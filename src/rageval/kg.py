"""既存 KG (Neo4j) を QA 生成の正解パス供給源として読むための層。**読み取り専用**。

背景: 生成の起点をチャンク抽選から KG パス抽選へ移す。従来は文書から関係を
再発見しようとしていたが、対象コーパス (chunks_plant) の KG は既に構築済みで、

  - 本文ノード (`Document`) の `id` が rageval の chunk_id と完全一致する
  - 実体間エッジが `source_chunks` (chunk_id のリスト) と `extraction_count` を持つ

ため、経路とその原文の裏付けをそのまま取り出せる。

注意: KG のエッジは LLM 抽出であって正解保証がない。汎用ハブ (「化学物質」
「ボイラー」等) を経由するだけの疑似経路が実在するので、
`pagerank` 上限・`extraction_count` 下限・値ノードの中継禁止で絞る必要がある。
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

# ---------------- 値ノードの判定 ----------------
# graphrag 側の `is_value` フラグは 16 ノードにしか付いていないため自前で判定する。
# 「様式第6号」「別表第2」「5年」のような値は **経路の終端にだけ**置く。
# 中継に使うと「様式番号を経由して別の条文へ」のような、実務では辿らない経路になる。
_VALUE_PATTERNS = [
    re.compile(r"^様式第[0-9０-９一二三四五六七八九十]+"),
    re.compile(r"^別表第?[0-9０-９一二三四五六七八九十]+"),
    re.compile(r"^附属書[A-ZＡ-Ｚ0-9０-９]"),
    re.compile(r"^[0-9０-９]+(?:\.[0-9]+)?\s*(?:年|か月|日|時間|mm|MPa|kW|℃|%|倍|m2|m3|l/min|V|A)$"),
]


def looks_like_value(name: str) -> bool:
    """ノード名が「値」か (様式番号・別表番号・数値+単位)。"""
    s = (name or "").strip()
    return any(p.match(s) for p in _VALUE_PATTERNS)


# ---------------- 経路のデータ構造 ----------------

def _surface_forms(raw: Any, fallback: str) -> list[str]:
    """ノードの surface_forms は文字列化されたリストで入っていることがある。"""
    out: list[str] = []
    if isinstance(raw, list):
        out = [str(x) for x in raw]
    elif isinstance(raw, str) and raw.strip().startswith("["):
        import ast
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                out = [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
    elif isinstance(raw, str) and raw.strip():
        out = [raw]
    if fallback and fallback not in out:
        out.append(fallback)
    return out


@dataclass
class PathEdge:
    subject: str
    predicate: str
    object: str
    source_chunks: list[str] = field(default_factory=list)
    extraction_count: int = 0
    subject_forms: list[str] = field(default_factory=list)
    object_forms: list[str] = field(default_factory=list)


@dataclass
class KGPath:
    """KG 上の1本の経路 (1辺 or 2辺)。"""
    edges: list[PathEdge]

    @property
    def nodes(self) -> list[str]:
        out = [self.edges[0].subject]
        for e in self.edges:
            out.append(e.object)
        return out

    @property
    def terminal_chunks(self) -> list[str]:
        """終端の辺の出典チャンク (= 答えが載っている見込みのチャンク)。"""
        return list(self.edges[-1].source_chunks)

    @property
    def all_chunks(self) -> list[str]:
        seen: list[str] = []
        for e in self.edges:
            for c in e.source_chunks:
                if c not in seen:
                    seen.append(c)
        return seen

    def chunks_are_disjoint(self) -> bool:
        """各辺の出典チャンクが互いに素か (= 1チャンクでは答えられないか)。"""
        if len(self.edges) < 2:
            return False
        seen: set[str] = set()
        for e in self.edges:
            cur = set(e.source_chunks)
            if cur & seen:
                return False
            seen |= cur
        return True

    def as_text(self) -> str:
        parts = [self.edges[0].subject]
        for e in self.edges:
            parts.append(f"-[{e.predicate}]->")
            parts.append(e.object)
        return " ".join(parts)


@dataclass(frozen=True)
class EdgeVerification:
    """原文で関係まで確認できたエッジと、その採用根拠。"""

    edge: PathEdge
    source_chunk_id: str
    reason: str


# ---------------- 接続 ----------------

@contextmanager
def session(uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
    """読み取り専用の Neo4j セッション。書き込みは一切行わない。"""
    from neo4j import GraphDatabase

    uri = uri or os.getenv("NEO4J_URI")
    user = user or os.getenv("NEO4J_USER")
    password = password or os.getenv("NEO4J_PW")
    if not uri:
        raise RuntimeError("NEO4J_URI が未設定です (.env を確認してください)")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as s:
            yield s
    finally:
        driver.close()


def snapshot_id(sess) -> str:
    """評価用 KG の版。生成した QA のメタデータに残して、KG 内推論の性能と
    実環境全体の性能を混同しないようにする。"""
    rec = sess.run(
        "MATCH (g:GraphProvenance) RETURN g.stamped_at AS at, g.kind AS kind, "
        "g.doc_count AS docs LIMIT 1"
    ).single()
    if not rec:
        return "unknown"
    return f"{rec['kind']}@{rec['at']}(docs={rec['docs']})"


# ---------------- 経路の列挙 ----------------

# 本文ノードや語彙ノードは経路の構成要素にしない (チャンク間エッジを辿らない)。
_EXCLUDED_LABELS = ("Document", "ProcessedChunk", "SchemaMeta", "GraphProvenance", "Doc")

# 経路の骨格に使わない関係型。
# `extraction_count` 順に並べると上位が IS_A / HAS_ATTRIBUTE / RELATED_TO で埋まる
# (頻出 = 汎用なので当然)。これらは「AだからB」の依存ではなく分類・連想であり、
# 辿っても「磁粉探傷試験は非破壊試験の一種」のような一般知識の問いにしかならない。
# 依存・手続き・規範の関係だけを骨格に使う。
TAXONOMIC_RELATIONS = frozenset({
    "IS_A", "RELATED_TO", "HAS_ATTRIBUTE", "SAME_AS", "MENTIONS",
    "PART_OF", "HAS_PART_OF", "PART_OF_DOCUMENT", "NEXT", "HAS_BODY",
    "SUBJECT_OF", "COMPARED_TO", "INVOLVES", "AFFILIATED_WITH",
})

# 中間ノードに許さないラベル。30問の全数検品 (2026-08-01) で、良い中間ノード
# (容器再検査・水圧試験・技能講習) は全て Process、悪い中間ノード (火災・危害・
# 過圧・規格) は Concept と綺麗に分かれた。抽象概念を中継すると「粉じん→火災→危害」
# のように無関係な規制体系同士が一般語で繋がる。Organization は「→労働省」のような
# 書誌問いにしかならないので中継にも使わない。
INTERMEDIATE_EXCLUDED_LABELS = frozenset({"Concept", "Organization"})

# 終端に許さないラベル。終端が Organization だと答えが「労働省」のような
# 発行主体名になり、規程の内容を問わない書誌問いに退化する。
TERMINAL_EXCLUDED_LABELS = frozenset({"Organization"})

# 規範・手続き・因果の関係 (これらを骨格にすると「辿らないと答えられない」問いになる)
DEPENDENCY_RELATIONS = frozenset({
    "REQUIRES", "REQUIRED_FOR", "APPLIES_TO", "DEFINED_BY", "REFERENCES",
    "REPLACES", "SUPERSEDES", "EXCLUDES", "BASED_ON", "CALCULATED_BY",
    "HAS_STEP", "FOLLOWS", "PRECEDES", "CAUSES", "CAUSED_BY", "PREVENTS",
    "DETECTS", "DETECTED_BY", "MEASURES", "MEASURED_BY", "VALIDATES",
    "CONTROLS", "MONITORS", "PRODUCES", "PRODUCED_BY", "USED_FOR",
    "ISSUED_BY", "SUBMITTED_TO", "REGISTERS", "PASSES_THROUGH",
})

_TWO_HOP_CYPHER = """
MATCH (a)-[e1]->(b)-[e2]->(c)
WHERE e1.source_chunks IS NOT NULL AND e2.source_chunks IS NOT NULL
  AND a <> b AND b <> c AND a <> c
  AND NOT any(l IN labels(a) WHERE l IN $excluded)
  AND NOT any(l IN labels(b) WHERE l IN $excluded)
  AND NOT any(l IN labels(c) WHERE l IN $excluded)
  AND coalesce(e1.extraction_count, 1) >= $min_ec
  AND coalesce(e2.extraction_count, 1) >= $min_ec
  AND ($max_pr IS NULL OR coalesce(b.pagerank, 0.0) <= $max_pr)
  AND (NOT $disjoint OR none(x IN e1.source_chunks WHERE x IN e2.source_chunks))
  AND ($allowed IS NULL OR (type(e1) IN $allowed AND type(e2) IN $allowed))
RETURN a.id AS a, type(e1) AS r1, b.id AS b, type(e2) AS r2, c.id AS c,
       a.surface_forms AS a_sf, b.surface_forms AS b_sf, c.surface_forms AS c_sf,
       e1.source_chunks AS sc1, e2.source_chunks AS sc2,
       coalesce(e1.extraction_count, 1) AS ec1, coalesce(e2.extraction_count, 1) AS ec2,
       coalesce(b.pagerank, 0.0) AS b_pr,
       labels(b) AS b_labels, labels(c) AS c_labels
ORDER BY (ec1 + ec2) DESC, b_pr ASC
LIMIT $limit
"""


def iter_two_hop_paths(
    sess,
    *,
    min_extraction_count: int = 2,
    max_pagerank: Optional[float] = 0.0005,
    require_disjoint_chunks: bool = True,
    exclude_value_intermediate: bool = True,
    allowed_relations: Optional[frozenset] = DEPENDENCY_RELATIONS,
    max_per_intermediate: int = 2,
    intermediate_excluded_labels: frozenset = INTERMEDIATE_EXCLUDED_LABELS,
    terminal_excluded_labels: frozenset = TERMINAL_EXCLUDED_LABELS,
    chunk_texts: Optional[dict[str, str]] = None,
    require_leak_free_terminal: bool = False,
    limit: int = 500,
) -> Iterator[KGPath]:
    """2辺の経路を、素材として使える条件つきで列挙する。

    既定の絞り込み:
      - 両辺が依存・手続き・規範の関係 (`DEPENDENCY_RELATIONS`)。
        分類・連想の関係 (IS_A 等) は骨格に使わない
      - 各辺が2チャンク以上から抽出されている (`extraction_count >= 2`)
      - 中間ノードが汎用ハブでない (`pagerank <= max_pagerank`)
      - 中間ノードが値ノードでない (様式番号・別表番号・数値+単位)
      - 中間ノードが抽象概念・組織でない (`INTERMEDIATE_EXCLUDED_LABELS`)
      - 終端が組織でない (`TERMINAL_EXCLUDED_LABELS`)
      - 2辺の出典チャンクが互いに素 (= 1チャンクでは答えられない)
      - `chunk_texts` を渡した場合は**始点閉包**: 始点の表記が終端辺の
        裏付けチャンク原文に現れること。中間ノードが手続き語でも、2辺が
        別の対象について語っている経路 (岩盤貯槽→耐圧試験→燃料電池設備)
        はここで落ちる
      - 同じ中間ノードを使う経路は `max_per_intermediate` 本まで。
        `pagerank` 下限をすり抜ける中程度のハブ (「実技講習」など) が
        ほぼ同一の経路を大量に出すので、ここで散らす
        (閉包などの検査は上限より先に掛け、落ちた経路が枠を食わないようにする)
      - `require_leak_free_terminal=True` (multi_hop 素材のみ) で**終端根拠の語彙断絶**:
        始点表記を含まない裏付け候補が終端辺に残る経路だけを通す
        (leak_free_terminal_chunks。閉包と根拠の役割分離、2026-08-07)
    """
    per_mid: dict[str, int] = {}
    rows = sess.run(
        _TWO_HOP_CYPHER,
        excluded=list(_EXCLUDED_LABELS),
        min_ec=min_extraction_count,
        max_pr=max_pagerank,
        disjoint=require_disjoint_chunks,
        allowed=(sorted(allowed_relations) if allowed_relations else None),
        limit=limit,
    )
    for r in rows:
        if exclude_value_intermediate and looks_like_value(r["b"]):
            continue
        if intermediate_excluded_labels & set(r["b_labels"] or []):
            continue
        if terminal_excluded_labels & set(r["c_labels"] or []):
            continue
        a_sf = _surface_forms(r["a_sf"], r["a"])
        b_sf = _surface_forms(r["b_sf"], r["b"])
        c_sf = _surface_forms(r["c_sf"], r["c"])
        path = KGPath(edges=[
            PathEdge(r["a"], r["r1"], r["b"], list(r["sc1"] or []), r["ec1"], a_sf, b_sf),
            PathEdge(r["b"], r["r2"], r["c"], list(r["sc2"] or []), r["ec2"], b_sf, c_sf),
        ])
        if chunk_texts is not None and not passes_start_closure(path, chunk_texts):
            continue
        # 語彙断絶 (multi_hop 素材のみ): 始点表記を含まない裏付け候補が終端辺に
        # 1つも無い経路は、どのチャンクを根拠にしても質問語が正解に漏れる
        if (require_leak_free_terminal and chunk_texts is not None
                and not leak_free_terminal_chunks(path, chunk_texts)):
            continue
        mid = r["b"]
        if per_mid.get(mid, 0) >= max_per_intermediate:
            continue
        per_mid[mid] = per_mid.get(mid, 0) + 1
        yield path


# ---------------- 星型 (aggregation 素材) ----------------

@dataclass
class KGStar:
    """同一中心から同型の依存辺が複数出る星型。「〜に要るものを全て」型の素材。

    KGPath と同じく .edges を持ち、検品側 (CheckContext) からは同型に見える。
    """
    center: str
    relation: str
    edges: list[PathEdge]          # 各枝。subject は全て center

    @property
    def branches(self) -> list[str]:
        return [e.object for e in self.edges]

    @property
    def all_chunks(self) -> list[str]:
        seen: list[str] = []
        for e in self.edges:
            for c in e.source_chunks:
                if c not in seen:
                    seen.append(c)
        return seen

    def as_text(self) -> str:
        inner = ", ".join(self.branches)
        return f"{self.center} -[{self.relation}]-> {{{inner}}}"


_CHUNK_POS_RE = re.compile(r"__c(\d+)$")


def _chunk_position(chunk_id: str) -> Optional[int]:
    m = _CHUNK_POS_RE.search(chunk_id or "")
    return int(m.group(1)) if m else None


def iter_star_groups(
    sess,
    *,
    min_branches: int = 3,
    max_branches: int = 8,
    max_position_spread: Optional[int] = None,
    chunk_texts: Optional[dict[str, str]] = None,
    limit: int = 2000,
) -> Iterator[KGStar]:
    """aggregation の素材になる星型を列挙する。

    設計表の条件:
      - 枝は依存・手続き・規範の関係 (`DEPENDENCY_RELATIONS`) の同型辺
      - 中心が Organization でない (「経済産業大臣が発する〜一覧」の書誌型を防ぐ)、
        値ノードでもない
      - **全枝の出典が同一文書内**。文脈混線 (「学科講習」に複数講習の科目が混ざる)
        の機械的な防波堤はこれだけ残し、同一文書内の混線 (「許容引張応力」が別表の
        別々の行を寄せ集めた実測例) の判定は段1の星合成判定に一本化する。
        位置差 ≤50 の近接要求は撤廃 (2026-08-07): 素材が位置幅 1〜8 の隣接チャンクに
        退化し、top10 被覆率 100% (v1 の 3/3) — 「検索で揃わない列挙」という型の
        存在意義を機械フィルタが先に消していた。`max_position_spread` は実測比較用に残す
      - 枝の出典が2チャンク以上に分散 (1チャンクの列挙表なら vector RAG で足りる)
      - 枝が多すぎる星は extraction_count 上位 `max_branches` 本に絞る
        (問いとして列挙可能な規模に保つ。正解は網羅主張にしない前提)
    `chunk_texts` を渡すと、両端の表記が出典に共起する枝だけを数える。
    """
    rows = sess.run("""
        MATCH (a)-[e]->(b)
        WHERE e.source_chunks IS NOT NULL
          AND a <> b
          AND NOT any(l IN labels(a) WHERE l IN $excluded)
          AND NOT any(l IN labels(b) WHERE l IN $excluded)
          AND NOT any(l IN labels(a) WHERE l = 'Organization')
          AND type(e) IN $dep
        RETURN a.id AS center, type(e) AS rel, b.id AS branch,
               a.surface_forms AS a_sf, b.surface_forms AS b_sf,
               e.source_chunks AS sc,
               coalesce(e.extraction_count, 1) AS ec
        LIMIT $limit
    """, excluded=list(_EXCLUDED_LABELS), dep=sorted(DEPENDENCY_RELATIONS),
        limit=limit * 50)

    groups: dict[tuple[str, str], list[PathEdge]] = {}
    for r in rows:
        if looks_like_value(r["center"]):
            continue
        a_sf = _surface_forms(r["a_sf"], r["center"])
        b_sf = _surface_forms(r["b_sf"], r["branch"])
        edge = PathEdge(r["center"], r["rel"], r["branch"],
                        list(r["sc"] or []), r["ec"], a_sf, b_sf)
        groups.setdefault((r["center"], r["rel"]), []).append(edge)

    emitted = 0
    for (center, rel), edges in groups.items():
        if emitted >= limit:
            break
        if chunk_texts is not None:
            edges = [e for e in edges
                     if candidate_support_chunks(e, chunk_texts)]
        if len(edges) < min_branches:
            continue
        edges = sorted(edges, key=lambda e: -e.extraction_count)[:max_branches]
        docs = {c.split("__")[0] for e in edges for c in e.source_chunks}
        if len(docs) != 1:
            continue
        chunks_used = {c for e in edges for c in e.source_chunks}
        if len(chunks_used) < 2:
            continue
        if max_position_spread is not None:
            positions = [p for p in (_chunk_position(c) for c in chunks_used)
                         if p is not None]
            if positions and max(positions) - min(positions) > max_position_spread:
                continue
        emitted += 1
        yield KGStar(center=center, relation=rel, edges=edges)


def iter_excludes_edges(
    sess,
    *,
    chunk_texts: Optional[dict[str, str]] = None,
    limit: int = 200,
) -> Iterator[KGPath]:
    """negation_exhaustive の素材になる EXCLUDES 辺を1辺経路として列挙する。

    否定・除外の問いは **原文に明記された除外だけ**から作る。KG に辺が無いことを
    「存在しない」と読む否定は、KG の欠損と実際の不存在を混同するため作らない
    (設計書の禁止事項)。供給は 62 本と少ないので、無理に増やさない。
    """
    rows = sess.run("""
        MATCH (a)-[e:EXCLUDES]->(b)
        WHERE e.source_chunks IS NOT NULL AND a <> b
          AND NOT any(l IN labels(a) WHERE l IN $excluded)
          AND NOT any(l IN labels(b) WHERE l IN $excluded)
        RETURN a.id AS a, b.id AS b,
               a.surface_forms AS a_sf, b.surface_forms AS b_sf,
               e.source_chunks AS sc, coalesce(e.extraction_count, 1) AS ec
        ORDER BY ec DESC
        LIMIT $limit
    """, excluded=list(_EXCLUDED_LABELS), limit=limit)
    for r in rows:
        edge = PathEdge(
            r["a"], "EXCLUDES", r["b"], list(r["sc"] or []), r["ec"],
            _surface_forms(r["a_sf"], r["a"]), _surface_forms(r["b_sf"], r["b"]),
        )
        if chunk_texts is not None and not candidate_support_chunks(edge, chunk_texts):
            continue
        yield KGPath(edges=[edge])


def verify_star_provenance(
    star: KGStar,
    chunk_text_by_id: dict[str, str],
    *,
    model: str,
    llm: Optional[LLMCaller] = None,
    min_branches: int = 3,
) -> Optional[list[EdgeVerification]]:
    """星の各枝を LLM 含意判定に掛け、裏付けの取れた枝だけを残す。

    連鎖 (verify_path_provenance) と違い、**落ちた枝は捨てるだけ**で星自体は
    生かす (正解は「原文で確定した要素の集合」であり、KG の枝全部ではない)。
    残った枝が min_branches 未満、または出典が1チャンクに縮んだら不成立。
    """
    verified: list[EdgeVerification] = []
    for edge in star.edges:
        accepted: Optional[EdgeVerification] = None
        for cid in candidate_support_chunks(edge, chunk_text_by_id)[:4]:
            accepted = verify_edge_relation(
                edge, cid, chunk_text_by_id[cid], model=model, llm=llm,
            )
            if accepted is not None:
                break
        if accepted is not None:
            verified.append(accepted)
    if len(verified) < min_branches:
        return None
    if len({v.source_chunk_id for v in verified}) < 2:
        return None
    return verified


# ---------------- provenance 検証 ----------------

def candidate_support_chunks(
    edge: PathEdge, chunk_text_by_id: dict[str, str],
) -> list[str]:
    """両端ノードの表記が同時に現れる source chunk を列挙する。

    これは関係の正しさを保証せず、LLM 含意判定へ渡す候補を安く絞るだけ。
    """
    from .evidence import normalize

    subj = [normalize(s) for s in (edge.subject_forms or [edge.subject]) if s]
    obj = [normalize(s) for s in (edge.object_forms or [edge.object]) if s]
    out: list[str] = []
    for cid in edge.source_chunks:
        text = normalize(chunk_text_by_id.get(cid, ""))
        if not text:
            continue
        if any(s and s in text for s in subj) and any(o and o in text for o in obj):
            out.append(cid)
    return out


def verify_edge(edge: PathEdge, chunk_text_by_id: dict[str, str]) -> Optional[str]:
    """エッジの両端が、出典チャンクの原文に実際に現れるかを確かめる。

    KG のエッジは LLM 抽出なので、原文に無い実体を繋いでいることがある
    (実例: 「高圧ガス保安法 -HAS_STEP-> 罰則」)。両端の表記が同じチャンクに
    揃って出てくることを最低条件にする。裏付けの取れたチャンク ID を返し、
    取れなければ None。
    """
    candidates = candidate_support_chunks(edge, chunk_text_by_id)
    return candidates[0] if candidates else None


def passes_start_closure(path: KGPath, chunk_text_by_id: dict[str, str]) -> bool:
    """始点の表記が、終端辺の裏付けチャンク原文に現れるか (= 対象が閉じているか)。

    中間ノードが「耐圧試験」のようなどの設備にも出てくる手続き語だと、
    2辺がそれぞれ別の対象について語っていても経路としては繋がってしまう
    (実例: 岩盤貯槽→耐圧試験→最高使用圧力、答えは燃料電池設備の基準)。
    終端側の原文に始点が登場することを求めると、この飛び移りを機械的に落とせる
    (30問検品時に該当3問を 3/3 捕捉)。裏付けチャンクは両端共起まで確認した
    候補 (candidate_support_chunks) に限る。
    """
    from .evidence import normalize

    if len(path.edges) < 2:
        return True
    start = path.edges[0]
    forms = [normalize(s) for s in (start.subject_forms or [start.subject])]
    forms = [f for f in forms if len(f) >= 2]
    if not forms:
        return False
    for cid in candidate_support_chunks(path.edges[-1], chunk_text_by_id):
        text = normalize(chunk_text_by_id.get(cid, ""))
        if any(f in text for f in forms):
            return True
    return False


def leak_free_terminal_chunks(
    path: KGPath, chunk_text_by_id: dict[str, str],
) -> list[str]:
    """終端辺の裏付け候補のうち、始点の表記が現れないチャンク (語彙断絶の根拠候補)。

    始点閉包 (passes_start_closure) は「始点表記が終端辺のどこかの裏付け候補に
    現れること」を求めるが、その閉包チャンクをそのまま答えの根拠にすると、
    始点を名指しする質問文が根拠チャンクに BM25 で直撃して検索で解ける問いになる
    (covered 28問の実測 2026-08-07: 質問と終端の語彙重なり ≥0.55 は 8/8 が top10 到達、
    不達4問は multi_hop に集中)。適用の証拠 (閉包) と答えの在り処 (根拠) を別チャンクに
    分離し、根拠側は始点語彙を含まない候補に限る。1辺経路には適用しない。
    """
    from .evidence import normalize

    candidates = candidate_support_chunks(path.edges[-1], chunk_text_by_id)
    if len(path.edges) < 2:
        return candidates
    start = path.edges[0]
    forms = [normalize(s) for s in (start.subject_forms or [start.subject])]
    forms = [f for f in forms if len(f) >= 2]
    out: list[str] = []
    for cid in candidates:
        text = normalize(chunk_text_by_id.get(cid, ""))
        if text and not any(f in text for f in forms):
            out.append(cid)
    return out


def verify_path(path: KGPath, chunk_text_by_id: dict[str, str]) -> Optional[list[str]]:
    """全ての辺で両端表記が共起するか。関係の検証には verify_path_provenance を使う。"""
    out: list[str] = []
    for e in path.edges:
        cid = verify_edge(e, chunk_text_by_id)
        if cid is None:
            return None
        out.append(cid)
    return out


LLMCaller = Callable[..., Any]


def _edge_verification_prompt(edge: PathEdge, chunk_id: str, text: str) -> str:
    return f"""あなたは知識グラフの provenance 検証者です。
次の原文が、指定された有向関係を明示的に裏付けるか判定してください。

subject: {edge.subject}
predicate: {edge.predicate}
object: {edge.object}
source_chunk_id: {chunk_id}

原文:
---
{text}
---

判定規則:
- subject と object が両方出るだけの共起は false。
- predicate の意味と向きまで、原文から直接読み取れる場合だけ true。
- 常識や外部知識で関係を補ってはいけない。
- 表記揺れは認めるが、別の実体を同一視してはいけない。

次の JSON object だけを返してください:
{{"entailed": true または false, "reason": "原文に即した短い理由"}}
"""


def verify_edge_relation(
    edge: PathEdge,
    source_chunk_id: str,
    source_text: str,
    *,
    model: str,
    llm: Optional[LLMCaller] = None,
) -> Optional[EdgeVerification]:
    """候補チャンクがエッジの述語・向きまで含意するか LLM で判定する。

    false は通常の棄却として None を返す。通信障害や不正 JSON は例外を伝播し、
    障害を「関係なし」として静かに混ぜない。
    """
    from .llm import LLMError, _parse_json, generate

    caller = llm or generate
    raw = caller(
        prompt=_edge_verification_prompt(edge, source_chunk_id, source_text),
        model=model,
        temperature=0.0,
        max_tokens=4096,
        response_model=None,
        force_json=True,
    )
    if hasattr(raw, "model_dump"):
        data = raw.model_dump()
    elif isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        data = _parse_json(raw)
    else:
        raise LLMError(f"Unsupported provenance response: {type(raw).__name__}")
    if not isinstance(data, dict) or type(data.get("entailed")) is not bool:
        raise LLMError(f"Invalid provenance response: {data!r}")
    reason = str(data.get("reason") or "").strip()
    if not reason:
        raise LLMError(f"Provenance response has no reason: {data!r}")
    if not data["entailed"]:
        return None
    return EdgeVerification(
        edge=edge, source_chunk_id=source_chunk_id, reason=reason,
    )


def verify_path_provenance(
    path: KGPath,
    chunk_text_by_id: dict[str, str],
    *,
    model: str,
    llm: Optional[LLMCaller] = None,
    terminal_leak_free: bool = False,
) -> Optional[list[EdgeVerification]]:
    """全エッジについて、共起候補の少なくとも1件で関係まで確認する。

    `terminal_leak_free=True` (multi_hop) では終端辺の候補を語彙断絶チャンク
    (始点表記を含まないもの) に絞る。素材列挙が leak-free 候補ありの経路だけを
    通していても、段0が閉包チャンクを根拠に採ってしまえば台無しになるため。
    """
    verified: list[EdgeVerification] = []
    used_chunks: set[str] = set()
    last_i = len(path.edges) - 1
    for i, edge in enumerate(path.edges):
        accepted: Optional[EdgeVerification] = None
        if terminal_leak_free and i == last_i and last_i >= 1:
            candidates = leak_free_terminal_chunks(path, chunk_text_by_id)
        else:
            candidates = candidate_support_chunks(edge, chunk_text_by_id)
        # 候補は上位4件まで。抽出回数の多い辺は候補が20件を超え、全滅パターンで
        # 1辺に数分溶ける (2026-08-03 の17時間無進捗の一因)
        for cid in candidates[:4]:
            if cid in used_chunks:
                continue
            accepted = verify_edge_relation(
                edge, cid, chunk_text_by_id[cid], model=model, llm=llm,
            )
            if accepted is not None:
                break
        if accepted is None:
            return None
        verified.append(accepted)
        used_chunks.add(accepted.source_chunk_id)
    return verified


def count_two_hop_paths(sess, **filters: Any) -> int:
    """絞り込み条件ごとの経路数 (供給量の実測用)。値ノード判定は Python 側なので
    ここには含めない — 呼び出し側で列挙して数えること。"""
    q = _TWO_HOP_CYPHER.replace(
        "RETURN a.id AS a", "RETURN count(*) AS n // "
    ).split("RETURN count(*) AS n")[0] + "RETURN count(*) AS n"
    allowed = filters.get("allowed_relations", DEPENDENCY_RELATIONS)
    return sess.run(
        q,
        excluded=list(_EXCLUDED_LABELS),
        min_ec=filters.get("min_extraction_count", 2),
        max_pr=filters.get("max_pagerank", 0.0005),
        disjoint=filters.get("require_disjoint_chunks", True),
        allowed=(sorted(allowed) if allowed else None),
        limit=filters.get("limit", 1_000_000),
    ).single()["n"]


def chunk_node_map(sess) -> dict[str, set]:
    """チャンクID → そのチャンクを出典に持つエッジが触るノード集合。

    「この2チャンクは KG 上で繋がっているか」(共有ノードの有無) の判定に使う。
    KG-missing 系統の供給 (明示参照のうち KG に対応が無い対) の判定用。
    """
    rows = sess.run("""
        MATCH (a)-[e]->(b) WHERE e.source_chunks IS NOT NULL
        RETURN a.id AS a, b.id AS b, e.source_chunks AS sc
    """)
    out: dict[str, set] = {}
    for r in rows:
        for cid in (r["sc"] or []):
            out.setdefault(cid, set()).update((r["a"], r["b"]))
    return out
