"""Anchor-chunk selection strategies for multi-chunk QA generation.

Each strategy returns a list of `Chunk` objects to feed into the generation
prompt. Dispatch is keyed on aspect — see `ASPECT_STRATEGIES` and
`AnchorSampler.select`.

All strategies must gracefully fall back to a single chunk when their preferred
condition cannot be satisfied (e.g., only one doc in corpus → MultiDocStrategy
falls back to single-chunk; embeddings unavailable → MultiDocByEmbedding falls
back to random pick).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol

import numpy as np

from .schema import Chunk, RetrievalDifficulty


# ---------------- 台帳・種の置き場 (パスの正本) ----------------
# 2026-08-08 監査 rank 30: 同じ台帳のパスが generate / filter / enrich / cli の
# 6箇所に直書きされており、コーパスを替えて台帳を作り直すと「CLI 経由だけ新台帳・
# 他は旧台帳」という食い違いが無警告で起きていた。台帳を読む側 (この module) に
# 1組だけ置き、他 module はここか generate の別名を引く。
SEEDS_DIR = Path("data/seeds")
DEFAULT_ANCHOR_PAIRS_PATH = SEEDS_DIR / "anchor_pairs_manual_v1.jsonl"
DEFAULT_DOC_ALIASES_PATH = SEEDS_DIR / "doc_aliases.json"
DEFAULT_STANDARD_PREFIXES_PATH = SEEDS_DIR / "standard_prefixes.json"


# ---------------- Aspect compatibility heuristics ----------------
# Goal: when an aspect is drawn, only consider chunks that *can plausibly
# support* that aspect. E.g. "complex_layout" needs structural markers, not a
# flat paragraph. Each predicate returns True if the chunk is a viable anchor
# for the aspect.

_TABLE_MARKERS = [
    "|---", "| ---",         # markdown table separator
    "<table", "<tr", "<td",  # html
]
_FLOWCHART_HINTS = ["フロー", "手順", "ステップ", "→", "▼", "(1)", "(2)", "①", "②"]
_CHART_HINTS = ["グラフ", "チャート", "折れ線", "棒グラフ", "円グラフ", "推移"]
_FIGURE_HINTS = ["![", "<img", "<figure", "図1", "図2", "図3", "Fig.", "Figure"]
_LIST_LINE_RE = re.compile(r"^\s*([-*・・]|\d+[\.)]|[①-⑳])", re.MULTILINE)
_NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?")
_COMPARISON_HINTS = ["以上", "以下", "未満", "超え", "より", "に対し", "倍", "%"]
_CAUSAL_HINTS = ["ため", "により", "原因", "結果", "ことから", "ゆえ"]
_TEMPORAL_HINTS = ["まず", "次に", "最後に", "後", "前", "までに", "経過", "順序"]
_NEGATION_HINTS = ["ない", "禁止", "除く", "適用しない", "してはならない"]
_ABSTENTION_HINTS_NONE: list[str] = []  # any chunk OK; LLM is the deciding factor


def _has_any(text: str, markers: list[str]) -> bool:
    return any(m in text for m in markers)


# v2.0 追加: チャンクが「リスト専用」(様式番号や別表番号を列挙しただけで
# 規定本文を持たない目次的チャンク) かを判定する。
# 09_kagi (条文本文) × 45_tokutei pos=45 (様式第1〜27 を丸ごと貼ったチャンク) の
# ような非対称ペアで Gemma が片 doc 完結問いを書く原因を素材レベルで絶つ。
_FORM_LIST_RE = re.compile(r"様式第[０-９0-9一二三四五六七八九十百千]+号?(?:[（(][^）)]{0,20}[）)])?")
_APPENDIX_LIST_RE = re.compile(r"別表第[０-９0-9一二三四五六七八九十百千]+(?:[（(][^）)]{0,20}[）)])?")
_ARTICLE_REF_RE = re.compile(r"第[０-９0-9]+条(?:第[０-９0-9]+項)?(?:第[０-９0-9]+号)?")


def _is_list_chunk(c: Chunk, list_density_threshold: float = 0.6, min_items: int = 5) -> bool:
    """様式 / 別表 を 5 件以上列挙し、それが本文の 60% 以上を占めるチャンクを True。

    例: 45_tokutei_setsubi pos=45 は「様式第1〜第27」を丸ごと貼ったチャンクで、
    list_density は 1.0 近く、min_items は 27 になる → True。
    一方、本文に「様式第6号を参照」と書いている条文チャンクは、list_density は
    0.1 未満なので False。
    """
    text = c.text
    if not text:
        return False
    form_hits = _FORM_LIST_RE.findall(text)
    app_hits = _APPENDIX_LIST_RE.findall(text)
    total_items = len(form_hits) + len(app_hits)
    if total_items < min_items:
        return False
    # 列挙の hit テキストが本文の何割を占めるか
    hit_chars = sum(len(h) for h in form_hits) + sum(len(h) for h in app_hits)
    density = hit_chars / len(text)
    return density >= list_density_threshold


def _refs_in_text(refs: set[str], text: str) -> bool:
    """anchor の references (= 言及参照 token) が target チャンクの本文に
    1 つでも逐語登場するなら True。"""
    if not refs or not text:
        return False
    return any(ref in text for ref in refs)


# ---------------- 文書スコープ規則 (v3: 別文書の同番号誤結合の根絶) ----------------
# 「第N条」「様式第N」「別表第N」等は文書ローカルな識別子であり、別文書の同番号と
# 突き合わせてはならない (plant_v15 で別法令の「第63条」「第7条」を同一視した捏造
# QA が 3/25 混入した実害あり)。異文書ペアの成立は次の3系統に限る:
#   (a) 全域一意な規格番号 (JIS/ISO/IEC/ASME/JPI 等) の共有
#   (b) 明示参照 — 相手文書の名称・別名を本文が名指ししている (doc_aliases 台帳)
#   (c) 解決済みエッジ台帳 (seeds/anchor_pairs_manual_v1.jsonl) に載っているチャンク対

# 規格団体の接頭辞はここが唯一の定義。異文書対の成立判定 (_is_global_ref) と
# standards_reference 観点の適合述語 (_has_standards_ref) の両方をこの1組から作る。
# 2026-08-08 監査 rank 27: 同一ファイル内に JIS|ISO|IEC|ASME だけの別実装があり、
# JPI/JEC/JEM/API/EN/NFPA を含むチャンクが「規格共有で対を組んでよい」のに
# 「規格参照ではない」と扱われる矛盾が既に起きていた。
# 社内文書番号の体系に替えるときは data/seeds の台帳 (下の load_standard_prefixes)
# で差し替える — コードを直さないと供給が消える状態にしない。
DEFAULT_STANDARD_PREFIXES: tuple[str, ...] = (
    "JIS", "ISO", "IEC", "ASME", "JPI", "JEC", "JEM", "API", "EN", "NFPA",
)


def _build_global_ref_re(prefixes: Iterable[str]) -> re.Pattern:
    alts = "|".join(re.escape(p) for p in prefixes if p)
    if not alts:
        # 接頭辞が空だと「^(?:)\b」が全てに当たってしまう。全域一意な参照は
        # 存在しない、と解釈できる正規表現にする。
        return re.compile(r"(?!)")
    return re.compile(rf"^(?:{alts})\b", re.IGNORECASE)


STANDARD_PREFIXES: tuple[str, ...] = DEFAULT_STANDARD_PREFIXES
_GLOBAL_REF_RE = _build_global_ref_re(STANDARD_PREFIXES)


def set_standard_prefixes(prefixes: Iterable[str]) -> tuple[str, ...]:
    """規格団体の接頭辞を差し替える (照合器を作り直す)。採用した集合を返す。"""
    global STANDARD_PREFIXES, _GLOBAL_REF_RE
    STANDARD_PREFIXES = tuple(dict.fromkeys(str(p).strip() for p in prefixes if str(p).strip()))
    _GLOBAL_REF_RE = _build_global_ref_re(STANDARD_PREFIXES)
    return STANDARD_PREFIXES


def load_standard_prefixes(path: Optional[Path]) -> Optional[tuple[str, ...]]:
    """台帳 (JSON の文字列配列) から規格接頭辞を読む。無ければ None。"""
    if path is None or not Path(path).exists():
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"[sampling] 警告: 規格接頭辞の台帳が壊れている ({path}) — 既定を使う")
        return None
    if isinstance(data, dict):
        data = data.get("prefixes", [])
    if not isinstance(data, list) or not data:
        print(f"[sampling] 警告: 規格接頭辞の台帳が空 ({path}) — 既定を使う")
        return None
    return tuple(str(x).strip() for x in data if str(x).strip())


def apply_standard_prefixes(path: Optional[Path]) -> tuple[str, ...]:
    """台帳があれば規格接頭辞を差し替え、実際に使う集合を返す。

    台帳が無いのは異常ではない (既定の規格番号体系で回る) が、採用した集合は
    必ず呼び出し側でログに出す — 異文書対の成立経路がここで決まるため。
    """
    loaded = load_standard_prefixes(path)
    if loaded is None:
        return STANDARD_PREFIXES
    return set_standard_prefixes(loaded)


# 「ボイラー則第百一条第二号ロ」のように、文書名の直後に来る条番号を拾う。
# コーパスは漢数字 (02_boiler_anzen) と全角算用数字 (09_kagi) が混在する。
# 「第56条」が「第56条の6の2」(枝番条文 = 別の条) に部分一致しないよう、
# 直後に「の+数字」が続く場合は不一致とする (2026-08-03: kg_missing 供給の
# 偽参照を Qwen 判定が発見したことで発覚)。
_ARTICLE_AFTER_ALIAS_RE = re.compile(
    r"第([0-9０-９一二三四五六七八九十百千]+)条(?!の[0-9０-９一二三四五六七八九十])")
_KANJI_DIGITS = {"〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                 "六": 6, "七": 7, "八": 8, "九": 9}


def _kanji_to_int(s: str) -> Optional[int]:
    """漢数字 (百一 / 十五 / 三) を整数にする。実装は evidence.kanji_to_int に集約。"""
    from .evidence import kanji_to_int
    return kanji_to_int(s)


def article_number(token: str) -> Optional[int]:
    """「第百一条」「第１０１条」「第101条」→ 101。解釈できなければ None。"""
    m = _ARTICLE_AFTER_ALIAS_RE.search(token)
    if not m:
        return None
    body = m.group(1)
    # 全角算用数字 → 半角
    trans = body.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if trans.isdigit():
        return int(trans)
    return _kanji_to_int(body)


def _article_variants(n: int) -> list[str]:
    """条番号 n を、コーパスに現れうる表記に展開する (半角/全角/漢数字)。"""
    half = str(n)
    full = half.translate(str.maketrans("0123456789", "０１２３４５６７８９"))
    out = [f"第{half}条", f"第{full}条"]
    kanji = _int_to_kanji(n)
    if kanji:
        out.append(f"第{kanji}条")
    return out


def _int_to_kanji(n: int) -> Optional[str]:
    """101 → 百一 (法令の条番号表記)。4桁までの簡易変換。"""
    if not (0 < n < 10000):
        return None
    digits = "〇一二三四五六七八九"
    units = [(1000, "千"), (100, "百"), (10, "十")]
    out = ""
    rest = n
    for value, unit in units:
        q, rest = divmod(rest, value)
        if q:
            out += ("" if q == 1 else digits[q]) + unit
    if rest:
        out += digits[rest]
    return out or None


def _is_global_ref(ref: str) -> bool:
    """コーパス全域で一意に解決できる参照 (規格番号) か。"""
    return bool(_GLOBAL_REF_RE.match(ref.strip()))


def _global_refs(refs) -> set[str]:
    return {r for r in (refs or []) if _is_global_ref(r)}


@dataclass
class CrossDocPolicy:
    """異文書チャンク対の成立可否を判定する規則の束。

    generate 側が load_cross_doc_policy() で台帳を読み込んで差し替える。
    台帳なしの既定状態でも (a) 規格番号の共有だけは機能する。

    doc_ns は文書の名前空間 (例: プラント名)。「設備保全管理規程」のような短名は
    コーパス全体では曖昧でも、同じ名前空間の中では一意に解決できる。名前空間が
    両方に定義されている場合、別名の名指しは同一名前空間内でのみ有効。
    """
    allowed_chunk_pairs: set[frozenset] = field(default_factory=set)
    doc_aliases: dict[str, list[str]] = field(default_factory=dict)
    doc_ns: dict[str, str] = field(default_factory=dict)

    def _ns_compatible(self, a: Chunk, b: Chunk) -> bool:
        ns_a, ns_b = self.doc_ns.get(a.doc_id), self.doc_ns.get(b.doc_id)
        if ns_a is None or ns_b is None:
            return True
        return ns_a == ns_b

    def _named_ok(self, a: Chunk, b: Chunk) -> bool:
        """一方の本文が他方の文書名 (別名) を名指ししているか。

        名前空間が異なる文書同士は、短名の名指しでは繋がない (双子プラントの
        「検査様式集」問題)。ただしプラント名付きの完全名・文書番号は
        名前空間を跨いでも一意なので、別名側に残っていれば成立する。
        """
        same_ns = self._ns_compatible(a, b)
        for alias in self.doc_aliases.get(b.doc_id, []):
            if alias and alias in a.text and (same_ns or self._alias_unique(alias)):
                return True
        for alias in self.doc_aliases.get(a.doc_id, []):
            if alias and alias in b.text and (same_ns or self._alias_unique(alias)):
                return True
        return False

    def _alias_unique(self, alias: str) -> bool:
        """別名がコーパス全体で一意 (1文書だけが名乗る) か。"""
        owners = sum(1 for names in self.doc_aliases.values() if alias in names)
        return owners <= 1

    def named_reference(self, anchor: Chunk, target_doc: str) -> Optional[int]:
        """anchor 本文が target_doc を名指ししているとき、その直後の条番号を返す。

        「ボイラー則第百一条第二号ロ」→ 101。条番号が付いていなければ None。
        """
        for alias in self.doc_aliases.get(target_doc, []):
            if not alias or alias not in anchor.text:
                continue
            pos = anchor.text.find(alias) + len(alias)
            tail = anchor.text[pos:pos + 20]     # 名指しの直後だけ見る
            n = article_number(tail)
            if n is not None:
                return n
        return None

    def resolve_named_target(
        self,
        anchor: Chunk,
        target_chunks: list[Chunk],
        *,
        similarity: Optional[Callable[[Chunk], float]] = None,
        sim_floor: float = 0.45,
        allow_similarity_fallback: bool = False,
    ) -> Optional[Chunk]:
        """名指しされた文書の中から、実際に参照している1チャンクを決める。

        文書名の名指しだけで接続を認めると、相手文書の**全チャンク**が候補になり
        (plant で1アンカーあたり中央値79・最大531)、無関係な章が対に選ばれていた。
        参照先を次の順で1つに絞る:
          1. 名指しの直後の条番号 (例「ボイラー則第百一条」) が見出し位置に現れるチャンク
          2. (1) が無ければ、その条番号を本文に含むチャンク
          3. 条番号が書かれていない場合、allow_similarity_fallback=True のときだけ
             埋め込み類似の最上位 (下限つき)
        どれも決まらなければ None (= その対は使わない)。

        類似フォールバックは既定で無効。相手文書に絞ってはいるものの、結局
        「類似＝関係」を持ち込み直しており検証手段が無いため (この誤りで
        「防毒呼吸具 × 液化ガス容器」のような対を掴んだ履歴がある)。
        """
        if not target_chunks:
            return None
        n = self.named_reference(anchor, target_chunks[0].doc_id)
        if n is not None:
            variants = _article_variants(n)
            heading: list[Chunk] = []
            body: list[Chunk] = []
            for c in target_chunks:
                for v in variants:
                    # 枝番条文 (第N条の6の2) への部分一致を除く (regex 側と同じ規則)
                    m_v = re.search(
                        re.escape(v) + r"(?!の[0-9０-９一二三四五六七八九十])", c.text)
                    idx = m_v.start() if m_v else -1
                    if idx < 0:
                        continue
                    # 条文の本体は「第N条 …」で始まる。チャンク先頭寄り or
                    # 直前が改行なら見出し (= 参照先そのもの) とみなす。
                    if idx < 80 or (idx > 0 and c.text[idx - 1] in "\n 　"):
                        heading.append(c)
                    else:
                        body.append(c)
                    break
            if heading:
                return heading[0]
            if body:
                return body[0]
            return None    # 名指しの条番号が相手文書に無い → 接続が成立しない
        if similarity is None or not allow_similarity_fallback:
            return None
        best = max(target_chunks, key=similarity)
        return best if similarity(best) >= sim_floor else None

    def describe_join(self, a: Chunk, b: Chunk) -> Optional[str]:
        """異文書対の接続根拠を、生成プロンプトに渡せる一文で返す。

        join_ok と同じ判定を「どの規則で成立したか」まで含めて言語化する。
        成立しない対 (または同一文書) は None。
        """
        if a.doc_id == b.doc_id:
            return None
        if frozenset((a.chunk_id, b.chunk_id)) in self.allowed_chunk_pairs:
            return "検証済みエッジ台帳に登録された対 (実在する参照関係を人手で確認済み)"
        a_glob, b_glob = _global_refs(a.references), _global_refs(b.references)
        shared = a_glob & b_glob
        if not shared:
            shared = {r for r in a_glob if r in b.text} | {r for r in b_glob if r in a.text}
        if shared:
            return f"両チャンクが同じ規格 {'、'.join(sorted(shared))} を参照している"
        same_ns = self._ns_compatible(a, b)
        for src, dst in ((a, b), (b, a)):
            for alias in self.doc_aliases.get(dst.doc_id, []):
                if not alias or alias not in src.text:
                    continue
                if not (same_ns or self._alias_unique(alias)):
                    continue
                n = self.named_reference(src, dst.doc_id)
                if n is not None:
                    return (f"一方の本文が『{alias}第{n}条』と名指しし、"
                            f"もう一方がその条文そのもの")
                return f"一方の本文が文書『{alias}』(もう一方のチャンクの文書) を名指しして参照している"
        return None

    # 異文書対を作れるチャンクの索引 (index_cross_capable が構築)。
    # 一様ランダム抽選だと台帳対・明示参照チャンクがほぼ引かれず、
    # 本丸 (文書跨ぎ) の供給が偶然任せになるため、抽選側が優先に使う。
    cross_capable_chunk_ids: set = field(default_factory=set)

    def index_cross_capable(self, chunks: list[Chunk]) -> None:
        """異文書対の成立根拠を持つチャンクを1回走査で索引化する。"""
        capable: set[str] = set()
        # (c) 台帳のチャンク対
        for pair in self.allowed_chunk_pairs:
            capable.update(pair)
        # (a) 規格番号: 同じ規格を2文書以上が参照していれば、その参照元は対を作れる
        ref_docs: dict[str, set[str]] = {}
        for c in chunks:
            for r in _global_refs(c.references):
                ref_docs.setdefault(r, set()).add(c.doc_id)
        shared_refs = {r for r, docs in ref_docs.items() if len(docs) >= 2}
        for c in chunks:
            if _global_refs(c.references) & shared_refs:
                capable.add(c.chunk_id)
        # (b) 明示参照: 他文書の別名を本文で名指ししているチャンク
        alias_owner: dict[str, str] = {}
        for doc, names in self.doc_aliases.items():
            for name in names:
                alias_owner[name] = doc
        for c in chunks:
            for name, owner in alias_owner.items():
                if owner != c.doc_id and name in c.text:
                    ns_c, ns_o = self.doc_ns.get(c.doc_id), self.doc_ns.get(owner)
                    if (ns_c is None or ns_o is None or ns_c == ns_o) or self._alias_unique(name):
                        capable.add(c.chunk_id)
                        break
        self.cross_capable_chunk_ids = capable

    def join_ok(self, anchor: Chunk, cand: Chunk) -> bool:
        """異文書チャンク対として成立するか。同一文書なら常に True。

        裸の条番号・様式番号の一致は根拠として認めない (別文書の同番号は別物)。
        """
        if anchor.doc_id == cand.doc_id:
            return True
        if frozenset((anchor.chunk_id, cand.chunk_id)) in self.allowed_chunk_pairs:
            return True  # 台帳のチャンク対はそれ自体が解決済みエッジ
        a_glob, c_glob = _global_refs(anchor.references), _global_refs(cand.references)
        if a_glob & c_glob:
            return True
        if _refs_in_text(a_glob, cand.text) or _refs_in_text(c_glob, anchor.text):
            return True
        return self._named_ok(anchor, cand)


# 既定は台帳なし。generate_batch が load_cross_doc_policy() で差し替える。
CROSS_DOC_POLICY = CrossDocPolicy()


def load_cross_doc_policy(
    anchor_pairs_path: Optional[Path] = None,
    aliases_path: Optional[Path] = None,
) -> CrossDocPolicy:
    """seeds のエッジ台帳・文書別名台帳から CrossDocPolicy を作る。

    - anchor_pairs_path: anchor_pairs_manual_v1.jsonl 形式
      (各行 {"anchors": [{"chunk_id": ...}, ...]})。anchors 内の全2点組を許可対に登録。
    - aliases_path: {"doc_id": ["名称", "略称", ...]} の JSON。無ければ空。
    """
    # 台帳が見つからないと規則が空になり、異文書対が「成立しない」側へ静かに倒れる
    # (kg_decoy はラベルの意味が反転する)。無警告で空にしない (2026-08-08 監査 rank 30)。
    for path, what in ((anchor_pairs_path, "解決済みエッジ台帳"),
                       (aliases_path, "文書別名台帳")):
        if path is not None and not Path(path).exists():
            print(f"[sampling] 警告: {what}が見つからない ({path}) — "
                  f"その規則は空のまま進む (異文書対は残る規則だけで成立させる)")
    pairs: set[frozenset] = set()
    if anchor_pairs_path and anchor_pairs_path.exists():
        with anchor_pairs_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ids = [a.get("chunk_id") for a in rec.get("anchors", []) if a.get("chunk_id")]
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        pairs.add(frozenset((ids[i], ids[j])))
    aliases: dict[str, list[str]] = {}
    doc_ns: dict[str, str] = {}
    if aliases_path and aliases_path.exists():
        try:
            data = json.loads(aliases_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        aliases[str(k)] = [str(x) for x in v]
                    elif isinstance(v, dict):
                        aliases[str(k)] = [str(x) for x in v.get("aliases", [])]
                        if v.get("ns"):
                            doc_ns[str(k)] = str(v["ns"])
        except json.JSONDecodeError:
            pass
    # 同じ名前空間の中で複数文書が名乗る別名は解決不能なので落とす
    # (名前空間を跨ぐ重複は _named_ok が判定時に処理する)。
    owner_count: dict[tuple, int] = {}
    for doc, names in aliases.items():
        ns = doc_ns.get(doc)
        for name in names:
            owner_count[(ns, name)] = owner_count.get((ns, name), 0) + 1
    dropped = {key for key, cnt in owner_count.items() if cnt >= 2}
    if dropped:
        print(f"[sampling] 名前空間内で曖昧な別名 {len(dropped)} 件を無効化: "
              f"{sorted(n for _, n in dropped)[:5]} …")
        aliases = {d: [n for n in names if (doc_ns.get(d), n) not in dropped]
                   for d, names in aliases.items()}
    return CrossDocPolicy(allowed_chunk_pairs=pairs, doc_aliases=aliases, doc_ns=doc_ns)


def _has_table(chunk: Chunk) -> bool:
    return _has_any(chunk.text, _TABLE_MARKERS)


def _has_complex_form(chunk: Chunk) -> bool:
    t = chunk.text
    # Tables with merged cells, multi-row headers, nested structure
    if "rowspan" in t or "colspan" in t:
        return True
    # crude proxy: a table AND at least 3 rows
    if _has_table(chunk) and t.count("\n|") >= 4:
        return True
    return False


def _has_long_list(chunk: Chunk) -> bool:
    return len(_LIST_LINE_RE.findall(chunk.text)) >= 5


def _has_multi_section(chunk: Chunk) -> bool:
    return len(chunk.section_path) >= 2


def _has_chart_marker(chunk: Chunk) -> bool:
    return _has_any(chunk.text, _CHART_HINTS) or _has_any(chunk.text, _FIGURE_HINTS)


def _has_flowchart_marker(chunk: Chunk) -> bool:
    return _has_any(chunk.text, _FLOWCHART_HINTS)


def _has_concept_diagram_marker(chunk: Chunk) -> bool:
    # We rarely have actual diagrams in text. Allow chunks that reference
    # a figure/diagram by name.
    return _has_any(chunk.text, ["構成図", "ブロック図", "概念図", "組織図"]) or _has_any(chunk.text, _FIGURE_HINTS)


def _has_standards_ref(chunk: Chunk) -> bool:
    """規格参照を持つチャンクか。判定は _is_global_ref (STANDARD_PREFIXES) に一本化。

    2026-08-08 監査 rank 27: ここに JIS|ISO|IEC|ASME だけの2つ目の列挙があり、
    JPI/JEC/JEM/API/EN/NFPA が「異文書対は組めるが規格参照ではない」という
    矛盾した扱いを受けていた。
    """
    return any(_is_global_ref(r) for r in chunk.references)


def _has_any_reference(chunk: Chunk) -> bool:
    return bool(chunk.references)


def _has_numeric_calc(chunk: Chunk) -> bool:
    # Need at least 2 numbers AND a comparator/operator hint for quantitative
    nums = _NUMERIC_RE.findall(chunk.text)
    return len(nums) >= 2 and _has_any(chunk.text, _COMPARISON_HINTS)


def _has_comparison(chunk: Chunk) -> bool:
    return _has_any(chunk.text, _COMPARISON_HINTS)


def _has_causal(chunk: Chunk) -> bool:
    return _has_any(chunk.text, _CAUSAL_HINTS)


def _has_temporal(chunk: Chunk) -> bool:
    return _has_any(chunk.text, _TEMPORAL_HINTS)


def _has_negation(chunk: Chunk) -> bool:
    return _has_any(chunk.text, _NEGATION_HINTS)


def _always(_: Chunk) -> bool:
    return True


ASPECT_COMPATIBILITY: dict[str, Callable[[Chunk], bool]] = {
    # --- Integration ---
    "multi_source_integration": _always,           # filtered later by MultiDocByEmbedding
    "multi_doc_reference":      _always,           # multi-doc constraint in strategy
    "remote_reference":         _always,           # same-doc distance check in strategy
    "standards_reference":      _has_standards_ref,
    # --- Reasoning ---
    "quantitative_calc":        _has_numeric_calc,
    "multi_hop":                _always,
    "negation":                 _has_negation,
    "causal":                   _has_causal,
    "temporal":                 _has_temporal,
    "comparison_conditional":   _has_comparison,
    # --- Logic ---
    "synonym_interpretation":   _always,
    "numeric_inclusion":        _has_numeric_calc,
    "concept_inclusion":        _always,
    "vocabulary_mismatch":      _always,           # any chunk with technical vocab
    "abstraction_gap":          _always,
    # --- Figure (CRITICAL: avoid pairing text-only chunks with figure aspects) ---
    "simple_table":             _has_table,
    "complex_form":             _has_complex_form,
    "concept_diagram":          _has_concept_diagram_marker,
    "flowchart":                _has_flowchart_marker,
    "chart_graph":              _has_chart_marker,
    "complex_layout":           _has_multi_section,  # multi-section ≒ layered layout
    "large_enumeration":        _has_long_list,
    # --- Abstention ---
    "insufficient_evidence":    _always,           # LLM crafts unanswerable question
    "contradictory_evidence":   _always,
    "fragmented_chunk":         _always,
    # --- kg_poc 専用キー ---
    "reference_follow":         _has_any_reference,
    # single_fact は1チャンクで完結する型なので素材を絞らない。
    # 2026-08-08 監査 rank 25: 既存観点 "quantitative_calc" を流用していたため
    # 互換述語 _has_numeric_calc が掛かり、アンカーが「数値2個以上+比較語」を含む
    # チャンクだけに縮んでいた (実測 chunks_plant 3175→1407 / chunks_synth 330→59)。
    # ASPECT_STRATEGIES には載せない = SingleChunk で1チャンクを引く。
    "kg_single_fact":           _always,
}


def find_compatible_chunks(aspect: str, chunks: list[Chunk]) -> list[Chunk]:
    """Return only chunks that pass the aspect's compatibility predicate.
    Falls back to all chunks if the aspect is unknown (defensive)."""
    predicate = ASPECT_COMPATIBILITY.get(aspect, _always)
    return [c for c in chunks if predicate(c)]


# ---------------- Strategy protocol ----------------

class Strategy(Protocol):
    def select(
        self,
        chunks: list[Chunk],
        embeddings: Optional[np.ndarray],
        rng: random.Random,
    ) -> list[Chunk]: ...


# ---------------- Single-chunk fallback ----------------

@dataclass
class SingleChunk:
    def select(self, chunks, embeddings, rng) -> list[Chunk]:
        return [rng.choice(chunks)]


# ---------------- Same-doc remote (for remote_reference) ----------------

@dataclass
class SameDocRemote:
    n: int = 2
    min_position_gap: int = 3
    # 同一文書内で組む戦略なので異文書跨ぎは要らない (generate 側の早期棄却の対象外)
    requires_multi_doc: bool = False

    def select(self, chunks, embeddings, rng) -> list[Chunk]:
        # Group chunks by doc_id
        by_doc: dict[str, list[Chunk]] = {}
        for c in chunks:
            by_doc.setdefault(c.doc_id, []).append(c)
        eligible_docs = [doc for doc, lst in by_doc.items()
                         if len(lst) >= self.n and
                         (max(c.position for c in lst) - min(c.position for c in lst)) >= self.min_position_gap]
        if not eligible_docs:
            return [rng.choice(chunks)]

        doc = rng.choice(eligible_docs)
        same_doc = sorted(by_doc[doc], key=lambda c: c.position)
        anchor = rng.choice(same_doc)
        # Pick partners whose position is far enough away
        candidates = [c for c in same_doc
                      if abs(c.position - anchor.position) >= self.min_position_gap]
        if not candidates:
            return [anchor]
        # Prefer different top-level section
        anchor_top = anchor.section_path[0] if anchor.section_path else None
        diff_section = [c for c in candidates
                        if (c.section_path[0] if c.section_path else None) != anchor_top]
        pool = diff_section if diff_section else candidates
        rng.shuffle(pool)
        return [anchor] + pool[: self.n - 1]


# ---------------- Reference-following (for standards_reference / kg relation cells) ----------------

# 「第N条」「別表N」「附属書X」「第N章」は文書内で参照先を特定できる。
# 「第N項」は多くの条に存在して曖昧 (自条の項を指すことが多い) なので劣後させる。
_SPECIFIC_REF_RE = re.compile(r"第\d+条|別表|附属書|第\d+章")


# 同一文書内で「遠隔参照」と認める位置差の既定値。ReferenceFollow.min_same_doc_gap が
# この値の唯一の置き場で、生成側の早期棄却 (generate) と難易度判定
# (compute_retrieval_difficulty / score_retrieval_level) はここから引く。
# 2026-08-08 監査 rank 15: 同じ 10 が3箇所に別々の直値で入っており、片方だけ動かすと
# 「素材選びは通したのに生成が黙って捨てる」「生成は通すのに remote_reference=False で
# 出荷される」というラベルと実体の食い違いが静かに起きていた。チャンク粒度の違う
# コーパスでは位置差の意味自体が変わるので、set_min_same_doc_gap で動かせるようにする。
DEFAULT_MIN_SAME_DOC_GAP = 10


@dataclass
class ReferenceFollow:
    n: int = 2
    # True なら特定性の高い参照 (条/別表/章) を優先し、無ければ全参照を使う
    prefer_specific: bool = False
    # True なら異 doc の target だけを採用 (kg_poc の関係系セル用)。
    # 2026-06-17: 同 doc 内で参照番号がマッチすると、見かけ上 2 chunk だが
    # 実体は 1 doc の偽多 doc を量産していた。
    force_distinct_doc: bool = False
    # 本丸のチャンク単位化 (2026-07-01) に伴う追加: 同 doc の target でも
    # 位置がこの値以上離れていれば採用する。一文書コーパス (モデル就業規則)
    # では文書間参照が存在しないため、遠隔チャンク間の条文参照が本丸素材になる。
    # None なら従来どおり (force_distinct_doc に従う)。
    min_same_doc_gap: Optional[int] = None
    # 名指しに条番号が無いとき、埋め込み類似で参照先チャンクを推測してよいか。
    # 既定 False。「類似＝関係」を相手文書の中に持ち込み直すだけで検証手段が無い。
    allow_similarity_fallback: bool = False
    # 異文書跨ぎが本質の用途か (kg_poc の関係系セル)。generate の早期棄却は
    # この宣言を読むだけにする — 集合を手で写すと戦略を足したとき二重管理になる。
    requires_multi_doc: bool = False

    def _same_doc_ok(self, anchor: Chunk, c: Chunk) -> bool:
        """同一 doc の target を採用してよいか。"""
        if self.min_same_doc_gap is not None:
            return abs(c.position - anchor.position) >= self.min_same_doc_gap
        return not self.force_distinct_doc

    def select(self, chunks, embeddings, rng) -> list[Chunk]:
        anchors_with_refs = [c for c in chunks if c.references]
        if self.prefer_specific:
            specific = [c for c in anchors_with_refs
                        if any(_SPECIFIC_REF_RE.search(r) for r in c.references)]
            if specific:
                anchors_with_refs = specific
        if not anchors_with_refs:
            return [rng.choice(chunks)]
        # v3: 異文書対を作れるアンカー (台帳対・規格共有・明示参照) を優先抽選する。
        # 一様ランダムだと 3,000+ チャンク中の数十箇所がほぼ引かれず、
        # 本丸 (文書跨ぎ) の供給が偶然任せになるため、7割の確率で優先枠から引く。
        capable_ids = CROSS_DOC_POLICY.cross_capable_chunk_ids
        capable = [c for c in anchors_with_refs if c.chunk_id in capable_ids]
        if capable and rng.random() < 0.7:
            anchor = rng.choice(capable)
        else:
            anchor = rng.choice(anchors_with_refs)
        refs = anchor.references
        if self.prefer_specific:
            spec_refs = [r for r in refs if _SPECIFIC_REF_RE.search(r)]
            refs = spec_refs or refs
        # Find chunks whose text or section_path mentions any of anchor's refs
        targets: list[Chunk] = []
        for c in chunks:
            if c.chunk_id == anchor.chunk_id:
                continue
            same_doc = c.doc_id == anchor.doc_id
            if same_doc:
                if not self._same_doc_ok(anchor, c):
                    continue
                # 同一文書内: 条番号・別表番号などのローカル参照で辿ってよい
                blob = c.text + " " + " ".join(c.section_path)
                if any(ref in blob for ref in refs):
                    targets.append(c)
            else:
                # 異文書: 裸の条番号一致では繋がない。台帳対・規格番号共有・
                # 明示参照 (文書スコープ規則) を満たす対だけ採用する。
                # ただし明示参照は「文書」までしか決めないので、後段で
                # 参照先チャンクまで解決する (下の by_doc 処理)。
                if CROSS_DOC_POLICY.join_ok(anchor, c):
                    targets.append(c)

        # v3.1: 名指し由来の候補は文書単位で来る。その文書の中から実際に
        # 参照している1チャンクへ絞り込む (絞れなければその文書は使わない)。
        by_target_doc: dict[str, list[Chunk]] = {}
        for c in targets:
            if c.doc_id != anchor.doc_id:
                by_target_doc.setdefault(c.doc_id, []).append(c)
        if by_target_doc:
            # 条番号が書かれていない名指しは、相手文書内の埋め込み類似で絞る。
            sim_fn = None
            if embeddings is not None:
                row = {c.chunk_id: i for i, c in enumerate(chunks)}
                a_idx = row.get(anchor.chunk_id)
                if a_idx is not None:
                    def sim_fn(c: Chunk, _a=a_idx) -> float:  # noqa: E306
                        j = row.get(c.chunk_id)
                        return float(embeddings[j] @ embeddings[_a]) if j is not None else -1.0
            resolved: list[Chunk] = [c for c in targets if c.doc_id == anchor.doc_id]
            for doc, cands in by_target_doc.items():
                if len(cands) == 1:
                    resolved.extend(cands)   # 台帳対・規格共有で既に1点に決まっている
                    continue
                picked = CROSS_DOC_POLICY.resolve_named_target(
                    anchor, cands, similarity=sim_fn,
                    allow_similarity_fallback=self.allow_similarity_fallback,
                )
                if picked is not None:
                    resolved.append(picked)
            targets = resolved
        # Deduplicate while preserving order
        seen: dict[str, Chunk] = {}
        for c in targets:
            seen.setdefault(c.chunk_id, c)
        ordered = list(seen.values())
        if not ordered:
            # 異 doc target が見つからなければ anchor 単独で返す。
            # 同 doc で水増しすると generate 側の early-skip が機能しなくなる。
            return [anchor]
        rng.shuffle(ordered)
        return [anchor] + ordered[: self.n - 1]


# ---------------- Embedding-based pairing ----------------

@dataclass
class MultiDocByEmbedding:
    n: int = 2
    force_distinct_doc: bool = False
    # Cosine-similarity floor. Pairs below this aren't considered topically
    # related and get rejected (we fall back to single-chunk).
    # Raised from 0.3 to 0.5 after observing nonsense cross-domain pairs like
    # "時間外労働 × 標準トルク" caused by topic-mismatched corpus halves.
    sim_floor: float = 0.5
    # v2.0 追加: anchor[0] と anchor[1] に共通する参照番号 (条文/様式/別表番号)
    # が無いペアは「偶然の話題類似」なので reject する。
    # cross-doc の意味的接続を担保するため、現コーパスで cross-doc を
    # 名乗るには参照番号レベルの overlap を必須にする。
    require_article_overlap: bool = False
    # v2.0 追加: anchor[1] 候補が「リスト専用チャンク」(様式番号や別表番号を
    # 列挙しただけで本文を持たない目次的チャンク) なら除外する。
    # 09×45 の「条文本文 × 様式番号リスト」型デコイの素材を絶つため。
    exclude_list_chunks: bool = False
    # 異文書跨ぎが本質の用途か。force_distinct_doc とは別建てにしてある:
    # multi_source_integration / multi_hop は戦略としては同一文書での穴埋めを許す
    # (fallback) が、生成の題としては跨ぎが要るため。
    requires_multi_doc: bool = False

    def select(self, chunks, embeddings, rng) -> list[Chunk]:
        if embeddings is None or len(chunks) < self.n:
            return [rng.choice(chunks)]
        idx_anchor = rng.randrange(len(chunks))
        anchor = chunks[idx_anchor]
        # cosine similarity against anchor
        emb = embeddings  # already L2-normalized
        sims = emb @ emb[idx_anchor]
        sims[idx_anchor] = -1.0  # exclude self

        order = np.argsort(-sims)
        picked: list[Chunk] = [anchor]
        seen_docs = {anchor.doc_id}
        anchor_refs = set(anchor.references or [])
        for i in order:
            if sims[i] < self.sim_floor:
                break
            c = chunks[int(i)]
            if self.force_distinct_doc and c.doc_id in seen_docs:
                continue
            if self.exclude_list_chunks and _is_list_chunk(c):
                continue
            if self.require_article_overlap:
                if c.doc_id == anchor.doc_id:
                    # 同一文書内: 従来どおり参照番号の重なりで接続を確認
                    cand_refs = set(c.references or [])
                    if not (anchor_refs & cand_refs) and not _refs_in_text(anchor_refs, c.text):
                        continue
                else:
                    # 異文書: 裸の条番号・様式番号の一致は「別法令の同番号」を
                    # 拾う誤結合源なので根拠にしない。文書スコープ規則で判定。
                    if not CROSS_DOC_POLICY.join_ok(anchor, c):
                        continue
            picked.append(c)
            seen_docs.add(c.doc_id)
            if len(picked) >= self.n:
                break
        if len(picked) == 1 and not self.force_distinct_doc:
            # 多 doc 制約が緩い戦略 (multi_source_integration / multi_hop) では、
            # 同 doc の高類似チャンクで 2 個目を埋めて運用する (fallback)。
            # v3: 同一文書に限定する。異文書を無条件で埋めると、上の文書スコープ
            # 検査を素通りして双子文書 (白波×青灘の同名記録など) が対になる。
            for i in order:
                if sims[i] < self.sim_floor:
                    break
                c = chunks[int(i)]
                if c.chunk_id != anchor.chunk_id and c.doc_id == anchor.doc_id:
                    picked.append(c)
                    break
        # 注意: force_distinct_doc=True (multi_doc_reference 用) のときは、
        # 異 doc の partner が見つからなければ picked=[anchor] のまま返す。
        # 同 doc で水増しすると「2 chunk だが 1 doc」の偽多 doc を生み、
        # generate 側の early-skip が機能しなくなる (2026-06-17 修正)。
        return picked


# ---------------- Dispatch table ----------------

ASPECT_STRATEGIES: dict[str, Strategy] = {
    "multi_source_integration": MultiDocByEmbedding(
        n=3, force_distinct_doc=False, sim_floor=0.5, requires_multi_doc=True,
    ),
    # force_distinct_doc=True needs a stricter floor: cross-doc pairs naturally
    # have lower similarity, but we'd rather emit a single-chunk fallback than
    # synthesize a nonsense cross-domain question.
    # 2026-06-17: プラント技報 + 法令の 158 文書間で sim_floor=0.55 だと
    # ほぼ全試行が「異 doc ペア無し」で単一 doc にフォールバックし、
    # traceability/multi_hop 系が偽の単一 doc 問いに化けていた。
    # 0.40 まで下げると話題関連性は維持しつつ異 doc ペアが安定して見つかる。
    # 2026-06-26: 0.40 は緩すぎて「防毒呼吸具 × 液化ガス容器」のような
    # 無関係ドメインのペアを採り、Gemma が経歴ピボット (Xの経験者がYに従事…)
    # で強引に接続する不良型を量産。0.50 に上げて関連性を担保。
    # v2.0: require_article_overlap=True で「条文番号/様式番号一致が無い偶然
    # ペア」を切る + exclude_list_chunks=True で「様式リスト専用チャンク」を素材
    # から外す。これで様式番号デコイ・ペルソナ装飾の素材を絶つ。
    "multi_doc_reference":      MultiDocByEmbedding(
        n=2, force_distinct_doc=True, sim_floor=0.50,
        require_article_overlap=True, exclude_list_chunks=True,
        requires_multi_doc=True,
    ),
    "remote_reference":         SameDocRemote(n=2, min_position_gap=3),
    "standards_reference":      ReferenceFollow(n=2),
    "multi_hop":                MultiDocByEmbedding(
        n=2, force_distinct_doc=False, sim_floor=0.5,
        require_article_overlap=True, exclude_list_chunks=True,
        requires_multi_doc=True,
    ),
    # kg_poc の関係系セル用: 実在する参照 (第N条→本文/別表) で繋がった2チャンク。
    # 埋め込み類似ペアは「似ているが関係が無い」問いを量産して判定で落ちるため、
    # 関係を問う組では参照辿りを使う。
    # min_same_doc_gap: 本丸のチャンク単位化に伴い、同一文書内でも位置が既定値
    # 以上離れた遠隔参照ペアを許可 (一文書コーパスの条文参照が本丸素材になる)。
    "reference_follow":         ReferenceFollow(
        n=2, prefer_specific=True, force_distinct_doc=True,
        min_same_doc_gap=DEFAULT_MIN_SAME_DOC_GAP, requires_multi_doc=True,
    ),
}


# ---------------- 戦略表から導く事実 (手書きの集合・直値を作らない) ----------------

def multi_doc_required_aspects() -> set[str]:
    """異文書跨ぎが必須の素材キーを戦略表から導く。

    2026-08-08 監査 rank 26: この集合が generate の繰り返しの内側に手書きされており、
    多文書戦略を足すと早期棄却をすり抜けて「2チャンクだが1文書」の偽の跨ぎ問題が
    出荷され、逆に古いキーが残ると素材を過剰に捨てていた。
    """
    return {
        key for key, strat in ASPECT_STRATEGIES.items()
        if getattr(strat, "requires_multi_doc", False)
    }


def same_doc_gap_for(aspect: str) -> Optional[int]:
    """素材キーが同一文書内の遠隔対を認めるなら、その位置差を返す (認めなければ None)。"""
    return getattr(ASPECT_STRATEGIES.get(aspect), "min_same_doc_gap", None)


def remote_reference_gap() -> int:
    """難易度判定が「遠隔参照級」と数える位置差。素材選びと同じ値を使う。

    素材側 (ReferenceFollow.min_same_doc_gap) と別値になると、素材として通した対が
    remote_reference=False のまま出荷される (ラベルと実体の食い違い)。
    """
    gap = same_doc_gap_for("reference_follow")
    return DEFAULT_MIN_SAME_DOC_GAP if gap is None else gap


def set_min_same_doc_gap(gap: int) -> int:
    """同一文書内の遠隔参照とみなす位置差を差し替える (素材選び・難易度判定に同時に効く)。

    チャンク粒度の違うコーパスでは位置差の意味自体が変わるので、実行時に動かせる
    ようにしておく。戻り値は採用した値 (呼び出し側でログに出すこと)。
    """
    if gap < 1:
        raise ValueError(f"min_same_doc_gap は 1 以上 (指定値: {gap})")
    strat = ASPECT_STRATEGIES.get("reference_follow")
    if strat is not None:
        ASPECT_STRATEGIES["reference_follow"] = replace(strat, min_same_doc_gap=gap)
    return gap


# ---------------- Retrieval difficulty scoring (deterministic) ----------------
# 難易度はチャンク組成から決定論的に測る (乱数ラベルは廃止)。
# retrieval_level は LLM を介さず、anchors のトポロジだけで決まる。

_LOW_LOCALITY_GAP = 3       # 同一文書で位置がこれ以上離れたら低局所性
_LARGE_CHUNK_CHARS = 1200   # チャンク単体が大きい
# 遠隔参照級 (Hard) の位置差は remote_reference_gap() = 素材選びと同じ出所から引く。


def _doc_count(anchors: list[Chunk]) -> int:
    return len({c.doc_id for c in anchors})


def _max_position_gap(anchors: list[Chunk]) -> int:
    """同一文書内の最大位置差。複数文書なら各文書ごとに見て最大を返す。"""
    by_doc: dict[str, list[int]] = {}
    for c in anchors:
        by_doc.setdefault(c.doc_id, []).append(c.position)
    gaps = [max(ps) - min(ps) for ps in by_doc.values() if len(ps) >= 2]
    return max(gaps) if gaps else 0


def compute_retrieval_difficulty(anchors: list[Chunk]) -> RetrievalDifficulty:
    """チャンク組成から検索難易度の診断 bool 群を埋める (LLM 不要)。

    意味的判定が要る軸 (abstraction_discrepancy / vocabulary_mismatch) は
    質問依存なのでここでは触らず False のままにする。
    """
    gap = _max_position_gap(anchors)
    return RetrievalDifficulty(
        multi_doc=_doc_count(anchors) >= 2,
        multi_chunk=len(anchors) >= 2,
        low_locality=gap >= _LOW_LOCALITY_GAP,
        remote_reference=gap >= remote_reference_gap(),
        doc_volume_large=False,  # 元文書総量は anchors からは判定不能
        chunk_size_large=any(len(c.text) >= _LARGE_CHUNK_CHARS for c in anchors),
        abstraction_discrepancy=False,
        vocabulary_mismatch=False,
    )


def score_retrieval_level(anchors: list[Chunk]) -> str:
    """anchors のトポロジから Easy/Medium/Hard を決定論的に算出する。

    - Easy   : 単一チャンクで完結
    - Medium : 同一文書・複数チャンク (言い換え/局所外の参照が必要)
    - Hard   : 複数文書横断、または同一文書でも遠隔参照級に離れている
    """
    if len(anchors) <= 1:
        return "Easy"
    if _doc_count(anchors) >= 2:
        return "Hard"
    if _max_position_gap(anchors) >= remote_reference_gap():
        return "Hard"
    return "Medium"


# ---------------- Sampler ----------------

class AnchorSampler:
    """Holds chunks + embedding cache, dispatches per aspect.

    On each `select(aspect, rng)`:
      1. Filter the chunk pool by `ASPECT_COMPATIBILITY[aspect]` so that the
         downstream strategy only ever sees plausible candidates
         (e.g. simple_table strategy won't be handed a plain paragraph).
      2. Slice embeddings to match the filtered pool when present.
      3. If the filtered pool is empty, log a warning and fall back to the
         full pool — this surfaces in the per-batch summary so users can spot
         "this aspect can't be supported by this corpus".
    """

    # 1バッチ内で同一の異文書対から作れる問いの上限 (近似重複の量産防止)
    MAX_PER_DOC_PAIR = 2

    def __init__(self, chunks: list[Chunk], embeddings: Optional[np.ndarray]):
        self.chunks = chunks
        self.embeddings = embeddings
        self._fallback = SingleChunk()
        self._idx_by_chunk_id = {c.chunk_id: i for i, c in enumerate(chunks)}
        # Counters for per-batch reporting
        self.compat_misses: dict[str, int] = {}
        # Multi-chunk aspect でコーパスが組成を作れず skip した回数
        self.difficulty_misses: dict[str, int] = {}
        # v3: アンカー分散制約 — 消費済みのチャンク組と異文書対の使用回数。
        # 同じアンカー対から複数問作ると uniqueness でまとめて落ちる
        # (kitei 初回バッチで 15問中9問が近似重複落ち) ため、バッチ内で分散させる。
        self.used_chunk_sets: set[frozenset] = set()
        self.doc_pair_counts: dict[frozenset, int] = {}
        self.diversity_rejects: int = 0

    def commit(self, anchors: list[Chunk]) -> None:
        """QA が全チェックを通過したときに、そのアンカー組を使用済みとして確定する。

        select() の時点で確定すると、後段 (LLM 出力不正・捏造用語・単一doc など)
        で棄却された分まで枠を消費し、指定数に届かないまま終わる。
        """
        if not anchors:
            return
        self.used_chunk_sets.add(frozenset(c.chunk_id for c in anchors))
        doc_set = frozenset(c.doc_id for c in anchors)
        if len(doc_set) >= 2:
            self.doc_pair_counts[doc_set] = self.doc_pair_counts.get(doc_set, 0) + 1

    def _subset_embeddings(self, subset: list[Chunk]) -> Optional[np.ndarray]:
        if self.embeddings is None:
            return None
        indices = [self._idx_by_chunk_id[c.chunk_id] for c in subset]
        return self.embeddings[indices]

    def select(
        self, aspect: str, rng: random.Random, max_resamples: int = 20
    ) -> Optional[list[Chunk]]:
        """観点に応じたアンカーチャンク群を選ぶ。

        Multi-chunk 観点 (ASPECT_STRATEGIES に定義のある観点) は、その戦略が
        意図するトポロジ (複数チャンク) を作れるまで最大 `max_resamples` 回まで
        再抽選する。どうしても作れない場合 (例: 単一文書コーパスで multi_doc) は
        `difficulty_misses` を立てて None を返す → 呼び出し側は skip する。
        単一チャンク観点はそのまま 1 チャンクを返す。
        """
        compat = find_compatible_chunks(aspect, self.chunks)
        if not compat:
            self.compat_misses[aspect] = self.compat_misses.get(aspect, 0) + 1
            # No suitable chunk in corpus — fall back to full pool so generation
            # still proceeds, but quality may suffer.
            compat = self.chunks
        embeddings = self._subset_embeddings(compat)
        strat = ASPECT_STRATEGIES.get(aspect, self._fallback)

        # 単一チャンク観点: 1 回引いて返す。
        if aspect not in ASPECT_STRATEGIES:
            anchors = strat.select(compat, embeddings, rng)
            return anchors or [rng.choice(self.chunks)]

        # Multi-chunk 観点: 複数チャンクの組成を作れるまで再抽選。
        for _ in range(max_resamples):
            anchors = strat.select(compat, embeddings, rng)
            if not anchors or len(anchors) < 2:
                continue
            # v3: 分散制約 — 使用済みのチャンク組は再抽選、異文書対は上限まで
            chunk_set = frozenset(c.chunk_id for c in anchors)
            doc_set = frozenset(c.doc_id for c in anchors)
            if chunk_set in self.used_chunk_sets:
                self.diversity_rejects += 1
                continue
            if len(doc_set) >= 2 and self.doc_pair_counts.get(doc_set, 0) >= self.MAX_PER_DOC_PAIR:
                self.diversity_rejects += 1
                continue
            # ここでは予約しない。QA が全チェックを通過してから commit() で確定する
            # (棄却された QA のぶんで枠を食い潰し、指定数・指定 mix に届かなくなる
            #  のを防ぐ)。同一呼び出し内での重複抽選は上の判定で弾かれる。
            return anchors
        # コーパスがこの観点の組成を作れない → skip させる。
        self.difficulty_misses[aspect] = self.difficulty_misses.get(aspect, 0) + 1
        return None


# ---------------- Embedding helper ----------------

def compute_embeddings(chunks: list[Chunk]) -> Optional[np.ndarray]:
    """Embed all chunks via the vLLM embedding endpoint. Returns None on failure."""
    from .llm import embed  # local import keeps module light when unused

    try:
        vectors = embed([c.text for c in chunks])
    except Exception as e:
        print(f"[sampling] embedding failed, multi-chunk strategies will fall back: {e}")
        return None
    arr = np.array(vectors, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms
