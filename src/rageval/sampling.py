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

import random
import re
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np

from .schema import Chunk, RetrievalDifficulty


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


_STANDARDS_RE = re.compile(r"^(?:JIS|ISO|IEC|ASME)")


def _has_standards_ref(chunk: Chunk) -> bool:
    return any(_STANDARDS_RE.match(r) for r in chunk.references)


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
        anchor = rng.choice(anchors_with_refs)
        refs = anchor.references
        if self.prefer_specific:
            spec_refs = [r for r in refs if _SPECIFIC_REF_RE.search(r)]
            refs = spec_refs or refs
        # Find chunks whose text or section_path mentions any of anchor's refs
        targets: list[Chunk] = []
        for ref in refs:
            for c in chunks:
                if c.chunk_id == anchor.chunk_id:
                    continue
                if c.doc_id == anchor.doc_id and not self._same_doc_ok(anchor, c):
                    continue
                blob = c.text + " " + " ".join(c.section_path)
                if ref in blob:
                    targets.append(c)
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
                cand_refs = set(c.references or [])
                if not (anchor_refs & cand_refs) and not _refs_in_text(anchor_refs, c.text):
                    # 参照番号レベルの重なりが無い → 意味的接続が弱い → skip
                    continue
            picked.append(c)
            seen_docs.add(c.doc_id)
            if len(picked) >= self.n:
                break
        if len(picked) == 1 and not self.force_distinct_doc:
            # 多 doc 制約が緩い戦略 (multi_source_integration / multi_hop) では、
            # 同 doc の高類似チャンクで 2 個目を埋めて運用する (fallback)。
            for i in order:
                if sims[i] < self.sim_floor:
                    break
                c = chunks[int(i)]
                if c.chunk_id != anchor.chunk_id:
                    picked.append(c)
                    break
        # 注意: force_distinct_doc=True (multi_doc_reference 用) のときは、
        # 異 doc の partner が見つからなければ picked=[anchor] のまま返す。
        # 同 doc で水増しすると「2 chunk だが 1 doc」の偽多 doc を生み、
        # generate 側の early-skip が機能しなくなる (2026-06-17 修正)。
        return picked


# ---------------- Dispatch table ----------------

ASPECT_STRATEGIES: dict[str, Strategy] = {
    "multi_source_integration": MultiDocByEmbedding(n=3, force_distinct_doc=False, sim_floor=0.5),
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
    ),
    "remote_reference":         SameDocRemote(n=2, min_position_gap=3),
    "standards_reference":      ReferenceFollow(n=2),
    "multi_hop":                MultiDocByEmbedding(
        n=2, force_distinct_doc=False, sim_floor=0.5,
        require_article_overlap=True, exclude_list_chunks=True,
    ),
    # kg_poc の関係系セル用: 実在する参照 (第N条→本文/別表) で繋がった2チャンク。
    # 埋め込み類似ペアは「似ているが関係が無い」問いを量産して判定で落ちるため、
    # 関係を問う組では参照辿りを使う。
    # min_same_doc_gap=10: 本丸のチャンク単位化に伴い、同一文書内でも位置が
    # 10 以上離れた遠隔参照ペアを許可 (一文書コーパスの条文参照が本丸素材になる)。
    "reference_follow":         ReferenceFollow(
        n=2, prefer_specific=True, force_distinct_doc=True, min_same_doc_gap=10,
    ),
}


# ---------------- Retrieval difficulty scoring (deterministic) ----------------
# 難易度はチャンク組成から決定論的に測る (乱数ラベルは廃止)。
# retrieval_level は LLM を介さず、anchors のトポロジだけで決まる。

_LOW_LOCALITY_GAP = 3       # 同一文書で位置がこれ以上離れたら低局所性
_REMOTE_GAP = 10            # さらに離れたら遠隔参照級 (Hard)
_LARGE_CHUNK_CHARS = 1200   # チャンク単体が大きい


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
        remote_reference=gap >= _REMOTE_GAP,
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
    if _max_position_gap(anchors) >= _REMOTE_GAP:
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

    def __init__(self, chunks: list[Chunk], embeddings: Optional[np.ndarray]):
        self.chunks = chunks
        self.embeddings = embeddings
        self._fallback = SingleChunk()
        self._idx_by_chunk_id = {c.chunk_id: i for i, c in enumerate(chunks)}
        # Counters for per-batch reporting
        self.compat_misses: dict[str, int] = {}
        # Multi-chunk aspect でコーパスが組成を作れず skip した回数
        self.difficulty_misses: dict[str, int] = {}

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
            if anchors and len(anchors) >= 2:
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
