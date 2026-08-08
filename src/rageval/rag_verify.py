"""Vector RAG ground-truth verification.

For each QA:
  1. Embed the question, retrieve top-k chunks by cosine similarity
  2. Ask a RAG model (typically the generation model) to answer using only those chunks
  3. Use a judge LLM to compare the candidate answer to the ground truth
  4. Set `qa.rag_verification = RAGVerification(...)`

これにより「生成→判定」の LLM 自己評価ループの外にある信号
(vector RAG が実際に解けるか) をデータセットに持ち込める。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .chunker import load_chunks
from .evidence import grounded_fraction, match_rationales
from .llm import LLMError, _parse_json, embed as llm_embed, generate as llm_generate
from .prompts import load_prompt
from .sampling import compute_embeddings
from .schema import Chunk, QAItem, RAGAnswerMatch, RAGVerification

DEFAULT_PROMPT = "prompts/rag_verify.md"
DEFAULT_TOP_K = 10   # 2026-08-04: 本番 KG-OFF (fetch20→rerank→top10) に合わせた

# モデルが「回答不能」の指示に従わず別表現で返す場合があるので、
# それらをまとめて「答えなし」扱いにして judge を呼ばないようにする。
# (judge は表現の揺れを許容する設計なので、ここで弾かないと
#  「不明」「該当なし」を partial や match と誤判定する余地が出る。)
_UNABLE_TOKENS: frozenset[str] = frozenset({
    "回答不能",
    "不明",
    "わからない",
    "分からない",
    "判断できない",
    "判断できません",
    "該当なし",
    "回答できません",
    "記載なし",
    "情報なし",
    "記述なし",
    "文書からは特定できません",
    "文書からは特定できない",
    "answernotavailable",
    "notavailable",
    "unknown",
})

# 文末記号・空白・かぎ括弧を落としてから語彙照合するための削除候補。
_UNABLE_STRIP_CHARS: tuple[str, ...] = (
    " ", "　", "。", "、", "．", "，",
    "「", "」", "『", "』", '"', "'", ".", ",",
)


def _is_unable_answer(text: str) -> bool:
    """rag_answer が実質「答えなし」を意味するかを判定する。

    Gemma 等は「回答不能と返せ」と指示しても「不明」「わからない」
    「文書からは特定できません」等の別表現を返しがちなので、
    まとめてここで吸収して judge 呼び出しを省く。
    """
    if not text:
        return True
    stripped = text.strip()
    for ch in _UNABLE_STRIP_CHARS:
        stripped = stripped.replace(ch, "")
    # 整形後に空文字になるもの (空白や記号のみ) も「答えなし」とみなす
    if not stripped:
        return True
    return stripped.lower() in _UNABLE_TOKENS


_SECTION_RE = re.compile(
    r"^##\s*\[(?P<name>[A-Z_]+)\]\s*\n(?P<body>.*?)(?=^##\s*\[|^---\s*$|\Z)",
    re.DOTALL | re.MULTILINE,
)


def _split_sections(body: str) -> dict[str, str]:
    return {m.group("name"): m.group("body").strip() for m in _SECTION_RE.finditer(body)}


LLMCaller = Callable[..., Any]
EmbedCaller = Callable[[list[str]], list[list[float]]]


def _parse_json_safely(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    parsed = _parse_json(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise LLMError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed


def _normalize_question_vec(vec: list[float]) -> np.ndarray:
    arr = np.array(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm == 0:
        # 全成分0の埋め込みをそのまま返すと、後段の内積が全0になり argsort で
        # 先頭チャンクが常に top-k 先頭に来る無言の誤りを生む。空質問または
        # 埋め込み API の異常の徴候として例外に倒し、上位で retrieved=[] 経路へ。
        raise LLMError(
            "質問の埋め込みベクトルの大きさが0です。"
            "空の質問または埋め込み API の異常の可能性があります。"
        )
    return arr / norm


def _format_chunks_block(chunks: list[Chunk]) -> str:
    """Render retrieved chunks for the RAG_ANSWER prompt (numbered, page-tagged)."""
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        section = " > ".join(c.section_path) if c.section_path else "(no section)"
        page_str = f"p.{c.page}" if c.page is not None else "p.?"
        header = f"--- 結果 {i}: {c.doc_id} ({page_str}, {c.chunk_id}) [{section}] ---"
        parts.append(f"{header}\n{c.text}")
    return "\n\n".join(parts)


# ---------------- BM25 (語彙検索) ----------------
# 実運用の RAG はハイブリッド検索 (密ベクトル + 語彙) がデファクトなので、
# スクリーニング (難易度実測・型確定) も同じ基準で行う。密ベクトルだけを
# 基準にすると、条番号や専門語の一致で語彙検索なら一発で当たる問いまで
# 「検索で届かない = KG が要る」と誤認し、KG の優位を過大評価する。

import math
import unicodedata
from collections import Counter

_LATIN_NUM_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _bm25_tokens(text: str) -> list[str]:
    """日本語向けの軽量トークナイザ (形態素解析なし・依存追加なし)。

    英数字の連続は1トークン、それ以外 (CJK) は文字バイグラム。
    条番号 (「第101条」→ '101' + 「第1」「1条」系のバイグラム) や規格番号
    (JIS B 8265 → 'jis' '8265') の一致が拾えれば十分。
    """
    s = unicodedata.normalize("NFKC", text or "").lower()
    tokens: list[str] = []
    for m in _LATIN_NUM_RE.finditer(s):
        tokens.append(m.group(0))
    cjk_runs = _LATIN_NUM_RE.sub(" ", s)
    for run in cjk_runs.split():
        run = "".join(ch for ch in run if not ch.isspace())
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class BM25Index:
    """Okapi BM25。チャンク集合に対して1回だけ構築する。"""

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._tfs: list[Counter] = []
        self._dls: list[int] = []
        df: Counter = Counter()
        for c in chunks:
            toks = _bm25_tokens(c.text)
            tf = Counter(toks)
            self._tfs.append(tf)
            self._dls.append(len(toks))
            df.update(tf.keys())
        n = max(len(chunks), 1)
        self._avgdl = (sum(self._dls) / n) if n else 1.0
        self._idf = {t: math.log((n - d + 0.5) / (d + 0.5) + 1.0)
                     for t, d in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = Counter(_bm25_tokens(query))
        out = np.zeros(len(self._tfs), dtype=float)
        for tok in q:
            idf = self._idf.get(tok)
            if idf is None:
                continue
            for i, tf in enumerate(self._tfs):
                f = tf.get(tok)
                if not f:
                    continue
                dl = self._dls[i] or 1
                out[i] += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
        return out


# リランク前の候補プール。2026-08-04 に 20→100 へ引き上げ (ユーザ判断):
# 「KG しか解けない」認定は、埋め込み候補100をリランカーで絞っても届かない、
# という現実的に最強の構成に耐えて初めて名乗れる。実測ではプール到達は
# fetch50 で飽和しており、100 に上げても偽の multi_hop を増やさない。
DEFAULT_FETCH_K = 100


def _rerank_via_api(query: str, docs: list[str], *, endpoint: str, model: str,
                    timeout: float = 60.0) -> list[float]:
    """交差エンコーダのリランク API (/v1/rerank) を呼び、文書ごとの関連度を返す。"""
    import json as _json
    import urllib.request

    body = _json.dumps({"model": model, "query": query, "documents": docs}).encode("utf-8")
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/rerank", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = _json.loads(r.read().decode("utf-8"))
    scores = [0.0] * len(docs)
    for item in data.get("results", []):
        scores[int(item["index"])] = float(item.get("relevance_score", 0.0))
    return scores


def build_reranker_from_env():
    """RERANK_ENDPOINT / RERANK_MODEL が設定されていればリランカー関数を返す。

    本番 RAG (KG-OFF) は「密+疎 → 交差エンコーダで並べ直し」の2段構成なので、
    スクリーニングも同じ基準に揃える (2026-08-04: リランク無し top5 の選別が
    本番 top10+リランクに全て解かれ、multi_hop 認定が過大だったため)。
    """
    endpoint = os.getenv("RERANK_ENDPOINT", "").strip()
    model = os.getenv("RERANK_MODEL", "").strip()
    if not endpoint or not model:
        return None
    def rerank(query: str, docs: list[str]) -> list[float]:
        return _rerank_via_api(query, docs, endpoint=endpoint, model=model)
    return rerank


def _rrf_fuse(rank_lists: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion。rank_lists は各手法の順位 (0始まり) の配列。"""
    score = np.zeros(len(rank_lists[0]), dtype=float)
    for ranks in rank_lists:
        score += 1.0 / (k + 1 + ranks)
    return score


def _retrieve_topk(
    question: str,
    chunks: list[Chunk],
    chunk_embeddings: np.ndarray,
    top_k: int,
    *,
    embed: EmbedCaller,
    bm25: Optional[BM25Index] = None,
    reranker=None,
    fetch_k: int = DEFAULT_FETCH_K,
) -> list[Chunk]:
    """質問に対する top-k チャンク。bm25 で RRF 融合、reranker で2段目の並べ直し。

    同点時は元の順序を保つため stable ソートを使う (埋め込み量子化等で同値が
    出たときに run ごとに retrieved_chunk_ids が揺れて rag_answer まで
    変わるのを防ぐ)。
    """
    if not chunks or top_k <= 0:
        return []
    q_vecs = embed([question])
    if not q_vecs:
        return []
    q_norm = _normalize_question_vec(q_vecs[0])
    if q_norm.shape[0] != chunk_embeddings.shape[1]:
        raise LLMError(
            f"embedding dim mismatch: question={q_norm.shape[0]} "
            f"chunks={chunk_embeddings.shape[1]}"
        )
    sims = chunk_embeddings @ q_norm
    k = min(top_k, len(chunks))
    if bm25 is None:
        top_idx = np.argsort(-sims, kind='stable')[:k]
        return [chunks[int(i)] for i in top_idx]
    # ハイブリッド: 密・語彙それぞれの順位を RRF で融合
    dense_rank = np.empty(len(chunks), dtype=int)
    dense_rank[np.argsort(-sims, kind='stable')] = np.arange(len(chunks))
    lex = bm25.scores(question)
    lex_rank = np.empty(len(chunks), dtype=int)
    lex_rank[np.argsort(-lex, kind='stable')] = np.arange(len(chunks))
    fused = _rrf_fuse([dense_rank, lex_rank])
    if reranker is None:
        top_idx = np.argsort(-fused, kind='stable')[:k]
        return [chunks[int(i)] for i in top_idx]
    # 2段目: 候補プール fetch_k を交差エンコーダで並べ直して top_k に選抜
    pool_idx = np.argsort(-fused, kind='stable')[:min(fetch_k, len(chunks))]
    pool = [chunks[int(i)] for i in pool_idx]
    scores = reranker(question, [c.text for c in pool])
    order = sorted(range(len(pool)), key=lambda i: -scores[i])
    return [pool[i] for i in order[:k]]


_CHUNK_ID_SUFFIX_RE = re.compile(r"__c\d+$")
_WS_RE = re.compile(r"\s+")


def _normalize_doc_id(raw: str) -> str:
    """根拠の doc_id にチャンクID('foo__c0023')が紛れたときに末尾を落とす。"""
    return _CHUNK_ID_SUFFIX_RE.sub("", raw)


def _normalize_for_match(s: str) -> str:
    """空白を全て削って比較用に整える (filter._normalize_for_match と同規則)。"""
    return _WS_RE.sub("", s)


def _retrieval_hits(qa: QAItem, retrieved: list[Chunk]) -> tuple[bool, bool, float]:
    """根拠が上位 k にどう当たったかを (文書一致, チャンク一致, 命中率) で返す。

    - 文書一致: 根拠と同じ doc_id のチャンクが1個でも上位 k に居れば真。
      技報のように1文書が数十チャンクに割れるコーパスでは、根拠とは
      無関係な章のチャンクが拾われただけで真になるので、検索健全性の
      判定指標として単独で使うと過大評価になる。
    - チャンク一致: **全ての必須根拠**が上位 k の中で逐語一致したときだけ真。
      2026-08-01 まで「いずれか1件が当たれば真」だったため、3根拠中1件しか
      取れていない問いが「命中 → 検索 Easy」と記録されていた (実測で
      複数根拠14問中5問が該当)。複数根拠を要する問いは、全部取れて初めて
      RAG が答えに到達できるので、全件一致を基準にする。
    - 命中率: 逐語一致した根拠の割合。部分命中を捨てずに持つ。

    照合は evidence.match_rationales に委譲し、**根拠ごとに、その根拠が
    名乗る文書のチャンク内だけ**で見る (別文書への取り違え・連結境界の
    偶然一致を防ぐ)。根拠が空のときは全て偽/0.0。
    """
    if not qa.rationale:
        return False, False, 0.0

    rationale_doc_ids = {_normalize_doc_id(r.doc_id) for r in qa.rationale}
    rationale_chunk_id_hints = {r.doc_id for r in qa.rationale}

    retrieved_doc_ids = {c.doc_id for c in retrieved}
    retrieved_chunk_ids = {c.chunk_id for c in retrieved}

    hit_doc = bool(rationale_doc_ids & retrieved_doc_ids)
    if not hit_doc and (rationale_chunk_id_hints & retrieved_chunk_ids):
        hit_doc = True

    matches = match_rationales(qa.rationale, retrieved)
    ratio = grounded_fraction(matches)
    hit_chunk = bool(matches) and all(m.matched for m in matches)

    return hit_doc, hit_chunk, ratio


def _retrieval_hit(qa: QAItem, retrieved: list[Chunk]) -> bool:
    """旧 API。文書一致 OR チャンク一致のどちらかが真なら真。

    既存テスト・既存呼び出し点との互換のために残す。新しいコードは
    `_retrieval_hits` を使い、文書一致とチャンク一致を分けて持つこと。
    """
    hit_doc, hit_chunk, _ = _retrieval_hits(qa, retrieved)
    return hit_doc or hit_chunk


def rag_verify_qa(
    qa: QAItem,
    *,
    chunks: list[Chunk],
    chunk_embeddings: np.ndarray,
    top_k: int,
    rag_model: str,
    judge_model: str,
    rag_section: str,
    judge_section: str,
    llm: LLMCaller,
    embed: EmbedCaller,
    bm25: Optional[BM25Index] = None,
    reranker=None,
    fetch_k: int = DEFAULT_FETCH_K,
) -> Optional[RAGVerification]:
    """Run vector RAG once and judge the answer.

    検索そのものが失敗した場合は **None を返す** (未検証として保留)。
    回答生成や judge が失敗した場合は judge_ok=False の RAGVerification を返す。
    どちらも「vector RAG が解けなかった (no_match)」という意味の判定とは
    区別する — 障害が KG候補ラベルに化けるのを防ぐため。
    """
    # Step 1: retrieve
    # 検索が落ちた場合は「検索も回答も全滅」に見えるレコードを作らない。
    # 作ってしまうと障害が retrieval_level=Hard / answer_match=no_match という
    # 意味のあるラベルに化ける (= 障害ほど「良いデータ」が増えて見える)。
    try:
        retrieved = _retrieve_topk(qa.question, chunks, chunk_embeddings, top_k,
                                   embed=embed, bm25=bm25, reranker=reranker,
                                   fetch_k=fetch_k)
    except Exception as e:
        print(f"[rag-verify] {qa.qa_id} retrieve failed (未検証として保留): "
              f"{type(e).__name__}: {e}")
        return None

    retrieved_ids = [c.chunk_id for c in retrieved]
    hit_doc, hit_chunk, hit_ratio = _retrieval_hits(qa, retrieved)

    # Step 2: RAG answer
    rag_answer = ""
    judge_ok = True
    if retrieved:
        chunks_block = _format_chunks_block(retrieved)
        rag_prompt = (
            rag_section
            .replace("{chunks_block}", chunks_block)
            .replace("{question}", qa.question)
        )
        try:
            raw = llm(
                prompt=rag_prompt,
                model=rag_model,
                temperature=0.0,
                max_tokens=4096,
                force_json=True,
            )
            rag_answer = _parse_json_safely(raw).get("answer", "").strip()
        except Exception as e:
            # APIConnectionError 等で全件落ちないように広めに捕捉。
            # ただし「回答できなかった」ではなく「聞けなかった」なので、
            # judge_ok=False にして後段の KG候補集計から外す。
            print(f"[rag-verify] {qa.qa_id} rag answer failed: {type(e).__name__}: {e}")
            judge_ok = False

    # Step 3: judge
    # match は judge が想定どおりに返したときだけ書き換える。
    # judge を呼ばなかった/落ちた/想定外を返したときはすべて 'no_match' のまま残るが、
    # 後段でこの3者を区別できるよう judge_raw (生の verdict) と judge_ok (照合が
    # 成立したか) を保存する。judge_ok=False の no_match は「vector RAG が解けなかった」
    # 証拠にはならないので、KG候補として数えてはならない。
    match: RAGAnswerMatch = "no_match"
    judge_raw: Optional[str] = None
    if judge_ok and not _is_unable_answer(rag_answer):
        judge_prompt = (
            judge_section
            .replace("{ground_truth}", qa.answer)
            .replace("{candidate}", rag_answer)
        )
        try:
            raw = llm(
                prompt=judge_prompt,
                model=judge_model,
                temperature=0.0,
                max_tokens=4096,
                force_json=True,
            )
            verdict = _parse_json_safely(raw).get("match", "").strip().lower()
            judge_raw = verdict
            if verdict in ("match", "partial", "no_match"):
                match = verdict  # type: ignore[assignment]
            else:
                # 想定外の verdict (例: 'matched', 'partial_match', 'unknown', 空文字)。
                # 黙って no_match に倒すと、--require-rag-fail などの後段ふるい分けが
                # 「judge が崩れた QA」を正規の no_match として通してしまうので警告を出す。
                print(
                    f"[rag-verify] {qa.qa_id} unexpected judge verdict: {verdict!r}"
                )
                judge_ok = False
        except Exception as e:
            print(f"[rag-verify] {qa.qa_id} judge failed: {type(e).__name__}: {e}")
            judge_ok = False

    return RAGVerification(
        top_k=top_k,
        retrieved_chunk_ids=retrieved_ids,
        retrieval_mode=("hybrid_rerank" if (bm25 is not None and reranker is not None)
                        else "hybrid" if bm25 is not None else "dense"),
        retrieval_hit_doc=hit_doc,
        retrieval_hit_chunk=hit_chunk,
        retrieval_hit_ratio=hit_ratio,
        # 旧フィールドは「どちらかが真」を入れて後方互換を保つ
        retrieval_hit=hit_doc or hit_chunk,
        rag_answer=rag_answer,
        answer_match=match,
        judge_raw=judge_raw,
        judge_ok=judge_ok,
        rag_model=rag_model,
        judge_model=judge_model,
        verified_at=datetime.now(),
    )


def apply_measured_retrieval_level(qa: QAItem) -> bool:
    """rag-verify の実測から retrieval_level を上書きする (P4: ラベルの実測化)。

    生成時の retrieval_level はチャンク組成からの見込み値で、実際に検索が
    届くかは実走するまで分からない。実測の対応:
    - 全ての必須根拠が top-k に入った (hit_ratio=1.0) → Easy (ベクトル検索で届く)
    - 一部の根拠だけ届いた / 文書までは当たるがチャンクは外れ → Medium
    - 文書にも届かない → Hard

    2026-08-01: 以前は「いずれか1件の根拠が当たれば Easy」だったため、
    3根拠中1件しか取れていない問いまで Easy に落ちていた。部分命中は
    Medium として扱う。書き換えた場合は履歴を difficulty_rationale に残す。
    """
    rv = qa.rag_verification
    if rv is None:
        return False
    hit_doc = rv.retrieval_hit_doc or rv.retrieval_hit
    ratio = rv.retrieval_hit_ratio
    if rv.retrieval_hit_chunk or ratio >= 1.0:
        measured = "Easy"
    elif ratio > 0.0 or hit_doc:
        measured = "Medium"
    else:
        measured = "Hard"
    if qa.retrieval_level == measured:
        return False
    old = qa.retrieval_level
    qa.retrieval_level = measured
    note = (f"[実測 {datetime.now():%Y-%m-%d}] 検索難易度 {old}→{measured} "
            f"(rag-verify: 根拠命中 {ratio:.0%}, hit_doc={bool(hit_doc)})")
    qa.difficulty_rationale = (qa.difficulty_rationale + " / " + note).strip(" /")
    return True


def sync_kg_query_type_from_retrieval(qa: QAItem) -> bool:
    """KG 経路由来 QA の型 (multi_hop / single_fact) を検索必要性で確定する。

    LOO (回答内容の必要性) は連鎖型を系統的に降格する (2026-08-01 実測 17/18):
    連鎖型は回答の固有値が終端チャンクに集中するため、終端だけで回答が
    導出でき、LOO は始点側を「不要」と判定してしまう。しかし始点側の価値は
    回答の導出ではなく**検索上の必要性** (どの規定が該当するかの特定) にある。

    そこで「質問だけのベクトル検索 (top-k) で終端の根拠チャンクが直接引けたか」
    で判定する:
      - 直接引けた → 経路を辿る必要が無い → single_fact
      - 引けない → 経路を辿らないと終端に到達できない → multi_hop (復元を含む)

    LOO と構造判別 (classify_evidence_structure) は束ね検出用として残る。
    """
    kp, rv = qa.kg_path, qa.rag_verification
    if kp is None or rv is None or len(kp.edges) < 2:
        return False
    # 「終端」概念は連鎖にしかない。星 (aggregation) や除外 (negation) の合否は
    # この関数では切らず、型別表の認定ゲート certify_kg_items が切る
    # (2026-08-08: それまで星は shape != "chain" でここを抜けたきり、設計書が
    #  定める被覆率の合格線を掛ける工程がどこにも無く、被覆率 1.0 の星が
    #  aggregation のまま出荷されていた)。
    if kp.shape != "chain":
        return False
    # decoy は「関係が無い」ことが正解で、終端の検索可否に意味が無い
    if getattr(kp, "source", "kg_covered") == "kg_decoy":
        return False
    # 型の再判定は multi_hop ⇄ single_fact の間だけ。traceability 等の型は
    # 設計で決まっており、検索必要性で上書きしない (2026-08-03: trace が
    # relabel される事故で追加)
    if qa.kg_query_type not in ("multi_hop", "single_fact"):
        return False
    terminal = kp.edges[-1].source_chunk_id
    direct = terminal in rv.retrieved_chunk_ids
    target = "single_fact" if direct else "multi_hop"
    if qa.kg_query_type == target:
        return False
    old = qa.kg_query_type
    qa.kg_query_type = target
    detail = ("終端チャンクが top-k に直接命中 (経路を辿る必要なし)" if direct
              else "終端チャンクは top-k に入らず経路経由でのみ到達可能")
    note = f"[型確定 {datetime.now():%Y-%m-%d}] {detail} のため {old}→{target}"
    qa.difficulty_rationale = (qa.difficulty_rationale + " / " + note).strip(" /")
    return True


# ---------------- 認定ゲート (2026-08-08 設計書「認定ゲート」節) ----------------
# 原則1「ラベルは実測で確定する」は、実測を記録することではなく**実測で合否を切る**こと。
# 合否は型別表 CERTIFY_BY_TYPE の1箇所だけで切り、結果は KGPathProvenance の
# certified / certification_reason に残す。**不認定でも項目は捨てない** —
# 対照群として持ち続け、「KG が要る」の集計から外すだけ。

# 設計書の合格線はすべて厳格2段検索 (密+BM25 → fetch100 → 交差エンコーダ → top10)
# の実測を前提にしている。密のみ・リランク無しの実測で「検索で届かない」を認めると、
# 語彙一致で解ける問いまで KG 必要と誤認して優位を過大評価するので、
# 検索の実測を根拠にする型 (multi_hop / aggregation) は方式もここで検査する。
STRICT_RETRIEVAL_MODE = "hybrid_rerank"

# aggregation の合格線 (設計書 型別定義表・認定ゲート表): 要素チャンクの
# top-k 被覆率がこの値**未満**であること。1.0 = 全要素が検索だけで揃う星は
# 「KG が要る」を名乗れない (v1 の3問はここを 3/3 で破ったまま出荷していた)。
AGGREGATION_COVERAGE_CEILING = 1.0

_NO_MEASUREMENT_REASON = "実測 (rag-verify) が無いため認定できない"


def _weak_mode_reason(rv: RAGVerification) -> str:
    return (f"実測が厳格2段検索でない (retrieval_mode={rv.retrieval_mode}) "
            f"ため認定できない")


def aggregation_coverage(qa: QAItem) -> Optional[float]:
    """星型の要素チャンクが top-k にどれだけ揃ったか (被覆率) を返す。

    要素チャンクの正本は `kg_path.edges[*].source_chunk_id` (枝ごとの含意を
    確認した出典)。rationale 本文の逐語一致 (retrieval_hit_ratio) とは別物で、
    「検索だけで列挙が揃うか」を見るこの型の合格線はチャンクの到達で測る。
    要素が特定できないときは None (認定側で不認定に倒す)。
    """
    kp, rv = qa.kg_path, qa.rag_verification
    if kp is None or rv is None:
        return None
    elements = list(dict.fromkeys(e.source_chunk_id for e in kp.edges))
    if not elements:
        return None
    got = set(rv.retrieved_chunk_ids)
    return sum(1 for c in elements if c in got) / len(elements)


def _certify_multi_hop(qa: QAItem) -> tuple[bool, str]:
    """終端チャンク ∉ top-k (検索必要性)。命中していれば single_fact へ降格済み。"""
    rv, kp = qa.rag_verification, qa.kg_path
    if rv is None:
        return False, _NO_MEASUREMENT_REASON
    if rv.retrieval_mode != STRICT_RETRIEVAL_MODE:
        return False, _weak_mode_reason(rv)
    if kp is None or not kp.edges:
        return False, "経路の終端チャンクが特定できない (kg_path.edges が空)"
    terminal = kp.edges[-1].source_chunk_id
    if terminal in rv.retrieved_chunk_ids:
        return False, f"終端チャンクが top-{rv.top_k} に直接命中 (経路を辿る必要がない)"
    return True, f"終端チャンクが厳格2段検索の top-{rv.top_k} に入らない"


def _certify_single_fact(qa: QAItem) -> tuple[bool, str]:
    return True, "降格先そのもの (常に認定)"


def _certify_aggregation(qa: QAItem) -> tuple[bool, str]:
    """要素チャンクの top-k 被覆率 < 100%。揃ってしまう星は対照群に落とす。"""
    rv = qa.rag_verification
    if rv is None:
        return False, _NO_MEASUREMENT_REASON
    if rv.retrieval_mode != STRICT_RETRIEVAL_MODE:
        return False, _weak_mode_reason(rv)
    coverage = aggregation_coverage(qa)
    if coverage is None:
        return False, "要素チャンクが特定できない (kg_path.edges が空)"
    if coverage < AGGREGATION_COVERAGE_CEILING:
        return True, (f"要素チャンクの top-{rv.top_k} 被覆率 {coverage:.0%} "
                      f"< {AGGREGATION_COVERAGE_CEILING:.0%}")
    # 理由文は集計画面・端末にそのまま出るので、cp932 で落ちる記号 (em ダッシュ等) は使わない
    return False, (f"要素チャンクが全て top-{rv.top_k} に揃う (被覆率 {coverage:.0%}) "
                   f"ため、検索だけで解ける星として不認定")


def _certify_traceability(qa: QAItem) -> tuple[bool, str]:
    """識別子の照合は段3 (生成時の型別検品) が済ませている。"""
    kp = qa.kg_path
    if kp is None or not kp.edges:
        return False, "経路の辺が無く出典識別子を照合できない"
    return True, "回答に各辺の出典識別子が全て含まれることを段3で判定済み"


def _certify_negation_exhaustive(qa: QAItem) -> tuple[bool, str]:
    """除外・禁止文言の原文含意は段0/段1で済ませている。"""
    kp = qa.kg_path
    if kp is None or not kp.edges:
        return False, "除外辺が無く原文の除外・禁止文言を照合できない"
    return True, "原文の除外・禁止文言の含意を段0/段1で判定済み"


def _certify_kg_missing(qa: QAItem) -> tuple[bool, str]:
    return True, "実在参照であることが機械供給の条件そのもの"


def _certify_kg_decoy(qa: QAItem) -> tuple[bool, str]:
    return True, "棄権が正解のため検索到達は問わない (常に認定)"


# 型別の認定条件。キーは kg_query_type、ただし供給系統が合格条件そのものを
# 決める kg_missing / kg_decoy は KGPathProvenance.source をキーにする
# (certify_key 参照)。**表に無いキーは不認定** (fail-closed)。
CERTIFY_BY_TYPE: dict[str, Callable[[QAItem], tuple[bool, str]]] = {
    "multi_hop": _certify_multi_hop,
    "single_fact": _certify_single_fact,
    "aggregation": _certify_aggregation,
    "traceability": _certify_traceability,
    "negation_exhaustive": _certify_negation_exhaustive,
    "kg_missing": _certify_kg_missing,
    "kg_decoy": _certify_kg_decoy,
}

# 設計書の表と型の列挙が食い違ったら (型を足して表を直し忘れたら) 気づけるように、
# 網羅性は tests/test_rag_verify.py で凍結する。実行時は fail-closed で不認定。
_UNKNOWN_TYPE_REASON = "認定表に無い型 ({key}) のため不認定 (fail-closed)"


def certify_key(qa: QAItem) -> str:
    """認定表の行を引くキー。供給系統 (kg_missing / kg_decoy) が型より優先する。"""
    kp = qa.kg_path
    if kp is None:
        return ""
    source = getattr(kp, "source", "kg_covered")
    if source in ("kg_missing", "kg_decoy"):
        return source
    return qa.kg_query_type or ""


def certify_kg_items(items: list[QAItem]) -> dict[str, int]:
    """型別表で「KG が要る」を認定し、KGPathProvenance に結果を書く。

    設計書「認定ゲート」節 (2026-08-08 追加) の実装。実測値を持つのがこの工程
    なのでここに置く。関数単体で呼べる形にしてあり、既に出荷済みの JSONL に
    後から掛け直せる (kg_path を持つ項目だけが対象)。

    - multi_hop の型確定 (終端が top-k に直接命中したら single_fact へ降格) は
      既存の sync_kg_query_type_from_retrieval をそのまま使う (挙動は不変)
    - 不認定でも項目は捨てず型も据え置く。落とすのは certified の旗だけで、
      対照群として残す
    - 返り値は件数の内訳 (total / certified / uncertified / ... と
      "certified:<キー>" 形式の型別内訳)。認定を通らなかった素材の数を
      必ず外へ出すため
    """
    counts: dict[str, int] = {}

    def bump(key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    for qa in items:
        bump("total")
        kp = qa.kg_path
        if kp is None:
            bump("skipped_no_kg_path")
            continue
        # 型の確定 (multi_hop ⇄ single_fact) は認定より前に済ませる。
        # 認定表は確定後の型で引かないと、降格すべき項目を multi_hop の
        # 行で不認定にして「対照群」と「降格」を取り違える。
        if sync_kg_query_type_from_retrieval(qa):
            bump("type_synced")
        key = certify_key(qa)
        rule = CERTIFY_BY_TYPE.get(key)
        if rule is None:
            ok, reason = False, _UNKNOWN_TYPE_REASON.format(key=key or "未設定")
        else:
            ok, reason = rule(qa)
        kp.certified = ok
        kp.certification_reason = reason
        bump("certified" if ok else "uncertified")
        bump(f"{'certified' if ok else 'uncertified'}:{key or '未設定'}")
    return counts


def format_certification_counts(counts: dict[str, int]) -> str:
    """認定ゲートの内訳を1行に整える (型別内訳はキー名の昇順)。"""
    detail = " ".join(
        f"{k}={v}" for k, v in sorted(counts.items()) if ":" in k
    )
    return (f"認定 {counts.get('certified', 0)} / "
            f"不認定 {counts.get('uncertified', 0)}"
            + (f" ({detail})" if detail else ""))


def rag_verify_batch(
    in_path: Path,
    *,
    chunks_dir: Path,
    top_k: int = DEFAULT_TOP_K,
    rag_model: str,
    judge_model: str,
    prompt_path: Path = Path(DEFAULT_PROMPT),
    llm: Optional[LLMCaller] = None,
    embed: Optional[EmbedCaller] = None,
    chunk_embeddings: Optional[np.ndarray] = None,
    out_path: Optional[Path] = None,
    update_labels: bool = True,
    hybrid: bool = True,
) -> Path:
    """Run RAG ground-truth verification on every QA in a JSONL.

    検索は既定でハイブリッド (密ベクトル + BM25 の RRF 融合)。実運用の RAG が
    ハイブリッドである以上、スクリーニングも同じ基準で行う (hybrid=False は
    旧来の密ベクトルのみとの比較用)。
    Writes results back. If `out_path` is None, overwrites `in_path`.
    """
    _, body = load_prompt(str(prompt_path))
    sections = _split_sections(body)
    rag_section = sections.get("RAG_ANSWER", "")
    judge_section = sections.get("JUDGE_MATCH", "")
    if not rag_section or not judge_section:
        raise RuntimeError(f"{prompt_path} missing RAG_ANSWER or JUDGE_MATCH section")

    caller = llm or llm_generate
    embedder = embed or llm_embed

    chunks = load_chunks(chunks_dir)
    if not chunks:
        raise RuntimeError(f"No chunks under {chunks_dir} — chunk first")
    if chunk_embeddings is None:
        emb = compute_embeddings(chunks)
        if emb is None:
            raise RuntimeError("chunk embedding failed — check VLLM_EMBEDDING_ENDPOINT")
        chunk_embeddings = emb
    if chunk_embeddings.shape[0] != len(chunks):
        raise RuntimeError(
            f"chunk_embeddings rows={chunk_embeddings.shape[0]} != chunks={len(chunks)} — "
            "古い埋め込みを使い回している恐れがあるので、もう一度埋め込みを取り直してください"
        )
    bm25 = BM25Index(chunks) if hybrid else None
    reranker = build_reranker_from_env() if hybrid else None
    mode = ("hybrid_rerank (密+BM25 → 交差エンコーダ)" if reranker is not None
            else "hybrid (密+BM25 RRF)" if hybrid else "dense のみ")
    print(f"[rag-verify] 検索方式: {mode} / top_k={top_k}")

    qas: list[QAItem] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qas.append(QAItem.model_validate_json(line))

    # 完走するまでは同じディレクトリの一時ファイルへ書く。既定の in-place
    # 実行で Ctrl-C / OOM / プロセス停止が起きても、元の全 QA を失わない。
    # 完走後だけ os.replace するため、別 out_path の既存内容も途中失敗から守る。
    out = out_path or in_path
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out.name}.", suffix=".tmp", dir=out.parent,
    )
    tmp_path = Path(tmp_name)

    n_match = n_partial = n_no_match = n_hit_doc = n_hit_chunk = 0
    n_failed = 0          # 検索が落ちて未検証のまま保留した件数
    n_judge_ng = 0        # 回答・judge が落ちて照合できなかった件数
    n_total = len(qas)
    certification: dict[str, int] = {}   # 認定ゲートの内訳 (型別)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for i, qa in enumerate(qas, 1):
                result = rag_verify_qa(
                    qa,
                    chunks=chunks,
                    chunk_embeddings=chunk_embeddings,
                    top_k=top_k,
                    rag_model=rag_model,
                    judge_model=judge_model,
                    rag_section=rag_section,
                    judge_section=judge_section,
                    llm=caller,
                    embed=embedder,
                    bm25=bm25,
                    reranker=reranker,
                )
                if result is None:
                    # 検索失敗。既存の検証結果があればそれを壊さず残す。
                    n_failed += 1
                    f.write(qa.model_dump_json() + "\n")
                    print(f"[rag-verify] [{i}/{n_total}] {qa.qa_id} → 未検証 (検索失敗)")
                    continue
                qa.rag_verification = result
                if update_labels:
                    apply_measured_retrieval_level(qa)
                    sync_kg_query_type_from_retrieval(qa)
                    # 認定ゲート: 実測を持っているこの工程で合否を切る。
                    # 不認定でも項目は書き出す (対照群として残す)。
                    for k, v in certify_kg_items([qa]).items():
                        certification[k] = certification.get(k, 0) + v
                if not result.judge_ok:
                    n_judge_ng += 1
                elif result.answer_match == "match":
                    n_match += 1
                elif result.answer_match == "partial":
                    n_partial += 1
                else:
                    n_no_match += 1
                if result.retrieval_hit_doc:
                    n_hit_doc += 1
                if result.retrieval_hit_chunk:
                    n_hit_chunk += 1
                f.write(qa.model_dump_json() + "\n")
                print(
                    f"[rag-verify] [{i}/{n_total}] {qa.qa_id} → "
                    f"{'照合不能' if not result.judge_ok else 'match=' + result.answer_match}"
                    f" hit_doc={result.retrieval_hit_doc}"
                    f" 根拠命中={result.retrieval_hit_ratio:.0%}"
                )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise

    print(
        f"[rag-verify] done: match={n_match} partial={n_partial} no_match={n_no_match} "
        f"(hit_doc={n_hit_doc}/{len(qas)} hit_chunk={n_hit_chunk}/{len(qas)}) → {out}"
    )
    if certification.get("total"):
        print(f"[rag-verify] 認定ゲート: {format_certification_counts(certification)}")
    if n_failed or n_judge_ng:
        print(f"[rag-verify] WARNING: 未検証 {n_failed} 件 / 照合不能 {n_judge_ng} 件 "
              f"— これらは no_match (KG候補) として数えないこと")
    return out
