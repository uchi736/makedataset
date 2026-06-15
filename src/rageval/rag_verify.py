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
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .chunker import load_chunks
from .llm import LLMError, _parse_json, embed as llm_embed, generate as llm_generate
from .prompts import load_prompt
from .sampling import compute_embeddings
from .schema import Chunk, QAItem, RAGAnswerMatch, RAGVerification

DEFAULT_PROMPT = "prompts/rag_verify.md"
DEFAULT_TOP_K = 5

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


def _retrieve_topk(
    question: str,
    chunks: list[Chunk],
    chunk_embeddings: np.ndarray,
    top_k: int,
    *,
    embed: EmbedCaller,
) -> list[Chunk]:
    """Embed the question and return the top-k chunks by cosine similarity.

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
    # 同点時は元の順序を保つため stable
    top_idx = np.argsort(-sims, kind='stable')[:k]
    return [chunks[int(i)] for i in top_idx]


_CHUNK_ID_SUFFIX_RE = re.compile(r"__c\d+$")
_WS_RE = re.compile(r"\s+")


def _normalize_doc_id(raw: str) -> str:
    """根拠の doc_id にチャンクID('foo__c0023')が紛れたときに末尾を落とす。"""
    return _CHUNK_ID_SUFFIX_RE.sub("", raw)


def _normalize_for_match(s: str) -> str:
    """空白を全て削って比較用に整える (filter._normalize_for_match と同規則)。"""
    return _WS_RE.sub("", s)


def _retrieval_hits(qa: QAItem, retrieved: list[Chunk]) -> tuple[bool, bool]:
    """根拠が上位 k にどう当たったかを (文書一致, チャンク一致) で返す。

    - 文書一致: 根拠と同じ doc_id のチャンクが1個でも上位 k に居れば真。
      技報のように1文書が数十チャンクに割れるコーパスでは、根拠とは
      無関係な章のチャンクが拾われただけで真になるので、検索健全性の
      判定指標として単独で使うと過大評価になる。
    - チャンク一致: 根拠本文 (rationale.text) を逐語で含むチャンクが
      上位 k に居れば真。filter 側の逐語照合と同じ空白正規化規則を使う。
      チャンク粒度の検索健全性を見る本命指標。

    根拠が空のときは両方とも偽。
    根拠の doc_id がチャンクID形式 (foo__c0023) のときは、その chunk_id
    自体が上位に居れば文書一致を真にする (既存挙動の維持)。
    """
    if not qa.rationale:
        return False, False

    rationale_doc_ids = {_normalize_doc_id(r.doc_id) for r in qa.rationale}
    rationale_chunk_id_hints = {r.doc_id for r in qa.rationale}

    retrieved_doc_ids = {c.doc_id for c in retrieved}
    retrieved_chunk_ids = {c.chunk_id for c in retrieved}

    hit_doc = bool(rationale_doc_ids & retrieved_doc_ids)
    if not hit_doc and (rationale_chunk_id_hints & retrieved_chunk_ids):
        hit_doc = True

    # チャンク一致は逐語で見る。空白差を吸収するため整形してから部分一致。
    retrieved_blob = "".join(_normalize_for_match(c.text) for c in retrieved)
    hit_chunk = False
    if retrieved_blob:
        for r in qa.rationale:
            needle = _normalize_for_match(r.text)
            if needle and needle in retrieved_blob:
                hit_chunk = True
                break

    return hit_doc, hit_chunk


def _retrieval_hit(qa: QAItem, retrieved: list[Chunk]) -> bool:
    """旧 API。文書一致 OR チャンク一致のどちらかが真なら真。

    既存テスト・既存呼び出し点との互換のために残す。新しいコードは
    `_retrieval_hits` を使い、文書一致とチャンク一致を分けて持つこと。
    """
    hit_doc, hit_chunk = _retrieval_hits(qa, retrieved)
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
) -> RAGVerification:
    """Run vector RAG once and judge the answer. Returns the RAGVerification.

    Conservative on errors: any LLM failure → answer_match='no_match', empty rag_answer.
    """
    # Step 1: retrieve
    try:
        retrieved = _retrieve_topk(qa.question, chunks, chunk_embeddings, top_k, embed=embed)
    except Exception as e:
        print(f"[rag-verify] {qa.qa_id} retrieve failed: {e}")
        retrieved = []

    retrieved_ids = [c.chunk_id for c in retrieved]
    hit_doc, hit_chunk = _retrieval_hits(qa, retrieved)

    # Step 2: RAG answer
    rag_answer = ""
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
                max_tokens=1024,
                force_json=True,
            )
            rag_answer = _parse_json_safely(raw).get("answer", "").strip()
        except Exception as e:
            # APIConnectionError 等で全件落ちないように広めに捕捉
            print(f"[rag-verify] {qa.qa_id} rag answer failed: {type(e).__name__}: {e}")

    # Step 3: judge
    # match は judge が想定どおりに返したときだけ書き換える。
    # judge を呼ばなかった/落ちた/想定外を返したときはすべて 'no_match' のまま残るが、
    # 後段でこの3者を区別できるよう、生の verdict を judge_raw に保存しておく。
    match: RAGAnswerMatch = "no_match"
    judge_raw: Optional[str] = None
    if not _is_unable_answer(rag_answer):
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
                max_tokens=512,
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
        except Exception as e:
            print(f"[rag-verify] {qa.qa_id} judge failed: {type(e).__name__}: {e}")

    return RAGVerification(
        top_k=top_k,
        retrieved_chunk_ids=retrieved_ids,
        retrieval_hit_doc=hit_doc,
        retrieval_hit_chunk=hit_chunk,
        # 旧フィールドは「どちらかが真」を入れて後方互換を保つ
        retrieval_hit=hit_doc or hit_chunk,
        rag_answer=rag_answer,
        answer_match=match,
        judge_raw=judge_raw,
        rag_model=rag_model,
        judge_model=judge_model,
        verified_at=datetime.now(),
    )


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
) -> Path:
    """Run vector RAG ground-truth verification on every QA in a JSONL.

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

    qas: list[QAItem] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qas.append(QAItem.model_validate_json(line))

    # 1問ずつ append 書き出し。Ctrl-C / Azure 全断 / OOM で途中で死んでも、
    # それまで verify した結果は out_path に残るようにする。
    out = out_path or in_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.open("w", encoding="utf-8").close()  # 既存内容を空にする

    n_match = n_partial = n_no_match = n_hit_doc = n_hit_chunk = 0
    n_total = len(qas)
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
        )
        qa.rag_verification = result
        if result.answer_match == "match":
            n_match += 1
        elif result.answer_match == "partial":
            n_partial += 1
        else:
            n_no_match += 1
        if result.retrieval_hit_doc:
            n_hit_doc += 1
        if result.retrieval_hit_chunk:
            n_hit_chunk += 1
        # 1問ずつ追記書き出し (途中で死んでも既処理分は失われない)
        with out.open("a", encoding="utf-8") as f:
            f.write(qa.model_dump_json() + "\n")
        print(
            f"[rag-verify] [{i}/{n_total}] {qa.qa_id} → match={result.answer_match}"
            f" hit_doc={result.retrieval_hit_doc} hit_chunk={result.retrieval_hit_chunk}"
        )

    print(
        f"[rag-verify] done: match={n_match} partial={n_partial} no_match={n_no_match} "
        f"(hit_doc={n_hit_doc}/{len(qas)} hit_chunk={n_hit_chunk}/{len(qas)}) → {out}"
    )
    return out
