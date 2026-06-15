"""Stage 2: Judge-based filtering + embedding-based dedup."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from .chunker import load_chunks
from .llm import LLMError, generate as llm_generate
from .prompts import load_prompt
from .schema import FilterScores, QAItem

DEFAULT_PROMPT = "prompts/judge.md"

# Threshold defaults (1-5 scales, higher is better).
DEFAULT_ANSWERABILITY_MIN = 4.0
DEFAULT_GROUNDING_MIN = 4.0
DEFAULT_UNIQUENESS_MAX = 0.92  # cosine similarity above this = duplicate
# 1.0 = 全 rationale が逐語必須。0.5 に緩めると LLM の整形差を許容できる。
# 2026-05-29 に Gemma 4 で 8% → 24% 通過率に改善した実測から 0.5 を既定化。
DEFAULT_RATIONALE_GROUNDED_MIN = 0.5

# Perspective ids used in prompts/judge.md
PERSPECTIVES = [
    "ANSWERABILITY",
    "LEAKAGE",
    "GROUNDING",
    "DIFFICULTY_MATCH",
]
# Note: RATIONALE_COMPLETENESS was removed — it duplicated GROUNDING and the
# returned score was discarded anyway (no field in FilterScores).


# ---------- prompt section extraction ----------

_SECTION_RE = re.compile(
    r"^##\s*\[(?P<name>[A-Z_]+)\]\s*\n(?P<body>.*?)(?=^##\s*\[|^---\s*$|\Z)",
    re.DOTALL | re.MULTILINE,
)


def _split_perspectives(judge_body: str) -> dict[str, str]:
    """Return {PERSPECTIVE: section_body} from the shared judge.md."""
    return {m.group("name"): m.group("body").strip() for m in _SECTION_RE.finditer(judge_body)}


def _build_judge_prompt(
    name: str,
    section: str,
    qa: QAItem,
    anchor_chunks: list[tuple[str, Optional[int], str]],
) -> str:
    if len(anchor_chunks) <= 1:
        chunk_block = anchor_chunks[0][2] if anchor_chunks else "(該当チャンクなし)"
        anchor_section = f"[元チャンク]\n{chunk_block}"
    else:
        parts = [f"[元チャンク {len(anchor_chunks)}個 — すべてQAの根拠候補]"]
        for i, (doc_id, page, text) in enumerate(anchor_chunks, 1):
            page_str = f"p.{page}" if page is not None else "p.?"
            parts.append(f"--- 元チャンク {i}/{len(anchor_chunks)} ({doc_id} {page_str}) ---\n{text}")
        anchor_section = "\n\n".join(parts)
    input_block = (
        f"[質問]\n{qa.question}\n\n"
        f"[回答]\n{qa.answer}\n\n"
        f"[rationale]\n"
        + "\n".join(f"- ({r.doc_id}:{r.page}) {r.text}" for r in qa.rationale)
        + f"\n\n{anchor_section}"
    )
    return f"## [{name}]\n\n{section}\n\n{input_block}"


_CHUNK_ID_SUFFIX_RE = re.compile(r"__c\d+$")
_WS_RE = re.compile(r"\s+")


def _normalize_doc_id(raw: str) -> str:
    """LLM sometimes echoes a chunk_id ('foo__c0154') as doc_id. Strip suffix."""
    return _CHUNK_ID_SUFFIX_RE.sub("", raw)


def _normalize_for_match(s: str) -> str:
    """Strip whitespace for substring comparison (LLM often reflows whitespace)."""
    return _WS_RE.sub("", s)


def compute_rationale_grounded(
    qa: QAItem,
    anchor_chunks: list[tuple[str, Optional[int], str]],
) -> float:
    """Return fraction of rationale entries whose .text appears verbatim (after
    whitespace normalization) in some anchor chunk. 1.0 = all grounded."""
    if not qa.rationale:
        return 0.0
    # Concatenate anchor chunk texts; LLM should quote from any of them
    anchor_blob = "".join(_normalize_for_match(text) for _, _, text in anchor_chunks)
    if not anchor_blob:
        return 0.0
    matched = 0
    for r in qa.rationale:
        needle = _normalize_for_match(r.text)
        if needle and needle in anchor_blob:
            matched += 1
    return matched / len(qa.rationale)


def _find_chunk_texts(
    qa: QAItem,
    chunk_index: dict[tuple[str, Optional[int]], list[str]],
    full_chunks: list,
) -> list[tuple[str, Optional[int], str]]:
    """Return one anchor chunk per rationale entry (deduped).

    同一 (doc_id, page) には複数チャンクがあり得る (PDF は1ページ最大6チャンク、
    .txt は page=None で文書全体が同一キー)。候補を全部連結すると判定LLMへの
    入力が文書まるごとに膨張するため、**rationale を逐語で含むチャンクを1つ選ぶ**。
    含むものが無ければ先頭候補 (逐語照合は 0 になり正しく落ちる)。

    Lookup strategy per rationale:
      1. (doc_id, page) exact match
      2. doc_id with chunk_id suffix stripped + page
      3. chunk_id direct match (if rationale.doc_id is actually a chunk_id)
      4. any chunk in the same normalized doc_id
    """
    # Build chunk_id → (doc_id, page, text) for case 3
    by_chunk_id: dict[str, tuple[str, Optional[int], str]] = {
        c.chunk_id: (c.doc_id, c.page, c.text) for c in full_chunks
    }
    # Build doc_id → list of texts for case 4
    by_doc: dict[str, list[str]] = {}
    for c in full_chunks:
        by_doc.setdefault(c.doc_id, []).append(c.text)

    def _candidates(r) -> Optional[tuple[str, Optional[int], list[str]]]:
        norm_doc = _normalize_doc_id(r.doc_id)
        for doc, page in ((r.doc_id, r.page), (norm_doc, r.page)):
            texts = chunk_index.get((doc, page))
            if texts:
                return doc, page, texts
        if r.doc_id in by_chunk_id:
            doc, page, text = by_chunk_id[r.doc_id]
            return doc, page, [text]
        if norm_doc in by_doc:
            return norm_doc, r.page, by_doc[norm_doc]
        return None

    out: list[tuple[str, Optional[int], str]] = []
    seen: set[tuple[str, Optional[int], int]] = set()

    for r in qa.rationale:
        found = _candidates(r)
        if not found:
            continue
        doc_id, page, texts = found
        needle = _normalize_for_match(r.text)
        best = next(
            (t for t in texts if needle and needle in _normalize_for_match(t)),
            texts[0],
        )
        key = (doc_id, page, hash(best))
        if key not in seen:
            out.append((doc_id, page, best))
            seen.add(key)

    if out:
        return out
    # last-resort fallback
    if qa.rationale:
        target_doc = _normalize_doc_id(qa.rationale[0].doc_id)
        for (doc_id, page), texts in chunk_index.items():
            if doc_id == target_doc and texts:
                return [(doc_id, page, texts[0])]
    return [("(unknown)", None, "(該当チャンクなし)")]


# ---------- LLM-based scoring ----------

LLMCaller = Callable[..., Any]


_LEVEL_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}


def _difficulty_match(declared: str, measured: str) -> Optional[str]:
    """宣言済み answer_level と判定LLMの実態レベルのズレをタグ化。"""
    if declared not in _LEVEL_ORDER or measured not in _LEVEL_ORDER:
        return None
    d, m = _LEVEL_ORDER[declared], _LEVEL_ORDER[measured]
    if m == d:
        return "aligned"
    return "too_easy" if m < d else "too_hard"


def _parse_json_safely(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    from .llm import _parse_json

    parsed = _parse_json(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise LLMError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed


def score_qa(
    qa: QAItem,
    anchor_chunks: list[tuple[str, Optional[int], str]],
    *,
    model: str,
    perspectives: dict[str, str],
    llm: LLMCaller,
) -> FilterScores:
    """Run each judge perspective and collate into FilterScores."""
    scores = FilterScores()
    for name in PERSPECTIVES:
        section = perspectives.get(name)
        if not section:
            continue
        prompt = _build_judge_prompt(name, section, qa, anchor_chunks)
        try:
            raw = llm(prompt=prompt, model=model, response_model=None, temperature=0.0)
            data = _parse_json_safely(raw)
        except (LLMError, json.JSONDecodeError) as e:
            print(f"[filter] judge {name} failed: {e}")
            continue

        if name == "ANSWERABILITY" and "answerability" in data:
            scores.answerability = float(data["answerability"])
        elif name == "LEAKAGE" and data.get("leakage") in ("pass", "fail"):
            scores.leakage = data["leakage"]
        elif name == "GROUNDING" and "grounding" in data:
            scores.grounding = float(data["grounding"])
        elif name == "DIFFICULTY_MATCH" and data.get("answer_level") in (
            "Easy", "Medium", "Hard"
        ):
            measured = data["answer_level"]
            # 宣言値 (生成時の暫定) と実態のズレをタグ化してから、実態で確定する。
            scores.difficulty_match = _difficulty_match(qa.answer_level, measured)
            qa.answer_level = measured
    return scores


# ---------- duplicate detection ----------

def _compute_uniqueness(qas: list[QAItem]) -> list[float]:
    """Return max cosine similarity against any other QA in the batch (0..1).

    Uses the vLLM-hosted embedding endpoint (VLLM_EMBEDDING_ENDPOINT +
    VLLM_EMBEDDING_MODEL).
    """
    try:
        import numpy as np  # type: ignore
    except ImportError as e:
        print(f"[filter] uniqueness skipped (numpy missing): {e}")
        return [0.0] * len(qas)

    from .llm import embed

    texts = [qa.question for qa in qas]
    try:
        vectors = embed(texts)
    except Exception as e:
        print(f"[filter] uniqueness skipped (embedding endpoint failed): {e}")
        return [0.0] * len(qas)

    emb = np.array(vectors, dtype=float)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    sim = emb @ emb.T
    np.fill_diagonal(sim, 0.0)
    return sim.max(axis=1).tolist()


# ---------- public entry ----------

def filter_batch(
    raw_path: Path,
    out_dir: Path,
    chunks_dir: Path,
    *,
    judge_model: str,
    prompt_path: Path = Path(DEFAULT_PROMPT),
    llm: Optional[LLMCaller] = None,
    answerability_min: float = DEFAULT_ANSWERABILITY_MIN,
    grounding_min: float = DEFAULT_GROUNDING_MIN,
    uniqueness_max: float = DEFAULT_UNIQUENESS_MAX,
    rationale_grounded_min: float = DEFAULT_RATIONALE_GROUNDED_MIN,
    compute_uniqueness: bool = True,
    require_leakage_pass: bool = True,
    reject_too_easy: bool = False,
    require_rag_fail: bool = False,
    require_rag_hit: bool = False,
    require_rationale_retrieved: bool = False,
) -> Path:
    """Score each QA with judge LLM + dedup, drop failures, write filtered JSONL."""
    # `--require-rag-fail` (answer_match != 'no_match' を落とす) と
    # `--require-rag-hit` (answer_match != 'match' を落とす) は相反する条件で、
    # 同時指定すると rag_verification 付きの QA は必ずどちらかに引っかかり
    # 全件 drop される。CLI 経由でない呼び出しでも気付けるよう、ここで早期に弾く。
    if require_rag_fail and require_rag_hit:
        raise ValueError(
            "require_rag_fail と require_rag_hit は同時指定できません "
            "(no_match と match の両方を満たす QA は存在せず、全件 drop されます)"
        )
    _, body = load_prompt(str(prompt_path))
    perspectives = _split_perspectives(body)

    # Load and index chunks for per-QA lookup.
    # 同一 (doc_id, page) の全チャンクを候補リストとして持ち、rationale を含む
    # チャンクを _find_chunk_texts が選ぶ。連結で1本にすると .txt 文書
    # (page=None) で文書全体に膨張し、判定LLMの入力が流量制限を突破する。
    chunks = load_chunks(chunks_dir) if chunks_dir.exists() else []
    chunk_index: dict[tuple[str, Optional[int]], list[str]] = {}
    for c in chunks:
        chunk_index.setdefault((c.doc_id, c.page), []).append(c.text)

    # Load QAs
    qas: list[QAItem] = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qas.append(QAItem.model_validate_json(line))

    caller = llm or llm_generate

    # Score each QA (LLM perspectives + deterministic rationale grounding)
    for qa in qas:
        anchor_chunks = _find_chunk_texts(qa, chunk_index, chunks)
        qa.filter_scores = score_qa(
            qa, anchor_chunks, model=judge_model, perspectives=perspectives, llm=caller
        )
        # Deterministic check: does rationale.text actually appear in the chunks?
        qa.filter_scores.rationale_grounded = compute_rationale_grounded(qa, anchor_chunks)

    # Compute uniqueness (1 - max similarity against other items)
    if compute_uniqueness and len(qas) > 1:
        sims = _compute_uniqueness(qas)
        for qa, s in zip(qas, sims):
            qa.filter_scores.uniqueness = 1.0 - float(s)

    # `--require-rag-*` 系は rag-verify 未実行時に全件 drop しないよう、
    # 1件でも rag_verification 持ちが居なければ警告を出して無視する。
    rag_required = require_rag_fail or require_rag_hit or require_rationale_retrieved
    rag_present = any(qa.rag_verification is not None for qa in qas)
    missing = sum(1 for qa in qas if qa.rag_verification is None)
    if rag_required and not rag_present:
        print(
            "[filter] WARNING: --require-rag-* 指定だが rag_verification が無い"
            " — rag-verify 未実行のため当該条件はスキップ"
        )
        require_rag_fail = False
        require_rag_hit = False
        require_rationale_retrieved = False
    elif rag_required and rag_present and missing > 0:
        # 一部の QA だけ rag-verify 済みで、残りは未付与というまだら状態。
        # 後段のガード (rv is not None) が掛かるため、未付与の QA は
        # --require-rag-* 条件を素通りして残ってしまう。利用者が
        # それに気付けるよう、件数を明示して通知する。
        print(
            f"[filter] WARNING: {missing}/{len(qas)} 件に rag_verification が無く、"
            "--require-rag-* 条件はそれらでは判定されません。"
            "rag-verify を流し直すか、対象を絞り込んでください"
        )

    # Apply thresholds
    kept: list[QAItem] = []
    for qa in qas:
        fs = qa.filter_scores
        rv = qa.rag_verification
        reasons: list[str] = []
        if fs.answerability is not None and fs.answerability < answerability_min:
            reasons.append(f"answerability={fs.answerability}")
        if require_leakage_pass and fs.leakage == "fail":
            reasons.append("leakage=fail")
        if fs.grounding is not None and fs.grounding < grounding_min:
            reasons.append(f"grounding={fs.grounding}")
        if fs.uniqueness is not None and fs.uniqueness < (1.0 - uniqueness_max):
            reasons.append(f"uniqueness={fs.uniqueness:.3f}")
        if fs.rationale_grounded is not None and fs.rationale_grounded < rationale_grounded_min:
            reasons.append(f"rationale_grounded={fs.rationale_grounded:.2f}")
        if reject_too_easy and fs.difficulty_match == "too_easy":
            reasons.append("difficulty_match=too_easy")
        if require_rag_fail and rv is not None and rv.answer_match != "no_match":
            reasons.append(f"rag_answer_match={rv.answer_match} (require no_match)")
        if require_rag_hit and rv is not None and rv.answer_match != "match":
            reasons.append(f"rag_answer_match={rv.answer_match} (require match)")
        if require_rationale_retrieved and rv is not None:
            # 旧 JSONL は retrieval_hit_chunk が未設定 (既定 False) で書かれているので、
            # 当該フィールドが両方とも False かつ retrieval_hit (旧フィールド) が真のときは
            # 旧フォーマットの「文書一致」として扱い、ふるい分けを通す。
            hit_chunk = rv.retrieval_hit_chunk
            legacy_only = not rv.retrieval_hit_chunk and not rv.retrieval_hit_doc
            if not hit_chunk and not (legacy_only and rv.retrieval_hit):
                reasons.append("rag_retrieval_hit_chunk=False")
        if reasons:
            print(f"[filter] drop {qa.qa_id}: {', '.join(reasons)} (scores={fs.model_dump()})")
            continue
        print(f"[filter] keep {qa.qa_id}: scores={fs.model_dump()}")
        kept.append(qa)

    # Write filtered
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / raw_path.name
    with out_path.open("w", encoding="utf-8") as f:
        for qa in kept:
            f.write(qa.model_dump_json() + "\n")

    print(f"[filter] kept {len(kept)}/{len(qas)} QAs → {out_path}")
    return out_path
