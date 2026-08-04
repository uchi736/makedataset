"""Stage 2: Judge-based filtering + embedding-based dedup."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from .chunker import load_chunks
from .evidence import EvidenceIndex, grounded_fraction, match_rationales
from .llm import LLMError, generate as llm_generate
from .prompts import load_prompt
from .sampling import CrossDocPolicy, load_cross_doc_policy
from .schema import Chunk, FilterScores, QAItem

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


def compute_rationale_grounded(qa: QAItem, chunks: list[Chunk]) -> float:
    """引用のうち、**自分が名乗る文書の中で**逐語一致した割合。1.0 = 全件一致。

    2026-08-01: 以前は全アンカーチャンクを連結した1本の文字列に対して部分一致を
    取っていたため、(a) doc A の引用が doc B の本文にあるだけで一致になる、
    (b) 連結の境目をまたいだ偶然一致が起きる、という2つの誤判定があった
    (どちらも実データで再現)。照合は evidence.match_rationales に一本化する。
    """
    if not qa.rationale:
        return 0.0
    return grounded_fraction(match_rationales(qa.rationale, chunks))


def _find_chunk_texts(
    qa: QAItem,
    chunks: list[Chunk],
    *,
    index: Optional[EvidenceIndex] = None,
) -> list[tuple[str, Optional[int], str]]:
    """判定LLM に見せるアンカーチャンクを引用ごとに1つずつ選ぶ (重複除去)。

    照合は evidence.match_rationales に委譲する。**引用が名乗る文書の中でしか
    探さない**ので、別文書のチャンクが根拠として提示されることはない。
    引用を含むチャンクが無ければ同じ文書の先頭チャンク (逐語照合は 0 になり
    正しく落ちる)。文書自体が見つからなければプレースホルダを返す。
    """
    matches = match_rationales(qa.rationale, chunks, index=index)
    out: list[tuple[str, Optional[int], str]] = []
    seen: set[str] = set()
    for m in matches:
        if m.chunk is None:
            continue
        if m.chunk.chunk_id not in seen:
            out.append((m.chunk.doc_id, m.chunk.page, m.chunk.text))
            seen.add(m.chunk.chunk_id)
    return out or [("(unknown)", None, "(該当チャンクなし)")]


# ---------- 誤結合の機械検出 (決定論) ----------
# plant_v15 で「別法令の第63条」「別法令の第7条」を同一視した捏造 QA が採用まで
# 素通りした実害への対策。引用が複数文書に跨り、質問/回答中の文書ローカルな番号
# (第N条・様式第N・別表第N) が両方の文書に別文脈で存在し、かつその文書対が
# 文書スコープ規則 (エッジ台帳・規格番号・明示参照) で正当化されない場合に検出する。

import unicodedata

_LOCAL_NUM_TOKEN_RE = re.compile(
    r"第[0-9]+条|様式第[0-9]+号?|別表第[0-9]+"
)


def _norm_ja(s: str) -> str:
    """全角→半角 (NFKC) + 空白除去。番号トークンの表記揺れを吸収する。"""
    return _WS_RE.sub("", unicodedata.normalize("NFKC", s))


def _resolve_rationale_chunks(qa: QAItem, full_chunks: list[Chunk]) -> list[Chunk]:
    """rationale ごとに、引用を逐語で含むチャンクを1つ選ぶ (無ければ同 doc の先頭)。"""
    from .evidence import matched_chunks
    return matched_chunks(match_rationales(qa.rationale, full_chunks))


def detect_cross_doc_number_collision(
    qa: QAItem,
    full_chunks: list[Chunk],
    policy: CrossDocPolicy,
) -> Optional[str]:
    """別文書の同番号条文を同一視した疑いを検出する。検出時は理由文字列を返す。"""
    cited = _resolve_rationale_chunks(qa, full_chunks)
    docs = {c.doc_id for c in cited}
    if len(docs) < 2:
        return None
    qa_text = _norm_ja(qa.question + " " + qa.answer)
    tokens = set(_LOCAL_NUM_TOKEN_RE.findall(qa_text))
    if not tokens:
        return None
    texts = {c.chunk_id: _norm_ja(c.text) for c in cited}
    for i in range(len(cited)):
        for j in range(i + 1, len(cited)):
            a, b = cited[i], cited[j]
            if a.doc_id == b.doc_id:
                continue
            if policy.join_ok(a, b):
                continue  # 台帳対・規格番号・明示参照で正当化された対
            shared = [t for t in tokens if t in texts[a.chunk_id] and t in texts[b.chunk_id]]
            if shared:
                return (f"{a.doc_id}×{b.doc_id} が正当化されない対のまま "
                        f"番号 {sorted(shared)} を共有")
    return None


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
) -> tuple[FilterScores, list[str]]:
    """判定観点ごとに judge を回して FilterScores にまとめる。

    Returns (scores, failed) — failed は判定できなかった観点名の一覧。
    「判定に失敗した」ことと「判定した結果スコアが低い」ことは意味が違うので、
    呼び出し側が区別できるように失敗を明示的に返す (以前は失敗すると
    スコアが None のまま残り、閾値判定を素通りして通過していた)。
    """
    scores = FilterScores()
    failed: list[str] = []
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
            failed.append(name)
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
        else:
            # 呼び出しは成功したが期待した項目が無い/値が不正 → 判定不成立
            print(f"[filter] judge {name} returned unusable payload: {str(data)[:120]}")
            failed.append(name)
    return scores, failed


# ---------- duplicate detection ----------

def _compute_uniqueness(qas: list[QAItem]) -> Optional[list[float]]:
    """バッチ内の他 QA との最大コサイン類似度 (0..1) を返す。

    埋め込みが使えなかった場合は **None** を返す (「全件ユニーク」ではない)。
    以前は失敗時に 0.0 を返しており、uniqueness=1.0 として重複判定が全件
    素通りしていた。呼び出し側は None を「未実施」として扱うこと。
    """
    try:
        import numpy as np  # type: ignore
    except ImportError as e:
        print(f"[filter] uniqueness 未実施 (numpy missing): {e}")
        return None

    from .llm import embed

    texts = [qa.question for qa in qas]
    try:
        vectors = embed(texts)
    except Exception as e:
        print(f"[filter] uniqueness 未実施 (embedding endpoint failed): {e}")
        return None

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
    check_cross_doc_collision: bool = True,
    anchor_pairs_path: Optional[Path] = Path("data/seeds/anchor_pairs_manual_v1.jsonl"),
    doc_aliases_path: Optional[Path] = Path("data/seeds/doc_aliases.json"),
    out_path: Optional[Path] = None,
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

    # Load and index chunks for per-QA lookup (evidence.EvidenceIndex に一本化)。
    chunks = load_chunks(chunks_dir) if chunks_dir.exists() else []
    ev_index = EvidenceIndex.build(chunks)

    # Load QAs
    qas: list[QAItem] = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qas.append(QAItem.model_validate_json(line))

    caller = llm or llm_generate

    # v3: 誤結合検出用の文書スコープ規則 (generate と同じ台帳を読む)
    policy = load_cross_doc_policy(anchor_pairs_path, doc_aliases_path) \
        if check_cross_doc_collision else CrossDocPolicy()

    # Score each QA (LLM perspectives + deterministic rationale grounding)
    collisions: dict[str, str] = {}
    judge_failures: dict[str, list[str]] = {}
    for qa in qas:
        anchor_chunks = _find_chunk_texts(qa, chunks, index=ev_index)
        qa.filter_scores, failed = score_qa(
            qa, anchor_chunks, model=judge_model, perspectives=perspectives, llm=caller
        )
        if failed:
            judge_failures[qa.qa_id] = failed
        # Deterministic check: does rationale.text actually appear in the chunks?
        qa.filter_scores.rationale_grounded = compute_rationale_grounded(qa, chunks)
        # Deterministic check: 別文書の同番号条文の誤結合
        if check_cross_doc_collision and chunks:
            hit = detect_cross_doc_number_collision(qa, chunks, policy)
            if hit:
                collisions[qa.qa_id] = hit

    # Compute uniqueness (1 - max similarity against other items)
    uniqueness_done = True
    if compute_uniqueness and len(qas) > 1:
        sims = _compute_uniqueness(qas)
        if sims is None:
            # 埋め込みが使えなかった。全件ユニーク扱いにはしない (未実施のまま)。
            uniqueness_done = False
        else:
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
        # decoy (棄権が正解) は answerability / leakage の観点が構造的に噛み合わない:
        # 「この根拠から答えられるか」→ 答えられない、が正解なので低スコアが正しく、
        # 落とすと系統ごと消える (2026-08-03 実測)。uniqueness と rationale_grounded
        # (引用の逐語性) だけで見る。
        is_decoy = (qa.kg_path is not None
                    and getattr(qa.kg_path, "source", "") == "kg_decoy")
        # 判定できなかったものは「合格」ではない。必須スコアが欠けていたら落とす。
        # (以前は None が閾値比較を素通りし、judge 障害時に未判定の QA が
        #  そのまま通過していた)
        missing_scores = [
            name for name, val in (
                ("answerability", "skip" if is_decoy else fs.answerability),
                ("grounding", "skip" if is_decoy else fs.grounding),
                ("leakage", fs.leakage if (require_leakage_pass and not is_decoy) else "skip"),
            ) if val is None
        ]
        if missing_scores:
            reasons.append(f"unevaluated: {'/'.join(missing_scores)} が未判定 "
                           f"(judge 失敗: {judge_failures.get(qa.qa_id, [])})")
        if (not is_decoy and fs.answerability is not None
                and fs.answerability < answerability_min):
            reasons.append(f"answerability={fs.answerability}")
        if require_leakage_pass and not is_decoy and fs.leakage == "fail":
            reasons.append("leakage=fail")
        if (not is_decoy and fs.grounding is not None
                and fs.grounding < grounding_min):
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
        if qa.qa_id in collisions:
            reasons.append(f"cross_doc_number_collision: {collisions[qa.qa_id]}")
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
    out_path = out_path or out_dir / raw_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for qa in kept:
            f.write(qa.model_dump_json() + "\n")

    print(f"[filter] kept {len(kept)}/{len(qas)} QAs → {out_path}")
    if judge_failures:
        print(f"[filter] WARNING: judge が {len(judge_failures)} 件で失敗し、"
              f"該当 QA は未判定として drop しました "
              f"(内訳: {sorted({p for v in judge_failures.values() for p in v})})")
    if not uniqueness_done and compute_uniqueness and len(qas) > 1:
        print("[filter] WARNING: uniqueness 未実施のまま通過しています "
              "(重複が残っている可能性あり — 埋め込みサーバを確認してください)")
    return out_path
