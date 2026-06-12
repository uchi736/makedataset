"""Stage 1: Generate QA items from anchor chunk groups using an LLM (R1 plan)."""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import ValidationError

from .aspects import (
    ALL_ASPECTS,
    ASPECT_BAD_PATTERNS,
    ASPECT_DESCRIPTIONS,
    ASPECT_EXAMPLES,
    ASPECT_GOOD_EXAMPLES,
    ASPECT_TO_CATEGORY,
)
from .chunker import load_chunks
from .llm import LLMError, generate as llm_generate
from .prompts import load_prompt, render
from .sampling import (
    AnchorSampler,
    compute_embeddings,
    compute_retrieval_difficulty,
    score_retrieval_level,
)
from .schema import Chunk, GenerationInfo, QAItem
from .tracks.kg_poc import (
    ALL_KG_NOVELTY,
    ALL_KG_QUERY_TYPES,
    KG_NOVELTY_DESCRIPTIONS,
    KG_QUERY_TYPE_DESCRIPTIONS,
)

DEFAULT_PROMPT = "prompts/generate.md"
DEFAULT_PROMPT_KG = "prompts/generate_kg_poc.md"
DEFAULT_SEEDS = "data/seeds/seeds.json"

# 難易度は乱数では割り当てない。retrieval_level はチャンク組成から決定論的に
# 算出 (sampling.score_retrieval_level)、answer_level は判定LLMが実態から確定する。

# Aspects that need multi-chunk context. Used to decide whether to compute
# embeddings for the batch (avoids the call if no one needs them).
MULTI_CHUNK_ASPECTS = {
    "multi_source_integration",
    "multi_doc_reference",
    "remote_reference",
    "standards_reference",
    "multi_hop",
}


LLMCaller = Callable[..., Any]


def _sample_seeds(seeds_path: Path, rng: random.Random, k: int = 3) -> list[dict]:
    if not seeds_path.exists():
        return []
    data = json.loads(seeds_path.read_text(encoding="utf-8"))
    return rng.sample(data, k=min(k, len(data)))


def _stable_qa_id(question: str, anchors: list[Chunk], aspect: str) -> str:
    key = "+".join(c.chunk_id for c in anchors) + f"::{aspect}::{question}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"auto_{h}"


def _build_spec(rng: random.Random) -> dict[str, Any]:
    aspect = rng.choice(list(ALL_ASPECTS))
    category = ASPECT_TO_CATEGORY[aspect]
    return {
        "aspect": aspect,
        "category": category,
    }


def _build_spec_kg_from_cell(rng: random.Random, qt: str, nov: str) -> dict[str, Any]:
    """Build spec for a given (query_type, novelty) cell already chosen by the
    distribution scheduler. 難易度はここでは決めない (生成後に確定する)。"""
    return {
        "kg_query_type": qt,
        "kg_query_type_desc": KG_QUERY_TYPE_DESCRIPTIONS[qt],
        "kg_novelty": nov,
        "kg_novelty_desc": KG_NOVELTY_DESCRIPTIONS[nov],
    }


def parse_kg_mix(spec: str) -> list[tuple[str, str, int]]:
    """Parse `--mix` string like
        "multi_hop:unknown_relation=5, traceability:procedural_relation=3"
    Returns [(query_type, novelty, count), ...]. Raises on invalid format/values.
    """
    out: list[tuple[str, str, int]] = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry or ":" not in entry:
            raise ValueError(
                f"--mix entry '{entry}' must be 'query_type:novelty=N' format"
            )
        cell_part, count_part = entry.split("=", 1)
        qt, nov = (s.strip() for s in cell_part.split(":", 1))
        if qt not in ALL_KG_QUERY_TYPES:
            raise ValueError(f"unknown kg_query_type '{qt}'. allowed: {ALL_KG_QUERY_TYPES}")
        if nov not in ALL_KG_NOVELTY:
            raise ValueError(f"unknown kg_novelty '{nov}'. allowed: {ALL_KG_NOVELTY}")
        try:
            n = int(count_part.strip())
        except ValueError:
            raise ValueError(f"count '{count_part}' is not an integer (entry: {entry})")
        if n <= 0:
            raise ValueError(f"count must be > 0 (entry: {entry})")
        out.append((qt, nov, n))
    if not out:
        raise ValueError("--mix is empty")
    return out


def _build_kg_cell_queue(
    n: int,
    mix: Optional[list[tuple[str, str, int]]],
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Return a list of (query_type, novelty) cells of length n.

    - If mix is given: expand each (qt, nov, count) into `count` copies, shuffle,
      and truncate/pad to n.
    - Otherwise: equal distribution across all 15 cells (round-robin shuffled).
    """
    queue: list[tuple[str, str]] = []
    if mix:
        for qt, nov, count in mix:
            queue.extend([(qt, nov)] * count)
        rng.shuffle(queue)
        if len(queue) >= n:
            return queue[:n]
        # If mix total < n, pad with random cells from the mix
        deficit = n - len(queue)
        pool = [(qt, nov) for qt, nov, _ in mix]
        queue.extend(rng.choices(pool, k=deficit))
        rng.shuffle(queue)
        return queue
    # equal distribution across all 15 cells
    all_cells = [(qt, nov) for qt in ALL_KG_QUERY_TYPES for nov in ALL_KG_NOVELTY]
    base, rem = divmod(n, len(all_cells))
    for cell in all_cells:
        queue.extend([cell] * base)
    if rem > 0:
        queue.extend(rng.sample(all_cells, k=rem))
    rng.shuffle(queue)
    return queue


def _format_examples_block(items: list[str], prefix: str = "- ") -> str:
    """Render a list of strings as a bullet block for prompt injection."""
    if not items:
        return "(該当なし)"
    return "\n".join(f"{prefix}{x}" for x in items)


def _format_anchor_chunks_block(anchors: list[Chunk]) -> str:
    n = len(anchors)
    lines = [f"# 元チャンク (n={n})", ""]
    for i, c in enumerate(anchors, 1):
        section = " > ".join(c.section_path) if c.section_path else "(no section)"
        page_str = f"p.{c.page}" if c.page is not None else "p.?"
        lines.append(f"## チャンク{i}: {c.doc_id} ({page_str}, {c.chunk_id}) [section: {section}]")
        if c.references:
            lines.append(f"_参照: {', '.join(c.references)}_")
        lines.append("```")
        lines.append(c.text)
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _render_generate_prompt(
    prompt_body: str, anchors: list[Chunk], spec: dict[str, Any], few_shot: list[dict]
) -> str:
    aspect = spec["aspect"]
    primary = anchors[0]
    variables = {
        "anchor_chunks_block": _format_anchor_chunks_block(anchors),
        # legacy single-chunk vars still referenced by some prompt parts
        "doc_id": primary.doc_id,
        "page": primary.page if primary.page is not None else "null",
        "page_or_null": primary.page if primary.page is not None else "null",
        "chunk_id": primary.chunk_id,
        "few_shot": json.dumps(few_shot, ensure_ascii=False, indent=2),
        "aspect": aspect,
        "aspect_description": ASPECT_DESCRIPTIONS.get(aspect, ""),
        "aspect_example": ASPECT_EXAMPLES.get(aspect, ""),
        "aspect_good_examples_block": _format_examples_block(
            ASPECT_GOOD_EXAMPLES.get(aspect, [])
        ),
        "aspect_bad_patterns_block": _format_examples_block(
            ASPECT_BAD_PATTERNS.get(aspect, [])
        ),
        "category": spec["category"],
        "aspect_json": json.dumps([aspect], ensure_ascii=False),
        "category_json": json.dumps([spec["category"]], ensure_ascii=False),
    }
    return render(prompt_body, variables)


def _render_generate_prompt_kg(
    prompt_body: str, anchors: list[Chunk], spec: dict[str, Any], few_shot: list[dict]
) -> str:
    """KG-PoC variant: substitute kg_query_type / kg_novelty into prompt."""
    variables = {
        "anchor_chunks_block": _format_anchor_chunks_block(anchors),
        "few_shot": json.dumps(few_shot, ensure_ascii=False, indent=2),
        "kg_query_type": spec["kg_query_type"],
        "kg_query_type_desc": spec["kg_query_type_desc"],
        "kg_novelty": spec["kg_novelty"],
        "kg_novelty_desc": spec["kg_novelty_desc"],
    }
    return render(prompt_body, variables)


def _kg_aspect_for_strategy(spec: dict[str, Any]) -> str:
    """Map KG spec → existing AnchorSampler aspect key so we can reuse strategies.

    Heuristic: 関係系/手順系の novelty では multi-anchor 戦略を選ぶ.
    (= 異なる doc 強制で、関係を辿る材料を出す).
    single_fact は SingleChunk で十分。
    """
    qt, nov = spec["kg_query_type"], spec["kg_novelty"]
    if qt == "single_fact":
        return "quantitative_calc"  # → SingleChunk
    # 関係を問う組は、埋め込み類似ペア (似ているだけで関係が無い) ではなく
    # 実在参照 (第N条→本文/別表) で繋がったペアを使う。類似ペアで作った
    # multi_hop/aggregation は判定で answerability/grounding=3 で大量死した。
    if qt in ("multi_hop", "traceability"):
        return "reference_follow"
    if nov in ("unknown_relation", "procedural_relation"):
        return "reference_follow"
    if qt == "aggregation":
        return "multi_source_integration"  # 列挙系は類似散在で可 (unknown_term のみ到達)
    return "reference_follow"


def generate_batch(
    chunks_dir: Path,
    out_dir: Path,
    *,
    n: int,
    model: str,
    track: str = "general",
    prompt_path: Optional[Path] = None,
    seeds_path: Path = Path(DEFAULT_SEEDS),
    seed: int = 42,
    llm: Optional[LLMCaller] = None,
    kg_mix: Optional[list[tuple[str, str, int]]] = None,
) -> Path:
    """Generate a batch of QAs. `track` chooses between general / kg_poc.

    For track='kg_poc':
      - If kg_mix is provided, generate exactly those (query_type, novelty)
        cells with the specified counts.
      - Otherwise, equal distribution across all 15 cells.
    """
    chunks = load_chunks(chunks_dir)
    if not chunks:
        raise RuntimeError(f"No chunks found under {chunks_dir}")

    # Resolve track-specific prompt + spec/render functions
    if track == "kg_poc":
        default_prompt = Path(DEFAULT_PROMPT_KG)
        render_fn = _render_generate_prompt_kg
        sampler_key_fn = _kg_aspect_for_strategy
    else:
        default_prompt = Path(DEFAULT_PROMPT)
        render_fn = _render_generate_prompt
        sampler_key_fn = lambda s: s["aspect"]
    prompt_path = prompt_path or default_prompt

    rng = random.Random(seed)
    meta, body = load_prompt(str(prompt_path))
    prompt_version = meta.get("prompt_version", "unknown")

    embeddings = compute_embeddings(chunks)
    sampler = AnchorSampler(chunks, embeddings)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"batch_{stamp}.jsonl"

    caller = llm or llm_generate

    # Build per-iteration spec source.
    # general: random aspect per iteration (existing behavior).
    # kg_poc: pre-built cell queue (explicit mix or equal distribution).
    cell_queue: list[tuple[str, str]] = []
    if track == "kg_poc":
        cell_queue = _build_kg_cell_queue(n, kg_mix, rng)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            if track == "kg_poc":
                qt, nov = cell_queue[i]
                spec = _build_spec_kg_from_cell(rng, qt, nov)
            else:
                spec = _build_spec(rng)
            sampler_key = sampler_key_fn(spec)
            anchors = sampler.select(sampler_key, rng)
            if anchors is None:
                # コーパスがこの観点の組成 (複数チャンク/複数文書) を作れない → skip
                print(f"[generate] skip ({sampler_key}): コーパスが必要なチャンク組成を作れず")
                continue
            few_shot = _sample_seeds(seeds_path, rng, k=3)
            prompt = render_fn(body, anchors, spec, few_shot)
            try:
                # max_tokens は llm.generate 既定 (8192) に任せる。
                # force_json=True で JSON mode 有効化 → 文字列途中切断やfence外
                # コメント等の "もっともらしいが壊れたJSON" を抑制
                raw = caller(prompt=prompt, model=model, response_model=None, force_json=True)
                data = _coerce_to_dict(raw)
                # Stable qa_id: track-specific key
                spec_key = spec.get("aspect") or spec.get("kg_query_type", "?")
                data["qa_id"] = _stable_qa_id(data.get("question", ""), anchors, spec_key)
                data["generation"] = GenerationInfo(
                    model=model,
                    prompt_version=prompt_version,
                    generated_at=datetime.now(),
                ).model_dump(mode="json")
                # Force-override what we asked the LLM to produce. The LLM
                # frequently downgrades (e.g., asked for multi_hop, returns
                # single_fact), which silently re-skews the distribution.
                if track == "kg_poc":
                    data["kg_query_type"] = spec["kg_query_type"]
                    data["kg_novelty"] = spec["kg_novelty"]
                qa = QAItem.model_validate(data)
                # 検索難易度は LLM の自己申告を使わず、チャンク組成から決定論的に確定。
                qa.retrieval_level = score_retrieval_level(anchors)
                qa.retrieval_difficulty = compute_retrieval_difficulty(anchors)
                # answer_level は LLM の暫定値のまま。filter の判定LLMが実態で上書き確定する。
            except (LLMError, ValidationError, json.JSONDecodeError) as e:
                # For Pydantic ValidationError the first line is generic
                # ("N validation errors for QAItem"); the next 1-2 lines have
                # the actual field + reason — keep those for diagnosability.
                lines = str(e).splitlines()
                detail = " | ".join(line.strip() for line in lines[1:4] if line.strip())[:300]
                print(f"[generate] skip ({type(e).__name__}): {lines[0]} :: {detail}")
                continue

            # Post-gen sanity check: katakana neologisms in question/answer
            # that don't appear in any anchor chunk → LLM fabricated jargon.
            fabricated = detect_fabricated_terms(
                qa.question + " " + qa.answer, anchors
            )
            if fabricated:
                print(f"[generate] skip {qa.qa_id} (fabricated terms): {fabricated}")
                continue

            f.write(qa.model_dump_json() + "\n")
            written += 1
            if len(anchors) > 1:
                doc_ids = "+".join(sorted({c.doc_id for c in anchors}))
                tag = spec.get("aspect") or f"{spec.get('kg_query_type')}×{spec.get('kg_novelty')}"
                print(f"[generate] {qa.qa_id} {tag} anchors={len(anchors)} docs={doc_ids}")

    print(f"[generate] wrote {written}/{n} QAs to {out_path} (track={track})")
    if sampler.compat_misses:
        print("[generate] aspect compatibility misses (fell back to full pool):")
        for asp, cnt in sorted(sampler.compat_misses.items(), key=lambda kv: -kv[1]):
            print(f"  {asp}: {cnt} (no chunk matched compatibility predicate)")
    if sampler.difficulty_misses:
        print("[generate] difficulty misses (コーパスが組成を作れず skip):")
        for asp, cnt in sorted(sampler.difficulty_misses.items(), key=lambda kv: -kv[1]):
            print(f"  {asp}: {cnt} (multi-chunk 組成を作れなかった)")
    return out_path


# 4+ chars of katakana / extended phrases — LLM 造語の典型形式
_KATAKANA_PHRASE_RE = re.compile(r"[ァ-ヴー・]{4,}")


def detect_fabricated_terms(qa_text: str, anchor_chunks: list[Chunk]) -> list[str]:
    """Return katakana neologisms in qa_text that don't appear in any anchor.

    These are typically LLM-fabricated jargon (e.g. ディスミッション・エクスクルージョン).
    Common loanwords (4+ chars katakana) get caught, so this is conservative.
    """
    blob = "".join(c.text for c in anchor_chunks)
    found: list[str] = []
    for m in _KATAKANA_PHRASE_RE.finditer(qa_text):
        phrase = m.group(0)
        if phrase not in blob and phrase not in found:
            found.append(phrase)
    return found


def _coerce_to_dict(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        from .llm import _parse_json

        parsed = _parse_json(raw)
        if not isinstance(parsed, dict):
            raise LLMError(f"Expected JSON object, got {type(parsed).__name__}")
        return parsed
    raise LLMError(f"Unsupported LLM output type: {type(raw).__name__}")
