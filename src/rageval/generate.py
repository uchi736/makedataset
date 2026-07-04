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
            # 異 doc 跨ぎが本質である戦略 (multi_doc_reference / multi_source_integration /
            # multi_hop) で sampler が単一 doc に縮退した場合は早期 skip。
            # こうしないと「multi-doc の問いを作れ」と LLM に頼んでも、anchor が1 doc
            # しか無いので物理的に成立せず、偽 traceability が量産される。
            _MULTI_DOC_REQUIRED = {
                "multi_doc_reference", "multi_source_integration", "multi_hop",
                "reference_follow",  # kg_poc の unknown_relation/procedural_relation 系
            }
            if sampler_key in _MULTI_DOC_REQUIRED:
                anchor_docs = {c.doc_id for c in anchors}
                if len(anchor_docs) < 2:
                    print(
                        f"[generate] skip ({sampler_key}): anchor が単一 doc に縮退 "
                        f"(docs={sorted(anchor_docs)}) — 異 doc 跨ぎが作れず"
                    )
                    continue
            few_shot = _sample_seeds(seeds_path, rng, k=3)
            prompt = render_fn(body, anchors, spec, few_shot)
            try:
                # max_tokens は 4096 に明示。
                # Gemma 4 26B-A4B (ctx=16384) に対し prompt 入力が ~8000 token 近くに
                # 達することがあり、既定 8192 だと合計 16384 を超えて 400 で落ちる事例があった。
                # QA 1問の JSON 出力は実測で 1500-2500 token なので 4096 で十分マージンあり。
                # force_json=True で JSON mode 有効化 → 文字列途中切断や fence 外
                # コメント等の "もっともらしいが壊れたJSON" を抑制
                raw = caller(
                    prompt=prompt, model=model, response_model=None,
                    force_json=True, max_tokens=4096,
                )
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

            # Post-gen sanity check: 多 doc アンカーを渡したのに rationale が 1 doc しか
            # 持たないと、traceability/multi_hop の本質である異 doc 横断が成立しない。
            # sampling 側は 2 doc 渡すよう指定済みなのに Gemma が 1 doc に縮める事例があった
            # (v1.7 smoke では traceability 18件中 全件が rationale 単一 doc に縮退)。
            mismatch = detect_single_doc_rationale_with_multi_anchor(qa, anchors)
            if mismatch:
                anchor_docs, rat_docs = mismatch
                print(
                    f"[generate] skip {qa.qa_id} (rationale 単一doc): "
                    f"anchors={sorted(anchor_docs)} rationale={sorted(rat_docs)}"
                )
                continue

            # Post-gen sanity check: 場所当て (条文の所在だけ尋ねる) は KG-RAG 評価に
            # 値しない。「どの条文」「第何条」等の場所探索表現を検出して落とす。
            place_findings = detect_place_finding_question(qa.question)
            if place_findings:
                print(
                    f"[generate] skip {qa.qa_id} (場所当て): "
                    f"phrases={place_findings} :: {qa.question[:80]}"
                )
                continue

            # Post-gen sanity check: 質問文に条文番号が 3 個以上列挙されているのは
            # 「実務者が聞かない」型 (答えのヒントを質問側に書き込んでいる)。
            article_count = detect_article_enumeration(qa.question)
            if article_count >= 3:
                print(
                    f"[generate] skip {qa.qa_id} (条文番号列挙 {article_count}個): "
                    f"{qa.question[:90]}"
                )
                continue

            # Post-gen sanity check: 「Xの経験者がYに従事する場合」のような
            # 経歴ピボット (無関係チャンク2題を架空経歴で強引に接続) を検出。
            pivot = detect_career_pivot(qa.question)
            if pivot:
                print(
                    f"[generate] skip {qa.qa_id} (経歴ピボット): "
                    f"phrases={pivot} :: {qa.question[:90]}"
                )
                continue

            # Post-gen sanity check: 「第N条に関連する事項を確認したい。第N条に
            # 関連する様式は」のような既知前提引用 (場所当ての変種) を検出。
            premise = detect_known_premise_quote(qa.question)
            if premise:
                print(
                    f"[generate] skip {qa.qa_id} (既知前提引用): "
                    f"phrases={premise} :: {qa.question[:90]}"
                )
                continue

            # Post-gen sanity check: rationale.text が条文番号断片だけのものを
            # 含む場合、多 doc 跨ぎを偽装した実質単 doc 問題なので skip。
            fragments = detect_fragment_rationale(qa)
            if fragments:
                snippets = " | ".join(f"#{i}:{t[:20]}" for i, t in fragments)
                print(
                    f"[generate] skip {qa.qa_id} (断片rationale): {snippets}"
                )
                continue

            # v2.0 追加: rationale.text の文字数が下限 30 未満は水増し疑い。
            short = detect_short_rationale(qa, min_chars=30)
            if short:
                snippets = " | ".join(f"#{i}({n}字):{t[:20]}" for i, n, t in short)
                print(
                    f"[generate] skip {qa.qa_id} (rationale 字数不足): {snippets}"
                )
                continue

            # v2.0 追加: answer 中の固有値 (様式番号/条文/数値+単位 等) が
            # rationale.text に 1 件も逐語存在しないなら、捏造 or 裏取り不足。
            ungrounded = detect_ungrounded_answer_tokens(qa)
            if ungrounded:
                print(
                    f"[generate] skip {qa.qa_id} (固有値 裏取り無し): "
                    f"tokens={ungrounded}"
                )
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


# v2.0 追加: answer 中の固有値を抽出する正規表現群。
# answer に登場する「答えそのもの」になる文字列で、これらは必ずいずれかの
# rationale.text に逐語で出てこなければならない (= 固有値の逐語裏取り)。
_ANSWER_TOKEN_PATTERNS = [
    r"様式第[０-９0-9一二三四五六七八九十百千]+号?",
    r"別表第[０-９0-9一二三四五六七八九十百千]+",
    r"附属書[A-ZＡ-Ｚ]",
    r"JIS\s*[A-Z]\s*\d{3,5}(?:[-－]\d+)?",
    r"ISO\s*\d{3,5}",
    r"ASME\s*[A-Z]+(?:\.\d+)?",
    r"\d+(?:\.\d+)?\s*(?:MPa|kPa|mm|cm|m|kg|t|℃|倍|時間|分|秒|日|年|kW|MW|A|V|Hz)",
    r"第[０-９0-9一二三四五六七八九十百千]+条(?:第[０-９0-9]+項)?(?:第[０-９0-9]+号)?",
]
_ANSWER_TOKEN_RE = re.compile("|".join(_ANSWER_TOKEN_PATTERNS))
_WS_RE = re.compile(r"\s+")


def _normalize_for_match(s: str) -> str:
    """空白を除去し全角数字を半角に揃える (逐語照合の前処理)。"""
    half = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    return _WS_RE.sub("", half)


def detect_ungrounded_answer_tokens(qa: QAItem) -> list[str]:
    """answer 中の固有値で、どの rationale.text にも逐語存在しないものを返す。

    answer に「様式第6号」「20MPa」「第59条」のような具体値が出てくるのに、
    その文字列が rationale.text のどれにも逐語で含まれないなら、その固有値は
    LLM が捏造したか rationale の引用が不十分。skip 対象。
    """
    answer_norm_tokens = list(dict.fromkeys(_ANSWER_TOKEN_RE.findall(qa.answer)))
    if not answer_norm_tokens:
        return []
    rationale_blob = _normalize_for_match("".join(r.text for r in qa.rationale))
    missing: list[str] = []
    for tok in answer_norm_tokens:
        if _normalize_for_match(tok) not in rationale_blob:
            missing.append(tok)
    return missing


def detect_short_rationale(qa: QAItem, min_chars: int = 30) -> list[tuple[int, int, str]]:
    """rationale.text の文字数が下限未満のエントリを (index, len, text) で返す。

    9字や 13字の条文番号断片で rationale 配列を水増しする型を機械的に弾く。
    detect_fragment_rationale が正規表現で「番号だけ」を検出するのに対し、
    こちらは内容問わず長さだけで切る。両方併用が安全。
    """
    bad: list[tuple[int, int, str]] = []
    for i, r in enumerate(qa.rationale):
        n = len(r.text.strip())
        if n < min_chars:
            bad.append((i, n, r.text))
    return bad


def detect_single_doc_rationale_with_multi_anchor(
    qa: QAItem, anchors: list[Chunk],
) -> Optional[tuple[set[str], set[str]]]:
    """anchor が 2 doc 以上なのに rationale が 1 doc しか持たない場合、(anchor_docs, rationale_docs) を返す。
    違反でなければ None。

    traceability / multi_hop / multi_doc_reference 等の「異 doc 跨ぎ」を意図して
    sampling が 2 doc 渡したのに、生成 LLM が「1 doc で書ける」と判断して片方を
    捨てているケースを検出する。
    """
    anchor_doc_ids = {c.doc_id for c in anchors}
    if len(anchor_doc_ids) < 2:
        return None  # 元から単一 doc なら検査対象外
    rationale_doc_ids = {r.doc_id for r in qa.rationale}
    if len(rationale_doc_ids) >= 2:
        return None  # rationale も 2 doc 以上なら OK
    return anchor_doc_ids, rationale_doc_ids


# v1.9 追加: 断片 rationale 検出。
# 「第N条」「第N条第M項」のような条文番号断片だけを rationale.text に置き、
# 実体は別 doc に閉じる cross-doc 偽装を弾く。
_FRAGMENT_RATIONALE_RE = re.compile(
    r"^\s*(?:第\d+(?:条|項|号)(?:[（(][^）)]{0,8}[）)])?[\s、,]*)+\s*$"
)


def detect_fragment_rationale(qa: QAItem) -> list[tuple[int, str]]:
    """rationale.text が条文番号や箇条見出し断片だけの実体不足エントリを返す。

    多 doc 跨ぎを偽装するために「第１０７条」「第１項」「（ロ）第１０７条第１項の試験」
    といった番号断片を rationale に貼り付ける型を機械的に検出する。
    返り値は (index, text) のリスト。空なら違反なし。
    """
    bad: list[tuple[int, str]] = []
    for i, r in enumerate(qa.rationale):
        text = r.text.strip()
        if not text:
            bad.append((i, text))
            continue
        # 全角数字を半角に正規化してマッチさせる
        norm = text.translate(str.maketrans("０１２３456789", "0123456789"))
        # 数字の半角化を完全にする
        norm = norm.translate(
            str.maketrans({"０":"0","１":"1","２":"2","３":"3","４":"4","５":"5","６":"6","７":"7","８":"8","９":"9"})
        )
        # 番号断片のみ
        if _FRAGMENT_RATIONALE_RE.match(norm):
            bad.append((i, text))
            continue
        # 極端に短い rationale (10文字未満) も実体不足とみなす
        if len(text) < 10:
            bad.append((i, text))
    return bad


# 場所当て (条文・章節の所在を問うだけ) の検出パターン。
# 「どこに」+ 「定め/規定/書か」 や、「どの条文」「第何条」を主に拾う。
# 「どの規定」「どの試験」「どの基準」は内容を問う文脈で正当に使われるので除外。
_PLACE_FINDING_PATTERNS = [
    r"どの条文",
    r"どの条",
    r"どこに定め",
    r"どこに規定",
    r"どこに書か",
    r"第何条",
    r"第何章",
    r"第何節",
    r"どの章に書",
    r"どの章で",
    r"何条で規定",
    r"何条に定め",
]
_PLACE_FINDING_RE = re.compile("|".join(_PLACE_FINDING_PATTERNS))


_ARTICLE_NUMBER_RE = re.compile(r"第\d+条")


def detect_article_enumeration(question: str) -> int:
    """質問文に含まれる条文番号 (「第N条」) の異なり数を返す。

    実務者は答え合わせ用に条文番号を覚えていないので、質問文に複数の
    条文番号が列挙されているのは「答えのヒントを質問側に書き込んだ」
    試験文体の徴候。3 個以上で skip 対象。
    """
    matches = _ARTICLE_NUMBER_RE.findall(question)
    return len(set(matches))


# 経歴ピボット (無関係な経歴で 2 題を強引に接続) の検出。
# 「Xの経験/従事者が Yに関わる/従事する/設計する場合」のような構文を捉える。
_CAREER_PIVOT_PATTERNS = [
    r"経験[ががのを]\S{0,8}従事",
    r"経験[ががのを]\S{0,8}関わる",
    r"経験[ががのを]\S{0,8}行う",
    r"経験[ががのを]\S{0,8}設計",
    r"従事[ ]?した?経験",
    r"携わ\S{0,4}者が",
    r"従事[ ]?する者が\S{0,8}関わ",
    r"設計\S{0,4}従事[ ]?した?経験",
]
_CAREER_PIVOT_RE = re.compile("|".join(_CAREER_PIVOT_PATTERNS))


# 既知前提引用: 質問者が「第N条が答えに関係する」と既に知っているように書く形。
# 「第N条に関連する事項を確認したい。第N条に関連する様式は」のような
# 同一条文の重複言及や、答えに使う先を質問側で指定する型を捉える。
_KNOWN_PREMISE_PATTERNS = [
    r"第\d+条に関連する事項",
    r"第\d+条に関連する[^。]{0,12}様式",
    r"第\d+条に関連する[^。]{0,12}記録",
    r"第\d+条に基づき[^。]{0,20}様式",
    r"第\d+条に基づく[^。]{0,12}様式",
    r"第\d+条に紐づ[^。]{0,12}様式",
    r"第\d+条に関する事項を確認",
    r"第\d+条の規定に基づき[^。]{0,12}様式は",
    # v1.9 追加 (audit で見つかった隠れ変種)
    r"第\d+条第\d+項の規定に準じて",       # 「第59条第3項の規定に準じて補強」
    r"第\d+条の規定に準じて",                # 「第14条の規定に準じて補強」
    r"第\d+条に基づいて",                    # 「第6条第5項の規定に基づいて」
    r"第\d+条第\d+項の規定に基づいて",
    r"第\d+条になった際",                    # 「第N条になった際」
    r"[（(]第\d+条[）)]",                    # 質問文中の括弧書き条文番号 (第77条) 系
    r"[（(]第\d+条第\d+項[）)]",
    r"第\d+条に関連して使用",                # 「第77条に関連して使用すべき様式」
]
_KNOWN_PREMISE_RE = re.compile("|".join(_KNOWN_PREMISE_PATTERNS))


def detect_known_premise_quote(question: str) -> list[str]:
    """質問文に「第N条が答えに関係する」と質問者が既に知っている前提のパターンを検出。

    実務者は条文番号を覚えていない前提なので、「第N条に関連する〜を確認したい」型は
    場所当ての変種 (= 既知前提引用) として弾く。
    """
    return list(dict.fromkeys(_KNOWN_PREMISE_RE.findall(question)))


def detect_career_pivot(question: str) -> list[str]:
    """質問文に「Xの経験者がYに関わる場合」型のピボット構文があれば返す。

    sampling が無関係ドメインの 2 doc をペアにしたとき、Gemma が両 doc
    を rationale に引くために「Xの経験者がYに従事する場合」のような
    架空経歴で強引に接続する不良型を量産する。これを後段で機械的に弾く。
    """
    return list(dict.fromkeys(_CAREER_PIVOT_RE.findall(question)))


def detect_place_finding_question(question: str) -> list[str]:
    """質問文に「条文の所在だけを問う」表現が含まれるなら、その表現を返す。

    「どの条文の規定に適合すればよいか」「第何条か」のような場所当ては、
    KG-RAG の評価では内容理解を測れないので落としたい。
    一方「どの試験」「どの基準」「どの規定 (内容)」のような中身を問う表現は
    場所当てではないので、ここでは弾かない。
    """
    return list(dict.fromkeys(_PLACE_FINDING_RE.findall(question)))


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
