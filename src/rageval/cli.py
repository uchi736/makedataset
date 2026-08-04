"""Typer-based CLI entry point.

Usage:
    rageval chunk    --docs data/docs/ --out data/chunks/
    rageval generate --chunks data/chunks/ --n 10 --out data/raw/
    rageval filter   --in data/raw/batch_xxx.jsonl --out data/filtered/ --chunks data/chunks/
    rageval review   --in data/filtered/batch_xxx.jsonl
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

from typing import Optional

from .chunker import (
    DISCOVERED_PATTERNS_PATH,
    chunk_directory,
    discover_patterns,
    load_chunks,
)
from .filter import filter_batch
from .generate import generate_batch
from .probe import probe_batch
from .rag_verify import DEFAULT_TOP_K, rag_verify_batch

load_dotenv()

# Windows コンソール (cp932) で「—」等の記号を含むログが UnicodeEncodeError で
# 工程全体を落とさないよう、符号化できない文字は置換して出力する。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _halt_on_unverified(path: Path, stage: str, allow_partial: bool, is_unverified) -> None:
    """工程の切れ目で未判定が残っていないか確かめ、残っていれば止める。

    障害のまま次工程へ進むと「未判定」が集計から静かに漏れ、全件検証済みの
    ように見えてしまう。--allow-partial なら警告だけで続行する。
    """
    from .schema import QAItem
    bad = [
        qa.qa_id for qa in (
            QAItem.model_validate_json(l)
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()
        ) if is_unverified(qa)
    ]
    if not bad:
        return
    msg = (f"[pipeline] {stage}: {len(bad)} 件が未判定のままです "
           f"(例: {bad[:3]})。判定サーバの状態を確認して当該工程を流し直してください")
    if allow_partial:
        typer.echo(msg + " — --allow-partial のため続行します")
        return
    typer.echo(msg, err=True)
    raise typer.Exit(1)


def _judge_default(fallback: str) -> str:
    """判定モデルの既定値。JUDGE_MODEL (オンプレ判定サーバ) があれば最優先、
    無ければ従来どおり Azure 配備名 → fallback の順で解決する。"""
    return os.getenv("JUDGE_MODEL") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", fallback)


def _provenance_default(fallback: str) -> str:
    """KG辺検証は生成モデルとの誤り相関を避けるため専用設定を優先する。

    2026-08-02: 解決順を Azure より JUDGE_MODEL (オンプレ) 優先に修正。
    旧順序は Azure 配備名が .env にあるだけで段0/段1 の判定が全部クラウドへ
    流れて課金されていた (オンプレ完結・費用最小の方針違反)。Azure は
    オンプレ判定サーバが無いときの最後の逃げ道としてだけ残す。
    """
    return (
        os.getenv("KG_PROVENANCE_MODEL")
        or os.getenv("JUDGE_MODEL")
        or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        or fallback
    )


@app.command()
def chunk(
    docs: Path = typer.Option(Path("data/docs"), help="Source docs directory"),
    out: Path = typer.Option(Path("data/chunks"), help="Output chunks directory"),
    chunk_size: int = typer.Option(800),
    chunk_overlap: int = typer.Option(100),
    pdf_backend: str = typer.Option(
        "auto", "--pdf-backend",
        help="PDF preprocessing backend: 'auto' (Azure DI if env set, else pypdf), 'pypdf', or 'azure_di'",
    ),
) -> None:
    """Split .txt/.md/.pdf docs into chunks (JSONL per doc)."""
    stats = chunk_directory(
        docs, out,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        pdf_backend=pdf_backend,
    )
    for doc_id, count in stats.items():
        typer.echo(f"  {doc_id}: {count} chunks")
    typer.echo(f"Total: {sum(stats.values())} chunks across {len(stats)} docs")


@app.command("discover-patterns")
def discover_patterns_cmd(
    chunks: Path = typer.Option(Path("data/chunks"), help="Chunks directory"),
    model: str = typer.Option(
        lambda: os.getenv("VLLM_MODEL", "openai/gpt-oss-120b"),
        help="LLM model to use for discovery",
    ),
    max_chars: int = typer.Option(12000, help="Max sample chars sent to LLM"),
    rechunk: bool = typer.Option(
        False, "--rechunk/--no-rechunk",
        help="After discovery, re-chunk docs so references is repopulated with new patterns",
    ),
    docs: Path = typer.Option(Path("data/docs"), help="(only used if --rechunk)"),
) -> None:
    """Discover corpus-specific reference patterns via LLM. One-shot.

    Saves the result to data/chunks/_discovered_patterns.json so that subsequent
    chunk operations pick them up.
    """
    loaded = load_chunks(chunks)
    if not loaded:
        typer.echo(f"No chunks found under {chunks}. Run `rageval chunk` first.")
        raise typer.Exit(1)

    typer.echo(f"Discovering patterns from {len(loaded)} chunks using model={model} ...")
    patterns = discover_patterns(loaded, model=model, max_chars=max_chars)
    typer.echo(f"Accepted {len(patterns)} new patterns → {DISCOVERED_PATTERNS_PATH}:")
    for p in patterns:
        typer.echo(f"  [{p['kind']}] {p['regex']}  e.g. {p['example']!r}  - {p['rationale']}")

    if rechunk:
        typer.echo("Re-chunking docs so references reflect newly discovered patterns ...")
        stats = chunk_directory(docs, chunks)
        for doc_id, count in stats.items():
            typer.echo(f"  {doc_id}: {count} chunks")


@app.command()
def generate(
    chunks: Path = typer.Option(Path("data/chunks"), help="Chunks directory"),
    out: Path = typer.Option(Path("data/raw"), help="Output dir for raw batches"),
    n: int = typer.Option(10, help="Number of QAs to generate"),
    model: str = typer.Option(
        lambda: os.getenv("VLLM_MODEL", "openai/gpt-oss-120b"),
        help="Generator model (vLLM / Azure / Claude)",
    ),
    seeds: Path = typer.Option(Path("data/seeds/seeds.json")),
    anchor_pairs: Path = typer.Option(
        Path("data/seeds/anchor_pairs_manual_v1.jsonl"), "--anchor-pairs",
        help="解決済みエッジ台帳 (異文書アンカー対の許可リスト)。無ければ規格番号・明示参照のみで成立",
    ),
    doc_aliases: Path = typer.Option(
        Path("data/seeds/doc_aliases.json"), "--doc-aliases",
        help="文書名の別名台帳 {doc_id: [名称,…]}。明示参照 (文書名の名指し) の判定に使う",
    ),
    prompt: Optional[Path] = typer.Option(
        None,
        help="Prompt file. Defaults: prompts/generate.md (general) or prompts/generate_kg_poc.md (kg_poc)",
    ),
    track: str = typer.Option(
        "general",
        help="Evaluation track: 'general' (25観点) or 'kg_poc' (KG-PoC 3軸)",
    ),
    mix: Optional[str] = typer.Option(
        None, "--mix",
        help=(
            "KG-PoC distribution. Format: 'qt:nov=N,qt:nov=N,...'. "
            "例: 'multi_hop:unknown_relation=5,traceability:procedural_relation=3'. "
            "未指定なら全 15 セルに等分配分。"
        ),
    ),
    seed: int = typer.Option(42),
) -> None:
    """Generate N QA items from anchor chunks."""
    from .generate import parse_kg_mix
    parsed_mix = parse_kg_mix(mix) if mix else None
    generate_batch(
        chunks_dir=chunks,
        out_dir=out,
        n=n,
        model=model,
        track=track,
        prompt_path=prompt,
        seeds_path=seeds,
        seed=seed,
        kg_mix=parsed_mix,
        anchor_pairs_path=anchor_pairs,
        doc_aliases_path=doc_aliases,
    )


@app.command("kg-generate")
def kg_generate(
    chunks: Path = typer.Option(Path("data/chunks_plant"), help="原文チャンク (KG と同じコーパス)"),
    out: Path = typer.Option(..., help="出力先ディレクトリ"),
    n: int = typer.Option(20, help="生成する QA 数の上限"),
    model: str = typer.Option(
        lambda: os.getenv("VLLM_MODEL", "google/gemma-4-26B-A4B-it"), help="生成モデル"),
    provenance_model: str = typer.Option(
        lambda: _provenance_default("gpt-4.1-mini"),
        "--provenance-model",
        help="KG の各辺が原文に含意されるかを、生成とは独立に判定するモデル",
    ),
    min_extraction_count: int = typer.Option(
        2, help="各辺が何チャンクから抽出されていれば採用するか (下げると供給↑・品質↓)"),
    max_pagerank: float = typer.Option(
        0.0005, help="中間ノードの pagerank 上限 (汎用ハブの除外)"),
    max_per_intermediate: int = typer.Option(
        2, help="同じ中間ノードを使う経路の上限 (中程度のハブによる重複を散らす)"),
    path_limit: int = typer.Option(500, help="KG から取り出す経路の候補数"),
    query_type: str = typer.Option(
        "multi_hop", "--query-type",
        help="生成する型: multi_hop (2辺連鎖) / aggregation (星型の一覧・集約) / "
             "traceability (連鎖の根拠系譜) / negation (EXCLUDES 辺の適用除外)"),
    kg_mix: Optional[str] = typer.Option(
        None, "--mix",
        help="型の配分をまとめて生成する。例: 'multi_hop=10,aggregation=8,"
             "traceability=5,negation=3'。指定時は --query-type/--n を無視"),
) -> None:
    """KG の経路・星・除外辺を原文検証し、通過した素材から QA を生成する。

    素材列挙 → 段0 両端含意 → 段1 合成判定 → 段2 作問 → 段3 機械検品 の順。
    """
    from .generate_kg import (
        WRITE_TRACE_PROMPT,
        generate_from_exclusions,
        generate_from_paths,
        generate_from_stars,
    )
    from . import kg as kg_mod

    if provenance_model == model:
        typer.echo(
            "[kg-generate] WARNING: 生成モデルと provenance 判定モデルが同一です。"
            "誤り相関を避けるには --provenance-model に別モデルを指定してください",
            err=True,
        )

    _ALLOWED = ("multi_hop", "aggregation", "traceability", "negation", "missing", "decoy")
    plan: list[tuple[str, int]] = []
    if kg_mix:
        for entry in kg_mix.split(","):
            entry = entry.strip()
            if not entry:
                continue
            qt, _, count = entry.partition("=")
            qt = qt.strip()
            if qt not in _ALLOWED or not count.strip().isdigit():
                typer.echo(f"[kg-generate] --mix の項目が不正: {entry!r} "
                           f"(型={_ALLOWED}, 'type=N' 形式)", err=True)
                raise typer.Exit(1)
            plan.append((qt, int(count)))
    else:
        if query_type not in _ALLOWED:
            typer.echo(f"[kg-generate] 未対応の --query-type: {query_type}", err=True)
            raise typer.Exit(1)
        plan = [(query_type, n)]

    # 始点閉包チェックのため原文を先に読む (経路列挙の段階で必要)
    texts = {c.chunk_id: c.text for c in load_chunks(chunks)}

    for qt, n_target in plan:
        typer.echo(f"[kg-generate] === {qt} ×{n_target} ===")
        with kg_mod.session() as sess:
            snapshot = kg_mod.snapshot_id(sess)
            typer.echo(f"[kg-generate] KG 版: {snapshot}")
            if qt == "aggregation":
                materials = list(kg_mod.iter_star_groups(
                    sess, chunk_texts=texts, limit=path_limit))
                typer.echo(f"[kg-generate] 候補の星: {len(materials)} 個")
            elif qt == "negation":
                materials = list(kg_mod.iter_excludes_edges(
                    sess, chunk_texts=texts, limit=path_limit))
                typer.echo(f"[kg-generate] 候補の除外辺: {len(materials)} 本")
            elif qt == "decoy":
                from .generate_kg import find_decoy_pairs
                from .sampling import compute_embeddings, load_cross_doc_policy
                policy = load_cross_doc_policy(
                    Path("data/seeds/anchor_pairs_manual_v1.jsonl"),
                    Path("data/seeds/doc_aliases.json"))
                node_map = kg_mod.chunk_node_map(sess)
                all_chunks = load_chunks(chunks)
                emb = compute_embeddings(all_chunks)
                if emb is None:
                    typer.echo("[kg-generate] 埋め込みが取れません", err=True)
                    raise typer.Exit(1)
                materials = find_decoy_pairs(all_chunks, emb, policy, node_map,
                                             top_n=max(n_target * 4, 20))
                typer.echo(f"[kg-generate] デコイ候補 (類似だが無関係): {len(materials)} 対")
            elif qt == "missing":
                from .generate_kg import find_missing_reference_pairs
                from .sampling import load_cross_doc_policy
                policy = load_cross_doc_policy(
                    Path("data/seeds/anchor_pairs_manual_v1.jsonl"),
                    Path("data/seeds/doc_aliases.json"))
                node_map = kg_mod.chunk_node_map(sess)
                all_chunks = load_chunks(chunks)
                materials = find_missing_reference_pairs(all_chunks, policy, node_map)
                typer.echo(f"[kg-generate] KG に対応の無い明示参照: {len(materials)} 対")
            else:   # multi_hop / traceability は同じ2辺素材
                materials = list(kg_mod.iter_two_hop_paths(
                    sess,
                    min_extraction_count=min_extraction_count,
                    max_pagerank=max_pagerank,
                    max_per_intermediate=max_per_intermediate,
                    chunk_texts=texts,
                    limit=path_limit,
                ))
                typer.echo(f"[kg-generate] 候補経路: {len(materials)} 本")
        common = dict(chunks_dir=chunks, out_dir=out, model=model,
                      provenance_model=provenance_model,
                      kg_snapshot=snapshot, max_items=n_target)
        if qt == "aggregation":
            generate_from_stars(materials, **common)
        elif qt == "negation":
            generate_from_exclusions(materials, **common)
        elif qt == "missing":
            from .generate_kg import generate_from_references
            generate_from_references(materials, **common, policy=policy)
        elif qt == "decoy":
            from .generate_kg import generate_from_decoys
            generate_from_decoys(materials, **common)
        elif qt == "traceability":
            generate_from_paths(materials, **common,
                                prompt_path=Path(WRITE_TRACE_PROMPT),
                                query_type_override="traceability")
        else:
            generate_from_paths(materials, **common)


@app.command()
def pipeline(
    chunks: Path = typer.Option(..., help="チャンクディレクトリ (生成元かつ照合先)"),
    out: Path = typer.Option(..., help="系統ディレクトリ (原石と全部入りの2ファイルをここに置く)"),
    n: int = typer.Option(25, help="生成する QA 数"),
    track: str = typer.Option("kg_poc", help="'general' or 'kg_poc'"),
    model: str = typer.Option(
        lambda: os.getenv("VLLM_MODEL", "google/gemma-4-26B-A4B-it"),
        help="生成・probe・RAG 回答モデル",
    ),
    judge_model: str = typer.Option(
        lambda: _judge_default("gpt-4.1-mini"),
        help="判定モデル (filter / answer-grounding / probe照合 / rag-verify照合)",
    ),
    in_path: Optional[Path] = typer.Option(
        None, "--in", help="既存の raw JSONL から始める (generate を飛ばす)"),
    mix: Optional[str] = typer.Option(None, "--mix", help="kg_poc のセル配分 (generate と同形式)"),
    top_k: int = typer.Option(DEFAULT_TOP_K, help="rag-verify の検索上位件数"),
    seed: int = typer.Option(42),
    allow_partial: bool = typer.Option(
        False, "--allow-partial",
        help="判定失敗が残っていても最後まで流す (既定は失敗を検出したら止めて報告)",
    ),
) -> None:
    """標準パイプライン: generate → filter → enrich → answer-grounding(降格) → probe → rag-verify(実測ラベル)。

    成果物は <out>/<batch>.jsonl (原石) と <out>/<batch>.filtered.jsonl (全部入り) の2つだけ。
    filter 以降の全工程は全部入りファイルに in-place で追記され、変種ファイルは増えない。

    判定 (LLM) が失敗した QA はラベルを付けずに残す。既定では工程の切れ目で
    未判定が残っていないかを確かめ、残っていれば止める — 障害のまま先へ進めると
    「未判定」が集計から漏れて、あたかも全件検証済みのように見えるため。
    """
    from .answer_grounding import grounding_batch
    from .enrich import enrich_batch
    from .generate import parse_kg_mix
    from .probe import probe_batch as probe_run

    # 1. 生成 (または既存 raw)
    if in_path is None:
        raw = generate_batch(
            chunks_dir=chunks, out_dir=out, n=n, model=model, track=track,
            seed=seed, kg_mix=parse_kg_mix(mix) if mix else None,
        )
    else:
        raw = in_path
    typer.echo(f"[pipeline] 1/6 原石: {raw}")

    # 2. 判定フィルタ → 全部入りファイル
    filtered = out / (raw.stem + ".filtered.jsonl")
    filter_batch(
        raw_path=raw, out_dir=out, chunks_dir=chunks,
        judge_model=judge_model, out_path=filtered,
    )
    n_kept = sum(1 for l in filtered.open(encoding="utf-8") if l.strip())
    typer.echo(f"[pipeline] 2/6 フィルタ通過: {n_kept} 問 → {filtered}")
    if n_kept == 0:
        typer.echo("[pipeline] 通過 0 問のため以降の工程を省略")
        return

    # 3. relation_edges 付与 (決定論)
    enrich_batch(filtered, chunks_dir=chunks)
    typer.echo("[pipeline] 3/6 relation_edges 付与済み")

    # 4. answer-grounding (leave-one-out) + 実態に合わせた降格
    grounding_batch(filtered, judge_model=judge_model, out_path=filtered, demote=True)
    typer.echo("[pipeline] 4/6 answer-grounding + 降格済み")

    _halt_on_unverified(filtered, "answer-grounding", allow_partial,
                        lambda qa: bool(qa.answer_grounding and qa.answer_grounding.errors))

    # 5. probe (LLM 既知性)
    probe_run(in_path=filtered, out_path=filtered, probe_model=model, judge_model=judge_model)
    typer.echo("[pipeline] 5/6 probe 済み")
    _halt_on_unverified(filtered, "probe", allow_partial,
                        lambda qa: qa.llm_knowledge is None)

    # 6. rag-verify (実測ラベル反映込み)
    rag_verify_batch(
        in_path=filtered, chunks_dir=chunks, top_k=top_k,
        rag_model=model, judge_model=judge_model,
    )
    _halt_on_unverified(filtered, "rag-verify", allow_partial,
                        lambda qa: qa.rag_verification is None
                        or not qa.rag_verification.judge_ok)
    typer.echo(f"[pipeline] 6/6 rag-verify 済み — 全部入り: {filtered}")


@app.command("filter")
def filter_cmd(
    in_path: Path = typer.Option(..., "--in", help="Raw JSONL path"),
    out: Path = typer.Option(Path("data/filtered"), help="Output dir"),
    chunks: Path = typer.Option(Path("data/chunks")),
    model: str = typer.Option(
        lambda: _judge_default("gpt-4.1-mini"),
        help="Judge model (Azure deployment name, vLLM model, or Claude id)",
    ),
    prompt: Path = typer.Option(Path("prompts/judge.md")),
    answerability_min: float = typer.Option(4.0),
    grounding_min: float = typer.Option(4.0),
    uniqueness_max: float = typer.Option(0.92),
    skip_uniqueness: bool = typer.Option(False, help="Skip embedding-based dedup"),
    skip_leakage: bool = typer.Option(False, help="Don't reject QAs with leakage=fail (judge is conservative)"),
    rationale_grounded_min: float = typer.Option(
        0.5,
        help="Min fraction of rationale entries verifiable as chunk substrings. "
             "0.5 = 半数以上が逐語必須 (既定)。1.0 = 全件逐語必須。0.0 = 無効化。",
    ),
    reject_too_easy: bool = typer.Option(
        False, "--reject-too-easy",
        help="filter_scores.difficulty_match=='too_easy' の QA を棄却",
    ),
    require_rag_fail: bool = typer.Option(
        False, "--require-rag-fail",
        help="rag_verification.answer_match=='no_match' のものだけ残す "
             "(KG-RAG が活きる問いの抽出)。rag-verify 未実行なら警告して無視",
    ),
    require_rag_hit: bool = typer.Option(
        False, "--require-rag-hit",
        help="rag_verification.answer_match=='match' のものだけ残す "
             "(vector RAG で解ける問いの sanity-check 用)",
    ),
    require_rationale_retrieved: bool = typer.Option(
        False, "--require-rationale-retrieved",
        help="rag_verification.retrieval_hit_chunk=True (根拠本文を逐語で含むチャンクが"
             "上位 k に入った) のものだけ残す。旧 JSONL は retrieval_hit (文書一致) で代用",
    ),
) -> None:
    """Apply 6-perspective judge scoring + threshold filter + dedup."""
    if require_rag_fail and require_rag_hit:
        typer.echo(
            "[filter] ERROR: --require-rag-fail と --require-rag-hit は相反します "
            "(no_match と match の両方は満たせず、全件 drop されます)",
            err=True,
        )
        raise typer.Exit(2)
    filter_batch(
        raw_path=in_path,
        out_dir=out,
        chunks_dir=chunks,
        judge_model=model,
        prompt_path=prompt,
        answerability_min=answerability_min,
        grounding_min=grounding_min,
        uniqueness_max=uniqueness_max,
        compute_uniqueness=not skip_uniqueness,
        require_leakage_pass=not skip_leakage,
        rationale_grounded_min=rationale_grounded_min,
        reject_too_easy=reject_too_easy,
        require_rag_fail=require_rag_fail,
        require_rag_hit=require_rag_hit,
        require_rationale_retrieved=require_rationale_retrieved,
    )


@app.command()
def probe(
    in_path: Path = typer.Option(..., "--in", help="JSONL of QAs to probe"),
    out_path: Optional[Path] = typer.Option(None, "--out", help="Output JSONL (default: overwrite --in)"),
    probe_model: str = typer.Option(
        lambda: os.getenv("VLLM_MODEL", "openai/gpt-oss-120b"),
        help="Base LLM to probe (should be the model used in deployment)",
    ),
    judge_model: str = typer.Option(
        lambda: _judge_default("gpt-4.1-mini"),
        help="Judge model to compare candidate vs ground truth",
    ),
) -> None:
    """KG-PoC: probe each QA against base LLM (no RAG). Sets llm_knowledge."""
    probe_batch(
        in_path=in_path,
        out_path=out_path,
        probe_model=probe_model,
        judge_model=judge_model,
    )


@app.command("answer-grounding")
def answer_grounding_cmd(
    in_path: Path = typer.Option(..., "--in", help="JSONL of QAs (must have rationale)"),
    out_path: Optional[Path] = typer.Option(None, "--out", help="Output JSONL"),
    judge_model: str = typer.Option(
        lambda: _judge_default("gpt-4.1"),
        help="Leave-one-out judge model",
    ),
    prompt: Path = typer.Option(Path("prompts/answer_grounding.md")),
    demote: bool = typer.Option(
        False, "--demote/--no-demote",
        help="全根拠必須が不成立の複数根拠 QA を single_fact に降格する (pipeline は常に有効)",
    ),
) -> None:
    """各 rationale の doc を leave-one-out で伏せて、answer の固有値が決まるかを judge LLM で判定。

    本丸 (KG-RAG の効きどころ) の必要条件「全 doc が answer に必要 (どの doc を伏せても
    answer が決まらない)」を意味的に確認する。結果は qa.answer_grounding に永続化される。
    """
    from .answer_grounding import grounding_batch
    grounding_batch(
        in_path=in_path,
        out_path=out_path,
        judge_model=judge_model,
        prompt_path=prompt,
        demote=demote,
    )


@app.command("rag-verify")
def rag_verify_cmd(
    in_path: Path = typer.Option(..., "--in", help="JSONL of QAs to verify"),
    out_path: Optional[Path] = typer.Option(None, "--out", help="Output JSONL (default: overwrite --in)"),
    chunks: Path = typer.Option(Path("data/chunks"), help="Chunks dir (検索対象)"),
    top_k: int = typer.Option(DEFAULT_TOP_K, help="検索結果の上位件数"),
    rag_model: str = typer.Option(
        lambda: os.getenv("VLLM_MODEL", "google/gemma-4-26B-A4B-it"),
        help="検索結果を context に回答するモデル (vector RAG 本番想定モデル)",
    ),
    judge_model: str = typer.Option(
        lambda: _judge_default("gpt-4.1-mini"),
        help="候補回答と ground truth を照合するモデル",
    ),
    prompt: Path = typer.Option(Path("prompts/rag_verify.md")),
    update_labels: bool = typer.Option(
        True, "--update-labels/--no-update-labels",
        help="実測 (hit_chunk/hit_doc) から retrieval_level を上書きする (履歴は difficulty_rationale に残る)",
    ),
) -> None:
    """vector RAG が解けるかを ground truth として測り rag_verification を後付け。

    質問を埋め込み → top-k チャンク取得 → そのチャンクだけを根拠に rag_model が回答
    → judge_model が ground truth と照合し match/partial/no_match を返す。
    """
    rag_verify_batch(
        in_path=in_path,
        out_path=out_path,
        chunks_dir=chunks,
        top_k=top_k,
        rag_model=rag_model,
        judge_model=judge_model,
        prompt_path=prompt,
        update_labels=update_labels,
    )


def _launch_review_ui(in_path: str, out: Path, chunks: Path, reviewer: str, track: str) -> None:
    module_path = Path(__file__).resolve().parent / "review_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(module_path),
        "--",
        "--in",
        in_path,
        "--out",
        str(out),
        "--chunks",
        str(chunks),
        "--reviewer",
        reviewer,
        "--track",
        track,
    ]
    subprocess.run(cmd, check=False)


@app.command()
def review(
    in_path: str = typer.Option(..., "--in", help="Filtered JSONL path or glob"),
    out: Path = typer.Option(Path("data/reviewed"), help="Output dir"),
    chunks: Path = typer.Option(Path("data/chunks"), help="Chunks dir (for anchor display)"),
    reviewer: str = typer.Option("unknown"),
    track: str = typer.Option(
        "general", "--track",
        help="どの track の QA を表示するか: general | kg_poc | all (既定: general)",
    ),
) -> None:
    """Launch the Streamlit review UI for general-track QAs (既定)."""
    _launch_review_ui(in_path, out, chunks, reviewer, track)


@app.command("review-kg")
def review_kg(
    in_path: str = typer.Option(..., "--in", help="Filtered JSONL path or glob"),
    out: Path = typer.Option(Path("data/reviewed"), help="Output dir"),
    chunks: Path = typer.Option(Path("data/chunks"), help="Chunks dir (for anchor display)"),
    reviewer: str = typer.Option("unknown"),
) -> None:
    """KG-PoC track 専用レビューUI (KG QA だけ表示・KG専用チェックリスト)."""
    _launch_review_ui(in_path, out, chunks, reviewer, "kg_poc")


def _launch_stats_ui(
    in_path: str, track: str, reviewed_out: Path, chunks: Path, reviewer: str,
) -> None:
    module_path = Path(__file__).resolve().parent / "stats_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(module_path),
        "--",
        "--in",
        in_path,
        "--track",
        track,
        "--reviewed-out",
        str(reviewed_out),
        "--chunks",
        str(chunks),
        "--reviewer",
        reviewer,
    ]
    subprocess.run(cmd, check=False)


@app.command()
def stats(
    in_path: str = typer.Option(
        "data/raw", "--in", help="QA JSONL file, directory, or glob (any stage)"
    ),
    track: str = typer.Option(
        "general", "--track",
        help="どの track の QA を表示するか: general | kg_poc | all (既定: general)",
    ),
    reviewed_out: Path = typer.Option(Path("data/reviewed"), "--reviewed-out",
        help="レビュー保存先 (レビュータブ用)"),
    chunks: Path = typer.Option(Path("data/chunks"), "--chunks",
        help="チャンクディレクトリ (元チャンク表示用)"),
    reviewer: str = typer.Option("unknown", "--reviewer"),
) -> None:
    """Launch the Streamlit dashboard (既定: general track, レビュータブ統合)."""
    _launch_stats_ui(in_path, track, reviewed_out, chunks, reviewer)


@app.command()
def web(
    port: int = typer.Option(8765, help="待ち受けポート"),
    host: str = typer.Option("127.0.0.1", help="待ち受けホスト"),
    data_root: Path = typer.Option(Path("data"), "--data-root",
        help="データ置き場 (dataset/chunks/reviewed をこの下から探す)"),
    reviewer: str = typer.Option("unknown", help="レビュー保存時の reviewed_by 既定値"),
    open_browser: bool = typer.Option(True, "--open/--no-open",
        help="起動時にブラウザを開くか"),
) -> None:
    """React 版 Web UI (ダッシュボード + レビュー) を起動する。

    データセット・トラック・チャンクディレクトリは画面上で切り替えられる。
    レビューの書き込み先は Streamlit 版と同じ data/reviewed/<元ファイル名>。
    """
    from .webapi.app import run_server
    run_server(
        host=host, port=port, data_root=data_root,
        reviewer=reviewer, open_browser=open_browser,
    )


@app.command("stats-kg")
def stats_kg(
    in_path: str = typer.Option(
        "data/raw", "--in", help="QA JSONL file, directory, or glob (any stage)"
    ),
    reviewed_out: Path = typer.Option(Path("data/reviewed"), "--reviewed-out"),
    chunks: Path = typer.Option(Path("data/chunks"), "--chunks"),
    reviewer: str = typer.Option("unknown", "--reviewer"),
) -> None:
    """KG-PoC track 専用ダッシュボード (KG QA のみ + KG向けタブ構成)."""
    _launch_stats_ui(in_path, "kg_poc", reviewed_out, chunks, reviewer)


if __name__ == "__main__":
    app()
