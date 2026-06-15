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

app = typer.Typer(add_completion=False, no_args_is_help=True)


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
    )


@app.command("filter")
def filter_cmd(
    in_path: Path = typer.Option(..., "--in", help="Raw JSONL path"),
    out: Path = typer.Option(Path("data/filtered"), help="Output dir"),
    chunks: Path = typer.Option(Path("data/chunks")),
    model: str = typer.Option(
        lambda: os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini"),
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
        lambda: os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini"),
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
        lambda: os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini"),
        help="候補回答と ground truth を照合するモデル",
    ),
    prompt: Path = typer.Option(Path("prompts/rag_verify.md")),
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
