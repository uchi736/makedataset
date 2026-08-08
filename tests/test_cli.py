"""CLI (typer) の規則そのものを凍結する試験。

ここで守りたいのは「命令側が第2・第3の正本を持ち始める」再発:
  - 既定の生成モデルが命令ごとに別の直値になる (rank 12)
  - 型別規則が素材選び・生成器選びの2本の if 連鎖に割れる (rank 23)
  - 台帳が data/seeds 決め打ちで、見失っても空規則のまま走る (rank 29)
  - 品質ゲートの合格線を命令側で再宣言する (rank 35)
どれも例外を出さずに結果だけ変える種類の壊れ方なので、値でなく
「どこから引いているか」を試験で押さえる。
"""

from __future__ import annotations

import inspect
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from rageval import cli
from rageval import filter as filter_mod
from rageval.tracks.kg_poc import ALL_KG_QUERY_TYPES

runner = CliRunner()


def _option_default(command_name: str, param: str):
    """登録済み命令の typer.Option 既定値 (OptionInfo.default) を取り出す。"""
    for command in cli.app.registered_commands:
        name = command.name or command.callback.__name__.replace("_", "-")
        if name == command_name:
            info = inspect.signature(command.callback).parameters[param].default
            return info.default
    raise AssertionError(f"命令が見つからない: {command_name}")


# ---------------- rank 12: 既定の生成モデル ----------------

# 生成側モデルを既定で決める全ての引数。ここに載っているものは1つ残らず
# _generator_default を指していなければならない
GENERATOR_MODEL_PARAMS = [
    ("discover-patterns", "model"),
    ("generate", "model"),
    ("kg-generate", "model"),
    ("pipeline", "model"),
    ("probe", "probe_model"),
    ("rag-verify", "rag_model"),
]


@pytest.mark.parametrize("command,param", GENERATOR_MODEL_PARAMS)
def test_generator_model_defaults_share_one_resolver(command, param):
    assert _option_default(command, param) is cli._generator_default


def test_generator_default_prefers_vllm_model(monkeypatch):
    monkeypatch.setenv("VLLM_MODEL", "on-prem/whatever")
    assert cli._generator_default() == "on-prem/whatever"


def test_generator_default_falls_back_to_one_constant(monkeypatch):
    monkeypatch.delenv("VLLM_MODEL", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "azure-deployment")
    # クラウド配備名へは落とさない (生成は件数が多く、設定漏れの課金が痛い)
    assert cli._generator_default() == cli.GENERATOR_MODEL_FALLBACK


def test_no_second_generator_model_literal_in_cli():
    """モデル名の直値が定数以外に現れたら落ちる (rank 12 の再発検出)。"""
    src = Path(cli.__file__).read_text(encoding="utf-8")
    literals = re.findall(r"[\"'][\w./-]*(?:gpt-oss|gemma|qwen|gpt-4)[\w./-]*[\"']", src)
    assert literals.count(f'"{cli.GENERATOR_MODEL_FALLBACK}"') == 1
    # 判定側 (_judge_default の fallback) は別系統なので除外して数える
    others = [lit for lit in literals
              if lit.strip("\"'") not in (cli.GENERATOR_MODEL_FALLBACK, "gpt-4.1", "gpt-4.1-mini")]
    assert others == []


# ---------------- rank 23: 型別表 ----------------

def test_kg_track_names_match_schema_vocabulary():
    """CLI 名がスキーマ上の型名と同じ綴りであること (第3の語彙を作らない)。"""
    from rageval.generate_kg import CHECKS_BY_TYPE

    for name, track in cli.KG_TRACKS.items():
        assert name == track.schema_type
        assert track.schema_type in CHECKS_BY_TYPE, f"検品テーブルに無い型: {name}"

    # single_fact は multi_hop の降格先なので専用生成しない。それ以外の
    # KGQueryType は全て表にある (negation ↔ negation_exhaustive のずれ再発検出)
    assert set(ALL_KG_QUERY_TYPES) - set(cli.KG_TRACKS) == {"single_fact"}


def test_kg_track_aliases_resolve_to_canonical_names():
    for old, new in cli.KG_TRACK_ALIASES.items():
        assert cli._resolve_kg_track(old) == new
    assert cli._resolve_kg_track("multi_hop") == "multi_hop"
    assert cli._resolve_kg_track("nonexistent") is None


# ---------------- kg-generate の配線 ----------------

class _FakeSession:
    pass


@pytest.fixture
def kg_stub(monkeypatch, tmp_path):
    """KG・埋め込み・生成器を差し替え、CLI の配線だけを見る。"""
    from rageval import generate_kg, kg, sampling

    calls: dict[str, object] = {}

    @contextmanager
    def _session():
        yield _FakeSession()

    monkeypatch.setattr(kg, "session", _session)
    monkeypatch.setattr(kg, "snapshot_id", lambda sess: "snap-1")
    monkeypatch.setattr(kg, "chunk_node_map", lambda sess: {"d__c0": {"N"}})
    monkeypatch.setattr(kg, "iter_two_hop_paths",
                        lambda sess, **kw: (calls.update(two_hop=kw), ["chain-material"])[1])
    monkeypatch.setattr(kg, "iter_star_groups",
                        lambda sess, **kw: (calls.update(stars=kw), ["star-material"])[1])
    monkeypatch.setattr(kg, "iter_excludes_edges",
                        lambda sess, **kw: (calls.update(excludes=kw), ["excl-material"])[1])
    monkeypatch.setattr(generate_kg, "find_decoy_pairs",
                        lambda *a, **kw: (calls.update(decoy=kw), ["decoy-material"])[1])
    monkeypatch.setattr(generate_kg, "find_missing_reference_pairs",
                        lambda *a, **kw: ["missing-material"])
    monkeypatch.setattr(sampling, "compute_embeddings", lambda chunks: [[1.0]])
    monkeypatch.setattr(sampling, "load_cross_doc_policy",
                        lambda anchors, aliases: SimpleNamespace(anchors=anchors, aliases=aliases))
    monkeypatch.setattr(cli, "load_chunks",
                        lambda d: [SimpleNamespace(chunk_id="d__c0", text="本文")])

    def _record(kind):
        def _fn(materials, **kwargs):
            calls[kind] = {"materials": materials, **kwargs}
            return tmp_path / "out.jsonl"
        return _fn

    for kind in ("paths", "stars", "exclusions", "references", "decoys"):
        monkeypatch.setattr(generate_kg, f"generate_from_{kind}", _record(kind))
    return calls


def _run_kg_generate(tmp_path, *args):
    return runner.invoke(cli.app, [
        "kg-generate", "--chunks", str(tmp_path / "chunks"),
        "--out", str(tmp_path / "out"), *args,
    ])


# (CLI 名, 呼ばれるべき生成器, 渡されるべき素材)
TRACK_WIRING = [
    ("multi_hop", "paths", "chain-material"),
    ("traceability", "paths", "chain-material"),
    ("aggregation", "stars", "star-material"),
    ("negation_exhaustive", "excl", "excl-material"),
    ("kg_missing", "references", "missing-material"),
    ("kg_decoy", "decoys", "decoy-material"),
]


@pytest.mark.parametrize("name,generator,material", [
    (n, g if g != "excl" else "exclusions", m) for n, g, m in TRACK_WIRING
])
def test_each_track_uses_its_own_materials_and_generator(
    kg_stub, tmp_path, name, generator, material,
):
    """素材と生成器が同じ表の行から来ること。

    以前は2本の if 連鎖がどちらも裸の else = multi_hop で終わっており、
    片方の追記漏れでその型だけ黙って multi_hop の素材・生成器で作られた。
    """
    result = _run_kg_generate(tmp_path, "--query-type", name, "--n", "3")
    assert result.exit_code == 0, result.output
    assert generator in kg_stub, f"{name} で {generator} が呼ばれていない"
    recorded = kg_stub[generator]
    assert recorded["materials"] == [material]
    assert recorded["max_items"] == 3
    assert recorded["kg_snapshot"] == "snap-1"
    # 他の生成器は呼ばれていない
    for other in ("paths", "stars", "exclusions", "references", "decoys"):
        if other != generator:
            assert other not in kg_stub


def test_multi_hop_requires_leak_free_terminal_but_traceability_does_not(kg_stub, tmp_path):
    assert _run_kg_generate(tmp_path, "--query-type", "multi_hop").exit_code == 0
    assert kg_stub["two_hop"]["require_leak_free_terminal"] is True
    assert kg_stub["paths"]["terminal_leak_free"] is True
    assert "query_type_override" not in kg_stub["paths"]

    kg_stub.clear()
    assert _run_kg_generate(tmp_path, "--query-type", "traceability").exit_code == 0
    assert kg_stub["two_hop"]["require_leak_free_terminal"] is False
    assert kg_stub["paths"]["terminal_leak_free"] is False
    assert kg_stub["paths"]["query_type_override"] == "traceability"
    # 段2プロンプトの正本は generate_kg 側の定数 (CLI に写しを作らない)
    from rageval.generate_kg import WRITE_TRACE_PROMPT
    assert kg_stub["paths"]["prompt_path"] == Path(WRITE_TRACE_PROMPT)


def test_old_query_type_names_still_work(kg_stub, tmp_path):
    result = _run_kg_generate(tmp_path, "--query-type", "negation")
    assert result.exit_code == 0, result.output
    assert kg_stub["exclusions"]["materials"] == ["excl-material"]


def test_mix_accepts_canonical_and_old_names(kg_stub, tmp_path):
    result = _run_kg_generate(tmp_path, "--mix", "aggregation=2,negation=1")
    assert result.exit_code == 0, result.output
    assert kg_stub["stars"]["max_items"] == 2
    assert kg_stub["exclusions"]["max_items"] == 1


def test_unknown_query_type_exits(kg_stub, tmp_path):
    assert _run_kg_generate(tmp_path, "--query-type", "temporal").exit_code == 1
    assert _run_kg_generate(tmp_path, "--mix", "temporal=3").exit_code == 1


def test_empty_materials_warns_and_skips_generation(kg_stub, tmp_path, monkeypatch):
    """素材 0 は例外にならず空ファイルで正常終了するので、警告を出して止める。"""
    from rageval import kg

    monkeypatch.setattr(kg, "iter_star_groups", lambda sess, **kw: [])
    result = _run_kg_generate(tmp_path, "--query-type", "aggregation")
    assert result.exit_code == 0
    assert "素材が 0" in result.output
    assert "stars" not in kg_stub


# ---------------- rank 29: 台帳 ----------------

def test_kg_generate_passes_ledger_options_through(kg_stub, tmp_path):
    anchors = tmp_path / "anchors.jsonl"
    aliases = tmp_path / "aliases.json"
    anchors.write_text("", encoding="utf-8")
    aliases.write_text("{}", encoding="utf-8")

    result = _run_kg_generate(
        tmp_path, "--query-type", "kg_missing",
        "--anchor-pairs", str(anchors), "--doc-aliases", str(aliases))
    assert result.exit_code == 0, result.output
    assert kg_stub["references"]["policy"].anchors == anchors
    assert kg_stub["references"]["policy"].aliases == aliases


@pytest.mark.parametrize("query_type", ["kg_missing", "kg_decoy"])
def test_missing_ledger_stops_instead_of_running_with_empty_policy(
    kg_stub, tmp_path, query_type,
):
    """台帳を見失ったら落とす。

    load_cross_doc_policy は台帳が無くても空規則を返すので、そのまま走ると
    kg_decoy は「本当は繋がっている対」を棄権が正解として出荷し、
    kg_missing は 0 対で正常終了する — ラベルの意味だけが静かに反転する。
    """
    result = _run_kg_generate(
        tmp_path, "--query-type", query_type,
        "--anchor-pairs", str(tmp_path / "nope.jsonl"))
    assert result.exit_code == 1
    assert "references" not in kg_stub and "decoys" not in kg_stub


def test_generate_and_kg_generate_share_ledger_defaults():
    assert _option_default("generate", "anchor_pairs") == cli.DEFAULT_ANCHOR_PAIRS
    assert _option_default("kg-generate", "anchor_pairs") == cli.DEFAULT_ANCHOR_PAIRS
    assert _option_default("generate", "doc_aliases") == cli.DEFAULT_DOC_ALIASES
    assert _option_default("kg-generate", "doc_aliases") == cli.DEFAULT_DOC_ALIASES


# ---------------- rank 10: デコイ選別の3値 ----------------

def test_decoy_knobs_are_passed_through(kg_stub, tmp_path):
    result = _run_kg_generate(
        tmp_path, "--query-type", "kg_decoy",
        "--decoy-sim-floor", "0.8", "--decoy-position-gap", "7",
        "--decoy-per-doc-pair", "3")
    assert result.exit_code == 0, result.output
    assert kg_stub["decoy"]["sim_floor"] == 0.8
    assert kg_stub["decoy"]["min_position_gap"] == 7
    assert kg_stub["decoy"]["max_per_doc_pair"] == 3
    assert "0.8 (指定)" in result.output


def test_decoy_knobs_default_to_generate_kg_constants(kg_stub, tmp_path):
    """未指定なら値を渡さない (既定の正本は generate_kg の名前つき定数)。"""
    result = _run_kg_generate(tmp_path, "--query-type", "kg_decoy", "--n", "5")
    assert result.exit_code == 0, result.output
    assert kg_stub["decoy"] == {"top_n": 20}   # max(n*4, 20)
    assert "generate_kg 既定" in result.output


# ---------------- rank 35: 品質ゲートの合格線 ----------------

@pytest.mark.parametrize("param,constant", [
    ("answerability_min", "DEFAULT_ANSWERABILITY_MIN"),
    ("grounding_min", "DEFAULT_GROUNDING_MIN"),
    ("uniqueness_max", "DEFAULT_UNIQUENESS_MAX"),
    ("rationale_grounded_min", "DEFAULT_RATIONALE_GROUNDED_MIN"),
])
def test_filter_thresholds_come_from_filter_module(param, constant):
    assert _option_default("filter", param) == getattr(filter_mod, constant)


def test_filter_prompt_default_comes_from_filter_module():
    assert _option_default("filter", "prompt") == Path(filter_mod.DEFAULT_PROMPT)


# ---------------- rank 18: ハブ判定の閾値 ----------------

def test_max_pagerank_default_comes_from_kg_module():
    """CLI が直値で上書きすると kg 側の定数が無音で空振りする。"""
    from rageval import kg

    assert _option_default("kg-generate", "max_pagerank") is cli._max_pagerank_default
    assert cli._max_pagerank_default() == kg.DEFAULT_MAX_PAGERANK
