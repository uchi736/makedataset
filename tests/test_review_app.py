"""Streamlit 画面 (review_app / stats_app) の回帰テスト。

レビュー結果を失わない保存、保存先の規約、表示ラベル・合格線の一本化を凍結する。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from rageval.review_app import (
    _checklist_groups,
    _merge_reviewed_items,
    _resolve_review_paths,
    _save_review_item,
)
from rageval.schema import (
    Explainability,
    GenerationInfo,
    QAItem,
    Rationale,
    ReasoningComplexity,
    RetrievalDifficulty,
    SourceStructure,
)


def _qa(qa_id: str, *, kg: bool = False) -> QAItem:
    return QAItem(
        qa_id=qa_id,
        question=f"{qa_id} question",
        answer=f"{qa_id} answer",
        rationale=[Rationale(doc_id="doc", text="根拠として十分な長さを持つテキストです。")],
        category=[] if kg else ["Reasoning"],
        aspect=[] if kg else ["quantitative_calc"],
        kg_query_type="multi_hop" if kg else None,
        kg_novelty="unknown_relation" if kg else None,
        reasoning_complexity=ReasoningComplexity(),
        retrieval_difficulty=RetrievalDifficulty(),
        source_structure=SourceStructure(),
        explainability=Explainability(evidence_strictness="hier-ref"),
        retrieval_level="Easy",
        answer_level="Easy",
        difficulty_rationale="test",
        generation=GenerationInfo(
            model="mock", prompt_version="test", generated_at=datetime.now(),
        ),
    )


def test_merge_reviewed_items_restores_rows_missing_from_old_partial_mirror():
    general = _qa("general")
    kg = _qa("kg", kg=True)
    reviewed_general = general.model_copy(deep=True)
    reviewed_general.review_status = "accepted"

    merged = _merge_reviewed_items([general, kg], [reviewed_general])

    assert [qa.qa_id for qa in merged] == ["general", "kg"]
    assert [qa.review_status for qa in merged] == ["accepted", "pending"]


def test_save_review_item_preserves_other_tracks_legacy_lines_and_prior_reviews(
    tmp_path: Path,
):
    source = tmp_path / "filtered" / "batch.jsonl"
    source.parent.mkdir()
    general = _qa("general")
    kg = _qa("kg", kg=True)
    legacy = {"qa_id": "legacy", "old_schema": True}
    source.write_text(
        general.model_dump_json()
        + "\n"
        + kg.model_dump_json()
        + "\n"
        + json.dumps(legacy, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    source_before = source.read_bytes()

    mirror = tmp_path / "reviewed" / source.name
    mirror.parent.mkdir()
    reviewed_general = general.model_copy(deep=True)
    reviewed_general.review_status = "accepted"
    # Simulate a mirror written by the old track-filtered implementation.
    mirror.write_text(reviewed_general.model_dump_json() + "\n", encoding="utf-8")

    reviewed_kg = kg.model_copy(deep=True)
    reviewed_kg.review_status = "rejected"
    _save_review_item(reviewed_kg, source, mirror)

    lines = mirror.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert QAItem.model_validate_json(lines[0]).review_status == "accepted"
    assert QAItem.model_validate_json(lines[1]).review_status == "rejected"
    assert json.loads(lines[2]) == legacy
    assert source.read_bytes() == source_before
    assert not list(mirror.parent.glob(f".{mirror.name}.*.tmp"))


def test_save_review_item_seeds_from_legacy_flat_mirror(tmp_path: Path):
    """回帰: 入れ子の保存先へ移った直後、土台を平たい旧配置から引くこと。

    旧配置を見ないと、1問保存した瞬間に「その1問以外すべて未判定」の
    ミラーが新規に作られ、以後そちらが優先されて過去の人手判定が消える。
    """
    source = tmp_path / "filtered_kg" / "batch.jsonl"
    source.parent.mkdir()
    general, kg = _qa("general"), _qa("kg", kg=True)
    source.write_text(
        general.model_dump_json() + "\n" + kg.model_dump_json() + "\n",
        encoding="utf-8",
    )

    # 旧配置 (data/reviewed/batch.jsonl) にだけ判定済みが残っている状態
    legacy_mirror = tmp_path / "reviewed" / source.name
    legacy_mirror.parent.mkdir()
    reviewed_general = general.model_copy(deep=True)
    reviewed_general.review_status = "accepted"
    legacy_mirror.write_text(
        reviewed_general.model_dump_json() + "\n" + kg.model_dump_json() + "\n",
        encoding="utf-8",
    )

    mirror = tmp_path / "reviewed" / "filtered_kg" / source.name
    mirror.parent.mkdir(parents=True)
    reviewed_kg = kg.model_copy(deep=True)
    reviewed_kg.review_status = "rejected"
    _save_review_item(reviewed_kg, source, mirror, legacy_mirror)

    lines = mirror.read_text(encoding="utf-8").splitlines()
    assert [QAItem.model_validate_json(l).review_status for l in lines] == [
        "accepted", "rejected",
    ]
    # 旧配置は読むだけ (書き戻さない)
    legacy_lines = legacy_mirror.read_text(encoding="utf-8").splitlines()
    assert QAItem.model_validate_json(legacy_lines[1]).review_status == "pending"


# ---------------- 保存先の規約 (rank 3) ----------------

def test_review_paths_do_not_collide_across_datasets(tmp_path: Path):
    """同名ファイルが保存先を奪い合わないこと。

    filter が入力名をそのまま出力名にするため raw_x/batch.jsonl と
    filtered_x/batch.jsonl は普通に併存する。平たい「保存先/入力ファイル名」だと
    片方を保存した瞬間にもう片方の人手判定が無警告で消えていた。
    """
    data = tmp_path / "data"
    out = data / "reviewed"
    raw, filtered = data / "raw_x" / "batch.jsonl", data / "filtered_x" / "batch.jsonl"
    for p in (raw, filtered):
        p.parent.mkdir(parents=True)
        p.write_text("", encoding="utf-8")

    raw_mirror, raw_legacy = _resolve_review_paths(raw, out)
    filtered_mirror, _ = _resolve_review_paths(filtered, out)

    assert raw_mirror != filtered_mirror
    assert raw_mirror == out / "raw_x" / "batch.jsonl"
    assert filtered_mirror == out / "filtered_x" / "batch.jsonl"
    # 平たい旧配置は読み込みの土台としてだけ返る (書き出しには使わない)
    assert raw_legacy == out / "batch.jsonl"


def test_review_paths_use_the_webapi_convention(tmp_path: Path):
    """Streamlit 版と Web 版が同じ場所を読み書きすること (規約の分裂を防ぐ)。"""
    from rageval.webapi.service import review_mirror_path

    data = tmp_path / "data"
    src = data / "raw" / "batch.jsonl"
    src.parent.mkdir(parents=True)
    src.write_text("", encoding="utf-8")

    mirror, _ = _resolve_review_paths(src, data / "reviewed")
    assert mirror == review_mirror_path(data, src)


# ---------------- ラベル・チェックリストの一本化 (rank 37/38) ----------------

def test_checklist_is_shared_with_the_web_ui():
    """減点表の項目が Streamlit と Web で一致すること。

    誤結合 (別文書の同番号の同一視) を指摘する f_join が Web 側にしか無く、
    同じ QA でも見る画面で指摘できる観点が違っていた。
    """
    from rageval.webapi.labels import CHECKLIST_GENERAL, CHECKLIST_KG

    for is_kg, expected in ((True, CHECKLIST_KG), (False, CHECKLIST_GENERAL)):
        groups = _checklist_groups(_qa("x", kg=is_kg))
        assert [g for g, _ in groups] == [g["group"] for g in expected]
        assert [
            [(k, label, desc) for k, label, desc in items] for _, items in groups
        ] == [
            [(it["key"], it["label"], it["desc"]) for it in g["items"]] for g in expected
        ]

    kg_keys = {k for _, items in _checklist_groups(_qa("x", kg=True)) for k, _, _ in items}
    assert "f_join" in kg_keys


def test_diagnostic_labels_come_from_the_shared_table():
    """診断3軸のラベル表を画面ごとに持ち直したら落ちる。"""
    from rageval import review_app
    from rageval.webapi import labels

    assert review_app._DIAG_LABELS is labels.DIAG_LABELS
    assert review_app._CATEGORY_LABELS is labels.CATEGORY_LABELS
    assert review_app._EVIDENCE_LABEL is labels.EVIDENCE_LABELS
    assert review_app._OUTPUT_TYPE_LABEL is labels.OUTPUT_TYPE_LABELS


# ---------------- 集計画面 (rank 31 / 45 / 46 / 47) ----------------

def test_stats_data_root_follows_the_given_input(tmp_path: Path, monkeypatch):
    """--in に渡した置き場を ./data があっても捨てないこと。

    以前は ./data が存在する限りそれを採用したので、別置き場を指定したつもりで
    data 配下の別コーパスの集計が無言で表示されていた。
    """
    from rageval.stats_app import _resolve_data_root

    (tmp_path / "data" / "raw").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    assert _resolve_data_root("外付け/raw") == Path("外付け")
    assert _resolve_data_root("外付け/raw/batch.jsonl") == Path("外付け")
    assert _resolve_data_root("data/raw") == Path("data")
    # 明示指定が最優先
    assert _resolve_data_root("data/raw", "別置き場") == Path("別置き場")


def test_stats_missing_dataset_is_an_error_not_a_silent_swap():
    """--in が候補に無いとき、先頭のデータセットへ差し替えないこと。"""
    from rageval.stats_app import _select_dataset

    assert _select_dataset(["raw", "filtered"], "filtered") == "filtered"
    with pytest.raises(ValueError):
        _select_dataset(["raw", "filtered"], "raw_kg_v1")


def test_stats_discovers_nested_reviewed_datasets(tmp_path: Path):
    """入れ子になったレビュー結果が集計画面の候補から消えないこと。"""
    from rageval.stats_app import _discover_data_sources

    root = tmp_path / "data"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "batch.jsonl").write_text("", encoding="utf-8")
    (root / "reviewed" / "raw").mkdir(parents=True)
    (root / "reviewed" / "raw" / "batch.jsonl").write_text("", encoding="utf-8")

    datasets, _ = _discover_data_sources(root)
    assert dict(datasets) == {"raw": ["batch.jsonl"], "reviewed/raw": ["batch.jsonl"]}


def test_stats_thresholds_and_targets_come_from_one_place():
    """合格線・目標値をこの画面が持ち直したら落ちる。"""
    from rageval import stats_app
    from rageval.webapi import labels

    assert stats_app.ANSWERABILITY_PASS == labels.ANSWERABILITY_PASS
    assert stats_app.GROUNDING_PASS == labels.GROUNDING_PASS
    assert stats_app.UNIQUENESS_PASS == labels.UNIQUENESS_PASS
    assert stats_app.TARGET_ABSTENTION == labels.TARGET_ABSTENTION
    assert list(stats_app.CATEGORY_ORDER) == list(labels.CATEGORY_ORDER)
    # 説明文も目標値から組む (見出しだけ旧値が残らないこと)
    assert stats_app._target_text({"Easy": 0.5, "Hard": 0.25}) == "易 50% / 難 25%"
