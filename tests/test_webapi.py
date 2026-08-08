"""Tests for the React web UI backend (rageval.webapi)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from rageval.schema import (
    Explainability,
    GenerationInfo,
    QAItem,
    Rationale,
    ReasoningComplexity,
    RetrievalDifficulty,
    SourceStructure,
)
from rageval.webapi.service import (
    ServiceError,
    data_root_for_reviewed_dir,
    legacy_review_mirror_path,
    list_sources,
    load_items,
    resolve_evidence,
    resolve_paths,
    review_mirror_path,
    save_review,
)

CHUNK_TEXT = "定格電圧は DC 24V ±10% とする。周囲温度は 0〜40℃ の範囲で使用すること。"
RATIONALE_TEXT = "定格電圧は DC 24V ±10% とする。"


def _make_qa(qa_id: str, *, kg: bool = False) -> QAItem:
    return QAItem(
        qa_id=qa_id,
        question=f"{qa_id} の質問?",
        answer=f"{qa_id} の回答",
        rationale=[Rationale(doc_id="sample", page=None, text=RATIONALE_TEXT)],
        category=[] if kg else ["Reasoning"],
        aspect=[] if kg else ["quantitative_calc"],
        kg_query_type="multi_hop" if kg else None,
        kg_novelty="unknown_relation" if kg else None,
        reasoning_complexity=ReasoningComplexity(quantitative=True),
        retrieval_difficulty=RetrievalDifficulty(),
        source_structure=SourceStructure(),
        explainability=Explainability(evidence_strictness="hier-ref"),
        retrieval_level="Easy",
        answer_level="Easy",
        difficulty_rationale="必要チャンク=1",
        generation=GenerationInfo(model="mock", prompt_version="v1", generated_at=datetime.now()),
    )


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    raw = root / "raw"
    raw.mkdir(parents=True)
    with (raw / "batch_test.jsonl").open("w", encoding="utf-8") as f:
        f.write(_make_qa("qa_general").model_dump_json() + "\n")
        f.write(_make_qa("qa_kg", kg=True).model_dump_json() + "\n")
        # 旧スキーマ行 (パース不能) — 保存時にも消えてはいけない
        f.write(json.dumps({"qa_id": "qa_legacy", "old_schema": True}, ensure_ascii=False) + "\n")
    chunks = root / "chunks"
    chunks.mkdir()
    chunk = {
        "chunk_id": "sample__c0",
        "doc_id": "sample",
        "page": None,
        "text": CHUNK_TEXT,
        "position": 0,
        "section_path": ["第1章"],
        "references": [],
    }
    (chunks / "sample.jsonl").write_text(
        json.dumps(chunk, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return root


def test_resolve_paths_rejects_outside(data_root: Path):
    with pytest.raises(ServiceError):
        resolve_paths(data_root, "../secrets")


def test_load_items_track_filter(data_root: Path):
    all_items = load_items(data_root, "raw", "all")
    assert {it["qa_id"] for it in all_items["items"]} == {"qa_general", "qa_kg"}
    assert all_items["skipped"] == {"batch_test.jsonl": 1}

    general = load_items(data_root, "raw", "general")
    assert [it["qa_id"] for it in general["items"]] == ["qa_general"]

    kg = load_items(data_root, "raw", "kg_poc")
    assert [it["qa_id"] for it in kg["items"]] == ["qa_kg"]


def test_load_items_counts_duplicate_qa_ids(data_root: Path):
    """同一バッチの派生ファイル (.enriched 等) を丸ごと読むと同じ qa_id が重複する。
    UI が注意を出せるよう duplicate_qa_ids で個数が返ること。"""
    # 重複なしのディレクトリでは 0
    assert load_items(data_root, "raw", "all")["duplicate_qa_ids"] == 0

    variants = data_root / "raw_variants"
    variants.mkdir()
    for name in ("batch_x.jsonl", "batch_x.enriched.jsonl"):
        with (variants / name).open("w", encoding="utf-8") as f:
            f.write(_make_qa("qa_dup1").model_dump_json() + "\n")
            f.write(_make_qa("qa_dup2").model_dump_json() + "\n")
    with (variants / "batch_y.jsonl").open("w", encoding="utf-8") as f:
        f.write(_make_qa("qa_unique").model_dump_json() + "\n")

    res = load_items(data_root, "raw_variants", "all")
    assert len(res["items"]) == 5
    assert res["duplicate_qa_ids"] == 2  # qa_dup1 と qa_dup2 (qa_unique は数えない)


def test_save_review_writes_mirror_and_keeps_other_lines(data_root: Path):
    updated = save_review(
        data_root, "raw", "batch_test.jsonl", "qa_general",
        status="accepted", reviewer="tester",
    )
    assert updated["review_status"] == "accepted"
    assert updated["reviewed_by"] == "tester"

    # 保存先はデータセット別 (data/reviewed/<データセット>/<ファイル名>)
    mirror = data_root / "reviewed" / "raw" / "batch_test.jsonl"
    assert mirror.exists()
    lines = mirror.read_text(encoding="utf-8").strip().splitlines()
    # 3行とも残る (旧スキーマ行も、track の異なる QA も落ちない)
    assert len(lines) == 3
    assert json.loads(lines[2])["qa_id"] == "qa_legacy"

    # 元ファイルは書き換えない
    src_lines = (data_root / "raw" / "batch_test.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(src_lines[0])["review_status"] == "pending"

    # ミラーがある状態で読み直すと accepted が見える
    reloaded = load_items(data_root, "raw", "all")
    assert reloaded["reviewed_override"] == ["batch_test.jsonl"]
    status = {it["qa_id"]: it["review_status"] for it in reloaded["items"]}
    assert status["qa_general"] == "accepted"

    # 2回目の保存 (edited + 本文書き換え) はミラーを土台にする
    save_review(
        data_root, "raw", "batch_test.jsonl", "qa_kg",
        status="edited", reviewer="tester",
        question="書き直した質問?", answer="書き直した回答",
    )
    again = load_items(data_root, "raw", "all")
    by_id = {it["qa_id"]: it for it in again["items"]}
    assert by_id["qa_general"]["review_status"] == "accepted"  # 1回目の結果が残る
    assert by_id["qa_kg"]["question"] == "書き直した質問?"


def test_save_review_unknown_qa_id(data_root: Path):
    with pytest.raises(ServiceError):
        save_review(data_root, "raw", "batch_test.jsonl", "no_such_id",
                    status="accepted", reviewer="t")


def test_resolve_evidence_spans(data_root: Path):
    qa = _make_qa("qa_general")
    ev = resolve_evidence(data_root, "chunks", [r.model_dump() for r in qa.rationale])
    assert ev["chunks_found"] is True
    assert ev["all_match"] is True
    assert ev["used_chunks_dir"] == "chunks"
    assert len(ev["chunks"]) == 1
    chunk = ev["chunks"][0]
    (start, end) = chunk["spans"][0]
    assert chunk["text"][start:end] == RATIONALE_TEXT


def test_resolve_evidence_auto_picks_matching_dir(data_root: Path):
    """"auto" と、明示指定が全滅した場合のフォールバックの両方を確認する。"""
    # 一致しないチャンクだけの chunks_other を作る
    other = data_root / "chunks_other"
    other.mkdir()
    (other / "x.jsonl").write_text(
        json.dumps({
            "chunk_id": "x__c0", "doc_id": "x", "page": None,
            "text": "無関係な本文", "position": 0, "section_path": [], "references": [],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rationale = [{"doc_id": "sample", "page": None, "text": RATIONALE_TEXT}]

    ev = resolve_evidence(data_root, "auto", rationale)
    assert ev["all_match"] is True
    assert ev["used_chunks_dir"] == "chunks"

    # 明示指定が外れていても、一致するディレクトリへフォールバックする
    ev2 = resolve_evidence(data_root, "chunks_other", rationale)
    assert ev2["all_match"] is True
    assert ev2["used_chunks_dir"] == "chunks"


def test_resolve_evidence_no_match(data_root: Path):
    ev = resolve_evidence(
        data_root, "chunks",
        [{"doc_id": "sample", "page": None, "text": "原文に存在しない捏造引用テキスト"}],
    )
    assert ev["all_match"] is False
    assert ev["rationales"][0]["matched"] is False


def test_api_round_trip(data_root: Path):
    from fastapi.testclient import TestClient
    from rageval.webapi.app import app

    app.state.data_root = data_root
    app.state.reviewer = "api-tester"
    client = TestClient(app)

    meta = client.get("/api/meta")
    assert meta.status_code == 200
    body = meta.json()
    assert body["defaults"]["reviewer"] == "api-tester"
    assert any(d["name"] == "raw" for d in body["sources"]["datasets"])
    assert "aspect_labels" in body["labels"]

    items = client.get("/api/items", params={"path": "raw", "track": "all"})
    assert items.status_code == 200
    assert len(items.json()["items"]) == 2

    bad = client.get("/api/items", params={"path": "no_such_dir", "track": "all"})
    assert bad.status_code == 400

    review = client.post("/api/review", json={
        "path": "raw",
        "source": "batch_test.jsonl",
        "qa_id": "qa_general",
        "status": "rejected",
        "reviewer": "api-tester",
    })
    assert review.status_code == 200
    assert review.json()["review_status"] == "rejected"

    ev = client.post("/api/evidence", json={
        "chunks_dir": "chunks",
        "rationale": [{"doc_id": "sample", "page": None, "text": RATIONALE_TEXT}],
    })
    assert ev.status_code == 200
    assert ev.json()["all_match"] is True


# ---------------- v3: 同名ファイル・引用照合・パス境界 ----------------

def test_review_mirrors_do_not_collide_across_datasets(data_root: Path):
    """別ディレクトリの同名ファイルのレビュー結果が混ざらないこと。

    data/raw_a/batch_x.jsonl と data/raw_b/batch_x.jsonl は
    このリポジトリでは普通に併存する (raw_X / filtered_X)。
    """
    for name, qid in (("raw_a", "qa_a"), ("raw_b", "qa_b")):
        d = data_root / name
        d.mkdir()
        (d / "batch_x.jsonl").write_text(_make_qa(qid).model_dump_json() + "\n", encoding="utf-8")

    save_review(data_root, "raw_a", "batch_x.jsonl", "qa_a",
                status="accepted", reviewer="t")

    # raw_b は raw_a のレビュー結果に汚染されない
    b = load_items(data_root, "raw_b", "all")
    assert [it["qa_id"] for it in b["items"]] == ["qa_b"]
    assert b["items"][0]["review_status"] == "pending"
    assert b["reviewed_override"] == []

    a = load_items(data_root, "raw_a", "all")
    assert a["items"][0]["review_status"] == "accepted"


def test_evidence_does_not_match_quote_from_other_doc(data_root: Path):
    """doc A に帰属する引用が doc B の本文にあっても一致にしない。"""
    other = data_root / "chunks"
    (other / "other.jsonl").write_text(
        json.dumps({
            "chunk_id": "other__c0", "doc_id": "other", "page": None,
            "text": CHUNK_TEXT, "position": 0, "section_path": [], "references": [],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # doc_id は "other" ではなく "sample" と名乗るが、sample のチャンクには無い文
    ev = resolve_evidence(data_root, "chunks", [
        {"doc_id": "sample", "page": None, "text": "周囲温度は 0〜40℃ の範囲で使用すること。"}
    ])
    # sample チャンクにこの文はある → 一致する (対照)
    assert ev["rationales"][0]["matched"] is True

    ev2 = resolve_evidence(data_root, "chunks", [
        {"doc_id": "no_such_doc", "page": None, "text": RATIONALE_TEXT}
    ])
    # 名乗った文書が存在しない → 他文書に同じ文があっても一致にしない
    assert ev2["rationales"][0]["matched"] is False


def test_resolve_paths_rejects_sibling_with_shared_prefix(tmp_path: Path):
    """data に対する兄弟ディレクトリ data_backup を接頭辞一致で通さないこと。"""
    root = tmp_path / "data"
    (root / "raw").mkdir(parents=True)
    sibling = tmp_path / "data_backup"
    sibling.mkdir()
    (sibling / "x.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ServiceError):
        resolve_paths(root, "../data_backup/x.jsonl")


# ---------------- 保存先の決定 (rank 3: 人手の成果を失わない) ----------------

def test_review_mirror_path_nests_by_dataset_and_never_re_nests(data_root: Path):
    """保存先はデータセットごとに掘り、reviewed/ 配下のファイルは掘り直さない。

    再入の判定を str(rel_dir).startswith("reviewed/") でやっていたため、区切りが
    バックスラッシュになる Windows では reviewed/raw/x.jsonl を開いたときに
    reviewed/reviewed/raw/x.jsonl へ書き出していた (=次に読むのは元の場所なので
    その判定が消える)。パス要素で判定する。
    """
    src = data_root / "raw" / "batch_test.jsonl"
    assert review_mirror_path(data_root, src) == data_root / "reviewed" / "raw" / "batch_test.jsonl"

    already = data_root / "reviewed" / "raw" / "batch_test.jsonl"
    already.parent.mkdir(parents=True)
    already.write_text("", encoding="utf-8")
    assert review_mirror_path(data_root, already) == already

    # 旧平置きは読み込みの土台としてだけ返る
    assert legacy_review_mirror_path(data_root, src) == data_root / "reviewed" / "batch_test.jsonl"

    # 保存先ディレクトリからデータ置き場を逆算できる (Streamlit 版の --out 用)
    assert data_root_for_reviewed_dir(Path("data/reviewed")) == Path("data")


def test_save_review_continues_from_legacy_flat_mirror(data_root: Path):
    """平たい旧配置のレビュー結果を土台にして、書き出しは入れ子へ移す。

    平たい配置に既存の判定が残っている状態で「入れ子へ移行」だけすると、過去の
    人手判定が丸ごと未判定に戻る。読み込みのみ後方互換とはそういう意味。
    """
    accepted = _make_qa("qa_general")
    accepted.review_status = "accepted"
    accepted.reviewed_by = "旧レイアウトの人"
    legacy = data_root / "reviewed" / "batch_test.jsonl"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        accepted.model_dump_json() + "\n"
        + _make_qa("qa_kg", kg=True).model_dump_json() + "\n",
        encoding="utf-8",
    )

    save_review(data_root, "raw", "batch_test.jsonl", "qa_kg",
                status="rejected", reviewer="tester")

    nested = data_root / "reviewed" / "raw" / "batch_test.jsonl"
    assert nested.exists()
    status = {
        json.loads(ln)["qa_id"]: json.loads(ln)["review_status"]
        for ln in nested.read_text(encoding="utf-8").splitlines()
    }
    assert status == {"qa_general": "accepted", "qa_kg": "rejected"}

    # 読み直しでも入れ子が優先される
    reloaded = load_items(data_root, "raw", "all")
    by_id = {it["qa_id"]: it for it in reloaded["items"]}
    assert by_id["qa_general"]["review_status"] == "accepted"
    assert by_id["qa_kg"]["review_status"] == "rejected"


def test_list_sources_shows_nested_reviewed_datasets(data_root: Path):
    """入れ子になったレビュー結果が候補一覧から消えないこと。"""
    save_review(data_root, "raw", "batch_test.jsonl", "qa_general",
                status="accepted", reviewer="tester")
    names = [d["name"] for d in list_sources(data_root)["datasets"]]
    assert "raw" in names
    assert "reviewed/raw" in names


# ---------------- ラベル・目標値の一本化 (rank 36/38/39) ----------------

def test_labels_are_derived_from_the_literals():
    """列挙の並びを手書きに戻したら落ちる。値が増えたら表示名も足させる。"""
    from typing import get_args

    from rageval.aspects import CategoryName
    from rageval.schema import (
        DifficultyLevel,
        DifficultyMatch,
        EvidenceStrictness,
        LeakageVerdict,
        OutputType,
        RAGAnswerMatch,
        ReviewStatus,
    )
    from rageval.webapi import labels

    pairs = [
        (CategoryName, labels.CATEGORY_ORDER, labels.CATEGORY_LABELS),
        (DifficultyLevel, labels.DIFF_ORDER, labels.DIFFICULTY_LABELS),
        (LeakageVerdict, labels.ALL_LEAKAGE, labels.LEAKAGE_LABELS),
        (DifficultyMatch, labels.ALL_DIFFICULTY_MATCH, labels.DIFFICULTY_MATCH_LABELS),
        (ReviewStatus, labels.ALL_REVIEW_STATUS, labels.REVIEW_STATUS_LABELS),
        (RAGAnswerMatch, labels.ALL_RAG_ANSWER_MATCH, labels.RAG_ANSWER_MATCH_LABELS),
        (OutputType, labels.ALL_OUTPUT_TYPE, labels.OUTPUT_TYPE_LABELS),
        (EvidenceStrictness, labels.ALL_EVIDENCE_STRICTNESS, labels.EVIDENCE_LABELS),
    ]
    for literal, order, label_map in pairs:
        assert tuple(get_args(literal)) == tuple(order)
        assert set(order) == set(label_map)


def test_diag_labels_cover_every_schema_flag():
    """診断3軸の項目を schema に足したとき、画面が黙って捨てないこと。"""
    from rageval.webapi import labels

    for axis, model in labels.DIAG_MODELS.items():
        flags = {
            name for name, field in model.model_fields.items()
            if field.annotation is bool
        }
        assert set(labels.DIAG_LABELS[axis]) == flags
        # 表示名の付け忘れ (英語のまま出ている項目) もここで気付ける
        assert all(v != k for k, v in labels.DIAG_LABELS[axis].items())


def test_targets_follow_the_filter_defaults():
    """画面の合格線は、実際に QA を落とす filter の既定と同じであること。"""
    from rageval.filter import (
        DEFAULT_ANSWERABILITY_MIN,
        DEFAULT_GROUNDING_MIN,
        DEFAULT_UNIQUENESS_MAX,
    )
    from rageval.webapi.labels import build_labels

    targets = build_labels()["targets"]
    assert targets["answerability_pass"] == DEFAULT_ANSWERABILITY_MIN
    assert targets["grounding_pass"] == DEFAULT_GROUNDING_MIN
    assert targets["uniqueness_pass"] == 1.0 - DEFAULT_UNIQUENESS_MAX
