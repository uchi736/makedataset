"""階層① 能力評価軸: 25観点を Literal で固定。

aspect は QAItem.aspect: list[str] に入れる値。表記揺れを防ぐため Literal で
列挙し、ASPECT_TO_CATEGORY でカテゴリへの逆引き、ASPECT_DESCRIPTIONS で抽象
定義、ASPECT_EXAMPLES で現場の具体例(プロンプト/UIヒント用)を提供する。

作問例 (ASPECT_GOOD_EXAMPLES / ASPECT_BAD_PATTERNS) だけは prompts/aspect_examples.md
が正本 (2026-08-08 分離)。あれは観点の定義ではなく生成プロンプトの本文で、コードに
置いていたために「作問の誘導を直すのに .py を触る・直しても prompt_version が動かず
バッチを区別できない」状態だった。観点の ID・ラベル・定義はコード側に残す。
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, get_args

from rageval.prompts import load_prompt


ASPECTS = Literal[
    # Integration
    "multi_source_integration",
    "multi_doc_reference",
    "remote_reference",
    "standards_reference",
    # Reasoning
    "quantitative_calc",
    "multi_hop",
    "negation",
    "causal",
    "temporal",
    "comparison_conditional",
    # Logic
    "synonym_interpretation",
    "numeric_inclusion",
    "concept_inclusion",
    "vocabulary_mismatch",
    "abstraction_gap",
    # Figure (コンテンツ構造ベース)
    "simple_table",
    "complex_form",
    "concept_diagram",
    "flowchart",
    "chart_graph",
    "complex_layout",
    "large_enumeration",
    # Abstention
    "insufficient_evidence",
    "contradictory_evidence",
    "fragmented_chunk",
]

ALL_ASPECTS: tuple[str, ...] = tuple(get_args(ASPECTS))


CategoryName = Literal["Integration", "Reasoning", "Logic", "Figure", "Abstention"]


ASPECT_TO_CATEGORY: dict[str, CategoryName] = {
    "multi_source_integration": "Integration",
    "multi_doc_reference":      "Integration",
    "remote_reference":         "Integration",
    "standards_reference":      "Integration",

    "quantitative_calc":        "Reasoning",
    "multi_hop":                "Reasoning",
    "negation":                 "Reasoning",
    "causal":                   "Reasoning",
    "temporal":                 "Reasoning",
    "comparison_conditional":   "Reasoning",

    "synonym_interpretation":   "Logic",
    "numeric_inclusion":        "Logic",
    "concept_inclusion":        "Logic",
    "vocabulary_mismatch":      "Logic",
    "abstraction_gap":          "Logic",

    "simple_table":             "Figure",
    "complex_form":             "Figure",
    "concept_diagram":          "Figure",
    "flowchart":                "Figure",
    "chart_graph":              "Figure",
    "complex_layout":           "Figure",
    "large_enumeration":        "Figure",

    "insufficient_evidence":    "Abstention",
    "contradictory_evidence":   "Abstention",
    "fragmented_chunk":         "Abstention",
}


# 観点ごとの抽象定義(汎用)
ASPECT_DESCRIPTIONS: dict[str, str] = {
    "multi_source_integration": "複数の情報源(文書・章節)の内容を統合して回答が必要",
    "multi_doc_reference":      "2つ以上の異なる文書を横断参照する必要がある",
    "remote_reference":         "同一文書内でも章をまたぐなど離れた場所を参照する必要がある",
    "standards_reference":      "規格・規定の番号による参照を解決する必要がある",
    "quantitative_calc":        "数値計算(合計・差分・比率など)が必要",
    "multi_hop":                "中間結論を経由する多段推論が必要",
    "negation":                 "「〜でない」等の否定条件を解釈する必要がある",
    "causal":                   "原因・結果・理由を導出する必要がある",
    "temporal":                 "時系列・順序関係を解釈する必要がある",
    "comparison_conditional":   "比較や条件分岐を含む推論が必要",
    "synonym_interpretation":   "同義語・言い換え表現の解釈が必要",
    "numeric_inclusion":        "数値の包含関係(範囲・以上以下)を解釈する必要がある",
    "concept_inclusion":        "概念の包含・上位下位関係を解釈する必要がある",
    "vocabulary_mismatch":      "質問と文書で表現が異なる(専門用語・略語・社内用語のずれ)",
    "abstraction_gap":          "抽象的な質問 vs 具体的な記述、またはその逆",
    "simple_table":             "単純な行列の表(読み下しで理解可能)",
    "complex_form":             "セル結合・ヘッダ階層・項目グループのある複雑帳票",
    "concept_diagram":          "概念図・構成図・ブロック図など関係性を表現する図",
    "flowchart":                "フローチャート・プロセス図・状態遷移図など手順を表現する図",
    "chart_graph":              "折れ線・棒・円グラフなど数値を可視化したチャート",
    "complex_layout":           "段組み・図表混在・サイドノートなど複雑なページレイアウト",
    "large_enumeration":        "長いリスト・大量の項目列挙",
    "insufficient_evidence":    "文書内に答えとなる情報が存在しない(回答拒否すべき)",
    "contradictory_evidence":   "複数箇所の情報が矛盾している(回答拒否すべき)",
    "fragmented_chunk":         "チャンク分割の都合で答えが分断されている",
}


# 観点ごとの現場での具体例(備考、プロンプトのヒント用)
ASPECT_EXAMPLES: dict[str, str] = {
    "multi_source_integration": "複数の規定・仕様書からの条件統合",
    "multi_doc_reference":      "仕様書↔規格書、上位規定↔下位規定、親規定↔子規定の参照",
    "remote_reference":         "附則・別表の参照、改訂履歴と本文の対応",
    "standards_reference":      "JIS/ISO/ASME 等の規格番号参照、社内規定番号の参照",
    "quantitative_calc":        "公差判定、許容値計算、単位変換(mm↔cm)、寸法合計",
    "multi_hop":                "規定A→規定B→規定Cと辿って答えに到達",
    "negation":                 "「適用しない」「除外する」条項の解釈",
    "causal":                   "不具合原因の特定、設計変更の理由、品質問題の根本原因",
    "temporal":                 "手順の順序、改訂前後の差分、製造工程の前後関係",
    "comparison_conditional":   "規定値との照合、適用条件判定(事業部別/製品別)、仕様比較",
    "synonym_interpretation":   "JIS規格名と通称、製品名と型番、部品の旧称と新称",
    "numeric_inclusion":        "「規定値±5%以内」「100以上」等の範囲判定",
    "concept_inclusion":        "全社規定→事業部規定→部門規定の階層、製品ファミリと型番",
    "vocabulary_mismatch":      "略語(QMS/P&ID/FMEA)、部署用語、業界用語と一般用語の差",
    "abstraction_gap":          "「品質を確保する方法は」のような抽象質問に対する具体手順の記述",
    "simple_table":             "部品リスト、仕様一覧、規定値一覧(単純な行列)",
    "complex_form":             "ヘッダ階層のある規定値表、セル結合のある帳票",
    "concept_diagram":          "システム構成図、機器ブロック図、組織図",
    "flowchart":                "QMSプロセス図、品質管理手順、設計レビューフロー",
    "chart_graph":              "試験結果のグラフ、トレンドチャート、不具合発生件数の推移",
    "complex_layout":           "図面と注記の混在、段組み仕様書",
    "large_enumeration":        "長い部品リスト、多数の項目を含む規定値表",
    "insufficient_evidence":    "文書外の情報を聞く質問(コーパスに答えがない)",
    "contradictory_evidence":   "旧版規定と新版規定の混在、矛盾する記述",
    "fragmented_chunk":         "ページまたぎ・章またぎで分断された記述",
}


# 作問例 (✓ 良い問いの例 / ✗ 避けるべきパターン) の本文は prompts/aspect_examples.md。
# ここに直書きしていた例は 2026-07-28 に退役したコーパス (就業規則・労務) の語彙で、
# 対象が法令・規格とプラント技報へ移った後も残って作問を引っ張っていた。本文を
# prompts 側へ出したので、以後は例文の差し替えでコードを触らない。
ASPECT_EXAMPLES_PROMPT = "prompts/aspect_examples.md"

# 既存の load_prompt は実行時の作業場所からの相対で読むが、作問例は UI・配信からも
# 引かれ得るので、置き場に依存しない __file__ 起点を先に試す。
_REPO_ROOT = Path(__file__).resolve().parents[2]


class AspectExamples(NamedTuple):
    """prompts/aspect_examples.md の中身 (観点ID → 箇条書き) と、その版数。"""

    good: Mapping[str, list[str]]
    bad: Mapping[str, list[str]]
    version: str


def _resolve_examples_path() -> Path:
    candidates = [_REPO_ROOT / ASPECT_EXAMPLES_PROMPT, Path(ASPECT_EXAMPLES_PROMPT)]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "観点の作問例が見つからない (探した場所: "
        + " / ".join(str(p) for p in candidates)
        + ")"
    )


def _parse_examples_body(body: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """`## 観点ID` → `### good` / `### bad` → `- 箇条書き` の3階層で読む。

    見出しの外にある説明文の箇条書きは、観点が確定していないので取り込まない。
    """
    good: dict[str, list[str]] = {}
    bad: dict[str, list[str]] = {}
    aspect: str | None = None
    bucket: list[str] | None = None
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            aspect = line[3:].strip()
            good.setdefault(aspect, [])
            bad.setdefault(aspect, [])
            bucket = None
        elif line.startswith("### "):
            kind = line[4:].strip().lower()
            if aspect is None:
                bucket = None
            elif kind.startswith("good"):
                bucket = good[aspect]
            elif kind.startswith("bad"):
                bucket = bad[aspect]
            else:
                bucket = None
        elif line.startswith("- ") and bucket is not None:
            bucket.append(line[2:].strip())
    return good, bad


def _validate_examples(
    good: dict[str, list[str]], bad: dict[str, list[str]]
) -> None:
    """観点の増減や綴り違いを黙って素通りさせない。

    片側だけ直すと「その観点だけ作問誘導が空のまま生成される」ため、
    欠落・未知の観点はここで落として知らせる (fail-closed)。
    """
    unknown = sorted(set(good) - set(ALL_ASPECTS))
    if unknown:
        raise ValueError(
            f"{ASPECT_EXAMPLES_PROMPT}: aspects.py に無い観点が書かれている {unknown}"
        )
    missing = [a for a in ALL_ASPECTS if not good.get(a) or not bad.get(a)]
    if missing:
        raise ValueError(
            f"{ASPECT_EXAMPLES_PROMPT}: 良い例/悪い例が揃っていない観点 {missing}"
        )


@lru_cache(maxsize=1)
def load_aspect_examples() -> AspectExamples:
    path = _resolve_examples_path()
    meta, body = load_prompt(str(path))
    good, bad = _parse_examples_body(body)
    _validate_examples(good, bad)
    return AspectExamples(
        good=MappingProxyType(good),
        bad=MappingProxyType(bad),
        version=meta.get("prompt_version", ""),
    )


def aspect_examples_version() -> str:
    """作問例の版数。生成バッチを版で区別できるよう frontmatter から引く。"""
    return load_aspect_examples().version


def __getattr__(name: str) -> Any:
    """旧来の定数名での参照を保つ (本文の読み込みは初回参照時)。"""
    if name == "ASPECT_GOOD_EXAMPLES":
        return load_aspect_examples().good
    if name == "ASPECT_BAD_PATTERNS":
        return load_aspect_examples().bad
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Display labels (日本語、グラフやレビューUI用)
ASPECT_LABELS: dict[str, str] = {
    "multi_source_integration": "複数情報源の統合",
    "multi_doc_reference":      "マルチドキュメント参照",
    "remote_reference":         "遠隔参照(章をまたぐ)",
    "standards_reference":      "規格・規定番号参照",
    "quantitative_calc":        "数値計算",
    "multi_hop":                "マルチホップ",
    "negation":                 "否定推論",
    "causal":                   "因果推論",
    "temporal":                 "時間推論",
    "comparison_conditional":   "比較・条件判断",
    "synonym_interpretation":   "同義関係の解釈",
    "numeric_inclusion":        "数値包含関係",
    "concept_inclusion":        "概念包含関係",
    "vocabulary_mismatch":      "語彙ミスマッチ(専門用語・略語)",
    "abstraction_gap":          "抽象度の乖離",
    "simple_table":             "単純表",
    "complex_form":             "複雑帳票",
    "concept_diagram":          "概念図・構成図",
    "flowchart":                "フローチャート",
    "chart_graph":              "グラフ・チャート",
    "complex_layout":           "複雑レイアウト",
    "large_enumeration":        "大量列挙",
    "insufficient_evidence":    "根拠不足",
    "contradictory_evidence":   "根拠の矛盾",
    "fragmented_chunk":         "不完全なチャンク区切り",
}


# 業務シナリオ参考値 (QAItem.business_scenario の選択肢ヒント)
BUSINESS_SCENARIOS: tuple[str, ...] = (
    "設計変更影響評価",
    "監査対応・規定遵守確認",
    "不具合原因調査",
    "安全基準レビュー",
    "新規部品・材料選定",
    "工程設計レビュー",
    "教育・OJT",
    "顧客問合せ対応",
    "規格適合性確認",
    "見積・調達仕様書作成",
)


def aspects_for_category(category: CategoryName) -> list[str]:
    return [a for a, c in ASPECT_TO_CATEGORY.items() if c == category]
