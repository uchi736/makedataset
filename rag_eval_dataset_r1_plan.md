# RAG評価データセット 半自動構築 実装方針

> Claude Code 向け実装指示書。最小構成から始めて、必要になったら足す方針。

---

## 前提・方針

- **小さく始める**。Langfuse / MLflow / Argilla / DSPy は **最初は入れない**。後で必要になったら足す
- **Python + CLI + JSONファイル** の素朴な構成。DBもUIも最初は不要
- **Gold Seed 50問を人手で作る** → これが生成プロンプトのFew-Shotになり、フィルタ閾値の基準にもなる
- **MVPで end-to-end を通す**（10問でいい）→ 動くもの確認してから拡張

---

## 最小スタック

| 層 | ツール | 備考 |
|---|---|---|
| 言語 | Python 3.11+ | uv で管理 |
| 生成LLM | vLLM on H200 の gpt-oss-20B | 既存環境を叩く |
| 判定LLM | Claude API or gpt-oss-120B | 生成と別モデルにする |
| スキーマ | Pydantic v2 | JSON入出力の型付け |
| CLI | Typer | `python -m rageval generate` みたいに |
| 埋め込み | sentence-transformers (Ruri-v3-310m) | 重複除去用 |
| レビューUI | **Streamlit** (1画面) | Argillaは後で検討 |
| テスト | pytest | 各ステージの単体テスト |

**あとで入れるかもしれんもの（今は入れない）：**
- Langfuse（トレースが欲しくなったら）
- MLflow（データセットバージョン管理を厳密にしたくなったら）
- Argilla（レビュアーが複数人になったら）
- DSPy（プロンプトを自動最適化したくなったら）
- ペルソナ多様化（生成の質的多様性が足りないと感じたら）

---

## 評価フレームワーク（3階層）

データセット設計の根幹。neoAI方式と富士通方式をマージした3階層構造で観点を整理する。

### 階層① 能力評価軸：カテゴリ × 評価観点（neoAI主体 + 富士通補完）

「どんなRAG能力を測るか」。**生成時のカバレッジ指示**に使う。

| カテゴリ | 評価観点 |
|---|---|
| **Integration** | 複数情報源からの統合 / マルチドキュメント参照 / 遠隔参照 / **規格・規定番号の参照** |
| **Reasoning** | 数値計算 / マルチホップ / 否定推論 / 因果推論 / 時間推論 / 比較・条件判断 |
| **Logic** | 同義関係の解釈 / 数値包含関係の解釈 / 概念包含関係の解釈 / **語彙ミスマッチ（専門用語・略語）** / 抽象度の乖離 |
| **Figure** | 単純表 / 複雑帳票（セル結合・ヘッダ階層） / 概念図・構成図 / フローチャート・プロセス図 / グラフ・チャート / 複雑レイアウト / 大量列挙 |
| **Abstention** | 根拠不足 / 根拠の矛盾 / 不完全なチャンク区切り |

計 **25観点**。生成時は「観点 × 難易度」のマトリクスで配分指定する。

> **Note**: neoAI の Table カテゴリ（HTML/Markdown/CSV のフォーマット軸）は実装の話なので、観点としては採用しない。フォーマットは評価実験時の比較条件（同じQAを HTML vs Markdown で食わせる等）として扱う。観点はコンテンツ構造の複雑さ（概念図・帳票・グラフ等）で切る。

### 階層② 診断軸：富士通4軸の bool タグ多重付与

「システムのどこが難しいか」。**評価結果の分析**に使う（例: vocabulary-mismatch=true のQAだけで精度集計）。

| 軸 | タグ |
|---|---|
| **Reasoning Complexity** | multi-step / quantitative / negation / cause-effect / comparison / temporal / output-type(summary/trans/list) |
| **Retrieval Difficulty** | multi-doc / multi-chunk / low-locality / remote-reference / doc-volume / chunk-size / abstraction-discrepancy / **vocabulary-mismatch** |
| **Source Structure** | tables-charts / complex-layout / specific-area-ref / logical-nesting / large-enumeration / redundancy |
| **Explainability** | evidence-strictness (no-evidence/hier-ref/coord-ref/multi-ref) |

階層①の観点と冗長な部分はあるが、用途が違う：
- **観点** = 生成時のカバレッジ指示
- **診断タグ** = 評価結果の分析

### 階層③ 難易度ラベル

- **retrieval_level**: Easy / Medium / Hard
- **answer_level**: Easy / Medium / Hard

定量基準を明示してアノテーター揺れを抑える。基準は以下：

| 軸 | Easy | Medium | Hard |
|---|---|---|---|
| **検索難易度** | 1チャンクで答え完結、キーワード一致で見つかる | 2-3チャンク必要、または言い換えで検索が必要 | 3チャンク以上、または専門用語・略語の解釈が必要 |
| **回答難易度** | 抜き出し型、推論不要 | 1-2ステップの推論、または比較・統合 | 3ステップ以上の推論、条件分岐、計算を含む |

`difficulty_rationale` フィールドに **「必要チャンク=3, 推論ステップ=2, 条件分岐あり」** のように定量理由を書く。

---

## バランス目標

- **規模**: パイロット30-50 → 標準100 → 拡張200+
- **検索難易度**: Easy 35-40% / Medium 35-40% / Hard 20-25%
- **回答難易度**: Easy 15-20% / Medium 60-65% / Hard 15-20%
- **各観点は最低3問**ずつカバー（25観点 × 3 = 最低75問のフロア）
- **Abstention**: 全体の10-15%

---

## ディレクトリ構成

```
rag_eval_dataset/
├── pyproject.toml
├── README.md
├── .env.example              # API_KEY等
├── src/rageval/
│   ├── __init__.py
│   ├── schema.py             # Pydantic モデル
│   ├── aspects.py            # 25観点の定義（Literal列挙）+ 備考・業務シナリオ
│   ├── generate.py           # Stage 1: 自動生成
│   ├── filter.py             # Stage 2: 自動フィルタ
│   ├── review_app.py         # Stage 3: Streamlitレビュー
│   ├── chunker.py            # ドキュメント→アンカーチャンク
│   ├── llm.py                # vLLM / Claude 呼び出しラッパ
│   └── cli.py                # Typerエントリポイント
├── prompts/
│   ├── generate.md           # 生成プロンプト本体
│   └── judge.md              # 判定プロンプト本体
├── data/
│   ├── docs/                 # 元ドキュメント (gitignore)
│   ├── chunks/               # チャンク化済み (gitignore)
│   ├── seeds/                # Gold Seed 50問 (人手, コミットする)
│   ├── raw/                  # 生成直後 (gitignore)
│   ├── filtered/             # フィルタ後 (gitignore)
│   └── reviewed/             # レビュー済み最終版 (コミットする)
└── tests/
    ├── test_schema.py
    ├── test_generate.py
    └── test_filter.py
```

---

## データスキーマ (`src/rageval/schema.py`)

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime

# ===== 根拠 =====
class Rationale(BaseModel):
    doc_id: str
    page: Optional[int] = None
    text: str

# ===== 階層② 診断軸（富士通4軸） =====
class ReasoningComplexity(BaseModel):
    multi_step: bool = False
    quantitative: bool = False
    negation: bool = False
    cause_effect: bool = False
    comparison: bool = False
    temporal: bool = False
    output_type: Literal["summary", "trans", "list", "none"] = "none"

class RetrievalDifficulty(BaseModel):
    multi_doc: bool = False
    multi_chunk: bool = False
    low_locality: bool = False
    remote_reference: bool = False
    doc_volume_large: bool = False        # >=1000p
    chunk_size_large: bool = False        # >=512tok
    abstraction_discrepancy: bool = False
    vocabulary_mismatch: bool = False     # 専門用語・略語

class SourceStructure(BaseModel):
    tables_charts: bool = False
    complex_layout: bool = False
    specific_area_ref: bool = False
    logical_nesting: bool = False
    large_enumeration: bool = False
    redundancy: bool = False

class Explainability(BaseModel):
    evidence_strictness: Literal["no-evidence", "hier-ref", "coord-ref", "multi-ref"]

# ===== 運用メタ =====
class GenerationInfo(BaseModel):
    model: str
    prompt_version: str
    generated_at: datetime

class FilterScores(BaseModel):
    answerability: Optional[float] = None       # 1-5
    leakage: Optional[Literal["pass", "fail"]] = None
    grounding: Optional[float] = None           # 1-5
    uniqueness: Optional[float] = None          # cos_sim (低いほど独自)
    difficulty_match: Optional[Literal["aligned", "too_easy", "too_hard"]] = None

# ===== QA本体 =====
class QAItem(BaseModel):
    qa_id: str
    question: str
    answer: str
    rationale: list[Rationale]

    # 階層① 能力評価
    category: list[Literal["Integration", "Reasoning", "Logic", "Figure", "Abstention"]]
    aspect: list[str]   # aspects.py の ASPECTS から選ぶ

    # 階層② 診断軸
    reasoning_complexity: ReasoningComplexity
    retrieval_difficulty: RetrievalDifficulty
    source_structure: SourceStructure
    explainability: Explainability

    # 階層③ 難易度
    retrieval_level: Literal["Easy", "Medium", "Hard"]
    answer_level: Literal["Easy", "Medium", "Hard"]
    difficulty_rationale: str   # "必要チャンク=3, 推論ステップ=2" など定量理由

    # 運用
    business_scenario: str   # aspects.py の BUSINESS_SCENARIOS 参照（free text）

    generation: GenerationInfo
    filter_scores: FilterScores = Field(default_factory=lambda: FilterScores())
    review_status: Literal["pending", "accepted", "edited", "rejected"] = "pending"
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
```

---

## 観点・備考・業務シナリオ定義 (`src/rageval/aspects.py`)

```python
from typing import Literal

# 階層① の25観点を Literal で固定
ASPECTS = Literal[
    # Integration
    "multi_source_integration", "multi_doc_reference", "remote_reference",
    "standards_reference",
    # Reasoning
    "quantitative_calc", "multi_hop", "negation", "causal", "temporal", "comparison_conditional",
    # Logic
    "synonym_interpretation", "numeric_inclusion", "concept_inclusion",
    "vocabulary_mismatch", "abstraction_gap",
    # Figure (コンテンツ構造ベース)
    "simple_table", "complex_form", "concept_diagram", "flowchart",
    "chart_graph", "complex_layout", "large_enumeration",
    # Abstention
    "insufficient_evidence", "contradictory_evidence", "fragmented_chunk",
]

# カテゴリへの逆引きマップ
ASPECT_TO_CATEGORY = {
    "multi_source_integration": "Integration",
    "multi_doc_reference": "Integration",
    "remote_reference": "Integration",
    "standards_reference": "Integration",
    "quantitative_calc": "Reasoning",
    "multi_hop": "Reasoning",
    "negation": "Reasoning",
    "causal": "Reasoning",
    "temporal": "Reasoning",
    "comparison_conditional": "Reasoning",
    "synonym_interpretation": "Logic",
    "numeric_inclusion": "Logic",
    "concept_inclusion": "Logic",
    "vocabulary_mismatch": "Logic",
    "abstraction_gap": "Logic",
    "simple_table": "Figure",
    "complex_form": "Figure",
    "concept_diagram": "Figure",
    "flowchart": "Figure",
    "chart_graph": "Figure",
    "complex_layout": "Figure",
    "large_enumeration": "Figure",
    "insufficient_evidence": "Abstention",
    "contradictory_evidence": "Abstention",
    "fragmented_chunk": "Abstention",
}

# 観点ごとの抽象的説明（汎用）
ASPECT_DESCRIPTIONS = {
    "multi_source_integration": "複数の情報源（文書・章節）の内容を統合して回答が必要",
    "multi_doc_reference": "2つ以上の異なる文書を横断参照する必要がある",
    "remote_reference": "同一文書内でも章をまたぐなど離れた場所を参照する必要がある",
    "standards_reference": "規格・規定の番号による参照を解決する必要がある",
    "quantitative_calc": "数値計算（合計・差分・比率など）が必要",
    "multi_hop": "中間結論を経由する多段推論が必要",
    "negation": "「〜でない」等の否定条件を解釈する必要がある",
    "causal": "原因・結果・理由を導出する必要がある",
    "temporal": "時系列・順序関係を解釈する必要がある",
    "comparison_conditional": "比較や条件分岐を含む推論が必要",
    "synonym_interpretation": "同義語・言い換え表現の解釈が必要",
    "numeric_inclusion": "数値の包含関係（範囲・以上以下）を解釈する必要がある",
    "concept_inclusion": "概念の包含・上位下位関係を解釈する必要がある",
    "vocabulary_mismatch": "質問と文書で表現が異なる(専門用語・略語・社内用語のずれ)",
    "abstraction_gap": "抽象的な質問 vs 具体的な記述、またはその逆",
    "simple_table": "単純な行列の表(読み下しで理解可能)",
    "complex_form": "セル結合・ヘッダ階層・項目グループのある複雑帳票",
    "concept_diagram": "概念図・構成図・ブロック図など関係性を表現する図",
    "flowchart": "フローチャート・プロセス図・状態遷移図など手順を表現する図",
    "chart_graph": "折れ線・棒・円グラフなど数値を可視化したチャート",
    "complex_layout": "段組み・図表混在・サイドノートなど複雑なページレイアウト",
    "large_enumeration": "長いリスト・大量の項目列挙",
    "insufficient_evidence": "文書内に答えとなる情報が存在しない(回答拒否すべき)",
    "contradictory_evidence": "複数箇所の情報が矛盾している(回答拒否すべき)",
    "fragmented_chunk": "チャンク分割の都合で答えが分断されている",
}

# 観点ごとの現場での具体例（備考、プロンプトのヒント用）
ASPECT_EXAMPLES = {
    "multi_source_integration": "複数の規定・仕様書からの条件統合",
    "multi_doc_reference": "仕様書↔規格書、上位規定↔下位規定、親規定↔子規定の参照",
    "remote_reference": "附則・別表の参照、改訂履歴と本文の対応",
    "standards_reference": "JIS/ISO/ASME 等の規格番号参照、社内規定番号の参照",
    "quantitative_calc": "公差判定、許容値計算、単位変換(mm↔cm)、寸法合計",
    "multi_hop": "規定A→規定B→規定Cと辿って答えに到達",
    "negation": "「適用しない」「除外する」条項の解釈",
    "causal": "不具合原因の特定、設計変更の理由、品質問題の根本原因",
    "temporal": "手順の順序、改訂前後の差分、製造工程の前後関係",
    "comparison_conditional": "規定値との照合、適用条件判定(事業部別/製品別)、仕様比較",
    "synonym_interpretation": "JIS規格名と通称、製品名と型番、部品の旧称と新称",
    "numeric_inclusion": "「規定値±5%以内」「100以上」等の範囲判定",
    "concept_inclusion": "全社規定→事業部規定→部門規定の階層、製品ファミリと型番",
    "vocabulary_mismatch": "略語(QMS/P&ID/FMEA)、部署用語、業界用語と一般用語の差",
    "abstraction_gap": "「品質を確保する方法は」のような抽象質問に対する具体手順の記述",
    "simple_table": "部品リスト、仕様一覧、規定値一覧(単純な行列)",
    "complex_form": "ヘッダ階層のある規定値表、セル結合のある帳票",
    "concept_diagram": "システム構成図、機器ブロック図、組織図",
    "flowchart": "QMSプロセス図、品質管理手順、設計レビューフロー",
    "chart_graph": "試験結果のグラフ、トレンドチャート、不具合発生件数の推移",
    "complex_layout": "図面と注記の混在、段組み仕様書",
    "large_enumeration": "長い部品リスト、多数の項目を含む規定値表",
    "insufficient_evidence": "文書外の情報を聞く質問(コーパスに答えがない)",
    "contradictory_evidence": "旧版規定と新版規定の混在、矛盾する記述",
    "fragmented_chunk": "ページまたぎ・章またぎで分断された記述",
}

# 業務シナリオの例（business_scenario フィールドの参考値）
BUSINESS_SCENARIOS = [
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
]
```

> **`ASPECT_EXAMPLES` の使い方**: スキーマには載らない補足情報。生成プロンプトに「この観点の現場例: 〜」として注入したり、レビューUIで観点選択時のツールチップに表示する。**現場感の維持**と**抽象的な定義の純度**を両立するための分離。

---

## Phase 0: Gold Seed 50問（人手作成）

**Claude Code にやらせる作業ではない。人手でやる。** ただしテンプレートJSON を用意しておく。

- `data/seeds/seeds.json` に人手で50問書く
- 最初は10問だけでもいい。徐々に追加
- **これが生成プロンプトの Few-Shot 例になる** から、25観点をバランスよくカバーするのを意識して書く
- 専門用語・略語を含むQA（vocabulary_mismatch）は必ず数問は入れる
- 概念図・フローチャートのQAも数問は入れる（設計基準書・QMS資料の典型）
- 規格番号参照のQA（standards_reference）も入れる

---

## Phase 1: MVP（end-to-end を10問で通す）

### ゴール
「ドキュメント投入 → 10問生成 → フィルタ → レビューUIで確認」が動く。

### タスク

1. **`schema.py`** + **`aspects.py`** を上記で実装
2. **`llm.py`** - vLLM と Claude API の呼び出しラッパ（OpenAI互換でよい）
    - 環境変数: `VLLM_BASE_URL`, `ANTHROPIC_API_KEY`
    - `generate(prompt, model, response_format)` のシンプルなI/F
    - JSON mode対応（Pydanticスキーマを渡したらバリデート済みオブジェクトを返す）
3. **`chunker.py`** - ドキュメント → チャンク
    - まずは単純に「文字数で分割」でいい。LangChain の `RecursiveCharacterTextSplitter` でも可
    - 出力: `data/chunks/*.jsonl`（1行1チャンク、`chunk_id / doc_id / page / text`）
4. **`prompts/generate.md`** - 生成プロンプト
    - Few-Shot に Gold Seed から3-5問を埋め込む
    - **観点 × 難易度マトリクスで配分指示**（カテゴリではなく観点を直接指定）
    - 観点ごとの **`ASPECT_DESCRIPTIONS` + `ASPECT_EXAMPLES`** をプロンプトに注入
    - JSON出力形式をスキーマに沿って厳密指定（診断軸4ブロックも全て出させる）
    - `business_scenario` は `BUSINESS_SCENARIOS` から選択 or 自由文字列
5. **`generate.py`** - Stage 1
    - 入力: `data/chunks/*.jsonl`, 生成数, 観点ごとの目標数
    - アンカーチャンクを層別サンプリング
    - プロンプト組み立て → LLM → JSON パース → `QAItem` バリデート
    - 出力: `data/raw/batch_YYYYMMDD_HHMM.jsonl`
6. **`prompts/judge.md`** - 判定プロンプト
    - 6観点それぞれを **別プロンプト** で判定（一気にやらせると精度下がる）
    - 各プロンプトはスコアと根拠を JSON で返す
7. **`filter.py`** - Stage 2
    - 入力: raw jsonl
    - 各QAに対し6観点をJudge LLMで評価
        - Answerability / Leakage / Grounding / Uniqueness / 難易度整合 / 根拠完全性
    - 閾値で自動棄却（Answerability < 4, Leakage = fail, Grounding < 4）
    - 埋め込みで重複除去（cos_sim > 0.92）
    - 出力: `data/filtered/batch_YYYYMMDD_HHMM.jsonl`（`filter_scores` 付き）
8. **`review_app.py`** - Stage 3 Streamlit
    - filtered jsonl を読む
    - 1問ずつ表示: 質問 / 回答 / 根拠 / カテゴリ / 観点（説明＋備考をツールチップ）/ 診断軸タグ / 難易度 / フィルタスコア
    - Accept / Edit / Reject の3ボタン
    - 結果を `data/reviewed/batch_YYYYMMDD_HHMM.jsonl` に書き出す
    - 起動: `streamlit run src/rageval/review_app.py`
9. **`cli.py`** - Typer
    ```bash
    rageval chunk --docs data/docs/ --out data/chunks/
    rageval generate --chunks data/chunks/ --n 10 --out data/raw/
    rageval filter --in data/raw/batch_*.jsonl --out data/filtered/
    rageval review --in data/filtered/batch_*.jsonl
    ```

### 完了条件
- 10問生成 → フィルタ → レビューUI表示まで動く
- バリデーションエラーが出ない
- 各ステージに最低1つの pytest がある
- **25観点のうち最低5観点は出現している**（カバレッジ抜き打ちチェック）

---

## Phase 2: 実用化（100問バッチ）

Phase 1 が動いてから着手。

- **観点 × 難易度マトリクスでの配分制御の厳密化**（足りないセルを補う再生成ループ）
- **25観点フルカバレッジ**を目標（各観点 最低3問）
- Evolution方式の追加（Seed → Multi-hop → Reasoning → Conditional の書き換えプロンプト）
- 難易度検証（宣言難易度 vs gpt-oss-20B実測正答率の乖離をフラグ）
- 診断軸タグの自動付与精度の確認（人手レビューでの修正率を監視）
- レビューUI強化（キーボードショートカット、スコア順ソート、観点フィルタ）

---

## Phase 3: スケール（必要になったら）

以下は必要性が出てから追加する。最初から入れない。

- **Langfuse**: プロンプトの差分で品質がどう変わるか追跡したくなったら
- **MLflow**: データセットのバージョン管理を厳密にしたくなったら
- **Argilla**: レビュアーが複数人になって協業が必要になったら
- **DSPy**: プロンプトを自動最適化したくなったら
- **ColQwen / PP-StructureV3**: 図表ヘビーな文書を扱うようになったら
- **フォーマット軸の評価**: 同一QAを HTML/Markdown/CSV で食わせて比較する実験
- **ペルソナ多様化**: 設計者/品証/現場オペ/新人 など視点別の問い方バリエーション

---

## 実装順序（Claude Code向け）

1. `pyproject.toml` + `src/rageval/__init__.py` 雛形
2. `aspects.py` で25観点 + 備考 + 業務シナリオを定義
3. `schema.py` + `tests/test_schema.py`
4. `llm.py`（vLLM と Claude の両対応、モックでテスト）
5. `chunker.py`（単純分割でよい）
6. `data/seeds/seeds.json` にサンプル5問だけ先に入れる（後で人手で拡張）
7. `prompts/generate.md` 起草
8. `generate.py` + `tests/test_generate.py`（モックLLMでテスト）
9. `prompts/judge.md` 起草（6観点分）
10. `filter.py` + `tests/test_filter.py`
11. `review_app.py`（Streamlit, シンプル1画面）
12. `cli.py` で全部つなぐ
13. README.md に使い方書く

各ステップで `uv run pytest` で緑にしてから次へ。

---

## 守ること

- **プロンプトは `prompts/*.md` にファイルで持つ**（Pythonにハードコードしない）
- **生成プロンプトにバージョンを埋める**（`prompt_version: "v1.0"` みたいにフロントマターで管理）
- **JSON Schema validation を必ず通す**（パースエラーで落ちない工夫。再生成 or スキップ）
- **LLM呼び出しは再試行・指数バックオフ**（tenacity使う）
- **ログは標準出力にJSONLで出す**（後から Langfuse 入れても移行しやすい形にしておく）
- **秘密情報は env から**（ドキュメントも含む。公開用テストだけリポジトリに入れる）
- **観点 (aspect) は必ず Literal で固定**（自由文字列にしない。タイポと表記揺れを潰す）
- **`ASPECT_DESCRIPTIONS` は抽象的に保つ**（具体例は `ASPECT_EXAMPLES` に分離）

---

## 参考（過去議論から）

- **富士通方式（Fujitsu RAG Hard Benchmark / AAAI 2026）**: 4軸診断メタデータ（Reasoning Complexity / Retrieval Difficulty / Source Structure / Explainability）、rationale (page+bbox)、Easy/Medium/Hard
- **neoAI方式（J-RAGBench）**: 5カテゴリ × 詳細観点の2階層、Abstention の3観点分解
- **RAGAS**: Evolution方式（Easy → Multi-hop → Reasoning → Conditional）

## 設計判断メモ

- **Table → Figure 改名**: neoAI の Table 観点は HTML/Markdown/CSV のフォーマット軸だったが、実装の話なので採用せず。コンテンツ構造軸（単純表/複雑帳票/概念図/フローチャート/グラフ/複雑レイアウト/大量列挙）で切り直し。フォーマットの違いは Phase 3 で評価実験条件として扱う。
- **question_type 不採用**: 富士通の Yes/No/Factoid/Definition は表面形式の分類で、カテゴリ×観点と診断軸タグに既に情報が含まれているので冗長と判断。
- **`standards_reference` 観点追加**: 製造業文書では「JIS Z 2241 に定める…」のような規格・規定番号参照が頻出。`multi_doc_reference` でも扱えるが、検索特性が独特（番号一致が決定的）なので独立観点とした。
- **`ASPECT_EXAMPLES` を分離**: 観点の抽象的定義（`ASPECT_DESCRIPTIONS`）と現場の具体例（`ASPECT_EXAMPLES`）を別フィールドにすることで、定義の純度を保ちつつ現場感を維持。プロンプト・レビューUI で適宜参照する。
- **ペルソナ不採用（Phase 1 では）**: 設計者/品証/現場オペ/新人 のペルソナ指定は生成時の手間に対して MVP では効果が薄いと判断。観点 × 難易度の配分制御で多様性は十分担保できる見込み。生成結果の質的多様性が足りないと感じたら Phase 3 で再導入。