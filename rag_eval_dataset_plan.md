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

class Rationale(BaseModel):
    doc_id: str
    page: Optional[int] = None
    text: str                              # 根拠テキストそのもの

class Diagnostic(BaseModel):
    retrieval_difficulty: Literal["Easy", "Medium", "Hard"]
    reasoning_difficulty: Literal["Easy", "Medium", "Hard"]
    doc_structure: Literal["text", "table", "figure", "text+table", "layout"]
    explainability: Literal["single_source", "multi_source"]

class Difficulty(BaseModel):
    search: Literal["Easy", "Medium", "Hard"]
    answer: Literal["Easy", "Medium", "Hard"]
    rationale: str                          # "必要チャンク=3, 推論ステップ=2" のような定量理由

class GenerationInfo(BaseModel):
    model: str
    prompt_version: str
    generated_at: datetime

class FilterScores(BaseModel):
    answerability: Optional[float] = None   # 1-5
    leakage: Optional[Literal["pass", "fail"]] = None
    grounding: Optional[float] = None       # 1-5
    uniqueness: Optional[float] = None      # cos_sim (低いほど独自)
    difficulty_match: Optional[Literal["aligned", "too_easy", "too_hard"]] = None

class QAItem(BaseModel):
    qa_id: str
    question: str
    answer: str
    rationale: list[Rationale]

    # メタデータ
    category: list[Literal["Integration", "Reasoning", "Logical", "Table", "Abstention"]]
    diagnostic: Diagnostic
    question_type: Literal["extractive", "multi_hop", "out_of_doc", "ambiguous"]
    difficulty: Difficulty
    persona: str
    business_scenario: str

    # 運用メタ
    generation: GenerationInfo
    filter_scores: FilterScores = Field(default_factory=FilterScores)
    review_status: Literal["pending", "accepted", "edited", "rejected"] = "pending"
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
```

---

## 5軸観点（生成・フィルタ・レビュー全てでこの軸を参照）

1. **カバレッジ**: カテゴリ (Integration/Reasoning/Logical/Table/Abstention) × 4軸診断 × 質問タイプ × 難易度
2. **ソース**: 文書起点70% / ユーザー起点30%、ペルソナ多様化 (設計/品証/現場/新人)
3. **品質**: Answerability / Leakage / Grounding / Uniqueness / 難易度整合 / 根拠完全性
4. **メタデータ**: rationale / カテゴリ / 診断スコア / 難易度+理由 / 生成元 / レビュー状況
5. **規模・バランス**:
    - 規模: パイロット30-50 → 標準100 → 拡張200+
    - 検索難易度: E 35-40% / M 35-40% / H 20-25%
    - 回答難易度: E 15-20% / M 60-65% / H 15-20%
    - Abstention: 全体の10-15%

---

## Phase 0: Gold Seed 50問 (人手作成)

**Claude Code にやらせる作業ではない。人手でやる。** ただしテンプレート JSON を用意しておく。

- `data/seeds/seeds.json` に人手で50問書く
- 最初は10問だけでもいい。徐々に追加
- **これが生成プロンプトの Few-Shot 例になる** から、5軸のバランスを意識して書く

---

## Phase 1: MVP (end-to-end を10問で通す)

### ゴール
「ドキュメント投入 → 10問生成 → フィルタ → レビューUIで確認」が動く。

### タスク

1. **`schema.py`** を上記で実装
2. **`llm.py`** - vLLM と Claude API の呼び出しラッパ (OpenAI互換でよい)
    - 環境変数: `VLLM_BASE_URL`, `ANTHROPIC_API_KEY`
    - `generate(prompt, model, response_format)` のシンプルなI/F
    - JSON mode対応 (Pydanticスキーマを渡したらバリデート済みオブジェクトを返す)
3. **`chunker.py`** - ドキュメント → チャンク
    - まずは単純に「文字数で分割」でいい。LangChainの `RecursiveCharacterTextSplitter` でも可
    - 出力: `data/chunks/*.jsonl` (1行1チャンク、`chunk_id / doc_id / page / text` を持つ)
4. **`prompts/generate.md`** - 生成プロンプト
    - Few-Shot に Gold Seed から3-5問を埋め込む
    - カテゴリ × 難易度マトリクスで配分を指示
    - JSON出力形式をスキーマに沿って厳密指定
    - ペルソナ指定（設計者/品証/現場/新人のうち1つ）
5. **`generate.py`** - Stage 1
    - 入力: `data/chunks/*.jsonl`, 生成数, 配分指示
    - アンカーチャンクを層別サンプリング
    - プロンプト組み立て → LLM → JSON パース → `QAItem` バリデート
    - 出力: `data/raw/batch_YYYYMMDD_HHMM.jsonl`
6. **`prompts/judge.md`** - 判定プロンプト
    - 6観点それぞれを **別プロンプト** で判定（一気にやらせると精度下がる）
    - 各プロンプトはスコアと根拠を JSON で返す
7. **`filter.py`** - Stage 2
    - 入力: raw jsonl
    - 各QAに対し6観点をJudgeLLMで評価
    - 閾値で自動棄却 (Answerability < 4, Leakage = fail, Grounding < 4)
    - 埋め込みで重複除去 (cos_sim > 0.92)
    - 出力: `data/filtered/batch_YYYYMMDD_HHMM.jsonl` (`filter_scores` 付き)
8. **`review_app.py`** - Stage 3 Streamlit
    - filtered jsonl を読む
    - 1問ずつ表示: 質問 / 回答 / 根拠 / カテゴリ / 難易度 / フィルタスコア
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
- 10問生成→フィルタ→レビューUI表示まで動く
- バリデーションエラーが出ない
- 各ステージに最低1つの pytest がある

---

## Phase 2: 実用化 (100問バッチ)

Phase 1 が動いてから着手。

- 配分制御の厳密化（カテゴリ × 難易度マトリクスで目標数を明示し、足りないセルを補う再生成ループ）
- Evolution方式の追加（Seed → Multi-hop → Reasoning → Conditional の書き換えプロンプト）
- ペルソナの完全網羅（4ペルソナ × 各チャンク）
- 難易度検証（宣言難易度 vs gpt-oss-20B実測正答率の乖離をフラグ）
- レビューUI強化（キーボードショートカット、スコア順ソート、カテゴリフィルタ）

---

## Phase 3: スケール (必要になったら)

以下は必要性が出てから追加する。最初から入れない。

- **Langfuse**: プロンプトの差分で品質がどう変わるか追跡したくなったら
- **MLflow**: データセットのバージョン管理を厳密にしたくなったら
- **Argilla**: レビュアーが複数人になって協業が必要になったら
- **DSPy**: プロンプトを自動最適化したくなったら
- **ColQwen / PP-StructureV3**: 図表ヘビーな文書を扱うようになったら

---

## 実装順序（Claude Code向け）

1. `pyproject.toml` + `src/rageval/__init__.py` 雛形
2. `schema.py` + `tests/test_schema.py`
3. `llm.py` (vLLM と Claude の両対応, モックでテスト)
4. `chunker.py` (単純分割でよい)
5. `data/seeds/seeds.json` にサンプル5問だけ先に入れる（後で人手で拡張）
6. `prompts/generate.md` 起草
7. `generate.py` + `tests/test_generate.py` (モックLLMでテスト)
8. `prompts/judge.md` 起草（6観点分）
9. `filter.py` + `tests/test_filter.py`
10. `review_app.py` (Streamlit, シンプル1画面)
11. `cli.py` で全部つなぐ
12. README.md に使い方書く

各ステップで `uv run pytest` で緑にしてから次へ。

---

## 守ること

- **プロンプトは `prompts/*.md` にファイルで持つ**（Pythonにハードコードしない）
- **生成プロンプトにバージョンを埋める**（`prompt_version: "v1.0"` みたいにフロントマターで管理）
- **JSON Schema validation を必ず通す**（パースエラーで落ちない工夫。再生成 or スキップ）
- **LLM呼び出しは再試行・指数バックオフ**（tenacity使う）
- **ログは標準出力にJSONLで出す**（後からLangfuse入れても移行しやすい形にしておく）
- **秘密情報はenvから**（ドキュメントも含む。公開用テストだけリポジトリに入れる）

---

## 参考（過去議論から）

- **富士通方式**: 4軸診断メタデータ、rationale (page+bbox)、Easy/Medium/Hard
- **neoAI方式**: 5カテゴリ、カテゴリ組合せ、Abstention 10-15%
- **RAGAS**: Evolution方式 (Easy → Multi-hop → Reasoning → Conditional)
- **ペルソナ多様化**: 同じチャンクでも問い方が変わる
