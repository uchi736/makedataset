# rageval

社内ドキュメントから **RAG 評価用 QA データセット** を半自動で構築する CLI ツールキット。
チャンク化 → 生成 → 判定で絞り込み → (任意のプロービング) → 人手レビュー → 可視化、を
1コマンドずつ実行できる。

設計の根拠は [rag_eval_dataset_plan.md](rag_eval_dataset_plan.md) / [rag_eval_dataset_r1_plan.md](rag_eval_dataset_r1_plan.md) を参照。

---

## 2つのトラック

このツールは **目的の違う2種類の評価セット** を同じパイプラインから作れる。
最初に「自分はどっちを作りたいのか」を決めてから読み進めると迷わない。

| 比較項目 | `general` トラック (既定) | `kg_poc` トラック |
|---|---|---|
| 何のための評価か | RAG が **広く** どんな質問形に弱いかを総合測定 | **KG-RAG が単純ベクター検索より勝てる** ことを示すための PoC 用 |
| 分類軸の数 | **25観点 × 5カテゴリ** の1階層分類 | **5クエリ型 × 3未知性 = 15セル** + プロービングで `LLM既知性 known/unknown` |
| 軸の定義場所 | [aspects.py](src/rageval/aspects.py) | [tracks/kg_poc.py](src/rageval/tracks/kg_poc.py) |
| 生成プロンプト | [prompts/generate.md](prompts/generate.md) | [prompts/generate_kg_poc.md](prompts/generate_kg_poc.md) |
| 配分の指定 | 観点はループ毎にランダム抽選 | `--mix qt:nov=N,...` で明示。未指定なら15セルに等分 |
| 専用コマンド | `review` / `stats` | `review-kg` / `stats-kg` / `probe` (任意) |
| QAItem 上の主タグ | `category[]` / `aspect[]` | `kg_query_type` / `kg_novelty` / `llm_knowledge` |
| 1ファイルの中で混在は? | 可。`review` 既定は general だけ、`review-kg` は kg_poc だけを表示 | 同上 |

両方とも 4軸の診断タグ (推論複雑度 / 検索難易度 / 構造 / 説明可能性) と Easy/Medium/Hard 難易度は共通で付く。違うのは「主タグ」だけ。

---

## 全体像 (どちらのトラックでも共通)

```
data/docs/         (元文書 .txt/.md/.pdf)
  │  ① chunk                                             ← トラック非依存
  ▼
data/chunks/       (チャンク JSONL, 1 doc = 1 ファイル)
  │  ② generate --track general|kg_poc                   ← ここで分岐
  ▼
data/raw/          (生成直後 QA JSONL, batch_YYYYMMDD_HHMM.jsonl)
  │  ③ filter                                            ← 判定LLM + 重複検知 (共通)
  ▼
data/filtered/     (合格 QA)
  │  ④ probe       (kg_poc のみ。llm_knowledge を後付け)
  │  ⑤ review / review-kg
  ▼
data/reviewed/     (人手 accept/edit/reject 済, コミット対象)
```

各段は同名のファイルに書き出すので、再実行で上書きされる。`stats` / `stats-kg` は
どの段の JSONL に対しても可視化できる。

---

## セットアップ

```bash
uv sync --extra dev
cp .env.example .env  # 後述の環境変数を埋める
```

Python 3.12+ / `uv` 0.8+。

### 環境変数 (`.env`)

| 用途 | 変数 | 既定値 / 例 |
|---|---|---|
| 生成LLM (vLLM, OpenAI互換) | `VLLM_ENDPOINT` (もしくは `VLLM_BASE_URL`) | `http://localhost:8000/v1` |
| 〃 | `VLLM_MODEL` | `openai/gpt-oss-120b` |
| 〃 | `VLLM_API_KEY` | `dummy` (vLLM 既定で OK) |
| 〃 | `VLLM_TIMEOUT` | `300` |
| 埋め込み (重複検知 / 多チャンク戦略) | `VLLM_EMBEDDING_ENDPOINT` | `http://localhost:8003/v1` |
| 〃 | `VLLM_EMBEDDING_MODEL` | `cl-nagoya/ruri-v3-310m` |
| 判定LLM (Azure OpenAI) | `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_API_VERSION` | — |
| 〃 | `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` | `gpt-4.1-mini` |
| 判定LLM (Claude) | `ANTHROPIC_API_KEY` | — |
| PDF 抽出 (Azure DI) | `AZURE_DI_ENDPOINT` / `AZURE_DI_API_KEY` / `AZURE_DI_MODEL` | `prebuilt-layout` |

モデル名で接続先を自動判定する ([llm.py](src/rageval/llm.py#L84-L102)):

- `claude*` / `anthropic/*` → Anthropic
- `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` と一致、または `azure/*` → Azure OpenAI
- それ以外 → vLLM (OpenAI 互換)

---

## コマンド対応表

| コマンド | `general` で使う? | `kg_poc` で使う? | 役割 |
|---|:---:|:---:|---|
| `chunk` | ◯ | ◯ | 文書をチャンクに分割 |
| `discover-patterns` | ◯ | ◯ | コーパス固有の参照識別子を LLM に発見させる (任意) |
| `generate --track general` | ◯ | — | 25観点ベースで QA 生成 |
| `generate --track kg_poc [--mix ...]` | — | ◯ | 3軸ベースで QA 生成 |
| `filter` | ◯ | ◯ | 判定LLMで6軸スコアリング + 重複検知 + 根拠検証 |
| `probe` | — | ◯ | 素の LLM に同じ質問を投げて `llm_knowledge` を付与 |
| `review` | ◯ | — | レビューUI (general 用チェックリスト) |
| `review-kg` | — | ◯ | レビューUI (KG適性チェックリスト付き) |
| `stats` | ◯ | — | ダッシュボード (25観点別の集計タブ) |
| `stats-kg` | — | ◯ | ダッシュボード (3軸別の集計タブ) |

> 1ファイルに両トラックの QA が混ざっていても、`review` / `stats` は `--track` で表示対象を
> 選べる (`general` | `kg_poc` | `all`、既定 `general`)。振り分けは `kg_query_type` /
> `kg_novelty` がセットされているかどうかで判定する ([review_app.py:342-363](src/rageval/review_app.py#L342-L363))。
> `review-kg` / `stats-kg` は `--track kg_poc` を固定した別名。

---

## 共通段: チャンク化と参照識別子

### `chunk` — 文書をチャンクに分割

```bash
uv run rageval chunk --docs data/docs --out data/chunks
# オプション:
#   --chunk-size 800 --chunk-overlap 100
#   --pdf-backend auto|pypdf|azure_di   (auto は AZURE_DI_* があれば azure_di)
```

[chunker.py](src/rageval/chunker.py) の振る舞い:

- 拡張子: `.txt` / `.md` / `.pdf`
- `.md` は見出し階層 (`#`〜`######`) を辿って各チャンクの `section_path` に格納
- `.pdf` (pypdf) はページ単位で抽出して再帰分割、`page` を保持
- `.pdf` (Azure DI) は `prebuilt-layout` の Markdown を経由してページ番号+見出しを両取り
  - Azure DI の結果は `<file>.pdf.di.md` にキャッシュされ、PDF より新しければ再呼び出ししない
- 全形式で「参照識別子」を正規表現で抽出 → `references` に格納
  (JIS/ISO/IEC/ASME、`第N章`、`第N節`、`第N条`、`第N項`、`別表N`、`附属書X`、`N.N.N項` 等)

### `discover-patterns` — 参照識別子の追加発見 (任意, 1回だけ)

社内文書 ID や独自の章節記法など、組み込みの正規表現では拾えない参照識別子を LLM に
列挙させる。発見したパターンは `data/chunks/_discovered_patterns.json` に保存され、
以降の `chunk` 実行で自動的に使われる。

```bash
uv run rageval discover-patterns --chunks data/chunks
# 発見したパターンを実 references に反映したいときは:
uv run rageval discover-patterns --chunks data/chunks --rechunk --docs data/docs
```

ガードレール ([chunker.py:439-461](src/rageval/chunker.py#L439-L461)):
- Python `re` でコンパイル可能か検査
- サンプル内で1件もマッチしないパターンは捨てる
- サンプルの5%以上を覆う貪欲パターン (例 `\d+`) も捨てる

---

## トラック A: `general` (25観点 × 5カテゴリ)

### A-1. 生成

```bash
uv run rageval generate \
  --chunks data/chunks --out data/raw --n 50 \
  --model "$VLLM_MODEL" --track general
```

毎ループで25観点から1つを抽選し、その観点に応じた **アンカーチャンク選択戦略**
([sampling.py](src/rageval/sampling.py)) で1〜3個のチャンクを選んで生成プロンプトに渡す:

| 戦略 | 適用観点 (内部値) |
|---|---|
| `MultiDocByEmbedding(n=3, sim_floor=0.5)` | `multi_source_integration` (複数情報源の統合) |
| `MultiDocByEmbedding(n=2, force_distinct_doc=True, sim_floor=0.55)` | `multi_doc_reference` (マルチドキュメント参照) |
| `SameDocRemote(n=2, gap=3)` | `remote_reference` (遠隔参照) |
| `ReferenceFollow(n=2)` | `standards_reference` (規格・規定番号参照) |
| `MultiDocByEmbedding(n=2)` | `multi_hop` (マルチホップ) |
| `SingleChunk` | 上記以外の20観点 |

さらに各観点には「適合述語」が紐づいており ([sampling.py:125-156](src/rageval/sampling.py#L125-L156))、
例えば `simple_table` には表マーカー (`|---`/`<table>`) を含むチャンクだけが渡される。
適合チャンクが無い場合は警告を出して全プールにフォールバック。

### A-2. 25観点の中身

[aspects.py](src/rageval/aspects.py) より、内部値と日本語ラベル:

| カテゴリ | 観点 (内部値 → 日本語ラベル) |
|---|---|
| **統合 (Integration)** | `multi_source_integration` 複数情報源の統合 / `multi_doc_reference` マルチドキュメント参照 / `remote_reference` 遠隔参照 / `standards_reference` 規格・規定番号参照 |
| **推論 (Reasoning)** | `quantitative_calc` 数値計算 / `multi_hop` マルチホップ / `negation` 否定推論 / `causal` 因果推論 / `temporal` 時間推論 / `comparison_conditional` 比較・条件判断 |
| **論理 (Logic)** | `synonym_interpretation` 同義関係の解釈 / `numeric_inclusion` 数値包含関係 / `concept_inclusion` 概念包含関係 / `vocabulary_mismatch` 語彙ミスマッチ / `abstraction_gap` 抽象度の乖離 |
| **図表 (Figure)** | `simple_table` 単純表 / `complex_form` 複雑帳票 / `concept_diagram` 概念図・構成図 / `flowchart` フローチャート / `chart_graph` グラフ・チャート / `complex_layout` 複雑レイアウト / `large_enumeration` 大量列挙 |
| **棄権 (Abstention)** | `insufficient_evidence` 根拠不足 / `contradictory_evidence` 根拠の矛盾 / `fragmented_chunk` 不完全なチャンク区切り |

### A-3. レビューと可視化

```bash
uv run rageval review --in "data/filtered/batch_*.jsonl"
uv run rageval stats  --in data/raw   # ディレクトリ / 単一ファイル / glob どれもOK
# 表示対象トラックの切替: --track general|kg_poc|all (既定 general)
```

`review` / `stats` は既定で **general トラックの QA** (`kg_query_type` 未設定) だけを表示する。
`--track kg_poc` で KG QA のみ、`--track all` で両方を表示できる。

---

## トラック B: `kg_poc` (3軸 = クエリ型 × 未知性 × LLM既知性)

KG-RAG が単純ベクター検索より勝てる場面を **計画的に** 並べた評価セットを作るための専用トラック。

### B-1. 生成

```bash
# 既定: 15セルへ等分
uv run rageval generate \
  --chunks data/chunks --out data/raw --n 30 \
  --model "$VLLM_MODEL" --track kg_poc

# セルごとの本数を明示 (推奨)
uv run rageval generate --track kg_poc --n 30 \
  --mix "multi_hop:unknown_relation=10, traceability:procedural_relation=8, aggregation:unknown_term=5"
```

`--mix` の文法: `<クエリ型>:<未知性>=<本数>` をカンマ区切りで列挙
([generate.py:103-132](src/rageval/generate.py#L103-L132))。
合計が `--n` より少なければ指定セルから埋め足し、多ければ切り詰める。

### B-2. 3軸の中身

[tracks/kg_poc.py](src/rageval/tracks/kg_poc.py) より:

**AXIS-1 クエリ型** (`kg_query_type`, 5値):

| 内部値 | ラベル | 何を見るか |
|---|---|---|
| `single_fact` | 単一ファクト | 一段で答えられるファクト確認 (基準点) |
| `multi_hop` | マルチホップ | 複数エンティティ・関係を辿る必要がある |
| `aggregation` | 一覧・集約 | 複数箇所にまたがる項目を一覧化 (ベクターRAGが弱い) |
| `traceability` | トレーサビリティ | 要求→手順→記録の追跡 (QMS固有) |
| `negation_exhaustive` | 否定・網羅 | 否定条件や網羅性 (高難度、少数で利く) |

**AXIS-2 未知性** (`kg_novelty`, 3値):

| 内部値 | ラベル | 何を見るか |
|---|---|---|
| `unknown_term` | 未知語 | 事前学習に無い語・表記揺れを KG のノードへ紐づけて解決 |
| `unknown_relation` | 未知の関係 | 関係そのものが事前学習に無い (**KG導入の最も強い主張根拠**) |
| `procedural_relation` | 手順的関係 | 順序・条件・依存。typed-edge (precedes/requires/triggers) で表現 |

**AXIS-3 LLM既知性** (`llm_knowledge`, 2値, **プロービングで決定**):

| 内部値 | ラベル | 何を見るか |
|---|---|---|
| `known` | 既知 | RAG 無しでも LLM が答えられる質問 (KG の貢献が見えにくい) |
| `unknown` | 未知 | RAG 必須 (KG-RAG で改善余地がある主戦場) |

5×3=15セル。生成時の事故防止 ([generate.py:333-353](src/rageval/generate.py#L333-L353)):
- LLM がスペックを下げてくる (例: `multi_hop` 指示で `single_fact` を返す) のを防ぐため、
  `kg_query_type` / `kg_novelty` は生成後に上書きで再代入
- 質問/回答に「アンカーチャンクに無い 4文字以上の片仮名語」が混じった QA は破棄
  ([generate.py:374-386](src/rageval/generate.py#L374-L386))
- pydantic スキーマ違反 / JSON パース失敗もスキップして次の問へ

### B-3. プロービング (AXIS-3 を埋める, 任意)

`filter` 通過後の QA に対して、AXIS-3 (`llm_knowledge`) を付与する。
これをやらないと「KG-RAG の主戦場 = unknown」セルに何問あるかが分からない。

```bash
uv run rageval probe \
  --in data/filtered/batch_YYYYMMDD_HHMM.jsonl \
  --probe-model "$VLLM_MODEL" \
  --judge-model "$AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
# --out を省略すると --in を上書き
```

[probe.py](src/rageval/probe.py) の流れ:

1. 質問だけ素の LLM (RAG 無し / チャンク非提示) に渡す → 候補回答を得る
2. 判定LLM に「候補回答 vs ground truth」を比較させ `match: known | unknown` を取得
3. `qa.llm_knowledge` にセット

候補が「不明」/空文字なら `unknown` とみなす保守的挙動。

### B-4. レビューと可視化

```bash
uv run rageval review-kg --in "data/filtered/batch_*.jsonl"
uv run rageval stats-kg  --in data/filtered
```

`review-kg` は **`kg_query_type` がセットされている QA だけ** を表示する。
チェックリストには KG 適性 (グラフ的アクセスが必要か / 文書中の実体か / 未知性タグが実態と合うか) が含まれる
([review_app.py:481-502](src/rageval/review_app.py#L481-L502))。

---

## 共通段: 判定で絞り込み (どちらのトラックでも同じ)

### `filter`

```bash
uv run rageval filter \
  --in data/raw/batch_YYYYMMDD_HHMM.jsonl \
  --chunks data/chunks --out data/filtered \
  --model "$AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"  # もしくは claude-sonnet-4-6 等
```

判定軸 ([filter.py](src/rageval/filter.py), [prompts/judge.md](prompts/judge.md)):

| 観点 | 種別 | 既定しきい値 |
|---|---|---|
| `answerability` | 1〜5 | `--answerability-min 4.0` |
| `grounding` | 1〜5 | `--grounding-min 4.0` |
| `leakage` | `pass`/`fail` | `--skip-leakage` で無効化可 |
| `difficulty_match` | `aligned`/`too_easy`/`too_hard` | 棄却条件には使わない (タグのみ) |
| `uniqueness` (埋め込みコサイン由来) | 0〜1 | `--uniqueness-max 0.92` 超で重複扱い |
| `rationale_grounded` (決定論) | 0〜1 | `--rationale-grounded-min 1.0` |

`rationale_grounded` は判定LLMを使わず、QAItem の `rationale[].text` がアンカーチャンクに
**空白を無視してそのまま含まれているか** で照合 ([filter.py:85-102](src/rageval/filter.py#L85-L102))。
既定は 1.0 (すべての rationale が逐語引用必須)。

`--skip-uniqueness` で埋め込み呼び出しを丸ごとスキップ。

### レビューUI 共通の振る舞い

`review` / `review-kg` ([review_app.py](src/rageval/review_app.py)):

- 上部: qa_id / トラック / 主タグ / 検索・回答難易度 / レビューstatus
- 左: 質問・回答 (Edit モードでテキストエリア化)
- 右: アンカーチャンク (rationale を黄色ハイライト) + rationale ごとの逐語一致◯✗
- 下部: トラック別チェックリスト (全てチェックで Accept ボタン解放) + Reject / Edit / Accept / Save snapshot
- 折り畳み: 判定LLMスコア / 生 JSON

`--in` は glob 可。glob のときは最新一致ファイルを開く。
書き戻し先は `data/reviewed/<元ファイル名>.jsonl`。

---

## QA データのスキーマ

[schema.py](src/rageval/schema.py) の `QAItem` 抜粋。**主タグ部分がトラックで分かれる** ことに注意:

```python
class QAItem:
    qa_id: str                          # 安定ID: sha1(anchors + 主タグ + question)[:8]
    question: str
    answer: str
    rationale: list[Rationale]          # {doc_id, page, text} の配列

    # ===== 階層① 能力評価軸 (general トラックで埋まる) =====
    category: list[CategoryName]        # Integration / Reasoning / Logic / Figure / Abstention
    aspect:   list[str]                 # 25観点のいずれか

    # ===== 階層② 診断軸 (両トラック共通) =====
    reasoning_complexity: ReasoningComplexity
        # multi_step / quantitative / negation / cause_effect / comparison / temporal + output_type
    retrieval_difficulty: RetrievalDifficulty
        # multi_doc / multi_chunk / low_locality / remote_reference / doc_volume_large
        # / chunk_size_large / abstraction_discrepancy / vocabulary_mismatch
    source_structure:     SourceStructure
        # tables_charts / complex_layout / specific_area_ref / logical_nesting
        # / large_enumeration / redundancy
    explainability:       Explainability
        # evidence_strictness: no-evidence | hier-ref | coord-ref | multi-ref

    # ===== 階層③ 難易度 (両トラック共通) =====
    retrieval_level: "Easy" | "Medium" | "Hard"
    answer_level:    "Easy" | "Medium" | "Hard"
    difficulty_rationale: str

    # ===== kg_poc トラック専用 (general では None) =====
    kg_query_type: KGQueryType | None   # single_fact / multi_hop / aggregation / traceability / negation_exhaustive
    kg_novelty:    KGNovelty   | None   # unknown_term / unknown_relation / procedural_relation
    llm_knowledge: "known" | "unknown" | None   # probe で後付け

    # ===== 運用メタ (両トラック共通) =====
    generation:    GenerationInfo       # model / prompt_version / generated_at
    filter_scores: FilterScores         # 判定LLMの各軸スコア
    review_status: "pending" | "accepted" | "edited" | "rejected"
    reviewed_by:   str | None
    reviewed_at:   datetime | None
```

**トラックの判定ロジック** ([review_app.py:357](src/rageval/review_app.py#L357)):
`kg_query_type` か `kg_novelty` のどちらかが non-None なら kg_poc、両方 None なら general。

---

## ディレクトリ

- [src/rageval/](src/rageval/) — Python 実装本体
  - [cli.py](src/rageval/cli.py) — Typer エントリポイント (9コマンド)
  - [chunker.py](src/rageval/chunker.py) — 文書 → チャンク + 参照識別子抽出 + 任意の LLM パターン発見
  - [azure_di.py](src/rageval/azure_di.py) — Azure Document Intelligence (prebuilt-layout) ラッパー
  - [generate.py](src/rageval/generate.py) — Stage 1: QA 生成 (両トラック対応)
  - [sampling.py](src/rageval/sampling.py) — アンカーチャンク選択戦略
  - [filter.py](src/rageval/filter.py) — Stage 2: 判定LLM + 重複検知 + 逐語根拠検証
  - [probe.py](src/rageval/probe.py) — kg_poc 用 LLM プロービング
  - [review_app.py](src/rageval/review_app.py) — レビューUI (general / kg_poc の両モード)
  - [stats_app.py](src/rageval/stats_app.py) — ダッシュボード (general / kg_poc の両モード)
  - [aspects.py](src/rageval/aspects.py) — **general トラックの 25観点定義**
  - [tracks/kg_poc.py](src/rageval/tracks/kg_poc.py) — **kg_poc トラックの 3軸定義**
  - [schema.py](src/rageval/schema.py) — `QAItem` / `Chunk` などの型定義
  - [llm.py](src/rageval/llm.py) — vLLM / Azure / Anthropic 共通ラッパー (tenacity で3回まで再試行)
  - [prompts.py](src/rageval/prompts.py) — frontmatter 付き Markdown プロンプトのローダ
- [prompts/](prompts/) — frontmatter 付き Markdown プロンプト
  - [generate.md](prompts/generate.md) — **general** 用生成プロンプト
  - [generate_kg_poc.md](prompts/generate_kg_poc.md) — **kg_poc** 用生成プロンプト
  - [judge.md](prompts/judge.md) — `[ANSWERABILITY]` / `[LEAKAGE]` / `[GROUNDING]` / `[DIFFICULTY_MATCH]` セクション分割 (両トラック共通)
  - [probe.md](prompts/probe.md) — `[PROBE]` + `[JUDGE_MATCH]` セクション (kg_poc 専用)
- [data/docs/](data/docs/) — 元文書 (gitignore)
- [data/chunks/](data/chunks/) — チャンク済み JSONL (gitignore)
- [data/seeds/seeds.json](data/seeds/seeds.json) — Gold Seed (人手で拡張する Few-Shot 用)
- [data/raw/](data/raw/) — 生成直後の QA (gitignore)
- [data/filtered/](data/filtered/) — フィルタ後 (gitignore)
- [data/reviewed/](data/reviewed/) — 人手レビュー済み (コミット対象)
- [tests/](tests/) — pytest (chunker / azure_di / filter / sampling / schema / generate)

## テスト

```bash
uv run pytest
```

`pythonpath` は `src` を見るよう [pyproject.toml](pyproject.toml#L38-L40) で設定済み。
