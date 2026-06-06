# RAG評価データセット作成手順書

## 目的

ドメイン特化KG-RAGシステムの精度評価および改善診断のための評価データセットを構築する。
本手順は以下2つのベンチマークの設計思想・手法を統合し、自社ドメインに適用可能な形にまとめたものである。

- **Fujitsu RAG Hard Benchmark**：4軸診断メタデータによる失敗要因分解
- **neoAI J-RAGBench（LIT-RAGBench）**：5つの評価カテゴリによるGenerator能力の体系的評価

---

## 全体フロー

```
Step 1: 評価軸の定義
  ↓
Step 2: 対象文書の選定・整備
  ↓
Step 3: QAペアの生成（文書起点 + ユーザー起点）
  ↓
Step 4: 診断メタデータの付与
  ↓
Step 5: 難易度ラベルの付与
  ↓
Step 6: 品質フィルタリング・バランス調整
  ↓
Step 7: 評価スクリプトの整備
```

---

## Step 1: 評価軸の定義

### 1-1. 診断4軸（Fujitsu方式）

RAGパイプラインのどこで失敗しているかを切り分けるための軸。

| 診断軸 | 何を測るか | 具体例 |
|--------|-----------|--------|
| **検索難易度**（Retrieval Difficulty） | 根拠文書を見つける難しさ | 単一チャンク / 複数チャンク / 複数文書 |
| **推論難易度**（Reasoning Complexity） | 回答を導出する推論の深さ | 単一ステップ / マルチステップ / 数値計算 |
| **文書構造・モダリティ**（Source Structure & Modality） | 文書形式の読解難易度 | テキストのみ / 表あり / 図あり / 複雑レイアウト |
| **説明可能性要件**（Explainability Requirement） | 根拠提示の厳密さ | 根拠不要 / 単一根拠 / 複数根拠の厳密提示 |

### 1-2. 評価5カテゴリ（neoAI方式）

Generator（LLM）の能力を体系化する軸。

| カテゴリ | 定義 | 評価観点の例 |
|---------|------|-------------|
| **Integration**（情報統合） | 2〜3文書の情報を抽出・統合して回答 | 複数根拠の統合、情報の突合 |
| **Reasoning**（推論） | 抽出情報に基づく多段階推論・数値計算 | 比較推論、条件分岐、計算 |
| **Logical**（論理条件解釈） | 質問と文書間の語彙・表現差異を解釈 | 言い換え理解、否定条件、包含関係 |
| **Table**（表形式解釈） | 表形式の根拠から情報を抽出 | セル値の特定、行列の対応付け |
| **Abstention**（回答拒否） | 根拠不在時に適切に「不明」と回答 | ハルシネーション抑制 |

### 1-3. 統合方針

- 各QAペアに対し、**4軸診断メタデータ**（パイプライン診断用）と**評価カテゴリ**（Generator能力評価用）の両方を付与する
- 4軸は主にシステム改善の診断に、5カテゴリはモデル選定・比較に活用する

---

## Step 2: 対象文書の選定・整備

### 2-1. 文書選定の基準

以下の性質をバランスよくカバーする文書セットを構成する。

| 文書特性 | 含めるべき理由 | 目安比率 |
|---------|---------------|---------|
| テキスト中心の文書 | ベースライン評価 | 30% |
| 表を多く含む文書 | Table評価カテゴリ | 30% |
| 図・フローチャートを含む文書 | マルチモーダル評価 | 20% |
| 複雑レイアウト（段組み・ヘッダ階層） | 構造理解の評価 | 20% |

### 2-2. 文書整備

- 各文書にID（file_name）を付与
- ページ番号を明確にする（根拠のトレーサビリティ確保のため）
- 可能であればPDF形式で管理し、バウンディングボックスによる領域指定に対応できるようにする

### 2-3. 架空シナリオの検討（任意だが推奨）

neoAI方式では、LLMの事前知識による「カンニング」を防ぐため架空シナリオに基づく文書を使用している。
ドメイン特化評価の場合は実文書が必要だが、汎用ベンチマーク用途も兼ねる場合は架空文書の混在を検討する。

---

## Step 3: QAペアの生成

### 3-1. 二つの起点で生成する

| 生成方式 | 手法 | 特徴 | 推奨比率 |
|---------|------|------|---------|
| **文書起点**（Document-grounded） | チャンクを先に選び、そこから質問を生成 | 回答可能性が担保される | 70% |
| **ユーザー起点**（User-grounded） | 実際のユーザーが聞きそうな質問を起点にする | リアルだが回答不可能が混じる | 30% |

### 3-2. 生成手順

#### 文書起点の場合

1. 対象文書からチャンク（段落・表・図）を選択
2. そのチャンクから回答可能な質問をLLMで生成（プロンプト例は後述）
3. 人手で質問の自然さ・妥当性をレビュー
4. 正解回答を作成（文書の該当箇所を明記）

#### ユーザー起点の場合

1. 実際の業務でユーザーが聞きそうな質問を収集（ヒアリング or ログ分析）
2. 各質問に対して文書コーパスを検索し、回答可能性を判定
3. 回答可能 → 正解回答を作成 / 回答不可能 → Abstention（回答拒否）カテゴリとして保持

### 3-3. QA生成プロンプト例（文書起点）

```
以下の文書チャンクに基づいて、RAGシステムの評価用QAペアを生成してください。

【文書チャンク】
{chunk_text}

【生成条件】
- 質問は日本語で、業務担当者が実際に聞きそうな自然な表現にする
- 以下の難易度のうちひとつを指定して生成する：
  - Easy: チャンク内の記載をそのまま抽出すれば回答可能
  - Medium: 複数箇所の情報を整理・要約する必要がある
  - Hard: 比較・条件判断・数値計算・多段推論を伴う
- 回答には根拠となる箇所（ページ番号、セクション）を明記する

【出力形式】
{
  "question": "...",
  "answer": "...",
  "difficulty": "Easy/Medium/Hard",
  "rationale_location": "ファイル名, ページX, セクションY"
}
```

### 3-4. 評価カテゴリの組み合わせ網羅（neoAI方式）

neoAI方式では、5カテゴリの1種または2種の全組み合わせを網羅するように設計する。
主要4カテゴリ（Integration, Reasoning, Logical, Table）の2カテゴリ組み合わせは C(4,2) = 6通り。
Abstentionは独立扱いとする。

| パターン | 例 |
|---------|-----|
| Integration単独 | 2文書の情報を統合して回答 |
| Reasoning単独 | 単一根拠から多段推論 |
| Integration × Reasoning | 複数文書を統合した上で比較推論 |
| Integration × Table | 複数文書の表を突き合わせて回答 |
| Reasoning × Table | 表の数値を使って計算 |
| Logical × Table | 表の条件を論理的に解釈 |
| ... | ... |

---

## Step 4: 診断メタデータの付与

各QAペアに以下のメタデータをアノテーションする。

### 4-1. アノテーション項目

```yaml
- no: "001"
  question: "..."
  answer: "..."
  
  # --- 根拠情報 ---
  rationales:
    - file_name: "manual_A.pdf"
      pages:
        - number: 5
          bounding_boxes:  # 任意（領域レベルの根拠提示）
            - top: 30.82
              left: 0.25
              width: 22.75
              height: 32.57
  
  # --- 診断4軸（Fujitsu方式） ---
  Retrieval_Difficulty:
    scope: "multi-chunk"          # single-chunk / multi-chunk / multi-document
    num_source_documents: 2
    num_source_chunks: 3
  
  Reasoning_Complexity:
    depth: "multi-step"           # single-step / multi-step
    involves_calculation: false
    involves_comparison: true
  
  Source_Structure_Modality:
    requires_table: true
    requires_figure: false
    complex_layout: false
  
  Explainability_Requirement:
    level: "multiple-rationales"  # none / single / multiple-rationales
  
  # --- 評価カテゴリ（neoAI方式） ---
  evaluation_categories:
    primary: "Integration"
    secondary: "Reasoning"        # null if single category
  
  # --- 難易度ラベル ---
  retrieval_level: "Medium"       # Easy / Medium / Hard
  answer_level: "Hard"            # Easy / Medium / Hard
```

### 4-2. 難易度ラベルの定義（定量基準付き）

Fujitsu方式の定義をベースに、定量的な閾値を追加してアノテーターの揺れを低減する。

#### 検索難易度（retrieval_level）

| ラベル | 定義 | 定量基準 |
|--------|------|---------|
| Easy | 根拠が比較的見つけやすく探索範囲が狭い | 根拠が1文書・1〜2チャンクに収まる |
| Medium | 複数箇所の探索や追加の読み取りが必要 | 根拠が1文書・3チャンク以上、または表/図の読解が必要 |
| Hard | 複数文書や離れた箇所から根拠を特定 | 根拠が2文書以上、または10ページ以上離れた箇所に散在 |

#### 回答難易度（answer_level）

| ラベル | 定義 | 定量基準 |
|--------|------|---------|
| Easy | 根拠が見つかればほぼ記載通りに答えられる | 抽出型（extractive）、推論ステップ0〜1 |
| Medium | 複数根拠の整理・要約・対応付けが必要 | 情報統合が必要、推論ステップ2〜3 |
| Hard | 比較・条件判断・数値処理・多段推論を伴う | 推論ステップ4以上、または数値計算・論理判断を含む |

---

## Step 5: 品質フィルタリング・バランス調整

### 5-1. QAペアの品質チェック

以下の観点で人手レビューを行い、不適切なQAを除外する。

| チェック項目 | 除外基準 |
|-------------|---------|
| 回答可能性 | 文書コーパスに回答根拠がない（Abstentionカテゴリを除く） |
| 質問の明確性 | 質問が曖昧で正解が一意に決まらない |
| 回答の正確性 | 正解回答が文書と矛盾している |
| 根拠の追跡可能性 | rationales（ファイル名・ページ）が不正確 |

### 5-2. バランス調整の目安

最終的なデータセットの構成バランス。規模は用途に応じて調整する。

#### 規模の目安
- 最小構成（パイロット評価）：30〜50問
- 標準構成（モデル比較・改善診断）：100問
- 拡張構成（包括的ベンチマーク）：200問以上

#### 難易度バランス（Fujitsu方式参考）
- 検索難易度：Easy 35〜40% / Medium 35〜40% / Hard 20〜25%
- 回答難易度：Easy 15〜20% / Medium 60〜65% / Hard 15〜20%

#### 評価カテゴリバランス（neoAI方式参考）
- 各主要カテゴリ（Integration, Reasoning, Logical, Table）が最低10問ずつ
- 2カテゴリ組み合わせパターンが各3〜5問
- Abstention（回答拒否）が全体の10〜15%

---

## Step 6: 評価スクリプトの整備

### 6-1. 評価指標

| 評価対象 | 指標 | 説明 |
|---------|------|------|
| 回答精度 | LLM-as-Judge（0/1） | LLMによる正誤判定 |
| 根拠一致率 | match-rate | 正解根拠との一致率 |
| 根拠網羅性 | full-coverage | 正解根拠をすべて含むか |
| カテゴリ別精度 | category-wise accuracy | 評価カテゴリごとの正解率 |
| 軸別精度 | axis-wise accuracy | 診断4軸の条件ごとの正解率 |

### 6-2. 分析の観点

診断メタデータを活用して、以下のような失敗パターンの切り分けを行う。

- 「検索は通るが表が読めずに誤答する」→ Table能力が弱い
- 「答えは合うが複数根拠を出し切れない」→ Explainability要件未達
- 「単一文書なら正解だが複数文書だと崩れる」→ Integration能力が弱い
- 「抽出型は正解だがマルチホップで精度が落ちる」→ Reasoning能力が弱い
- 「根拠不在時にハルシネーションする」→ Abstention能力が弱い

---

## Step 7: 運用・拡張

### 7-1. データセットの継続的拡張

- 富士通PoCから判明した課題を新規QAとして追加
- 実運用中のRAGシステムのエラーケースをデータセットに還元
- 評価カテゴリ・診断軸の追加拡張（例：時系列推論、因果推論）

### 7-2. 外部ベンチマークとの併用

| ベンチマーク | 用途 | 補完ポイント |
|-------------|------|-------------|
| Fujitsu RAG Hard Benchmark | E2E評価（検索〜回答〜根拠提示） | 公開PDFベースで再現可能 |
| J-RAGBench / LIT-RAGBench | Generator能力の横断比較 | 架空シナリオで事前知識排除 |
| JEMHopQA | マルチホップQA特化 | Wikipedia基盤、KG-RAG評価向き |
| JQaRA | Retriever単体の検索性能 | nDCG@10ベースの検索評価 |
| 自社ドメインデータセット（本手順で構築） | ドメイン特化の改善診断 | 実文書・実業務に即した評価 |

---

## 参考リソース

- Fujitsu RAG Hard Benchmark: https://github.com/FujitsuResearch/Fujitsu-RAG-Hard-Benchmark
- J-RAGBench (neoAI): https://huggingface.co/datasets/neoai-inc/Japanese-RAG-Generator-Benchmark
- LIT-RAGBench 論文: https://arxiv.org/html/2603.06198
- JEMHopQA: https://github.com/aiishii/JEMHopQA
- JQaRA: https://huggingface.co/datasets/hotchpotch/JQaRA
