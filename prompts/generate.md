---
prompt_version: v5.2
description: RAG評価用QA生成プロンプト(R1 25観点版, 難易度は後段で確定。観点別の作問ガイド + 共通禁止ルール)
---

# 役割

あなたは社内ドキュメントから RAG 評価用の QA ペアを1問だけ生成する専門家です。
以下の「元チャンク群」に基づき、3階層の評価メタデータ(能力評価軸 / 診断軸 / 難易度)を
全て出力してください。

複数チャンクが渡された場合は、**それら全てを参照しないと答えられない質問**を作ること。
1チャンクだけ渡された場合は、そのチャンクから答えられる質問を作ること。

{anchor_chunks_block}

# 生成条件

- **観点 (aspect)**: `{aspect}` を満たす質問を作る
- **カテゴリ (category)**: 観点に対応する `{category}` を使う
- **難易度**: 検索難易度は後段でチャンク組成から自動判定するので指定しない。回答難易度は質問が要求する推論の量で決まる — 観点に見合うだけの推論・統合を要する問いにすること(抜き出しで済む安直な問いにしない)。

## 観点別の作問ガイド

- **{aspect} の定義**: {aspect_description}
- **{aspect} の現場例**: {aspect_example}

### この観点で ✓ 良い問いの例

{aspect_good_examples_block}

### この観点で ✗ 避けるべき問いのパターン

{aspect_bad_patterns_block}

### 共通の禁止ルール(全観点)

- 質問文に **答えそのもの**(具体的な数値・固有名・ステップ名)を書かない
- 「○○の個数はいくつか」「全部で何件か」のように **列挙を数えるだけ** の問いは作らない
- 元チャンクの本文をそのまま抜き出すだけの問いは作らない(理解・統合・推論を要求する形にする)
- 観点を満たす根拠が元チャンクに無いと判断した場合は、**無理に問いを作らず**、`category=["Abstention"]` / `aspect=["insufficient_evidence"]` に倒して、回答を「本ドキュメントには該当する情報がありません」で始めること

# Few-Shot 例

人手で作成された QA 例です。質問粒度・回答明瞭さ・診断軸タグの埋め方を参考にしてください。

```json
{few_shot}
```

# 出力スキーマ(厳守)

以下の JSON オブジェクトを **1件だけ** 返してください。前後にコメント・説明は禁止。

```json
{{
  "qa_id": "auto_<8桁ハッシュ>",
  "question": "<質問文。日本語。曖昧さを避け1問1答>",
  "answer": "<簡潔な模範回答。単位・数値は明示>",
  "rationale": [
    {{"doc_id": "<上記元チャンクの doc_id をそのまま (NG例: chunk_id 'foo__c0154' を入れない、必ず 'foo' 形式)>", "page": <page or null>, "text": "<該当チャンク内の連続部分文字列>"}}
  ],

  "category": {category_json},
  "aspect": {aspect_json},

  "reasoning_complexity": {{
    "multi_step": false, "quantitative": false, "negation": false,
    "cause_effect": false, "comparison": false, "temporal": false,
    "output_type": "none"
  }},
  "retrieval_difficulty": {{
    "multi_doc": false, "multi_chunk": false, "low_locality": false,
    "remote_reference": false, "doc_volume_large": false, "chunk_size_large": false,
    "abstraction_discrepancy": false, "vocabulary_mismatch": false
  }},
  "source_structure": {{
    "tables_charts": false, "complex_layout": false, "specific_area_ref": false,
    "logical_nesting": false, "large_enumeration": false, "redundancy": false
  }},
  "explainability": {{"evidence_strictness": "hier-ref"}},

  "retrieval_level": "<Easy|Medium|Hard。下の基準で自己判定。後段でチャンク組成から上書き確定される>",
  "answer_level": "<Easy|Medium|Hard。下の回答難易度基準で判定。後段で判定LLMが上書き確定する>",
  "difficulty_rationale": "<必要チャンク数・推論ステップ数を定量で>"
}}
```

# 診断軸タグ付け規則

各 bool フィールドは「質問とその回答に該当する性質」を true にする。例:
- 計算が必要 → `reasoning_complexity.quantitative=true`, `reasoning_complexity.multi_step=true`
- 専門用語と質問語彙が違う → `retrieval_difficulty.vocabulary_mismatch=true`
- 表/図/帳票からの抽出 → `source_structure.tables_charts=true`
- 章をまたぐ参照 → `retrieval_difficulty.remote_reference=true`
- 規格・規定の番号参照 → 観点 `standards_reference` + `retrieval_difficulty.multi_doc=true` の可能性
- 根拠1箇所 → `explainability.evidence_strictness="hier-ref"`
- 並列に複数根拠 → `"coord-ref"`、厳密に複数根拠を提示すべき → `"multi-ref"`
- 根拠なし(Abstention) → `"no-evidence"`

# 難易度判定基準

| 軸 | Easy | Medium | Hard |
|---|---|---|---|
| 検索難易度 | 1チャンクで答え完結、キーワード一致 | 2-3チャンク、言い換え検索が必要 | 3チャンク以上、専門用語・略語の解釈 |
| 回答難易度 | 抜き出し型、推論不要 | 1-2ステップの推論、比較・統合 | 3ステップ以上、条件分岐、計算 |

# 品質ルール(必ず守る)

1. 回答は **元チャンクのテキストだけから一意に決まる**こと(憶測・外部知識を混ぜない)。
2. 質問文に **答えそのもの**(具体的な数値・固有名・ステップ名)を書かないこと。
3. rationale.text は **元チャンクのいずれかの内に連続した部分文字列**であること(改変・要約しない)。
4. **複数元チャンクが与えられた場合**、rationale には参照した各チャンクから引用を入れる(2チャンクなら rationale も2件)。
5. category が "Abstention" の場合、answer は「本ドキュメントには該当する情報がありません」で始めること。
6. 自己判定した難易度と実態(必要チャンク数・推論ステップ数)を difficulty_rationale で必ず突き合わせる。
7. aspect は **指定された値だけ**を含むこと(勝手に追加・差し替えない)。
