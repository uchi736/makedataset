---
prompt_version: kg_poc_v1.2
description: KG-RAG PoC 評価用 QA 生成プロンプト (3軸タグ + KG主戦場狙い + 造語抑制。難易度は後段で確定)
---

# 役割

あなたは社内ドキュメントから **KG-RAG (Knowledge Graph 強化 RAG) の評価用 QA** を1問だけ生成する専門家です。
ベクター RAG では取り切れない「未知の関係」「マルチホップ」「手順的関係」を主戦場として狙います。

複数元チャンクが渡された場合は、**それら全てを参照しないと答えられない質問**を作ること。

{anchor_chunks_block}

# 生成条件

- **クエリ型 (kg_query_type)**: `{kg_query_type}`  — {kg_query_type_desc}
- **未知性 (kg_novelty)**: `{kg_novelty}`  — {kg_novelty_desc}
- **難易度**: 検索難易度は後段でチャンク組成から自動判定するので指定しない。回答難易度は質問が要求する推論の量で決まる — 主戦場(マルチホップ/未知の関係/手順的関係)に見合うだけの推論を要する問いにすること。

## クエリ型ごとの作問ガイド

- **single_fact**: 1チャンク・1段で答えられるファクト型(baseline 用、なるべく単純に)
- **multi_hop**: A→B→C と複数エンティティを辿る。中間結論を経由する
- **aggregation**: 「全項目を列挙せよ」「該当する全条文を抽出せよ」型
- **traceability**: 要求→手順→記録 のような追跡。QMS 固有の系譜
- **negation_exhaustive**: 「適用されない場合は?」「除外条件は?」のような否定/網羅

## 未知性ごとの作問ガイド

- **unknown_term**: **元チャンクに実際に登場する用語・社内呼称・略語のみ**を使う。
  - ✓ 良い例(チャンクに「振替休日」と書いてある場合): 「振替休日を設定する際に必要な要件は?」
  - ✗ 悪い例: 「ディスミッション・エクスクルージョン条件として就業規則に記載すべき事項は?」
    - ← チャンクに無い造語を作るのは禁止
- **unknown_relation**: ★主役。事前学習では結びつかない関係を辿らせる。
  - **必ず文書中に明示されている関係**を扱う(例: 「Aは Bにより上書きされる」「Cの場合は Dを適用しない」)
  - ✓ 良い例: 「就業規則 第19条 の所定労働時間を、業務都合で超えることが認められる根拠条文は?」
  - ✗ 悪い例: 自分で勝手に発明した関係名(「サブオーディネーション条項」等)を使うのは禁止
- **procedural_relation**: 順序・条件・依存。typed-edge (precedes/requires/triggers) で表現できる関係
  - ✓ 良い例: 「振替休日の通知タイミングと、通知が間に合わない場合の代替手段の順序は?」

# Few-Shot 例

```json
{few_shot}
```

# 出力スキーマ (厳守)

以下の JSON オブジェクトを **1件だけ** 返してください。前後にコメント・説明は禁止。

```json
{{
  "qa_id": "auto_<8桁ハッシュ>",
  "question": "<質問文。日本語。曖昧さを避け1問1答>",
  "answer": "<簡潔な模範回答。単位・数値は明示>",
  "rationale": [
    {{"doc_id": "<元チャンクの doc_id をそのまま (chunk_id ではない)>", "page": <page or null>, "text": "<該当チャンク内の連続部分文字列>"}}
  ],

  "category": [],
  "aspect": [],

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

  "retrieval_level": "<Easy|Medium|Hard。自己判定。後段でチャンク組成から上書き確定される>",
  "answer_level": "<Easy|Medium|Hard。回答難易度を判定。後段で判定LLMが上書き確定する>",
  "difficulty_rationale": "<必要チャンク数・推論ステップ数・経由エンティティ数を定量で>",

  "kg_query_type": "{kg_query_type}",
  "kg_novelty": "{kg_novelty}"
}}
```

# 品質ルール (必ず守る)

1. 回答は **元チャンクのテキストだけから一意に決まる**こと(憶測・外部知識を混ぜない)。
2. 質問文に **答えそのもの**(具体的な数値・固有名・ステップ名)を書かないこと。
3. rationale.text は **チャンク内の連続した部分文字列**であること(改変・要約しない)。
4. **複数元チャンクが与えられた場合**、rationale には参照した各チャンクから引用を入れる(2チャンクなら rationale も2件)。
5. kg_query_type / kg_novelty は **指定された値だけ**を含むこと(勝手に変えない)。
   `aspect` と `category` は **空配列 `[]` のままにする**(KG-PoC track では使わない)。
6. `unknown_relation` を指定された場合: 「LLMが事前学習だけでは結びつけられない関係」を意図的に作る。汎用知識で答えられる質問は不可。
7. `procedural_relation` を指定された場合: 順序・依存・条件のいずれかを含む。
8. **★ 造語禁止 (最重要)**:
   - 質問・回答・rationale に登場する **すべての固有名詞・専門用語・カタカナ語・略語** は、
     元チャンクの本文に **そのままの文字列で必ず登場する**ものに限る。
   - チャンクに無い英語ベースの造語(「〜・〜条件」「〜エクスクルージョン」等)を作るのは厳禁。
   - 既知の労働基準法用語(時間外労働・有給休暇・産前産後休業 等)も、
     **チャンクに書かれていなければ使わない**。
   - 用語の言い換え・要約も禁止。チャンクに「振替休日」とあれば「振り替え休日」「代休日」と書き換えない。
9. **「文書外の情報で答えられる」質問は禁止**。
   - 例えば一般的な労働基準法の常識で答えられる質問は、KG-RAG の検証価値が無い。
   - 「この文書ではどう定めているか」「この文書ではどの条文が…」のように、
     **文書固有の取り決め**を聞く形にすること。
