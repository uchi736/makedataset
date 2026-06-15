---
prompt_version: rag_verify_v1.0
description: vector RAG が解けるかを ground truth として測るための2段プロンプト
---

## [RAG_ANSWER]

以下の検索結果(top-k チャンク)だけを根拠に、質問に答えてください。
チャンクに無い情報を補ったり、事前学習知識で推測したりしてはいけません。

答えがチャンクから一意に導けないと判断した場合は、answer に
**正確に「回答不能」という4文字だけ**を返してください。
「不明」「わからない」「判断できません」「該当なし」「記載なし」
「文書からは特定できません」など、別の言い回しに置き換えてはいけません。
迷ったときも、必ず「回答不能」と返してください。

[検索結果]
{chunks_block}

[質問]
{question}

出力(JSON のみ):
```json
{"answer": "<チャンクから導ける回答 or '回答不能'>"}
```

---

## [JUDGE_MATCH]

ground truth と候補回答が事実として一致するかを判定してください。
表現の違い・言い回しの差異は許容します(意味が合っていれば一致)。

判定基準:
- match: 主要事実が一致(細かい言い換え・補足は許容)
- partial: 一部一致、欠落あり、または曖昧
- no_match: 主要事実が違う、または候補が "回答不能" 等で答えになっていない

[ground truth]
{ground_truth}

[candidate]
{candidate}

出力(JSON のみ):
```json
{"match": "match|partial|no_match", "reason": "<1-2文>"}
```
