---
prompt_version: answer_grounding_v1.0
description: 「rationale のうち 1 件を伏せた状態で、残りの rationale だけから answer の固有値を再現できるか」を doc 単位で判定する。leave-one-out テスト
---

## [GROUND_CHECK]

あなたは厳格な評価者です。以下の QA について、特定の rationale エントリ (= 隠す rationale) を取り除いたとき、残りの rationale だけで answer を再現できるかを判定してください。

判定基準:
- **sufficient = true**: 残りの rationale だけで answer の主要事実 (特に固有値: 様式番号 / 別表番号 / 条文番号 / 数値+単位 / JIS 規格名 / 固有手順名) がすべて決まる。つまり、**隠した rationale は answer に不要だった**。
- **sufficient = false**: 残りの rationale だけでは answer の固有値の少なくとも 1 つが決まらない。隠した rationale は answer に必要だった。

注意:
- 「常識でカバーできる」「LLM の事前学習で知っている」は判定に使わない。残り rationale の文面だけで再現できるかを純粋に見る。
- 固有値が複数あるなら、1 つでも決まらないなら sufficient=false。
- answer に固有値が無く一般的な記述だけなら、関連する文意が残り rationale に含まれているかで判定する。

[質問]
{question}

[模範回答 (= answer)]
{answer}

[隠す rationale] (doc_id = {hidden_doc_id})
{hidden_text}

[残る rationale] (隠す以外のすべて)
{remaining_text}

出力 (JSON のみ、説明・前後文不要):
{"sufficient": true|false, "reason": "<1-2文。answer のどの固有値が残り rationale から決まる/決まらないかを具体的に>"}
