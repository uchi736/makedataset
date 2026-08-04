# データセット台帳

最終更新: 2026-07-29

**このファイルの役割**: `data/` に何があるか、どれが現役か、何から作られたかを1枚で確定させる。
評価結果側（どの数字を引用してよいか）の台帳は `C:/work/RAG/graphrag/_bench/plant/MANIFEST.md`。

`data/` は git 管理外なので、ここに書かれた内容だけが素性の記録になる。

---

## 0. 用語

- **設問セット** — 質問と正解（模範解答＋根拠文書）のセット。JSONL 1行1問
- **コーパス** — 設問セットが参照する文書チャンク群。設問セットは対応するコーパスとペアでしか成立しない
- **正解参照** — その問に答えるのに必要な文書ID。`rationale[].doc_id` を重複排除したもの
- **review_status** — `pending`（人手レビュー未通過）/ `accepted`（人手承認）/ `edited` /
  `auto_fixed`（後から機械＋LLMで正解を直した）

---

## 1. 現役の設問セット

| ファイル | 件数 | review_status | 正解参照 | コーパス | PGVector コレクション |
|---|---|---|---|---|---|
| `data/reviewed/plant_v15.jsonl` | 25 | **accepted 25**（人手承認済み） | 30 | `data/chunks_plant/` | `plant_v15` |
| `data/dataset_kg_v3_fixed.jsonl` | 23 | auto_fixed 20 / pending 3 | 26 | `data/chunks_plant/` | `plant_v15` |
| `data/kg_synth_v1/dataset_kg_synth_v1.jsonl` | 77 | pending 77 | 136 | `data/chunks_synth/` | `synth_v1` |

**品質の序列**: `plant_v15` が唯一の人手承認済み。`dataset_kg_v3_fixed` は人手レビューを
通しておらず（`reviewed_by: null`）、後から見つかった正解データの誤りを機械＋LLMで直したもの。
`dataset_kg_synth_v1` は人手未通過だが、正解がテンプレートから決定論的に組み立てられているため
正解の正しさは構造的に保証されている。

### コーパス実測値

| ディレクトリ | 文書数 | チャンク数 | 内容 |
|---|---|---|---|
| `data/chunks_plant/` | 158 | 3,175 | 実在の日本の法令・規程（ボイラー安全規則、容器保安規則、火技解釈等） |
| `data/chunks_synth/` | 25 | 330 | 架空の白波／青灘火力発電所の規程・様式集・点検実績・不適合事例集 |
| `data/chunks_kitei/` | 78 | 531 | HS_RHAT 社内規程類（**仕掛中**、まだどのコレクションにも未投入） |

---

## 2. バックアップ・派生物（評価には使わない）

| ファイル | 説明 |
|---|---|
| `data/dataset_kg_v3_fixed.bak.jsonl` | `dataset_kg_v3_fixed.jsonl` の**正解参照の監査（2026-07-29）を当てる前**の原本。各レコードの `fix_note.ref_audit` に変更理由が入っている |
| `data/kg_synth_v1/dataset_kg_synth_v1.rv.jsonl` | 無印に `rag_verification` / `relation_edges` を足した版。**正解は無印と完全に同一**（77問・参照136件が一致） |
| `data/kg_synth_v1/dataset_kg_synth_v1.grounded.jsonl` | 無印に `answer_grounding`（leave-one-out 判定）を足した版。正解は同一 |
| `data/kg_synth_v1/dataset_kg_synth_v1.csv` | 目視用のCSV書き出し |

---

## 3. 旧世代（`data/archive/loose/` ほか）

`dataset_kg_v1`〜`v8` は積み上げ式で作られている。qa_id の突き合わせで確認した系譜:

| ファイル | 件数 | 積み上げ内容 |
|---|---|---|
| `dataset_kg_v1.jsonl` | 30 | gemma 生成の初回プール（prompt v1.7 / v1.8） |
| `dataset_kg_v2.jsonl` | 25 | v1 から間引き |
| `dataset_kg_v2_clean.jsonl` | 19 | さらにクリーニング |
| **`dataset_kg_v3.jsonl`** | **23** | v2_clean 19 + prompt v1.9 の新規4。**graphrag の plant23 ベンチの原本。ただし23問中6問が偽の正解（後述）** |
| `dataset_kg_v4.jsonl` | 27 | v1.9 の4問を v2.0 の8問に差し替え |
| `dataset_kg_v5.jsonl` | 37 | 手書きQA（Claude Opus 生成）10問が合流 |
| `dataset_kg_v6.jsonl` | 51 | 就業規則コーパス由来が合流 |
| `dataset_kg_v7.jsonl` | 56 | チャンクレベル生成 v3 が合流 |
| `dataset_kg_v8.jsonl` | 68 | 全部入り |
| ├ `dataset_kg_v8_plant.jsonl` | 48 | うち plant コーパス由来。`pred_v8_*` の入力 |
| └ `dataset_kg_v8_kitei.jsonl` | 20 | うち就業規則由来 |

その他の旧世代:

| ファイル | 件数 | 説明 |
|---|---|---|
| `data/reviewed/plant_29.jsonl` | 29 | `plant_v15` の前身。pending 27 / accepted 1 / edited 1。**plant_v15 と qa_id の重複は0（別の問題集）** |
| `data/kg_synth_v1/dataset_kg_synth_v1.rv_k5.jsonl` | 53 | synth77 の前身（53問版）。`pred_synth_kgoff` / `pred_synth_kgon` の入力 |
| `data/v5_manual/v5_manual.final.jsonl` | 10 | 手書きQAの実験 |
| `data/archive/axes_v4/` `data/archive/chunklevel/` `data/archive/shuki_trial5/` | 各2〜12 | 生成方式ごとの試作 |
| `data/archive/review_queue/` | 25 / 29 | `reviewed/` に入る前のレビュー待ち版（review_status が全て pending） |

---

## 4. 仕掛中

| ファイル | 件数 | 状態 |
|---|---|---|
| `data/chunks_kitei/`（78文書 / 531チャンク） | — | HS_RHAT 社内規程の新コーパス（2026-07-28作成）。**PGVector 未投入・対応コレクション無し** |
| `data/raw_kg_kitei_v1/batch_20260728_2346.jsonl` | — | 生成直後の生バッチ |
| `data/filtered_kitei_v1/batch_20260728_2346.jsonl` | 6 | フィルタ通過分。`review_status: pending`（**人手レビュー未実施**） |

このコーパスで評価するには、`_bench/plant/ingest_plant.py --chunks-dir` で PGVector へ投入し、
`build_kg_plant.py --chunks-dir --provenance` でグラフを作る必要がある。

---

## 5. 生成の来歴

| 設問セット | 生成モデル | prompt_version | 生成日 | 方式 |
|---|---|---|---|---|
| `reviewed/plant_v15.jsonl` | `azure/gpt-4.1` | `kg_poc_v1.5` | 2026-06-12 | LLM合成 → **人手レビュー** |
| `reviewed/plant_29.jsonl` | `azure/gpt-4.1` | `kg_poc_v1.3` / `v1.4` | 2026-06-12 | LLM合成（レビューほぼ未了） |
| `dataset_kg_v3(_fixed).jsonl` | `google/gemma-4-26B-A4B-it` | `kg_poc_v1.7` / `v1.8` / `v1.9` | 2026-06-21〜27 | LLM合成のみ |
| `kg_synth_v1/dataset_kg_synth_v1.jsonl` | `template-from-ground-truth-edges` | `synth_qa_v1.0` | 2026-07-08 | **LLM不使用**。正解エッジからテンプレート組み立て |

生成パイプライン（`rageval` CLI）:
`chunk` → `generate` → `filter` → `probe` → `answer-grounding` → `rag-verify` → `review`

- 生成器: `src/rageval/generate.py` + `prompts/generate_kg_poc.md`（3軸でセル分割）
- アンカー選択: `src/rageval/sampling.py`（`ReferenceFollow` / `MultiDocByEmbedding`）
- 人手レビューUI: `src/rageval/review_app.py` / `src/rageval/webapi/` → `data/reviewed/` へ出力
- 合成コーパス生成: `scripts/gen_synth_corpus.py`（決定論的テンプレート、LLM不使用）
- 合成QA生成: `scripts/gen_synth_qa.py`（同上）
- 手書きQA生成（旧）: `scripts/legacy_manual_qa/`

---

## 6. 既知の欠陥と再発防止

### 6-1. 裸の条番号による規程横断の偽結合（生成器のバグ）

`dataset_kg_v3.jsonl` の23問中6問が、**別々の規程にある同じ条番号を取り違えて**
自動生成されていた（2026-07-22 確定）。例: 火技解釈の第46条（燃料電池の気密性能）と、
特定設備検査規則の「様式第3（第46条関係）」を「第46条」の一致だけで結びつけ、
「気密試験の記録は様式第3」という**コーパスに存在しない正解**を合成していた。

機序は `scripts/legacy_manual_qa/enrich_v19_aug.py` の `_enrich_one()` にある:

```python
for ref in a.references:
    if ref in blob:   # 「第5条」という裸トークンの部分文字列一致だけで別文書を結合
        edges.append({...})
```

`src/rageval/sampling.py` の `ReferenceFollow` も同じ照合でアンカーペアを作っている。

**対策**: 条番号は「文書ID + 条番号」の複合キーで扱うこと。graphrag 側は
`graphrag_core/graph/scoping.py` で同じ問題を修正済み（`文書ID#第N条` で一意化）。
データセット生成側は**未対策**なので、新しく作る設問セットでも同じ誤りが出る。

### 6-2. 正解参照にも同じ偽結合が残っていた

2026-07-29 の監査で、`dataset_kg_v3_fixed.jsonl` の正解参照から14件を除去した
（「第5条の規定を準用する」という回答に対し、別規程の「様式第1（第5条関係）」を
条番号一致で結合していたもの）。除去の判断根拠は
`C:/work/RAG/graphrag/_bench/plant/_ref_audit.json`。

### 6-3. 修正スクリプトが残っていない

上記6-1・6-2の修正はアドホック作業で、再現できるコードがどちらのリポジトリにも無い
（`dataset_kg_v1`〜`v8` のマージ処理も `_tmp_merge_v3.py` が削除済みで再現不可）。
残っている痕跡は各レコードの `fix_note` フィールドのみ。

### 6-4. `data/` が git 管理外

README には「`data/reviewed/` — 人手レビュー済み（コミット対象）」とあり、
`.gitignore` にも `!data/reviewed/` の例外があるが、**実際には一度も add されていない**。
全コミット中、データ由来のファイルは0件。ローカルディスク上にしか存在しない。
