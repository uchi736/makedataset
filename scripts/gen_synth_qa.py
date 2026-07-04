# -*- coding: utf-8 -*-
"""合成コーパス用 QA 生成器: 正解エッジからテンプレートで QA を組み立てる.

生成器 (gen_synth_corpus.py) のテーブルを直接 import して使うので、
問い・答え・根拠のすべてが構造的に正しいことが保証される。
rationale.text はチャンク全文の逐語引用。

出力: data/dataset_kg_synth_v1.jsonl

実行: uv run python scripts/gen_synth_qa.py
"""

from __future__ import annotations

import json
import hashlib
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gen_synth_corpus import ARTICLES, FORM_SUBJECTS, GLOSSARY, EQUIPMENT, INCIDENTS, PROCEDURE_BOILER_START  # noqa: E402

SYNTH_DIR = Path("data/chunks_synth")
GEN_MODEL = "template-from-ground-truth-edges"
PROMPT_VERSION = "synth_qa_v1.0"

# ---------- チャンク読み込みと検索 ----------

CHUNKS: dict[str, list[dict]] = {}
for p in sorted(SYNTH_DIR.glob("SYN*.jsonl")):
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            CHUNKS.setdefault(c["doc_id"], []).append(c)


def find_chunk(doc_id: str, needle: str) -> dict:
    for c in CHUNKS[doc_id]:
        if needle in c["text"]:
            return c
    raise KeyError(f"chunk not found: {doc_id} :: {needle[:40]}")


def rat(doc_id: str, needle: str) -> dict:
    c = find_chunk(doc_id, needle)
    return {"doc_id": doc_id, "chunk_id": c["chunk_id"], "page": None, "text": c["text"]}


def qa_item(question, answer, rationale, *, qt, nov, notes):
    return {
        "qa_id": f"synth_{hashlib.sha1(question.encode('utf-8')).hexdigest()[:8]}",
        "question": question,
        "answer": answer,
        "rationale": rationale,
        "category": [], "aspect": [],
        "reasoning_complexity": {
            "multi_step": qt in ("multi_hop", "traceability"),
            "quantitative": False, "negation": qt == "negation_exhaustive",
            "cause_effect": False, "comparison": False,
            "temporal": nov == "procedural_relation",
            "output_type": "list" if qt == "aggregation" else "none",
        },
        "retrieval_difficulty": {
            "multi_doc": len({r["doc_id"] for r in rationale}) >= 2,
            "multi_chunk": len(rationale) >= 2,
            "low_locality": len(rationale) >= 2, "remote_reference": len(rationale) >= 2,
            "doc_volume_large": False, "chunk_size_large": False,
            "abstraction_discrepancy": False, "vocabulary_mismatch": nov == "unknown_term",
        },
        "source_structure": {
            "tables_charts": False, "complex_layout": False, "specific_area_ref": True,
            "logical_nesting": True, "large_enumeration": qt == "aggregation", "redundancy": False,
        },
        "explainability": {"evidence_strictness": "hier-ref" if len(rationale) >= 2 else "no-evidence"},
        "retrieval_level": "Hard" if len(rationale) >= 2 else "Easy",
        "answer_level": "Medium" if len(rationale) >= 2 else "Easy",
        "difficulty_rationale": notes,
        "kg_query_type": qt, "kg_novelty": nov,
        "generation": {"model": GEN_MODEL, "prompt_version": PROMPT_VERSION,
                       "generated_at": datetime.now().isoformat()},
        "filter_scores": {}, "review_status": "pending",
    }


QAS = []

# ============================================================
# A. 様式 lookup (3ホップ: 条文 → 様式マップ → 様式実体)
#    traceability × unknown_relation
# ============================================================

ART_PARAMS = {a: prm for a, _, kind, prm in ARTICLES if kind == "form_ref"}

def form_lookup_qas(plant: str, reg_doc: str, forms_doc: str, article_offset: int, form_offset: int, limit=None):
    out = []
    n = 6 + form_offset
    for art in sorted(FORM_SUBJECTS):
        no = n; n += 1
        name, dest, keep = FORM_SUBJECTS[art]
        prm = ART_PARAMS[art]
        a = art + article_offset
        q = (f"{plant}火力発電所で、{prm['subject']}を計画しています。"
             f"実施方法の要求事項と、{prm['record']}の記録に使う様式・提出先・保存期間を教えてください。")
        ans = (f"{prm['action']}こととされています。"
               f"記録は様式第{no}号により作成し、{dest}に提出します。保存期間は{keep}です。")
        r = [
            rat(reg_doc, f"第{a}条 {prm['subject']}は"),
            rat(forms_doc, f"様式第{no}号 記載事項"),
        ]
        out.append(qa_item(q, ans, r, qt="traceability", nov="unknown_relation",
                           notes=f"様式lookup 3ホップ: 第{a}条→様式第{no}号→実体。{plant}"))
    return out[:limit] if limit else out

QAS += form_lookup_qas("白波", "SYN01_白波_設備保全管理規程", "SYN02_白波_検査様式集", 0, 0)
# 青灘は発電所スコープの取り違えを試す 3 問だけ
QAS += form_lookup_qas("青灘", "SYN09_青灘_設備保全管理規程", "SYN10_青灘_検査様式集", 2, 10, limit=3)

# ============================================================
# B. 名寄せ (unknown_term): 旧称・略称で聞いて正式情報に辿る
# ============================================================

G_DOC = "SYN03_白波_用語集"
R_DOC = "SYN04_白波_機器台帳"
N_DOC = "SYN05_白波_不適合事例集"
REG = "SYN01_白波_設備保全管理規程"

# B-1: 旧称 → 正式名称+機器ID (用語集+台帳)
for official, abbr, old in GLOSSARY[:5]:
    eq = next((e for e in EQUIPMENT if e[1] == official), None)
    if not eq:
        continue
    q = f"以前の文書で「{old}」と呼ばれていた設備の、現在の正式名称と機器ID、所属系統を教えてください。"
    ans = f"「{old}」は旧称であり、現行文書では使用しません。正式名称は{official}、機器IDは{eq[0]}、所属系統は{eq[2]}です。"
    r = [rat(G_DOC, f"旧称: {old}"), rat(R_DOC, f"機器ID: {eq[0]}。")]
    QAS.append(qa_item(q, ans, r, qt="multi_hop", nov="unknown_term",
                       notes=f"名寄せ: {old}→{official}→台帳"))

# B-2: 旧称で不適合を聞く (用語集+事例集)
ALIAS_INCIDENTS = [
    ("三号瓦斯機", "第3ガスタービン発電設備", "NCR-2024-01"),
    ("一号缶水ポンプ", "第1給水ポンプ", "NCR-2023-02"),
    ("非常用発電機", "非常用ディーゼル発電設備", "NCR-2024-02"),
]
for old, official, ncr in ALIAS_INCIDENTS:
    inc = next(i for i in INCIDENTS if i[0] == ncr)
    q = f"{official}で過去に発生した不適合({ncr})の事象と根本原因を教えてください。"
    ans = f"この設備は不適合事例集では旧称・略称の「{inc[1]}」で記載されています。事象は「{inc[2]}」で、原因は{inc[3]}でした。"
    r = [rat(G_DOC, f"旧称: {old}"), rat(N_DOC, f"{ncr}。")]
    QAS.append(qa_item(q, ans, r, qt="multi_hop", nov="unknown_term",
                       notes=f"名寄せ経由の不適合参照: 事例集は「{inc[1]}」表記"))

# B-3: 旧称で規程を聞く (用語集+条文)
q = "「ガスリークセンサ」の校正はどのような頻度で行い、警報設定値はどう定められていますか。"
ans = ("「ガスリークセンサ」は燃料ガス漏えい検知装置の旧称です。"
       "校正は検知器ごとに標準ガスにより6か月に1回実施し、警報設定値は爆発下限界の4分の1とします。")
r = [rat(G_DOC, "旧称: ガスリークセンサ"), rat(REG, "第27条 燃料ガス漏えい検知装置の機能点検は")]
QAS.append(qa_item(q, ans, r, qt="traceability", nov="unknown_term", notes="旧称→正式名→条文"))

# ============================================================
# C. 因果 (multi_hop × unknown_relation): 不適合 → 原因 → 関連規程の要求
# ============================================================

for ncr, equip, event, cause, art, corrective in INCIDENTS[:6]:
    prm = ART_PARAMS.get(art)
    if prm:
        req = prm["action"]
        needle = f"第{art}条 {prm['subject']}は"
    else:
        plain = {a: p for a, _, k, p in ARTICLES if k in ("plain", "appendix")}
        prm2 = plain[art]
        req = prm2.get("text", prm2.get("subject", ""))[:60]
        needle = f"第{art}条 "
    q = (f"白波火力発電所で「{event}」という不適合が発生しました。"
         f"根本原因は何でしたか。また、原因に関連する保全規程ではどのような実施要求が定められていますか。")
    ans = f"根本原因は、{cause}です。関連する規程の要求は「{req}」です。"
    r = [rat(N_DOC, f"{ncr}。"), rat(REG, needle)]
    QAS.append(qa_item(q, ans, r, qt="multi_hop", nov="unknown_relation",
                       notes=f"因果: {ncr} 原因→第{art}条の要求"))

# ============================================================
# D. 手順 (procedural_relation)
# ============================================================

P_DOC = "SYN06_白波_ボイラー起動手順書"

# D-1: 前提工程の連鎖
for no, name, prereq, form_art in PROCEDURE_BOILER_START:
    if not prereq:
        continue
    pre = next(s for s in PROCEDURE_BOILER_START if s[0] == prereq)
    q = f"ボイラー起動作業で「{name}」に着手する直前に、完了確認が必要な工程は何ですか。"
    ans = f"工程{prereq}「{pre[1]}」の完了確認後でなければ、工程{no}「{name}」に着手してはなりません。"
    r = [rat(P_DOC, f"工程{no}: {name}"), rat(P_DOC, f"工程{prereq}: {pre[1]}")]
    QAS.append(qa_item(q, ans, r, qt="traceability", nov="procedural_relation",
                       notes=f"工程{prereq}→工程{no} の順序依存"))

# D-2: 工程 → 条文 → 様式 の 3 ホップ (step_article)
for no, name, prereq, form_art in PROCEDURE_BOILER_START:
    if not form_art:
        continue
    fnames = {a: FORM_SUBJECTS[a] for a in FORM_SUBJECTS}
    if form_art in fnames:
        fidx = sorted(FORM_SUBJECTS).index(form_art)
        fno = 6 + fidx
        fname, dest, keep = FORM_SUBJECTS[form_art]
        q = (f"ボイラー起動作業の「{name}」は、どの工程の完了後に実施し、"
             f"確認結果はどの様式に記録して誰に提出しますか。")
        pre2 = next(sname for sno, sname, _, _ in PROCEDURE_BOILER_START if sno == prereq)
        ans = (f"工程{prereq}の完了確認後に工程{no}として実施します。"
               f"確認結果は様式第{fno}号に記録し、{dest}に提出します。")
        r = [
            rat(P_DOC, f"工程{no}: {name}"),
            rat("SYN02_白波_検査様式集", f"様式第{fno}号 記載事項"),
        ]
        QAS.append(qa_item(q, ans, r, qt="traceability", nov="procedural_relation",
                           notes=f"工程{no}→第{form_art}条→様式第{fno}号 の3ホップ"))

# ============================================================
# E. 集約 (aggregation)
# ============================================================

# E-1: 保存期間 10 年の様式をすべて (様式実体 3 チャンク)
ten_year = [(6 + sorted(FORM_SUBJECTS).index(a), FORM_SUBJECTS[a][0]) for a in sorted(FORM_SUBJECTS) if FORM_SUBJECTS[a][2] == "10年"]
q = "白波火力発電所の記録様式のうち、保存期間が10年と定められているものをすべて挙げてください。"
ans = "、".join(f"様式第{no}号" for no, name in ten_year) + " の3様式です。"
r = [rat("SYN02_白波_検査様式集", f"様式第{no}号 記載事項") for no, name in ten_year]
QAS.append(qa_item(q, ans, r, qt="aggregation", nov="unknown_relation", notes="保存期間10年の様式の集約 (3チャンク)"))

# E-2: 汽力系統の機器一覧 (台帳 4 チャンク)
steam = [e for e in EQUIPMENT if e[2] == "汽力系統"]
q = "白波火力発電所の機器台帳で、汽力系統に属する機器の機器IDをすべて挙げてください。"
ans = "、".join(e[0] for e in steam) + " の4機器です。"
r = [rat(R_DOC, f"機器ID: {e[0]}。") for e in steam]
QAS.append(qa_item(q, ans, r, qt="aggregation", nov="unknown_relation", notes="汽力系統の機器集約 (4チャンク)"))

# E-3: 提出先が統括責任者の様式 (実体 3 チャンク)
tousei = [(6 + sorted(FORM_SUBJECTS).index(a), FORM_SUBJECTS[a][0]) for a in sorted(FORM_SUBJECTS) if FORM_SUBJECTS[a][1] == "統括責任者"]
q = "白波火力発電所の記録様式のうち、提出先が統括責任者と定められているものをすべて挙げてください。"
ans = "、".join(f"様式第{no}号" for no, name in tousei) + " です。"
r = [rat("SYN02_白波_検査様式集", f"様式第{no}号 記載事項") for no, name in tousei]
QAS.append(qa_item(q, ans, r, qt="aggregation", nov="unknown_relation", notes="提出先=統括責任者の様式の集約"))

# ============================================================
# F. 否定・網羅 (negation_exhaustive)
# ============================================================

# F-1: 旧様式は使えるか (SUPERSEDES)
q = ("現在、ボイラー水圧耐圧試験の記録を旧様式第3号で作成してもよいですか。"
     "使えない場合、現行のどの様式を使うべきかも教えてください。")
ans = ("使えません。旧様式第3号は廃止された旧規程に基づくもので、"
       "現行規程では様式番号が変更されています。"
       "現行はボイラー水圧耐圧試験記録の様式第6号を使用します。")
r = [rat("SYN08_白波_旧版規程_廃止", "旧様式第3号"),
     rat(REG, "第6条 ボイラー設備の耐圧部分の水圧耐圧試験は"),
     rat(REG, "様式第6号(第6条関係)")]
QAS.append(qa_item(q, ans, r, qt="negation_exhaustive", nov="unknown_relation", notes="改廃 (SUPERSEDES): 旧様式の使用可否"))

# F-2: 使ってはいけない呼称 (用語集)
q = "「排煙脱硝設備」という呼称を現行文書で使用してもよいですか。正式にはどう呼ぶべきですか。"
ans = "使用できません。「排煙脱硝設備」は旧称であり、現行文書では正式名称の脱硝装置(社内略称 SCR-1)を使用します。"
r = [rat(G_DOC, "旧称: 排煙脱硝設備")]
QAS.append(qa_item(q, ans, r, qt="negation_exhaustive", nov="unknown_term", notes="旧称の使用禁止 (単一)"))

# F-3: 規程の適用除外
q = "白波火力発電所の設備保全管理規程が適用されない設備はありますか。"
ans = "あります。建屋及び構内道路等の土木建築設備には適用されません。適用対象は機器台帳に登録された発電用設備です。"
r = [rat(REG, "第2条 この規程は")]
QAS.append(qa_item(q, ans, r, qt="negation_exhaustive", nov="unknown_relation", notes="適用除外 (単一)"))

# F-4: 準用の読み替え (ガスタービンは 1.4倍でない)
q = ("ガスタービン車室の水圧耐圧試験は、ボイラーと同じ試験圧力・保持時間で実施してよいですか。"
     "異なる場合は正しい条件を教えてください。")
ans = ("同じではありません。ガスタービン車室の水圧耐圧試験はボイラーの規定を準用しますが、"
       "「1.4倍の圧力で30分間」を「1.2倍の圧力で10分間」と読み替えて実施します。")
r = [rat(REG, "第26条 ガスタービン車室の水圧耐圧試験については"),
     rat(REG, "第6条 ボイラー設備の耐圧部分の水圧耐圧試験は")]
QAS.append(qa_item(q, ans, r, qt="negation_exhaustive", nov="unknown_relation", notes="準用+読み替え: 1.4倍/30分→1.2倍/10分"))

# ============================================================
# G. 単一ファクト (対照群)
# ============================================================

SINGLES = [
    ("蒸気タービンの軸振動の警報設定値はいくつですか。",
     "軸振動の警報設定値は125マイクロメートルです (軸受メタル温度は95度)。",
     REG, "第18条 蒸気タービンの軸受温度及び軸振動は"),
    ("脱硝装置の性能試験で維持を確認すべき脱硝効率は何パーセント以上ですか。",
     "脱硝効率80パーセント以上を維持していることを確認します。性能試験は毎年1回実施します。",
     REG, "第35条 脱硝装置の脱硝効率の測定は"),
    ("非常用ディーゼル発電設備の始動所要時間の基準は何秒以内ですか。",
     "始動所要時間は40秒以内であることを確認します。",
     REG, "第41条 非常用ディーゼル発電設備の始動試験は"),
    ("ボイラー給水のpHの管理基準値の範囲を教えてください。",
     "ボイラー給水のpHは8.5以上9.6以下です (缶水は9.0以上9.8以下)。",
     "SYN07_白波_別表集", "別表第1(水質管理基準値表)"),
    ("フェライト系耐熱鋼を過熱器管に使用する場合の使用温度上限は何度ですか。",
     "使用温度上限は580度です。",
     REG, "第5条 ボイラー設備の耐圧部分に使用する材料は"),
]
for q, ans, doc, needle in SINGLES:
    QAS.append(qa_item(q, ans, [rat(doc, needle)], qt="single_fact", nov="unknown_relation", notes="単一ファクト対照"))


def main():
    out = Path("data/dataset_kg_synth_v1.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for qa in QAS:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {len(QAS)} QAs to {out}")
    print("クエリ型:", dict(Counter(q['kg_query_type'] for q in QAS)))
    print("未知性:", dict(Counter(q['kg_novelty'] for q in QAS)))
    print("根拠単位数:", dict(Counter(len(q['rationale']) for q in QAS)))


if __name__ == "__main__":
    main()
