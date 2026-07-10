"""チャンク単位本丸の書き増し第 3 弾 (Claude 手書き、7 件)。

- 就業規則 2 件: 産前産後×解雇制限 (gap 73)、賞与×年休不利益 (gap 66)
- plant 3 件: レジューサ準用 (gap 11)、保安区画準用 (gap 41)、離隔距離×別表第5 (gap 354)
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path

PROMPT_VERSION = "manual_claude_chunklevel_v3.0"
GEN_MODEL = "claude-fable-5-via-claude-code"


def _qa_id(question: str) -> str:
    return f"manual3_{hashlib.sha1(question.encode('utf-8')).hexdigest()[:8]}"


def _qa(question, answer, rationale, *, kg_query_type, notes):
    return {
        "qa_id": _qa_id(question),
        "question": question,
        "answer": answer,
        "rationale": rationale,
        "category": [],
        "aspect": [],
        "reasoning_complexity": {
            "multi_step": True, "quantitative": False, "negation": False,
            "cause_effect": False, "comparison": False, "temporal": False,
            "output_type": "none",
        },
        "retrieval_difficulty": {
            "multi_doc": False, "multi_chunk": True, "low_locality": True,
            "remote_reference": True, "doc_volume_large": True, "chunk_size_large": False,
            "abstraction_discrepancy": False, "vocabulary_mismatch": False,
        },
        "source_structure": {
            "tables_charts": False, "complex_layout": False, "specific_area_ref": True,
            "logical_nesting": True, "large_enumeration": False, "redundancy": False,
        },
        "explainability": {"evidence_strictness": "hier-ref"},
        "retrieval_level": "Hard",
        "answer_level": "Medium",
        "difficulty_rationale": notes,
        "kg_query_type": kg_query_type,
        "kg_novelty": "unknown_relation",
        "generation": {
            "model": GEN_MODEL,
            "prompt_version": PROMPT_VERSION,
            "generated_at": datetime.now().isoformat(),
        },
        "filter_scores": {},
        "review_status": "pending",
    }


S = "モデル就業規則"
P = "09_kagi_kaishaku_240315"

QAS = [
    # S6. 産後就業の例外 (c0121) × 解雇制限期間 (c0194)、位置差 73
    _qa(
        question=(
            "産後8週間を経過していない女性労働者は原則として就業させられないと理解していますが、"
            "例外的に就業させることができる条件はありますか。"
            "また、産前産後の休業に関連して解雇が制限されるのは、いつまでの期間ですか。"
        ),
        answer=(
            "産後6週間を経過した女性労働者から請求があったときは、医師が支障がないと認めた業務に"
            "就かせることができます (労基法第65条第2項)。"
            "解雇については、産前産後の休業の期間とその後30日間は解雇が禁止されています (労基法第19条)。"
        ),
        rationale=[
            {"doc_id": S, "chunk_id": f"{S}__c0121", "page": None,
             "text": "2 産後8週間を経過しない女性労働者を就業させてはいけません。ただし、産後6週間を 経過した女性労働者から請求があったときは、医師が支障がないと認めた業務には就か せることができます(労基法第65条第2項)。"},
            {"doc_id": S, "chunk_id": f"{S}__c0194", "page": None,
             "text": "3 労働者の業務上の負傷、疾病による休業期間とその後30日間及び産前産後 の休業の期間(産前6週間(多胎妊娠の場合は14週間)以内又は産後8週間以 内の女性が休業する期間)とその後30日間の解雇(労基法第19条)。"},
        ],
        kg_query_type="multi_hop",
        notes="産後就業の医師例外(c0121)と解雇制限30日(c0194)、位置差73",
    ),

    # S7. 賞与の支給対象 (c0182) × 年休の不利益取扱い禁止 (c0116)、位置差 66
    _qa(
        question=(
            "賞与の額を算定する際に、年次有給休暇を取得した日を欠勤として扱うことは許されますか。"
            "また、賞与はどのような労働者に対して支給するものと定められていますか。"
        ),
        answer=(
            "年次有給休暇を取得した労働者に対して、賞与の額の算定に際して年次有給休暇取得日を"
            "欠勤として取扱うなどの不利益な取扱いをしてはいけません (労基法附則第136条)。"
            "賞与は、原則として算定対象期間に在籍した労働者に対し、会社の業績等を勘案して支給します。"
        ),
        rationale=[
            {"doc_id": S, "chunk_id": f"{S}__c0116", "page": None,
             "text": "12 年次有給休暇を取得した労働者に対して、賃金の減額や精皆勤手当、賞与の額の算定 に際しての年次有給休暇取得日を欠勤として取扱う等の不利益な取扱いをしてはいけま せん(労基法附則第136条)。"},
            {"doc_id": S, "chunk_id": f"{S}__c0182", "page": None,
             "text": "第50条 賞与は、原則として、下記の算定対象期間に在籍した労働者に対し、会社 の業績等を勘案して下記の支給日に支給する。ただし、会社の業績の著しい低下そ の他やむを得ない事由により、支給時期を延期し、又は支給しないことがある。"},
        ],
        kg_query_type="multi_hop",
        notes="不利益取扱い禁止(c0116)と賞与支給対象(c0182)、位置差66",
    ),

    # P5. レジューサ部分の厚さ計算 (c0016 × c0005)、位置差 11
    _qa(
        question=(
            "ボイラーの管のうちレジューサの部分の厚さを計算します。"
            "この部分に準用される円すい形の規定では、胴の形状にどのような条件がありますか。"
            "また、水管や過熱管などにおける付け代の特例もあわせて教えてください。"
        ),
        answer=(
            "レジューサの部分には容器の胴の円すい形に係る規定が準用されます。"
            "円すい形の胴は図1から図5までに示す形であることが必要で、円筒形又は同軸円すい形の胴では、"
            "軸に垂直な同一断面における最大内径と最小内径との差が当該断面の基準内径の1%以下で"
            "なければなりません。"
            "水管、過熱管、再熱管、節炭器管 (鋳鉄管を除く)、下降管、上昇管、管寄せ連絡管等については、"
            "付け代を管の外径の0.005倍とします。"
        ),
        rationale=[
            {"doc_id": P, "chunk_id": f"{P}__c0016", "page": None,
             "text": "２ 管のうちレジューサの部分にあっては、第６条第２項の規定中円すい形に係る部分を 準用する。ただし、水管、過熱管、再熱管、節炭器管（鋳鉄管を使用するものを除く。）、 下降管、上昇管、管寄せ連絡管並びにボイラーに最も近い給水止め弁からボイラーに最 も近い蒸気止め弁までの部分の蒸気管及び給水管にあっては付け代を管の外径の0.005 倍とする。"},
            {"doc_id": P, "chunk_id": f"{P}__c0005", "page": None,
             "text": "一 円筒形又は図１から図５までに示す円すい形（ボイラー等及び独立節炭器に係る容 4 器にあっては、図１及び図２に示すものに限る。）であること。 二 円筒形（板を曲げて製作する物に限る。第５９条第１項第二号において同じ。）又は 同軸円すい形の胴にあっては、その軸に垂直な同一断面における最大内径と最小内径 との差は、当該断面の基準内径の1 %以下であること。"},
        ],
        kg_query_type="multi_hop",
        notes="レジューサ準用+付け代0.005倍(c0016)と円すい形の形状条件(c0005)、位置差11",
    ),

    # P6. ガス化炉の保安区画 (c0095 × c0054)、位置差 41
    _qa(
        question=(
            "ガス化炉設備について「保安上適切な区画」を設けることを検討しています。"
            "準用される規定では、保安区画の面積と、1つの保安区画内の燃焼熱量の合計に"
            "どのような上限が定められていますか。"
        ),
        answer=(
            "ガス化炉設備の保安上適切な区画は、第86条により液化ガス設備の保安区画規定のうち"
            "第52条第2項第一号、第3項及び第4項を準用します。"
            "具体的には、特定発電所に属する設備では、保安区画の面積が20,000 m2以下であること、"
            "1の保安区画内の液化ガス設備の燃焼熱量の数値の合計が6.0×10の8乗以下であることが必要です。"
        ),
        rationale=[
            {"doc_id": P, "chunk_id": f"{P}__c0095", "page": None,
             "text": "（保安区画） 第８６条 省令第５６条に規定する「保安上適切な区画」とは、第５２条第２項第一号、 第３項及び第４項の規定を準用する。"},
            {"doc_id": P, "chunk_id": f"{P}__c0054", "page": None,
             "text": "２ 省令第３８条に規定する「保安上適切な区画」とは、次の各号に掲げるものをいう。 一 特定発電所に属する液化ガス設備にあっては、次に掲げるもの イ 第３項に定める方法により算出した保安区画の面積が、20,000 m2 以下であるもの ロ １の保安区画内の液化ガス設備の燃焼熱量の数値の合計が、6.0×108 以"},
        ],
        kg_query_type="traceability",
        notes="ガス化炉の準用接続(c0095 第86条)と保安区画の実体(c0054 第52条)、位置差41",
    ),

    # P7. 離隔距離の計算式 (c0095) × 別表第5 の K 値 (c0449)、位置差 354
    _qa(
        question=(
            "ガス化炉設備の離隔距離を計算しています。"
            "距離の計算式と、算出した値が50 mを下回った場合の扱いを教えてください。"
            "また、メタンを通ずる設備で常用の温度が-80以上の場合、計算に使う係数はどの値になりますか。"
        ),
        answer=(
            "離隔距離Lは L = 0.576 × KW の立方根 (KにWを乗じた値の3乗根に0.576を乗じた値) で算出し、"
            "50 m未満の場合は50 mとします。"
            "Kはガスの種類及び常用の温度の区分に応じて別表第5に定める値で、"
            "表中のkの値に1,000を乗じて得た値です。メタンで常用の温度が-80以上の場合はk=714です。"
        ),
        rationale=[
            {"doc_id": P, "chunk_id": f"{P}__c0095", "page": None,
             "text": "次の計算式より算出した値以上とし、50 m 未満の場合にあっては、50 67 m とする。ただし、ガス化炉設備に2 以上のガスがある場合においては、それぞれの ガスについてK にW を乗じた値を算出し、その数値の合計により、L を算出するもの とする。 3 576 .0 KW L  = L は、離隔距離（m を単位とする。） K は、ガスの種類及び常用の温度の区分に応じて別表第５に定める値"},
            {"doc_id": P, "chunk_id": f"{P}__c0449", "page": None,
             "text": "別表第５　Kの値　（第50条、第85条関係） Kの値は次に掲げるガスの種類及び常用の温度の区分に応じ表中のkの値に1,000を乗じて得た値 1 メタン 常用の温度 -110未満 -110以上　-80未満 -80以上 - - - k 143 357 714"},
        ],
        kg_query_type="traceability",
        notes="計算式と50m下限(c0095 第85条)とK値表(c0449 別表第5)、位置差354の遠隔参照",
    ),
]


def main():
    out_path = Path("data/dataset_chunklevel_v3.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for qa in QAS:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    print(f"wrote {len(QAS)} QAs to {out_path}")


if __name__ == "__main__":
    main()
