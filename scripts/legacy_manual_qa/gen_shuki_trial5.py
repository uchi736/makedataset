"""就業規則コーパスでの試作 5 件 (Claude 手書き、チャンク単位 leave-one-out 検証用).

rationale.doc_id にチャンク ID を入れることで、既存 AnswerGrounding の
doc 単位 leave-one-out がそのままチャンク単位判定として機能する。
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path

PROMPT_VERSION = "manual_claude_shuki_v1.0"
GEN_MODEL = "claude-fable-5-via-claude-code"


def _qa_id(question: str) -> str:
    return f"shuki_{hashlib.sha1(question.encode('utf-8')).hexdigest()[:8]}"


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


QAS = [
    # --- 1. 復職の原則 (c0037) × 退職時証明書 (c0190)、位置差 153 ---
    _qa(
        question=(
            "休職していた労働者の休職事由が消滅したので復職させたいのですが、"
            "復帰先の原則と、元の職務への復帰が難しい場合の扱いを教えてください。"
            "また、その後この労働者が退職する際、本人から請求があった場合に"
            "交付する証明書へ記載すべき事項もあわせて知りたいです。"
        ),
        answer=(
            "休職事由が消滅したときは、原則として元の職務に復帰させます。"
            "元の職務への復帰が困難又は不適当な場合には、他の職務に就かせることがあります。"
            "退職時に本人から請求があった場合は、使用期間、業務の種類、地位、賃金又は退職の事由を"
            "記載した証明書を遅滞なく交付します。"
        ),
        rationale=[
            {"doc_id": "モデル就業規則__c0037", "page": None,
             "text": "休職期間中に休職事由が消滅したときは、原則として元の職務に復帰させる。ただ し、元の職務に復帰させることが困難又は不適当な場合には、他の職務に就かせるこ とがある。"},
            {"doc_id": "モデル就業規則__c0190", "page": None,
             "text": "労働者が退職し、又は解雇された場合、その請求に基づき、使用期間、業務の種 類、地位、賃金又は退職の事由を記載した証明書を遅滞なく交付する。"},
        ],
        kg_query_type="multi_hop",
        notes="復職原則(c0037)と証明書事項(c0190)、位置差153の遠隔参照。両チャンク必須",
    ),

    # --- 2. 民法627条14日 (c0191) × 休職期間満了退職 (c0190) ---
    _qa(
        question=(
            "期間の定めのない雇用の労働者が退職を願い出ましたが、会社として承認していません。"
            "法律上いつ退職となりますか。また、休職中の労働者については、"
            "どのような場合に退職の扱いになるのかも教えてください。"
        ),
        answer=(
            "会社の承認がなくても、民法第627条第1項の規定により、退職の申出をした日から起算して"
            "原則として14日を経過したときに退職となります。"
            "休職中の労働者については、第9条に定める休職期間が満了してもなお休職事由が"
            "消滅しないときに退職となります。"
        ),
        rationale=[
            {"doc_id": "モデル就業規則__c0191", "page": None,
             "text": "会社の承認がなくても、民法(明治29年法律第89号)の規定により退職の申出をした 日から起算して原則として14日を経過したときは、退職となります(民法第627条第 1項)。"},
            {"doc_id": "モデル就業規則__c0190", "page": None,
             "text": "3 第9条に定める休職期間が満了し、なお休職事由が消滅しないとき"},
        ],
        kg_query_type="multi_hop",
        notes="法定退職(民法627/14日、c0191)と規程上の休職満了退職(c0190)。字数29字ぎりぎり→要確認",
    ),

    # --- 3. 年休 20日 (c0109 表) × 基準日統一の切り上げ (c0113) ---
    _qa(
        question=(
            "勤続6年6か月以上の通常の労働者に付与する年次有給休暇の日数を教えてください。"
            "また、年次有給休暇の基準日を毎年4月1日に統一した場合、"
            "1月1日に採用した労働者への初年度の付与はどう扱えばよいですか。"
        ),
        answer=(
            "勤続6年6か月以上の労働者への年次有給休暇の付与日数は20日です。"
            "基準日を4月1日に統一した場合、勤務期間の切捨ては認められず常に切り上げるため、"
            "1月1日に採用した労働者にも、3か月間継続勤務した後の4月1日の時点、"
            "すなわち法定より3か月前倒しで初年度の年次有給休暇を付与しなければなりません。"
        ),
        rationale=[
            {"doc_id": "モデル就業規則__c0109", "page": None,
             "text": "その後1年間継続勤務するごとに、当該 1年間において所定労働日の8割以上出勤した労働者に対しては、下の表のとおり 勤続期間に応じた日数の年次有給休暇を与える。 <table> <tr> <th>勤続期間</th> <th>6か月</th> <th>1年 6か月</th> <th>2年 6か月</th> <th>3年 6か月</th> <th>4年 6か月</th> <th>5年 6か月</th> <th>6年 6か月以上</th> </tr> <tr> <td>付与日数</td> <td>10 日</td> <td>11 日</td> <td>12 日</td> <td>14 日</td> <td>16 日</td> <td>18 日</td> <td>20 日</td> </tr> </table>"},
            {"doc_id": "モデル就業規則__c0113", "page": None,
             "text": "年次有給休暇の基準日を個々の労働者の採用日に関係なく統一的に定めることもでき ます。この場合、勤務期間の切捨ては認められず、常に切り上げなければなりません。例 えば、基準日を4月1日に統一した場合には、その年の1月1日に採用した労働者につい ても3か月間継続勤務した後の4月1日の時点、すなわち法定の場合よりも3か月間前 倒しで初年度の年次有給休暇を付与しなければなりません。"},
        ],
        kg_query_type="traceability",
        notes="付与日数表(c0109)と基準日統一運用(c0113)。表の20日はc0109のみ、切り上げ・前倒しはc0113のみ",
    ),

    # --- 4. 非常時払いの事由 (c0178) × 法的根拠 労基法25条 (c0179) ---
    _qa(
        question=(
            "労働者から、賃金支払日前に既往の労働に対する賃金を支払ってほしいと請求がありました。"
            "この非常時払いに応じる必要があるのはどのような場合ですか。"
            "また、この制度の法的根拠となる労働基準法の条文もあわせて教えてください。"
        ),
        answer=(
            "やむを得ない事由によって1週間以上帰郷する場合、結婚又は死亡の場合、"
            "出産、疾病又は災害の場合、退職又は解雇により離職した場合に、"
            "労働者からの請求に基づき賃金支払日前であっても既往の労働に対する賃金を支払います。"
            "この非常時払いの法的根拠は労基法第25条です。"
        ),
        rationale=[
            {"doc_id": "モデル就業規則__c0178", "page": None,
             "text": "第48条 労働者又はその収入によって生計を維持する者が、次のいずれかの場合に 該当し、そのために労働者から請求があったときは、賃金支払日前であっても、既往 の労働に対する賃金を支払う。 1 やむを得ない事由によって1週間以上帰郷する場合 2 結婚又は死亡の場合 3 出産、疾病又は災害の場合 4 退職又は解雇により離職した場合"},
            {"doc_id": "モデル就業規則__c0179", "page": None,
             "text": "本規程例は、労働者又はその収入によって生計を維持する者に出産、疾病、災害等の臨 時の出費を必要とする事情が生じた場合に、当該労働者は賃金支払日前であっても既往 の労働に対する賃金の払いを請求できることとしたものです(労基法第25条)。"},
        ],
        kg_query_type="traceability",
        notes="事由リスト(c0178条文)と法的根拠・労基法25条(c0179解説)。条文本文は労基法25条に言及しない",
    ),

    # --- 5. 業務外傷病休職の帰結 (c0037) × 業務上傷病の解雇制限 (c0191)、位置差 154 ---
    _qa(
        question=(
            "長期間就業できない労働者の扱いについて、業務外の傷病で休職している場合と、"
            "業務上の負傷や疾病で療養している場合とで、退職・解雇の扱いがどう違うのか教えてください。"
        ),
        answer=(
            "業務外の傷病による休職の場合は、休職期間が満了してもなお傷病が治癒せず就業が困難なときに、"
            "休職期間の満了をもって退職となります。"
            "一方、業務上の負傷又は疾病による療養の場合は、療養の開始後3年を経過しても治らず、"
            "労働者が傷病補償年金を受けているとき又は受けることとなったとき"
            "（会社が打ち切り補償を支払ったときを含む）に解雇されることがあります。"
        ),
        rationale=[
            {"doc_id": "モデル就業規則__c0037", "page": None,
             "text": "3 第1項第1号により休職し、休職期間が満了してもなお傷病が治癒せず就業が困 難な場合は、休職期間の満了をもって退職とする。"},
            {"doc_id": "モデル就業規則__c0191", "page": None,
             "text": "3 業務上の負傷又は疾病による療養の開始後3年を経過しても当該負傷又は疾病 が治らない場合であって、労働者が傷病補償年金を受けているとき又は受けるこ ととなったとき(会社が打ち切り補償を支払ったときを含む。)。"},
        ],
        kg_query_type="multi_hop",
        notes="業務外(休職満了退職、c0037)と業務上(3年経過・傷病補償年金で解雇、c0191)の対比。位置差154",
    ),
]


def main():
    out_path = Path("data/dataset_shuki_trial5.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for qa in QAS:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    print(f"wrote {len(QAS)} QAs to {out_path}")


if __name__ == "__main__":
    main()
