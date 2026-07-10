"""観点補完の書き増し 12 件 (Claude 手書き)。

欠けていた観点を埋める:
- クエリ型: single_fact / aggregation / negation_exhaustive を新規カバー
- 未知性: unknown_term / procedural_relation を新規カバー
- 規定 8 件 (12→20 件へ)、plant 4 件

単一チャンクで完結する問い (対照群设計) も含む — その場合 rationale は 1 単位。
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path

PROMPT_VERSION = "manual_claude_axes_v4.0"
GEN_MODEL = "claude-fable-5-via-claude-code"


def _qa_id(question: str) -> str:
    return f"manual4_{hashlib.sha1(question.encode('utf-8')).hexdigest()[:8]}"


def _qa(question, answer, rationale, *, kg_query_type, kg_novelty, notes):
    return {
        "qa_id": _qa_id(question),
        "question": question,
        "answer": answer,
        "rationale": rationale,
        "category": [],
        "aspect": [],
        "reasoning_complexity": {
            "multi_step": kg_query_type in ("multi_hop", "traceability"),
            "quantitative": False,
            "negation": kg_query_type == "negation_exhaustive",
            "cause_effect": False, "comparison": False,
            "temporal": kg_novelty == "procedural_relation",
            "output_type": "list" if kg_query_type == "aggregation" else "none",
        },
        "retrieval_difficulty": {
            "multi_doc": False, "multi_chunk": len(rationale) >= 2, "low_locality": len(rationale) >= 2,
            "remote_reference": len(rationale) >= 2, "doc_volume_large": True, "chunk_size_large": False,
            "abstraction_discrepancy": False, "vocabulary_mismatch": kg_novelty == "unknown_term",
        },
        "source_structure": {
            "tables_charts": False, "complex_layout": False, "specific_area_ref": True,
            "logical_nesting": True, "large_enumeration": kg_query_type == "aggregation", "redundancy": False,
        },
        "explainability": {"evidence_strictness": "hier-ref" if len(rationale) >= 2 else "no-evidence"},
        "retrieval_level": "Hard" if len(rationale) >= 2 else "Easy",
        "answer_level": "Medium" if len(rationale) >= 2 else "Easy",
        "difficulty_rationale": notes,
        "kg_query_type": kg_query_type,
        "kg_novelty": kg_novelty,
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
Y = "34_yousetu_kaisyaku"

QAS = [
    # ===== 規定 (就業規則) 8 件 =====

    # K1. negation_exhaustive × unknown_relation: 解雇禁止の列挙 × 即時解雇の手続き (gap 39)
    _qa(
        question=(
            "労働者を解雇することが法律で禁止されているのはどのような場合ですか。"
            "また、それらに該当せず懲戒解雇する場合でも、解雇予告手当を支払わずに"
            "即時解雇するには何が必要ですか。"
        ),
        answer=(
            "解雇が禁止されるのは、労働者の国籍・信条・社会的身分を理由とする解雇 (労基法第3条)、"
            "性別を理由とする解雇 (均等法第6条)、業務上の負傷・疾病による休業期間とその後30日間"
            "及び産前産後の休業期間とその後30日間の解雇 (労基法第19条) などです。"
            "これらに該当しない懲戒解雇でも、解雇予告手当なしで即時解雇するには、"
            "あらかじめ所轄の労働基準監督署長に解雇予告除外認定の申請をし、認定を受ける必要があります。"
        ),
        rationale=[
            {"doc_id": S, "chunk_id": f"{S}__c0194", "page": None,
             "text": "※ 解雇が禁止されている場合 1 労働者の国籍、信条、社会的身分を理由とする解雇(労基法第3条)。 2 労働者の性別を理由とする解雇(均等法第6条)。 3 労働者の業務上の負傷、疾病による休業期間とその後30日間及び産前産後 の休業の期間(産前6週間(多胎妊娠の場合は14週間)以内又は産後8週間以 内の女性が休業する期間)とその後30日間の解雇(労基法第19条)。"},
            {"doc_id": S, "chunk_id": f"{S}__c0233", "page": None,
             "text": "労働者を懲戒解雇として平均賃金の30日分以上の解雇予告手当を支給せずに即時に 解雇する場合、あらかじめ、所轄の労働基準監督署長に解雇予告除外認定の申請をし、そ の認定を受けることが必要です(労基法第20条)。"},
        ],
        kg_query_type="negation_exhaustive", kg_novelty="unknown_relation",
        notes="解雇禁止列挙(c0194)と除外認定(c0233)、位置差39。否定・網羅型",
    ),

    # K2. traceability × procedural_relation: 休職解除→復職→満了退職の時系列 (gap 152)
    _qa(
        question=(
            "私傷病で休職中の労働者について、休職期間の途中で治癒した場合と、"
            "休職期間が満了しても治癒しなかった場合とで、その後の流れがどうなるか"
            "時系列で教えてください。"
        ),
        answer=(
            "休職期間中に休職事由がなくなった場合は、当然に休職が解除され復職となります。"
            "一方、休職期間が満了してもなお休職事由が消滅しないときは、退職となります。"
        ),
        rationale=[
            {"doc_id": S, "chunk_id": f"{S}__c0038", "page": None,
             "text": "2 休職期間中に休職事由がなくなった場合は、当然に休職が解除され復職となります。"},
            {"doc_id": S, "chunk_id": f"{S}__c0190", "page": None,
             "text": "3 第9条に定める休職期間が満了し、なお休職事由が消滅しないとき"},
        ],
        kg_query_type="traceability", kg_novelty="procedural_relation",
        notes="休職解除の手順(c0038)と満了退職(c0190)、位置差152。手順的関係",
    ),

    # K3. single_fact × unknown_term: 「番方編成による交替制」(c0066+c0067、隣接)
    _qa(
        question=(
            "「番方編成による交替制」とはどのような勤務形態ですか。"
            "この場合、休日を暦日ではなく与えることが認められる要件も教えてください。"
        ),
        answer=(
            "番方編成による交替制とは、8時間3交替勤務のような交替制勤務をいいます。"
            "この場合、就業規則等により定められて制度として運用されていること、"
            "各番方の交替が規則的に定められており勤務割表等によりその都度設定されるものではないこと、"
            "という要件を満たせば、休日は暦日ではなく継続した24時間を与えれば差し支えありません。"
        ),
        rationale=[
            {"doc_id": S, "chunk_id": f"{S}__c0066", "page": None,
             "text": "番方編成による交替制(8時間3交替勤務のような 場合をいいます。)を導入するような場合、以下の要件を満たせば休日は暦日ではなく、 継続した24時間を与えれば差し支えないとされています"},
            {"doc_id": S, "chunk_id": f"{S}__c0067", "page": None,
             "text": "(イ) 番方編成による交替制によることが就業規則等により定められており、制度と して運用されていること。 (ロ) 各番方の交替が規則的に定められているものであって、勤務割表等によりその 都度設定されるものではないこと。"},
        ],
        kg_query_type="single_fact", kg_novelty="unknown_term",
        notes="社内用語「番方編成」の定義(c0066)と要件(c0067)。名寄せ型",
    ),

    # K4. aggregation × unknown_relation: 副業を制限できる場合の列挙 (c0240 単一)
    _qa(
        question=(
            "労働者の副業・兼業は勤務時間外であれば原則自由と理解していますが、"
            "会社が禁止又は制限できるのはどのような場合ですか。すべて挙げてください。"
        ),
        answer=(
            "労務提供上の支障がある場合、企業秘密が漏洩する場合、"
            "会社の名誉や信用を損なう行為や信頼関係を破壊する行為がある場合、"
            "競業により企業の利益を害する場合、の4つの場合に禁止又は制限できます。"
        ),
        rationale=[
            {"doc_id": S, "chunk_id": f"{S}__c0240", "page": None,
             "text": "2 会社は、労働者からの前項の業務に従事する旨の届出に基づき、当該労働者が当該 業務に従事することにより次の各号のいずれかに該当する場合には、これを禁止又 は制限することができる。 1 労務提供上の支障がある場合 2 企業秘密が漏洩する場合 3 会社の名誉や信用を損なう行為や、信頼関係を破壊する行為がある場合 4 競業により、企業の利益を害する場合"},
        ],
        kg_query_type="aggregation", kg_novelty="unknown_relation",
        notes="副業制限4事由の列挙(c0240 単一チャンク)。集約型・対照群",
    ),

    # K5. single_fact × unknown_relation: 産後の就業禁止期間 (c0121 単一)
    _qa(
        question="産後の女性労働者を就業させてはいけない期間は何週間ですか。",
        answer=(
            "産後8週間を経過しない女性労働者を就業させてはいけません。"
            "ただし、産後6週間を経過した女性労働者から請求があったときは、"
            "医師が支障がないと認めた業務に就かせることができます。"
        ),
        rationale=[
            {"doc_id": S, "chunk_id": f"{S}__c0121", "page": None,
             "text": "2 産後8週間を経過しない女性労働者を就業させてはいけません。ただし、産後6週間を 経過した女性労働者から請求があったときは、医師が支障がないと認めた業務には就か せることができます(労基法第65条第2項)。"},
        ],
        kg_query_type="single_fact", kg_novelty="unknown_relation",
        notes="単一ファクト対照群(c0121)",
    ),

    # K6. multi_hop × unknown_term: 「時間単位年休」(c0117+c0118、隣接)
    _qa(
        question=(
            "「時間単位年休」とはどのような制度ですか。年に何日分まで認められ、"
            "所定労働時間が7時間30分の労働者の場合、1日分に相当する時間数は何時間になりますか。"
        ),
        answer=(
            "時間単位年休とは、労働者代表との書面による労使協定に基づき、"
            "年次有給休暇を時間単位で取得できる制度です (労基法第39条第4項)。"
            "1年について5日の範囲内で認められます。"
            "所定労働時間が7時間を超え8時間以下の者の1日相当時間数は8時間です。"
        ),
        rationale=[
            {"doc_id": S, "chunk_id": f"{S}__c0117", "page": None,
             "text": "第24条 労働者代表との書面による協定に基づき、前条の年次有給休暇の日数のう ち、1年について5日の範囲で次により時間単位の年次有給休暇(以下「時間単位 年休」という。)を付与する。 (1)時間単位年休付与の対象者は、すべての労働者とする。 (2)時間単位年休を取得する場合の、1日の年次有給休暇に相当する時間数は、 以下のとおりとする。 1 所定労働時間が5 時間を超え6 時間以下の者…6 時間 2 所定労働時間が6 時間を超え7 時間以下の者…7 時間 3 所定労働時間が7 時間を超え8 時間以下の者…8 時間"},
            {"doc_id": S, "chunk_id": f"{S}__c0118", "page": None,
             "text": "1 労使協定を締結すれば、年に5日を限度として、時間単位で年次有給休暇を与えること ができます(労基法第39条第4項)。"},
        ],
        kg_query_type="multi_hop", kg_novelty="unknown_term",
        notes="用語「時間単位年休」の定義と換算(c0117/c0118)。※7時間超8時間以下→8時間は c0117 の表",
    ),

    # K7. multi_hop × unknown_relation: 休日深夜の割増率 × 振替の通知期限 (gap 84)
    _qa(
        question=(
            "法定休日の労働が深夜にまで及んだ場合の割増率を教えてください。"
            "また、そもそも休日労働扱いを避けるために休日をあらかじめ振り替える場合、"
            "いつまでに通知する必要がありますか。"
        ),
        answer=(
            "休日労働が深夜に及んだ場合には6割以上の割増率で計算した割増賃金の支払いが必要です。"
            "休日の振替は、振替休日を特定し4週4日の休日が確保される範囲のできるだけ近接した日とした上で、"
            "前日までに通知する必要があります。"
        ),
        rationale=[
            {"doc_id": S, "chunk_id": f"{S}__c0152", "page": None,
             "text": "なお、時間外労働が深夜に及んだ場合には5割以上、休日労働が深夜に及んだ場合には 6割以上の割増率で計算した割増賃金をそれぞれ支払わなければなりません。"},
            {"doc_id": S, "chunk_id": f"{S}__c0068", "page": None,
             "text": "ア 就業規則に振替休日の規程を置くこと。 イ 振替休日は特定すること。 ウ 振替休日は4週4日の休日が確保される範囲のできるだけ近接した日とする こと。 エ 振替は前日までに通知すること。"},
        ],
        kg_query_type="multi_hop", kg_novelty="unknown_relation",
        notes="休日深夜6割(c0152)と振替要件(c0068)、位置差84",
    ),

    # K8. single_fact × unknown_term: 「基準日」(c0113 単一)
    _qa(
        question="年次有給休暇の「基準日」とは何を指しますか。",
        answer=(
            "基準日とは、年次有給休暇の権利が発生した日のことであり、"
            "雇入れ後6か月経過した日、その後は1年ごとの日のことをいいます。"
        ),
        rationale=[
            {"doc_id": S, "chunk_id": f"{S}__c0113", "page": None,
             "text": "ここでいう「基準日」とは、年次有給休暇の権利が発生した日のことであり、雇入れ後6 か月経過した日、その後は1年ごとの日のことをいいます。"},
        ],
        kg_query_type="single_fact", kg_novelty="unknown_term",
        notes="用語「基準日」の定義(c0113 単一)。名寄せ対照群",
    ),

    # ===== plant 4 件 =====

    # P8. negation_exhaustive × unknown_relation: 穴補強の準用が免除される場合 (c0071 単一)
    _qa(
        question=(
            "ボイラーの管に穴を設ける場合には平板の穴補強の規定が準用されると理解していますが、"
            "この準用が免除される (適用しなくてよい) のはどのような穴ですか。"
        ),
        answer=(
            "当該穴の径が61 mm以下で、かつ、管の外径の1/4以下のものにあっては、"
            "補強規定の準用は適用されません。"
        ),
        rationale=[
            {"doc_id": P, "chunk_id": f"{P}__c0071", "page": None,
             "text": "３ 第５９条第３項の規定は、管に穴を設ける場合に準用する。ただし、当該穴の径が61 mm 以下で、かつ、管の外径の1/4 以下のものにあっては、この限りでない。"},
        ],
        kg_query_type="negation_exhaustive", kg_novelty="unknown_relation",
        notes="準用の除外条件(c0071 ただし書き)。否定型",
    ),

    # P9. aggregation × unknown_relation: 安全弁の吹出し圧力の場合分け (c0021 単一)
    _qa(
        question=(
            "過熱器のある循環ボイラーについて、ドラムと過熱器に設ける安全弁の吹出し圧力の"
            "設定条件を、場合分けしてすべて教えてください。"
        ),
        answer=(
            "ドラムに設ける安全弁が1個の場合はボイラーの最高使用圧力以下の圧力とします。"
            "ただし最高使用圧力以下で自動的に作動する圧力逃がし装置がある場合は"
            "最高使用圧力の1.03倍以下とすることができます。"
            "安全弁が2個以上の場合は、1個は前記に準ずる圧力、他は最高使用圧力の1.03倍以下の圧力とします。"
            "過熱器に設ける安全弁は、ドラムに設ける安全弁に先行して動作する圧力であることが必要です。"
        ),
        rationale=[
            {"doc_id": P, "chunk_id": f"{P}__c0021", "page": None,
             "text": "ドラムに設ける安全弁の吹出し圧力は、次によること。 （イ）安全弁が1 個の場合は、ボイラーの最高使用圧力以下の圧力。ただし、当該 ボイラーにボイラーの最高使用圧力以下の圧力で自動的に作動する圧力逃がし 装置がある場合は、ボイラーの最高使用圧力の1.03 倍以下の圧力とすることが できる。 （ロ）安全弁が2 個以上の場合は、1 個は(イ)の規定に準ずる圧力、他はボイラーの 最高使用圧力の1.03 倍以下の圧力 ホ 過熱器に設ける安全弁の吹出し圧力は、ドラムに設ける安全弁に先行して動作す る圧力であること。"},
        ],
        kg_query_type="aggregation", kg_novelty="unknown_relation",
        notes="安全弁設定の場合分け列挙(c0021 単一)。集約型",
    ),

    # P10. single_fact × unknown_term: 「耐圧部分」の定義 (c0001 単一)
    _qa(
        question="発電用火力設備の技術基準でいう「耐圧部分」とは、具体的にどの部分を指しますか。",
        answer=(
            "耐圧部分とは、内面に0 MPaを超える圧力 (ゲージ圧力) を受ける部分をいいます。"
        ),
        rationale=[
            {"doc_id": P, "chunk_id": f"{P}__c0001", "page": None,
             "text": "第２条 省令第５条に規定する「耐圧部分」とは、内面に0 MPa を超える圧力（ゲージ圧 力をいう。以下同じ。）を受ける部分をいう。"},
        ],
        kg_query_type="single_fact", kg_novelty="unknown_term",
        notes="用語「耐圧部分」の定義(c0001 単一)。名寄せ対照群",
    ),

    # P11. traceability × procedural_relation: 溶接自主検査の2段階 (34_yousetu c0002 単一)
    _qa(
        question=(
            "ボイラー等の耐圧部分について溶接をする場合の溶接自主検査は、"
            "どの段階で何を検査する構成になっていますか。工程の順に教えてください。"
        ),
        answer=(
            "まず電気工作物の溶接をしようとする前に、溶接施工法に関することと溶接士の技能に関することを、"
            "別表1に示す溶接自主検査の工程ごとの検査の方法により確認します。"
            "その後、溶接施工した構造物に対して、ボイラー等であって耐圧部分について溶接をするものごとに"
            "別表2に示す溶接自主検査の工程ごとの検査の方法により検査を行います。"
        ),
        rationale=[
            {"doc_id": Y, "chunk_id": f"{Y}__c0002", "page": None,
             "text": "（１） あらかじめ確認すべき事項に対する溶接自主検査 \n① 次に掲げる事項については、電気工作物の溶接をしようとする前に別表１に示す溶\n接自主検査の工程ごとの検査の方法により行うこと。 イ 溶接施工法に関すること \nロ 溶接士の技能に関すること \n② ①のイに掲げる事項については、客観性を有する方法により発電用火力設備の技術\n基準の解釈（平成２５年５月１７日付け２０１３０５０７商局第２号。以下「技術\n基準解釈」という。）第１０７条第１項に規定する試験を実施し、当該試験に合格す\nること。 ③ ①のロに掲げる事項については、客観性を有する方法により技術基準解釈第１１０\n条第１項及び同条第２項第２号イに規定する試験を実施し、当該試験に合格するこ\nと。 （２） 溶接施工した構造物に対する溶接自主検査 \n次に掲げる事項については、ボイラー等であって耐圧部分について溶接をするものご\nとに別表２に示す溶接自主検査の工程ごとの検査の方法により行うこと。"},
        ],
        kg_query_type="traceability", kg_novelty="procedural_relation",
        notes="溶接前検査(別表1)→施工後検査(別表2)の工程順序。手順的関係 (単一チャンク・対照)",
    ),
]


def main():
    out_path = Path("data/dataset_axes_v4.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for qa in QAS:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    print(f"wrote {len(QAS)} QAs to {out_path}")


if __name__ == "__main__":
    main()
