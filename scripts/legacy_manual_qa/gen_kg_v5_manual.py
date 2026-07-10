"""Hand-authored QA dataset for KG-PoC, written by Claude (Opus 4.7 via Claude Code).

Each QA satisfies the leave-one-out invariant: if either rationale doc is hidden,
at least one answer-token cannot be reproduced from the remaining rationale.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path

PROMPT_VERSION = "manual_claude_opus_4_7_v1.0"
GEN_MODEL = "claude-opus-4-7-via-claude-code"


def _qa_id(question: str) -> str:
    h = hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]
    return f"manual_{h}"


def _qa(
    question: str,
    answer: str,
    rationale: list[dict],
    *,
    kg_query_type: str,
    kg_novelty: str = "unknown_relation",
    notes: str | None = None,
) -> dict:
    return {
        "qa_id": _qa_id(question),
        "question": question,
        "answer": answer,
        "rationale": rationale,
        "category": [],
        "aspect": [],
        "reasoning_complexity": {
            "multi_step": kg_query_type in ("multi_hop", "traceability"),
            "quantitative": False, "negation": False, "cause_effect": False,
            "comparison": False, "temporal": kg_query_type == "traceability",
            "output_type": "none",
        },
        "retrieval_difficulty": {
            "multi_doc": True, "multi_chunk": True, "low_locality": True,
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
        "difficulty_rationale": notes or "両 rationale の固有値が answer に必須 (leave-one-out 通過)",
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


QAS = [
    # --- pair_09: 耐圧試験(第130条) + 溶接自主検査(別表1) ---
    _qa(
        question=(
            "ボイラー等の容器または管の溶接部について耐圧試験を実施したいと考えています。"
            "試験圧力の準用規定はどこを参照すればよいでしょうか。"
            "また、耐圧試験を実施する前段階で、溶接施工法と溶接士の技能について"
            "あらかじめどんな方法で検査を行うべきかも教えてください。"
        ),
        answer=(
            "耐圧試験そのものについては、第130条により第5条の規定を準用して実施します。"
            "それより前の段階で、溶接施工法と溶接士の技能に関する事項は、"
            "「別表１に示す溶接自主検査の工程ごとの検査の方法」により確認する必要があります。"
        ),
        rationale=[
            {"doc_id": "09_kagi_kaishaku_240315", "page": None,
             "text": "（耐圧試験） 第１３０条 ボイラー等に係る容器又は管の溶接部に対する耐圧試験については、第５条 の規定を準用する。"},
            {"doc_id": "34_yousetu_kaisyaku", "page": None,
             "text": "次に掲げる事項については、電気工作物の溶接をしようとする前に別表１に示す溶\n接自主検査の工程ごとの検査の方法により行うこと。 イ 溶接施工法に関すること \nロ 溶接士の技能に関すること"},
        ],
        kg_query_type="traceability",
        notes="耐圧試験(09第130条)と事前検査(34別表1)の二段階を両方引かないと答えが揃わない",
    ),

    # --- pair_01: 非破壊試験技術者(JIS Z 2305) + 溶接施工法試験(第107条第1項) ---
    _qa(
        question=(
            "圧力容器の溶接部について非破壊試験を行う技術者の資格条件と、"
            "事前に行う溶接施工法に関する自主検査として実施すべき試験は何かを教えてください。"
        ),
        answer=(
            "非破壊試験を実施する技術者は、日本産業規格 JIS Z 2305（2013）「非破壊試験技術者の資格及び認証」"
            "に基づき認証された者でなければなりません。"
            "また、溶接施工法に関する事前検査としては、技術基準解釈 第１０７条第１項に規定する試験を実施し、"
            "当該試験に合格する必要があります。"
        ),
        rationale=[
            {"doc_id": "09_kagi_kaishaku_240315", "page": None,
             "text": "第２項の非破壊試験は、次の各号のいずれかの者により行われなければならない。 一 日本産業規格 JIS Z 2305 （2013 ）「非破壊試験技術者の資格及び認証」"},
            {"doc_id": "34_yousetu_kaisyaku", "page": None,
             "text": "①のイに掲げる事項については、客観性を有する方法により発電用火力設備の技術\n基準の解釈（平成２５年５月１７日付け２０１３０５０７商局第２号。以下「技術\n基準解釈」という。）第１０７条第１項に規定する試験を実施し、当該試験に合格す\nること。"},
        ],
        kg_query_type="multi_hop",
        notes="非破壊試験者資格(JIS Z 2305、09)と施工法試験(第107条、34) 別固有値",
    ),

    # --- pair_17: 安全弁吹出し圧力(ドラム) + 溶接記録の報告(第85条) ---
    _qa(
        question=(
            "循環ボイラーのドラムに安全弁を1個だけ設置する場合、"
            "吹出し圧力をどう設定すればよいか、また圧力逃がし装置を持つ場合の特例も教えてください。"
            "あわせて、ボイラー設置後にこの溶接自主検査の結果は誰に対していつまでに報告する必要があるかも知りたいです。"
        ),
        answer=(
            "ドラムに設ける安全弁が1個の場合、吹出し圧力はボイラーの最高使用圧力以下に設定します。"
            "ただし、当該ボイラーに最高使用圧力以下の圧力で自動的に作動する圧力逃がし装置がある場合は、"
            "最高使用圧力の1.03倍以下とすることができます。"
            "溶接自主検査の実施状況及び結果については、毎年度6月末日までに、"
            "電気関係報告規則 第2条の表 第9号に基づき国へ報告する必要があります。"
        ),
        rationale=[
            {"doc_id": "09_kagi_kaishaku_240315", "page": None,
             "text": "ドラムに設ける安全弁の吹出し圧力は、次によること。 （イ）安全弁が1 個の場合は、ボイラーの最高使用圧力以下の圧力。ただし、当該 ボイラーにボイラーの最高使用圧力以下の圧力で自動的に作動する圧力逃がし 装置がある場合は、ボイラーの最高使用圧力の1.03 倍以下の圧力とすることが できる。"},
            {"doc_id": "34_yousetu_kaisyaku", "page": None,
             "text": "溶接自主検査を実施した電気工作物を設置する者は、前年度に実施した溶接自主検査\nの実施状況及びその結果を電気関係報告規則（昭和４０年通商産業省令第５４号）第２\n条の表第９号に基づき、毎年度６月末日までに国へ報告しなければならない。"},
        ],
        kg_query_type="multi_hop",
        notes="安全弁(09第6項)と報告義務(34第85条)で別固有値",
    ),

    # --- pair_07: 容器胴(第6条 JIS B 8201) + 溶接施工試験(第107条第1項) ---
    _qa(
        question=(
            "ボイラー用容器の円筒胴を新規に設計する際、"
            "胴の最小厚さの計算に従うべき規格はどれですか。"
            "また、その容器を製作する際の溶接については、施工法に関する自主検査で"
            "何条の試験に合格しておく必要があるか、あわせて教えてください。"
        ),
        answer=(
            "ボイラー等及び独立節炭器に属する円筒胴の最小厚さは、"
            "日本産業規格 JIS B 8201（2013）「陸用鋼製ボイラ－構造」の「6.2.2 内圧胴の最小厚さ」"
            "に規定されている計算式により算出します。"
            "あわせて、溶接施工法に関する事項については、技術基準解釈 第１０７条第１項に規定する試験を実施し、"
            "当該試験に合格する必要があります。"
        ),
        rationale=[
            {"doc_id": "09_kagi_kaishaku_240315", "page": None,
             "text": "円筒形の胴にあっては日本産業規格 JIS B 8201（2013）「陸用鋼製ボイラ－構造」の 「6.2.2 内圧胴の最小厚さ」に規定されている計算式により算出した値"},
            {"doc_id": "34_yousetu_kaisyaku", "page": None,
             "text": "①のイに掲げる事項については、客観性を有する方法により発電用火力設備の技術\n基準の解釈（平成２５年５月１７日付け２０１３０５０７商局第２号。以下「技術\n基準解釈」という。）第１０７条第１項に規定する試験を実施し、当該試験に合格す\nること。"},
        ],
        kg_query_type="multi_hop",
        notes="設計規格(JIS B 8201)と溶接試験(第107条第1項) 別固有値",
    ),

    # --- pair_15: 容器胴の穴補強(JIS B 8267) + 溶接士技能試験(第110条) ---
    _qa(
        question=(
            "圧力容器の円筒胴と円すい形胴の接続部について大径端部と小径端部の計算厚さを求めたい。"
            "どの規格のどの附属書を参照すればよいですか。"
            "またこの容器を溶接する溶接士の技能について、客観性を有する方法でどの試験を実施し合格させる必要があるかも教えてください。"
        ),
        answer=(
            "大径端部および小径端部は、日本産業規格 JIS B 8267（2015）「圧力容器の設計」の"
            "「附属書E（規定）圧力容器の胴及び鏡板」の「E.2.4 円すい胴の計算厚さ」の"
            "「b)大径端」および「c)小径端」によります。"
            "溶接士の技能に関しては、技術基準解釈 第１１０条第１項及び同条第２項第２号イに規定する試験を実施し、"
            "当該試験に合格する必要があります。"
        ),
        rationale=[
            {"doc_id": "09_kagi_kaishaku_240315", "page": None,
             "text": "大径端部及び小径端部は、日本産業規格 JIS B 8267（2015）「圧力容器の設計」の「附 属書E（規定）圧力容器の胴及び鏡板」の「E.2.4 円すい胴の計算厚さ」の「b)大径 端」及び「c)小径端」によること。"},
            {"doc_id": "34_yousetu_kaisyaku", "page": None,
             "text": "①のロに掲げる事項については、客観性を有する方法により技術基準解釈第１１０\n条第１項及び同条第２項第２号イに規定する試験を実施し、当該試験に合格するこ\nと。"},
        ],
        kg_query_type="multi_hop",
        notes="設計規格(JIS B 8267)と技能試験(第110条) 別固有値",
    ),

    # --- pair_04: 管厚計算(第59条第3項 / 第61条) + 溶接施工試験(第107条) ---
    _qa(
        question=(
            "ボイラー用の管に穴を設けて差し込み閉止板で塞ぐ設計をしたい。"
            "差し込み閉止板の最小厚さを求めるための条件式と、係数の定義はどこに従えばよいですか。"
            "また、この溶接構造物を作る際の溶接施工法に対する事前の自主検査として何の試験に合格すればよいかも教えてください。"
        ),
        answer=(
            "差し込み閉止板の最小厚さは t = d × √(3P/(16a)) の式により算出した値以上で、"
            "P および a は第59条第2項第二号に定めるところによります。"
            "また溶接施工法に関する事前の自主検査としては、技術基準解釈 第１０７条第１項に規定する試験を実施し、"
            "当該試験に合格する必要があります。"
        ),
        rationale=[
            {"doc_id": "09_kagi_kaishaku_240315", "page": None,
             "text": "管に取り付ける平板の厚さは、差し込み閉止板以外のものにあっては第６１条に掲げ る計算式により算出した値以上、差し込み閉止板にあっては次の計算式により算出した 値以上であること。 a B P d t  16 3 = t は、差し込み閉止板の最小厚さ（mm を単位とする。） P 及び a は、それぞれ第５９条第２項第二号に定めるところによる。"},
            {"doc_id": "34_yousetu_kaisyaku", "page": None,
             "text": "①のイに掲げる事項については、客観性を有する方法により発電用火力設備の技術\n基準の解釈（平成２５年５月１７日付け２０１３０５０７商局第２号。以下「技術\n基準解釈」という。）第１０７条第１項に規定する試験を実施し、当該試験に合格す\nること。"},
        ],
        kg_query_type="multi_hop",
        notes="差し込み閉止板計算式(09)と施工法試験(34) 別固有値",
    ),

    # --- pair_18: 技報PRB炭ボイラ材料 + 技報石炭タワー型ボイラ材料 ---
    _qa(
        question=(
            "USC蒸気条件の石炭専焼ボイラの過熱器/再熱器用材料として、"
            "アメリカ向けPRB炭専焼ボイラ計画と、ヨーロッパ向け石炭専焼タワー型ボイラ実績とでは"
            "それぞれ何が採用されていたか、その違いを教えてください。"
        ),
        answer=(
            "アメリカ向けPRB炭専焼ボイラ計画の最終過熱器および再熱器には ASME Code Case 2328-1 "
            "（火SUS304J1HTB相当）が採用されていました。"
            "一方、ヨーロッパ向け石炭専焼タワー型ボイラでは、過熱器・再熱器の加熱部に SUPER304H、"
            "非加熱部に9Cr材が採用されています。"
        ),
        rationale=[
            {"doc_id": "v54n1_17_アメリカにおける亜瀝青炭専焼超臨界圧ボイラの900MW計画と試運転実績", "page": None,
             "text": "3次過熱器(最終過熱器) ASME CodeCase2328-1 [火SUS304JIHTB相当] 最終過熱器出口管寄 ASME SA335P91 再熱器 ASME CodeCase2328-1 [火SUS304J1HTB相当]"},
            {"doc_id": "v54n1_18_石炭専焼タワー型ボイラの運転実績813MWUSC", "page": None,
             "text": "USC蒸気条件のボイラの信頼性を確保するため, EN規格(ヨーロッパ統一規格)との整合性を十分確認のうえ、豊富な実績と知見をもつ材料を採用した.過熱器や再熱器における加熱部にはSUPER304H®を,非加熱部には9Cr材を採用した."},
        ],
        kg_query_type="multi_hop",
        kg_novelty="unknown_relation",
        notes="技報2件の材料選定の比較。両方の固有名(材料種別)が answer に必要",
    ),

    # --- pair_02: 圧力容器附属書(E.3.4 / E.3.2 / 第60条) + 02_boiler 様式第6号(第60条関係) ---
    _qa(
        question=(
            "圧力容器の半だ円形鏡板を新規に設計します。"
            "計算厚さを求めるためにどの規定のどの附属書の規定を参照すればよいですか。"
            "また、設計した容器が技術基準に適合しているかの検査結果を記録するのに使う様式はどれか、あわせて教えてください。"
        ),
        answer=(
            "半だ円形鏡板の計算厚さは、「圧力容器の設計」の「附属書E（規定）圧力容器の胴及び鏡板」の "
            "「E.3.4 半だ円形鏡板」または「圧力容器の構造－一般事項」の「E.3.4 半だ円形鏡板の計算厚さ」"
            "により算出します。"
            "検査結果の記録には、第60条関係に該当する様式として様式第６号（裏面）を使用します。"
        ),
        rationale=[
            {"doc_id": "09_kagi_kaishaku_240315", "page": None,
             "text": "第６０条 「圧力容器の設計」の「附属書E（規 定）圧力容器の胴及び鏡板」の「E.3.2 全半球形鏡板」、「E.3.3 皿形鏡板」、 「E.3.4 半だ円形鏡板」及び「E.3.5 円すい形鏡板」 「圧力容器の構造－一般事項」の「附 属書E（規定）圧力容器の胴及び鏡板」 の「E.3.2 全半球形鏡板の計算厚さ」、 「E.3.3 皿形鏡板の計算厚さ」、「E.3.4 半だ円形鏡板の計算厚さ」"},
            {"doc_id": "02_boiler_anzen", "page": None,
             "text": "様式第６号（第５条、第１１条、第１２条、第１５条、第５１条、第５６条の２、第５７条、第６０条関係）（裏面）"},
        ],
        kg_query_type="traceability",
        notes="附属書E.3.4(09)と様式第6号(02、第60条関係) を両方引かないと答えが揃わない",
    ),

    # --- pair_06: 別表第3 貯槽支持物許容応力 + 45_tokutei 様式 ---
    _qa(
        question=(
            "貯槽の支持物および基礎に使用する炭素鋼鋳鋼品について、厚さ40 mm以下の材料の許容引張応力はどの規定を見れば確認できますか。"
            "また、その特定設備の検査結果を第５条関係として届け出るために使う様式番号もあわせて教えてください。"
        ),
        answer=(
            "貯槽の支持物および基礎に使用される炭素鋼鋳鋼品(JIS G 5101)で厚さ40 mm以下のものの許容引張応力は、"
            "別表第３「貯槽及びガスホルダーの支持物及び基礎に使用される主要材料の許容応力」（第５６条、第５８条、"
            "第６４条、第７６条及び第８２条関係）に従い、引張で156 N/mm²と定められています。"
            "また、第５条関係として届け出に使用する様式は、様式第１（第５条関係）または様式第２（第５条関係）です。"
        ),
        rationale=[
            {"doc_id": "09_kagi_kaishaku_240315", "page": None,
             "text": "別表第３　貯槽及びガスホルダーの支持物及び基礎に使用される主要材料の許容応力　（第５６条、第５８条、第６４条、第７６条及び第８２条関係） 引張 圧縮 曲げ せん断 側圧 (...略...) 厚さ40 mm以下 156 156 156 90 293 (...略...) 炭素鋼鋳鋼品 JIS G 5101(1991)"},
            {"doc_id": "45_tokutei_setsubi", "page": None,
             "text": "様式第１（第５条関係）様式第２（第５条関係）様式第３（第４６条関係）様式第４（第４７条及び第４８条関係）様式第５（第４９条関係）様式第６（第５０条関係）"},
        ],
        kg_query_type="traceability",
        notes="許容応力表(09 別表第3、JIS G 5101、40 mm以下、156)と様式(45 第5条関係) 両方の固有値が必要",
    ),

    # --- pair_08: JIS鋼板使用制限 + 溶接士技能試験(第110条) ---
    _qa(
        question=(
            "JIS G 3106「溶接構造用圧延鋼材」を使って圧力容器を設計したいのですが、"
            "胴や鏡板などに使用してはならない圧力条件を教えてください。"
            "また、こうした圧力容器を溶接する溶接士の技能を確認するには、客観性を有する方法で何条の試験に合格させる必要があるかも知りたい。"
        ),
        answer=(
            "JIS G 3106「溶接構造用圧延鋼材」のSM400A、SM490A及びSM490YA等は、"
            "設計圧力が1.6 MPaを超える圧力容器の胴・鏡板などに使用してはなりません。"
            "また圧力が1 MPaを超え胴に長手溶接継手があるものや鏡板に溶接継手があるものにも使用できません。"
            "これら溶接構造物の溶接士の技能は、技術基準解釈 第１１０条第１項及び同条第２項第２号イに規定する試験を実施し、"
            "当該試験に合格する必要があります。"
        ),
        rationale=[
            {"doc_id": "09_kagi_kaishaku_240315", "page": None,
             "text": "JIS G 3101（2010）「一般構造用圧延鋼材」、JIS G 3106（2008）「溶接構造用圧延鋼材」のSM400A、SM490A及びSM490YA、(...略...)は、次に掲げる圧力容器の部分に使用してはならない。 イ 設計圧力が1.6 MPaを超える圧力容器の胴、鏡板、その他これらに類する部分 ロ 圧力が1 MPaを超える圧力容器で、胴に長手溶接継手があるもの、及び鏡板に溶接継手のあるもの"},
            {"doc_id": "34_yousetu_kaisyaku", "page": None,
             "text": "①のロに掲げる事項については、客観性を有する方法により技術基準解釈第１１０\n条第１項及び同条第２項第２号イに規定する試験を実施し、当該試験に合格するこ\nと。"},
        ],
        kg_query_type="multi_hop",
        notes="鋼材使用制限(09 JIS G 3106)と溶接士技能試験(34 第110条) 別固有値",
    ),
]


def main():
    out_path = Path("data/dataset_kg_v5_manual.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for qa in QAS:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    print(f"wrote {len(QAS)} QAs to {out_path}")


if __name__ == "__main__":
    main()
