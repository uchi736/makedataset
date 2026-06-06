"""階層① 能力評価軸: 25観点を Literal で固定。

aspect は QAItem.aspect: list[str] に入れる値。表記揺れを防ぐため Literal で
列挙し、ASPECT_TO_CATEGORY でカテゴリへの逆引き、ASPECT_DESCRIPTIONS で抽象
定義、ASPECT_EXAMPLES で現場の具体例(プロンプト/UIヒント用)を提供する。
"""

from __future__ import annotations

from typing import Literal, get_args


ASPECTS = Literal[
    # Integration
    "multi_source_integration",
    "multi_doc_reference",
    "remote_reference",
    "standards_reference",
    # Reasoning
    "quantitative_calc",
    "multi_hop",
    "negation",
    "causal",
    "temporal",
    "comparison_conditional",
    # Logic
    "synonym_interpretation",
    "numeric_inclusion",
    "concept_inclusion",
    "vocabulary_mismatch",
    "abstraction_gap",
    # Figure (コンテンツ構造ベース)
    "simple_table",
    "complex_form",
    "concept_diagram",
    "flowchart",
    "chart_graph",
    "complex_layout",
    "large_enumeration",
    # Abstention
    "insufficient_evidence",
    "contradictory_evidence",
    "fragmented_chunk",
]

ALL_ASPECTS: tuple[str, ...] = tuple(get_args(ASPECTS))


CategoryName = Literal["Integration", "Reasoning", "Logic", "Figure", "Abstention"]


ASPECT_TO_CATEGORY: dict[str, CategoryName] = {
    "multi_source_integration": "Integration",
    "multi_doc_reference":      "Integration",
    "remote_reference":         "Integration",
    "standards_reference":      "Integration",

    "quantitative_calc":        "Reasoning",
    "multi_hop":                "Reasoning",
    "negation":                 "Reasoning",
    "causal":                   "Reasoning",
    "temporal":                 "Reasoning",
    "comparison_conditional":   "Reasoning",

    "synonym_interpretation":   "Logic",
    "numeric_inclusion":        "Logic",
    "concept_inclusion":        "Logic",
    "vocabulary_mismatch":      "Logic",
    "abstraction_gap":          "Logic",

    "simple_table":             "Figure",
    "complex_form":             "Figure",
    "concept_diagram":          "Figure",
    "flowchart":                "Figure",
    "chart_graph":              "Figure",
    "complex_layout":           "Figure",
    "large_enumeration":        "Figure",

    "insufficient_evidence":    "Abstention",
    "contradictory_evidence":   "Abstention",
    "fragmented_chunk":         "Abstention",
}


# 観点ごとの抽象定義(汎用)
ASPECT_DESCRIPTIONS: dict[str, str] = {
    "multi_source_integration": "複数の情報源(文書・章節)の内容を統合して回答が必要",
    "multi_doc_reference":      "2つ以上の異なる文書を横断参照する必要がある",
    "remote_reference":         "同一文書内でも章をまたぐなど離れた場所を参照する必要がある",
    "standards_reference":      "規格・規定の番号による参照を解決する必要がある",
    "quantitative_calc":        "数値計算(合計・差分・比率など)が必要",
    "multi_hop":                "中間結論を経由する多段推論が必要",
    "negation":                 "「〜でない」等の否定条件を解釈する必要がある",
    "causal":                   "原因・結果・理由を導出する必要がある",
    "temporal":                 "時系列・順序関係を解釈する必要がある",
    "comparison_conditional":   "比較や条件分岐を含む推論が必要",
    "synonym_interpretation":   "同義語・言い換え表現の解釈が必要",
    "numeric_inclusion":        "数値の包含関係(範囲・以上以下)を解釈する必要がある",
    "concept_inclusion":        "概念の包含・上位下位関係を解釈する必要がある",
    "vocabulary_mismatch":      "質問と文書で表現が異なる(専門用語・略語・社内用語のずれ)",
    "abstraction_gap":          "抽象的な質問 vs 具体的な記述、またはその逆",
    "simple_table":             "単純な行列の表(読み下しで理解可能)",
    "complex_form":             "セル結合・ヘッダ階層・項目グループのある複雑帳票",
    "concept_diagram":          "概念図・構成図・ブロック図など関係性を表現する図",
    "flowchart":                "フローチャート・プロセス図・状態遷移図など手順を表現する図",
    "chart_graph":              "折れ線・棒・円グラフなど数値を可視化したチャート",
    "complex_layout":           "段組み・図表混在・サイドノートなど複雑なページレイアウト",
    "large_enumeration":        "長いリスト・大量の項目列挙",
    "insufficient_evidence":    "文書内に答えとなる情報が存在しない(回答拒否すべき)",
    "contradictory_evidence":   "複数箇所の情報が矛盾している(回答拒否すべき)",
    "fragmented_chunk":         "チャンク分割の都合で答えが分断されている",
}


# 観点ごとの現場での具体例(備考、プロンプトのヒント用)
ASPECT_EXAMPLES: dict[str, str] = {
    "multi_source_integration": "複数の規定・仕様書からの条件統合",
    "multi_doc_reference":      "仕様書↔規格書、上位規定↔下位規定、親規定↔子規定の参照",
    "remote_reference":         "附則・別表の参照、改訂履歴と本文の対応",
    "standards_reference":      "JIS/ISO/ASME 等の規格番号参照、社内規定番号の参照",
    "quantitative_calc":        "公差判定、許容値計算、単位変換(mm↔cm)、寸法合計",
    "multi_hop":                "規定A→規定B→規定Cと辿って答えに到達",
    "negation":                 "「適用しない」「除外する」条項の解釈",
    "causal":                   "不具合原因の特定、設計変更の理由、品質問題の根本原因",
    "temporal":                 "手順の順序、改訂前後の差分、製造工程の前後関係",
    "comparison_conditional":   "規定値との照合、適用条件判定(事業部別/製品別)、仕様比較",
    "synonym_interpretation":   "JIS規格名と通称、製品名と型番、部品の旧称と新称",
    "numeric_inclusion":        "「規定値±5%以内」「100以上」等の範囲判定",
    "concept_inclusion":        "全社規定→事業部規定→部門規定の階層、製品ファミリと型番",
    "vocabulary_mismatch":      "略語(QMS/P&ID/FMEA)、部署用語、業界用語と一般用語の差",
    "abstraction_gap":          "「品質を確保する方法は」のような抽象質問に対する具体手順の記述",
    "simple_table":             "部品リスト、仕様一覧、規定値一覧(単純な行列)",
    "complex_form":             "ヘッダ階層のある規定値表、セル結合のある帳票",
    "concept_diagram":          "システム構成図、機器ブロック図、組織図",
    "flowchart":                "QMSプロセス図、品質管理手順、設計レビューフロー",
    "chart_graph":              "試験結果のグラフ、トレンドチャート、不具合発生件数の推移",
    "complex_layout":           "図面と注記の混在、段組み仕様書",
    "large_enumeration":        "長い部品リスト、多数の項目を含む規定値表",
    "insufficient_evidence":    "文書外の情報を聞く質問(コーパスに答えがない)",
    "contradictory_evidence":   "旧版規定と新版規定の混在、矛盾する記述",
    "fragmented_chunk":         "ページまたぎ・章またぎで分断された記述",
}


# 観点ごとの「✓ 良い問いの例」(プロンプトで作問を誘導する具体例)
ASPECT_GOOD_EXAMPLES: dict[str, list[str]] = {
    # --- Integration ---
    "multi_source_integration": [
        "第○章と第△章の規定を併せて読み解いた場合の××の取扱いを答えよ",
        "親規定の制約と子規定の手順を統合した結論を答えよ",
    ],
    "multi_doc_reference": [
        "仕様書と規格書の対応条文を参照して○○の結論を導け",
        "上位規定の要件と下位規定の手順を併せて参照する必要のある問い",
    ],
    "remote_reference": [
        "本文第○条が参照する別表△で、××に該当する項目を答えよ",
        "附則で改訂された本文条項の現在の内容を、附則と本文を照合して答えよ",
    ],
    "standards_reference": [
        "本文で参照される規格番号(例: JIS Z 2241)の規定内容を、本文の手順と紐づけて答えよ",
        "規格番号で示された要件が本文のどの条項に対応するか答えよ",
    ],
    # --- Reasoning ---
    "quantitative_calc": [
        "月額○万円の労働者が時間外労働を△時間した場合の追加賃金額",
        "年次有給休暇△日を取得した場合の残日数を計算せよ",
    ],
    "multi_hop": [
        "条件Aを満たす場合の規定B、その例外Cを順に辿った最終的な取扱いを答えよ",
        "第○条の対象→第△条の手順→第□条の例外を経由した結論",
    ],
    "negation": [
        "本規定が適用されない条件を全て挙げよ",
        "○○を除く対象に該当するのはどのような場合か",
    ],
    "causal": [
        "△△が認められる根拠条文と、そう取り扱う理由を答えよ",
        "○○の取扱いがそうなっている理由を文書から導いて答えよ",
    ],
    "temporal": [
        "申請から実施までに要する所定期間と、その経過手順を答えよ",
        "通知の必要日数と、間に合わない場合の代替手段の順序",
    ],
    "comparison_conditional": [
        "正社員と契約社員における○○の取扱いの差を答えよ",
        "業務都合と労働者都合の場合で△△の取扱いがどう変わるか",
    ],
    # --- Logic ---
    "synonym_interpretation": [
        "本規程の『所定労働時間』と『勤務時間』の関係を答えよ",
        "△△と同義で使われている別の表現を本文中から特定し、対応する条項を答えよ",
    ],
    "numeric_inclusion": [
        "年△日以上の範囲に含まれる条件と、その場合の取扱いを答えよ",
        "月○時間超の労働に該当する場合の取扱いを答えよ",
    ],
    "concept_inclusion": [
        "育児休業と介護休業の包含関係を、根拠条文と共に答えよ",
        "年次有給休暇と特別休暇の上下関係(片方が他方を含むか)を答えよ",
    ],
    "vocabulary_mismatch": [
        "『QMS』『FMEA』などの略語が本文中ではどの正式表記で書かれているか答えよ",
        "略語と本文の正式表記の対応から、該当条項を特定せよ",
    ],
    "abstraction_gap": [
        "『休暇制度の概要は』のような抽象的な問いに対し、本文の具体条項で答えよ",
        "○○の方針について、本文の具体的手順を整理して答えよ",
    ],
    # --- Figure (素材が元チャンクに実在する場合のみ) ---
    "simple_table": [
        "表の中で○○の条件を満たす行を全て挙げよ",
        "列Aの値が△を超える行において、列Bが示す値を答えよ",
    ],
    "complex_form": [
        "セル結合で同一カテゴリに括られている項目を全て挙げよ",
        "ヘッダ階層から、○○項目の上位カテゴリ名を答えよ",
    ],
    "concept_diagram": [
        "構成図に示された A と B の関係性を答えよ",
        "機器構成図から、○○の上流コンポーネントを答えよ",
    ],
    "flowchart": [
        "フロー図のStep3で判断分岐がYesの場合、次に実行されるステップは",
        "○○プロセスにおいて、決定権を持つステップを答えよ",
    ],
    "chart_graph": [
        "グラフから最大値を示した年度を答えよ",
        "○年と△年の値の差分を読み取って答えよ",
    ],
    "complex_layout": [
        "サイドノートの注釈が本文○○についてどう補足しているか答えよ",
        "段組み右側に置かれた条件と本文の規定の対応関係を答えよ",
    ],
    "large_enumeration": [
        "列挙された項目のうち○○の条件を満たすものを全て挙げよ",
        "○○に該当しない項目を抜き出して答えよ",
    ],
    # --- Abstention ---
    "insufficient_evidence": [
        "本文に存在しない情報(他社事例・最新法改正など)を要求する問い。回答は『本ドキュメントには該当する情報がありません』で始めること",
        "文書では言及されていない一般常識を要求する問い",
    ],
    "contradictory_evidence": [
        "旧版規程と新版規程で矛盾する記述がある場合、どちらが優先するか。回答は矛盾の存在を明示すること",
        "複数箇所の記述が両立しない場合の取扱いを問う",
    ],
    "fragmented_chunk": [
        "チャンク境界で途切れた条文の続きを推測する問い。回答は『チャンク末尾で記述が分断されている』ことを明示すること",
        "ページまたぎ・章またぎで分断された記述を扱う問い",
    ],
}


# 観点ごとの「✗ 避けるべき問いのパターン」(LLM がよく逃げる trivial 形)
ASPECT_BAD_PATTERNS: dict[str, list[str]] = {
    # --- Integration ---
    "multi_source_integration": [
        "1チャンクだけ見れば答えられる単純抽出",
        "複数情報源の統合が必要ない、片方の情報だけで完結する問い",
    ],
    "multi_doc_reference": [
        "元チャンクが単一文書のみの場合は無理に作問しない。`insufficient_evidence` に倒すこと",
        "1チャンクで完結する単純抽出",
    ],
    "remote_reference": [
        "同じ条文内・同じセクション内で完結する問い",
        "離れた箇所への参照が問いの解決に不要な問い",
    ],
    "standards_reference": [
        "規格番号(JIS/ISO/IEC/ASME等)が元チャンクに無い場合は無理に作問しない。`insufficient_evidence` に倒すこと",
        "規格番号を引用するだけで内容との対応が問われていない問い",
    ],
    # --- Reasoning ---
    "quantitative_calc": [
        "文書にそのまま書いてある数値を抽出するだけ(計算ステップが無い)",
        "計算問題に見えるが実は単一値の抜き出しで完結する問い",
    ],
    "multi_hop": [
        "1チャンクで答えが完結する単純抽出",
        "中間結論を経由せず、直接答えに到達する問い",
    ],
    "negation": [
        "肯定形で答えられる、否定が本質でない問い",
        "『○○である』とだけ問う、除外・例外の判断が不要な問い",
    ],
    "causal": [
        "結果だけ聞いて理由(根拠条文)を要求しない問い",
        "『何が起きるか』だけで『なぜそうなるか』を要求しない問い",
    ],
    "temporal": [
        "単なる日付・時刻の抽出で時系列の順序が問われていない問い",
        "前後関係を問わずに単独の時点を聞くだけ",
    ],
    "comparison_conditional": [
        "1ケースだけ聞いて比較対象が無い問い",
        "条件分岐が無く、単一条件だけで完結する問い",
    ],
    # --- Logic ---
    "synonym_interpretation": [
        "同義語ペアが元チャンクに明示されていないのに無理に作問する",
        "用語の単純な定義抽出で言い換えが関与しない問い",
    ],
    "numeric_inclusion": [
        "数値の単純抽出だけで範囲・包含関係が問われていない問い",
        "『○分です』と数値だけ答えさせる問い",
    ],
    "concept_inclusion": [
        "単純な定義抽出で包含関係(上位下位)が無い問い",
        "別概念どうしを並べるだけで、包含が成立しない関係を扱う",
    ],
    "vocabulary_mismatch": [
        "表記揺れ・略語が元チャンクに無いのに混乱を装う問い",
        "用語が一致している箇所で無理に表記の差を問う",
    ],
    "abstraction_gap": [
        "抽象問いに対して抽象的にだけ答える(具体条項に落とせていない)問い",
        "具体的すぎる抜き出しで抽象-具体の橋渡しが無い問い",
    ],
    # --- Figure (★ 事故が多発する) ---
    "simple_table": [
        "「表に何行あるか」「項目の個数は何個か」など、列挙の数を数えるだけの問い",
        "表の見出し名をそのまま答えさせる単純抽出",
    ],
    "complex_form": [
        "ヘッダ名や項目名をそのまま抽出するだけで階層・結合の理解が要らない問い",
        "セル結合・階層構造を一切活用しない単純抽出",
    ],
    "concept_diagram": [
        "概念図・構成図・ブロック図の素材(『構成図』『ブロック図』等の語、図キャプション)が元チャンクに無い場合は無理に作問しない。`insufficient_evidence` に倒すこと",
        "図に描かれた要素間の関係性を問わずに、単に登場物を列挙させる問い",
    ],
    "flowchart": [
        "フロー記号・ステップ番号・分岐表記が元チャンクに無い場合は無理に作問しない。`insufficient_evidence` に倒すこと",
        "ステップ名を抽出するだけで、順序・分岐の理解が不要な問い",
    ],
    "chart_graph": [
        "グラフ凡例・軸単位・系列名が元チャンクに無い場合は無理に作問しない。`insufficient_evidence` に倒すこと",
        "数値読み取りが不要な、グラフのタイトルを答えるだけの問い",
    ],
    "complex_layout": [
        "「カレンダーに月が何個あるか」「表に何項目あるか」など、列挙の個数を数えるだけの問い ★最頻出の事故パターン",
        "「ページに何が書かれているか」のような抽象的すぎる総括問い",
        "サイドノート・段組み・図表混在のような構造要素が元チャンクに無い場合は無理に作問しない",
    ],
    "large_enumeration": [
        "「全部で何項目あるか」「○番目の項目は何か」など、個数や順番を抽出するだけの問い",
        "列挙の中身の意味的な条件分け(該当する/しない)が不要な単純抽出",
    ],
    # --- Abstention ---
    "insufficient_evidence": [
        "本文内で答えられる問いを無理に Abstention に倒すこと",
        "あまりに本文と無関係な問い(評価として無意味)",
    ],
    "contradictory_evidence": [
        "実際には矛盾していない複数記述を矛盾と装う問い",
        "単に異なる事項を扱った別記述を矛盾扱いする問い",
    ],
    "fragmented_chunk": [
        "実際にはチャンク境界で分断されていない箇所を分断扱いにする問い",
        "本来1チャンクで完結している問いを分断問題と装う",
    ],
}


# Display labels (日本語、グラフやレビューUI用)
ASPECT_LABELS: dict[str, str] = {
    "multi_source_integration": "複数情報源の統合",
    "multi_doc_reference":      "マルチドキュメント参照",
    "remote_reference":         "遠隔参照(章をまたぐ)",
    "standards_reference":      "規格・規定番号参照",
    "quantitative_calc":        "数値計算",
    "multi_hop":                "マルチホップ",
    "negation":                 "否定推論",
    "causal":                   "因果推論",
    "temporal":                 "時間推論",
    "comparison_conditional":   "比較・条件判断",
    "synonym_interpretation":   "同義関係の解釈",
    "numeric_inclusion":        "数値包含関係",
    "concept_inclusion":        "概念包含関係",
    "vocabulary_mismatch":      "語彙ミスマッチ(専門用語・略語)",
    "abstraction_gap":          "抽象度の乖離",
    "simple_table":             "単純表",
    "complex_form":             "複雑帳票",
    "concept_diagram":          "概念図・構成図",
    "flowchart":                "フローチャート",
    "chart_graph":              "グラフ・チャート",
    "complex_layout":           "複雑レイアウト",
    "large_enumeration":        "大量列挙",
    "insufficient_evidence":    "根拠不足",
    "contradictory_evidence":   "根拠の矛盾",
    "fragmented_chunk":         "不完全なチャンク区切り",
}


# 業務シナリオ参考値 (QAItem.business_scenario の選択肢ヒント)
BUSINESS_SCENARIOS: tuple[str, ...] = (
    "設計変更影響評価",
    "監査対応・規定遵守確認",
    "不具合原因調査",
    "安全基準レビュー",
    "新規部品・材料選定",
    "工程設計レビュー",
    "教育・OJT",
    "顧客問合せ対応",
    "規格適合性確認",
    "見積・調達仕様書作成",
)


def aspects_for_category(category: CategoryName) -> list[str]:
    return [a for a, c in ASPECT_TO_CATEGORY.items() if c == category]
