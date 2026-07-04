# -*- coding: utf-8 -*-
"""合成コーパス生成器: 架空の「白波火力発電所」文書一式を決定論で生成する.

目的: KG-PoC の機構実証用に、6 関係カテゴリすべての素材を構造から保証した
コーパスを作る。相互参照はテーブル駆動で生成するので、参照切れが原理的に
起きない。正解エッジ一覧 (_ground_truth_edges.json) を副産物として出力し、
QA 作問と KG 構築側の両方の正解データとする。

安全条件 (プロジェクトルール):
- 出力先は data/chunks_synth/ 専用。実コーパス (chunks_plant / chunks) と混ぜない
- 固有名はすべて架空 (白波/青灘火力発電所、株式会社シラナミエナジー)
- 全チャンクの doc_id に SYN プレフィックスを付け、一目で合成と分かる

実行: uv run python scripts/gen_synth_corpus.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

OUT_DIR = Path("data/chunks_synth")

# ============================================================
# 世界設定 (すべて架空)
# ============================================================

# 機器台帳: (機器ID, 正式名称, 系統, 種別, 部位リスト)
EQUIPMENT = [
    ("B-1",   "第1ボイラー設備",          "汽力系統", "ボイラー",     ["火炉", "過熱器", "再熱器", "節炭器", "ドラム"]),
    ("B-2",   "第2ボイラー設備",          "汽力系統", "ボイラー",     ["火炉", "過熱器", "再熱器", "節炭器", "ドラム"]),
    ("ST-1",  "第1蒸気タービン発電設備",   "汽力系統", "蒸気タービン", ["車室", "ロータ", "軸受", "調速装置"]),
    ("ST-2",  "第2蒸気タービン発電設備",   "汽力系統", "蒸気タービン", ["車室", "ロータ", "軸受", "調速装置"]),
    ("GT-1",  "第1ガスタービン発電設備",   "ガス系統", "ガスタービン", ["圧縮機", "燃焼器", "タービン翼"]),
    ("GT-2",  "第2ガスタービン発電設備",   "ガス系統", "ガスタービン", ["圧縮機", "燃焼器", "タービン翼"]),
    ("GT-3",  "第3ガスタービン発電設備",   "ガス系統", "ガスタービン", ["圧縮機", "燃焼器", "タービン翼"]),
    ("FWP-1", "第1給水ポンプ",            "給水系統", "ポンプ",       ["ケーシング", "羽根車", "軸封部"]),
    ("FWP-2", "第2給水ポンプ",            "給水系統", "ポンプ",       ["ケーシング", "羽根車", "軸封部"]),
    ("FGC-1", "第1燃料ガス圧縮機",         "燃料系統", "圧縮機",       ["シリンダ", "ピストン", "冷却器"]),
    ("SCR-1", "脱硝装置",                 "環境系統", "環境設備",     ["触媒層", "アンモニア注入装置"]),
    ("EP-1",  "電気集じん装置",           "環境系統", "環境設備",     ["放電極", "集じん極", "槌打装置"]),
    ("EDG-1", "非常用ディーゼル発電設備",  "所内系統", "内燃機関",     ["シリンダ", "過給機", "燃料噴射弁"]),
]

# 用語集: (正式名称, 社内略称, 旧称) — 名寄せ (⑤SAME_AS) の素材
GLOSSARY = [
    ("第3ガスタービン発電設備", "GT-3", "三号瓦斯機"),
    ("第1給水ポンプ", "FWP-1", "一号缶水ポンプ"),
    ("脱硝装置", "SCR-1", "排煙脱硝設備"),
    ("電気集じん装置", "EP-1", "集塵機"),
    ("非常用ディーゼル発電設備", "EDG-1", "非常用発電機"),
    ("定期事業者検査", "定検", "法定検査"),
    ("使用前自主確認", "使用前確認", "竣工検査"),
    ("水圧耐圧試験", "水圧試験", "耐水圧検査"),
    ("燃料ガス漏えい検知装置", "ガス検知器", "ガスリークセンサ"),
    ("保全部設備管理課", "設管課", "保修課"),
]

# ============================================================
# 白波規程 SHK-R-001: 条文テーブル
# 各条: (条番号, 見出し, 本文生成種別, パラメータ)
# 本文種別:
#   plain     : 単独で完結する規定
#   form_ref  : 「所定の様式により…提出」(様式番号はマップにのみ存在 → 2ホップ)
#   junyou    : 他条の準用 (読み替えつき)
#   appendix  : 別表参照
# ============================================================

# (条, 見出し, 種別, dict)
ARTICLES = [
    (1, "目的", "plain", {"text": "この規程は、株式会社シラナミエナジー白波火力発電所における発電用設備の保全管理に関する基本的事項を定め、設備の安全かつ安定な運転を確保することを目的とする。"}),
    (2, "適用範囲", "plain", {"text": "この規程は、白波火力発電所に設置された発電用設備のうち、機器台帳(SHK-T-002)に登録されたすべての設備に適用する。ただし、建屋及び構内道路等の土木建築設備には適用しない。"}),
    (3, "用語の定義", "plain", {"text": "この規程において使用する用語は、白波火力発電所用語集(SHK-G-003)に定めるところによる。用語集に定めのない用語は、発電用火力設備に関する一般的な慣用によるものとする。"}),
    (4, "保全管理体制", "plain", {"text": "設備保全の統括責任者は保全部長とし、系統ごとの保全実施責任者は保全部設備管理課長とする。保全実施責任者は、毎月の保全実施状況を統括責任者に報告しなければならない。"}),

    (5, "ボイラー設備の材料", "plain", {"text": "ボイラー設備の耐圧部分に使用する材料は、最高使用圧力及び最高使用温度において十分な機械的強度及び化学的安定性を有するものでなければならない。フェライト系耐熱鋼を過熱器管に使用する場合は、使用温度上限を580度とする。"}),
    (6, "ボイラー設備の水圧耐圧試験", "form_ref", {"subject": "ボイラー設備の耐圧部分の水圧耐圧試験", "action": "最高使用圧力の1.4倍の圧力で30分間保持し、漏えい及び異常変形がないことを確認する", "record": "試験の結果"}),
    (7, "安全弁の整定", "form_ref", {"subject": "ボイラー設備の安全弁の吹出し圧力の整定", "action": "ドラム安全弁は最高使用圧力以下、過熱器安全弁はドラム安全弁に先行して動作する圧力に整定する", "record": "整定の結果"}),
    (8, "給水装置の点検", "form_ref", {"subject": "給水ポンプ及び給水制御装置の機能点検", "action": "毎月1回、予備機への自動切替動作を含む機能試験を行う", "record": "点検の結果"}),
    (9, "ボイラー計測装置", "plain", {"text": "ボイラー設備の運転状態を計測する装置として、ドラム水位、ドラム圧力、過熱器出口蒸気温度及び再熱器出口蒸気温度を常時計測し、中央制御室に表示しなければならない。"}),
    (10, "ボイラー水質管理", "appendix", {"subject": "ボイラー給水及び缶水の水質", "appendix_no": 1, "appendix_name": "水質管理基準値表"}),
    (11, "ボイラー定期点検周期", "appendix", {"subject": "ボイラー設備各部位の定期点検の周期", "appendix_no": 2, "appendix_name": "定期点検周期表"}),
    (12, "ボイラー補修溶接", "form_ref", {"subject": "ボイラー耐圧部分の補修溶接", "action": "溶接施工要領書を保全部長に提出して承認を受けた後でなければ施工してはならない。施工後は非破壊検査を実施する", "record": "非破壊検査の結果"}),

    (15, "蒸気タービンの材料", "junyou", {"subject": "蒸気タービン設備の耐圧部分に使用する材料", "target": 5, "yomikae": ("ボイラー設備", "蒸気タービン設備")}),
    (16, "蒸気タービンの水圧耐圧試験", "junyou", {"subject": "蒸気タービン車室の水圧耐圧試験", "target": 6, "yomikae": ("1.4倍", "1.2倍")}),
    (17, "非常調速装置の試験", "form_ref", {"subject": "蒸気タービン非常調速装置の動作試験", "action": "定格回転速度の1.11倍以下で確実に動作することを年1回確認する", "record": "試験の結果"}),
    (18, "タービン軸受の監視", "plain", {"text": "蒸気タービンの軸受温度及び軸振動は常時監視とし、警報設定値は軸受メタル温度95度、軸振動125マイクロメートルとする。警報発生時は速やかに負荷を降下させる。"}),
    (19, "タービン定期点検周期", "appendix", {"subject": "蒸気タービン設備各部位の定期点検の周期", "appendix_no": 2, "appendix_name": "定期点検周期表"}),

    (25, "ガスタービンの材料", "junyou", {"subject": "ガスタービン設備の耐圧部分に使用する材料", "target": 5, "yomikae": ("ボイラー設備", "ガスタービン設備")}),
    (26, "ガスタービンの水圧耐圧試験", "junyou", {"subject": "ガスタービン車室の水圧耐圧試験", "target": 6, "yomikae": ("1.4倍の圧力で30分間", "1.2倍の圧力で10分間")}),
    (27, "燃料ガス漏えい検知", "form_ref", {"subject": "燃料ガス漏えい検知装置の機能点検", "action": "検知器ごとに標準ガスによる校正を6か月に1回実施し、警報設定値は爆発下限界の4分の1とする", "record": "校正の結果"}),
    (28, "燃料ガス圧縮機の開放点検", "form_ref", {"subject": "燃料ガス圧縮機の開放点検", "action": "運転時間20,000時間ごと又は4年ごとのいずれか早い時期に実施する", "record": "開放点検の結果"}),
    (29, "ガスタービン翼の検査", "form_ref", {"subject": "ガスタービン高温部品(燃焼器及びタービン翼)の検査", "action": "等価運転時間24,000時間ごとに抜取検査を行い、き裂その他の欠陥の有無を確認する", "record": "検査の結果"}),

    (35, "脱硝装置の性能管理", "form_ref", {"subject": "脱硝装置の脱硝効率の測定", "action": "毎年1回性能試験を実施し、脱硝効率80パーセント以上を維持していることを確認する", "record": "性能試験の結果"}),
    (36, "電気集じん装置の管理", "form_ref", {"subject": "電気集じん装置の集じん効率の測定", "action": "2年に1回性能試験を実施する。放電極の断線本数が全体の5パーセントを超えた場合は交換する", "record": "性能試験の結果"}),
    (37, "排ガス測定", "appendix", {"subject": "ばい煙量及びばい煙濃度の測定項目、頻度", "appendix_no": 3, "appendix_name": "排ガス測定基準表"}),

    (41, "非常用電源設備の試験", "form_ref", {"subject": "非常用ディーゼル発電設備の始動試験", "action": "毎月1回無負荷始動試験、6か月に1回実負荷試験を実施し、始動所要時間が40秒以内であることを確認する", "record": "試験の結果"}),
    (42, "保護継電器の試験", "form_ref", {"subject": "発電機保護継電器の動作特性試験", "action": "2年に1回、整定値に対する動作値の誤差が5パーセント以内であることを確認する", "record": "試験の結果"}),

    (47, "定期事業者検査", "plain", {"text": "定期事業者検査は、ボイラー設備及び蒸気タービン設備にあっては2年ごと、ガスタービン設備にあっては3年ごとに実施する。検査の実施に当たっては、あらかじめ検査計画書を作成し、保全部長の承認を受けなければならない。"}),
    (48, "検査記録の保存", "plain", {"text": "この規程に基づき作成した検査及び試験の記録は、次回の同種検査の完了後まで、かつ、作成の日から5年間保存しなければならない。ただし、第12条の非破壊検査記録は当該設備の廃止まで保存する。"}),
    (49, "検査結果の報告", "form_ref", {"subject": "定期事業者検査の結果の統括責任者への報告", "action": "検査完了後30日以内に報告する", "record": "検査結果の総括"}),

    (57, "不適合の報告", "form_ref", {"subject": "設備の不適合(故障、損傷、機能低下等)を発見した場合の報告", "action": "発見者は速やかに保全実施責任者に報告し、保全実施責任者は24時間以内に統括責任者に報告する", "record": "不適合の内容"}),
    (58, "原因分析と是正処置", "form_ref", {"subject": "不適合の原因分析及び是正処置", "action": "保全実施責任者は不適合の根本原因を分析し、再発防止のための是正処置を立案・実施する", "record": "原因分析及び是正処置の内容"}),
    (59, "水平展開", "plain", {"text": "是正処置のうち同種設備に共通に適用すべきものは、保全実施責任者が水平展開計画を作成し、対象設備のすべてについて実施状況を確認しなければならない。水平展開の完了は統括責任者が確認する。"}),

    (63, "規程の改廃", "plain", {"text": "この規程の改廃は、保全部長の発議により、設備保全委員会の審議を経て所長が決定する。この規程は白波火力発電所設備保全管理規程(旧版SHK-R-000、2018年制定)を廃止し、これに代わるものである。"}),
]

# 様式マップ: form_ref 種別の条に順番に様式番号を割り当てる (7号から開始して実在感を出す)
FORM_SUBJECTS = {
    6:  ("ボイラー水圧耐圧試験記録", "保全部長", "5年"),
    7:  ("安全弁整定記録", "保全実施責任者", "5年"),
    8:  ("給水装置機能点検記録", "保全実施責任者", "3年"),
    12: ("補修溶接非破壊検査記録", "保全部長", "設備廃止まで"),
    17: ("非常調速装置動作試験記録", "保全部長", "5年"),
    27: ("燃料ガス漏えい検知装置校正記録", "保全実施責任者", "3年"),
    28: ("燃料ガス圧縮機開放点検記録", "保全部長", "5年"),
    29: ("ガスタービン高温部品検査記録", "保全部長", "設備廃止まで"),
    35: ("脱硝効率性能試験記録", "保全実施責任者", "5年"),
    36: ("集じん効率性能試験記録", "保全実施責任者", "5年"),
    41: ("非常用電源始動試験記録", "保全実施責任者", "3年"),
    42: ("保護継電器試験記録", "保全部長", "5年"),
    49: ("定期事業者検査結果総括表", "統括責任者", "10年"),
    57: ("不適合報告書", "統括責任者", "10年"),
    58: ("是正処置報告書", "統括責任者", "10年"),
}

# ============================================================
# 不適合事例 (④因果): (ID, 設備呼称(略称/旧称を混ぜる), 事象, 原因, 違反/関連条文, 是正)
# ============================================================
INCIDENTS = [
    ("NCR-2023-01", "GT-3", "起動時に燃焼器の失火が頻発", "燃料ガス漏えい検知装置の校正期限超過により誤警報でガス遮断弁が閉止していた", 27, "校正計画を年間保全計画に組み込み、期限管理を設管課の月次確認事項とした"),
    ("NCR-2023-02", "一号缶水ポンプ", "軸封部から蒸気漏えい", "軸封部の定期取替周期を超過して運転を継続していた", 11, "定期点検周期表の周期を機器台帳と連動させ、超過時は自動的に作業指示が発行される仕組みとした"),
    ("NCR-2023-03", "B-2", "過熱器管1本に膨出を発見", "水質管理基準値の逸脱(缶水pH低下)が3日間放置されていた", 10, "水質基準値逸脱時の当直から設管課への通報基準を明確化した"),
    ("NCR-2024-01", "三号瓦斯機", "タービン翼に規定を超えるき裂", "等価運転時間の算定誤りにより高温部品検査が7,000時間遅延していた", 29, "等価運転時間の算定を手計算から運転データ自動集計に変更した"),
    ("NCR-2024-02", "非常用発電機", "実負荷試験で始動所要時間が55秒", "燃料噴射弁の劣化。無負荷始動試験のみでは検出できなかった", 41, "実負荷試験の頻度を6か月から3か月に短縮する所内基準を追加した"),
    ("NCR-2024-03", "EP-1", "集じん効率の急低下", "放電極の断線本数が管理基準の5パーセントを超えていたが交換が未実施だった", 36, "断線率4パーセント到達時点で交換計画を立案する予防基準を新設した"),
    ("NCR-2024-04", "ST-1", "軸振動警報の頻発", "軸受メタルの部分摩耗。定期点検周期表の軸受開放周期を超えて運転していた", 11, "周期超過機器の一覧を毎月の保全会議で確認する運用とした"),
    ("NCR-2025-01", "FGC-1", "シリンダ内圧力の異常上昇", "開放点検(20,000時間ごと)の起算時間の記録誤り", 28, "運転時間の起算点を様式に明記する欄を追加した"),
]

# ============================================================
# 作業手順書 (③手順): 工程リスト (番号, 工程名, 前提, 記録様式条)
# ============================================================
PROCEDURE_BOILER_START = [
    (1, "起動前弁体ラインナップ確認", None, None),
    (2, "ボイラー水張り及び水位確認", 1, None),
    (3, "缶水水質確認", 2, 10),
    (4, "点火前パージ(炉内換気5分以上)", 3, None),
    (5, "点火及び昇圧開始", 4, None),
    (6, "昇圧中の膨張量確認", 5, None),
    (7, "安全弁整定値の確認", 6, 7),
    (8, "タービン通気及び併入", 7, None),
]

# ============================================================
# 生成ヘルパ
# ============================================================

def _split_chunks(doc_id: str, blocks: list[tuple[str, str]], max_chars: int = 480) -> list[dict]:
    """(section, text) ブロック列をチャンク列に変換。長いブロックは分割する。"""
    chunks = []
    pos = 0
    for section, text in blocks:
        parts = []
        t = text.strip()
        while len(t) > max_chars:
            cut = t.rfind("。", 0, max_chars)
            if cut < max_chars // 2:
                cut = max_chars
            parts.append(t[: cut + 1])
            t = t[cut + 1:]
        parts.append(t)
        for p in parts:
            if not p.strip():
                continue
            refs = sorted(set(re.findall(r"第\d+条|様式第\d+号|別表第\d+", p)))
            chunks.append({
                "chunk_id": f"{doc_id}__c{pos:04d}",
                "doc_id": doc_id,
                "page": None,
                "text": p.strip(),
                "position": pos,
                "section_path": [section] if section else [],
                "references": refs,
            })
            pos += 1
    return chunks


def build_regulation(doc_id: str, plant: str, reg_no: str, form_offset: int, article_offset: int):
    """規程本体を生成。form_offset/article_offset で青灘 (妨害用) の番号をずらす。"""
    # 様式番号の割り当て (条番号順に 6号 から)
    form_map = {}
    n = 6 + form_offset
    for art in sorted(FORM_SUBJECTS):
        form_map[art + article_offset] = (n, *FORM_SUBJECTS[art])
        n += 1

    blocks = []
    blocks.append(("", f"{plant}火力発電所 設備保全管理規程 ({reg_no}) 株式会社シラナミエナジー {plant}火力発電所 制定: 2024年4月1日 この文書は架空の演習用文書である。"))
    edges = []
    for art, title, kind, prm in ARTICLES:
        a = art + article_offset
        if kind == "plain":
            body = prm["text"]
        elif kind == "form_ref":
            body = (f"{prm['subject']}は、{prm['action']}。"
                    f"{prm['record']}は、所定の様式により記録を作成し、遅滞なく提出しなければならない。"
                    f"様式の種別及び提出先は第{66 + article_offset}条に定めるところによる。")
            edges.append({"type": "article_form", "doc_id": doc_id, "article": f"第{a}条",
                          "form": f"様式第{form_map[a][0]}号", "form_name": form_map[a][1]})
        elif kind == "junyou":
            t = prm["target"] + article_offset
            y = prm["yomikae"]
            body = (f"{prm['subject']}については、第{t}条の規定を準用する。"
                    f"この場合において、同条中「{y[0]}」とあるのは「{y[1]}」と読み替えるものとする。")
            edges.append({"type": "junyou", "doc_id": doc_id, "article": f"第{a}条", "target": f"第{t}条",
                          "yomikae_from": y[0], "yomikae_to": y[1]})
        elif kind == "appendix":
            body = (f"{prm['subject']}は、別表第{prm['appendix_no']}({prm['appendix_name']})に定めるところによる。"
                    f"別表の値を逸脱した場合は、第{57 + article_offset}条の不適合として扱う。")
            edges.append({"type": "article_appendix", "doc_id": doc_id, "article": f"第{a}条",
                          "appendix": f"別表第{prm['appendix_no']}", "name": prm["appendix_name"]})
        blocks.append((f"第{a}条({title})", f"第{a}条 {body}"))

    # 様式マップ (第66条相当、文書末尾 = 条文から位置的に遠い)。
    # 実コーパス (特定設備省令) と同じく様式は無名で「(第N条関係)」のみ。
    # 名前を書くと様式チャンク単独で主題が特定でき 3 ホップが崩れる。
    map_lines = [f"第{66 + article_offset}条 この規程に基づく記録の様式は次のとおりとし、様式の実体は{plant}火力発電所検査様式集に定める。"]
    for a in sorted(form_map):
        no, name, dest, keep = form_map[a]
        map_lines.append(f"様式第{no}号(第{a}条関係)")
    blocks.append((f"第{66 + article_offset}条(様式)", " ".join(map_lines)))
    return blocks, form_map, edges


def build_forms_doc(doc_id: str, plant: str, form_map: dict, article_offset: int):
    blocks = [("", f"{plant}火力発電所 検査様式集 (架空の演習用文書)。本様式集は設備保全管理規程に基づく記録様式の記載事項、提出先及び保存期間を定める。")]
    edges = []
    for a in sorted(form_map):
        no, name, dest, keep = form_map[a]
        # 様式実体も無名 (主題語彙を含めない)。含めると条文・マップを飛ばして
        # 様式チャンク単独で答えられてしまい、参照ホップの評価にならない。
        body = (f"様式第{no}号 記載事項: 実施年月日、対象設備の機器ID、実施者及び立会者、実施内容の要点、判定結果。"
                f"提出先: {dest}。保存期間: {keep}。"
                f"本様式は電子様式管理システムに登録された最新版を使用すること。")
        blocks.append((f"様式第{no}号", body))
        edges.append({"type": "form_detail", "doc_id": doc_id, "form": f"様式第{no}号",
                      "form_name": name, "dest": dest, "keep": keep})
    return blocks, edges


def build_glossary(doc_id: str):
    blocks = [("", "白波火力発電所 用語集 (SHK-G-003) (架空の演習用文書)。本用語集は所内文書で使用する用語の正式名称、社内略称及び旧称の対応を定める。旧称は2018年以前の文書にのみ使用されており、現行文書では正式名称又は社内略称を使用する。")]
    edges = []
    for official, abbr, old in GLOSSARY:
        blocks.append((official, f"正式名称: {official}。社内略称: {abbr}。旧称: {old}。旧称「{old}」は現行文書では使用しない。"))
        edges.append({"type": "alias", "doc_id": doc_id, "official": official, "abbr": abbr, "old": old})
    return blocks, edges


def build_registry(doc_id: str):
    blocks = [("", "白波火力発電所 機器台帳 (SHK-T-002) (架空の演習用文書)。設備保全管理規程第2条の適用対象設備を登録する。")]
    edges = []
    for eq_id, name, system, kind, parts in EQUIPMENT:
        body = (f"機器ID: {eq_id}。正式名称: {name}。所属系統: {system}。設備種別: {kind}。"
                f"主要部位: {'、'.join(parts)}。")
        blocks.append((eq_id, body))
        edges.append({"type": "equipment", "doc_id": doc_id, "id": eq_id, "name": name,
                      "system": system, "kind": kind, "parts": parts})
    return blocks, edges


def build_incidents(doc_id: str):
    blocks = [("", "白波火力発電所 不適合事例集 (SHK-N-005) (架空の演習用文書)。設備保全管理規程第57条及び第58条に基づき報告された不適合の記録。")]
    edges = []
    for ncr, equip, event, cause, art, corrective in INCIDENTS:
        body = (f"{ncr}。対象設備: {equip}。事象: {event}。"
                f"原因: {cause}(設備保全管理規程第{art}条関連)。"
                f"是正処置: {corrective}。水平展開: 同種設備について実施済み。")
        blocks.append((ncr, body))
        edges.append({"type": "incident", "doc_id": doc_id, "ncr": ncr, "equipment": equip,
                      "cause_article": f"第{art}条", "event": event})
    return blocks, edges


def build_procedure(doc_id: str):
    blocks = [("", "白波火力発電所 ボイラー起動作業手順書 (SHK-P-011) (架空の演習用文書)。B-1及びB-2ボイラーの冷態起動に適用する。各工程は前提工程の完了確認後でなければ着手してはならない。")]
    edges = []
    for no, name, prereq, form_art in PROCEDURE_BOILER_START:
        parts = [f"工程{no}: {name}。"]
        if prereq:
            parts.append(f"前提: 工程{prereq}の完了確認。")
            edges.append({"type": "precedes", "doc_id": doc_id, "before": f"工程{prereq}", "after": f"工程{no}"})
        if form_art:
            parts.append(f"本工程の確認結果は設備保全管理規程第{form_art}条に基づく記録の対象とする。")
            edges.append({"type": "step_article", "doc_id": doc_id, "step": f"工程{no}", "article": f"第{form_art}条"})
        parts.append("本工程の完了は当直長が確認し、運転日誌に記載する。")
        blocks.append((f"工程{no}", " ".join(parts)))
    return blocks, edges


def build_appendices(doc_id: str):
    blocks = [("", "白波火力発電所 設備保全管理規程 別表集 (架空の演習用文書)。")]
    blocks.append(("別表第1", "別表第1(水質管理基準値表)(第10条関係) ボイラー給水: pH 8.5以上9.6以下、溶存酸素 7ppb以下、鉄 20ppb以下。缶水: pH 9.0以上9.8以下、シリカ 0.3ppm以下、電気伝導率 25マイクロジーメンス毎センチメートル以下。基準値を逸脱した場合は2時間以内に設管課へ通報する。"))
    blocks.append(("別表第2", "別表第2(定期点検周期表)(第11条及び第19条関係) ボイラー火炉・過熱器・再熱器: 2年ごと開放点検。ドラム: 2年ごと内部点検。給水ポンプ軸封部: 2年ごと取替。蒸気タービン車室: 4年ごと開放。蒸気タービン軸受: 2年ごと開放点検。調速装置: 1年ごと動作確認。"))
    blocks.append(("別表第3", "別表第3(排ガス測定基準表)(第37条関係) ばいじん濃度: 2か月に1回。硫黄酸化物: 常時(連続分析計)。窒素酸化物: 常時(連続分析計)。連続分析計の校正は6か月に1回とする。"))
    edges = [
        {"type": "appendix_detail", "doc_id": doc_id, "appendix": "別表第1", "articles": ["第10条"]},
        {"type": "appendix_detail", "doc_id": doc_id, "appendix": "別表第2", "articles": ["第11条", "第19条"]},
        {"type": "appendix_detail", "doc_id": doc_id, "appendix": "別表第3", "articles": ["第37条"]},
    ]
    return blocks, edges


def build_old_regulation(doc_id: str):
    """旧版 (廃止済み)。様式番号が現行とずれている = 改廃 distractor + SUPERSEDES 素材。"""
    blocks = [("", "白波火力発電所 設備保全管理規程 (旧版 SHK-R-000) 制定: 2018年。本規程は2024年4月1日をもって廃止され、SHK-R-001に置き換えられた。記録の様式番号は現行規程と異なるため、2024年3月以前の記録の参照時のみ使用すること。(架空の演習用文書)")]
    olds = [(6, 3, "ボイラー水圧耐圧試験記録"), (17, 9, "調速機試験記録"), (41, 15, "非常用電源試験記録")]
    for art, old_no, name in olds:
        blocks.append((f"旧様式第{old_no}号", f"旧規程第{art}条に基づく記録は旧様式第{old_no}号({name})により作成していた。現行規程では様式番号が変更されているため、現行の様式は現行規程の様式の条を参照すること。"))
    edges = [{"type": "supersedes", "doc_id": doc_id, "old": "SHK-R-000", "new": "SHK-R-001"}]
    return blocks, edges


def build_patrol_records(doc_id: str, plant: str, year: int, half: int):
    """点検実績記録 (ノイズ兼リアリティ用)。機器を略称/正式名を混ぜて言及するが、
    様式マップの情報は含まない。検索空間を現実的な密度にするための量的素材。"""
    months = range(1, 7) if half == 1 else range(7, 13)
    statuses = [
        "異常なし", "軽微な発錆を確認し清掃実施", "指示値正常", "にじみを確認し監視継続",
        "振動値は管理値内", "異音なし", "絶縁抵抗値良好", "作動試験良好",
    ]
    blocks = [("", f"{plant}火力発電所 定期巡視点検実績記録 {year}年{'上' if half == 1 else '下'}期 (架空の演習用文書)。設備保全管理規程第4条に基づく月次報告の添付記録。")]
    for m in months:
        lines = [f"{year}年{m}月度 定期巡視点検実績。"]
        for i, (eq_id, name, system, kind, parts) in enumerate(EQUIPMENT):
            label = eq_id if (i + m) % 3 == 0 else name
            st = statuses[(i * 7 + m * 3) % len(statuses)]
            part = parts[(i + m) % len(parts)]
            lines.append(f"{label} {part}: {st}。")
        blocks.append((f"{year}年{m}月", " ".join(lines)))
    return blocks


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_edges = []
    total = 0

    docs = []

    # 白波 (本命)
    reg_blocks, form_map, e1 = build_regulation("SYN01_白波_設備保全管理規程", "白波", "SHK-R-001", 0, 0)
    forms_blocks, e2 = build_forms_doc("SYN02_白波_検査様式集", "白波", form_map, 0)
    docs += [("SYN01_白波_設備保全管理規程", reg_blocks), ("SYN02_白波_検査様式集", forms_blocks)]
    all_edges += e1 + e2

    g_blocks, e3 = build_glossary("SYN03_白波_用語集")
    r_blocks, e4 = build_registry("SYN04_白波_機器台帳")
    i_blocks, e5 = build_incidents("SYN05_白波_不適合事例集")
    p_blocks, e6 = build_procedure("SYN06_白波_ボイラー起動手順書")
    a_blocks, e7 = build_appendices("SYN07_白波_別表集")
    o_blocks, e8 = build_old_regulation("SYN08_白波_旧版規程_廃止")
    docs += [("SYN03_白波_用語集", g_blocks), ("SYN04_白波_機器台帳", r_blocks),
             ("SYN05_白波_不適合事例集", i_blocks), ("SYN06_白波_ボイラー起動手順書", p_blocks),
             ("SYN07_白波_別表集", a_blocks), ("SYN08_白波_旧版規程_廃止", o_blocks)]
    all_edges += e3 + e4 + e5 + e6 + e7 + e8

    # 青灘 (妨害用: 条番号 +2、様式番号 +10 ずらし)
    reg2_blocks, form_map2, e9 = build_regulation("SYN09_青灘_設備保全管理規程", "青灘", "SHK-R-101", 10, 2)
    forms2_blocks, e10 = build_forms_doc("SYN10_青灘_検査様式集", "青灘", form_map2, 2)
    docs += [("SYN09_青灘_設備保全管理規程", reg2_blocks), ("SYN10_青灘_検査様式集", forms2_blocks)]
    all_edges += e9 + e10

    # 点検実績記録 (量的ノイズ: 2023-2025 の 5 半期 × 2 発電所)
    for plant, prefix in [("白波", "SYN2"), ("青灘", "SYN3")]:
        idx = 0
        for year, half in [(2023, 1), (2023, 2), (2024, 1), (2024, 2), (2025, 1)]:
            doc_id = f"{prefix}{idx}_{plant}_点検実績_{year}{'上' if half == 1 else '下'}期"
            docs.append((doc_id, build_patrol_records(doc_id, plant, year, half)))
            idx += 1

    for doc_id, blocks in docs:
        chunks = _split_chunks(doc_id, blocks)
        with (OUT_DIR / f"{doc_id}.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        total += len(chunks)
        print(f"{doc_id}: {len(chunks)} chunks")

    with (OUT_DIR / "_ground_truth_edges.json").open("w", encoding="utf-8") as f:
        json.dump({
            "description": "合成コーパスの正解エッジ一覧 (生成器の構造から導出)。QA 作問と KG 構築の正解データ",
            "edge_count": len(all_edges),
            "edges": all_edges,
        }, f, ensure_ascii=False, indent=1)

    print(f"\ntotal: {total} chunks, {len(all_edges)} ground-truth edges → {OUT_DIR}")


if __name__ == "__main__":
    main()
