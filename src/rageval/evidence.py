"""引用 (rationale) と原文チャンクの照合を1箇所に集約したモジュール。

これまで filter / rag_verify / webapi.service / enrich の4箇所が別々に
「引用がチャンク本文に逐語で現れるか」を実装しており、いずれも

  - 全チャンクのテキストを**連結**した1本の文字列に対して部分一致を取る
  - 引用の doc_id を見ない

という同じ弱点を持っていた。その結果、

  - doc A に帰属する引用が doc B の本文に含まれるだけで「一致」になる
  - 連結の境目をまたいだ文字列が偶然一致する (空白除去で境界が消えるため)

が実データで再現していた。本モジュールは **引用ごとに、その引用が名乗る
文書 (または chunk_id) の中だけで照合する** ことを唯一の規則とし、
連結は一切行わない。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .schema import Chunk

_WS_RE = re.compile(r"\s+")
_CHUNK_ID_SUFFIX_RE = re.compile(r"__c\d+$")


def normalize(s: str) -> str:
    """空白を除去して比較用に整える (LLM は改行・空白を勝手に入れ替えるため)。"""
    return _WS_RE.sub("", s or "")


# ---------------- 数値の表記ゆれ正規化 ----------------
# 法令コーパスは「五年」「二十年」「一・三倍」「〇・二メガパスカル」の漢数字と
# 「5年」「20年」「1.3倍」「0.2 MPa」の算用数字が混在する。回答とチャンクを
# 突き合わせるにはどちらも同じ表記に落とす必要がある。

_KANJI_DIGITS = {"〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                 "六": 6, "七": 7, "八": 8, "九": 9}
_KANJI_NUM_RUN_RE = re.compile(r"[〇一二三四五六七八九十百千]+(?:・[〇一二三四五六七八九]+)?")


def kanji_to_int(s: str) -> Optional[int]:
    """漢数字 (百一 / 十五 / 三 / 二〇二六) を整数にする。四桁までの表記のみ対応。"""
    if not s:
        return None
    # 位取り記号なしの並び (二〇二六) は桁として読む
    if all(ch in _KANJI_DIGITS for ch in s):
        return int("".join(str(_KANJI_DIGITS[ch]) for ch in s))
    section, digit = 0, 0
    for ch in s:
        if ch in _KANJI_DIGITS:
            digit = _KANJI_DIGITS[ch]
        elif ch == "十":
            section += (digit or 1) * 10
            digit = 0
        elif ch == "百":
            section += (digit or 1) * 100
            digit = 0
        elif ch == "千":
            section += (digit or 1) * 1000
            digit = 0
        else:
            return None
    return section + digit


def _kanji_run_to_ascii(m: re.Match) -> str:
    run = m.group(0)
    if "・" in run:                     # 一・三 → 1.3 (小数)
        ip, fp = run.split("・", 1)
        i = kanji_to_int(ip)
        if i is None or not all(ch in _KANJI_DIGITS for ch in fp):
            return run
        return f"{i}." + "".join(str(_KANJI_DIGITS[ch]) for ch in fp)
    n = kanji_to_int(run)
    return run if n is None else str(n)


# 単位の同義表記 (カタカナ ↔ 記号)。normalize_numbers の後に適用する。
_UNIT_SYNONYMS = [
    ("メガパスカル", "MPa"), ("キロパスカル", "kPa"), ("パスカル", "Pa"),
    ("パーセント", "%"), ("ミリメートル", "mm"), ("センチメートル", "cm"),
    ("キロワット", "kW"), ("リットル", "L"),
]


def normalize_numbers(s: str) -> str:
    """NFKC + 空白除去 + 漢数字→算用数字 + 単位の同義吸収。

    「二十年未満のものは五年」→「20年未満のものは5年」、
    「〇・二メガパスカル」→「0.2MPa」。数値を持つ固有値の照合はこの表記で行う。
    """
    s = unicodedata.normalize("NFKC", s or "")
    s = _WS_RE.sub("", s)
    s = _KANJI_NUM_RUN_RE.sub(_kanji_run_to_ascii, s)
    for kana, sym in _UNIT_SYNONYMS:
        s = s.replace(kana, sym)
    return s


# 原文の表記をそのまま拾う側 (chunker の参照抽出) が使う数字の文字クラス。
# 抽出した表記は原文照合にそのまま使うため normalize_numbers を通せない。
# 半角・全角の算用数字と漢数字を1つの定義で持ち、コーパスごとの表記の違い
# (chunks_plant の条番号は漢数字 4200 対 算用 1052) を正規表現側で吸収する。
NUMERAL_CHARS = "0-9０-９〇一二三四五六七八九十百千"


# ---------------- 根拠構造の判別 (chained / parallel) ----------------
# leave-one-out は「全根拠が必要」を証明するが「連鎖」は証明しない。
# 独立2問を「また」で束ねた問いも、両方の根拠が要るので LOO を通ってしまう。
# ここでは回答の固有値 (条番号・様式番号・数値+単位) が根拠のどこに載っているかで
#   chained  : 固有値が1つの根拠単位に集中 = 他の根拠は「見つけるため」に必要
#   parallel : 固有値が複数の根拠単位に分散 = 実体は束ね問い
# を機械判別する。

# 文書ローカルな番号トークン (第N条・様式第N・別表第N) の唯一の定義。
# **normalize_numbers を通した表記に当てる**前提なので算用数字だけで書く。
# 2026-08-08: filter が独自の正規化 (NFKC + 空白除去のみ) と独自の番号正規表現で
# 同じことをしており、漢数字の条番号を1件も拾えないまま誤結合検出が無反応に
# なっていた。両者がずれないよう定義をここに1つだけ置く。
LOCAL_NUMBER_TOKEN_PATTERNS = [
    r"第\d+条(?:の\d+)?",
    r"様式第?\d+(?:号)?",
    r"別表第?\d+",
]
LOCAL_NUMBER_TOKEN_RE = re.compile("|".join(f"(?:{p})" for p in LOCAL_NUMBER_TOKEN_PATTERNS))

_SALIENT_TOKEN_RES = [re.compile(p) for p in LOCAL_NUMBER_TOKEN_PATTERNS] + [
    # 数値+単位。単位を伴わない裸の数値は誤検出が多いので拾わない
    re.compile(r"\d+(?:\.\d+)?(?:年|か月|箇月|日|時間|分|回|件|人|号|倍|%|"
               r"mm|cm|m|MPa|kPa|Pa|kW|MW|kg|t|L|V|A|℃|度)"),
]


@dataclass
class EvidenceStructure:
    """回答の固有値と根拠単位の対応。"""
    label: str                                   # chained / parallel / unverifiable
    covering_unit: Optional[str] = None          # chained のとき固有値を全て含む単位
    token_units: dict[str, list[str]] = field(default_factory=dict)


def salient_tokens(answer: str) -> list[str]:
    """回答から固有値 (条番号・様式番号・数値+単位) を取り出す (正規化表記)。"""
    text = normalize_numbers(answer)
    out: list[str] = []
    for pat in _SALIENT_TOKEN_RES:
        for m in pat.finditer(text):
            if m.group(0) not in out:
                out.append(m.group(0))
    return out


def classify_evidence_structure(qa) -> EvidenceStructure:
    """回答の固有値が根拠のどの単位に載っているかで chained / parallel を判別する。

    単位は chunk_id、無ければ引用1件を1単位 (`doc_id#i`) とする。
    doc_id で潰すと、同一文書の別チャンクから引いた束ね問い
    (実例: auto_f9e3662a) が1単位に合流して「集中」に見えてしまう。
    固有値がどの根拠にも見つからない場合や、回答に固有値が無い場合は
    unverifiable (判定不能) とし、棄却判断には使わない。
    """
    rationales = list(getattr(qa, "rationale", []) or [])
    tokens = salient_tokens(getattr(qa, "answer", "") or "")
    if not tokens or not rationales:
        return EvidenceStructure(label="unverifiable")

    unit_texts: dict[str, str] = {}
    for i, r in enumerate(rationales):
        if isinstance(r, dict):
            chunk_id, doc_id = r.get("chunk_id"), r.get("doc_id")
            text = r.get("text")
        else:
            chunk_id, doc_id = getattr(r, "chunk_id", None), getattr(r, "doc_id", "")
            text = getattr(r, "text", None)
        unit = chunk_id or f"{normalize_doc_id(doc_id or '')}#{i}"
        unit_texts[unit] = unit_texts.get(unit, "") + "\n" + (text or "")

    token_units: dict[str, list[str]] = {}
    for tok in tokens:
        hits = [u for u, t in unit_texts.items() if tok in normalize_numbers(t)]
        if hits:
            token_units[tok] = hits
    if not token_units:
        return EvidenceStructure(label="unverifiable")

    # どれか1つの単位が、見つかった固有値を全て含むなら「集中」= chained
    for unit in unit_texts:
        if all(unit in hits for hits in token_units.values()):
            return EvidenceStructure(label="chained", covering_unit=unit,
                                     token_units=token_units)
    return EvidenceStructure(label="parallel", token_units=token_units)


def normalize_doc_id(raw: str) -> str:
    """LLM が chunk_id ('foo__c0154') を doc_id 欄に書く事故を吸収する。"""
    return _CHUNK_ID_SUFFIX_RE.sub("", raw or "")


def normalize_rationale_ids(qa) -> int:
    """QA の rationale について doc_id を正規化し、chunk_id 形式なら chunk_id 欄へ移す。

    生成LLM が doc_id 欄に `09_kagi…__c0118` のようなチャンクIDを書くことがあり、
    そのまま保存すると「同一文書の2チャンク」が「異文書」に見える。実際に
    跨ぎ判定と multi_doc ラベルの両方が誤っていた。書き換えた件数を返す。
    """
    fixed = 0
    for r in getattr(qa, "rationale", []) or []:
        raw = getattr(r, "doc_id", "") or ""
        norm = normalize_doc_id(raw)
        if norm != raw:
            if not getattr(r, "chunk_id", None):
                r.chunk_id = raw       # 元の値はチャンクIDなので所定の欄へ移す
            r.doc_id = norm
            fixed += 1
    return fixed


@dataclass
class RationaleMatch:
    """引用1件の照合結果。"""
    index: int                      # qa.rationale 内の位置
    doc_id: str                     # 引用が名乗る文書 (正規化済み)
    text: str                       # 引用本文
    matched: bool                   # 名乗った文書の中で逐語一致したか
    chunk: Optional[Chunk] = None   # 一致したチャンク (無ければ同 doc の先頭候補)
    candidates: int = 0             # 照合対象になったチャンク数 (0 = 文書自体が無い)


@dataclass
class EvidenceIndex:
    """チャンク集合を doc_id / chunk_id で引ける形にした索引。"""
    by_chunk_id: dict[str, Chunk] = field(default_factory=dict)
    by_doc_id: dict[str, list[Chunk]] = field(default_factory=dict)

    @classmethod
    def build(cls, chunks: Iterable[Chunk]) -> "EvidenceIndex":
        idx = cls()
        for c in chunks:
            idx.by_chunk_id[c.chunk_id] = c
            idx.by_doc_id.setdefault(c.doc_id, []).append(c)
        for lst in idx.by_doc_id.values():
            lst.sort(key=lambda c: c.position)
        return idx

    def candidates_for(self, doc_id: str, chunk_id: Optional[str]) -> list[Chunk]:
        """引用の照合対象になるチャンク。**他文書は決して含めない**。

        優先順: chunk_id 直指定 → doc_id → doc_id (chunk_id 接尾辞を剥がしたもの)
        """
        if chunk_id and chunk_id in self.by_chunk_id:
            return [self.by_chunk_id[chunk_id]]
        if doc_id in self.by_doc_id:
            return self.by_doc_id[doc_id]
        norm = normalize_doc_id(doc_id)
        if norm in self.by_doc_id:
            return self.by_doc_id[norm]
        # doc_id 欄に chunk_id が入っていた場合
        if doc_id in self.by_chunk_id:
            return [self.by_chunk_id[doc_id]]
        return []


def match_rationales(
    rationales: Sequence,
    chunks: Iterable[Chunk],
    *,
    index: Optional[EvidenceIndex] = None,
) -> list[RationaleMatch]:
    """引用ごとに、その引用が名乗る文書の中だけで逐語一致を確かめる。

    rationales は Rationale モデルでも dict でもよい (doc_id / chunk_id / text)。
    """
    idx = index or EvidenceIndex.build(chunks)
    out: list[RationaleMatch] = []
    for i, r in enumerate(rationales):
        if isinstance(r, dict):
            doc_id = r.get("doc_id") or ""
            chunk_id = r.get("chunk_id")
            text = r.get("text") or ""
        else:
            doc_id = getattr(r, "doc_id", "") or ""
            chunk_id = getattr(r, "chunk_id", None)
            text = getattr(r, "text", "") or ""
        cands = idx.candidates_for(doc_id, chunk_id)
        needle = normalize(text)
        hit: Optional[Chunk] = None
        if needle:
            hit = next((c for c in cands if needle in normalize(c.text)), None)
        out.append(RationaleMatch(
            index=i,
            doc_id=normalize_doc_id(doc_id),
            text=text,
            matched=hit is not None,
            chunk=hit or (cands[0] if cands else None),
            candidates=len(cands),
        ))
    return out


def grounded_fraction(matches: Sequence[RationaleMatch]) -> float:
    """逐語一致した引用の割合。引用が無ければ 0.0。"""
    if not matches:
        return 0.0
    return sum(1 for m in matches if m.matched) / len(matches)


def matched_chunks(matches: Sequence[RationaleMatch]) -> list[Chunk]:
    """一致したチャンクを重複なく順序を保って返す。"""
    out: list[Chunk] = []
    seen: set[str] = set()
    for m in matches:
        if m.chunk is not None and m.chunk.chunk_id not in seen:
            out.append(m.chunk)
            seen.add(m.chunk.chunk_id)
    return out


# ---------------- ハイライト範囲 ----------------

def _stripped_to_orig_map(text: str) -> list[int]:
    """空白除去後の位置 k → 元テキストでの位置、の対応表。"""
    return [i for i, ch in enumerate(text) if not ch.isspace()]


def find_spans(chunk_text: str, texts: Sequence[str], min_len: int = 4) -> list[list[int]]:
    """引用文が chunk_text 中に現れる範囲 [start, end) を列挙する (重なりはマージ)。

    まず逐語、無ければ空白無視で探す。**このチャンクに帰属する引用だけ**を
    渡すこと (他文書の引用を混ぜると誤ったハイライトになる)。
    """
    spans: list[list[int]] = []
    stripped = normalize(chunk_text)
    mapping = _stripped_to_orig_map(chunk_text)
    for raw in texts:
        if not raw or len(raw) < min_len:
            continue
        idx = chunk_text.find(raw)
        if idx >= 0:
            spans.append([idx, idx + len(raw)])
            continue
        needle = normalize(raw)
        if not needle:
            continue
        sidx = stripped.find(needle)
        if sidx < 0:
            continue
        start = mapping[sidx]
        end_stripped = sidx + len(needle) - 1
        end = mapping[end_stripped] + 1 if end_stripped < len(mapping) else len(chunk_text)
        spans.append([start, end])
    spans.sort()
    merged: list[list[int]] = []
    for s in spans:
        if merged and s[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s[1])
        else:
            merged.append(list(s))
    return merged
