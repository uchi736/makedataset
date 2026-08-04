import type { ReactNode } from "react";
import type { Labels, QAItem } from "../types";
import { isKg, pct } from "../util";
import CoverageTab from "./CoverageTab";
import KgTab from "./KgTab";
import OverviewTab from "./OverviewTab";
import QualityTab from "./QualityTab";
import RagVerifyTab from "./RagVerifyTab";

/** 折りたたみ節。見出しに要点の数字を出し、開かなくても現在地が分かるようにする。 */
function Section(props: {
  title: string;
  headline: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  return (
    <details className="section" open={props.defaultOpen}>
      <summary>
        <span className="sec-title">{props.title}</span>
        <span className="sec-headline">{props.headline}</span>
        <span className="sec-arrow" aria-hidden>▸</span>
      </summary>
      <div className="sec-body">{props.children}</div>
    </details>
  );
}

export default function AnalysisTab(props: {
  items: QAItem[];
  labels: Labels;
  onGoReview: (qaId: string) => void;
}) {
  const { items, labels, onGoReview } = props;
  const t = labels.targets;
  const generalItems = items.filter((it) => !isKg(it));
  const kgItems = items.filter(isKg);
  const ragVerified = items.filter((it) => it.rag_verification != null);
  const scored = items.filter((it) => it.filter_scores.answerability != null);

  // 見出し用の要点
  const gaps = generalItems.length
    ? labels.category_order.reduce((acc, c) => {
        for (const d of labels.diff_order) {
          if (!generalItems.some((it) => (it.category[0] ?? "") === c && it.retrieval_level === d)) acc += 1;
        }
        return acc;
      }, 0)
    : 0;
  const aspectCounts = new Map<string, number>();
  for (const it of generalItems) for (const a of it.aspect) aspectCounts.set(a, (aspectCounts.get(a) ?? 0) + 1);
  const aspectsOk = labels.all_aspects.filter((a) => (aspectCounts.get(a) ?? 0) >= t.aspect_min).length;

  const passed = scored.filter(
    (it) => (it.filter_scores.answerability ?? 0) >= t.answerability_pass && it.filter_scores.leakage !== "fail",
  ).length;
  const qualityHeadline = scored.length
    ? `ゲート通過 ${passed}/${scored.length} 件 (${pct(passed / scored.length)})`
    : "判定LLM 未実行";

  const ragNoMatch = ragVerified.filter((it) => it.rag_verification?.answer_match === "no_match").length;
  const kgUnknown = kgItems.filter((it) => it.llm_knowledge === "unknown").length;

  return (
    <div>
      <OverviewTab items={items} labels={labels} />

      {generalItems.length > 0 && (
        <Section
          title="カバレッジ (25観点)"
          headline={`目標達成 ${aspectsOk}/25 観点 ・ 未カバーセル ${gaps} 個`}
        >
          <CoverageTab items={items} labels={labels} />
        </Section>
      )}

      <Section title="品質スコア (判定LLM)" headline={qualityHeadline}>
        <QualityTab items={items} labels={labels} />
      </Section>

      {ragVerified.length > 0 && (
        <Section
          title="RAG検証 (実走)"
          headline={`検証済 ${ragVerified.length} 件 ・ 不一致 (KG候補) ${ragNoMatch} 件`}
        >
          <RagVerifyTab items={items} labels={labels} onGoReview={onGoReview} />
        </Section>
      )}

      {kgItems.length > 0 && (
        <Section
          title="KG-PoC (3軸)"
          headline={`${kgItems.length} 件 ・ LLM未知 ${kgUnknown} 件`}
        >
          <KgTab items={items} labels={labels} />
        </Section>
      )}
    </div>
  );
}
