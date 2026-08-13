"""Create the Stage-B report from immutable real-benchmark ranking artifacts."""
from __future__ import annotations
import json
from pathlib import Path

def main():
 root=Path(__file__).resolve().parents[1];p=root/'release_audit/reranker_real_artifact_shortlisting_v2';m=json.loads((p/'ranking_metrics.json').read_text());d=json.loads((p/'catastrophic_demotions.json').read_text());perf=json.loads((p/'performance.json').read_text());st=json.loads((p/'stability.json').read_text());budget=json.loads((p/'verifier_budget.json').read_text())
 b,u,r=m['BGE'],m['UNION'],m['RERANKER'];report=root/'release_audit/RERANKER_REAL_ARTIFACT_SHORTLISTING_V2_REPORT.md'
 report.write_text(f"""# Reranker Real-Artifact Shortlisting V2 Report

## Verdict

**RERANKER REAL-ARTIFACT SHORTLISTING FAILED**

On 51 frozen real positive queries, BGE R@3/R@5 was {b['recall_at_3']:.3f}/{b['recall_at_5']:.3f}; candidate union was {u['recall_at_3']:.3f}/{u['recall_at_5']:.3f}; forward R2 `bge-reranker-v2-m3` fell to {r['recall_at_3']:.3f}/{r['recall_at_5']:.3f}. Reranking caused {len(d['top3'])} top-3 and {len(d['top5'])} top-5 catastrophic demotions.

Reranker metrics: R@1 {r['recall_at_1']:.3f}, R@2 {r['recall_at_2']:.3f}, R@3 {r['recall_at_3']:.3f}, R@5 {r['recall_at_5']:.3f}, R@10 {r['recall_at_10']:.3f}, MRR {r['mrr']:.3f}, NDCG@10 {r['ndcg_at_10']:.3f}. The pools average {perf['mean_candidates']:.2f} candidates and contain reviewed multiple distractors.

Five-run stability on 30 difficult queries: **{st['status']}**; top-3 ordered stability {st['top3_ordered_stability']:.3f}, max true-rank drift {st['max_true_rank_drift']}.

Operationally, {perf['reranker_pairs']} pairs took {perf['reranker_seconds']:.2f}s ({perf['pairs_per_second']:.2f} pairs/s). Top-3 reduction was {budget['3']['reduction']:.1%} at observed recall {budget['3']['observed_recall']:.3f}; top-5 reduction was {budget['5']['reduction']:.1%} at recall {budget['5']['observed_recall']:.3f}. Neither is safe.

Graph replay boundary passed: zero cross-group candidates, zero reranker-induced merges, and no graph mutation. Production was not modified.

Recommended top-K: **NONE**. Next research step: high-recall candidate generation -> identity-specific verifier, bypassing this reranker.
""",encoding='utf-8');return 0
if __name__=='__main__':raise SystemExit(main())
