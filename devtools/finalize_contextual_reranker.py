"""Materialize Phase-3R analysis from immutable score artifacts."""
import json, statistics
from pathlib import Path

P=Path("release_audit/context_reranker")
def read(n): return json.loads((P/n).read_text(encoding="utf-8"))
def write(n,v): (P/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

rows=read("reranker_directional_scores.json"); comp=read("representation_comparison.json")
write("score_distributions.json",{rep:{label:{"mean":statistics.mean(v),"min":min(v),"max":max(v)} for label in ("MERGE","SPLIT")
    if (v:=[r["mean"] for r in rows if r["representation"]==rep and r["expected"]==label])} for rep in ("R1","R2","R3","R4")})
write("threshold_selection.json",{rep:{k:v["selection"] for k,v in comp[rep].items()} for rep in comp})
write("held_out_results.json",{rep:{k:v["held_out"] for k,v in comp[rep].items()} for rep in comp})
write("category_metrics.json",{"note":"120 controlled-synthetic pairs; categories are true_alias and RELATED_BUT_DISTINCT"})
write("related_but_distinct_analysis.json",{rep:{"held_out_false_merges":comp[rep]["mean"]["held_out"]["false_merges"]} for rep in comp})
write("profile_size_ablation.json",{"status":"NOT_EXECUTED","reason":"Primary R1-R4 result is not eligible for a generalization PASS; avoid additional endpoint use before fresh real-artifact coverage."})
write("context_ablation.json",{"status":"NOT_EXECUTED","reason":"R4 synthetic improvement is confounded by trivial exact-name positives and lacks real-artifact held-out coverage."})
write("context_swap_diagnostic.json",{"status":"NOT_EXECUTED","reason":"Reserved until a non-trivial fresh benchmark is available."})
write("threshold_robustness.json",{"status":"FAIL","reason":"R1/R2 scores remain saturated; R4 perfect synthetic held-out does not establish robust unseen separation."})
asym={}
for rep in comp:
    d=sorted(abs(r["ab"]-r["ba"]) for r in rows if r["representation"]==rep)
    asym[rep]={"mean":statistics.mean(d),"median":statistics.median(d),"p95":d[int(.95*len(d))-1],"max":max(d)}
write("stability.json",{"status":"NOT_EXECUTED","directional_asymmetry":asym,"reason":"Five-repeat gate deferred because fresh real-artifact coverage is insufficient."})
write("graph_replay.json",{"status":"NOT_AVAILABLE","large_graph_replay":"NOT_AVAILABLE","reason":"Saved A/B/C artifacts are historical development data and no fresh >=50-entity pre-resolution artifact is available for this hypothesis."})

report='''# Contextual Reranker Re-evaluation Report

## Executive verdict

**CONTEXTUAL RERANKER PROMISING BUT INCONCLUSIVE.** R4 recovered perfect classification on the 24-pair controlled-synthetic held-out partition with mean bidirectional scoring, while R3 collapsed to zero recall. This cannot establish generalization: all 120 fresh pairs are controlled synthetic, R1 exact-name positives make the benchmark too easy, no fresh real-artifact held-out set exists, and robustness/stability/graph-replay gates are incomplete.

The prior conclusion is therefore scoped as **R2 RERANKER GENERALIZATION FAILED**. Rich-context reranking had not previously been tested.

## Historical rich BGE representation

Git objects recover the historical report and aggregate. The original function itself is no longer reachable, so Phase 3R records a reconstructed-compatible contract, not byte-identical recovery: name/type/aliases, up to five source-order evidence snippets truncated to 160 characters, up to eight source-order one-hop incidents, total 800 characters. Serializer version/hash is in `profile_contract.json`.

## Matched-input methodology and provenance

For every R1-R4 profile, exactly the same serialized string was sent independently to BGE-M3 and to both reranker directions. Candidate retrieval was not changed. Profiles are pair-independent and contain only local pre-resolution data. The benchmark has 120 eligible pairs: 40 MERGE and 80 SPLIT, all `CONTROLLED_SYNTHETIC`; split is 72/24/24 by identity cluster. Real-artifact benchmark coverage is zero and is the principal limitation.

## Central held-out result

| Representation | BGE P/R/F1 | BGE false/missed | Reranker mean P/R/F1 | Reranker false/missed | Reranker ROC-AUC |
| --- | --- | --- | --- | --- | ---: |
| R1 name | 1.00/1.00/1.00 | 0/0 | 1.00/1.00/1.00 | 0/0 | 1.000 |
| R2 name/type/aliases | 1.00/1.00/1.00 | 0/0 | 1.00/.75/.857 | 0/2 | 1.000 |
| R3 + evidence | 1.00/1.00/1.00 | 0/0 | 0/.00/.00 | 0/8 | .916 |
| R4 + local relations | 1.00/1.00/1.00 | 0/0 | 1.00/1.00/1.00 | 0/0 | .953 |

PR-AUC was not independently materialized; ROC-AUC is secondary. Recall at zero observed false merges equals held-out recall above. The synthetic construction makes R1/BGE perfect, so the table is diagnostic rather than a fair proof that context improves either model.

## Directional asymmetry

Mean absolute difference fell from R2 `.03668` to R3 `.00866`, then rose to R4 `.01782`; R4 p95 `.05517`, max `.09718`. Context reduced the extreme Phase-3 asymmetry but did not remove direction sensitivity.

## Related-but-distinct and same-context negatives

R4 mean had zero observed held-out false merges on 16 synthetic RELATED_BUT_DISTINCT cases. R3 avoided false merges only by rejecting every positive. Because these negatives share templated context and all positives use identical names, this does not meet the requested semantic-difficulty standard.

## Robustness, ablations, stability and replay

Threshold robustness: **FAIL/incomplete**. Profile-size, context removal, context swap, and five-repeat stability were not executed after the benchmark validity limitation became clear. Graph replay and a fresh >=50-local-entity replay are not available. Saved A/B/C graphs and V1/V2/V3.1 remain historical development data and were not misrepresented as unseen validation.

## Production regression boundary

Production remains BGE-M3 `canonical_name_only`, candidate `.76`, automatic merge `.90`, top-K `10`; no production or batch reranker call was added. DeepSeek calls were zero. Prompt, schema, FinalGraph and Neo4j are unchanged.

## Final recommendation

Do not integrate. Build a genuinely difficult, fresh benchmark with at least 100 R3/R4-eligible real-artifact pairs, non-exact alias positives, same-span negatives and cluster-isolated held-out data. Then repeat the frozen R1-R4 matched-input protocol and only proceed to expanded retrieval if R3/R4 wins robustly.
'''
(P.parent/"CONTEXTUAL_RERANKER_REEVALUATION_REPORT.md").write_text(report,encoding="utf-8")
