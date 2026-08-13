"""Materialize Phase-3R analysis from immutable score artifacts."""
import json, statistics
from pathlib import Path

P=Path("release_audit/context_reranker")
def read(n): return json.loads((P/n).read_text(encoding="utf-8"))
def write(n,v): (P/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

rows=read("reranker_directional_scores.json"); comp=read("representation_comparison.json")
def pr_auc(items,key):
    ranked=sorted(items,key=lambda r:r[key],reverse=True); total=sum(r["expected"]=="MERGE" for r in ranked); tp=fp=0; area=last=0.0
    for r in ranked:
        tp += r["expected"]=="MERGE"; fp += r["expected"]=="SPLIT"; recall=tp/total; precision=tp/(tp+fp)
        area += (recall-last)*precision; last=recall
    return area
write("score_distributions.json",{rep:{label:{"mean":statistics.mean(v),"min":min(v),"max":max(v)} for label in ("MERGE","SPLIT")
    if (v:=[r["mean"] for r in rows if r["representation"]==rep and r["expected"]==label])} for rep in ("R1","R2","R3","R4")})
write("threshold_selection.json",{rep:{k:v["selection"] for k,v in comp[rep].items()} for rep in comp})
write("held_out_results.json",{rep:{k:v["held_out"] for k,v in comp[rep].items()} for rep in comp})
write("category_metrics.json",{rep:{cat:{"count":len(v),"mean_score":statistics.mean(x["mean"] for x in v)} for cat in sorted({x["category"] for x in rows}) if (v:=[x for x in rows if x["representation"]==rep and x["category"]==cat])} for rep in comp})
write("related_but_distinct_analysis.json",{rep:{"held_out_false_merges":comp[rep]["mean"]["held_out"]["false_merges"],"mean_score":statistics.mean(r["mean"] for r in rows if r["representation"]==rep and r["expected"]=="SPLIT")} for rep in comp})
write("profile_size_ablation.json",{"status":"NOT_EXECUTED","reason":"Primary R3/R4 generalization failed; additional size tuning would be post-hoc."})
for rep in comp:
    for key in ("bge","mean","min","max"): comp[rep][key]["pr_auc_all"]=pr_auc([r for r in rows if r["representation"]==rep],key)
write("representation_comparison.json",comp)
robust={}
for rep in comp:
    threshold=comp[rep]["mean"]["selection"]["threshold"]; held=[r for r in rows if r["representation"]==rep and r["partition"]=="HELD_OUT"]
    robust[rep]={"threshold":threshold,"changes":{str(d):sum((r["mean"]>=threshold)!=(r["mean"]>=threshold+d) for r in held) for d in (-.01,-.001,-.0001,-.00001,.00001,.0001,.001,.01)},
        "positive_margins":sorted(abs(r["mean"]-threshold) for r in held if r["expected"]=="MERGE"),"negative_margins":sorted(abs(r["mean"]-threshold) for r in held if r["expected"]=="SPLIT")}
write("threshold_robustness.json",robust)
asym={}
for rep in comp:
    d=sorted(abs(r["ab"]-r["ba"]) for r in rows if r["representation"]==rep)
    asym[rep]={"mean":statistics.mean(d),"median":statistics.median(d),"p95":d[int(.95*len(d))-1],"max":max(d)}
stability=read("stability.json") if (P/"stability.json").exists() else {}
stability["directional_asymmetry"]=asym
write("stability.json",stability)
write("graph_replay.json",{"status":"NOT_AVAILABLE","large_graph_replay":"NOT_AVAILABLE","reason":"Saved A/B/C artifacts are historical development data and no fresh >=50-entity pre-resolution artifact is available for this hypothesis."})

report='''# Contextual Reranker Re-evaluation Report

## Executive verdict

**CONTEXTUAL RERANKER FAILED.** On a corrected 24-pair held-out set of non-exact aliases and same-context negatives, R2 mean achieved precision 1.00/recall .875/F1 .933, R3 achieved zero recall, and R4 achieved zero recall with one false merge. Rich entity-local context worsened both BGE and reranker discrimination.

The prior conclusion is therefore scoped as **R2 RERANKER GENERALIZATION FAILED**. Rich-context reranking had not previously been tested.

## Historical rich BGE representation

Git objects recover the historical report and aggregate. The original function itself is no longer reachable, so Phase 3R records a reconstructed-compatible contract, not byte-identical recovery: name/type/aliases, up to five source-order evidence snippets truncated to 160 characters, up to eight source-order one-hop incidents, total 800 characters. Serializer version/hash is in `profile_contract.json`.

## Matched-input methodology and provenance

For every R1-R4 profile, exactly the same serialized string was sent independently to BGE-M3 and to both reranker directions. Candidate retrieval was not changed. Profiles are pair-independent and contain only local pre-resolution data. The benchmark has 120 eligible pairs: 40 MERGE and 80 SPLIT, all `CONTROLLED_SYNTHETIC`; split is 72/24/24 by identity cluster. Real-artifact benchmark coverage is zero and is the principal limitation.

## Central held-out result

| Representation | BGE P/R/F1 | BGE false/missed | Reranker mean P/R/F1 | Reranker false/missed | Reranker ROC-AUC |
| --- | --- | --- | --- | --- | ---: |
| R1 name | 1/.50/.667 | 0/4 | .75/.375/.50 | 1/5 | .933 |
| R2 name/type/aliases | 1/1/1 | 0/0 | 1/.875/.933 | 0/1 | .995 |
| R3 + evidence | 0/0/0 | 0/8 | 0/0/0 | 0/8 | .563 |
| R4 + local relations | 0/0/0 | 0/8 | 0/0/0 | 1/8 | .466 |

PR-AUC is persisted for every representation/model. Recall at zero observed false merges is R1 .25, R2 .875, R3 0, R4 0 for the conservative calibrated mean/min operating points.

## Directional asymmetry

Mean absolute difference fell from R2 `.03668` to R3 `.00866`, then rose to R4 `.01782`; R4 p95 `.05517`, max `.09718`. Context reduced the extreme Phase-3 asymmetry but did not remove direction sensitivity.

## Related-but-distinct and same-context negatives

R4 mean produced one held-out false merge among 16 related-but-distinct cases. R3 avoided false merges only by rejecting every positive. Same-span co-occurrence raised similarity rather than improving identity contrast.

## Robustness, ablations, stability and replay

Threshold robustness: **FAIL** for contextual identity quality. Context removal/swap and five-repeat stability were executed. Swapping evidence/relations changed R4 mean score by .0197 on average, proving context sensitivity, but not useful discrimination. R4 raw scores were stable (one of 24 pairs exceeded 1e-5 range), while R2 was noisier. Graph replay and a fresh >=50-local-entity replay are unavailable.

## Production regression boundary

Production remains BGE-M3 `canonical_name_only`, candidate `.76`, automatic merge `.90`, top-K `10`; no production or batch reranker call was added. DeepSeek calls were zero. Prompt, schema, FinalGraph and Neo4j are unchanged.

## Final recommendation

Do not integrate. The corrected controlled-synthetic experiment supports Outcome B: richer context harms both bi-encoder and cross-encoder identity discrimination. A real-artifact replication could refine external validity, but does not justify production work or expanded retrieval now.
'''
(P.parent/"CONTEXTUAL_RERANKER_REEVALUATION_REPORT.md").write_text(report,encoding="utf-8")
