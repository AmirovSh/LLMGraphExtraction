"""Finalize candidate-ranking artifacts from immutable live scores without model calls."""
from __future__ import annotations
import json,statistics
from pathlib import Path
from devtools.entity_candidate_ranking import acronym_match,rank_metrics
from devtools.run_reranker_candidate_ranking import benchmark,neutral_key

P=Path("release_audit/reranker_candidate_ranking")
rows=json.loads((P/"candidate_generator_results.json").read_text())
queries=benchmark(); match=[q for q in queries if not q.no_match]
lookup={(q.query_id,c.candidate_id):(q.query_entity,c.entity) for q in queries for c in q.candidates}
for row in rows:
 q,c=lookup[(row["query_id"],row["candidate_id"])]; row["acronym_match"]=acronym_match(q.canonical_name,c.canonical_name)
 if row["acronym_match"] and "ACRONYM" not in row["candidate_sources"]:row["candidate_sources"].append("ACRONYM")
def ranking(key,eligible=lambda r:True):
 return [[(r["candidate_id"],r["relevance"]) for r in sorted([x for x in rows if x["query_id"]==q.query_id and eligible(x)],key=lambda x:(-x[key],neutral_key(q.query_id,x["candidate_id"])))] for q in match]
methods={"BGE only":ranking("bge_score"),"Lexical only":ranking("lexical_score"),"Acronym only":ranking("acronym_match",lambda x:x["acronym_match"]),"Candidate union":ranking("bge_score",lambda x:bool(x["candidate_sources"])),"BGE to reranker":ranking("ab",lambda x:"BGE" in x["candidate_sources"]),"Union to reranker":ranking("ab",lambda x:bool(x["candidate_sources"]))}
metrics={k:rank_metrics(v) for k,v in methods.items()}
def subset(category):
 ids={r["query_id"] for r in rows if r["relevance"] and r["category"]==category}; return {k:rank_metrics([v[i] for i,q in enumerate(match) if q.query_id in ids]) for k,v in methods.items()}
cats={c:subset(c) for c in sorted({r["category"] for r in rows if r["relevance"]})}
union,rera=methods["Candidate union"],methods["Union to reranker"]; gains=[]; dem=[]
for q,u,rr in zip(match,union,rera,strict=True):
 ur={cid:i for i,(cid,_) in enumerate(u,1)}; ar={cid:i for i,(cid,_) in enumerate(rr,1)}
 for c in q.candidates:
  if c.relevance and c.candidate_id in ur and c.candidate_id in ar:
   gains.append(ur[c.candidate_id]-ar[c.candidate_id]);
   if ur[c.candidate_id]<=5<ar[c.candidate_id]:dem.append(q.query_id)
source={s:sum(r["relevance"] and s in r["candidate_sources"] for r in rows) for s in ("BGE","LEXICAL","ACRONYM")}
art={"ranking_metrics.json":metrics,"category_metrics.json":cats,"abbreviation_metrics.json":cats.get("abbreviation",{}),"cross_document_metrics.json":cats.get("cross_document_alias",{}),"candidate_source_contribution.json":source,"rank_gain.json":{"mean":statistics.mean(gains),"median":statistics.median(gains),"improved":sum(x>0 for x in gains),"unchanged":sum(x==0 for x in gains),"worsened":sum(x<0 for x in gains)},"catastrophic_demotions.json":dem,"candidate_union.json":union,"bge_ranking.json":methods["BGE only"],"reranker_ranking.json":rera}
for n,v in art.items():(P/n).write_text(json.dumps(v,indent=2)+"\n")
print(json.dumps({"metrics":metrics,"categories":cats,"rank_gain":art["rank_gain.json"],"demotions":len(dem),"sources":source},indent=2))
