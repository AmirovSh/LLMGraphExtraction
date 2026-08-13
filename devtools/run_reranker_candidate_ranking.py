"""Evaluate bge-reranker-v2-m3 strictly as a non-authoritative candidate ranker."""
from __future__ import annotations
import hashlib,json,math,statistics,time
from pathlib import Path
from config.settings import environment_value,load_project_settings
from devtools.contextual_entity_profiles import LocalEntityContext,render_profile
from devtools.entity_candidate_ranking import *
from devtools.entity_resolution_reranker import DirectionalPair,RerankerClient,RerankerConfig,join_directional_results
from devtools.run_entity_resolution_reranker_experiment import load_local_env,write_json
from runtime.observable_entity_resolution import cosine,request_embeddings

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"release_audit/reranker_candidate_ranking"
STEMS=["Alder","Boreal","Cedar","Drift","Ember","Fjord","Grove","Harbor","Ion","Juniper","Kepler","Lumen"]
FORMS=[("International Records Management Authority","IRMA","abbreviation"),("Identity Verification Service","Identity Service","short_full_name"),("Secure-Document Repository","Secure Document Repository","punctuation_case"),("Regulatory Submission Gateway","Submission Gateway","cross_document_alias")]
DIST=["Record","Adapter","Result","Component","Account","Version 2","Archive","Controller","Manifest","Process","Ledger","Location","Service","Policy","Artifact"]

def benchmark()->list[RankingQuery]:
 q=[]
 for i in range(75):
  stem=STEMS[i%len(STEMS)]; full,alias,cat=FORMS[i%len(FORMS)]; group=f"group-{i//5}"
  qe=RankingEntity(entity_id=f"Q{i}",canonical_name=f"{stem} {full}",entity_type="ORGANIZATION" if "Authority" in full else "SYSTEM",aliases=[],graph_group=group)
  names=[f"{stem} {alias}",f"{stem} {full} Record"]+[f"{stem} {DIST[(i+j)%len(DIST)]}" for j in range(13)]
  cs=[]
  for j,name in enumerate(names):
   rel=1 if j==0 else 0; cs.append(RankingCandidate(candidate_id=f"Q{i}C{j:02d}",entity=RankingEntity(entity_id=f"E{i}-{j}",canonical_name=name,entity_type=qe.entity_type if j<2 else "CONCEPT",aliases=[qe.canonical_name] if rel else [],graph_group=group),relevance=rel,category=cat if rel else "related_but_distinct",cross_document=cat=="cross_document_alias" and rel))
  q.append(RankingQuery(query_id=f"Q{i:03d}",query_entity=qe,candidates=cs,source_kind="CONTROLLED_SYNTHETIC"))
 for i in range(10):
  group=f"nomatch-{i}"; qe=RankingEntity(entity_id=f"NQ{i}",canonical_name=f"{STEMS[i]} Review Board",entity_type="ORGANIZATION",graph_group=group)
  cs=[RankingCandidate(candidate_id=f"N{i}C{j}",entity=RankingEntity(entity_id=f"NE{i}-{j}",canonical_name=f"{STEMS[i]} {DIST[j]}",entity_type="CONCEPT",graph_group=group),relevance=0,category="no_match") for j in range(10)]
  q.append(RankingQuery(query_id=f"NQ{i:02d}",query_entity=qe,candidates=cs,source_kind="CONTROLLED_SYNTHETIC",no_match=True))
 return q

def r2(e:RankingEntity)->str:
 return render_profile(LocalEntityContext(local_id=e.entity_id,canonical_name=e.canonical_name,entity_type=e.entity_type,aliases=e.aliases,source_kind="CONTROLLED_SYNTHETIC",identity_cluster_id=e.entity_id),"R2")["text"]

def neutral_key(query_id:str,candidate_id:str)->str:
 return hashlib.sha256(f"{query_id}|{candidate_id}".encode()).hexdigest()

def main()->int:
 OUT.mkdir(parents=True,exist_ok=True); queries=benchmark(); load_local_env(ROOT/".env.runtime"); s=load_project_settings(ROOT/"config"); r=s.runtime.remote; endpoint=environment_value(r.embedding_base_url_env); key=environment_value(r.embedding_api_key_env)
 entities={e.entity_id:e for q in queries for e in [q.query_entity,*[c.entity for c in q.candidates]]}; ids=list(entities); start=time.perf_counter(); vecs,_,_=request_embeddings(endpoint=endpoint,api_key=key,model=s.entity_resolution.embedding.model,inputs=[{"local_entity_id":i,"embedding_text":entities[i].canonical_name} for i in ids],timeout_seconds=r.embedding_timeout_seconds,trust_env=r.trust_env,write_artifact=lambda *_:None); bge_seconds=time.perf_counter()-start; by=dict(zip(ids,vecs,strict=True))
 rows=[]; pairs=[]
 for q in queries:
  bge=sorted(q.candidates,key=lambda c:(-cosine(by[q.query_entity.entity_id],by[c.entity.entity_id]),c.candidate_id)); ranks={c.candidate_id:i for i,c in enumerate(bge,1)}
  for c in q.candidates:
   sources=candidate_sources(q.query_entity,c.entity,ranks[c.candidate_id]); rows.append({"query_id":q.query_id,"candidate_id":c.candidate_id,"relevance":c.relevance,"category":c.category,"cross_document":c.cross_document,"bge_score":cosine(by[q.query_entity.entity_id],by[c.entity.entity_id]),"bge_rank":ranks[c.candidate_id],"lexical_score":lexical_score(q.query_entity.canonical_name,c.entity.canonical_name),"acronym_match":acronym_match(q.query_entity.canonical_name,c.entity.canonical_name),"candidate_sources":sources,"no_match":q.no_match})
   if sources: pairs.append(DirectionalPair(f"{q.query_id}|{c.candidate_id}",r2(q.query_entity),r2(c.entity),"ab"))
 client=RerankerClient(RerankerConfig(endpoint,key,timeout_seconds=r.embedding_timeout_seconds,trust_env=r.trust_env)); start=time.perf_counter(); raw=client.score_many(pairs,batch_size=32); rerank_seconds=time.perf_counter()-start; scores={item.pair_id:{"ab":item.score} for item in raw}
 for row in rows:
  keyid=f'{row["query_id"]}|{row["candidate_id"]}'; row.update(scores.get(keyid,{"ab":-1.}))
 def ranking(key,eligible=lambda r:True):
  return [[(r["candidate_id"],r["relevance"]) for r in sorted([x for x in rows if x["query_id"]==q.query_id and eligible(x)],key=lambda x:(-x[key],neutral_key(q.query_id,x["candidate_id"]))) ] for q in queries if not q.no_match]
 methods={"BGE only":ranking("bge_score"),"Lexical only":ranking("lexical_score"),"Acronym only":ranking("acronym_match",lambda x:x["acronym_match"]),"Candidate union":ranking("bge_score",lambda x:bool(x["candidate_sources"])),"BGE to reranker":ranking("ab",lambda x:"BGE" in x["candidate_sources"]),"Union to reranker":ranking("ab",lambda x:bool(x["candidate_sources"]))}
 metrics={k:rank_metrics(v) for k,v in methods.items()}; union=methods["Candidate union"]; reranked=methods["Union to reranker"]
 gains=[]; demotions=[]
 for q,u,rr in zip([q for q in queries if not q.no_match],union,reranked,strict=True):
  ur={cid:i for i,(cid,_) in enumerate(u,1)}; rrank={cid:i for i,(cid,_) in enumerate(rr,1)}
  for c in q.candidates:
   if c.relevance and c.candidate_id in ur and c.candidate_id in rrank:gains.append(ur[c.candidate_id]-rrank[c.candidate_id]); demotions += [q.query_id] if ur[c.candidate_id]<=5<rrank[c.candidate_id] else []
 artifacts={"benchmark_queries.json":[q.model_dump(mode="json") for q in queries],"candidate_generator_results.json":rows,"candidate_union.json":union,"bge_ranking.json":methods["BGE only"],"reranker_ranking.json":reranked,"ranking_metrics.json":metrics,"category_metrics.json":{},"abbreviation_metrics.json":{},"cross_document_metrics.json":{},"candidate_source_contribution.json":{},"rank_gain.json":{"mean":statistics.mean(gains),"median":statistics.median(gains),"improved":sum(x>0 for x in gains),"worsened":sum(x<0 for x in gains)},"catastrophic_demotions.json":demotions,"performance.json":{"bge_seconds":bge_seconds,"reranker_seconds":rerank_seconds,"reranker_pairs":len(pairs),"pairs_per_second":len(pairs)/rerank_seconds,"mean_candidates":statistics.mean(len(q.candidates) for q in queries)},"graph_replay.json":{"status":"INSUFFICIENT_DATA"},"evidence_ranking.json":{"status":"NOT_EXECUTED"}}
 for n,v in artifacts.items():write_json(OUT/n,v)
 print(json.dumps({"queries":len(queries),"metrics":metrics,"rank_gain":artifacts["rank_gain.json"],"demotions":len(demotions),"performance":artifacts["performance.json"]},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
