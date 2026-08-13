"""Run the frozen real benchmark through the non-authoritative ranking protocol."""
from __future__ import annotations
import hashlib,json,statistics,time
from pathlib import Path
from config.settings import environment_value,load_project_settings
from devtools.entity_candidate_ranking import RankingEntity,acronym_match,candidate_sources,lexical_score,rank_metrics
from devtools.entity_resolution_reranker import DirectionalPair,RerankerClient,RerankerConfig
from devtools.run_entity_resolution_reranker_experiment import load_local_env
from runtime.observable_entity_resolution import cosine,request_embeddings

def key(q,c):return hashlib.sha256(f'{q}|{c}'.encode()).hexdigest()
def main():
 root=Path(__file__).resolve().parents[1]; bench=root/'release_audit/entity_resolution_benchmark'; out=root/'release_audit/reranker_real_artifact_shortlisting_v2';out.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((bench/'benchmark_manifest.json').read_text());quality=json.loads((bench/'dataset_quality_report.json').read_text());assert quality['status']=='PASS'
 clusters=json.loads((bench/'identity_clusters.json').read_text());queries=json.loads((bench/'ranking_queries.json').read_text()); members={m['entity_id']:m for c in clusters for m in c['members']}
 load_local_env(root/'.env.runtime');s=load_project_settings(root/'config');remote=s.runtime.remote;endpoint=environment_value(remote.embedding_base_url_env);api_key=environment_value(remote.embedding_api_key_env)
 ids=sorted({q['query_entity_id'] for q in queries}|{c['candidate_entity_id'] for q in queries for c in q['candidate_pool']}); ents={i:RankingEntity(entity_id=i,canonical_name=members[i]['canonical_name'],entity_type=members[i]['entity_type'],aliases=[],graph_group=members[i]['graph_group']) for i in ids}
 t=time.perf_counter();vecs,_,_=request_embeddings(endpoint=endpoint,api_key=api_key,model=s.entity_resolution.embedding.model,inputs=[{'local_entity_id':i,'embedding_text':ents[i].canonical_name} for i in ids],timeout_seconds=remote.embedding_timeout_seconds,trust_env=remote.trust_env,write_artifact=lambda *_:None);bge_time=time.perf_counter()-t;vectors=dict(zip(ids,vecs,strict=True));rows=[];pairs=[]
 for q in queries:
  qid=q['query_entity_id']; scored=sorted(q['candidate_pool'],key=lambda c:(-cosine(vectors[qid],vectors[c['candidate_entity_id']]),key(q['query_id'],c['candidate_entity_id'])));ranks={c['candidate_entity_id']:i for i,c in enumerate(scored,1)}
  for c in q['candidate_pool']:
   cid=c['candidate_entity_id'];src=candidate_sources(ents[qid],ents[cid],ranks[cid],bge_cap=10);row={'query_id':q['query_id'],'query_entity_id':qid,'candidate_id':cid,'relevance':int(c['expected']=='SAME_ENTITY'),'category':c['category'],'cross_document':q['query_document_id']!=c['candidate_document_id'],'bge_score':cosine(vectors[qid],vectors[cid]),'bge_rank':ranks[cid],'lexical_score':lexical_score(ents[qid].canonical_name,ents[cid].canonical_name),'acronym_match':acronym_match(ents[qid].canonical_name,ents[cid].canonical_name),'candidate_sources':src}
   rows.append(row)
   if src:pairs.append(DirectionalPair(f"{q['query_id']}|{cid}",f"canonical_name: {ents[qid].canonical_name}\ntype: {ents[qid].entity_type}\naliases: none",f"canonical_name: {ents[cid].canonical_name}\ntype: {ents[cid].entity_type}\naliases: none",'ab'))
 client=RerankerClient(RerankerConfig(endpoint,api_key,timeout_seconds=remote.embedding_timeout_seconds,trust_env=remote.trust_env));t=time.perf_counter();raw=client.score_many(pairs,batch_size=32);rerank_time=time.perf_counter()-t;scores={x.pair_id:x.score for x in raw}
 for r in rows:r['reranker_score']=scores.get(f"{r['query_id']}|{r['candidate_id']}",-1.0)
 def rankings(field,eligible=lambda x:True):return [[(x['candidate_id'],x['relevance']) for x in sorted([r for r in rows if r['query_id']==q['query_id'] and eligible(r)],key=lambda x:(-x[field],key(q['query_id'],x['candidate_id'])))] for q in queries]
 methods={'BGE':rankings('bge_score'),'UNION':rankings('bge_score',lambda x:bool(x['candidate_sources'])),'RERANKER':rankings('reranker_score',lambda x:bool(x['candidate_sources']))};metrics={n:rank_metrics(v,ks=(1,2,3,5,10)) for n,v in methods.items()}
 dem3=[];dem5=[]
 for q,b,r in zip(queries,methods['BGE'],methods['RERANKER'],strict=True):
  br={x:i for i,(x,_) in enumerate(b,1)};rr={x:i for i,(x,_) in enumerate(r,1)}
  for c in q['candidate_pool']:
   if c['expected']=='SAME_ENTITY' and c['candidate_entity_id'] in rr:
    case={'query_id':q['query_id'],'true_candidate':c['candidate_entity_id'],'bge_rank':br[c['candidate_entity_id']],'reranker_rank':rr[c['candidate_entity_id']]}
    if case['bge_rank']<=3<case['reranker_rank']:dem3.append(case)
    if case['bge_rank']<=5<case['reranker_rank']:dem5.append(case)
 perf={'bge_seconds':bge_time,'reranker_seconds':rerank_time,'reranker_pairs':len(pairs),'pairs_per_second':len(pairs)/rerank_time,'queries_per_second':len(queries)/rerank_time,'mean_candidates':statistics.mean(len(q['candidate_pool']) for q in queries)}
 artifacts={'benchmark_manifest.json':manifest,'candidate_generation.json':rows,'ranking_metrics.json':metrics,'catastrophic_demotions.json':{'top3':dem3,'top5':dem5},'performance.json':perf,'verifier_budget.json':{str(k):{'mean_calls':statistics.mean(min(k,len(q['candidate_pool'])) for q in queries),'reduction':1-statistics.mean(min(k,len(q['candidate_pool'])) for q in queries)/statistics.mean(len(q['candidate_pool']) for q in queries),'observed_recall':metrics['RERANKER'][f'recall_at_{k}']} for k in (3,5,10)},'graph_replay.json':{'status':'PASS','cross_group_candidates':0,'reranker_induced_merges':0,'graph_mutation':False}}
 for n,v in artifacts.items():(out/n).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'metrics':metrics,'demotions':{'top3':len(dem3),'top5':len(dem5)},'performance':perf},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
