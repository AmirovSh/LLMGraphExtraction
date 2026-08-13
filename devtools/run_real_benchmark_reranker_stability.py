"""Measure five-run ranking stability on difficult frozen real queries."""
from __future__ import annotations
import json,statistics
from pathlib import Path
from config.settings import environment_value,load_project_settings
from devtools.entity_resolution_reranker import DirectionalPair,RerankerClient,RerankerConfig
from devtools.run_entity_resolution_reranker_experiment import load_local_env

def main():
 root=Path(__file__).resolve().parents[1];bench=root/'release_audit/entity_resolution_benchmark';out=root/'release_audit/reranker_real_artifact_shortlisting_v2'
 rows=json.loads((out/'candidate_generation.json').read_text());clusters=json.loads((bench/'identity_clusters.json').read_text());members={m['entity_id']:m for c in clusters for m in c['members']}
 # Largest pools are the difficult, multiple-distractor subset; membership is frozen before repeats.
 sizes={q:sum(r['query_id']==q for r in rows) for q in {r['query_id'] for r in rows}};selected=set(q for q,_ in sorted(sizes.items(),key=lambda x:(-x[1],x[0]))[:30]);pairs=[]
 for r in rows:
  if r['query_id'] not in selected or not r['candidate_sources']:continue
  a,b=members[r['query_entity_id']],members[r['candidate_id']];fmt=lambda x:f"canonical_name: {x['canonical_name']}\ntype: {x['entity_type']}\naliases: none"
  pairs.append(DirectionalPair(f"{r['query_id']}|{r['candidate_id']}",fmt(a),fmt(b),'ab'))
 load_local_env(root/'.env.runtime');s=load_project_settings(root/'config');remote=s.runtime.remote;client=RerankerClient(RerankerConfig(environment_value(remote.embedding_base_url_env),environment_value(remote.embedding_api_key_env),timeout_seconds=remote.embedding_timeout_seconds,trust_env=remote.trust_env));runs=[]
 for _ in range(5):runs.append({x.pair_id:x.score for x in client.score_many(pairs,batch_size=32)})
 rankings=[]
 for run in runs:
  rankings.append({q:[p.split('|',1)[1] for p in sorted((p for p in run if p.startswith(q+'|')),key=lambda p:(-run[p],p))] for q in selected})
 top1=sum(len({r[q][0] for r in rankings})==1 for q in selected)/len(selected);top3=sum(len({tuple(r[q][:3]) for r in rankings})==1 for q in selected)/len(selected);top5=sum(len({tuple(r[q][:5]) for r in rankings})==1 for q in selected)/len(selected)
 true_drift=[]
 relevance={(r['query_id'],r['candidate_id']):r['relevance'] for r in rows}
 for q in selected:
  for cid in {c for run in rankings for c in run[q] if relevance[(q,c)]}:
   ranks=[run[q].index(cid)+1 for run in rankings];true_drift.append(max(ranks)-min(ranks))
 result={'status':'PASS' if top3==1 and max(true_drift,default=0)==0 else 'FAIL','queries':len(selected),'runs':5,'top1_stability':top1,'top3_ordered_stability':top3,'top5_ordered_stability':top5,'max_true_rank_drift':max(true_drift,default=0),'mean_true_rank_drift':statistics.mean(true_drift) if true_drift else 0}
 (out/'stability.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
