"""Run the isolated oracle-pair identity verifier experiment (audit artifacts only)."""
from __future__ import annotations

import argparse, json, os, random, statistics, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from config.settings import environment_value,load_project_settings
from devtools.identity_specific_verifier import (EntityProfile,SYSTEM_PROMPT,VerifierDecision,
    assert_pre_resolution,canonical_json_hash,deterministic_baseline,metrics,render_profile,verifier_tool)
from devtools.probe_model_compatibility import _profile_payload
from devtools.run_entity_resolution_reranker_experiment import load_local_env

BENCH=ROOT/'release_audit/entity_resolution_benchmark'; OUT=ROOT/'release_audit/identity_verifier'
EXPECTED_BENCH='725ec53fa5fe43070f6b02f9cfbf3a266c5fa3942f835772ffc900708cf4936e'
EXPECTED_SPLIT='3d92ca1209ddb32aa00e9046a07da7ad49704c85d26769d287e14b578a63212f'
REPS=('V1','V2','V3','V4','V4_LITE')

def read(path:Path): return json.loads(path.read_text(encoding='utf-8'))
def write(path:Path,value:Any): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def load_inputs():
    manifest=read(BENCH/'benchmark_manifest.json')
    if manifest['benchmark_version']!='1.0.0' or manifest['benchmark_hash']!=EXPECTED_BENCH or manifest['split_hash']!=EXPECTED_SPLIT:
        raise RuntimeError('frozen benchmark identity mismatch')
    clusters=read(BENCH/'identity_clusters.json'); pairs=read(BENCH/'identity_pairs.json')
    split=read(BENCH/'split_manifest.json')['by_identity_cluster_id']
    members={m['entity_id']:m for c in clusters for m in c['members']}
    membership={m['entity_id']:c['identity_cluster_id'] for c in clusters for m in c['members']}
    return manifest,pairs,split,members,membership

def source_context(member:dict[str,Any])->tuple[tuple[tuple[str,str],...],tuple[str,...]]:
    evidence=[]; relations=[]; doc=member['document_id']; source=(BENCH/'benchmark_sources'/f'{doc}.txt').read_text(encoding='utf-8')
    units={x['span_id']:x for x in read(BENCH/'benchmark_pipeline_run_v2'/'documents'/doc/'units.json')}
    for ref in member.get('evidence_refs',[]):
        span=ref.split(':')[-1]; u=units.get(span)
        if not u: continue
        evidence.append((ref,source[u['source_start']:u['source_end']]))
        artifact=BENCH/'benchmark_pipeline_run_v2'/'documents'/doc/'units'/f"span_{u['document_local_order']:06d}"/'span_extraction_results.json'
        if not artifact.exists(): continue
        for result in read(artifact):
            entities={x['local_id']:x for x in result.get('entities',[])}
            ids={i for i,x in entities.items() if x.get('name')==member['canonical_name']}
            for rel in result.get('relations',[]):
                s,t=rel['source_local_id'],rel['target_local_id']; desc=rel.get('relation_description') or rel.get('raw_relation') or 'related to'
                if s in ids and t in entities: relations.append(f"{desc} -> {entities[t]['name']} [{ref}]")
                if t in ids and s in entities: relations.append(f"{desc} <- {entities[s]['name']} [{ref}]")
    return tuple(evidence),tuple(dict.fromkeys(relations))

def profiles(members):
    out={}
    for eid,m in members.items():
        evidence,relations=source_context(m)
        out[eid]=EntityProfile(eid,m['canonical_name'],m['entity_type'],(),evidence,relations)
    return out

def dataset(pairs,split,membership):
    rows=[]; excluded=[]
    for i,p in enumerate(pairs):
        a,b=membership[p['left_entity_id']],membership[p['right_entity_id']]; pa,pb=split[a],split[b]
        row={**p,'pair_id':f'P{i+1:04d}'}
        (rows if pa==pb else excluded).append(row|({'partition':pa} if pa==pb else {'left_partition':pa,'right_partition':pb}))
    return rows,excluded

class Client:
    def __init__(self):
        load_local_env(ROOT/'.env.runtime'); s=load_project_settings(ROOT/'config'); remote=s.runtime.remote; self.profile=remote.selected_provider_profile
        self.url=environment_value(remote.llm_base_url_env).rstrip('/')+'/chat/completions'; self.headers={'Authorization':f'Bearer {environment_value(remote.llm_api_key_env)}'}
        self.timeout=remote.capability_probe.timeout_seconds; self.trust_env=remote.trust_env
    def call(self,row,left,right,rep,tag='primary'):
        a,b=render_profile(left,rep),render_profile(right,rep); public={'profile_a':a,'profile_b':b}; assert_pre_resolution(public)
        payload=_profile_payload(self.profile,messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':json.dumps(public,ensure_ascii=False)}],max_tokens=256)
        payload['tools']=[verifier_tool()]; payload['tool_choice']={'type':'function','function':{'name':'verify_entity_identity'}}; payload['parallel_tool_calls']=False
        started=time.perf_counter(); result={'pair_id':row['pair_id'],'partition':row['partition'],'representation':rep,'tag':tag,'attempt_count':1,'label':row['label'],'category':row.get('category'),'left_entity_id':row['left_entity_id'],'right_entity_id':row['right_entity_id'],'profile_a_hash':a['sha256'],'profile_b_hash':b['sha256']}
        try:
            response=httpx.post(self.url,headers=self.headers,json=payload,timeout=self.timeout,trust_env=self.trust_env); result['http_status']=response.status_code; response.raise_for_status(); body=response.json()
            msg=body['choices'][0]['message']; calls=msg.get('tool_calls') or []
            if len(calls)!=1: raise ValueError(f'expected one tool call, got {len(calls)}')
            parsed=VerifierDecision.model_validate_json(calls[0]['function']['arguments']); result.update(parsed.model_dump()); result['status']='SUCCESS'; result['finish_reason']=body['choices'][0].get('finish_reason'); result['usage']=body.get('usage',{})
        except Exception as e: result.update(status='FAILED',decision='UNCERTAIN',reason_codes=['TRANSPORT_OR_CONTRACT_FAILURE'],evidence_refs=[],error=f'{type(e).__name__}: {e}')
        result['latency_ms']=round((time.perf_counter()-started)*1000,2); return result

def run_calls(client,work,cache_path):
    cache={}
    if cache_path.exists(): cache={(x['pair_id'],x['representation'],x.get('tag','primary')):x for x in read(cache_path)}
    todo=[x for x in work if (x[0]['pair_id'],x[3],x[4]) not in cache]
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures=[pool.submit(client.call,*x) for x in todo]
        for n,f in enumerate(as_completed(futures),1):
            x=f.result(); cache[(x['pair_id'],x['representation'],x['tag'])]=x
            if n%25==0: write(cache_path,list(cache.values())); print(f'completed {n}/{len(todo)}',flush=True)
    write(cache_path,list(cache.values())); return list(cache.values())

def choose(summary):
    # precision-first; then false merges, selective accuracy, coverage, compactness
    order={'V1':0,'V2':1,'V3':2,'V4_LITE':3,'V4':4}
    return max(REPS,key=lambda r:(summary[r]['same_precision'],-summary[r]['false_merges'],summary[r]['selective_accuracy'],summary[r]['coverage'],-order[r]))

def bootstrap(rows,n=2000):
    rng=random.Random(20260813); keys=('accuracy','coverage','selective_accuracy','same_precision','same_recall','false_merges','false_splits'); samples={k:[] for k in keys}
    for _ in range(n):
        m=metrics([rng.choice(rows) for _ in rows])
        for k in keys: samples[k].append(m[k])
    return {k:{'low':sorted(v)[int(.025*n)],'high':sorted(v)[int(.975*n)-1]} for k,v in samples.items()}

def diagnostics(client,rows,prof,selected):
    primary=read(OUT/'verifier_call_records.json')
    selected_rows=[x for x in primary if x['representation']==selected and x.get('tag')=='primary']
    difficult=[x for x in selected_rows if x['decision']!=x['label']]
    remaining=[x for x in selected_rows if x['decision']==x['label']]
    sample=(difficult+remaining)[:30]
    by_id={x['pair_id']:x for x in rows}
    stability_work=[]
    for result in sample:
        row=by_id[result['pair_id']]
        for run in range(1,6): stability_work.append((row,prof[row['left_entity_id']],prof[row['right_entity_id']],selected,f'stability_{run}'))
    calls=run_calls(client,stability_work,OUT/'diagnostic_call_records.json')
    stability=[]
    for result in sample:
        decisions=[x['decision'] for x in calls if x['pair_id']==result['pair_id'] and x['tag'].startswith('stability_')]
        stability.append({'pair_id':result['pair_id'],'decisions':decisions,'stable':len(set(decisions))==1})
    write(OUT/'stability_results.json',{'pairs':len(stability),'runs_per_pair':5,'stable_pairs':sum(x['stable'] for x in stability),'results':stability})
    order_sample=selected_rows[:60]; order_work=[]
    for result in order_sample:
        row=by_id[result['pair_id']]
        order_work.append((row,prof[row['right_entity_id']],prof[row['left_entity_id']],selected,'order_swap'))
    calls=run_calls(client,order_work,OUT/'diagnostic_call_records.json')
    swaps=[]
    primary_by={x['pair_id']:x['decision'] for x in order_sample}
    for x in calls:
        if x['tag']=='order_swap': swaps.append({'pair_id':x['pair_id'],'original':primary_by[x['pair_id']],'swapped':x['decision'],'changed':primary_by[x['pair_id']]!=x['decision']})
    write(OUT/'order_swap_results.json',{'pairs':len(swaps),'changed':sum(x['changed'] for x in swaps),'results':swaps})
    # Swap only the evidence/relations while retaining each entity's independent name/type.
    context_sample=[x for x in selected_rows if x['partition']=='VALIDATION'][:20]; swap_work=[]
    for result in context_sample:
        row=by_id[result['pair_id']]; a,b=prof[row['left_entity_id']],prof[row['right_entity_id']]
        a2=EntityProfile(a.local_id,a.canonical_name,a.entity_type,a.aliases,b.evidence,b.relations)
        b2=EntityProfile(b.local_id,b.canonical_name,b.entity_type,b.aliases,a.evidence,a.relations)
        swap_work.append((row,a2,b2,selected,'context_swap'))
    calls=run_calls(client,swap_work,OUT/'diagnostic_call_records.json')
    swapped=[x for x in calls if x['tag']=='context_swap']; original={x['pair_id']:x['decision'] for x in selected_rows}
    write(OUT/'context_ablation.json',{'representation_ablation':'see metrics.json representation_comparison','context_swap_pairs':len(swapped),'context_swap_changed':sum(x['decision']!=original[x['pair_id']] for x in swapped),'results':swapped})
    return len(calls)

def main():
    manifest,pairs,split,members,membership=load_inputs(); rows,excluded=dataset(pairs,split,membership); prof=profiles(members); OUT.mkdir(parents=True,exist_ok=True)
    integrity={'benchmark_version':manifest['benchmark_version'],'benchmark_hash':manifest['benchmark_hash'],'split_hash':manifest['split_hash'],'status':'PASS','total_pairs':len(pairs),'eligible_pairs':len(rows),'cross_partition_pairs_excluded':len(excluded),'eligible_by_partition':Counter(x['partition'] for x in rows),'exclusion_reason':'cluster-disjoint split cannot assign a pair whose endpoints belong to different partitions'}
    write(OUT/'benchmark_integrity.json',integrity); write(OUT/'excluded_cross_partition_pairs.json',excluded)
    profile_rows=[render_profile(p,r) for p in prof.values() for r in REPS]; write(OUT/'profile_manifest.json',{'pair_independent':True,'pre_resolution_only':True,'profiles':profile_rows,'sha256':canonical_json_hash(profile_rows)})
    baselines=[]
    for row in rows: baselines.append(row|{'decision':deterministic_baseline(prof[row['left_entity_id']],prof[row['right_entity_id']])})
    write(OUT/'baseline_results.json',{'by_partition':{p:metrics([x for x in baselines if x['partition']==p]) for p in ('DEVELOPMENT','VALIDATION','HELD_OUT')},'rows':baselines})
    client=Client(); selection=[x for x in rows if x['partition'] in {'DEVELOPMENT','VALIDATION'}]
    work=[(x,prof[x['left_entity_id']],prof[x['right_entity_id']],r,'primary') for x in selection for r in REPS]
    calls=run_calls(client,work,OUT/'verifier_call_records.json')
    comparison={}
    for rep in REPS:
        comparison[rep]={p:metrics([x for x in calls if x['representation']==rep and x['partition']==p and x['tag']=='primary']) for p in ('DEVELOPMENT','VALIDATION')}
    selected=choose({r:comparison[r]['VALIDATION'] for r in REPS}); frozen={'representation':selected,'system_prompt_sha256':canonical_json_hash(SYSTEM_PROMPT),'tool_schema_sha256':canonical_json_hash(verifier_tool()),'selection_rule':'validation precision-first, then false merges, selective accuracy, coverage, compactness','frozen_before_held_out':True}; write(OUT/'selected_config.json',frozen)
    held=[x for x in rows if x['partition']=='HELD_OUT']; calls=run_calls(client,[(x,prof[x['left_entity_id']],prof[x['right_entity_id']],selected,'primary') for x in held],OUT/'verifier_call_records.json')
    primary=[x for x in calls if x['representation']==selected and x['tag']=='primary']; full=metrics(primary); held_metrics=metrics([x for x in primary if x['partition']=='HELD_OUT'])
    report={'selected_config':frozen,'representation_comparison':comparison,'held_out':held_metrics,'full_audit':full,'bootstrap_95_percent':bootstrap(primary),'by_category':{k:metrics(v) for k,v in defaultdict(list).items()}}
    cats=defaultdict(list)
    for x in primary: cats[x.get('category') or 'UNSPECIFIED'].append(x)
    report['by_category']={k:metrics(v) for k,v in cats.items()}; write(OUT/'metrics.json',report)
    failures=[x for x in primary if x['decision']!=x['label']]; write(OUT/'error_analysis.json',{'errors':failures,'counts':Counter(('FALSE_MERGE' if x['decision']=='SAME_ENTITY' else 'FALSE_SPLIT' if x['decision']=='DIFFERENT_ENTITY' else 'UNCERTAIN') for x in failures)})
    latency=[x['latency_ms'] for x in primary]; cost={'semantic_calls':len(calls),'primary_selected_calls':len(primary),'failed_calls':sum(x['status']!='SUCCESS' for x in calls),'attempts_per_call':1,'latency_ms_mean':statistics.mean(latency),'latency_ms_p95':sorted(latency)[int(.95*len(latency))-1],'usage':dict(Counter({k:sum((x.get('usage') or {}).get(k,0) for x in calls) for k in ('prompt_tokens','completion_tokens','total_tokens')}))}; write(OUT/'cost_and_latency.json',cost)
    diagnostic_calls=diagnostics(client,rows,prof,selected)
    stability=read(OUT/'stability_results.json'); order_swap=read(OUT/'order_swap_results.json'); context_swap=read(OUT/'context_ablation.json')
    safety={'policy':'automatic merge only on SAME_ENTITY; DIFFERENT_ENTITY blocks merge; UNCERTAIN requires review/no automatic merge','eligible_pairs':len(primary),'automatic_merges':sum(x['decision']=='SAME_ENTITY' for x in primary),'false_automatic_merges':full['false_merges'],'blocked_distinct_pairs':sum(x['label']=='DIFFERENT_ENTITY' and x['decision']=='DIFFERENT_ENTITY' for x in primary),'review_or_no_merge':sum(x['decision']=='UNCERTAIN' for x in primary),'production_integration':'NONE'}
    write(OUT/'safety_simulation.json',safety); cost['diagnostic_semantic_calls']=diagnostic_calls; cost['total_semantic_calls']=len(calls)+diagnostic_calls; write(OUT/'cost_and_latency.json',cost)
    md=f"""# Identity-Specific Entity Verifier Experiment\n\nBenchmark 1.0.0 integrity: **PASS** (`{EXPECTED_BENCH}`). The frozen labels were not changed.\n\n## Isolation\n\nProfiles are pair-independent and use only pre-resolution names, types, source evidence, and local span relations. No retrieval scores, cluster IDs, gold labels, post-resolution neighbors, or final graph fields enter model payloads. Each configuration makes exactly one structured semantic call per pair with no retry, repair, voting, or fallback.\n\nThe cluster-disjoint split yields {len(rows)} eligible oracle pairs. {len(excluded)} cross-partition negative pairs remain in the frozen benchmark but are excluded from split metrics to prevent identity-cluster leakage.\n\n## Selection and held-out result\n\nSelected representation: **{selected}**. Held-out n={held_metrics['n']}, accuracy={held_metrics['accuracy']:.3f}, coverage={held_metrics['coverage']:.3f}, same precision={held_metrics['same_precision']:.3f}, false merges={held_metrics['false_merges']}, false splits={held_metrics['false_splits']}.\n\nFull eligible audit: n={full['n']}, accuracy={full['accuracy']:.3f}, coverage={full['coverage']:.3f}, selective accuracy={full['selective_accuracy']:.3f}, false merges={full['false_merges']}, false splits={full['false_splits']}.\n\n## Robustness diagnostics\n\nFive-run stability was exact on {stability['stable_pairs']}/{stability['pairs']} pairs. Order swapping changed {order_swap['changed']}/{order_swap['pairs']} decisions. Swapping evidence/relation context changed {context_swap['context_swap_changed']}/{context_swap['context_swap_pairs']} validation decisions. These sensitivities preclude unattended production merging.\n\n## Safety conclusion\n\nThe conservative simulation automatically merges {safety['automatic_merges']} pairs with {safety['false_automatic_merges']} observed false merges, blocks {safety['blocked_distinct_pairs']} distinct pairs, and leaves {safety['review_or_no_merge']} for review/no merge. This remains an offline oracle-pair experiment and is not integrated into production resolution.\n"""
    (ROOT/'release_audit/IDENTITY_SPECIFIC_VERIFIER_EXPERIMENT.md').write_text(md,encoding='utf-8')
    print(json.dumps({'selected':selected,'held_out':held_metrics,'full':full,'primary_and_selection_calls':len(calls),'diagnostic_calls':diagnostic_calls},indent=2))

if __name__=='__main__': main()
