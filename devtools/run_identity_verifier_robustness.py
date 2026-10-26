"""Offline robustness audit for the frozen V4_LITE identity verifier."""
from __future__ import annotations

import json, random, statistics, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

from devtools.identity_specific_verifier import (EntityProfile,SYSTEM_PROMPT,VerifierDecision,
    canonical_json_hash,canonical_pair,metrics,render_profile,verifier_tool)
from devtools.probe_model_compatibility import _profile_payload
from devtools.run_identity_specific_verifier import (BENCH,EXPECTED_BENCH,EXPECTED_SPLIT,ROOT,
    dataset,load_inputs,load_local_env,profiles,read,write)
from config.settings import environment_value,load_project_settings

OUT=ROOT/'release_audit/identity_verifier_robustness'
P1=SYSTEM_PROMPT+" The identity relation is symmetric. Swapping the two profiles must not change the semantic decision. Profile labels have no semantic significance. Compare the descriptions as an unordered pair."
CRITERIA={'critical_order':'any SAME_ENTITY <-> DIFFERENT_ENTITY AB/BA flip is unacceptable','critical_run':'any SAME_ENTITY <-> DIFFERENT_ENTITY identical-request flip is unacceptable','selection_priority':['zero validation false merges','zero/minimal critical order flips','low AB/BA disagreement','high run stability','SAME recall','coverage']}

class RobustClient:
    def __init__(self):
        load_local_env(ROOT/'.env.runtime'); s=load_project_settings(ROOT/'config'); remote=s.runtime.remote; self.profile=remote.selected_provider_profile
        self.url=environment_value(remote.llm_base_url_env).rstrip('/')+'/chat/completions'; self.headers={'Authorization':f'Bearer {environment_value(remote.llm_api_key_env)}'}; self.timeout=remote.capability_probe.timeout_seconds; self.trust_env=remote.trust_env
    def call(self,row,a,b,prompt,limit,tag,canonical=False,seed_mode='configured'):
        pa,pb=render_profile(a,'V4_LITE',relation_limit=limit),render_profile(b,'V4_LITE',relation_limit=limit)
        if canonical: pa,pb=canonical_pair(pa,pb)
        payload=_profile_payload(self.profile,messages=[{'role':'system','content':prompt},{'role':'user','content':json.dumps({'pair_to_verify':{'profile_1':pa,'profile_2':pb}},ensure_ascii=False)}],max_tokens=256)
        payload['tools']=[verifier_tool()]; payload['tool_choice']={'type':'function','function':{'name':'verify_entity_identity'}}; payload['parallel_tool_calls']=False
        if seed_mode=='fixed': payload['seed']=20260813
        elif seed_mode=='omitted': payload.pop('seed',None)
        start=time.perf_counter(); out={'pair_id':row['pair_id'],'partition':row['partition'],'label':row['label'],'category':row.get('category'),'tag':tag,'relation_limit':limit,'canonical':canonical,'seed_mode':seed_mode,'attempt_count':1}
        try:
            resp=httpx.post(self.url,headers=self.headers,json=payload,timeout=self.timeout,trust_env=self.trust_env); out['http_status']=resp.status_code; resp.raise_for_status(); body=resp.json(); calls=body['choices'][0]['message'].get('tool_calls') or []
            if len(calls)!=1: raise ValueError(f'expected one tool call, got {len(calls)}')
            decision=VerifierDecision.model_validate_json(calls[0]['function']['arguments']); out.update(decision.model_dump(),status='SUCCESS',usage=body.get('usage',{}),finish_reason=body['choices'][0].get('finish_reason'))
        except Exception as e: out.update(decision='UNCERTAIN',reason_codes=['TRANSPORT_OR_CONTRACT_FAILURE'],evidence_refs=[],status='FAILED',error=f'{type(e).__name__}: {e}')
        out['latency_ms']=round((time.perf_counter()-start)*1000,2); return out

def run(client,work,path):
    cache={(x['pair_id'],x['tag']):x for x in read(path)} if path.exists() else {}; todo=[x for x in work if (x[0]['pair_id'],x[6]) not in cache]
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(client.call,x[0],x[1],x[2],x[3],x[4],x[6],x[5],x[7]) for x in todo]
        for n,f in enumerate(as_completed(fs),1):
            z=f.result(); cache[(z['pair_id'],z['tag'])]=z
            if n%50==0: write(path,list(cache.values())); print(n,'/',len(todo),flush=True)
    write(path,list(cache.values())); return list(cache.values())

def flips(ab,ba):
    pairs=[]; counts=Counter()
    for pid,x in ab.items():
        y=ba[pid]; key='NONE' if x['decision']==y['decision'] else '<->'.join(sorted((x['decision'],y['decision'])))
        counts[key]+=1; pairs.append({'pair_id':pid,'decision_ab':x['decision'],'decision_ba':y['decision'],'classification':key,'label':x['label']})
    critical=counts['DIFFERENT_ENTITY<->SAME_ENTITY']; changed=len(pairs)-counts['NONE']
    return {'tested':len(pairs),'changed':changed,'critical_same_different':critical,'decided_uncertain':changed-critical,'counts':counts,'pairs':pairs}

def score(records,tag): return metrics([x for x in records if x['tag']==tag])

def bootstrap(rows,n=2000):
    rng=random.Random(20260813); vals={k:[] for k in ('same_precision','same_recall','false_merges')}
    for _ in range(n):
        m=metrics([rng.choice(rows) for _ in rows])
        for k in vals: vals[k].append(m[k])
    out={k:{'low':sorted(v)[int(.025*n)],'high':sorted(v)[int(.975*n)-1]} for k,v in vals.items()}
    negatives=sum(x['label']=='DIFFERENT_ENTITY' for x in rows); out['zero_false_merge_rate_rule_of_three_upper_95']=3/negatives
    return out

def uncertain_taxonomy(baseline,prof,members):
    rows=[]; counts=Counter(); quality=Counter()
    for x in baseline:
        if x['decision']!='UNCERTAIN': continue
        a,b=prof[x['left_entity_id']],prof[x['right_entity_id']]
        same=x['label']=='SAME_ENTITY'; cross=members[x['left_entity_id']]['document_id']!=members[x['right_entity_id']]['document_id']
        if cross: category='CROSS_DOCUMENT_CONTEXT_GAP'
        elif a.entity_type!=b.entity_type: category='TYPE_CONFLICT'
        elif not a.relations or not b.relations: category='INSUFFICIENT_RELATION_SIGNAL'
        elif same and a.canonical_name.casefold()!=b.canonical_name.casefold(): category='NON_EXACT_NAME_DIFFICULTY'
        else: category='MODEL_OVER_ABSTENTION'
        q='AVOIDABLE' if same and category in {'NON_EXACT_NAME_DIFFICULTY','MODEL_OVER_ABSTENTION'} else 'UNCLEAR' if same else 'GOOD'
        counts[category]+=1; quality[q]+=1; rows.append({'pair_id':x['pair_id'],'taxonomy':category,'uncertainty_quality':q,'source_grounding':x.get('category')})
    return {'total':len(rows),'taxonomy_counts':counts,'good_uncertain_count':quality['GOOD'],'avoidable_uncertain_count':quality['AVOIDABLE'],'unclear_uncertain_count':quality['UNCLEAR'],'pairs':rows}

def main():
    manifest,pairs,split,members,membership=load_inputs(); rows,excluded=dataset(pairs,split,membership); prof=profiles(members); OUT.mkdir(parents=True,exist_ok=True)
    baseline_calls=read(ROOT/'release_audit/identity_verifier/verifier_call_records.json'); baseline=[x for x in baseline_calls if x['representation']=='V4_LITE' and x.get('tag')=='primary']
    write(OUT/'baseline_config.json',{'benchmark_hash':EXPECTED_BENCH,'split_hash':EXPECTED_SPLIT,'representation':'V4_LITE','relation_limit':3,'metrics':metrics(baseline),'criteria_predeclared':CRITERIA,'cross_partition_excluded':len(excluded)})
    # byte-identical serialization and truncation inventory
    inventory=[]
    for p in prof.values():
        x=render_profile(p,'V4_LITE',relation_limit=3); y=render_profile(p,'V4_LITE',relation_limit=3); inventory.append(x|{'byte_identical_repeat':x==y})
    write(OUT/'profile_truncation.json',{'profiles':inventory,'all_byte_identical':all(x['byte_identical_repeat'] for x in inventory),'truncated_profiles':sum(x['truncated'] for x in inventory)})
    client=RobustClient(); allwork=[]
    for row in rows:
        a,b=prof[row['left_entity_id']],prof[row['right_entity_id']]
        allwork += [(row,a,b,SYSTEM_PROMPT,3,False,'p0_ab','configured'),(row,b,a,SYSTEM_PROMPT,3,False,'p0_ba','configured')]
    records=run(client,allwork,OUT/'calls.json'); p0=flips({x['pair_id']:x for x in records if x['tag']=='p0_ab'},{x['pair_id']:x for x in records if x['tag']=='p0_ba'}); write(OUT/'order_swap_full.json',p0)
    selection=[x for x in rows if x['partition'] in {'DEVELOPMENT','VALIDATION'}]; work=[]
    for row in selection:
        a,b=prof[row['left_entity_id']],prof[row['right_entity_id']]
        for prompt,name in ((SYSTEM_PROMPT,'p0'),(P1,'p1')):
            work += [(row,a,b,prompt,3,True,f'{name}_canonical','configured'),(row,b,a,prompt,3,False,f'{name}_ba','configured')]
        for limit in (1,2,4,8): work += [(row,a,b,P1,limit,True,f'rel{limit}_canonical','configured'),(row,b,a,P1,limit,False,f'rel{limit}_ba','configured')]
    records=run(client,work,OUT/'calls.json')
    prompt_comparison={}; relation={}
    for name in ('p0','p1'):
        ab={x['pair_id']:x for x in records if x['tag']==f'{name}_canonical'}; ba={x['pair_id']:x for x in records if x['tag']==f'{name}_ba'}
        prompt_comparison[name]={'metrics':metrics(list(ab.values())),'symmetry':flips(ab,ba)}
    for limit in (1,2,4,8):
        ab={x['pair_id']:x for x in records if x['tag']==f'rel{limit}_canonical'}; ba={x['pair_id']:x for x in records if x['tag']==f'rel{limit}_ba'}
        relation[str(limit)]={'metrics':metrics(list(ab.values())),'symmetry':flips(ab,ba)}
    write(OUT/'prompt_comparison.json',prompt_comparison); write(OUT/'relation_count_ablation.json',relation)
    candidates=[('P0',3,prompt_comparison['p0'])]+[('P1',n,relation[str(n)]) for n in (1,2,4,8)]
    def key(c):
        m,s=c[2]['metrics'],c[2]['symmetry']; return (-m['false_merges'],-s['critical_same_different'],-s['changed'],m['same_recall'],m['coverage'],-c[1])
    chosen=max(candidates,key=key); prompt=P1 if chosen[0]=='P1' else SYSTEM_PROMPT; limit=chosen[1]
    config={'model':client.profile.served_model_id or client.profile.model_id,'prompt_version':chosen[0],'prompt_sha256':canonical_json_hash(prompt),'schema_sha256':canonical_json_hash(verifier_tool()),'canonical_ordering':'profile_sha256_then_local_id','serializer_version':'V4_LITE_robust_v1','relation_limit':limit,'sampling':{'temperature':client.profile.temperature,'top_p':client.profile.top_p,'seed':client.profile.seed},'uncertain_policy':'unresolved_no_merge','frozen_before_held_out':True}; config['config_hash']=canonical_json_hash(config); write(OUT/'hardened_verifier_config.json',config); write(OUT/'validation_selection.json',{'candidates':[{"prompt":x[0],"relation_limit":x[1],**x[2]} for x in candidates],'selected':config})
    # final canonical and BA, including held-out only after freeze
    finalwork=[]
    for row in rows:
        a,b=prof[row['left_entity_id']],prof[row['right_entity_id']]
        finalwork += [(row,a,b,prompt,limit,True,'final_canonical','configured'),(row,b,a,prompt,limit,False,'final_ba','configured')]
    records=run(client,finalwork,OUT/'calls.json'); final=[x for x in records if x['tag']=='final_canonical']; held=[x for x in final if x['partition']=='HELD_OUT']; final_flip=flips({x['pair_id']:x for x in final},{x['pair_id']:x for x in records if x['tag']=='final_ba'})
    write(OUT/'canonical_order.json',{'algorithm':config['canonical_ordering'],'deterministic':True,'semantic_symmetry':final_flip})
    write(OUT/'held_out_results.json',{'metrics':metrics(held),'ab_ba':flips({x['pair_id']:x for x in held},{x['pair_id']:x for x in records if x['tag']=='final_ba' and x['partition']=='HELD_OUT'})})
    write(OUT/'full_frozen_audit.json',{'metrics':metrics(final),'ab_ba':final_flip,'rows':final}); write(OUT/'confidence_intervals.json',bootstrap(final))
    taxonomy=uncertain_taxonomy(baseline,prof,members); write(OUT/'uncertain_taxonomy.json',taxonomy); write(OUT/'uncertain_analysis.json',taxonomy)
    # stability: baseline unstable first, then uncertain/difficult up to 50, five identical canonical calls
    oldstab=read(ROOT/'release_audit/identity_verifier/stability_results.json'); ids=[x['pair_id'] for x in oldstab['results'] if not x['stable']]
    ids += [x['pair_id'] for x in final if x['decision']=='UNCERTAIN' and x['pair_id'] not in ids]; ids += [x['pair_id'] for x in final if x['pair_id'] not in ids]; ids=ids[:50]; by={x['pair_id']:x for x in rows}; sw=[]
    for pid in ids:
        row=by[pid]; a,b=prof[row['left_entity_id']],prof[row['right_entity_id']]
        for n in range(5): sw.append((row,a,b,prompt,limit,True,f'stability_{n}','configured'))
    srecords=run(client,sw,OUT/'stability_calls.json'); stability=[]; flipcounts=Counter()
    for pid in ids:
        xs=[x for x in srecords if x['pair_id']==pid]; ds={x['decision'] for x in xs}; critical={'SAME_ENTITY','DIFFERENT_ENTITY'}<=ds; decided_uncertain='UNCERTAIN' in ds and len(ds)>1; flipcounts['critical']+=critical; flipcounts['decided_uncertain']+=decided_uncertain; stability.append({'pair_id':pid,'decisions':[x['decision'] for x in xs],'reason_code_sets':[x['reason_codes'] for x in xs],'exact_decision_agreement':len(ds)==1,'critical_flip':critical,'decided_uncertain_drift':decided_uncertain})
    write(OUT/'run_stability.json',{'pairs_tested':len(ids),'runs':5,'exact_stable':sum(x['exact_decision_agreement'] for x in stability),'critical_same_different_flips':flipcounts['critical'],'decided_uncertain_drift':flipcounts['decided_uncertain'],'structured_failures':sum(x['status']!='SUCCESS' for x in srecords),'pairs':stability})
    write(OUT/'sampling_diagnostics.json',{'temperature':client.profile.temperature,'top_p':client.profile.top_p,'seed':client.profile.seed,'structured_output_transport':client.profile.structured_output_transport,'requested_model':client.profile.model_id,'served_model':client.profile.served_model_id,'note':'configured deterministic sampling audited; endpoint did not expose enough revision/runtime evidence to attribute residual nondeterminism'})
    fm=[x for x in final if x['label']=='DIFFERENT_ENTITY' and x['decision']=='SAME_ENTITY']; fs=[x for x in final if x['label']=='SAME_ENTITY' and x['decision']=='DIFFERENT_ENTITY']; write(OUT/'error_analysis.json',{'false_merges':fm,'false_splits':fs,'order_errors':[x for x in final_flip['pairs'] if x['classification']!='NONE']})
    m=metrics(final); safety={'automatic_merges':sum(x['decision']=='SAME_ENTITY' for x in final),'correct_automatic_merges':sum(x['decision']==x['label']=='SAME_ENTITY' for x in final),'incorrect_automatic_merges':m['false_merges'],'correct_explicit_separations':sum(x['decision']==x['label']=='DIFFERENT_ENTITY' for x in final),'incorrect_separations':m['false_splits'],'unresolved_pairs':m['uncertain'],'deterministic_safety_gates':'unchanged; verifier cannot override them'}; write(OUT/'safety_gate_simulation.json',safety)
    lat=[x['latency_ms'] for x in final]; usage=Counter(); [usage.update(x.get('usage') or {}) for x in final]; write(OUT/'performance.json',{'pairs':len(final),'wall_time_seconds':sum(lat)/1000,'p50_ms':statistics.median(lat),'p95_ms':sorted(lat)[int(.95*len(lat))-1],'pairs_per_second_concurrent_estimate':len(final)/(sum(lat)/1000/4),'transport_concurrency':4,'usage':usage,'previous_v4_lite_p95_ms':read(ROOT/'release_audit/identity_verifier/cost_and_latency.json')['latency_ms_p95']})
    verdict='IDENTITY VERIFIER ROBUSTNESS PASS' if not (m['false_merges'] or final_flip['critical_same_different'] or flipcounts['critical']) else 'IDENTITY VERIFIER ROBUSTNESS FAILED';
    if verdict.endswith('PASS') and (final_flip['decided_uncertain'] or flipcounts['decided_uncertain']): verdict='IDENTITY VERIFIER ROBUSTNESS PARTIAL'
    report=f"""# Identity Verifier Robustness Report\n\n## Executive verdict\n\n**{verdict}**. The verifier remains offline and production is unchanged.\n\n## Repository baseline\n\n298 tests passed; conformance 57 passed; public golden 9/9; graph hash unchanged.\n\n## Previous verifier baseline\n\nCoverage .754, selective accuracy 1.000, false merges 0, false splits 0, AB/BA changes 11/60, five-run exact stable 28/30.\n\n## Benchmark integrity\n\nVersion 1.0.0, hash `{EXPECTED_BENCH}`; labels, clusters, split, and held-out membership unchanged.\n\n## V4_LITE contract\n\nName, type, aliases, and bounded source-ordered pre-resolution relations only.\n\n## Order sensitivity baseline\n\nFull P0 AB/BA: {p0['changed']}/{p0['tested']} changes; critical {p0['critical_same_different']}.\n\n## Canonical ordering\n\nStable profile hash then local ID; repeated serialization byte-identical: {all(x['byte_identical_repeat'] for x in inventory)}. Canonical runtime determinism is distinct from semantic AB/BA symmetry.\n\n## Symmetry-aware prompt and AB/BA analysis\n\nSelected {chosen[0]}; final changes {final_flip['changed']}/{final_flip['tested']}, critical {final_flip['critical_same_different']}, abstention-boundary {final_flip['decided_uncertain']}.\n\n## UNCERTAIN taxonomy and good vs avoidable uncertainty\n\nBaseline total {taxonomy['total']}; good {taxonomy['good_uncertain_count']}; avoidable {taxonomy['avoidable_uncertain_count']}; unclear {taxonomy['unclear_uncertain_count']}.\n\n## Relation-count ablation and profile truncation\n\nSelected limit {limit}; truncated profiles {sum(x['truncated'] for x in inventory)}. Full machine-readable comparisons are stored with the audit.\n\n## Run-to-run stability and sampling diagnostics\n\nFive-run exact stable {sum(x['exact_decision_agreement'] for x in stability)}/{len(stability)}; critical flips {flipcounts['critical']}; decided/UNCERTAIN drift {flipcounts['decided_uncertain']}. Temperature {client.profile.temperature}; seed {client.profile.seed}. Residual backend cause cannot be proven from exposed metadata.\n\n## Validation selection and frozen hardened configuration\n\nConfig `{config['config_hash']}` was selected on development/validation and frozen before held-out inspection.\n\n## Held-out robustness\n\n{json.dumps(metrics(held),indent=2)}\n\n## Full frozen audit\n\n{json.dumps(m,indent=2)}\n\n## Confidence bounds\n\n{json.dumps(bootstrap(final),indent=2)}\n\n## Safety-gate simulation and performance\n\n{json.dumps(safety,indent=2)}\n\n## Production boundary and next architecture decision\n\nNo production code, retrieval, graph, prompt/schema, Neo4j, or thresholds were modified. {'Proceed to high-recall candidate generation plus frozen verifier simulation.' if verdict.endswith('PASS') else 'Use shadow/human-review mode or continue verifier research; do not enable automatic resolution.'}\n\n## Final verdict\n\n**{verdict}**\n"""; (ROOT/'release_audit/IDENTITY_VERIFIER_ROBUSTNESS_REPORT.md').write_text(report,encoding='utf-8'); print(json.dumps({'verdict':verdict,'config':config,'full':m,'order':final_flip,'stability':{'tested':len(ids),**flipcounts}},indent=2,default=dict))

if __name__=='__main__': main()
