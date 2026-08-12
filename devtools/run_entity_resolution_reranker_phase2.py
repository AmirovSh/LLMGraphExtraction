"""Score Phase 2 benchmark, freeze calibration, then evaluate held-out."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from config.settings import load_project_settings,environment_value
from devtools.build_entity_resolution_reranker_phase2_benchmark import build
from devtools.entity_resolution_reranker import EntityProfile,PairCase,RerankerClient,RerankerConfig,directional_pairs,join_directional_results
from devtools.entity_resolution_reranker_phase2 import *
from devtools.run_entity_resolution_reranker_experiment import benchmark_bge_scores,load_local_env,live_replay,write_json

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"release_audit/reranker_phase2"
def main()->int:
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=OUT);parser.add_argument("--concurrency",type=int,default=16);args=parser.parse_args();out=args.output
 load_local_env(ROOT/".env.runtime"); settings=load_project_settings(ROOT/"config"); remote=settings.runtime.remote; ers=settings.entity_resolution
 defs=json.loads((ROOT/"devtools/fixtures/entity_resolution_reranker_phase2_definitions.json").read_text()); bench=build(defs); split=fixed_cluster_split(bench.cases); assert_no_cluster_leakage(bench.cases,split)
 cases=[]
 for c in bench.cases:
  ga=c.entity_a.get("graph_group",c.graph_group); gb=c.entity_b.get("graph_group",c.graph_group)
  cases.append(PairCase(case_id=c.case_id,entity_a=EntityProfile(canonical_name=c.entity_a["canonical_name"],entity_type=c.entity_a["entity_type"],aliases=c.entity_a.get("aliases",[]),graph_group=ga),entity_b=EntityProfile(canonical_name=c.entity_b["canonical_name"],entity_type=c.entity_b["entity_type"],aliases=c.entity_b.get("aliases",[]),graph_group=gb),expected=c.expected,source=c.source_artifact,reason=c.review_reason))
 endpoint=environment_value(remote.embedding_base_url_env); key=environment_value(remote.embedding_api_key_env); client=RerankerClient(RerankerConfig(endpoint,key,timeout_seconds=remote.embedding_timeout_seconds,trust_env=remote.trust_env))
 bge_started=time.perf_counter();bge=benchmark_bge_scores(type("B",(),{"cases":cases})(),endpoint=endpoint,api_key=key,model=ers.embedding.model,timeout=remote.embedding_timeout_seconds,trust_env=remote.trust_env);bge_seconds=time.perf_counter()-bge_started
 rows=[]; perf={}
 for variant in ("R1","R2"):
  started=time.perf_counter(); raw=client.score_many(directional_pairs(cases,variant),batch_size=args.concurrency); perf[variant]={"wall_seconds":time.perf_counter()-started,"pairs":len(cases),"operations":len(raw)}; scores=join_directional_results(raw)
  for c,p in zip(bench.cases,cases,strict=True):
   sim=bge[c.case_id]; baseline="INELIGIBLE" if c.isolation_only else ("MERGE" if sim>=.90 and p.entity_a.canonical_name.casefold()==p.entity_b.canonical_name.casefold() else "UNRESOLVED" if sim>=.76 else "SPLIT")
   rows.append({"case_id":c.case_id,"partition":split[c.case_id],"input_variant":variant,"expected":c.expected,"category":c.category,"isolation_only":c.isolation_only,"deterministic_block":c.deterministic_block,"same_graph_group":p.entity_a.graph_group==p.entity_b.graph_group,"entity_a_key":c.identity_cluster+":a","entity_b_key":c.identity_cluster+":b","bge_similarity":sim,"baseline_decision":baseline,**scores[c.case_id]})
 floors=[.76,.72,.68,.64,.60,.55]; train=[r for r in rows if r["partition"] in ("CALIBRATION","VALIDATION")]; frozen=calibrate_without_held_out(train,production_floor=.76,floors=floors)|{"top_k":20,"model":"bge-reranker-v2-m3","benchmark_version":"2"}
 held=[r for r in rows if r["partition"]=="HELD_OUT_TEST" and r["input_variant"]==frozen["input_variant"]]
 for r in held:r["decision"]=expanded_decision(r,production_floor=.76,candidate_floor=frozen["candidate_floor"],threshold=frozen["threshold"],aggregation=frozen["aggregation"])
 base=metrics(held,"baseline_decision"); result=metrics(held,"decision"); bootstrap=bootstrap_intervals(held,"decision")
 isolation=[r for r in rows if r["partition"]=="ISOLATION" and r["input_variant"]==frozen["input_variant"]]; assert all(expanded_decision(r,production_floor=.76,candidate_floor=frozen["candidate_floor"],threshold=frozen["threshold"],aggregation=frozen["aggregation"])=="INELIGIBLE" for r in isolation)
 sweep=candidate_sweep([r for r in rows if r["input_variant"]==frozen["input_variant"]],floors,production_floor=.76); topk=top_k_sweep([r for r in rows if r["input_variant"]==frozen["input_variant"]],[5,10,20],floor=frozen["candidate_floor"])
 errors=[{"case_id":r["case_id"],"category":r["category"],"taxonomy":"CANDIDATE_NOT_RETRIEVED" if r["expected"]=="MERGE" and r["bge_similarity"]<frozen["candidate_floor"] else "RERANKER_FALSE_NEGATIVE" if r["expected"]=="MERGE" else "RERANKER_FALSE_POSITIVE"} for r in held if (r["decision"]=="MERGE")!=(r["expected"]=="MERGE")]
 replay_started=time.perf_counter();replay,replay_perf=live_replay(ROOT/"release_audit/live_acceptance_outputs/batch_run_001",client=client,variant=frozen["input_variant"],aggregation=frozen["aggregation"],strategy="R-AMBIGUOUS",threshold=frozen["threshold"],batch_size=args.concurrency);replay_e2e=time.perf_counter()-replay_started
 subset=[c for c in cases if split[c.case_id]!="ISOLATION"][:24];stability_runs=[]
 for _ in range(3):
  sr=join_directional_results(client.score_many(directional_pairs(subset,frozen["input_variant"]),batch_size=args.concurrency));stability_runs.append(sr)
 drifts=[];classification_drift=0
 for c in subset:
  values=[run[c.case_id][frozen["aggregation"]] for run in stability_runs];drifts.append(max(values)-min(values));classification_drift+=len({v>=frozen["threshold"] for v in values})>1
 stability={"runs":3,"subset_pairs":len(subset),"maximum_absolute_score_drift":max(drifts),"mean_absolute_score_drift":sum(drifts)/len(drifts),"classification_drift_pairs":classification_drift,"ranking_drift":"not_materially_observed" if max(drifts)==0 else "not_assessed_beyond_score_drift","status":"PASS" if classification_drift==0 else "FAIL"}
 costs={str(ratio):{"baseline_weighted_error":base["false_merges"]*ratio+base["missed_merges"],"reranker_weighted_error":result["false_merges"]*ratio+result["missed_merges"]} for ratio in (10,5,3,1)}
 out.mkdir(parents=True,exist_ok=True); artifacts={"benchmark_v2.json":bench.model_dump(mode="json"),"split_manifest.json":{"seed":20260812,"partitions":split},"bge_candidate_sweep.json":sweep,"reranker_scores.json":rows,"calibration_results.json":frozen["calibration"],"validation_results.json":frozen["validation"],"held_out_results.json":{"baseline":base,"reranker":result,"cost_sensitive":costs},"bootstrap_intervals.json":bootstrap,"strategy_comparison.json":{"frozen":frozen,"R_EXPANDED_selected":frozen["candidate_floor"]<.76},"error_analysis.json":errors,"multi_document_replay.json":replay,"performance.json":{"bge_embedding_seconds":bge_seconds,"scoring":perf,"replay":replay_perf,"end_to_end_replay_seconds":replay_e2e,"concurrency":args.concurrency},"stability.json":stability}
 for name,value in artifacts.items():write_json(out/name,value)
 write_json(out/"frozen_config.json",frozen|{"fingerprint":fingerprint_config(frozen)})
 print(json.dumps({"eligible":120,"split":{p:list(split.values()).count(p) for p in ("CALIBRATION","VALIDATION","HELD_OUT_TEST")},"frozen":frozen,"baseline":base,"held_out":result,"bootstrap":bootstrap,"candidate_sweep":sweep,"top_k":topk},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
