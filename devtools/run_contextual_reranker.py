"""Run the fixed-pair matched-input Phase-3R experiment (never production imported)."""
from __future__ import annotations

import json, math, statistics, time
from pathlib import Path
from typing import Any

from config.settings import environment_value, load_project_settings
from devtools.contextual_entity_profiles import LocalEntityContext, LocalIncident, eligibility, render_profile, serializer_contract
from devtools.entity_resolution_reranker import DirectionalPair, RerankerClient, RerankerConfig, join_directional_results
from devtools.entity_resolution_reranker_phase3 import canonical_json_hash, metrics
from devtools.run_entity_resolution_reranker_experiment import load_local_env, write_json
from runtime.observable_entity_resolution import cosine, request_embeddings

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "release_audit/context_reranker"


def benchmark() -> list[dict[str, Any]]:
    nouns = [("Registry", "Record"), ("Gateway", "Manifest"), ("Policy", "Result"), ("Service", "Component"),
             ("Person", "Account"), ("Container", "Artifact"), ("Approval Process", "Approval Record"),
             ("Facility", "Location"), ("Build", "Definition"), ("Release", "Ledger")]
    rows=[]
    for i in range(120):
        left, right = nouns[i % len(nouns)]; merge = i % 3 == 0
        base = f"Delta {left} {i//10}"; other = base if merge else f"Delta {left} {right} {i//10}"
        doc = (f"The {base} validates approved inputs and records the decision. "
               f"The {other} is referenced by the local workflow and preserves its own lifecycle.")
        def ent(local_id: str, name: str, relation: str, neighbour: str) -> LocalEntityContext:
            return LocalEntityContext(local_id=local_id, canonical_name=name, entity_type="COMPONENT" if "Record" not in name and "Account" not in name else "RECORD",
                aliases=([name.replace("Delta ", "")] if merge else []), evidence=[doc],
                incidents=[LocalIncident(direction="outgoing", relation=relation, neighbour_name=neighbour, source_order=0)],
                source_kind="CONTROLLED_SYNTHETIC", identity_cluster_id=f"cluster-{i}")
        a=ent(f"C{i}A",base,"validates",other); b=ent(f"C{i}B",other,"preserves",base)
        rows.append({"case_id":f"CTX{i:03d}","identity_cluster_id":f"cluster-{i}","category":"true_alias" if merge else "RELATED_BUT_DISTINCT",
                     "expected":"MERGE" if merge else "SPLIT","source_kind":"CONTROLLED_SYNTHETIC","document":doc,"entity_a":a,"entity_b":b})
    return rows


def split(rows: list[dict]) -> dict[str,str]:
    return {r["case_id"]:("CALIBRATION" if i%5<3 else "VALIDATION" if i%5==3 else "HELD_OUT") for i,r in enumerate(rows)}


def auc(rows: list[dict], key: str) -> float:
    pos=[r[key] for r in rows if r["expected"]=="MERGE"]; neg=[r[key] for r in rows if r["expected"]=="SPLIT"]
    return sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))


def best_threshold(rows: list[dict], key: str) -> dict:
    choices=sorted({r[key] for r in rows}, reverse=True); best=None
    for t in choices:
        evaluated=[r|{"decision":"MERGE" if r[key]>=t else "SPLIT"} for r in rows]; m=metrics(evaluated)
        candidate={"threshold":t,**m}
        if best is None or (-m["false_merges"],m["precision"],m["recall"],m["f1"]) > (-best["false_merges"],best["precision"],best["recall"],best["f1"]): best=candidate
    return best


def main() -> int:
    OUT.mkdir(parents=True,exist_ok=True); cases=benchmark(); partitions=split(cases)
    load_local_env(ROOT/".env.runtime"); settings=load_project_settings(ROOT/"config"); remote=settings.runtime.remote
    endpoint=environment_value(remote.embedding_base_url_env); key=environment_value(remote.embedding_api_key_env)
    client=RerankerClient(RerankerConfig(endpoint,key,timeout_seconds=remote.embedding_timeout_seconds,trust_env=remote.trust_env))
    all_rows=[]; performance={}; profile_examples={}; eligible=[]
    for rep in ("R1","R2","R3","R4"):
        profiles={}
        for case in cases:
            for side in ("entity_a","entity_b"):
                e=case[side]; profiles[e.local_id]=render_profile(e,rep)["text"]
        profile_examples[rep]=list(profiles.items())[:4]
        started=time.perf_counter(); ids=list(profiles); vectors,body,_=request_embeddings(endpoint=endpoint,api_key=key,model=settings.entity_resolution.embedding.model,
            inputs=[{"local_entity_id":i,"embedding_text":profiles[i]} for i in ids],timeout_seconds=remote.embedding_timeout_seconds,trust_env=remote.trust_env,write_artifact=lambda *_:None)
        bge_time=time.perf_counter()-started; by_id=dict(zip(ids,vectors,strict=True))
        directional=[]
        for case in cases:
            a,b=case["entity_a"].local_id,case["entity_b"].local_id
            directional += [DirectionalPair(case["case_id"],profiles[a],profiles[b],"ab"),DirectionalPair(case["case_id"],profiles[b],profiles[a],"ba")]
        started=time.perf_counter(); raw=client.score_many(directional,batch_size=16); rerank_time=time.perf_counter()-started; joined=join_directional_results(raw)
        for case in cases:
            a,b=case["entity_a"].local_id,case["entity_b"].local_id; scores=joined[case["case_id"]]
            all_rows.append({"case_id":case["case_id"],"partition":partitions[case["case_id"]],"representation":rep,"expected":case["expected"],"category":case["category"],
                "source_kind":case["source_kind"],"bge":cosine(by_id[a],by_id[b]),**scores})
        performance[rep]={"profiles":len(profiles),"bge_seconds":bge_time,"reranker_seconds":rerank_time,"pairs_per_second":len(cases)/rerank_time,
            "mean_input_characters":statistics.mean(len(v) for v in profiles.values())}
    comparisons={}
    for rep in ("R1","R2","R3","R4"):
        rr=[r for r in all_rows if r["representation"]==rep]; cal=[r for r in rr if r["partition"]=="CALIBRATION"]
        comparisons[rep]={}
        for model_key in ("bge","mean","min","max"):
            frozen=best_threshold(cal,model_key); held=[r|{"decision":"MERGE" if r[model_key]>=frozen["threshold"] else "SPLIT"} for r in rr if r["partition"]=="HELD_OUT"]
            comparisons[rep][model_key]={"selection":frozen,"held_out":metrics(held),"roc_auc_all":auc(rr,model_key)}
    write_json(OUT/"profile_contract.json",serializer_contract()); write_json(OUT/"profile_examples.json",profile_examples)
    write_json(OUT/"benchmark_context_v1.json",[{k:(v.model_dump(mode="json") if hasattr(v,"model_dump") else v) for k,v in r.items()} for r in cases])
    manifest={"seed":20260813,"partitions":partitions}; manifest["split_hash"]=canonical_json_hash(manifest); write_json(OUT/"split_manifest.json",manifest)
    write_json(OUT/"representation_eligibility.json",[{"case_id":r["case_id"],"a":eligibility(r["entity_a"]),"b":eligibility(r["entity_b"])} for r in cases])
    write_json(OUT/"bge_matched_scores.json",[{k:v for k,v in r.items() if k not in {"ab","ba","mean","min","max"}} for r in all_rows])
    write_json(OUT/"reranker_directional_scores.json",all_rows); write_json(OUT/"representation_comparison.json",comparisons); write_json(OUT/"performance.json",performance)
    print(json.dumps({"benchmark_pairs":len(cases),"split_hash":manifest["split_hash"],"comparison":comparisons,"performance":performance},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
