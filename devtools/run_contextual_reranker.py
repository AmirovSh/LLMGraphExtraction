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
    stems = ["Alder", "Boreal", "Cedar", "Drift", "Ember", "Fjord", "Grove", "Harbor", "Ion", "Juniper"]
    positive_forms = [("Records Management Authority", "Records Authority", "ORGANIZATION"),
        ("Identity Verification Service", "Identity Service", "SERVICE"),
        ("Secure Document Repository", "Document Repository", "REPOSITORY"),
        ("Regulatory Submission Gateway", "Submission Gateway", "SYSTEM")]
    negative_forms = [("Build Registry", "Build Registry Record", "COMPONENT", "RECORD", "process_record"),
        ("Access Policy", "Access Policy Result", "POLICY", "RESULT", "policy_result"),
        ("Identity Service", "Identity Service Adapter", "SERVICE", "COMPONENT", "service_component"),
        ("Evidence Container", "Evidence Artifact", "CONTAINER", "ARTIFACT", "container_contained"),
        ("Review Process", "Review Record", "PROCESS", "RECORD", "process_record"),
        ("Research Facility", "Research District", "FACILITY", "LOCATION", "facility_location"),
        ("Operator", "Operator Account", "PERSON", "ACCOUNT", "person_account"),
        ("Protocol Version 3", "Protocol Version 4", "VERSION", "VERSION", "version")]
    rows=[]
    def ent(i: int, side: str, name: str, kind: str, alias: str, evidence: str, direction: str,
            relation: str, neighbour: str) -> LocalEntityContext:
        return LocalEntityContext(local_id=f"C{i}{side}", canonical_name=name, entity_type=kind,
            aliases=[alias] if alias else [], evidence=[evidence],
            incidents=[LocalIncident(direction=direction, relation=relation, neighbour_name=neighbour, source_order=0)],
            source_kind="CONTROLLED_SYNTHETIC", identity_cluster_id=f"cluster-{i}")
    for i in range(120):
        stem=stems[(i//4)%len(stems)]
        if i % 3 == 0:
            full, short, kind = positive_forms[(i//3) % len(positive_forms)]
            left=f"{stem} {full}"; right=f"{stem} {short}"
            evidence_a=f"The {left} receives approved submissions under the {stem} charter."
            evidence_b=f"Staff refer to the {right} when routing approved submissions under that charter."
            a=ent(i,"A",left,kind,right,evidence_a,"outgoing","receives","Approved Submission")
            b=ent(i,"B",right,kind,left,evidence_b,"incoming","routes","Approved Submission")
            expected="MERGE"; category="true_alias"
            document=evidence_a+" "+evidence_b
        else:
            left_form,right_form,left_type,right_type,category=negative_forms[(i-(i//3)-1)%len(negative_forms)]
            left=f"{stem} {left_form}"; right=f"{stem} {right_form}"
            document=f"The {left} validates the request and writes the {right}. The {right} preserves the resulting lifecycle state."
            a=ent(i,"A",left,left_type,"",document,"outgoing","writes",right)
            b=ent(i,"B",right,right_type,"",document,"incoming","written_by",left)
            expected="SPLIT"
        rows.append({"case_id":f"CTX{i:03d}","identity_cluster_id":f"cluster-{i}","category":category,
                     "expected":expected,"source_kind":"CONTROLLED_SYNTHETIC","document":document,"entity_a":a,"entity_b":b})
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
