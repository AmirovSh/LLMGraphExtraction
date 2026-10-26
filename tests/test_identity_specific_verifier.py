from devtools.identity_specific_verifier import (EntityProfile,VerifierDecision,
    assert_pre_resolution,canonical_pair,deterministic_baseline,metrics,render_profile,verifier_tool)


def profile(name="Example Service"):
    return EntityProfile("local:1",name,"SERVICE",("ES",),(('doc:S001','Example Service (ES) is available.'),),("uses -> Component [doc:S001]",))


def test_representations_are_bounded_and_pre_resolution():
    for representation in ("V1","V2","V3","V4","V4_LITE"):
        rendered=render_profile(profile(),representation)
        assert rendered["characters"] <= 800
        assert "identity_cluster_id" not in rendered["text"]


def test_forbidden_payload_fields_are_rejected():
    try: assert_pre_resolution({"profile":{"expected":"SAME_ENTITY"}})
    except ValueError: pass
    else: raise AssertionError("gold label leaked")


def test_contract_is_strict_and_has_three_decisions():
    schema=verifier_tool()["function"]["parameters"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["decision"]["enum"] == ["SAME_ENTITY","DIFFERENT_ENTITY","UNCERTAIN"]
    assert VerifierDecision(decision="UNCERTAIN",reason_codes=[],evidence_refs=[]).decision == "UNCERTAIN"


def test_baseline_and_metrics_are_conservative():
    assert deterministic_baseline(profile(),profile("example-service")) == "SAME_ENTITY"
    assert deterministic_baseline(profile(),profile("Different Service")) == "UNCERTAIN"
    result=metrics([{"label":"SAME_ENTITY","decision":"SAME_ENTITY"},{"label":"DIFFERENT_ENTITY","decision":"UNCERTAIN"}])
    assert result["same_precision"] == 1.0 and result["coverage"] == .5


def test_relation_limit_and_serialization_are_deterministic():
    p=EntityProfile("x","X","SERVICE",relations=("r1","r2","r3","r4"))
    first=render_profile(p,"V4_LITE",relation_limit=2)
    assert first == render_profile(p,"V4_LITE",relation_limit=2)
    assert first["raw_relation_count"] == 4
    assert first["retained_relation_count"] == 2


def test_canonical_pair_is_orientation_invariant():
    a=render_profile(profile("Alpha"),"V4_LITE",relation_limit=4)
    b=render_profile(profile("Beta"),"V4_LITE",relation_limit=4)
    assert canonical_pair(a,b) == canonical_pair(b,a)
