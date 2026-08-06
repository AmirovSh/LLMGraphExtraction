# DeepSeek and BGE-M3 live acceptance summary

**Verdict: LIVE ACCEPTANCE PASS WITH SEMANTIC QUALITY FINDINGS**

The live review covered branch `feature/multi-document-graph-groups`, the
legacy single-document path, grouped 2 + 3 + 1 processing, explicit DeepSeek
served-model identity, BGE-M3 embeddings, and JSON/Neo4j parity. This is
sanitized acceptance evidence for an experimental release candidate.

## Provider compatibility

- Requested DeepSeek identity: `deepseek-ai/DeepSeek-V4-Flash-0731`.
- Explicit configured and observed served identity: `deepseek-v4-flash`.
- Native automatic, required, and named tool calls passed. Complex arguments
  including strings, integers, booleans, arrays, and nested objects passed.
- The production-schema probe passed structural, evidence, temporal, finish
  reason, usage, and truncation checks.
- BGE-M3 identity: `bge-m3`; every reviewed dense vector had 1024 dimensions.
- No automatic retry, model fallback, LLM judge, or semantic repair occurred.

The endpoint's exact vLLM version and deployment flags were not observable.
The explicit served alias was added in `f24e483`; generic compatibility probes
were corrected to honor it and validate complex tool arguments in `34a8924`.

## Runtime results

The single-document smoke accepted one span with one primary request, zero
post-extraction LLM calls, one embedding stage, a 2-node/1-edge FinalGraph, and
exact Neo4j parity. A false graph-view warning caused by comparing output-folder
basename with generated run identity was subsequently fixed in `07927f6`; the
viewer now validates identities from run artifacts while corrupted metadata
still fails.

The live grouped batch processed six documents as 2 + 3 + 1:

| Graph | Documents | Nodes | Edges | Result |
| --- | ---: | ---: | ---: | --- |
| A | 2 | 4 | 2 | completed, exact parity |
| B | 3 | 6 | 7 | completed, exact parity |
| C | 1 | 2 | 1 | completed, exact parity |

Each group used its own embedding/resolution matrix and one embedding call.
Graph IDs and Neo4j namespaces were distinct. Evidence prefixes, nodes, edges,
candidate matrices, and resolution results stayed within the owning group.
Document provenance and completion-order ownership checks passed. Across all
three graphs there were zero JSON-only or Neo4j-only nodes/edges, endpoint
mismatches, import errors, and unexpected graph IDs.

## Manual semantic review

For this specific six-document acceptance fixture only:

- expected facts: 11;
- recalled facts: 10;
- unsupported facts: 0;
- false merges: 0;
- missed merges: 0;
- temporal errors: 0;
- provenance errors: 0;
- structural failures: 0.

The omitted fact was a source-supported office-location relation to Almaty. No
prompt, schema, threshold, extraction-policy, or repair change was made; a
single-pass extraction does not guarantee complete recall, and 10/11 is not a
universal model-quality score.

An evidence-backed alias/equivalence edge in graph B became a redundant
self-loop after its endpoint entities merged. The final edge does not carry a
reliable typed equivalence signal sufficient for narrow deterministic
suppression. It therefore remains structurally valid with provenance rather
than being removed by a lexical raw-relation heuristic. Node aliases and the
resolution evidence remain observable.

## Remaining limitations

Semantic output is probabilistic even at temperature zero. Queue-wait and
per-document processing durations remain uninstrumented (`null`). Resume and
automatic recovery are unsupported. Raw local run artifacts can contain source
content and are intentionally ignored. The reviewed evidence supports
compatibility and structural acceptance, not production readiness or guaranteed
factual correctness.
