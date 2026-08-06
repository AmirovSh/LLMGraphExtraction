# Contributing

This project extracts evidence-backed fact graphs from formal documents. JSON is the authoritative output; Neo4j is a derived projection.

## Development setup

Create a Python environment, install the repository dependencies, and copy only placeholder environment names from `.env.example`. For local Neo4j, copy `.env.neo4j.example` to ignored `.env.neo4j` and follow the Neo4j section in `README.md`.

Install the tested development environment with:

```powershell
python -m pip install -c constraints.txt -e ".[dev]"
```

Supported installation variants are:

```powershell
python -m pip install -c constraints.txt -e .
python -m pip install -c constraints.txt -e ".[test]"
python -m pip install -c constraints.txt -e ".[docs]"
python -m pip install -c constraints.txt -e ".[dev]"
```

Update compatible dependency bounds and `constraints.txt` together, never by
relying on ambient packages. Run the complete offline suite after every
dependency update.

Use a topic branch based on the intended development branch. Do not rewrite protected release branches or tags.

## Code organization

- `prompts/`: versioned prompt contracts and registry.
- `config/`: strict YAML-backed runtime settings.
- `runtime/`: production stages, validation, resolution, artifacts, and projection.
- `scripts/`: supported command-line entrypoints.
- `tests/`: offline characterization and regression coverage.
- `README.md`: the consolidated public operating guide.

## Testing

Run focused tests while developing, then:

```powershell
python -m pytest
python -m devtools.check_project_conformance --offline
python -m devtools.check_public_golden --offline
git diff --check
```

The project does not currently enforce an external formatter or linter. Follow existing Python style, keep typed boundaries explicit, and require `git diff --check` plus the offline test gates before review.

Prompt/schema meaning, extraction behavior, thresholds, embedding inputs, resolution decisions, stable identifiers, graph semantics, projection semantics, and model-call stages require explicit approval and proportionate acceptance evidence. Keep mechanical refactors behavior-preserving.

Update documentation when public configuration, commands, artifacts, or module ownership changes. Keep reports concise and do not describe experiments as production features.

## Pull-request checklist

- Scope is focused and unrelated changes are excluded.
- Protected semantic invariants are unchanged or explicitly approved.
- Focused and full offline tests pass; required acceptance/parity evidence is attached.
- Configuration rejects unsupported modes before transport.
- Documentation and migration notes are current.
- No credentials, private source data, raw model responses, run outputs, dumps, caches, or generated bulky files are included.
- No private endpoint, local absolute path, customer document, or embedding artifact is included.

Report suspected secret exposure privately; do not open an issue containing credentials. No license has been selected, so the repository currently grants no explicit reuse rights.
