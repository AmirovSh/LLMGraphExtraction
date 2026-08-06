# Model Connection Policy

Model and embedding connections are infrastructure adapters, not semantic fallback mechanisms.

- Load endpoints, models, provider profiles, overrides, and timeouts from validated configuration.
- Keep secret values in environment variables and redact diagnostics.
- Reject unsupported request modes before transport.
- Preserve provider-generic errors, truthful attempt accounting, and completed-unit resume protection.
- Do not add retries, concurrency, model switching, or prompt changes while diagnosing connectivity unless separately approved.
- Never call a model or embedding endpoint during an offline test.

Reports may include presence/absence, sanitized endpoint class, status, timing, exception type, finish reason, usage counts, and schema result. Never include credentials, authorization headers, or raw private payloads.
