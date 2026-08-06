# LLM Pipeline Boundaries

- Resolve prompt identity and schema before transport.
- Reject unsupported configuration before any request.
- Record every HTTP attempt, timeout, failure, success, usage count, and manual resume.
- Accept only the configured structured-output contract.
- Never add semantic retries, gleaning, merge verification, or post-extraction repair.
- Keep endpoint-specific compatibility in validated provider configuration.
