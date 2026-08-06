# Long-Running Work Policy

Record process identity, sanitized logs, artifact paths, stage boundaries, and progress. Reuse a live process and poll it through completion; a tool timeout does not prove the child process ended. Do not start duplicate servers, acceptance runs, or benchmarks.

Persist compact failure evidence. Diagnose local defects, apply the smallest justified fix, run focused checks, and resume the interrupted stage. Do not weaken a gate or introduce a fallback to make progress appear successful.

A blocked handoff states the exact failing stage, evidence, attempts, completed and missing artifacts, whether user input is required, and the next executable action.
