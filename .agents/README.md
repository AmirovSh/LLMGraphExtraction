# Agent Guidance

`AGENTS.md` is the concise always-on repository entry point.

- `policies/` contains supporting project policies. Read a policy only when `AGENTS.md`, a skill, or the task scope requires it.
- `skills/` contains task-specific Agent Skills. Load the matching `SKILL.md` only when its description triggers.
- `templates/` contains reusable task and review-report structures.

All files here are public repository content. They must not contain secrets, private URLs, personal absolute paths, local audit reports, runtime outputs, or development history.

Golden validation belongs to prompt/model/schema development and release
regression. Production completion depends only on extraction, validation,
authoritative JSON persistence, configured projection, and structural parity.
