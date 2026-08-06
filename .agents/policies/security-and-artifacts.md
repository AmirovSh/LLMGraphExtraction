# Security and Artifact Policy

Secrets belong only in environment variables or ignored local files. Never print, persist, stage, or commit secret values, private endpoints, credentials, authorization headers, or personal paths.

Do not commit:

- local environment or credential files;
- raw model requests and responses;
- production outputs, databases, dumps, logs, or browser state;
- coverage, caches, builds, IDE files, or endpoint probe output;
- local audit and diagnostic reports.

Tracked examples may contain environment-variable names and obvious placeholders only. Public fixtures must contain reviewed non-private data.

Before committing, inspect status, the staged diff, ignored artifacts, and secret-pattern matches. Stage only reviewed source, tests, public policies, skills, templates, and documentation.
