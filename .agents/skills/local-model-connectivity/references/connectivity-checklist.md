# Connectivity Checklist

- Report required environment variables as present or absent only.
- Confirm configured endpoint class, model, timeout, and provider profile without secrets.
- Record start/end time, duration, HTTP status, exception type, finish reason, and usage when available.
- Distinguish DNS, connection, queue, generation, client timeout, server timeout, and structured-output failure.
- Make network calls only when the task explicitly requires live diagnosis.
