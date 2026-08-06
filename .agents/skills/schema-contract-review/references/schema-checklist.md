# Schema Contract Checklist

- Validate types, required fields, non-empty identifiers, uniqueness, and endpoints.
- Reject unknown or unsupported modes before transport or persistence.
- Preserve canonical deterministic serialization and stable content-derived IDs.
- Redact secrets while retaining reproducible active/static configuration.
- Test invalid input, round trips, backward compatibility, and projection parity.
