# Text Processing Policy

Text processing may normalize representation but must not infer meaning.

Allowed deterministic operations include Unicode normalization, line-ending normalization, configured whitespace cleanup, deterministic unit/span splitting, and structural validation.

Do not use:

- regexes to infer relations, entities, aliases, polarity, or causality;
- domain keyword maps or lexical tables to create graph meaning;
- language-specific stemming, lemmatization, transliteration, or synonym expansion;
- substring similarity as proof of entity equivalence;
- destructive source rewriting or sample-specific corrections.

Preserve original source text and evidence offsets. If a representation change could alter identity, relation meaning, negation, version, quantity, condition, or temporal scope, treat it as semantic and require regression plus clean end-to-end evidence.
