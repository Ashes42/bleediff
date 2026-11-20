Preserved numerical evidence from the v0.1.0 live smoke test (llama.cpp b11963, Qwen2.5-1.5B-Instruct Q4_K_M,
5 trials, offset 24 words). Used only by regression tests; no inference is ever run from here.

- `traces.jsonl`: byte-for-byte copy of the original trace (15-field schema, unchanged).
- `baseline.sanitized.json`: the original baseline minus machine-specific path fields (see `_derivation`).

The originals remain in the (git-ignored) `runs/` directory and were not modified.
