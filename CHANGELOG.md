# Changelog

All notable changes. Versions identify the software that wrote a file, not an experiment or an algorithm; protocols and
control algorithms have their own identifiers (see `docs/TECHNICAL.md`).

## [0.2.0] - exploratory research release

### Added
- `research/` research drivers (`run_arms.py`, `arms.py`, `measure_fragments.py`): run any set of frozen window texts plus the
  12 preserved neutral controls through the frozen fold pipeline, twice, with a fresh or compatibility-checked baseline.
- `research/data/`: the five frozen treatment-type texts, the verified 12-control manifest, sanitised traces of the real runs
  (Qwen2.5-1.5B and Qwen2.5-7B) and `results-v0.2.0.json` with hashes.
- New README (findings, limitations, reproduction), `docs/TECHNICAL.md` (the former README), release notes.
- Tests: `tests/test_v020_research.py` (request plans, saved-evidence equations, analysis, mock-server end-to-end).

### Fixed
- The research summary no longer fails for a run that has no `B_original` arm or no controls of its own (a cue-probe-only
  run crashed at its final summary step in the untracked driver used for the original experiments); it falls back to a
  supplied reference range or reports the fields as absent.
- Tracked research data is byte-stable across Windows checkouts (`.gitattributes`), so its recorded SHA-256 values verify in a
  fresh clone; the git-identity test no longer fails in a fresh clone (it checks the repo-local author only when one is set).

### Findings recorded (exploratory; see README)
- config-audit-v2 original B residual +0.857 (Qwen2.5-1.5B) and +0.654 (Qwen2.5-7B), above all 12 neutral controls on both.
- Paraphrase +0.713 / +0.479; polarity-reversed -0.183 / -0.668; neutral-control ranges -0.241 to +0.061 and -0.698 to -0.673.
- Raw argmax stayed A on 1.5B and flipped to B on 7B; repeated measurements bit-identical; cue interactions not strictly additive.

### Unchanged
- The contrast equation, fold, normalisation, thresholds, 15-field trace schema, validity gates, the `neutral-prose/1`
  generator and lexicon, every protocol and preregistration document, and all historical evidence.

## [0.1.6] - robustness amendment (internal)
Disclosed AMENDMENT-001 to `config-audit-robustness/1` (annotation-only review, fixed draw budget, byte-identical regeneration
verification, anchor evidence, provenance binding); `neutral-prose-replicates/1`; robustness analysis module.

## [0.1.5] - config-audit robustness protocol (internal)
Preregistration, replicate-control generator and exploratory analysis; static prompt-identity and fold-recoverability checks.

## [0.1.4] - `semantic-v2/1` (internal)
Five protocol-bound semantic fixtures, per-scenario windows, 12-control-capable manifests and baseline-identity gate.
First exploratory run: INCONCLUSIVE.

## [0.1.3] - `window-repair/1` (internal)
Per-scenario displacement windows; separate identities for protocol, control algorithm and software version.

## [0.1.2] - `neutral-prose/1` (internal)
Deterministic fluent matched neutral controls replace the word-salad filler.

## [0.1.1] - validity gates and provenance (internal)
Explicit validity preconditions, provenance sidecar, choice-surface coverage diagnostic.

## [0.1.0] - initial baseline
Frozen architecture: three-request fold, contrast against a clean baseline, 15-field trace, mock-server test suite.
