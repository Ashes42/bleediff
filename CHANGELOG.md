# Changelog

All notable changes. Versions identify the software that wrote a file, not an experiment or an algorithm; protocols and
control algorithms have their own identifiers (see `docs/TECHNICAL.md`).

## [0.3.0] - decision tracker, batch influence ranking and influence atlas (exploratory)

### Added
- `bleediff track`: an interactive decision-margin tracker. Edit a UTF-8 fragment file, press Enter, and see the normalized
  A/B/C/D probabilities, the signed `margin_AB = ln(p_A/p_B)`, changes from the baseline and the previous step, and the overall
  preferred answer (a pairwise crossing is not reported as an answer flip). Single-prompt measurement of request #3 of a frozen
  scenario (default `config-audit-v2`); reuses the validated `LlamaClient.choice_probs` extraction unchanged. Session history
  `<out>/history.jsonl` (schema `bleediff.track-history/1`): provenance, hashes, probabilities, margins, changes, errors; prompts
  only with `--retain-prompts`; never overwrites an existing session. Module `bleediff/track.py`.
- `bleediff track --arms FILE` (batch mode): scores a labelled list of candidate fragments against one clean baseline with the same
  single-prompt measurement and writes `history.jsonl`, a compact `leaderboard.csv` and `batch-manifest.json` (hashes, provenance).
  The terminal summary is ranked by observed movement of the A/B margin toward the designated incorrect answer (`--incorrect`,
  default B), shows fragment and prompt token counts so unmatched lengths are visible, and labels results as observed candidate
  effects (no causal claim). Failed requests, invalid probabilities, invalid candidates, ties and empty lists are handled; the run
  stops after repeated unreachable-server failures. Module `bleediff/batch.py`.
- `bleediff atlas`: an influence atlas over existing batch sessions (reads saved results; never contacts a model). Validates the
  sources (schemas, recorded hashes, scenario and model identity, evidence files), re-derives each margin and preferred answer from
  the saved probabilities with the unchanged functions, and writes `atlas.json`, `candidates.csv`, `category-summary.csv`,
  `model-comparison.csv` and `report.md`. Effects are shifts from each model's own baseline; paired category differences are computed
  inside each task/model cell; outcomes distinguish induced errors from already-wrong baselines; A/B crossings are kept apart from
  overall flips; saturated baselines, low coverage, token mismatches and incompatible presentations are flagged; filters by model,
  task, category and outcome; no global ranking. The `topic` category is labelled as on-topic text that also mentions option B's
  subject. Module `bleediff/atlas.py`.
- `examples/atlas/`: three small sanitized example datasets of saved batch sessions from real runs (pilot: 4 tasks x 3 models;
  answer-mention 2x2; slot-swap), their candidate files, fixtures and atlas input manifests, plus `mention_contrasts.py`. They
  reproduce the documented findings offline, without a model.
- Tests: `tests/test_v030_track.py`, `tests/test_v031_batch.py`, `tests/test_v032_atlas.py`, `tests/test_v030_release.py`
  (mock-server tests; example data, documentation and version consistency).
- README for v0.3.0 and `RELEASE_NOTES_v0.3.0.md`.

### Findings recorded (exploratory; see README and the release notes)
- Three local models (Qwen2.5-1.5B, Llama 3.2 3B Instruct, Qwen2.5-7B; Q4_K_M, CPU, raw prompts) across four independent synthetic
  tasks. On-topic text that also mentions option B's subject often moved the decision toward B (11 of 12 cells above off-topic text,
  11 of 12 above neutral prose), but the original fixed-position results were confounded with the position of the subject.
- One newly induced incorrect answer: Llama 3B, sensor task, when B's subject was mentioned without evidence for B. It survived
  swapping the subject positions. Qwen 1.5B showed smaller effects at both positions; Qwen 7B showed strong positional sensitivity.
  Models differ substantially, including on neutral text.
- No universal priming mechanism and no real tenant-isolation vulnerability was established. Deterministic repeats are
  reproducibility checks, not independent samples.

### Changed
- Package version 0.3.0. `.gitattributes` keeps `examples/atlas/**` byte-stable across checkouts so recorded SHA-256 values verify.

### Unchanged
- The research fold, contrast equation, thresholds, trace schema, validity gates, fixtures, protocols and all v0.2.0 results.

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
