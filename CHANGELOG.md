# Changelog

All notable changes. Versions identify the software that wrote a file, not an experiment or an algorithm; protocols and
control algorithms have their own identifiers (see `docs/TECHNICAL.md`).

## Experiment authoring (after 0.5.0; no version bump)

### Added
- `bleediff experiment init|validate` (`bleediff/experiment.py`): an offline authoring step for answer-only experiments. A short spec (questions, named context fragments, explicit letter mappings) is validated and turned into a `bleediff.hosted-prompts/1` dataset that the existing `anthropic`/`openai`/`gemini` commands run unchanged, with an estimated worst-case budget and a demonstration-versus-evidence label. No measurement, schema or existing command changed.
- `docs/FIRST_EXPERIMENT.md` and `tests/test_experiment.py`.
- Per-question conditions for `bleediff.experiment/1`: `conditions` may be a list of names with each question supplying its own `condition_fragments` (every name exactly once, no fallback to a global fragment; mixing it with a `{NAME: fragment}` object is rejected). Global specs and the starter generate byte-identical prompt datasets. `examples/experiment-multi-question/` is a three-question example.

### Fixed
- `experiment validate` applied one global fragment set to every question without saying so; it now warns about that, and its control and length warnings are computed on the fragments each question actually receives and name the question, condition and matched term. The control option-text check matches standalone words or phrases (the `K` of an answer no longer matches the `k` in `bakery`).

## [0.5.0] - hosted answer-only support and the five-model comparison (exploratory)

### Added
- `docs/HOSTED_MODELS.md` (usage guide: DIRECT_PROBABILITY vs ANSWER_ONLY, strict parsing vs secondary diagnostics, provider refusals vs incorrect answers, estimated budget vs provider billing, replay, redaction).
- `examples/hosted-comparison/`: the five-model comparison as a replayable offline bundle (sanitized Claude Haiku 5.5 and Sonnet 5.5 sessions, atlas manifest, `verify_comparison.py`, `SHA256SUMS.txt`, results table and interpretation limits).
- `docs/EVIDENCE_RECONCILIATION.md`, `docs/CASE_NOTE_HOSTED_REFUSALS_AND_FORMAT.md`, `docs/evidence/` (per-request index of 522 raw records, pseudonymous ids) and `research/reconcile_evidence.py`: reconciliation of the exploratory Claude runs v0.10-v0.15 (raw runs stay git-ignored); the post-hoc final-line extraction is documented as a separately labelled secondary rule (FLR-v1, Final-Line Recovery).
- `tests/test_v050_release.py`.

### Fixed
- Saved hosted records no longer contain opaque thinking signatures or reasoning text: every live run scrubs them (`hosted.scrub_private_reasoning`) before writing. The committed Gemini example records had carried `thoughtSignature` values; they are replaced by placeholders, and provider response ids in the Gemini and OpenAI example records are pseudonymized (answers, statuses and usage unchanged; both `SHA256SUMS.txt` files regenerated).
- `bleediff anthropic --help` showed Gemini/OpenAI model examples; each provider now shows its own.
- `examples/hosted-comparison/README.md` no longer says the Claude runs have not been performed.

- Estimated-budget accounting under retries: previously a retry was not admitted against the remaining estimate and failed attempts that were then retried were never charged. Now a retry is sent only if one more worst-case attempt fits, and every failed attempt that may have been billed (transport error, HTTP 5xx) is charged at the worst-case bound (records carry `billable_attempts`; 429/4xx are not charged). The cap remains an estimate, not a guaranteed billing limit. Regression tests added.

### Changed
- Version 0.5.0. The atlas manifest `atlas-input.template.json` became `atlas-input.json` now that the Claude sessions exist in the repository.

## Hosted answer-only runner (included in 0.5.0)

### Added
- `bleediff anthropic` (`bleediff/anthropic_messages.py`): the same ANSWER_ONLY engine for the Anthropic Messages API (`ANTHROPIC_API_KEY`, `x-api-key` + `anthropic-version`; `--effort`, `--thinking`, `--thinking-budget`; thinking blocks never count as the answer; thinking tokens counted once from `output_tokens`; 529 overload is retried within bounds). A real `claude-haiku-5-5` probe with `max_tokens` 8 returned only an empty thinking block, recorded as an empty non-answer. `examples/hosted-comparison/` is the replayable five-model comparison: sanitized Claude Haiku 5.5 and Sonnet 5.5 sessions, the Atlas manifest `atlas-input.json`, runbook, offline `estimate_pilot_cost.py` and `verify_comparison.py`.
- `bleediff openai` (`bleediff/openai_chat.py`): the same ANSWER_ONLY engine for the OpenAI Chat Completions API (`OPENAI_API_KEY`; model-family parameter rules; reasoning tokens counted once), and `examples/openai-answer-only/`, the saved GPT-4.1 mini v0.7 pilot as a replayable bundle (logprobs removed). The shared engine moved to `bleediff/hosted.py`; `bleediff/gemini.py` is now only the Gemini provider record.
- Budget semantics reviewed for both hosted adapters: the cap applies to an estimate from your prices and reported usage (not a provider billing total); the per-request reserve covers the output-token limit and thinking/reasoning tokens; a response exceeding its admitted bound stops the run (`COST_BOUND_EXCEEDED`); non-retryable quota errors are not retried.
- `bleediff gemini` (`bleediff/gemini.py`, `bleediff/answers.py`): ANSWER_ONLY measurements on the Gemini `generateContent` API with frozen prompt datasets, A-D answer
  parsing, answer-letter permutations (correct versus competing semantic answers), repeated passes, response validation, usage and cost recording, an enforced
  `--budget` with explicit list prices (fail closed on missing usage), bounded retries, request/response provenance, and `--replay` of saved evidence offline. The key is
  read from `GEMINI_API_KEY` only and is redacted from every message and record. No probabilities, margins or confidence are ever derived from answers.
- Atlas: an optional `answer_only_sessions` manifest list; `answer_only` block in `atlas.json`, `answer-only.csv`, `answer-level-comparison.csv` and separate report
  sections. Local cells are now labelled `measurement_type: DIRECT_PROBABILITY`; hosted answer-only sessions are `ANSWER_ONLY` and never enter margins, shifts, ranks or paired differences.
- `examples/gemini-answer-only/`: the v0.8 Gemini pilot (2 models x 80 prompts x 2 passes) as a sanitized, hashed bundle that replays offline.
- Tests: `tests/test_gemini_adapter.py` (mock HTTP, secrets, budget, parsing, replay, Atlas separation).

### Unchanged
- All direct-probability calculations, the research fold, published releases and tags.

## [0.4.0] - semantic answer roles and the four-mapping evidence bundle (exploratory)

### Added
- `examples/four-mapping/`: a sanitized evidence bundle (48 saved sessions, frozen texts, candidate files, fixtures, Atlas input manifest,
  `SHA256SUMS.txt`) and `verify_summary.py`, an offline script that recomputes every number in `docs/FOUR_MAPPING_SUMMARY.md` from it.
- `RELEASE_NOTES_v0.4.0.md`.
- `bleediff/roles.py` and `track --arms --correct/--incorrect A-D`: semantic-role scoring for the correct and the competing answers wherever
  their letters are (correct-versus-competing margin, correct-versus-strongest-incorrect margin, movement toward the competing answer,
  overall flips, role-order crossings, newly induced incorrect answers only when the baseline was correct). Batch sessions record
  `semantic_roles` records and the designated correct answer; the Atlas re-derives role scores from the saved probabilities, flags
  cells whose A/B margin is not the role margin (`AB_MARGIN_NOT_ROLE_MARGIN`), adds semantic columns, a quantities legend and a semantic
  roles section.
- `leaderboard.csv` gains five trailing columns (`margin_correct_vs_competing`, `margin_correct_vs_strongest`,
  `newly_induced_incorrect`, `correct_letter`, `competing_letter`); the earlier columns are unchanged.
- `docs/FOUR_MAPPING_SUMMARY.md`: a concise, exploratory summary of the four-mapping counterfactual experiment, reproducible from the bundle.
- `.gitattributes`: `examples/four-mapping/** -text` so the hashed bundle files are byte-identical on every platform.

### Changed
- The batch baseline print says "Margin A/B (letters A and B only; NOT the role margin)" and prints the role margin when the roles are not
  at A and B; the batch summary reports newly induced incorrect answers separately from flips. Output for roles at A and B is otherwise unchanged.
- Atlas wording for the historical `topic` category now says "the competing answer's subject (option B's subject in the original A/B
  experiments)"; the historical definition is preserved.

### Unchanged
- The research fold, all v0.3.0 evidence and the published releases; margins, probabilities and rankings for roles at A and B.

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
