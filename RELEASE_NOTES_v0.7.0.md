# Bleediff v0.7.0 release notes

**Status: exploratory research release. Not a security finding.** v0.7.0 adds **offline behavioral telemetry**: it reads hosted sessions you have already saved and reports what the model did and what it cost, one dimension at a time. It is not a security scanner, and nothing in it demonstrates a jailbreak or a vulnerability.

**Published evidence.** The Research 002 study that motivated this release is public: **<https://github.com/Ashes42/bleediff-research-002>** (release v1.0.0: frozen protocol, sanitized run records, report, errata, provenance and offline reproduction commands). v0.7.0 reproduces that study's published condition-level and paired figures from its published records (see "Research 002 compatibility demonstration").

**The new telemetry layer has been exercised offline only**: unit tests, synthetic edge cases, and archived evidence from earlier live runs. v0.7.0 itself makes no request to any provider, needs no key, and has not been validated through a new live provider experiment.

## What is new in v0.7.0 (since v0.6.0)

* **`bleediff behavior ingest` and `bleediff behavior report`** (offline: no key, no network, no model). They read a saved hosted session (`session.json` + `records/`), the legacy pilot layout (with `--prompts` for the scoring key) or an observations file, and never modify the evidence: every observation records the SHA-256 of the file it came from, and files that cannot be ingested are listed, never silently dropped. Output files are never overwritten.
* **13 measurements, kept separate.** Every request becomes one observation (schema `bleediff.behavior-observation/1`) with these measurements, grouped in four report blocks that are never combined:

  | Dimension | Measurements |
  |---|---|
  | Answer behavior | `answer_outcome`, `answer_role` (correct / competing / other_incorrect / no_answer) |
  | Provider-reported thinking | `thinking_tokens`, `thinking_activation` (thinking tokens > 0) |
  | Resource expenditure | `input_tokens`, `cache_read_tokens`, `cache_write_tokens`, `output_tokens`, `non_thinking_output_tokens`, `latency_s`, `cost_usd` (an estimate, not a provider bill) |
  | Format and refusal | `format_compliant`, `refusal` |

* **Five measurement-validity states.** Each measurement is exactly one of **OBSERVED** (the provider or harness reported a valid value; **zero is a legitimate observed value**), **MISSING** (defined for this provider but absent from the record), **UNSUPPORTED** (this adapter does not establish it), **INCONSISTENT** (values contradict each other or a documented invariant) and **INVALID** (present but not a valid measurement). **Absent thinking telemetry is never converted to zero**: a missing `thinking_tokens` gives a missing `thinking_activation`, rates use observed records as the denominator and show how many were excluded, and inconsistent or invalid values never enter an aggregate.
* **Adapters only where the semantics are established.** The Anthropic adapter is validated against the published Research 002 records (all 2,560 are ingested with every measurement OBSERVED). The OpenAI and Gemini adapters cover answers, format, refusal and input tokens; their reasoning/thinking telemetry (and Gemini output totals) is UNSUPPORTED, because providers do not expose equivalent reasoning measurements. Other providers have no adapter and everything provider-specific is UNSUPPORTED.
* **Condition-level summaries and explicitly paired contrasts, without inference.** Repeated requests of one prompt are repeated measurements, not independent units: each condition reports its requests, distinct prompts and units, and pooled figures sit beside unit-averaged ones. Contrasts pair records that share (session, cell, pass), never across sessions, and use a pair for a measurement only when both records are OBSERVED. There is **no composite behavioral score, no interval, no test and no significance statement.**
* **`--cost-from-usage`** prices every record from reported usage and the prices you supply, because the harness's recorded cost can include a worst-case charge for failed attempts.

Guide: [`docs/BEHAVIORAL_TELEMETRY.md`](docs/BEHAVIORAL_TELEMETRY.md). Measurement boundaries: [`docs/ARCHITECTURE_V07_BEHAVIOR.md`](docs/ARCHITECTURE_V07_BEHAVIOR.md).

## Research 002 compatibility demonstration

`python examples/behavior/r002_compat_check.py PATH_TO_bleediff-research-002-publication` reads the published records and the frozen analysis output of <https://github.com/Ashes42/bleediff-research-002> (only read; nothing there is changed or recomputed) and compares, with no Research 002 specific code, the layer's per-condition family-averaged thinking activation, correct rate, format-failure rate, thinking tokens and cost, and the paired CONTRA - CONSIST contrast over 384 pairs, with the published figures. All match. The published study's classification and numbers are unchanged, and the demonstration claims nothing beyond them. One reconciliation is documented: the harness's recorded cost for the two retried requests includes a worst-case charge for the failed attempt (about $0.0006 each), while the published analysis prices from usage; `--cost-from-usage` reproduces the published costs and a test asserts that the whole difference is exactly that charge.

## What already existed (unchanged by this release)
`track`, `track --arms`, `atlas`, the v0.2.0 fold (`calibrate`, `bounce`, `diff`, `inspect`), semantic answer-role scoring and four-mapping evidence (v0.4.0), the hosted adapters, estimated spend cap, offline replay and Atlas (v0.5.0), and experiment authoring (`bleediff experiment init|validate`, v0.6.0) are unchanged. Historical research results, archived responses and bundled datasets are unchanged. See [`RELEASE_NOTES_v0.6.0.md`](RELEASE_NOTES_v0.6.0.md) and earlier notes.

## Compatibility
Additive. v0.6.0 sessions, datasets and manifests load unchanged; no existing command, record format or schema was redefined, and `bleediff/cli.py` only gains the `behavior` command. The only changes to existing tests are their package-version assertions. Python 3.11 or newer and the standard library only: **zero third-party dependencies**, as before.

## Limitations
* Provider-reported thinking telemetry exists only for the Anthropic adapter; OpenAI and Gemini reasoning telemetry is UNSUPPORTED.
* Contrasts are descriptive. Choosing a unit-level estimator, an interval and a decision rule is the analyst's job (Research 002's frozen analysis is the worked example and is unchanged).
* The layer reads records the v0.6 harness already wrote; it does not collect anything live. `latency_s` is the whole-request wall time recorded by the harness (it includes retries) and exists only where recorded.
* A `thinking` block in the reply is reported as corroboration of the thinking counter, not used as a rule, because that is not established.
* `cost_usd` is an estimate from list prices; cache pricing is not modelled.
* Python 3.11 compatibility is covered by syntax checks and the CI matrix; the local verification for this release ran on Python 3.14.
