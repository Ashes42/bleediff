# Behavioral telemetry (v0.7 Phase 1)

> Status: **feature branch `feature/behavioral-telemetry-v07`, cut from the v0.6.0 release. Not released; the package version is still 0.6.0.** Everything here runs offline.

Bleediff began as an answer-influence measurement tool. Research 002 showed why that is not enough: a contextual change can sharply change how often a model's provider reports
*thinking* while leaving the *answer* almost unchanged. `bleediff behavior` measures these dimensions **separately**, from evidence you already saved:

| Dimension | Measurements | Block in the report |
|---|---|---|
| Answer selection and correctness | `answer_outcome`, `answer_role` (correct / competing / other_incorrect / no_answer) | ANSWER BEHAVIOR |
| Provider-reported thinking | `thinking_tokens`, `thinking_activation` (= thinking_tokens > 0) | THINKING |
| Resource expenditure | `input_tokens`, `output_tokens`, `non_thinking_output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `latency_s`, `cost_usd` | RESOURCES |
| Format and refusal | `format_compliant`, `refusal` | FORMAT/REFUSAL |

There is **no composite behavioral score**, no probability, no confidence and no inference (no interval or test): the report is descriptive and says so.

## Commands (no key, no network, no model)

```powershell
# summarise a saved session by condition (any session written by `bleediff anthropic|openai|gemini`, or the legacy pilot layout with --prompts)
python -m bleediff behavior report path\to\session
python -m bleediff behavior report path\to\session --baseline BASELINE
python -m bleediff behavior report A_dir B_dir --contrast CONTRA CONSIST --unit-field question --select-unit "^F[0-9]+D$" --json-out report.json

# write the observations (one JSON line per request, schema bleediff.behavior-observation/1) to a NEW file, then report from the file
python -m bleediff behavior ingest path\to\session --out observations.jsonl
python -m bleediff behavior report observations.jsonl

# cost: records normally carry the harness's own estimate; to price EVERY record from reported usage instead (USD per 1M tokens)
python -m bleediff behavior report path\to\session --price-in 0.1 --price-out 0.5 --cost-from-usage

# the legacy v0.8 pilot layout carries no scoring key: supply the prompt dataset
python -m bleediff behavior report runs\v0.8-gemini-pilot-...\responses\g38flash --prompts runs\v0.7-frontier-pilot-...\frozen-requests.json --baseline BASELINE
```

Output files are never overwritten. `--strict` makes an un-ingestable file exit 2 (otherwise it is listed under INGEST PROBLEMS and counted, never silently dropped). Evidence is only read;
each observation records the SHA-256 of the file it came from.

## The five measurement states

Every measurement of every observation is exactly one of these. Only OBSERVED holds a value.

| State | Meaning | Example |
|---|---|---|
| **OBSERVED** | the provider or harness reported a valid value. **Zero is a legitimate observed value.** | `thinking_tokens = 0` with `output_tokens_details.thinking_tokens: 0` in the response |
| **MISSING** | the adapter defines the measurement but the record does not contain it (field absent or null, no response, error record) | a response whose usage has no `output_tokens_details`; an HTTP-error record |
| **UNSUPPORTED** | the adapter does not establish this measurement for this provider or layout; the raw provider field is preserved passively under `provider_fields` | OpenAI `reasoning_tokens`, Gemini `thoughtsTokenCount`, latency in the legacy layout |
| **INCONSISTENT** | values are present but contradict each other or a documented invariant; the conflicting value is kept in `reported` and never aggregated | thinking tokens above output tokens; output tokens above `max_tokens`; a stored usage figure that disagrees with the saved response |
| **INVALID** | the field is present but is not a valid measurement of its kind | `"thinking_tokens": "5"`, `true`, `-3`, `2.5`; a non-finite latency; `usage` that is not an object |

Rules that follow from this:

- **Absent thinking telemetry is never converted to zero.** `thinking_activation` and `non_thinking_output_tokens` take the worst state of their inputs, so a MISSING `thinking_tokens` gives a MISSING `thinking_activation` (value `null`), never `false`.
- **Derived measurements say so**: `source: "derived: thinking_tokens > 0"` and `derived_from`.
- **Denominators are explicit.** `activation rate = activated / OBSERVED`; records in any other state are counted as "excluded" and shown per measurement in `NOT-OBSERVED STATES`. A rate over zero observed records is `null`, not 0.
- **Corroboration is separate from measurement.** The counter is the measure. Whether the saved reply also contains a `thinking` block is reported as `thinking_corroboration` (`agrees`, `counter_positive_no_thinking_block`, `counter_zero_with_thinking_block`, `not_assessable`); a disagreement is surfaced, not used to exclude a record, because Phase 1 has not established that a block must exist for every positive counter.

## What each measurement is, and is not

- `thinking_tokens`: the provider-reported `usage.output_tokens_details.thinking_tokens`, a **subset** of `output_tokens` (Anthropic). It is a provider-reported counter, **not** a measure of reasoning quality, effort, or hidden-reasoning content (the content is not stored). Token counts of requests that hit the output cap are right-censored; `output_cap` counts them.
- `thinking_activation`: `thinking_tokens > 0`. Research 002's outcome T is the same definition.
- `input_tokens`: the provider's `usage.input_tokens` (uncached input); cache reads/writes are separate measurements and are never folded in. (The v0.6 usage mapping adds them for budget bounds; telemetry does not.)
- `latency_s`: the harness's `elapsed_s`, the client-measured wall time of the **whole request, including retries and waits** when `attempts > 1` (both the v0.6 hosted engine and the Research 002 harness start the clock before the first attempt). It is not model inference time. Where the layout never recorded it (legacy pilot files) it is UNSUPPORTED; where it should exist and does not it is MISSING.
- `cost_usd`: an **estimate** from reported usage and list prices (or the harness's recorded estimate), **not a provider bill**. A recorded cost that is the harness's worst-case **upper bound** because usage was missing is MISSING (the bound is kept in `reported`). A recorded cost that includes a worst-case charge for a failed attempt is OBSERVED and flagged `includes_worst_case_charge_for_failed_attempts`; `--cost-from-usage` prices every record from usage instead.
- `answer_outcome`: the letter A-D the model returned, or `no_answer` when the reply was not exactly one letter (format-invalid, empty, or refused); `no_answer` is an observed behavior, not missing telemetry. `format_compliant` is true only when the reply was exactly one letter. `refusal` is true when the provider reported a refusal or block; a refusal is not a format failure.
- `answer_role` needs the scoring key (`correct_letter`/`competing_letter` in the record, or `--prompts`); without it the role is MISSING and the record is "unscored", counted separately.

## Adapters: only where the semantics are established

| Adapter | Answer / format / refusal | Input tokens | Output tokens | Thinking telemetry | Validation |
|---|---|---|---|---|---|
| `anthropic-messages` v1 | yes (v0.6 parser) | yes | yes | **yes** | validated against the archived Research 002 evidence (2,560 Messages-API records) |
| `openai-chat` v1 | yes (v0.6 parser) | yes | yes (`completion_tokens`, billed total) | **UNSUPPORTED** | answers and usage only; reasoning semantics not established in Phase 1 |
| `gemini-generate` v1 | yes (v0.6 parser) | yes | **UNSUPPORTED** (candidate/thought split not established) | **UNSUPPORTED** | answers and input only |
| any other provider | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | no adapter |

Providers do not expose equivalent reasoning measurements, and Phase 1 does not assume they do. Adding a provider means writing an adapter that states its semantics and validating it against saved evidence.

## Condition summaries, units and paired contrasts

- **Units.** Repeated requests of one prompt (passes, sessions) are **repeated measurements, not independent units.** Each condition reports its requests, its **distinct prompts** (by prompt hash) and its distinct `--unit-field` units (default `question`). The pooled-request figure sits next to the **unit-averaged** figure (mean over units of each unit's own value; `units_without_observed` tells you how many units had nothing observed).
- **Scopes.** With several sessions you get a `pooled` scope and one scope per session.
- **Pairing.** A contrast A - B pairs records sharing **(session, cell, pass)**: the same prompt template and mapping in the same session, differing in the condition. Pairs never cross sessions; conditions sharing no such key print `NO PAIRS` and are not contrasted. A pair counts for a measurement only if **both** records are OBSERVED for it; the excluded pairs are listed by state pair (for example `CONTRA:MISSING / CONSIST:OBSERVED`).
- **What a contrast reports**: pooled and unit-averaged mean difference, sign counts, the 2x2 discordance table for boolean measures, and (for answers) how often the two records returned the same or a different outcome. **No interval or test.**

## Research 002 compatibility demonstration

The published Research 002 evidence is the real-world validation fixture. The generic layer, with no Research 002 specific code, reproduces the published condition-level and paired figures:

```powershell
python examples\behavior\r002_compat_check.py C:\path\to\bleediff-research-002-publication
```

It ingests all 2,560 records (0 problems; every measurement of every record OBSERVED; all 290 positive thinking counters corroborated by a thinking block), selects the 24 derivation families (`--select-unit "^F[0-9]+D$"`) and prints, per condition, the family-averaged thinking activation, correct rate, format failure rate, thinking tokens and cost next to the figures in the frozen `analysis_results.json` (all match), and the paired CONTRA - CONSIST contrast over 384 pairs: thinking activation +0.46875 (180 pairs CONTRA-only, 0 CONSIST-only, 25 both, 179 neither), correct -0.0234375, matching the published values. The published package is only read, and the tests assert that it is unchanged afterwards. The published study's classification and numbers are **not** recomputed or altered, and the demonstration makes no claim beyond them.

One reconciliation worth knowing: the harness's recorded cost for the two requests that were retried includes a worst-case charge for the failed attempt (about $0.0006 each), while the published analysis prices every request from usage. `cost_usd` therefore differs for those records unless you pass `--cost-from-usage` (the demonstration does, with the Research 002 list prices of $0.10 and $0.50 per 1M tokens, which reproduce every usage-based cost exactly). The test suite asserts that the whole difference is exactly that charge.

Other archived evidence exercised (read only): the Haiku 5.5 and Sonnet 5.5 answer-only pilots (`runs/v0.10`, `v0.11`; session layout, thinking telemetry OBSERVED) and the Gemini pilot (`runs/v0.8`; legacy layout, needs `--prompts`; there thinking and output tokens are UNSUPPORTED and latency is UNSUPPORTED, while answers, format and input tokens are OBSERVED). In the Sonnet 5.5 pilot's COMPETING condition every reply is format-invalid, thinking telemetry is 0, and yet output tokens average about 289 per request: visible text, not provider-reported thinking, carried the expenditure, which is exactly why the dimensions are kept apart.

## Remaining limitations (Phase 1)

- **Anthropic only for thinking.** OpenAI and Gemini reasoning telemetry, and Gemini output-token totals, are UNSUPPORTED until their semantics are validated against saved evidence.
- **No inference.** Contrasts are descriptive. Choosing a unit, an estimator and an interval for a given design is the analyst's job (Research 002's frozen analysis is the worked example and is unchanged).
- **Offline only.** There is no live collection of the new telemetry: the layer reads records the v0.6 harness already wrote. Latency exists only where the harness recorded `elapsed_s`.
- **The thinking block is corroboration, not a rule.** Whether a positive counter must always come with a `thinking` block is not established.
- **Cost is an estimate** from list prices; cache pricing is not modelled (a record with cache activity cannot be priced by `--cost-from-usage` and is MISSING for cost).
- **Latency includes retries**; the two are not separated per attempt in the record.
- **No streaming/time-to-first-token, no per-attempt telemetry, no multi-turn.**
- Python 3.11 compatibility is checked by syntax (`ast` feature version) and by the CI matrix; it was not executed under 3.11 on this machine.

Schema and boundaries are described in [`ARCHITECTURE_V07_BEHAVIOR.md`](ARCHITECTURE_V07_BEHAVIOR.md).
