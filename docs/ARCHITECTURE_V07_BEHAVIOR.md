# Architecture notes: behavioral telemetry (v0.7.0)

Scope: how the new layer sits on the v0.6 code, and where each measurement boundary is drawn and why.

## Placement: one reading path, no parallel framework

```
saved hosted session (session.json + records/*.json)     <- written by bleediff.hosted (unchanged)
        |  read only
bleediff/behavior.py   load_evidence()   -> one observation per record, problems listed
        |
bleediff/telemetry.py  observe_record()  -> observation (schema bleediff.behavior-observation/1)
        |       uses bleediff.answers: PARSERS (answer/format/refusal), detect_provider, usage_from_anthropic (cross-check only)
bleediff/behavior.py   summarize_group() / contrast() / build_report() / report_lines()
        |
bleediff behavior ingest | report         <- registered in bleediff/cli.py like `experiment`
```

- **Reused, unchanged:** the record format (`bleediff.hosted-answer/1`), the answer parsers (`answers.PARSERS`) and the usage mapping, the CLI parser pattern (`add_parser`, as `experiment.py`), `errors.Inconclusive` (exit code 2), the offline-replay idea. Nothing in `hosted.py`, `answers.py`, the research harnesses or any saved evidence was modified; `bleediff/cli.py` gains two lines (import and registration).
- **New:** `bleediff/telemetry.py` (schema, states, adapters, one-record extraction), `bleediff/behavior.py` (ingestion, observation files, summaries, contrasts, report, CLI handlers), `tests/test_v070_behavior.py`, `examples/behavior/r002_compat_check.py`, these docs.
- **Why a separate module rather than extending `answers.analyze`:** `analyze` is an answer-only analysis that sums missing thinking tokens as zero (`x or 0`) and requires a prompt dataset. The new layer must never do either, so it reads the raw response itself and leaves `analyze` and every v0.6 output untouched (backward compatibility is tested: the v0.6 reader and the new reader agree on every answer status of the same session).

## The observation (schema `bleediff.behavior-observation/1`)

```
identity   session, name, cell, question, mapping, condition, pass, prompt_sha256, provider, model, model_version
evidence   file (relative), sha256 of the record file, layout (session|legacy), record_schema, http_status
adapter    id, version, provider, validation (what the semantics rest on)
answer_status / stored_status      the v0.6 six-valued status (ok, invalid, blocked, empty, malformed, error) as recomputed from the saved response, and as stored
measures   13 measurements, each {state, value, source, [reason], [reported], [derived_from]}
context    finish_reason, block_reason, output_cap_hit, attempts, cost_basis, thinking_block_present, thinking_corroboration   (not measurements; never aggregated as values)
provider_fields   raw provider fields kept passively where the adapter does not interpret them (UNSUPPORTED)
problems   record-level inconsistency codes (for example stored_usage_disagrees_with_response:thoughtsTokenCount)
```

`validate_observation` enforces the state rule on read-back: OBSERVED needs a value of the right kind; any other state must carry none.

## Measurement boundaries

1. **Raw response is the source of truth.** Token telemetry is read from `response.usage...` and named by path (`source`). The harness's normalised `usage` is only a cross-check: a disagreement makes the affected measures INCONSISTENT and adds a problem code. The raw provider evidence is never rewritten.
2. **Three levels of "no value".** *Not provided by this provider/adapter* (UNSUPPORTED), *provided in principle, absent here* (MISSING), *present but unusable* (INVALID or INCONSISTENT). They answer different questions (is the instrument blind here, did the run lose data, can the data be trusted), so they are never merged and never defaulted to zero.
3. **Observed behavior is not telemetry state.** A reply that is not one letter is an observed `no_answer` (OBSERVED), not a missing answer; an error record has no response, so its answer is MISSING.
4. **Answer, thinking, resources and format are four blocks.** No function combines them; the report and JSON have no composite field, and a test searches the JSON for composite-score and probability-like keys. Cost and latency are shown beside, never inside, the thinking block.
5. **Counter versus block.** The provider counter is the measure; the thinking block is corroboration that is reported, not enforced (v0.7.0 has no evidence that it must always exist).
6. **Per-provider semantics are declared, not assumed.** `Adapter.validation` states what each adapter's semantics rest on and `Adapter.thinking` gates whether thinking measurements exist at all.
7. **Units.** Requests are the rows of evidence; prompts (by hash) and `--unit-field` units are counted next to them; unit-averaged figures weight units equally. Pairing is within a session and a cell, so repeated sessions are never merged into one pair.
8. **Descriptive only.** There is no interval, test, weighting scheme or model; those belong to a design-specific analysis plan (Research 002's frozen analysis is the example).

## Provenance and privacy

Observations hold relative evidence paths (`session_dir/records/file.json`) and SHA-256 digests, no absolute paths and no credentials; the report records each input's digest (over file names and hashes, like `answers.read_session`) and an `observations_digest`. Observation files are written exclusive-create (never overwritten). The module opens no socket and reads no environment variable; a test patches the network and the key variables away and asserts that a report still runs.

## Compatibility

- v0.6.0 commands, record formats and saved evidence are unchanged; the full v0.6 test suite passes unchanged beside the new tests (only the package-version assertions moved to 0.7.0).
- Python 3.11+, standard library only (`ast.parse(..., feature_version=(3, 11))` is part of the tests; no third-party import).
- The legacy pilot layout needs `--prompts` for the scoring key; its latency is UNSUPPORTED because that layout never recorded it.

## Extension points

A new provider adapter = an `Adapter` in `telemetry.ADAPTERS` with a `usage` function returning the five usage measurements (input, cache read/write, output, thinking), a `corroboration` function, a cap `finish_reason`, and an honest `validation` sentence; its answer parsing goes through `answers.PARSERS`. Until validated against saved evidence, return UNSUPPORTED for anything whose semantics are not established.
