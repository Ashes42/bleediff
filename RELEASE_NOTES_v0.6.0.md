# Bleediff v0.6.0 release notes

**Status: exploratory research release. Not a security finding.** v0.6.0 adds tooling to *author* and *report* small answer-only experiments. It is not a security scanner, and nothing in it demonstrates a jailbreak or a vulnerability.
**The new authoring workflow (`bleediff experiment`) has been exercised offline only** (unit tests with mocked providers, and offline validation of the example specs). It has not been validated through a live provider experiment, and no new measurements or model results are published in this release.

## What is new in v0.6.0 (since v0.5.0)

* **`bleediff experiment init` and `bleediff experiment validate`** (offline: no model, key or network). A short `bleediff.experiment/1` spec (questions, named context fragments, explicit letter mappings) is validated and turned into a
  `bleediff.hosted-prompts/1` dataset, which is re-loaded with the same loader the live runners use, so a file that passes is accepted by `bleediff anthropic|openai|gemini`. Existing output files are never overwritten.
  Guide: [`docs/FIRST_EXPERIMENT.md`](docs/FIRST_EXPERIMENT.md).
* **Provider-aware budget estimation.** With `--provider`, `--price-in` and `--price-out`, `validate` prints four estimated figures (one successful attempt each; budget to admit every request; the same if replies fill over half the reserve; the same if
  every request needs retries) computed with the hosted engine's own reserve and admission rules, which differ per provider. They are estimates from the prices you supply, not a provider bill or a guarantee that a run completes. Bleediff has no built-in prices.
* **Per-question conditions (backward compatible).** In `bleediff.experiment/1`, `conditions` may now be a list of names with each question supplying its own `condition_fragments`, so each question gets its own treatment and neutral control.
  Every question must define exactly the listed names; missing, extra, duplicate, empty or ambiguous definitions are errors, and there is no fallback to a global fragment. This fixes a usability-test finding: a global competing claim was sent to every question
  of a multi-question spec, including unrelated ones, and validation said nothing. **Global `{NAME: fragment}` specs and the starter are unchanged and generate byte-identical prompt datasets** (pinned in the tests); with several questions they now get a warning.
* **Multi-question example and safeguards.** `examples/experiment-multi-question/` (Canberra/Sydney, Pacific/Atlantic, Na/K: 3 questions x 4 mappings x 3 conditions = 36 prompts; public, synthetic facts). `validate` checks control and length heuristics on the fragments each
  question actually receives, names the question, condition and matched term, and matches option text as whole words or phrases (an answer `K` no longer matches the `k` in `bakery`). These are heuristics: they cannot establish neutrality or relevance,
  a missing control is a warning (exploratory specs are allowed), and `validate` labels any spec under 30 questions a DEMONSTRATION (an illustrative convention, not a power analysis).
* **Hosted answer diagnostics and FLR-v2.** `bleediff <provider> --replay`, `answers.analyze` and the Atlas answer-only tables now split the strict non-answers into explicit refusals, format-invalid replies that end in one letter, and ambiguous or missing replies, and
  print an optional, labelled **FLR-v2** diagnostic (last non-empty line, markdown emphasis stripped, exactly one letter A-D; conflicting standalone letters pick nothing). **Strict classification is unchanged and remains the primary result**; FLR-v2 never changes a strict count,
  never applies to refusals, and gives no probability or confidence. FLR-v1 (`research/reconcile_evidence.py`, v0.5.0) is unchanged; on the saved Haiku and Sonnet sessions v1 and v2 extract the same letters.
* **Documentation and usability.** An install section with per-platform virtual-environment commands, reproducibility tiers, a strict-versus-FLR worked example, the actual prompt template, what `--passes` repeats (replicates of the same prompt, not independent samples),
  and clearer budget wording. A GitHub Actions workflow runs the unittest suite on Ubuntu/Python 3.11 and Windows/Python 3.14.

## What already existed (unchanged by this release)
`track`, `track --arms`, `atlas`, the v0.2.0 fold (`calibrate`, `bounce`, `diff`, `inspect`), semantic answer-role scoring and four-mapping evidence (v0.4.0), and the hosted adapters, estimated spend cap, offline replay, redacted evidence and five-model comparison (v0.5.0) are as released.
Historical research results, archived responses and bundled datasets are unchanged. See [`RELEASE_NOTES_v0.5.0.md`](RELEASE_NOTES_v0.5.0.md) and earlier notes.

## Compatibility
Additive. v0.5.0 sessions, datasets and manifests load unchanged, and the hosted runners, scoring and Atlas schemas were not redefined. Output that grew: the answer-only Atlas CSV and `report.md` gain no-answer-split and FLR-v2 columns/sections, and the offline
analysis carries the new counters; readers that expect an exact column list for those CSVs should use column names. Internally, the hosted engine's output-reserve and admission arithmetic and its parameter ranges moved into named helpers/constants so that `experiment validate` uses the
runner's own rules; behaviour is unchanged.

## Verify offline
```
python -m unittest discover -s tests
python -m bleediff experiment validate examples/experiment-multi-question/experiment.json --write-prompts runs/mq-prompts.json --provider anthropic --price-in 1 --price-out 5
python -m bleediff anthropic --replay examples/hosted-comparison/haiku55
```
(`--write-prompts` needs an existing `runs/` directory; the price figures above are placeholders.)

## Limits
A spec that validates shows only that its prompts are well-formed, not that any context caused any change. Mappings re-letter the same question and `--passes` repeats the same prompt, so neither adds independent samples. Hosted answer-only results carry no probabilities, margins or confidence.
Refusal categories are provider metadata. Costs are estimates, not billing totals.
