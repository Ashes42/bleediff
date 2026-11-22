# Bleediff v0.5.0 release notes

**Status: exploratory research release. Not a security finding.** v0.5.0 adds hosted-model support and an offline-replayable five-model comparison. It is not a security scanner, and nothing in it demonstrates
a jailbreak or a vulnerability.

## What is new

* **Three hosted adapters**: `bleediff gemini`, `bleediff openai`, `bleediff anthropic` on one shared engine. They measure models that expose no usable A-D probabilities as **`ANSWER_ONLY`**: a frozen prompt goes in as one
  user message and the reply is classified (correct / competing / other incorrect / non-answer). No probability, margin, confidence or ranking is ever derived. Local runs stay **`DIRECT_PROBABILITY`**, and the Atlas
  reports the two apart (`answer_only_sessions` manifest list, separate sections and CSVs).
* **Estimated spend cap** (`--budget` with explicit `--price-in/--price-out`): a worst-case check before every request and a stop when a response exceeds its admitted bound. It caps an **estimate**, not the provider's bill.
* **Offline replay** (`--replay SESSION_DIR`): re-validates and re-analyses saved sessions with no key, network or model.
* **Redacted evidence**: saved responses are scrubbed of opaque thinking signatures and reasoning text at write time; keys are never recorded.
* **`examples/hosted-comparison/`**: Claude Haiku 5.5, Claude Sonnet 5.5, Gemini 3.8 Flash, Gemini 4 Carbon and GPT-4.1 mini on the same 80 frozen prompts x 2 passes, a results table, an Atlas manifest and
  `verify_comparison.py` (replays everything and scans the bundle for secrets, signatures, reasoning text and local identifiers).
* **Evidence notes**: `docs/HOSTED_MODELS.md` (usage guide), `docs/EVIDENCE_RECONCILIATION.md` (522 unique raw request records reconciled; strict classifications preserved; the post-hoc final-line rule documented as a
  separate secondary rule FLR-v1 (Final-Line Recovery)), `docs/CASE_NOTE_HOSTED_REFUSALS_AND_FORMAT.md`, and a per-request index in `docs/evidence/`.

## What the comparison shows (strict classification; 32 requests per condition)
Under COMPETING evidence the models mostly followed the competing letter (correct/competing: Haiku 9/23, Carbon 4/28, GPT-4.1 mini 2/30, Flash 0/32; Sonnet had no valid answers). The more interesting differences are in
**non-answers**: Haiku refused 12 RANDOM_OTHER requests (provider category `bio`, a provider label, not a finding about cause) that Sonnet answered; Sonnet broke the one-letter format in 60 of 160 requests (prose ending in a
letter). A single-swap contrast in one question context did not reproduce in a second. See the case note for what remains unknown.

## Fixed
* Live runs and the committed Gemini example records no longer hold opaque thinking signatures (`thoughtSignature` values in 320 example records are replaced by placeholders; provider response ids in the Gemini and OpenAI
  example records are pseudonymized; answers, statuses and usage are unchanged).
* `bleediff anthropic --help` shows an Anthropic model id as the example.
* Estimated-budget accounting under retries: each retry must fit under the remaining estimate as one more worst-case attempt, and every failed attempt that may have been billed (network error, HTTP 5xx) is charged at the worst-case bound. Previously retries were not admitted against the budget and failed-then-retried attempts were not charged. The cap is still an estimate, not a guaranteed billing limit.

## Compatibility
Additive. The v0.2.0 fold, `track`, `track --arms` and Atlas results for local sessions are unchanged; Atlas cells now carry `measurement_type: DIRECT_PROBABILITY`. v0.4.0 sessions and manifests load unchanged. The atlas
manifest template in `examples/hosted-comparison/` was renamed `atlas-input.json`.

## Reproduce offline
```
python -m unittest discover -s tests
python -m bleediff anthropic --replay examples/hosted-comparison/haiku55
python -m bleediff atlas --manifest examples/hosted-comparison/atlas-input.json --out runs/hosted-comparison
python examples/hosted-comparison/verify_comparison.py
```

## Limits
Answer-level data from one prompt family: 4 questions x 4 mappings (16 non-independent cells), one filler string, one wording, two passes (replicates, not independent samples). Hosted chat presentation, hidden reasoning,
defaults and refusal behaviour differ per provider and can change over time, so differences are not attributable to capability. Refusal categories are provider metadata. Costs are estimates, not billing totals.
