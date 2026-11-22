# Gemini answer-only example (saved evidence from the v0.8 pilot)

A compact, sanitized copy of the real Gemini pilot, in the `bleediff.hosted-answer/1` record format, so that `bleediff gemini --replay` and the Atlas
can reproduce its tables **offline: no key, no network, no model, no cost**.

```
python -m bleediff gemini --replay examples/gemini-answer-only/flash
python -m bleediff gemini --replay examples/gemini-answer-only/carbon
python -m bleediff atlas --manifest examples/gemini-answer-only/atlas-input.json --out runs/gemini-atlas
```

## Measurement type: ANSWER_ONLY
Every tested Gemini model rejected `responseLogprobs` ("Logprobs is not enabled for this model"), so these sessions contain **returned answers only**. Bleediff never
derives probabilities, log-odds margins, confidence or a leaderboard from them. The Atlas reports them in separate `ANSWER_ONLY` sections and CSVs; the local
sessions in the same report stay `DIRECT_PROBABILITY`.

## Contents
| Path | What it is |
|---|---|
| `prompts.json` | the 80 frozen prompts: 4 synthetic questions x 4 answer-letter mappings x BASELINE, PROSE, RANDOM_AB, RANDOM_OTHER, COMPETING (correct and competing letter named per prompt) |
| `flash/`, `carbon/` | one session each: `session.json` and `records/<cell>__<condition>__pass<k>.json` (request configuration without credentials, full response, usage, estimated cost); 80 prompts x 2 passes = 160 records per model |
| `atlas-input.json` | Atlas manifest: the two hosted sessions next to the local `DIRECT_PROBABILITY` sessions of the same questions q1-q4 and four mappings (`../four-mapping`) |
| `SHA256SUMS.txt` | SHA-256 of every file except this README |

Models: `gemini-3.8-flash` (thinking level low) and `gemini-4-carbon` (thinking level low), temperature 1, seed 0, chat-style single user message (the local experiments used raw
completion). Estimated cost of the original pilot: $0.0898 and $0.0696 at list prices.

## Expected replay (pooled over the 2 passes: 16 cells x 2 = 32 answers per condition)
| condition | gemini-3.8-flash | gemini-4-carbon |
|---|---|---|
| BASELINE, PROSE, RANDOM_AB, RANDOM_OTHER | 32 correct each | 32 correct each |
| COMPETING (explicit competing evidence) | 32 competing | 28 competing, 4 correct (league question under m1 and m2) |

The two passes gave the same answer in 80 of 80 prompts for both models (replicate stability, not confidence).

## Provenance and limits
The records were converted from the local, git-ignored pilot directory (the original files are not distributed); only records, request configuration and usage were kept.
One wording per fragment, 16 non-independent cells (4 questions x 4 mappings), two models, hosted decoding and hidden reasoning differ from the local raw-completion runs: the
comparison with the local sessions is descriptive and answer-level only.
