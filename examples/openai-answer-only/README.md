# OpenAI answer-only example (saved GPT-4.1 mini evidence from the v0.7 pilot)

A compact, sanitized copy of the real GPT-4.1 mini pilot in the `bleediff.hosted-answer/1` record format, so that `bleediff openai --replay` and the Atlas can reproduce its
tables **offline: no key, no network, no model, no cost**. It is the OpenAI counterpart of `examples/gemini-answer-only/` (the same 80 frozen prompts).

```
python -m bleediff openai --replay examples/openai-answer-only/gpt41mini
python -m bleediff atlas --manifest examples/openai-answer-only/atlas-input.json --out runs/hosted-atlas
```

The Atlas command shows three kinds of evidence in one descriptive report, never merged: local `DIRECT_PROBABILITY` sessions (questions q1-q4 of `examples/four-mapping`, Qwen2.5-1.5B and
Llama 3.2 3B), the two `ANSWER_ONLY` Gemini sessions, and this `ANSWER_ONLY` GPT-4.1 mini session.

## Measurement type: ANSWER_ONLY
The adapter requests no logprobs and derives no probabilities, log-odds, confidence or leaderboard. The original pilot did also request top-20 logprobs and found all four answer letters in only
28 of its 160 responses (partial exposure); those lists were **removed** from this bundle and are not used. Answers come from the returned message.

## Contents
| Path | What it is |
|---|---|
| `prompts.json` | the 80 frozen prompts (byte-identical to the Gemini bundle's) |
| `gpt41mini/` | `session.json` and `records/<cell>__<condition>__pass<k>.json`: request parameters without credentials, the response (message, finish reason, usage; logprobs removed), estimated cost; 80 prompts x 2 passes = 160 records |
| `atlas-input.json` | Atlas manifest: local sessions + the Gemini bundle's two sessions + this session |
| `SHA256SUMS.txt` | SHA-256 of every file except this README |

Model: `gpt-4.1-mini` (snapshot `gpt-4.1-mini-2025-04-14`), chat completions, the frozen prompt as one user message, temperature 1, top_p 1, seed 0, `max_completion_tokens` 1.
Estimated cost of the original pilot: $0.0104 at list prices ($0.40 / $1.60 per 1M tokens) - an estimate from reported usage, not a provider billing total.

## Expected replay (pooled over the 2 passes: 16 cells x 2 = 32 answers per condition)
| condition | answers |
|---|---|
| BASELINE, PROSE, RANDOM_AB, RANDOM_OTHER | 32 correct each; no answer changed |
| COMPETING (explicit competing evidence) | 30 competing, 2 correct (route question under m4, both passes) |

The two passes gave the same answer in 80 of 80 prompts (replicate stability, not confidence). Replay recomputes all of this from the saved records.

## Provenance and limits
Converted from the local, git-ignored pilot directory (not distributed); only records, request parameters, usage and the returned message were kept. One wording per fragment, 16 non-independent cells
(4 questions x 4 mappings), one model, chat presentation (the local runs used raw completion): comparison with the local sessions is descriptive and answer-level only.
