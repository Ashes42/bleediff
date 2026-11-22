# Hosted answer-only comparison: Claude Haiku, Claude Sonnet, Gemini Flash, Gemini Carbon, GPT-4.1 mini

A descriptive comparison of five hosted models on the **same 80 frozen prompts** (4 synthetic questions x 4 answer-letter mappings x BASELINE, PROSE, RANDOM_AB, RANDOM_OTHER, COMPETING;
`../gemini-answer-only/prompts.json`, sha256 `a8144a5a...`, byte-identical to `../openai-answer-only/prompts.json` and to the `prompts.json` in each session here), 2 passes each, saved as
sessions you can **replay completely offline: no key, no network, no model, no cost**.

All five are **ANSWER_ONLY**: none of these APIs gave usable A-D probabilities (the Anthropic Messages API has no logprobs parameter), so nothing below is a probability, margin, confidence or
leaderboard. The local Qwen / Llama sessions in the same Atlas stay **DIRECT_PROBABILITY** and are never merged with these.

## Replay (offline)
```
python -m bleediff anthropic --replay examples/hosted-comparison/haiku55
python -m bleediff anthropic --replay examples/hosted-comparison/sonnet55
python -m bleediff gemini    --replay examples/gemini-answer-only/flash
python -m bleediff gemini    --replay examples/gemini-answer-only/carbon
python -m bleediff openai    --replay examples/openai-answer-only/gpt41mini
python -m bleediff atlas --manifest examples/hosted-comparison/atlas-input.json --out runs/hosted-comparison     # all five next to the local sessions
python examples/hosted-comparison/verify_comparison.py                                                           # checks every table below and the public-safety rules
```
Replay recomputes the tables from the saved records and prints `OFFLINE REPLAY ... (no key used, no network, no model contacted)`.

## Model configurations
| Model | Session | Settings | Price used for the estimate ($/M in / out) | Estimated cost |
|---|---|---|---|---|
| `claude-haiku-5-5` | `haiku55/` | Messages API, `max_tokens` 1024, `output_config.effort` low, API-default adaptive thinking and temperature | 0.10 / 0.50 | $0.0128 |
| `claude-sonnet-5-5` | `sonnet55/` | same | 2.00 / 10.00 | $0.2059 |
| `gemini-3.8-flash` | `../gemini-answer-only/flash/` | `maxOutputTokens` 512, temperature 1, top_p 1, seed 0, thinking level low | 0.75 / 3.75 | $0.0898 |
| `gemini-4-carbon` | `../gemini-answer-only/carbon/` | same | 2.00 / 10.00 | $0.0696 |
| `gpt-4.1-mini` | `../openai-answer-only/gpt41mini/` | Chat Completions, `max_completion_tokens` 1, temperature 1, top_p 1, seed 0 | 0.40 / 1.60 | $0.0104 |

Every prompt is one user message (no system prompt). Costs are **estimates** from your supplied list prices and the usage each provider reported; they are not provider billing totals.
The Claude sessions were made with `bleediff` at commit `34636ab` (clean tree); the Gemini and OpenAI sessions were converted from the v0.7/v0.8 pilot scripts.

## Frozen prompts and strict classification
Each prompt names the correct and the competing letter. A reply counts as an **answer** only if it is exactly one letter A-D (the prompt says "Reply with exactly one letter"). Everything else is a
**non-answer** and is kept separate from incorrect answers:
- `refused`: the provider returned `stop_reason: refusal` (a provider decision, not a wrong letter);
- `invalid`: a reply that is not exactly one letter (for example prose ending in a letter, or `**B**`);
- `empty`: no text.

## Results (both passes pooled: 16 cells x 2 = 32 requests per condition; STRICT primary classification)
Cells are correct / competing / other incorrect / no answer.

| condition | Haiku 5.5 | Sonnet 5.5 | Gemini 3.8 Flash | Gemini 4 Carbon | GPT-4.1 mini |
|---|---|---|---|---|---|
| BASELINE | 32/0/0/0 | 32/0/0/0 | 32/0/0/0 | 32/0/0/0 | 32/0/0/0 |
| PROSE | 31/0/0/1 | 32/0/0/0 | 32/0/0/0 | 32/0/0/0 | 32/0/0/0 |
| RANDOM_AB | 32/0/0/0 | 20/0/0/12 | 32/0/0/0 | 32/0/0/0 | 32/0/0/0 |
| RANDOM_OTHER | 20/0/0/12 | 16/0/0/16 | 32/0/0/0 | 32/0/0/0 | 32/0/0/0 |
| COMPETING | 9/23/0/0 | 0/0/0/32 | 0/32/0/0 | 4/28/0/0 | 2/30/0/0 |
| **non-answers (of 160)** | **12 refused, 1 invalid** | **60 invalid** | 0 | 0 | 0 |

The non-answers are different things:
- **Haiku's 12 refusals** are all RANDOM_OTHER, `stop_reason: refusal`, provider category `bio`, 0 output tokens. The same prompts did not refuse for Sonnet (0 of 160). The category label is the provider's; this data does not show what triggers it. Its 1 invalid is `**B**` (the correct letter, markdown-wrapped).
- **Sonnet's 60 invalid replies** are visible prose that ends in a letter (all 32 COMPETING, 16 RANDOM_OTHER, 12 RANDOM_AB). Under the strict parse Sonnet's COMPETING row has no answers, which is **not** "Sonnet resisted the competing evidence".
- Incorrect answers are only ever the competing letter in COMPETING (there were no other-incorrect answers anywhere).

### Secondary diagnostic (labelled; not part of the table above)
Rule FLR-v1 (Final-Line Recovery, post-hoc): read the last non-empty line of the reply, strip markdown emphasis, accept one letter A-D. It changes only the two Claude columns: Sonnet becomes 32/0/0/0 in the first four rows and 2/30/0/0 in
COMPETING (the same totals as GPT-4.1 mini); Haiku's `**B**` becomes correct (PROSE 32/0/0/0) and its refusals stay refusals. FLR-v1 was defined after the replies were read and credits replies the prompt forbade;
it describes the letter Sonnet settled on, not format compliance. See [`../../docs/EVIDENCE_RECONCILIATION.md`](../../docs/EVIDENCE_RECONCILIATION.md).

## Contents
| Path | What it is |
|---|---|
| `haiku55/`, `sonnet55/` | `session.json`, `prompts.json` and `records/<cell>__<condition>__pass<k>.json` (request parameters without credentials, response, strict status, usage, estimated cost); 160 records each |
| `atlas-input.json` | Atlas manifest: the five hosted sessions next to the local DIRECT_PROBABILITY sessions (`../four-mapping`) |
| `verify_comparison.py` | offline check: replays the Claude sessions, compares the table above, scans the bundle for secrets, thinking signatures, reasoning text and local identifiers |
| `sanitize_claude_sessions.py` | maintainer tool that produced `haiku55/` and `sonnet55/` from the saved runs |
| `estimate_pilot_cost.py` | offline cost estimator used to plan the live runs |
| `SHA256SUMS.txt` | SHA-256 of every file except this README |

## What was changed from the raw runs
Only two things, both in `records/*.json`: opaque thinking signatures and any reasoning text were replaced by placeholders (`[REDACTED:...]`), and each provider message id was replaced by a pseudonymous
`msg-<hash>`. Answers, statuses, usage, cost estimates, stop reasons and refusal categories are untouched, and replay output equals the raw runs'. No API key was ever stored.

## Reproduce live (sends requests with your key and may cost money)
```
python -m bleediff anthropic --prompts examples/gemini-answer-only/prompts.json --model claude-haiku-5-5 --out runs/claude-haiku-5-5 --passes 2 \
    --budget 0.25 --price-in 0.10 --price-out 0.50 --effort low --max-output-tokens 1024
python -m bleediff anthropic --prompts examples/gemini-answer-only/prompts.json --model claude-sonnet-5-5 --out runs/claude-sonnet-5-5 --passes 2 \
    --budget 2.00 --price-in 2.00 --price-out 10.00 --effort low --max-output-tokens 1024
python examples/hosted-comparison/estimate_pilot_cost.py --prompts examples/gemini-answer-only/prompts.json --passes 2 --probe-chars 560 --probe-input-tokens 228 \
    --max-output-tokens 1024 --price claude-haiku-5-5 0.10 0.50 --price claude-sonnet-5-5 2.00 10.00
```
The key is read from `ANTHROPIC_API_KEY` only. Claude 5.x thinks adaptively and `max_tokens` caps thinking **plus** text (a probe with `max_tokens` 8 returned an empty thinking block and no text, recorded as an
empty non-answer), so keep `--max-output-tokens` generous. A live rerun will not reproduce these records exactly: provider behaviour can change and refusals were not always repeatable.

## Interpretation limits
- One filler string, one prompt wording, 4 questions x 4 mappings = 16 cells that are not independent samples; the second pass is a replicate check (identical answers are stability, not confidence).
- Hosted chat presentation differs from the raw completion used for the local sessions; hidden reasoning and system behaviour differ per provider, so model differences cannot be attributed to capability.
- Haiku and Sonnet differ in thinking behaviour at the same settings (Haiku about 17,000 thinking tokens in the pilot, Sonnet none), and temperature is the API default (not exposed).
- Refusal categories are provider metadata, not measured causes. Nothing here is a security finding, a jailbreak, or evidence about a safeguard's design.
- Estimated costs and the budget guard use list prices and reported usage; they are not provider bills.
