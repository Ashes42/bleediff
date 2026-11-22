# Hosted answer-only comparison: Claude Haiku, Claude Sonnet, Gemini Flash, Gemini Carbon, GPT-4.1 mini

Preparation for one descriptive comparison of five hosted models on the **same 80 frozen prompts** (4 synthetic questions x 4 answer-letter mappings x BASELINE, PROSE, RANDOM_AB,
RANDOM_OTHER, COMPETING; `examples/gemini-answer-only/prompts.json`, byte-identical to the one in `examples/openai-answer-only/`). **The Claude runs have not been performed yet**; the
Gemini and GPT-4.1 mini sessions are the saved pilots (offline, free).

| Model | Evidence | Status |
|---|---|---|
| `claude-haiku-5-5` | `bleediff anthropic` live run, 2 passes | **to run** (about $0.01 to $0.09 estimated) |
| `claude-sonnet-5-5` | `bleediff anthropic` live run, 2 passes | **to run** (about $0.13 to $1.76 estimated) |
| `gemini-3.8-flash` (Flash) | `examples/gemini-answer-only/flash` | saved, replays offline |
| `gemini-4-carbon` | `examples/gemini-answer-only/carbon` | saved, replays offline |
| `gpt-4.1-mini` | `examples/openai-answer-only/gpt41mini` | saved, replays offline |

All of them are **ANSWER_ONLY**: the Anthropic Messages API has no logprobs parameter, so no A-D probabilities exist for Claude, and the Gemini and OpenAI sessions are answer-only as well. The
local Qwen / Llama sessions in the same Atlas stay **DIRECT_PROBABILITY**. Nothing here computes a probability, margin, confidence or leaderboard from answers.

## Why the Claude settings matter (adaptive thinking)
Claude 5.x models think adaptively and `max_tokens` is a hard cap on thinking **plus** text. A real probe of `claude-haiku-5-5` with `max_tokens` 8 returned an empty thinking block and **no answer
text** (`stop_reason: max_tokens`, 8 thinking tokens), which the adapter records as an empty non-answer. For the pilot use `--max-output-tokens 1024` (also the per-request spending reserve) and
`--effort low` (soft guidance toward less thinking). Whether `--effort low` gives clean single-letter answers has **not** been verified live yet; the Haiku run below shows it in its first records, and its budget caps the exposure at $0.25.

## Cost estimate (ESTIMATE from list prices and one calibration request; not a provider billing total)
Prices (official pricing page, per million tokens, checked 2026-11-22): Haiku 5.5 $0.10 input / $0.50 output; Sonnet 5.5 $2.00 / $10.00. Calibration: the real probe sent a 560-character
prompt and the API reported 228 input tokens (0.407 tokens/character; 1.5x more than GPT-4.1 mini's 149 for the same prompt). Complete pilot = 80 prompts x 2 passes = 160 requests per model:

| model | input tokens (calibrated x1.5 margin) | answer only (5 out tokens) | light thinking (100) | moderate thinking (300) | all at the 1024 cap (worst case) |
|---|---|---|---|---|---|
| claude-haiku-5-5 | about 58,600 | $0.006 | $0.014 | $0.030 | $0.088 |
| claude-sonnet-5-5 | about 58,600 | $0.125 | $0.277 | $0.597 | $1.756 |

The live guard's hard per-request bound is $0.0006 (Haiku) and $0.012 (Sonnet), at most about $0.10 and $1.91 for the whole run if every request hit the cap. Both models together: about $0.14 to
$0.9 expected, never more than your `--budget` caps (the guard stops before a request that could exceed it). Reproduce the table offline:

```
python examples/hosted-comparison/estimate_pilot_cost.py --prompts examples/gemini-answer-only/prompts.json --passes 2 --probe-chars 560 --probe-input-tokens 228 \
    --max-output-tokens 1024 --price claude-haiku-5-5 0.10 0.50 --price claude-sonnet-5-5 2.00 10.00
```

## Runbook (nothing below has been executed)
The key is read from `ANTHROPIC_API_KEY` (never printed). Use new output directories.

```
# Haiku first: even at the cap it costs about $0.10, and its first records show whether these settings give clean single-letter answers
python -m bleediff anthropic --prompts examples/gemini-answer-only/prompts.json --model claude-haiku-5-5 --out runs/claude-haiku-5-5 --passes 2 \
    --budget 0.25 --price-in 0.10 --price-out 0.50 --effort low --max-output-tokens 1024

python -m bleediff anthropic --prompts examples/gemini-answer-only/prompts.json --model claude-sonnet-5-5 --out runs/claude-sonnet-5-5 --passes 2 \
    --budget 2.00 --price-in 2.00 --price-out 10.00 --effort low --max-output-tokens 1024

python -m bleediff atlas --manifest examples/hosted-comparison/atlas-input.template.json --out runs/hosted-comparison
```

Recommended order: run Haiku first (at most about $0.10 even at the cap) and inspect the first records (`records/*.json`: `status` should be `ok`); if the thinking still consumes the cap, raise
`--max-output-tokens` or try `--thinking disabled` if the model accepts it, then run Sonnet. `--effort low` and the other optional parameters are sent only when you pass them, and a value the
model rejects is reported once as an error, never retried.

## What the comparison can and cannot show
The same prompts and mappings make the answers comparable at the answer level (correct, competing, other incorrect, newly induced incorrect, flips against each model's own baseline, agreement
between the two passes). It cannot show confidence, probabilities or margins; repeated identical answers are replicate stability. Chat presentation (hosted) differs from the raw-completion local runs,
hidden reasoning differs per provider, 16 cells are not independent, and one wording per fragment was used.
