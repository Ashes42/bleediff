# Your first experiment (about ten minutes, no model, no key to prepare)

This walks you from nothing to a validated, budgeted, ready-to-run **answer-only** experiment. Steps 1-3 are offline and free. Step 4 is the only one that costs money, and you choose whether to run it.
Run everything from the repository root (`python -m bleediff ...`, or `bleediff ...` if installed). No GGUF file or `llama-server` is involved.

**What this is, and is not.** The starter is a **demonstration**: one question, re-lettered four ways. It shows the whole pipeline working and what the output looks like. It is **not statistical evidence**: the four
mappings are the same question, and `--passes` repeats are replicate checks, not independent samples. Bleediff computes no p-values or confidence intervals. See [What it can support](#what-it-can-support).

## 1. Create the starter spec
```
python -m bleediff experiment init runs/my-first
```
This writes `runs/my-first/experiment.json` (`runs/` is git-ignored; the command refuses to overwrite). The spec holds:

| Field | Meaning |
|---|---|
| `questions[]` | `question`, `evidence`, and four `options`: `correct`, `competing` (the plausible wrong answer you are testing), `distractor_1`, `distractor_2`. |
| `conditions` | `{NAME: fragment}`. A fragment is a line of extra context placed **after the evidence**. The clean `BASELINE` (no fragment) is implicit. The starter has a `TREATMENT_COMPETING_CLAIM` and a length-matched `CONTROL_NEUTRAL`. |
| `mappings` | `{name: {letter: role}}`: **explicitly** which letter holds which answer. The starter moves the correct answer to A, B, C and D (and the competing one with it) so a letter preference cannot pass as an effect. |

Edit the question, options, evidence and both fragments. Keep the control about as long as the treatment and unrelated to the answer.

## 2. Validate offline and write the prompts
```
python -m bleediff experiment validate runs/my-first/experiment.json --write-prompts runs/my-first/prompts.json --passes 2 --price-in 1.00 --price-out 5.00
```
It checks the spec, prints every letter mapping, warns about a missing or non-neutral control, mismatched fragment lengths and a correct answer that never changes letter, writes `prompts.json`
(schema `bleediff.hosted-prompts/1`), and **re-loads it with the loader the live runners use**, so a file that passes is accepted by `bleediff anthropic|openai|gemini`. With both prices it prints the estimated worst-case
cost and a suggested `--budget`. Use your provider's current list prices: Bleediff has none built in. The starter is 12 prompts, 24 requests at `--passes 2`; at the example prices above (1.00 / 5.00 USD per million tokens) the worst case is about $0.14,
and real answer-only replies usually cost much less (this is an estimate, never a bill). Nothing here contacts an API.

## 3. See what the report will look like (free)
```
python -m bleediff anthropic --replay examples/hosted-comparison/haiku55
```
This replays a saved session of the same shape. Your own run is reported the same way, with two kinds of numbers kept apart:
* **Strict (primary):** a reply counts only if it is exactly one letter `A`-`D`. Counts per condition: correct, competing, other incorrect, non-answer; pairs against the baseline: changed, newly induced incorrect. Refusals and prose are **non-answers, never "incorrect"**.
* **Diagnostic (secondary, labelled):** FLR-v2 recovers the final letter of format-invalid replies. It is post-hoc, never changes a strict count and is shown in its own columns.

## 4. Run it live (paid; only when you decide to)
Set a key in your shell (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `GEMINI_API_KEY`; it is never saved or printed), then, using the suggested budget from step 2:
```
python -m bleediff anthropic --prompts runs/my-first/prompts.json --model claude-haiku-5-5 --out runs/my-first/haiku --passes 2 --budget 0.14 --price-in 1.00 --price-out 5.00 --effort low --max-output-tokens 1024
python -m bleediff anthropic --replay runs/my-first/haiku --json-out runs/my-first/haiku-analysis.json
```
`--out` must not exist, and the run stops before a request could exceed the estimated cap. Read [HOSTED_MODELS.md](HOSTED_MODELS.md) section 3 for what the cap is and is not. Prompts and replies are kept locally and are sent to the provider.
Leave the saved session in `runs/` (git-ignored). Only share a session after reading the redaction notes in HOSTED_MODELS.md section 5.

## What it can support
* **Can show:** what this model did on exactly these prompts: how often each answer appeared per condition, and whether a treatment fragment changed answers compared with the baseline and the neutral control.
* **Cannot show:** that an effect is real, general or causal; anything about other wording, questions or models; a probability, confidence or ranking (hosted models give none).
* **Toward evidence:** many *independent* questions (each with its own evidence, not re-lettered copies); write down the questions, conditions and how you will score before running; keep the control; report strict counts as the primary result and label any diagnostic reading as post-hoc; keep failed and refused records. `validate` switches its label from DEMONSTRATION only at 30 questions, and that is a wording floor, not a power analysis.
