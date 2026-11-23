# Your first experiment (about ten minutes, no model, no key to prepare)

This walks you from nothing to a validated, budgeted, ready-to-run **answer-only** experiment. Steps 1-3 are offline and free. Step 4 is the only one that costs money, and you choose whether to run it.
Run everything from the repository root (`python -m bleediff ...`, or `bleediff ...` if installed). No GGUF file or `llama-server` is involved.

**What this is, and is not.** The starter is a **demonstration**: one question, re-lettered four ways. It shows the whole pipeline working. It is **not statistical evidence**: the four mappings are the same question,
`--passes` repeats are replicate checks, and **a spec that validates shows only that the prompts are well-formed, not that any context caused any change.** Bleediff computes no p-values or confidence intervals.
See [What it can support](#what-it-can-support).

## 1. Create the starter spec
```
python -m bleediff experiment init runs/my-first
```
This writes `runs/my-first/experiment.json` (`runs/` is git-ignored; the command refuses to overwrite). The spec holds:

| Field | Meaning |
|---|---|
| `questions[]` | `question`, `evidence`, and four `options`: `correct`, `competing` (the plausible wrong answer you are testing), `distractor_1`, `distractor_2`. |
| `conditions` | `{NAME: fragment}`. A fragment is a line of extra context placed **after the evidence**. The clean `BASELINE` (no fragment) is implicit. The starter has a `TREATMENT_COMPETING_CLAIM` and a `CONTROL_NEUTRAL`. |
| `mappings` | `{name: {letter: role}}`: **explicitly** which letter holds which answer. The starter moves the correct answer to A, B, C and D (and the competing one with it) so a letter preference cannot pass as an effect. |

Edit the question, options, evidence and both fragments. Give the control **similar character length; token counts may differ** (characters do not establish tokenizer parity), and keep it unrelated to the answer.

## 2. Validate offline, write the prompts, estimate the budget
```
python -m bleediff experiment validate runs/my-first/experiment.json --write-prompts runs/my-first/prompts.json --provider anthropic --price-in 1.00 --price-out 5.00
```
**The prices above are placeholders.** Bleediff has no built-in prices: look up your provider's current list prices before any paid run and use those.

It checks the spec, prints every letter mapping, writes `prompts.json` (schema `bleediff.hosted-prompts/1`) and **re-loads it with the loader the live runners use**, so a file that passes is accepted by
`bleediff anthropic|openai|gemini`. Existing files are never overwritten, and `--write-prompts` needs an existing directory. Nothing here contacts an API.

Its warnings are **heuristics**: a condition counts as a control if its *name* contains `CONTROL`, and lengths are compared in characters. **No warning does not prove a control is neutral.** Judge that yourself.

**Budget figures follow the provider** (`--provider anthropic|openai|gemini`), because the runner reserves different output amounts. For the starter at two passes (24 requests) and the placeholder prices:

| Provider (default output cap) | 1. one successful attempt each, ceiling | 2. budget to admit all requests | 3. if a reply fills over half the reserve | 4. if every request needs 2 retries |
|---|---|---|---|---|
| anthropic (1024) | $0.1356 | $0.14 | $0.15 | $0.65 |
| openai (512) | $0.0742 | $0.08 | $0.08 | $0.35 |
| gemini (512; reserve is 2x the cap for thinking) | $0.1356 | $0.14 | $0.15 | $0.65 |

Real answer-only replies usually cost far less than figure 1. But the live guard decides whether to *send* each request from worst cases, not from what you will actually be charged, so a `--budget` below figure 2 stops
early. Two things push the need higher: once any reply uses more than half the reserve, the guard doubles the reserve for later requests (figure 3); and a failed attempt (network error or HTTP 5xx) is charged at the worst case
and a retry is sent only if one more worst case still fits (figure 4, from `--max-retries`, default 2). A run that stops early is saved as `partial` with fewer records. These are estimates from your prices, **not a provider billing limit
or a guarantee that a run completes**; check the provider's dashboard for what you were charged.

## 3. See what the report looks like (free)
```
python -m bleediff anthropic --replay examples/hosted-comparison/haiku55
```
This replays the bundled Haiku session: a **different experiment** (80 prompts, conditions `COMPETING`, `PROSE`, `RANDOM_*`). It demonstrates only the **reporting layout**, not the conditions of your starter. Your own run is reported the same way, with two kinds of numbers kept apart:
* **Strict (primary):** a reply counts only if it is exactly one letter `A`-`D`. Counts per condition: correct, competing, other incorrect, non-answer; pairs against the baseline: changed, newly induced incorrect. Refusals and prose are **non-answers, never "incorrect"**.
* **Diagnostic (secondary, labelled):** FLR-v2 recovers the final letter of format-invalid replies. It is post-hoc, never changes a strict count and is shown in its own columns.

## 4. Run it live (paid; only when you decide to)
Confirm current prices first. Set a key in your shell (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `GEMINI_API_KEY`; it is never saved or printed), then choose a `--budget` from step 2 (the example uses figure 4, which leaves room for retries; prices are the same placeholders):
```
python -m bleediff anthropic --prompts runs/my-first/prompts.json --model claude-haiku-5-5 --out runs/my-first/haiku --passes 2 --budget 0.65 --price-in 1.00 --price-out 5.00 --effort low --max-output-tokens 1024
python -m bleediff anthropic --replay runs/my-first/haiku --json-out runs/my-first/haiku-analysis.json
```
`--out` must not exist. Read [HOSTED_MODELS.md](HOSTED_MODELS.md) section 3 for what the cap is and is not. Prompts and replies are kept locally and are sent to the provider.
Leave the saved session in `runs/` (git-ignored). Only share a session after reading the redaction notes in HOSTED_MODELS.md section 5.

## What it can support
* **Can show:** what this model did on exactly these prompts: how often each answer appeared per condition, and whether answers differ between the baseline, the treatment and the control.
* **Cannot show:** that an effect is real, general or causal; anything about other wording, questions or models; a probability, confidence or ranking (hosted models give none). Validation never establishes a causal result.
* **Toward evidence:** many *independent* questions, each with its own evidence and its own controls (rewording one question 30 times does not make 30 questions); write down the questions, conditions and scoring before running; report strict counts as primary and label any diagnostic reading post-hoc; keep failed and refused records. `validate` changes its label at 30 questions, an illustrative convention, **not a power analysis**.
