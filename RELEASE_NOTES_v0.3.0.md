# Bleediff v0.3.0 release notes

**Status: exploratory research release. Not a security finding.** v0.3.0 adds three commands on top of the unchanged v0.2.0
research fold, and reports exploratory single-prompt measurements on three local models.

## What is new

| Command | Purpose |
|---|---|
| `bleediff track` | Interactive decision-margin tracker: edit a fragment, press Enter, see A/B/C/D probabilities, the signed A/B margin, changes and the preferred answer. |
| `bleediff track --arms FILE` | Batch influence ranking of labelled candidate fragments against one clean baseline, with a leaderboard, a manifest with hashes and tokenizer counts. |
| `bleediff atlas` | Cross-task, cross-model reporting over saved batch sessions: validation, paired category effects, correctness outcomes, filters. Offline; never queries a model. |

Also: `examples/atlas/` (three small sanitized datasets that reproduce the findings below without a model), schemas documented in
`docs/TECHNICAL.md`, and a v0.3.0 README. Package version 0.3.0.

## How to read the measurements

Each score is one prompt (request #3 of a frozen scenario, clean or with a fragment appended), not the three-request fold, so values
are not `delta_contrast`. `margin_AB = ln(p_A / p_B)`. A candidate's *shift* is movement toward the designated incorrect answer from
**that model's own baseline**; compare signs and ranks across models, not magnitudes. An A/B crossing is not an overall
preferred-answer change; an error is *newly induced* only if the baseline was correct; raw-prompt and chat-template presentations are
never combined; low answer-letter coverage and saturated baselines are reported as separate findings.

## Exploratory findings

Qwen2.5-1.5B, Llama 3.2 3B Instruct and Qwen2.5-7B (Q4_K_M, CPU, llama.cpp b11963, raw prompts) on four independent synthetic tasks
(backup verification, network access, software integrity, sensor monitoring). Repeats were bit-identical and are reproducibility
checks, not independent samples.

* On-topic text that **also mentions option B's subject** (no evidence for B) moved the decision toward B more than off-topic text
  in 11 of 12 task-by-model cells (8 of 12 by at least 0.25 nats) and more than neutral prose in 11 of 12 (10 of 12). An explicit
  competing claim exceeded it in 0 of 4 tasks (Qwen 1.5B), 3 of 4 (Llama 3B) and 4 of 4 (Qwen 7B).
* **Confounded:** in the follow-up 2x2 mention experiment, B's subject beat the neutral fragment in 11 of 12 cells, but A's subject was
  always in slot 1 and B's in slot 2. The original fixed-position result is not evidence of a shared causal mechanism.
* **Llama 3B, sensor task:** mentioning B's subject without evidence for B newly induced an incorrect answer (A to B; A 0.491 / B 0.095
  at baseline, A 0.222 / B 0.486 with the fragment). It is the only newly induced incorrect answer in the 2x2 experiment.
* **Position swap (B in slot 1, A in slot 2):** the Llama effect and the answer flip survived (A 0.293 / B 0.386; the both-subjects
  fragment also flipped; B-subject effect +1.88 nats at slot 2, +1.37 at slot 1). Qwen 1.5B showed smaller effects at both positions
  (+0.55, +0.35). Qwen 7B was strongly position-sensitive (+0.21 at slot 2, -0.26 at slot 1) on a saturated baseline.
* **Models differ substantially**, including on neutral text (toward A for Qwen 1.5B, toward B for Llama 3B, mixed for Qwen 7B).
* Qwen 1.5B's network baseline already preferred D; later changes there are not induced errors.

## Limitations (read before citing)

* **No universal priming mechanism and no real tenant-isolation vulnerability was established.** These are observed effects of
  appended text on a model's answer probabilities.
* Topic relevance and option-subject priming are not separated; neither are answer labels, distinct noun phrases and tokenization.
* Four hand-written tasks, three models, one quantization each, single wordings per cell, one machine and one llama.cpp build;
  no spread estimate beyond exact repeats. The Qwen 7B baselines are saturated, so no fragment could plausibly flip them.
* The native-chat-template check on Llama 3B (one scenario) is not bundled; the examples contain raw-prompt sessions only.
* The v0.2.0 results, limitations and the INCONCLUSIVE verdict of the original five-scenario experiment are unchanged.

## Reproducing

Offline, no model: see `examples/atlas/README.md` (the `atlas` command on the bundled sessions). With models you supply yourself
(GGUF files and `llama-server`): see the README; the model file hashes used are listed there.

## Distribution

The release is the git tag `v0.3.0`. Model files, run directories and per-run sidecars are not distributed with the repository.
