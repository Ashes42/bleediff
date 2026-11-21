# Bleediff

> What survives subtraction was already in the composite.

**Bleediff is a local GGUF evaluation CLI that measures how added context changes a model's A/B/C/D choice
probabilities, by folding three requests into one composite and comparing it with a clean baseline.** It runs against a
`llama-server` you start yourself, uses only the Python 3.11+ standard library, and keeps no per-request probability
vectors or prompts. Version **0.2.0** is an **exploratory research toolkit** with measurements on two models. It is not a
security scanner and has not demonstrated a vulnerability.

## What it does

1. A fixed multiple-choice prompt (question, options A-D, evidence from three authorized contexts) is sent to the model
   three times, once per authorized context. The model's answer-letter probabilities are read before sampling and
   renormalized over A, B, C, D.
2. The three probability vectors are averaged into a composite `M = (p1 + p2 + p3) / 3` and then discarded.
3. A **clean baseline** `B_ref` is the same fold with nothing added. The **residual** for choice c is
   `R_c = log(M_c + eps) - log(B_ref_c + eps)` and the headline number is `delta_contrast = R_B - R_A` (nats): how much
   the composite moved toward B relative to A.
4. In the displacement experiments, request #3's evidence is extended by 25 words of an *adjacent* tenant's text (the
   "treatment"), or by matched neutral prose (the "control"). Everything else in the prompt is byte-identical.

The question the project asked is whether such displaced context leaves a measurable trace in the output distribution
even when the final answer does not change. The detailed method, validity gates, trace schema and history are in
[`docs/TECHNICAL.md`](docs/TECHNICAL.md).

## Findings (config-audit-v2; Qwen2.5-1.5B and Qwen2.5-7B, Q4_K_M, CPU, llama.cpp b11963)

`delta_contrast` in nats; the raw answer (argmax of the composite) is in brackets. Full numbers, including every control,
are in [`research/data/results-v0.2.0.json`](research/data/results-v0.2.0.json).

| | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---|---|
| Clean baseline (p_A / p_B) | 0.769 / 0.208 (A) | 0.555 / 0.445 (A) |
| Original B treatment | **+0.857** (A) | **+0.654** (B) |
| Paraphrase of the B evidence (no option-B words) | +0.713 (A) | +0.479 (B) |
| Polarity reversed (passed / logging disabled) | -0.183 (A) | -0.668 (A) |
| Cue probe: failed but logging disabled | +0.409 (A) | +0.400 (B) |
| Cue probe: passed but logging enabled | +0.726 (A) | +0.261 (B) |
| 12 neutral controls: range | -0.241 to +0.061 | -0.698 to -0.673 |
| 12 neutral controls: median | -0.115 | -0.694 |
| Controls inside the historical ±0.1 tolerance | 5 of 12 | 0 of 12 |
| Original B vs controls | above all 12 | above all 12 |

What the measurements show:

* The original treatment exceeded all 12 neutral controls on both models, and the paraphrase kept most of the effect
  (83% on 1.5B, 73% on 7B), so the effect does not require the exact answer-option wording.
* Reversing the stated state ("passed", "disabled") removed the effect on both models: it fell to the neutral level.
* On the 1.5B model the raw answer stayed A in every arm (the residual was hidden in the composite). On the 7B model the
  treatment **flipped the raw answer to B**, so there it is an overt answer change, not a hidden residual.
* The two polarity cues (verdict word "failed/passed" and logging state "enabled/disabled") each contribute, and their
  effects are not strictly additive (interaction contrast -0.46 on 1.5B and -0.68 on 7B; additive cues would give 0).
  On 1.5B the logging state moved the result more; on 7B the verdict word moved it slightly more.
* Repeated measurements were bit-identical in every tested configuration (each experiment was run twice).
* Neutral text itself shifts the composite: on both models it pushes toward A, on 7B by about 0.69 nats.

## What this does and does not show

* **The simulated displacement is ordinary text concatenation at the model interface.** The model receives the same bytes
  whether the text arrives as an "over-read" of an adjacent buffer or as authorized in-slot text; the harness cannot tell
  the two delivery routes apart.
* **The fold is algebraically recoverable** under the documented assumptions (requests 1 and 2 unchanged, a deterministic
  model and a known clean baseline): the request-3 probability change equals three times the composite change.
* **The historical ±0.1-nat control tolerance failed frequently** (5 of 12 neutral controls on 1.5B, 0 of 12 on 7B).
  Empirical neutral-control distributions **need not be centered at zero**.
* **Lexical overlap alone is insufficient to explain** the observed pattern (the paraphrase kept most of the effect and
  an overlapping-but-reversed text did not), but a **semantic interpretation is not uniquely established**: the probes
  change single cue words, and shallow polarity cues cannot be separated from meaning.
* **No real tenant-isolation vulnerability was demonstrated.** These are measurements of how a model's output
  distribution responds to appended text.
* **The original five-scenario experiment remains INCONCLUSIVE** (one of five trials valid; two clean baselines were not
  A-dominant; four of five matched controls were out of tolerance). Nothing later changes that verdict.
* **Scope:** one model family (Qwen2.5), one quantization per model (Q4_K_M), one scenario chosen after the fact, selected
  prompts, single probe texts, CPU runs on one machine and one llama.cpp build. Model-family differences are untested.

## Quick start

```
python -m unittest discover -s tests              # mock llama-server; no model needed
```

Reproduce the measurements (needs your own GGUF files and a llama.cpp `llama-server`, build `b11963-39febd2ad`;
CPU only, deterministic on the tested setup but not guaranteed identical on other hardware or builds):

```
llama-server -m qwen2.5-1.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8099 -c 4096 -ngl 0 --no-webui

python research/run_arms.py --server http://127.0.0.1:8099 \
    --frozen research/data/frozen-arms.json \
    --controls research/data/neutral-controls-replicates-manifest.json \
    --out runs/my-run
```

This does a fresh clean calibration, runs the five frozen treatment-type texts and the 12 neutral controls twice, writes
15-field traces with provenance sidecars and an `analysis.json`. Expected file hashes for the models:

| Model file | SHA-256 |
|---|---|
| `qwen2.5-1.5b-instruct-q4_k_m.gguf` | `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |
| `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` | `dfce12e3862a5283…` (full values in the results file) |

Both models share the Qwen2.5 tokenizer, so every frozen text and control keeps exact token parity (34 fragment tokens,
245 request-3 prompt tokens). Compare your `analysis.json` with `research/data/results-v0.2.0.json`; the saved traces
in `research/data/` are the ones the numbers above come from. The original CLI (`bleediff calibrate | bounce | diff`)
and the frozen five-scenario protocol are documented in [`docs/TECHNICAL.md`](docs/TECHNICAL.md).

## Repository layout

* `bleediff/` the frozen measurement code (fold, contrast, validity gates, trace schema, control generator, protocols).
* `research/` the supported research drivers (`run_arms.py`, `arms.py`, `measure_fragments.py`) and `research/data/`
  (frozen texts, the 12 preserved neutral controls, saved traces and the results summary).
* `fixtures/` scenario fixtures, preserved controls and the protocol and preregistration documents.
* `docs/TECHNICAL.md` the detailed method and history; `CHANGELOG.md`; `RELEASE_NOTES_v0.2.0.md`.
* `tests/` stdlib `unittest` suite (a mock llama-server stands in for the model).

Local run directories (`runs/`), model files and per-run sidecars are git-ignored and are not part of the repository.

## Git identity and license

The repository is configured with a repo-local, non-personal author (`bleediff <noreply@bleediff.invalid>`), and a test fails
if a file contains a personal Gmail address. MIT license.
