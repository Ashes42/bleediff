# Bleediff

> What survives subtraction was already in the composite.

**Bleediff is a local GGUF evaluation CLI that measures how added context changes a model's A/B/C/D choice probabilities.** It runs
against a `llama-server` you start yourself, uses only the Python 3.11+ standard library, and keeps no prompts or fragment text
unless you ask. Version **0.3.0** is an **exploratory research toolkit** with measurements on three local models. It is not a
security scanner and has not demonstrated a vulnerability.

## What is in v0.3.0

| Command | What it does | Needs a model? |
|---|---|---|
| `bleediff track` | Interactive decision-margin tracker: edit a text fragment, press Enter, see how the answer probabilities and the A/B margin move. | yes (`llama-server`) |
| `bleediff track --arms FILE` | Batch influence ranking: score a list of labelled candidate fragments against one clean baseline and rank the observed effects. | yes |
| `bleediff atlas` | Cross-task, cross-model report over saved batch sessions: validated sources, paired category effects, correctness outcomes, filters. | **no** (reads saved results) |
| `calibrate`, `bounce`, `diff`, `inspect` | The v0.2.0 three-request fold research CLI (unchanged). | yes |

The v0.2.0 research fold (`M = (p1 + p2 + p3) / 3` against a clean baseline, `delta_contrast = R_B - R_A`), its equations, thresholds,
trace schema and results are **unchanged**; see [Research findings from v0.2.0](#research-findings-from-v020) and
[`docs/TECHNICAL.md`](docs/TECHNICAL.md). The three v0.3.0 commands are a different measurement: **one prompt per score** (request
#3 of a frozen scenario, clean or with a fragment appended after its authorized context), so their values are not
`delta_contrast`.

## Quick start (no model needed)

```
python -m unittest discover -s tests                 # mock llama-server; takes a few minutes

# the Atlas on the bundled example sessions (saved results of real runs; see examples/atlas/README.md)
python -m bleediff atlas --manifest examples/atlas/pilot/atlas-input.json --out runs/atlas-pilot
python -m bleediff atlas --manifest examples/atlas/slot-swap/atlas-input.json --out runs/atlas-slot-swap
python examples/atlas/mention_contrasts.py runs/atlas-slot-swap/atlas.json
```

`--out` directories must not exist; reports are never overwritten. Nothing here downloads or runs a model.

## The three commands

The real-model commands below need **your own GGUF files and a llama.cpp `llama-server`** (the examples used build
`b11963-39febd2ad`, CPU, `-c 4096 -ngl 0`). Without them only the test suite (mock server) and `atlas` run.

```
llama-server -m qwen2.5-1.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8099 -c 4096 -ngl 0 --no-webui
```

### `bleediff track`: interactive

```
python -m bleediff track --server http://127.0.0.1:8099 --out runs/my-session
# edit runs/my-session/fragment.txt in any editor and save it, then press Enter to rescore; `again` remeasures; `quit` ends
```

The clean baseline is scored first, then each saved edit. Every step shows the normalized A/B/C/D probabilities, the signed margin
`margin_AB = ln(p_A / p_B)`, the change from the baseline and from the previous step, and the overall preferred answer. The session
is written to `<out>/history.jsonl` (hashes, probabilities, margins, model and inference provenance); prompts and fragment text are
**not** saved unless you pass `--retain-prompts`. Missing or non-finite probabilities are errors, never zeros.

### `bleediff track --arms`: batch ranking

```
python -m bleediff track --server http://127.0.0.1:8099 --arms candidate-fragments.json --out runs/batch-001
```

`candidate-fragments.json` is a UTF-8 list of `{"label": "...", "text": "..."}` (or `{"candidates": [...]}`, or the
`{"arms": {label: text}}` / `{"fragments": {...}}` shape; an optional `sha256` per candidate is verified). The baseline is measured
**once** (tokenizer and server checks, model hash and flags recorded) and every candidate is scored with the same unchanged
measurement. Outputs: `history.jsonl` (full A-D probabilities, margins, changes, tokenizer counts), `leaderboard.csv` and
`batch-manifest.json` (hashes, provenance). The terminal summary is ranked by observed movement toward the designated incorrect
answer (`--incorrect B` by default; ties share a rank) and shows fragment and prompt token counts, so length-unmatched lists are
visible. Failed requests, invalid probabilities and invalid candidates are recorded as not scored and never imputed. There is no
automatic prompt generation. Rankings are **observed candidate effects**: a large shift does not show that a particular word caused it.

**Answers at other letters.** The A/B margin `ln(p_A / p_B)` describes the letters A and B only. When an answer-choice permutation puts the
correct and the competing answers elsewhere, pass `--correct C --incorrect D` (any two different letters; `--correct` is optional while
`--incorrect` is A or B, as in v0.3.0). Bleediff then scores the **semantic roles**: the correct-versus-competing margin
`ln(p_correct / p_competing)`, the correct-versus-strongest-incorrect margin, the movement toward the competing answer, overall
preferred-answer flips and newly induced incorrect answers (only when the clean baseline was correct). The terminal, `leaderboard.csv`
(last five columns) and the Atlas label which margin is which, and the fixed A/B margin stays available as `margin_AB`. A summary of the
first four-mapping experiment is in [`docs/FOUR_MAPPING_SUMMARY.md`](docs/FOUR_MAPPING_SUMMARY.md).

### `bleediff atlas`: cross-task and cross-model report

```
python -m bleediff atlas --manifest atlas-input.json --out runs/atlas-report [--model M] [--task T] [--category C] [--outcome O]
```

The input manifest (`bleediff.atlas-input/1`; see the docstring of `bleediff/atlas.py` and
[`examples/atlas/pilot/atlas-input.json`](examples/atlas/pilot/atlas-input.json)) maps each batch-session directory to its task,
model, prompt presentation, correct and designated incorrect answer, candidate categories and evidence files. The atlas validates
every source (schemas, recorded hashes, scenario and model identity, evidence-file hashes), re-derives each saved margin and
preferred answer from the saved probabilities with the unchanged functions, and writes `atlas.json`, `candidates.csv`,
`category-summary.csv`, `model-comparison.csv`, `report.md` and a terminal summary. It never contacts a model. To import your own
runs, point a manifest at your batch-session directories (copy the example and edit the paths, ids and candidate categories).
Filters (`--model`, `--task`, `--category`, `--outcome`) apply to the CSVs, report and terminal; `atlas.json` always keeps the
complete data.

## Reading the numbers

* **A/B margin movement.** `margin_AB = ln(p_A / p_B)`; positive favors A. A candidate's **shift** (movement toward the designated
  incorrect answer B) is the baseline margin minus the candidate margin, in nats, measured from *that model's own clean baseline*.
  Magnitudes are model-specific: across models compare signs and within-cell ranks, not raw margins.
* **A/B ranking crossing vs overall preferred-answer change.** A crossing is the margin changing sign (A above B becomes B above A).
  An overall preferred-answer change (**flip**) is the highest-probability answer among A-D changing. The two are different: with D
  leading, A and B can cross without the preferred answer changing. Bleediff records both separately.
* **Newly induced incorrect answer.** The clean baseline preferred the correct answer and a candidate made the model prefer a wrong
  one. If the baseline was already wrong (for example Qwen2.5-1.5B preferred D on one task) later changes are *not* reported as
  newly induced errors; the atlas labels them `baseline_incorrect_*` or `recovered_to_correct`.
* **Raw vs chat-template measurements.** All bundled results use the *raw* prompt: the frozen text is sent to `/completion` with no
  chat template, and the answer letter is read at the next-token position. An instruct model's *native chat template* is a different
  presentation (system and user turns, answer position after the assistant header); its results must not be mixed with raw ones. The
  atlas groups by declared presentation and never combines them. A native-template check on Llama 3.2 3B (one scenario, six
  conditions; a local run whose files are not bundled) kept the direction and order of the effects while the shifts grew several-fold
  from a much more confident baseline and letter coverage fell.
* **Answer-letter probability coverage.** The share of the model's first-token probability mass that lies on the four answer-letter
  surfaces before renormalization. Low coverage (flagged below 0.90) means the A-D probabilities rest on less of the model's actual
  output distribution. **Saturated baselines** (highest probability at least 0.995; Qwen2.5-7B here) leave little room for a visible
  flip and make shifts large logit changes of an extremely confident answer. Both are reported as separate findings, never as noise.

## Exploratory findings in v0.3.0

Three local models (Qwen2.5-1.5B, Llama 3.2 3B Instruct, Qwen2.5-7B; Q4_K_M, CPU, raw prompts) on four independent synthetic
tasks (backup verification, network access, software integrity, sensor monitoring). Every session was run twice and was bit-identical;
those repeats check reproducibility and are **not** independent statistical samples. The bundled example sessions reproduce the
numbers below (`examples/atlas/README.md`).

* **Mentioning B's subject often increased B preference.** In the pilot, on-topic text that *also mentions option B's subject* (with
  no evidence for B) moved the decision toward B more than off-topic text in 11 of 12 task-by-model cells (8 of 12 by at least 0.25
  nats) and more than neutral prose in 11 of 12 (10 of 12). An explicit competing claim exceeded that on-topic text in 0 of 4 tasks for
  Qwen 1.5B, 3 of 4 for Llama 3B and 4 of 4 for Qwen 7B.
* **The fixed-position result was confounded.** In the follow-up 2x2 (A's and B's subjects in two slots), mentioning B beat the neutral
  fragment in 11 of 12 cells, but A's subject always sat in slot 1 and B's in slot 2, so the answer subject and its position were
  not separated. The original 11/12 results are not evidence of a shared causal mechanism.
* **One newly induced incorrect answer: Llama 3B, sensor task.** Naming B's subject without evidence for B moved the preferred answer
  from A to B (A 0.491 / B 0.095 at baseline; A 0.222 / B 0.486 with the fragment).
* **The Llama effect and the flip survived swapping the subject positions** (B in slot 1: A 0.293 / B 0.386; the both-subjects
  fragment also flipped). Its B-subject effect was +1.88 nats at slot 2 and +1.37 at slot 1.
* **Qwen 1.5B showed smaller B-subject effects at both positions** (+0.55 and +0.35 nats) and no flips.
* **Qwen 7B was strongly position-sensitive** in the sensor experiment: the B-subject effect was +0.21 at slot 2 and -0.26 at slot 1
  (a sign reversal), on a saturated baseline (margin about +16).
* **Models differ substantially, including on neutral text**, which moved Qwen 1.5B toward A, Llama 3B toward B, and Qwen 7B in
  mixed directions.
* **Not established:** a universal priming mechanism, any real tenant-isolation vulnerability, or a distinction between topic
  relevance and option-subject priming (the `topic` category is *on-topic text that also mentions option B's subject*).
  Four hand-written tasks, three models, single wordings per cell and one llama.cpp build are a small exploratory sample.

## Research findings from v0.2.0

Findings of the three-request fold (config-audit-v2; Qwen2.5-1.5B and Qwen2.5-7B, Q4_K_M, CPU, llama.cpp b11963). `delta_contrast`
in nats; the raw answer (argmax of the composite) is in brackets. Full numbers, including every control, are in
[`research/data/results-v0.2.0.json`](research/data/results-v0.2.0.json). These results are unchanged.

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

### What the v0.2.0 results do and do not show

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
* **Scope of v0.2.0:** one model family (Qwen2.5), one quantization per model (Q4_K_M), one scenario chosen after the fact, selected
  prompts, single probe texts, CPU runs on one machine and one llama.cpp build. Model-family differences were untested there (v0.3.0
  adds Llama 3.2 3B; see above).

### Reproducing the v0.2.0 fold measurements

```
python research/run_arms.py --server http://127.0.0.1:8099 \
    --frozen research/data/frozen-arms.json \
    --controls research/data/neutral-controls-replicates-manifest.json \
    --out runs/my-run
```

This does a fresh clean calibration, runs the five frozen treatment-type texts and the 12 neutral controls twice, writes
15-field traces with provenance sidecars and an `analysis.json` (deterministic on the tested setup, not guaranteed identical on other
hardware or builds). Expected file hashes for the models:

| Model file | SHA-256 |
|---|---|
| `qwen2.5-1.5b-instruct-q4_k_m.gguf` | `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |
| `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` | `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db` |
| `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf` | `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a` |
| `Llama-3.2-3B-Instruct-Q4_K_M.gguf` (v0.3.0 sessions) | `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff` |

Both Qwen models share the Qwen2.5 tokenizer, so every frozen text and control keeps exact token parity (34 fragment tokens,
245 request-3 prompt tokens). Compare your `analysis.json` with `research/data/results-v0.2.0.json`. The original CLI
(`bleediff calibrate | bounce | diff`) and the frozen five-scenario protocol are documented in
[`docs/TECHNICAL.md`](docs/TECHNICAL.md).

## Repository layout

* `bleediff/` the measurement code: the frozen fold, contrast, validity gates and trace schema, plus `track.py`, `batch.py` and
  `atlas.py` (v0.3.0).
* `research/` the supported research drivers (`run_arms.py`, `arms.py`, `measure_fragments.py`) and `research/data/`
  (frozen texts, the 12 preserved neutral controls, saved traces and the v0.2.0 results summary).
* `examples/atlas/` small sanitized example datasets (saved batch sessions, candidate files, fixtures, atlas input manifests) that
  reproduce the v0.3.0 findings offline, and `mention_contrasts.py`.
* `fixtures/` scenario fixtures, preserved controls and the protocol and preregistration documents.
* `docs/TECHNICAL.md` the detailed method and history; `CHANGELOG.md`; `RELEASE_NOTES_v0.2.0.md`, `RELEASE_NOTES_v0.3.0.md`.
* `tests/` stdlib `unittest` suite (a mock llama-server stands in for the model).

Local run directories (`runs/`), model files and per-run sidecars are git-ignored and are not part of the repository.

## Git identity and license

The repository is configured with a repo-local, non-personal author (`bleediff <noreply@bleediff.invalid>`), and a test fails
if a file contains a personal Gmail address. MIT license.
