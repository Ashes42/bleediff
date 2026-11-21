# Four-mapping counterfactual evidence bundle

A compact, **sanitized** copy of the saved measurements behind [`docs/FOUR_MAPPING_SUMMARY.md`](../../docs/FOUR_MAPPING_SUMMARY.md)
(about 0.9 MB, 164 files). It lets you recompute every number in that summary **offline, without a model, a server or a download**.

```
python -m bleediff atlas --manifest examples/four-mapping/atlas-input.json --out runs/four-mapping
python examples/four-mapping/verify_summary.py runs/four-mapping/atlas.json
```

(`--out` must not exist yet; run both commands from the repository root.) The second command prints one `PASS` or `FAIL` line per documented
statement and exits 1 if any fails. The first command validates all 48 sources (schemas, recorded hashes, scenario and model identity,
evidence-file hashes), re-derives every margin, role score and preferred answer from the saved probabilities and writes the Atlas report
(`report.md`, `atlas.json`, CSVs).

## What was measured

Six synthetic questions with objectively verifiable answers, each under four answer-letter assignments (m1: correct A / competing B,
m2: correct B / competing A, m3: correct C / competing D, m4: correct D / competing C), on two local models (Qwen2.5-1.5B-Instruct and
Llama-3.2-3B-Instruct, both Q4_K_M, CPU, llama.cpp b11963, raw prompts). m1 and m2 have five treatments (neutral, on-topic mention,
competing claim, negated claim, repeated claim); m3 and m4 have three (neutral, competing claim, repeated claim). One cell is one question
under one mapping: 24 cells per model, 48 in total, 192 scored candidates.

## Contents

| Path | What it is |
|---|---|
| `atlas-input.json` | Atlas input manifest (`bleediff.atlas-input/1`): for each of the 48 sessions its task, model (with model hash), correct and competing letters, candidate categories and evidence files with SHA-256 values |
| `sessions/<model>/<question>-<mapping>/` | `history.jsonl`, `batch-manifest.json`, `leaderboard.csv` exactly as the batch runner wrote them (full A to D probabilities, margins, semantic-role records, tokenizer counts, model and inference provenance). Only the first of each pair of identical repeat runs is included |
| `arms/m1m2/`, `arms/m3m4/` | The frozen candidate-fragment files (`<question>.arms.json`) |
| `fixtures/` | The synthetic scenarios (questions, options, correct answers) |
| `frozen/` | The frozen texts with hashes and tokenizer counts, written before any scoring |
| `SHA256SUMS.txt` | SHA-256 of every data file in the bundle (`sha256sum -c` format; paths relative to this directory) |
| `verify_summary.py` | Offline verification script (stdlib only) |

Not included: model files, prompts or fragment text beyond the frozen candidate files, server logs, absolute paths, credentials or any
other local run artifact.

## What the verification script checks

Hashes of all bundled files, then the numbers quoted in the summary: Llama 3.2 3B shift toward the competing answer positive in 24 of 24 cells,
competing evidence above neutral text in 24 of 24, repeated above single in 23 of 24, the mean shifts by mapping; Qwen2.5-1.5B's
baseline probability of the correct answer by letter, its letter B and D changes under competing evidence, its sign reversals across
mappings; and every newly-induced-incorrect-answer count (Qwen 1.5B: 9 under m1 and 1 under m2 across the five treatments).

## Reading the evidence

* Every cell is a single deterministic measurement. Every session was also run a second time and repeated bit-identically; repeats are
  reproducibility checks, **not** independent samples, so no confidence intervals are given.
* Four of the 24 possible answer-letter permutations, six hand-written questions, two models, one quantization each, one llama.cpp
  build. Letter position changes baseline confidence as well, so position and confidence effects are not fully separated.
* The evidence suggests **semantic-answer sensitivity in the tested Llama configuration** and **strong answer-letter sensitivity in the tested
  Qwen configuration**. It does not establish a model-family mechanism, a statistical effect size, or a security vulnerability.
