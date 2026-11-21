# Example Atlas datasets

Small **sanitized** datasets of real, saved batch sessions (about 0.5 MB in total) so the Influence Atlas and the central v0.3.0
findings can be reproduced **offline, without a model, a server or a download**:

```
python -m bleediff atlas --manifest examples/atlas/pilot/atlas-input.json --out runs/atlas-pilot
python -m bleediff atlas --manifest examples/atlas/answer-mention/atlas-input.json --out runs/atlas-answer-mention
python -m bleediff atlas --manifest examples/atlas/slot-swap/atlas-input.json --out runs/atlas-slot-swap
python examples/atlas/mention_contrasts.py runs/atlas-slot-swap/atlas.json
```

(`--out` directories must not exist yet.)

## What each dataset is

| Dataset | Sessions | Candidates per session | What it shows |
|---|---|---|---|
| `pilot/` | 4 tasks x 3 models = 12 | 5: explicit B claim, on-topic text that also mentions option B's subject (no evidence), off-topic text, neutral prose, matching identifier | the 12-cell comparison (on-topic vs off-topic and neutral; claim vs on-topic) and the one newly induced incorrect answer (Llama 3B, sensor) |
| `answer-mention/` | 4 tasks x 3 models = 12 | 4: neither / A's subject / B's subject / both (A always in slot 1, B in slot 2) | the 2x2 mention contrasts, confounded with slot position |
| `slot-swap/` | sensor task x 3 models = 3 | 8: the four original-position conditions re-measured plus the four position-swapped ones | whether the B-subject effect and the Llama flip survive moving B's subject to slot 1 |

Each dataset directory holds `sessions/<model>/<task>/` (`history.jsonl`, `batch-manifest.json`, `leaderboard.csv` exactly as the
batch runner wrote them), `arms/` (the frozen candidate files), `fixtures/` (the four synthetic scenarios), `frozen.json` (the frozen
texts with their hashes and tokenizer counts, written before any scoring) and `atlas-input.json`. Prompts and per-session probability
vectors beyond the recorded results are not included. Only the first of each pair of identical repeat runs is bundled; the repeats
were bit-identical and are reproducibility checks, **not** independent samples.

## Expected numbers (raw prompt presentation)

* `pilot/`: on-topic > off-topic in 11 of 12 cells (8 of 12 by at least 0.25 nats); on-topic > neutral in 11 of 12 (10 of 12); explicit
  claim > on-topic in 0 of 4 (Qwen 1.5B), 3 of 4 (Llama 3B), 4 of 4 (Qwen 7B); induced incorrect answers: Llama 3B / sensor only
  (claim, on-topic and identifier fragments, A to B); Qwen 1.5B's network baseline already prefers D (not counted as induced).
* `answer-mention/`: B's subject alone above the neutral fragment in 11 of 12 cells; adding B's subject to A's raises the shift in
  12 of 12; one induced incorrect answer (Llama 3B / sensor, `B_ONLY`: A 0.222, B 0.486 against a baseline of A 0.491, B 0.095).
* `slot-swap/` (`mention_contrasts.py`): B-subject effect (`B_ONLY - NEITHER`, nats toward B) at slot 2 and at slot 1: Llama 3B
  +1.88 and +1.37 (the flip survives; `BOTH_S` also flips), Qwen 1.5B +0.55 and +0.35, Qwen 7B +0.21 and -0.26.

The `topic` category is on-topic text that **also mentions option B's subject**: topic relevance and option-subject priming cannot be
separated by these experiments. Shifts are measured from each model's own baseline; compare signs and ranks, not magnitudes, across
models. Qwen 7B's baselines are saturated.

## Provenance

* Models (Q4_K_M, CPU, llama.cpp `b11963-39febd2ad`, `-c 4096 -ngl 0 --no-webui`, raw `/completion` prompts):

| Model | File SHA-256 |
|---|---|
| Qwen2.5-1.5B-Instruct (`qwen2.5-1.5b-instruct-q4_k_m.gguf`) | `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |
| Qwen2.5-7B-Instruct (two shards) | `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db`, `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a` |
| Llama-3.2-3B-Instruct (`Llama-3.2-3B-Instruct-Q4_K_M.gguf`) | `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff` |

  The Qwen files are from the Qwen GGUF repositories; the Llama file is a community GGUF conversion whose provenance is only as
  good as its hosting repository's listing (the hash matched that listing; nothing beyond transfer integrity is claimed). Each
  session records the model's header hash, which the atlas manifests check.
* The atlas validates the recorded SHA-256 values in each `batch-manifest.json` against the files here. `.gitattributes` marks
  `examples/atlas/**` as `-text` so a checkout with any line-ending setting keeps these bytes (and hashes) unchanged.

## Importing your own full local runs

Run the batch mode for each task and model (`bleediff track --arms ... --scenario TASK --out DIR`, see the README), then write an
`atlas-input.json` like the bundled ones: one entry per session directory with the task id (equal to the scenario id, or give
`scenario_id`), model id, prompt `presentation` (`raw` or your own label; presentations are never combined), `correct_answer`,
`incorrect_answer`, a `candidates` map from candidate label to category (`claim`, `topic`, `offtopic`, `neutral`, `id`, or any string
of your own), and optional `evidence` file entries with SHA-256 values. Mark a repeated run with `replicate_of` so it is checked for
bit-identity instead of being counted. Then `python -m bleediff atlas --manifest atlas-input.json --out runs/atlas-report`.
