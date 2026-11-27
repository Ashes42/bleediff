# Bleediff

> What survives subtraction was already in the composite.

**Bleediff is an evaluation CLI that measures how added context changes a model's A/B/C/D choices.** It uses only the Python 3.11+ standard library.

- **Local:** GGUF evaluation runs against a local `llama-server` you start yourself, and keeps no prompts or fragment text unless you ask.
- **Hosted:** `bleediff gemini`, `bleediff openai` and `bleediff anthropic` send your prompts to the third-party API provider you select. Hosted sessions persist prompt and response data locally so they can be replayed offline. Do not submit sensitive prompt content unless you understand these implications.

> **Published evidence.** The Research 002 study behind v0.7.0 is public: **[https://github.com/Ashes42/bleediff-research-002](https://github.com/Ashes42/bleediff-research-002)** (release v1.0.0: frozen protocol, sanitized run records, report, errata, provenance and offline reproduction commands). `bleediff behavior` reads its records offline and reproduces its published condition-level and paired figures; see [What is in v0.7.0](#what-is-in-v070).

Version **0.7.0** is an **exploratory research toolkit** with measurements on three local models and, answer-level only, five hosted models. It is not a
security scanner and has not demonstrated a vulnerability.

## What is in v0.7.0

v0.7.0 adds **behavioral telemetry**: `bleediff behavior ingest|report` reads saved hosted sessions **offline** (no key, no network, no model) and reports four things separately, because Research 002 showed that a contextual change can sharply change how often a provider reports *thinking* while the *answer* barely moves: **answer behavior** (selection, correctness), **provider-reported thinking** (tokens, activation), **resource expenditure** (input/output/cache tokens, latency, estimated cost) and **format and refusal**. Every one of the 13 measurements carries exactly one of five validity states: **OBSERVED** (zero is a legitimate value), **MISSING**, **UNSUPPORTED**, **INCONSISTENT** or **INVALID**, and absent thinking telemetry is never converted to zero. Condition-level summaries sit beside unit-averaged ones, contrasts are explicitly paired, and there is no composite score and no inference. The Anthropic adapter is validated against the published Research 002 records; OpenAI and Gemini cover answers, format, refusal and input tokens only. Guide: [`docs/BEHAVIORAL_TELEMETRY.md`](docs/BEHAVIORAL_TELEMETRY.md); boundaries: [`docs/ARCHITECTURE_V07_BEHAVIOR.md`](docs/ARCHITECTURE_V07_BEHAVIOR.md); Research 002 compatibility demonstration: [`examples/behavior/r002_compat_check.py`](examples/behavior/r002_compat_check.py) against the [published repository](https://github.com/Ashes42/bleediff-research-002). Release notes: [`RELEASE_NOTES_v0.7.0.md`](RELEASE_NOTES_v0.7.0.md).

v0.6.0 had added **experiment authoring**: `bleediff experiment init|validate` writes and checks a small answer-only experiment offline (starter spec, per-question or global conditions, a prompt dataset the hosted commands accept, provider-aware budget estimates; guide: [`docs/FIRST_EXPERIMENT.md`](docs/FIRST_EXPERIMENT.md)), and the hosted replay and Atlas reports now split refusals from format-invalid replies and print a labelled FLR-v2 diagnostic beside the unchanged strict counts. The authoring workflow has been exercised offline only. Release notes: [`RELEASE_NOTES_v0.6.0.md`](RELEASE_NOTES_v0.6.0.md). v0.5.0 had added **hosted-model support**: `bleediff gemini`, `bleediff openai` and `bleediff anthropic` (`ANSWER_ONLY` runs with an enforced estimated-spend cap, redacted evidence and offline `--replay`), Atlas sections that keep
`ANSWER_ONLY` and `DIRECT_PROBABILITY` apart, and a replayable five-model comparison ([`examples/hosted-comparison/`](examples/hosted-comparison/README.md); guide: [`docs/HOSTED_MODELS.md`](docs/HOSTED_MODELS.md)).
Release notes: [`RELEASE_NOTES_v0.5.0.md`](RELEASE_NOTES_v0.5.0.md). v0.4.0 had added **semantic answer-role scoring** (the correct and the competing answers can sit at any letters A to D, with answer-letter
permutation support and clearer labels) and a sanitized, offline-reproducible **four-mapping evidence bundle**
([`examples/four-mapping/`](examples/four-mapping/README.md)); v0.3.0 sessions and datasets load unchanged ([`RELEASE_NOTES_v0.4.0.md`](RELEASE_NOTES_v0.4.0.md)).

| Command | What it does | Needs a model? |
|---|---|---|
| `bleediff track` | Interactive decision-margin tracker: edit a text fragment, press Enter, see how the answer probabilities and the A/B margin move. | yes (`llama-server`) |
| `bleediff track --arms FILE` | Batch influence ranking: score a list of labelled candidate fragments against one clean baseline and rank the observed effects. | yes |
| `bleediff atlas` | Cross-task, cross-model report over saved batch sessions: validated sources, paired category effects, correctness outcomes, filters. | **no** (reads saved results) |
| `bleediff gemini`, `bleediff openai`, `bleediff anthropic` | ANSWER_ONLY runs on the Gemini / OpenAI / Anthropic APIs (hosted models without usable A-D probabilities), or `--replay` of saved evidence. | live: API key and budget; replay: **no** |
| `calibrate`, `bounce`, `diff`, `inspect` | The v0.2.0 three-request fold research CLI (unchanged). | yes |
| `bleediff behavior ingest / report` | Offline behavioral telemetry over saved hosted sessions: answer behavior, provider-reported thinking, resources and format/refusal as separate blocks, with explicit measurement states and paired contrasts. Guide: [`docs/BEHAVIORAL_TELEMETRY.md`](docs/BEHAVIORAL_TELEMETRY.md). | **no** |
| `bleediff experiment init / validate` | Author your own small answer-only experiment: a starter spec, offline validation, a prompt dataset the hosted commands accept, and an estimated budget. Guide: [`docs/FIRST_EXPERIMENT.md`](docs/FIRST_EXPERIMENT.md). | **no** |
| `bleed` | v0.2.0 helper: offline preview of the exact arena boundary displacement against the matched control (see `bleediff bleed --help`). | **no** |

The v0.2.0 research fold (`M = (p1 + p2 + p3) / 3` against a clean baseline, `delta_contrast = R_B - R_A`), its equations, thresholds,
trace schema and results are **unchanged**; see [Research findings from v0.2.0](#research-findings-from-v020) and
[`docs/TECHNICAL.md`](docs/TECHNICAL.md). The three v0.3.0 commands are a different measurement: **one prompt per score** (request
#3 of a frozen scenario, clean or with a fragment appended after its authorized context), so their values are not
`delta_contrast`.

## Install

Requires **Python 3.11 or newer** and `git`. There are no third-party dependencies.

```
git clone https://github.com/Ashes42/bleediff.git
cd bleediff
python -m venv .venv
```

Activate the virtual environment (run **one** of these, for your platform, before installing, so Bleediff does not go into your system Python):

Windows PowerShell:

```
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```
source .venv/bin/activate
```

Then, with the environment active (your prompt usually shows `(.venv)`):

```
python -m pip install .
bleediff --version        # prints: bleediff 0.7.0
bleediff --help           # lists every command
```

`pip install .` is optional for the examples below: run from the repository root, `python -m bleediff ...` works without installing. All
example paths (`examples/...`, `runs/...`) are relative to the repository root, which is also where the bundled example data lives (it is
not installed into the venv).

## Quick start (no model needed)

```
python -m unittest discover -s tests                 # mock llama-server; takes several minutes (about 6 on one test machine)

# the Atlas on the bundled example sessions (saved results of real runs; see examples/atlas/README.md)
python -m bleediff atlas --manifest examples/atlas/pilot/atlas-input.json --out runs/atlas-pilot
python -m bleediff atlas --manifest examples/atlas/slot-swap/atlas-input.json --out runs/atlas-slot-swap
python examples/atlas/mention_contrasts.py runs/atlas-slot-swap/atlas.json

# the four-mapping experiment: rebuild the Atlas and check every documented number offline (examples/four-mapping/README.md)
python -m bleediff atlas --manifest examples/four-mapping/atlas-input.json --out runs/four-mapping
python examples/four-mapping/verify_summary.py runs/four-mapping/atlas.json

# hosted models, replayed offline (no key, no network): five models on the same 80 frozen prompts (examples/hosted-comparison/README.md)
python -m bleediff anthropic --replay examples/hosted-comparison/haiku55
python -m bleediff atlas --manifest examples/hosted-comparison/atlas-input.json --out runs/hosted-comparison
python examples/hosted-comparison/verify_comparison.py
```

`--out` directories must not exist; reports are never overwritten. Nothing here downloads or runs a model.

**About the test run.** The full suite takes several minutes. It can print `ResourceWarning` lines and one line starting `ERROR: ... provider message id(s) appear in more than one record`; that line is the expected output of a negative test (a deliberate duplicate-record check). What matters is the final `OK` from `unittest` (some tests are skipped by design).

## What you can reproduce

Not every number in this repository is reproducible in the same way. There are three tiers:

1. **Recomputed offline from bundled records.** The Atlas runs, the four-mapping numbers (`verify_summary.py`) and the hosted-comparison tables
   (`verify_comparison.py`) are recalculated from files in this repository, with no key, network or model. Run the Quick start commands above.
2. **Archived model responses: replayed and hash-verified.** The hosted sessions under `examples/hosted-comparison/`, `examples/gemini-answer-only/`
   and `examples/openai-answer-only/` are saved provider responses (with thinking signatures and reasoning text replaced by placeholders).
   `--replay` re-reads and re-validates them (schemas, hashes) and recomputes the counts, and `examples/hosted-comparison/SHA256SUMS.txt` lists a
   SHA-256 for each file. **Replay does not rerun model inference**: it shows that the saved records give the published tables, not that a model
   would answer the same way today. A live rerun will not reproduce these records exactly.
3. **Historical experiments that need material not in the repository.** The v0.2.0 fold and local-model measurements need your own GGUF files
   and `llama-server` (expected model hashes are listed below), and a bit-identical rerun is only claimed for the tested setup. The exploratory
   hosted refusal runs behind `docs/CASE_NOTE_HOSTED_REFUSALS_AND_FORMAT.md` are git-ignored raw runs that are **not published**; only their
   per-request index (`docs/evidence/`) is public, so those counts can be inspected but not re-derived. The Gemini and OpenAI pilots were
   converted from earlier pilot scripts that are not in the repository, and the native-template Llama check is not bundled.

**Version labels.** Labels such as v0.7 to v0.15 in the docs and in some file comments name **internal research iterations** (pilots and
follow-up runs; the internal "v0.7" frontier pilot is unrelated to the 0.7.0 release). They are not published package versions. The released versions are 0.2.0 to 0.7.0 (see `CHANGELOG.md` and the `RELEASE_NOTES_v*.md` files; the
0.1.x entries there are internal); the current package version is 0.7.0, as in [What is in v0.7.0](#what-is-in-v070).

## Local and hosted commands

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
first four-mapping experiment is in [`docs/FOUR_MAPPING_SUMMARY.md`](docs/FOUR_MAPPING_SUMMARY.md) and summarized below.

### `bleediff gemini`, `bleediff openai` and `bleediff anthropic`: hosted answer-only runs

Some hosted models return an answer but **no usable A-D probabilities**: every tested Gemini model rejects `responseLogprobs`, and the tested OpenAI model (GPT-4.1 mini) returned only a top-20 list that held all four letters in 17.5% of its responses. the Anthropic Messages API has no logprobs parameter at all. `bleediff gemini` (Gemini `generateContent`), `bleediff openai` (Chat Completions) and `bleediff anthropic` (Claude Messages) measure such models
as **`ANSWER_ONLY`**: it sends the prompts of a frozen dataset, parses each reply as one letter A-D, tracks the correct versus the competing answer across
answer-letter permutations, and reports correct / competing / other-incorrect answers, newly induced incorrect answers, answer flips and agreement between repeated
passes. It never produces probabilities, log-odds margins, confidence or a leaderboard (existing `DIRECT_PROBABILITY` measurements are unchanged). Full guide: [`docs/HOSTED_MODELS.md`](docs/HOSTED_MODELS.md).

**Four distinctions to keep in mind when reading hosted results**

* **`DIRECT_PROBABILITY` versus `ANSWER_ONLY`.** Local runs measure the probabilities of A-D; hosted runs observe only the letter returned. The two are reported separately and never merged; no hosted result has a margin, confidence or rank.
* **Provider refusals versus incorrect answers.** A refusal (for example Anthropic `stop_reason: refusal`) is a provider decision to return nothing. It is a non-answer, never counted as incorrect or competing, and its category label is provider metadata, not a measured cause.
* **Strict parsing versus secondary diagnostics.** The primary parse accepts only a reply that is exactly one letter A-D; prose, `**B**` and empty replies are `invalid`/`empty` non-answers. Lenient readings (for example "last line is one letter") are separate, labelled and post-hoc, and never replace the strict result.
* **Estimated budget enforcement versus provider billing.** `--budget` caps an estimate computed from the prices you supply and the usage reported, with a worst-case reserve checked before every request. It is not a guaranteed billing limit: every retry must fit under the remaining estimate as one more worst-case attempt, and a failed attempt that may have been billed (network error, HTTP 5xx) is charged at the worst-case bound, but caching, price tiers and the provider's real billing are not modelled.

**Strict scoring versus FLR diagnostics (worked example).** The primary result is always the strict parse. FLR is an optional, post-hoc, labelled
diagnostic on replies the strict parse rejected as `invalid`; neither version changes a primary classification or the strict columns.

* **FLR-v1** (historical Final-Line Recovery): take the last non-empty line, strip markdown emphasis, accept one letter A-D. Used in the original reconciliation notes and tables (`research/reconcile_evidence.py`, unchanged).
* **FLR-v2** (conflict-aware): the same, but if standalone letters on different lines disagree it reports a conflict and picks nothing. It is what `--replay` and the Atlas print. On the 61 format-invalid replies in the saved Haiku and Sonnet sessions (12 refusals excluded) v1 and v2 extract the same letter every time; none of those replies has conflicting letters, so the difference is not exercised by this data.
* Neither applies to provider refusals, and neither produces a probability or confidence.

The prompt says "Reply with exactly one letter". Claude Sonnet 5.5 instead wrote visible prose that ends in a letter in 60 of 160 requests,
including all 32 COMPETING requests. Those are **format failures, not answers**: the strict parse scores them `invalid`, so the strict COMPETING
row reads 0 correct / 0 competing / 0 other / 32 no answer. That does **not** mean Sonnet resisted the competing evidence, only that none of those
replies followed the format. FLR-v2 reads the letter each reply ended on and reports 2 correct / 30 competing for those 32 replies (the same totals as GPT-4.1 mini); it
describes the letter Sonnet settled on, was defined after the replies were read, and credits replies the prompt forbade. Haiku's 12 refusals are a different kind of non-answer and stay
refusals. Details: [`docs/HOSTED_MODELS.md`](docs/HOSTED_MODELS.md) and [`examples/hosted-comparison/README.md`](examples/hosted-comparison/README.md).

```
# offline, free: replay saved evidence (the v0.8 pilot, converted; see examples/gemini-answer-only/README.md)
python -m bleediff gemini --replay examples/gemini-answer-only/flash
python -m bleediff openai --replay examples/openai-answer-only/gpt41mini      # the saved GPT-4.1 mini pilot (examples/openai-answer-only/README.md)
python -m bleediff atlas --manifest examples/openai-answer-only/atlas-input.json --out runs/hosted-atlas   # local + Gemini + OpenAI, measurement types kept apart

# live (sends requests with your key and may cost money): the key is read from the GEMINI_API_KEY environment variable only
python -m bleediff gemini --prompts examples/gemini-answer-only/prompts.json --model gemini-3.8-flash --out runs/my-gemini-run \
    --passes 2 --budget 1.00 --price-in 0.75 --price-out 3.75 --thinking-level low
# Claude: the key comes from the ANTHROPIC_API_KEY environment variable only (Claude 5.x thinks adaptively: max_tokens caps thinking + text, so keep it generous)
python -m bleediff anthropic --prompts examples/gemini-answer-only/prompts.json --model claude-haiku-5-5 --out runs/my-claude-run \
    --passes 2 --budget 0.25 --price-in 0.10 --price-out 0.50 --effort low --max-output-tokens 1024
# OpenAI: the key comes from the OPENAI_API_KEY environment variable only
python -m bleediff openai --prompts examples/openai-answer-only/prompts.json --model gpt-4.1-mini --out runs/my-openai-run \
    --passes 2 --budget 1.00 --price-in 0.40 --price-out 1.60
```

A live run needs an explicit `--budget` (USD) and list prices (`--price-in` / `--price-out`, USD per 1M tokens; there are no built-in prices). **The budget is a cap on an
ESTIMATE, not on the provider's bill**: it uses the prices you supply and the usage the provider reports (cached-token discounts, tiers and later price changes are not modelled). Before
every request the worst case of the NEXT request must fit under the remaining budget: the prompt (one token per UTF-8 byte, an upper bound) at the input price plus a reserve for
output at the output price. The reserve is `max_tokens` for Anthropic (a hard cap on thinking plus text), `max_completion_tokens` for OpenAI (it includes reasoning tokens), and `maxOutputTokens` plus a thinking allowance for Gemini (the explicit
`--thinking-budget`, or another full cap when none is set); it grows to twice the largest output seen so far. Reasoning and thinking tokens are billed as output (OpenAI's
`completion_tokens` already contains them, so they are not counted twice). Missing or inconsistent usage metadata is charged at the worst-case bound and flagged, a response that costs more
than the bound it was admitted under stops the run (`COST_BOUND_EXCEEDED`), and retries are bounded (`--max-retries`, default 2) and budget-admitted: a retry is sent only if one more full worst-case attempt fits under the remaining estimate, and attempts that may have been billed (transport errors, HTTP 5xx; not 429 or other 4xx) are charged at the worst-case bound. Only parameters the selected model supports are sent
(for OpenAI: `--temperature`/`--top-p` only for `gpt-4*`, `--reasoning-effort` only for reasoning models; for Claude: `--effort`, `--thinking`, never `--temperature` with `--top-p`; others are refused before any request). A real probe showed that `claude-haiku-5-5` with `max_tokens` 8 spends all of it on thinking and returns no text; such a reply is recorded as an empty non-answer, never guessed. `examples/hosted-comparison/` is the five-model comparison (Claude Haiku and Sonnet, Gemini Flash and Carbon, GPT-4.1 mini), replayable offline.  The session directory (`session.json`, a copy of the prompts, one record per request with the request configuration,
full response with opaque thinking signatures and any reasoning text replaced by placeholders, usage and cost, never the key) is new each time and can be replayed offline. The Atlas accepts such sessions in an optional `answer_only_sessions` list and
reports them in separate `ANSWER_ONLY` sections and CSVs next to the `DIRECT_PROBABILITY` results (`examples/gemini-answer-only/atlas-input.json`).

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

## Exploratory findings in v0.4.0: four answer-letter mappings

Six synthetic questions, each under four letter assignments (correct answer at A, B, C or D, with the competing answer at B, A, D or C),
on Qwen2.5-1.5B and Llama 3.2 3B Instruct (Q4_K_M, CPU, raw prompts); 24 cells per model. Four of the 24 possible permutations; repeats
were bit-identical and are not independent samples. Recompute everything with `examples/four-mapping/verify_summary.py`.

* **Llama 3B (tested configuration): semantic-answer sensitivity.** Competing evidence moved the decision toward the competing answer in
  24 of 24 cells, exceeded neutral text in 24 of 24, and repeating it added shift in 23 of 24. The sign did not depend on the letter; the
  size did (mean +3.99, +2.64, +2.52, +3.14 nats for the correct answer at A, B, C, D). Newly induced incorrect answers occurred only with
  the correct answer at A (5 of 18) or D (3 of 18).
* **Qwen 1.5B (tested configuration): strong answer-letter sensitivity.** The shift reversed with the mapping (+1.18, -1.17, -0.70,
  +0.92) and was positive under all four mappings for 0 of 6 questions. The baseline p(correct) was 0.51, 0.23, 0.09, 0.59 for A, B, C, D;
  competing evidence raised letter B and lowered letter D under every mapping. Newly induced incorrect answers over the five m1/m2
  treatments: 9 under m1, 1 under m2.
* **Not established:** a model-family mechanism (one model per family, one quantization), a statistical effect size, position effects
  separated from baseline confidence, or any real tenant-isolation vulnerability.

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
* `bleediff/roles.py` (v0.4.0) the semantic answer-role scoring used by `track --arms` and the Atlas.
* `examples/atlas/` small sanitized example datasets (saved batch sessions, candidate files, fixtures, atlas input manifests) that
  reproduce the v0.3.0 findings offline, and `mention_contrasts.py`.
* `bleediff/answers.py`, `bleediff/hosted.py`, `bleediff/gemini.py`, `bleediff/openai_chat.py`, `bleediff/anthropic_messages.py` answer-only analysis, the shared hosted engine and the three provider adapters (v0.5.0).
* `examples/gemini-answer-only/`, `examples/openai-answer-only/` the Gemini and GPT-4.1 mini pilots as replayable ANSWER_ONLY bundles; `examples/hosted-comparison/` the five-model comparison (Claude Haiku and Sonnet sessions, results table, `verify_comparison.py`).
* `bleediff/telemetry.py`, `bleediff/behavior.py`, `docs/BEHAVIORAL_TELEMETRY.md`, `docs/ARCHITECTURE_V07_BEHAVIOR.md`, `examples/behavior/` offline behavioral telemetry (v0.7.0): the measurement schema and adapters, ingestion and reporting, and the Research 002 compatibility check.
* `bleediff/experiment.py`, `docs/FIRST_EXPERIMENT.md`, `examples/experiment-multi-question/` experiment authoring (v0.6.0): spec validation, prompt-dataset generation and budget estimates; the multi-question example has per-question conditions.
* `docs/HOSTED_MODELS.md` the hosted-model usage guide; `docs/EVIDENCE_RECONCILIATION.md`, `docs/CASE_NOTE_HOSTED_REFUSALS_AND_FORMAT.md` and `docs/evidence/` the reconciliation of the exploratory Claude runs (v0.10-v0.15), a short case note and a per-request index; `research/reconcile_evidence.py` regenerates the index.
* `examples/four-mapping/` the sanitized four-mapping evidence bundle (48 sessions, SHA-256 list) and `verify_summary.py` (v0.4.0).
* `fixtures/` scenario fixtures, preserved controls and the protocol and preregistration documents.
* `docs/TECHNICAL.md` the detailed method and history; `CHANGELOG.md`; `RELEASE_NOTES_v0.2.0.md`, `RELEASE_NOTES_v0.3.0.md`, `RELEASE_NOTES_v0.4.0.md`, `RELEASE_NOTES_v0.5.0.md`, `RELEASE_NOTES_v0.6.0.md`, `RELEASE_NOTES_v0.7.0.md`; `docs/FOUR_MAPPING_SUMMARY.md`.
* `tests/` stdlib `unittest` suite (a mock llama-server stands in for the model).

Local run directories (`runs/`), model files and per-run sidecars are git-ignored and are not part of the repository.

## Git identity and license

The repository is configured with a repo-local, non-personal author (`bleediff <noreply@bleediff.invalid>`), and a test fails
if a file contains a personal Gmail address. MIT license.
