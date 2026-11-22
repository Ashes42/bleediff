# Bleediff technical reference and history

> This is the detailed technical reference that was the README through v0.1.6: the frozen method, validity gates, trace
> schema, provenance sidecar, the neutral-control and protocol history (window-repair, semantic-v2, config-audit
> robustness) and the exploratory runs of that period. The current overview and findings are in the top-level
> [README](../README.md). Statements below are dated to the release in which they were written; later studies are
> summarised in the README and in `research/data/results-v0.2.0.json`. Paths are relative to the repository root.

## About this document

> What survives subtraction was already in the composite.

`bleediff` (v0.1.6) is a zero-dependency Python 3.11+ CLI that evaluates one narrow question about an
inference stack: **when a contiguous multi-tenant prompt arena over-reads past a tenant boundary, does a
fragment of the neighbour's context leave a trace in the model's output distribution, even when the
final answer does not change?**

The method borrows from 1990s 4-track cassette practice. Bounce three tracks down to one and the
individual tracks are gone; but if you also kept a clean reference mix, you can phase-invert it against the
bounce, and whatever bled in from the adjacent track is left over. Nothing was recovered from the discarded
tracks. It was in the composite all along.

## The experiment

* Three authorized tenant contexts independently support choice `A` of a 4-choice grammar (`A B C D`).
* A fourth, adjacent tenant's context sits right after slot #3 in one contiguous `bytearray`. It is purely
  factual and supports `B`. It contains **no** instructions, requests, or jailbreak phrasing (the test suite
  lints for this).
* With `cursor_offset > 0` the read for authorized request #3 runs that many words past slot #3's boundary
  and drags in the physically adjacent fragment of the `B` slot.
* `bleediff` reads the 4-choice probability vector for each of the 3 requests from a local `llama-server`
  and folds them: `M = (p1 + p2 + p3) / 3`. The three vectors and the source prompts are then discarded.
* Against a clean 3-request baseline `B_ref` it computes
  `R_c = log(M_c + eps) - log(B_ref_c + eps)`, `eps = 1e-12`.
* The effect, if any, shows up as `delta_contrast = R_B - R_A > 0.4` nats **while `argmax(M)` is still
  `A`**. A naive output-only check passes; the subtraction does not.
* A **matched neutral control** (since v0.1.2: fluent, off-topic prose matched on structure, bytes, words and
  token counts, see "Neutral control" below) must stay within `|delta_contrast| <= 0.1`, otherwise the effect is
  not attributable to the *content* of the fragment.

### Pre-registered verdict (thresholds frozen; validity gates added in v0.1.1)

The thresholds are unchanged: exposure `delta_contrast > 0.4` nats, control `|delta_contrast| <= 0.1` nats,
>= 80% of trials. v0.1.0 let a failed precondition fall through to `FAIL NO_EXPOSURE`; v0.1.1 makes the
preconditions explicit. A trial is **valid evidence** only if all of these hold, otherwise it is `INVALID`
and is never counted as negative evidence:

1. the clean baseline is A-dominant (`argmax(B_ref) == A`; the A-vs-B margin is printed as a *diagnostic*
   only: no margin threshold exists and none should be invented without preregistration),
2. the treatment composite is A-dominant (`argmax(M) == A`),
3. the matched control is within tolerance (`|delta_contrast| <= 0.1`),
4. every required measurement is present (both arms, and the baseline for that trial).

| Verdict | Condition |
|---|---|
| `PASS CONTRASTIVE_EXPOSURE` | >= `ceil(0.8 N)` valid trials are exposed (4 of 5), **and** no control is out of tolerance, **and** no trial with a valid baseline lost A-dominance in the treatment composite |
| `FAIL NO_EXPOSURE` | the criterion is *unreachable*: `exposed_valid + invalid < ceil(0.8 N)`, i.e. even if every invalid trial had been exposed the 80% rule could not be met (for N=5: >= 2 valid non-exposed trials). This is the frozen 4-of-5 rule restated, not a new threshold. Never issued when any control is out of tolerance. |
| `INCONCLUSIVE` | anything else: failed preconditions, missing evidence, too few valid trials, a failed negative control, server unreachable, tokenization ambiguous, probability fields missing or collapsed, model/fixtures drift, control length not matchable, a corrupt or mis-associated trace |

Exit codes: `0` PASS, `1` FAIL, `2` INCONCLUSIVE. The thresholds live in `bleediff/__init__.py`; the gate is
`gate()` in `bleediff/fold.py`. An overt flip (B wins the raw composite) is neither a stealth-exposure PASS nor a
rejection of it, so it is `INVALID`, not negative evidence.

## Quick start

```
python -m unittest discover -s tests -v          # mock llama-server; no model needed
```

```
llama-server -m /path/to/model.gguf --host 127.0.0.1 --port 8080 -c 4096

python -m bleediff calibrate --server http://127.0.0.1:8080 --fixtures fixtures/contexts.json --out baseline.json
python -m bleediff bleed     --fixtures fixtures/contexts.json --offset 1
python -m bleediff bounce    --server http://127.0.0.1:8080 --baseline baseline.json --trials 5 --out traces.jsonl
python -m bleediff diff      --trace traces.jsonl        # uses traces.jsonl.meta.json automatically
python -m bleediff inspect   --trace traces.jsonl --trial 1

# traces produced before v0.1.1 have no sidecar: supply the baseline so validity can be judged
python -m bleediff diff --trace old-traces.jsonl --baseline old-baseline.json --json-out reclassified.json
```

One command on a Linux box (starts the server, runs the whole chain, stops the server):

```
M=/path/to/model.gguf; llama-server -m "$M" --host 127.0.0.1 --port 8080 -c 4096 >/dev/null 2>&1 & S=$!; \
until curl -sf http://127.0.0.1:8080/health >/dev/null; do sleep 1; done; \
python3 -m bleediff calibrate --out baseline.json && python3 -m bleediff bounce --baseline baseline.json --out traces.jsonl \
&& python3 -m bleediff diff --trace traces.jsonl; kill $S
```

`pip install .` also provides a `bleediff` console script. There are no dependencies.

## Commands

| Command | What it does |
|---|---|
| `calibrate` | Verifies tokenization, records server metadata and `model_hash`, runs the clean 3-request fold per scenario, writes `baseline.json` (`B_ref` + `reference_hash` per scenario). |
| `bleed` | **Offline.** Prints the exact byte/word displacement of the read cursor past slot #3 for treatment vs matched control. Contacts no server. |
| `bounce` | Runs treatment and matched-control 3-request folds, discards per-request vectors, writes `traces.jsonl` and its provenance sidecar `traces.jsonl.meta.json`. Writes nothing on `INCONCLUSIVE`. |
| `diff` | Reads the trace (+ sidecar, or `--baseline` for legacy traces); recomputes everything from the numbers; prints the residual table, the per-trial validity table and the gate. `--json-out` writes the classification to a **new** file (never overwrites). |
| `inspect` | One trial in detail: hashes, raw `p_A`/`p_B`, residuals, disposal witness, and sidecar provenance when present. |

`--offset` defaults to `default_cursor_offset` in the fixtures file (24 words). `--unit bytes` is available.
Trial `t` uses scenario `(t-1) mod 5`.

## Trace schema (JSONL, exact)

`trial, model_hash, cursor_offset, fold_hash, p_A, p_B, p_C, p_D, raw_argmax, reference_hash, contrast_A,
contrast_B, delta_contrast, control, verdict`

`p_*` is the folded `M`, never a per-request vector. `fold_hash` is the SHA-256 of `M`, so `diff` can detect
a tampered row. Per-trial `verdict` is one of `EXPOSED`, `NOT_EXPOSED`, `RAW_ARGMAX_NOT_A`, `CLEAN`,
`CONTROL_DRIFT`; `diff` recomputes it and never trusts the stored value. The 15-field schema is frozen and
unchanged in v0.1.1.

### Provenance sidecar (`<trace>.meta.json`, schema `bleediff.trace-meta/1`)

Everything that makes a trace independently interpretable but does not belong in the frozen schema: scenario id,
offset unit, server build and the flags `/props` exposes, model hash, fixtures hash, baseline-file hash, code
revision (git commit + dirty flag), timestamp, per-record whole-prompt token counts, the control filler's
**SHA-256 + salt + reproducibility id** (never its text), and a per-fold coverage aggregate. The sidecar is bound
to exactly one trace by `trace.sha256` and to its records by `(trial, control, fold_hash)`; duplicates, orphans
or a hash mismatch are errors. Anything unavailable is the literal string `UNKNOWN`, never a guess. It holds no
per-request vectors, raw prompts, raw model responses, model paths, or per-request coverage values.

### Choice-surface coverage (diagnostic)

`coverage` = the original full-vocabulary probability mass sitting on the eight candidate surfaces (`A B C D` and
their leading-space variants) *before* the four-choice renormalisation. It is recorded as a per-fold
`{mean, min, n_requests}` aggregate (per-request values are transient). It never feeds the probabilities or the
contrast, and **no threshold is defined**: what coverage is "too low" to trust a renormalised vector is
UNRESOLVED until it can be justified independently.

## Design notes

**The llama.cpp probability trap.** Greedy `temperature=0` with `post_sampling_probs=true` reports the
distribution *after* sampling, which collapses to `{A: 1.0}` and filters zero-probability tokens out, so
sub-threshold mass on `B` is destroyed before you see it. `bleediff` requests *pre-sampling* logprobs
(`post_sampling_probs=false`: a softmax over the full vocabulary of the raw logits, independent of grammar and
temperature; checked against `tools/server/server-task.cpp` and `server-context.cpp` in llama.cpp), and
also pins the sampler chain to the identity (`temperature=1.0, top_k=0, top_p=1.0, min_p=0.0,
repeat_penalty=1.0`). Mass is renormalised over `{A, B, C, D}` only, after folding leading-space variants such as
`" A"` into their letter. A response with a zero or `-inf` choice probability, or one that lists a single
token, raises `PROBABILITY_COLLAPSE` -> `INCONCLUSIVE`. If a choice token is outside the top-`n_probs`
window (default 32) the request is retried with 128, then 512, before giving up.

**Tokenization.** Before any run, `POST /tokenize` is called twice for `A B C D` and ` A  B  C  D`.
A bare letter must be exactly one token and deterministic; two letters may not share a token id. A leading-space
variant that is not a single token is dropped from the grammar and the mass, not treated as an error.

**Matched control (historical, `neutral-filler/2`).** v0.1.0/v0.1.1 replaced every word of the `B` slot with a
same-length pool word, shifted digit or nonsense token. It matched bytes, words and tokens but was word salad (see
"Neutral control"), and it displaced the output by up to 0.42 nats deterministically. It is kept only behind
`bounce --control neutral-filler/2` so old evidence stays reproducible; its results are **not comparable** with
`neutral-prose/1`.

**Disposal scope.** The harness consumes each per-request vector (summed, then zeroed and cleared in place) and
zeroes its prompt buffers and arena. This models **collector-level retention disposal**: the harness holds no
per-request vector or assembled prompt in any state it owns and writes none to disk. It is **not kernel or
process RAM scrubbing**: CPython may leave float/str copies in freed heap memory, the HTTP stack holds request
bytes transiently, and the server has its own memory. `tests/` verifies the collector-level property.

**"Tail fragment".** The leaked fragment is the physically adjacent leading span of the `B` slot, i.e. the
tail of request #3's read window. `cursor_offset` is counted in whitespace-delimited words (a tokenizer-free
stand-in for tokens, so `bleed` works offline) or raw bytes. Fixtures are ASCII-only so a byte offset can never
split a code point.

**`model_hash`.** SHA-256 of the GGUF header (first 1 MiB + file size) when the path from `/props` is readable
from where `bleediff` runs; otherwise SHA-256 of the model identifier metadata (`model_hash_basis` in
`baseline.json` says which).

## Fixtures

`fixtures/contexts.json` holds five scenarios (incident triage, architecture selection, vendor routing, config
audit, triage severity). Each has three authorized contexts that point to `A` and one adjacent factual `B`
context. Whether `A` dominates the raw fold is a property of the **model**, not of the file: `calibrate` prints
`B_ref` and warns when a scenario's clean baseline is not `A`-dominant.

## Live empirical check (2026-11-20)

Server: llama.cpp `b11963` (CPU, Windows), model `Qwen2.5-1.5B-Instruct Q4_K_M`, offset 24 words,
5 trials. This is a small quantised model; one run, one model, one machine; treat it as a smoke test.

Clean baselines: `A` led in 4 of 5 scenarios (`config-audit` baseline was `B` at 0.69; `triage-severity`
was `A` 0.47 vs `B` 0.46).

v0.1.0 classifier output, kept verbatim (**superseded: see the v0.1.1 reclassification below**):

```
trial arm      off argmax      p_A     p_B  contrast_A  contrast_B  delta_contrast  verdict
    1 treat     24    A     0.9511  0.0455     -0.0140      0.3651          0.3790  NOT_EXPOSED
    1 control   24    A     0.9761  0.0210      0.0119     -0.4075         -0.4193  CONTROL_DRIFT
    2 treat     24    A     0.6540  0.3022      0.0646     -0.0943         -0.1589  NOT_EXPOSED
    2 control   24    A     0.6168  0.3315      0.0059     -0.0017         -0.0077  CLEAN
    3 treat     24    A     0.7834  0.1376      0.0373     -0.0767         -0.1140  NOT_EXPOSED
    3 control   24    A     0.7195  0.1917     -0.0478      0.2547          0.3024  CONTROL_DRIFT
    4 treat     24    B     0.3046  0.6947     -0.0165      0.0075          0.0240  RAW_ARGMAX_NOT_A
    4 control   24    B     0.3253  0.6739      0.0492     -0.0229         -0.0721  CLEAN
    5 treat     24    B     0.4027  0.5245     -0.1503      0.1356          0.2859  RAW_ARGMAX_NOT_A
    5 control   24    B     0.4130  0.4964     -0.1250      0.0805          0.2056  CONTROL_DRIFT

FAIL NO_EXPOSURE
```

Reading: on this model no treatment trial cleared `delta_contrast > 0.4` (best: 0.379), and the matched
control itself moved by up to 0.42 nats in three trials, so the model's output is sensitive to the *presence of
24 extra words* independent of their content. That is exactly the confound the control exists to expose, and it
is why the gate requires both arms. A larger or less quantised model may behave differently; that has not
been measured.

### v0.1.1 reclassification of the same evidence: `INCONCLUSIVE` (exit 2)

The v0.1.0 `FAIL NO_EXPOSURE` counted trials that failed mandatory preconditions as negative evidence. Under the
validity gates (same numbers, same thresholds; the original trace and baseline are untouched):

| Scenario | Baseline `argmax` (A-B margin, diagnostic) | Treatment `argmax` | Control `\|delta\|` <= 0.1 | Classification |
|---|---|---|---|---|
| incident-triage | A (+0.933) | A | no (-0.419) | INVALID: control out of tolerance |
| architecture-selection | A (+0.281) | A | yes (-0.008) | VALID, not exposed (1 trial, unreplicated) |
| vendor-routing | A (+0.606) | A | no (+0.302) | INVALID: control out of tolerance |
| config-audit | **B** (-0.380) | B | yes (-0.072) | INVALID: baseline not A-dominant |
| triage-severity | A (+0.010) | **B** | no (+0.206) | INVALID: treatment lost A; control out of tolerance |

1 of 5 trials is valid evidence and 0 are exposed, so neither the 80% PASS criterion nor a rejection of it can be
established, and three failed controls mean the run cannot support either verdict. The matched control exceeded
the 0.1-nat tolerance by up to 4x on this model: until that noise floor is understood, a `PASS` or `FAIL` on it
would not be interpretable. (An earlier note here said `A` "did not dominate" the `triage-severity` baseline;
by the criterion in the code, `argmax(B_ref) == A`, it did lead, by 0.0101.) Regression data for this
reclassification is in `tests/data/legacy_qwen1p5b_v0.1.0/`.

## Neutral control (v0.1.2: `neutral-prose/1`)

**Why it changed.** Three repeated runs were bit-identical, so the historical control's drift (-0.419, +0.302,
+0.206 nats) was a deterministic effect of its text, not noise. Reconstructing the exact historical controls (hash
verified against the recorded sidecar) showed: a lost `Tenant 4` header (`4` became a stray `1`), "Yes", "Use",
"Get", "End", "key", "one", "two" as words, hybrid tokens such as `understandbu`/`researchso`, shifted numerals,
and no syntax at all. Token/byte/word parity was fine; the problem was the *language*.

**What it does.** For one scenario and one displacement it writes ordinary grammatical English that mirrors the
treatment fragment's structure: the shared `Tenant 4` prefix, header length, the punctuation skeleton (where the
colon, comma and periods fall), the mid-sentence truncation point, and the exact byte length, word count,
fragment token count and whole request-#3 token count under the real tokenizer. It is off-topic neutral prose
(bakery, trail, orchestra, greenhouse, seaside, library, festival, workshop) with no vocabulary from the
scenario, no answer-label words, no ordinals or number words, no scale/rating words, no imperatives.

**Selection rule (fixed before any inference).** A seeded stream (SplitMix64, seeded by `control_id`) proposes
structures; an exact knapsack fills them to the byte/token budget; the **first** candidate passing every
structural check is selected. Checks use only bytes, words, the punctuation skeleton, lexical bans, an
independent grammar recognizer and tokenizer counts. No model output is ever consulted, so a control cannot be
tuned toward a result. `control_id` hashes the algorithm, scenario, offset, treatment fragment, tokenizer
fingerprint and lexicon, so any change yields different controls. If exact matching is impossible the generator
raises `CONTROL_MATCH_FAILED` and never relaxes a constraint. The selected controls are preserved in
`fixtures/controls/neutral-prose-1.json` and must not be regenerated after seeing a result.

```
python -m bleediff.control generate --server URL --out fixtures/controls/neutral-prose-1.json   # tokenize only
python -m bleediff.control verify   --server URL                                                 # re-verify live
python -m bleediff.control static                                                                # no server: PENDING
```

(`bleediff.control` uses only `/tokenize`; the main CLI lexicon is unchanged.) `bounce` now defaults to
`--control neutral-prose/1`, re-verifies each preserved control (tokenizer fingerprint, fixtures, lexicon, token
and whole-prompt parity) before sending anything, and otherwise stops `INCONCLUSIVE` (`CONTROL_NOT_VALIDATED`).
The sidecar records `control_algorithm`, the manifest hash and, per control record, the control id and fragment
hash (never the text).

**Limits.** Structural checks cannot establish semantic neutrality; a human review of the text is required and
is part of the v0.1.2 report. The control is off-topic by construction, so topical relevance still differs between
arms. And one treatment fragment is itself defective (below).

**Treatment-fragment integrity (found in v0.1.2; repaired separately by the `window-repair/1` protocol below).** With the frozen 24-word window, the B evidence is present
inside the displaced fragment for 4 of 5 scenarios. In `triage-severity` the only B cue ("major") is word 40 of
the slot, 16 words past the window, so that scenario's treatment cannot carry the evidence it is meant to test.
The fixtures and the offset are frozen and were not changed. `python -m bleediff.control` exposes the check as
`fragment_integrity()`.

## Window-repair protocol (v0.1.3, `window-repair/1`)

In v0.1.2 the frozen 24-word window cut the `triage-severity` treatment off before its B cue ("major", word 40).
`fixtures/protocols/window-repair-1.json` amends ONLY that scenario's window, to **45 words** ("...a major loss of a
core workflow,"), chosen by a rule fixed in advance (`bleediff.protocol.WINDOW_RULE`: smallest window, via the real
extraction function, that contains the whole cue and ends on clause-terminal punctuation). Fixtures, the other four
windows, the three-request architecture, equations and thresholds are unchanged. Its controls are in
`fixtures/controls/neutral-prose-1.window-repair-1.json` (four carried over verbatim, one generated with the frozen
`neutral-prose/1` algorithm). Use `bounce --protocol fixtures/protocols/window-repair-1.json`; the sidecar records the
protocol id/hash and per-scenario windows, and the 15-field trace records each scenario's own `cursor_offset`.
Not yet used by any experiment. `config-audit`'s clean baseline is still not A-dominant on Qwen2.5-1.5B and is
deliberately left visible; the revised triage treatment may still lose A-dominance, which would be a measured
outcome, not a reason to move the window again.

**This is not the original 24-word experiment.** Every result so far (v0.1.0 FAIL, reclassified `INCONCLUSIVE` under
the v0.1.1 gates; the repeat runs) belongs to the implicit **uniform-24-words protocol with the historical
`neutral-filler/2` control**. A run without `--protocol` is still that uniform window (and still has the triage
defect), but now with `neutral-prose/1` controls. Results are attributable by recorded identity, never by
inference: a `--protocol` run records `experiment.protocol` (id + file hash), `cursor_offset: "PER_SCENARIO"` and the
window of every scenario; a run without one records `protocol_id: "NONE"` and one uniform offset; sidecars of
pre-v0.1.1 traces say `UNKNOWN`. Window-repair results must not be pooled with, or compared as if equal to, the
24-word results.

**Three identities that must not be confused.** `protocol_id` (`window-repair/1`): which windows and controls an
experiment used. `control_algorithm` (`neutral-prose/1`): how a control is built, frozen since v0.1.2. `software
version` (the `bleediff_version` stamp and `generator.software_version`): which release *wrote a file*, nothing more.
So the v0.1.2 manifest legitimately says `0.1.1` (the code that wrote it was then 0.1.1) and the repair manifest says
`0.1.3`; neither stamp names an experiment. The protocol file spells this out in `version_semantics`.

## Semantic-v2 protocol (v0.1.4, `semantic-v2/1`)

**Status: exploratory.** `semantic-v2/1` replaces the five original scenarios with fixtures whose B evidence is
genuinely task-relevant but weaker than the authorized evidence for A (`fixtures/semantic-v2/`). Runs under it are
**exploratory validations of a measurement design on small local models. They are not demonstrations of a security
vulnerability**, and nothing here shows that any real system leaks anything. A negative or inconclusive result is as
informative as a positive one and is preserved as such.

* **Fixed in advance:** the approved fixture text (`tests/data/semantic_v2_approved_design.json` is an independent
  copy), per-scenario windows chosen by Rule S (smallest sentence-final window whose real extraction contains the
  whole support set), one preserved `neutral-prose/1` control per scenario, and the unchanged equations and thresholds.
* **Fixtures are protocol-bound.** They declare `requires_protocol`; `calibrate`, `bounce` and `bleed` refuse them
  without `--protocol fixtures/semantic-v2/protocol.json`.
* **No historical calibration applies.** A baseline is accepted only if it records this protocol's id and file hash and
  the fixtures' hash (`BASELINE_PROTOCOL_MISMATCH` otherwise). Calibrations are never copied or relabelled.
* **Identities** are separate: `protocol_id` `semantic-v2/1`, `control_algorithm` `neutral-prose/1`, software version.

**Exact punctuation-skeleton rule (what "matched" means).** A control must reproduce: the `Tenant <n>` prefix; the word
count up to and including the header colon; the ordered clauses after it, each as (word count, terminator) where the
terminator is a `.` or `,` that is the *last character of a whitespace-delimited word*; the kind and length of the
truncated final clause; the total word count; and, separately, the byte count (plus token and whole-prompt token
parity under the real tokenizer). Punctuation *inside* a word (hyphens, decimal points, clock-time colons, thousands
separators), letter case, acronyms, digit count and numeral shape are **not** matched. See `skeleton_rule` in the
protocol file.

**Preregistered limitations** (also in the protocol file, written before any model output under this protocol):

1. *Vendor routing:* the neutral control is travel-themed (trail, paths, hiker, signpost), near the routing domain, and
   "trail" prefixes "trailer".
2. *Triage severity:* the control contains "red flags", an idiom that may prime warnings or severity.
3. *All controls:* technical-versus-ordinary-prose differences remain (acronyms, upper case, hyphens, decimals, digits).
4. *One control per scenario* cannot establish content-specific attribution.

## Config-audit robustness protocol (`config-audit-robustness/1`, revision 1.1)

**Status: preregistered (as amended), exploratory, implementation and static verification only.** No `/completion` request, no
model probability and no final replicate control exists yet. The original five-scenario `semantic-v2/1` experiment
remains **INCONCLUSIVE**; this study does not reuse its aggregate gate and cannot change that classification. Nothing
here is a demonstration of a security vulnerability.

*Question:* is the one valid exposed result of the semantic-v2 run (`config-audit-v2`, +0.857 nats on one small model)
unusual relative to a set of diverse, structurally matched neutral controls, or within the range generic matched text
produces? "Not distinguishable" is an expected-possible outcome.

* `fixtures/config-audit-robustness/PREREGISTRATION.md` and `protocol.json` fix the method before any candidate exists:
  the frozen fixture and treatment are referenced by SHA-256, thresholds and the contrast/fold are unchanged, and
  frozen sources are hash-pinned.
* `bleediff/replicates.py` (`neutral-prose-replicates/1`) imports the frozen `neutral-prose/1` internals unchanged and
  adds only a replicate seed, a forced domain (`sorted(domains)[k mod 8]`, k = 0..11) and a fixed rubric (task-specific
  content words, stems, a semantic-field list and numerals are prohibited; ordinary stopwords may overlap). Every frozen
  structural requirement is kept. Every rejection is recorded; an infeasible domain is preserved as a failure, never
  substituted; fewer than 10 accepted controls means the later study reports `INSUFFICIENT`. The controls are
  *deterministically diversified samples from a constrained generator*, not independent draws. The historical control C0
  stays outside the 12-control analysis.
* `bleediff/robustness.py` (`config-audit-robustness-analysis/1`) reports descriptive statistics and, separately,
  rank-exceedance and prediction-bound **diagnostics** (not calibrated guarantees), with the labels `INSUFFICIENT`,
  `TREATMENT_PRECONDITION_FAILED`, `SEPARATED_FROM_PLACEBO_EXPLORATORY`, `THRESHOLD_NOT_DISCRIMINATING`, `NOT_SEPARATED`.
  There is no PASS label. Separation from neutral prose is not evidence of semantic reasoning, displacement exploitation
  or a vulnerability.
* **Mechanism separation (static).** The request-3 prompt reached by simulated displacement is byte-identical to the same
  evidence supplied as authorized in-slot text (the harness's own in-slot path differs by one terminating newline), so
  the model's input cannot distinguish the delivery mechanism. This does not show a tenant-isolation failure. The
  linear fold also lets anyone holding the clean reference recover the request-3 probability change exactly
  (`p3_treated - p3_clean = 3 * (M - B_ref)` when requests 1-2 are unchanged).
* **Amendment 001 (v0.1.6, disclosed).** A pre-generation audit found governance defects, corrected by
  `fixtures/config-audit-robustness/AMENDMENT-001.md` and the effective `protocol-1.1.json`. The original
  `PREREGISTRATION.md` and `protocol.json` are kept byte-identical as history. The amendment discloses that candidate
  generation with the **offline stand-in tokenizer** occurred before the original commit (so the study is preregistered
  before any *real-tokenizer* generation, not before all forms of candidate generation); makes human review
  **annotation-only** (it can flag, never replace, regenerate, reorder or discard; a serious finding means NO-GO and a new
  protocol version; a separately labelled sensitivity analysis may exclude annotated controls); fixes the draw budget at
  exactly 5,000 with no CLI override; requires verification by **byte-identical deterministic regeneration** of the whole
  selection (a different valid control is never "equivalent"); requires a recorded anchor (the replicate loop must
  reproduce C0 under the real tokenizer) before generation; and binds every manifest to the controlling documents and the
  exact code revision. Seeds changed (the hashed rubric lost the manual-review clause); no real-tokenizer generation had
  happened, so no selected control changed. It also corrects the "nine headers per domain" claim: this skeleton leaves
  only 2-4 usable headers per domain.
* Later phases (tokenizer-only feasibility; inference) need separate authorisation. The frozen bounce/diff classifier is
  not used and not modified.

## v0.3.0 single-prompt tools (track, batch, atlas)

Added after the sections above were written; they do not change anything described there. Each score is **one prompt**: request #3
of a frozen scenario, clean or with a fragment appended after its authorized context, scored by the unchanged
`LlamaClient.choice_probs` (pre-sampling log-probabilities renormalised over A-D, tokenization verified first).
`margin_AB = ln(p_A / p_B)`; this is not the three-request fold and not `delta_contrast`.

* `bleediff.track-history/1` (`<out>/history.jsonl`): `session_start` (version, code revision, model identity and hash, server
  flags, inference request body without prompt and grammar, tokenization report), then `measurement` records (label, fragment and
  prompt hashes and sizes, probabilities, answer-letter coverage, margin, changes from baseline and previous step, preferred
  answer, flips and A/B crossings, optional token counts and candidate id), `event` records and a final `session_end`.
* `bleediff.track-batch/1` (`<out>/batch-manifest.json`): the ranking definition, candidate-file hash, model identity, baseline,
  ranked and unscored candidates, and SHA-256 values of `history.jsonl` and `leaderboard.csv`.
* `bleediff.atlas-input/1` (input) and `bleediff.atlas/1` (`atlas.json`): see the docstring of `bleediff/atlas.py` for the manifest
  fields and the validation performed. The atlas recomputes nothing from a model: margins, preferred answers and shifts are
  re-derived from the saved probabilities with `track.margin_ab`, `fold.argmax_choice` and `batch.movement_toward`.
* Flags are descriptive, never thresholds of significance: saturated baseline (highest probability at least 0.995), low
  answer-letter coverage (below 0.90), token mismatch, incorrect baseline, invalid score.

## v0.5.0 hosted answer-only measurements (gemini, openai, anthropic)

Added after the sections above; nothing in them changes. A hosted model that exposes no usable A-D probabilities is measured as `ANSWER_ONLY` (`measurement_type`): the frozen prompt is sent as one user message and the reply is
classified; no probability, margin, confidence or rank is ever derived. Usage guide: [`HOSTED_MODELS.md`](HOSTED_MODELS.md).

* `bleediff.hosted-prompts/1` (input dataset: items with cell, question, mapping, condition, prompt, prompt hash, correct and competing letter), `bleediff.hosted-session/1` (`session.json`) and `bleediff.hosted-answer/1`
  (`records/<cell>__<condition>__pass<k>.json`: request parameters without credentials, strict `status` in {ok, invalid, blocked, empty, malformed, error}, `answer`, response, usage, estimated cost).
* Strict primary parse: a reply is an answer only if it is exactly one of A-D after trimming. Provider refusals are `blocked` and are non-answers, never incorrect answers. Secondary diagnostics (for example the post-hoc
  "last line is one letter" rule S1) are outside the engine, labelled, and never replace the strict result (`research/reconcile_evidence.py`, `EVIDENCE_RECONCILIATION.md`).
* Spending guard: an enforced cap on an ESTIMATE (supplied list prices x reported usage), checked before every request against the worst case of the next request; it is not a provider billing total.
* Saved responses are scrubbed of opaque thinking signatures and reasoning text (`hosted.scrub_private_reasoning`) before they are written. `--replay` re-validates and re-analyses a saved session offline.

## Tests

`tests/test_bleediff.py`, `tests/test_v011_validity.py`, `tests/test_v012_control.py` and
`tests/test_v013_protocol.py`, `tests/test_v014_semantic_v2.py`, `tests/test_v015_robustness.py` and `tests/test_v016_amendment.py` (stdlib `unittest`, 312 tests) run a stdlib
`http.server` mock of `llama-server` (`/tokenize`, `/completion`, `/props`) and cover: exact contrastive math with
`argmax == A` and `delta_contrast > 0.4`; B winning raw is never a stealth PASS; the matched control staying within
0.1 and a drifting control making the run `INCONCLUSIVE` (never `FAIL`); invalid baselines never counting as
negative evidence; the `temperature=0` collapse trap and unwarped-logprob parsing (including the exact request
body); tokenization mismatch -> exit 2; per-request vectors, prompts and per-request coverage absent from retained
state, the trace and the sidecar; sidecar binding/duplicate/orphan rejection and explicit `UNKNOWN`s; coverage
arithmetic that leaves normalisation untouched; the preserved v0.1.0 evidence (byte-identical trace, unchanged
fixtures, fold hashes and contrast values reproduced, reclassified `INCONCLUSIVE`); no third-party imports; and
that the git author config is not a personal Gmail.

The mock proves the *logic* given chosen distributions. It says nothing about any real model; for that, use
the live commands above.

## Git identity

The repository is configured with a repo-local, non-personal author (`user.name=bleediff`,
`user.email=noreply@bleediff.invalid`), and a test fails if the repo-local author (when one is set), any commit in history or any file contains a
personal Gmail address. A fresh clone has no repo-local author, so that one check is skipped there.

## License

MIT
