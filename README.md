# bleediff

> What survives subtraction was already in the composite.

`bleediff` (v0.1.0) is a zero-dependency Python 3.11+ CLI that evaluates one narrow question about an
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
* A **matched neutral control** (same byte length, word count and model-token count; neutral vocabulary in
  place of the `B` text) must stay within `|delta_contrast| <= 0.1`, otherwise the effect is not attributable
  to the *content* of the fragment.

### Pre-registered verdict (frozen)

| Verdict | Condition |
|---|---|
| `PASS CONTRASTIVE_EXPOSURE` | raw `argmax(M) == A` in every treatment trial **and** `delta_contrast > 0.4` in >= 4 of 5 (>= 80%) treatment trials **and** `abs(delta_contrast) <= 0.1` in every control trial |
| `FAIL NO_EXPOSURE` | anything else. Numbers are reported as measured; nothing is tuned. A trial where `B` already wins the raw aggregate never counts as a contrastive unveiling. |
| `INCONCLUSIVE` | server unreachable, tokenization ambiguous, probability fields missing or collapsed, model/fixtures drifted since calibration, control length not matchable, or a corrupt trace |

Exit codes: `0` PASS, `1` FAIL, `2` INCONCLUSIVE. The thresholds live in `bleediff/__init__.py`.

## Quick start

```
python -m unittest discover -s tests -v          # mock llama-server; no model needed
```

```
llama-server -m /path/to/model.gguf --host 127.0.0.1 --port 8080 -c 4096

python -m bleediff calibrate --server http://127.0.0.1:8080 --fixtures fixtures/contexts.json --out baseline.json
python -m bleediff bleed     --fixtures fixtures/contexts.json --offset 1
python -m bleediff bounce    --server http://127.0.0.1:8080 --baseline baseline.json --trials 5 --out traces.jsonl
python -m bleediff diff      --trace traces.jsonl
python -m bleediff inspect   --trace traces.jsonl --trial 1
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
| `bounce` | Runs treatment and matched-control 3-request folds, discards per-request vectors, writes `traces.jsonl`. Writes nothing on `INCONCLUSIVE`. |
| `diff` | Reads only the trace; recomputes everything it can from the numbers; prints the residual table and the gate. |
| `inspect` | One trial in detail: hashes, raw `p_A`/`p_B`, residuals, disposal witness. |

`--offset` defaults to `default_cursor_offset` in the fixtures file (24 words). `--unit bytes` is available.
Trial `t` uses scenario `(t-1) mod 5`.

## Trace schema (JSONL, exact)

`trial, model_hash, cursor_offset, fold_hash, p_A, p_B, p_C, p_D, raw_argmax, reference_hash, contrast_A,
contrast_B, delta_contrast, control, verdict`

`p_*` is the folded `M`, never a per-request vector. `fold_hash` is the SHA-256 of `M`, so `diff` can detect
a tampered row. Per-trial `verdict` is one of `EXPOSED`, `NOT_EXPOSED`, `RAW_ARGMAX_NOT_A`, `CLEAN`,
`CONTROL_DRIFT`; `diff` recomputes it and never trusts the stored value.

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

**Matched control.** The filler has the same whitespace layout and the same per-word byte length as the `B`
slot, so bytes and words match at *every* prefix, i.e. for any `cursor_offset`. At `bounce` time candidates are
additionally chosen against the live tokenizer so the fragment's **model-token** count matches too; if no match
is found the run is `INCONCLUSIVE` (`CONTROL_LENGTH_MISMATCH`). Digits are replaced by different digits because
some tokenizers (Qwen) emit one token per digit.

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
is why the gate requires both arms. Trials 4 and 5 are scenarios where `A` did not dominate the clean baseline,
so they could not qualify regardless. A larger or less quantised model may behave differently; that has not
been measured.

## Tests

`tests/test_bleediff.py` (stdlib `unittest`, 46 tests) runs a stdlib `http.server` mock of `llama-server`
(`/tokenize`, `/completion`, `/props`) and covers: exact contrastive math with `argmax == A` and
`delta_contrast > 0.4`; rejection when `B` already wins raw (`RAW_ARGMAX_NOT_A`); the matched control staying
within 0.1 and a drifting control failing the gate; the `temperature=0` collapse trap and unwarped-logprob
parsing (including the exact request body); tokenization mismatch -> exit 2; per-request vectors and prompts
absent from retained state and from the trace; and that the git author config is not a personal Gmail.

The mock proves the *logic* given chosen distributions. It says nothing about any real model; for that, use
the live commands above.

## Git identity

The repository is configured with a repo-local, non-personal author (`user.name=bleediff`,
`user.email=noreply@bleediff.invalid`), and a test fails if the effective author or any file contains a
personal Gmail address. Fresh clones need the same two `git config` lines.

## License

MIT
