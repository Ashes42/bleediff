# Amendment 001 to the `config-audit-robustness/1` preregistration

Status: **EXPLORATORY.** This is a disclosed governance amendment. It does not change the research question, the
hypothesis, the frozen measurement, or any threshold. The original preregistration and the original protocol file are
kept byte-for-byte as historical evidence; this amendment says which of their statements it supersedes.

## 1. Identity of what is amended

| Item | Value |
|---|---|
| Protocol | `config-audit-robustness/1`, revision 1.0 amended to **revision 1.1** |
| Original preregistration commit | `58a00b1d149994eb3775570e85174fe4a3e2df21` (parent of that commit: `4805abb76fe111048a36df3eee80ef84abe6c04a`) |
| Original preregistration (`PREREGISTRATION.md`) SHA-256 | `74c118cc9c14c892ee5915fe20e83229e558783031e64d872db62e82e2ba437b` |
| Original protocol (`protocol.json`) SHA-256 | `d9cc96c6cd3ebcabd2bc9e2cc16eb43dbce9de368d4ebf0ff1883b74c81aac4c` |
| Original rubric hash (`rubric/1`) | `020552891747c7161e3277d8e8dc9a5e4690f3902da7c100410a016030110264` |
| Amended rubric hash (`rubric/1.1`) | `ee30f0978bccad62ae0ac56eb1168d9423d2555ef7a1c3b0f510ad0e7719413d` |
| Software that wrote the amendment | bleediff 0.1.6 |
| Effective protocol file | `protocol-1.1.json` (it cites this amendment's hash; this amendment deliberately does not cite it, to avoid a circular dependency) |

## 2. Disclosure: candidate generation that DID occur before the original preregistration commit

The original preregistration (line 3) said it was written "BEFORE any tokenizer-based candidate generation". **That was
inaccurate.** Before commit `58a00b1` the implementer ran the candidate generator with the **offline stand-in tokenizer**
(`tests/_tok.py`, a deterministic stand-in, not the real Qwen tokenizer):

1. a smoke comparison of the draw-loop copy against the frozen generator on all five semantic-v2 scenarios (<= 300 draws);
2. one exploratory run that generated all twelve replicates for `config-audit-v2` (<= 150 draws each) and printed the
   first 60 characters of each accepted text, the draws used and the rubric-rejection counts;
3. the regression tests, which generate candidate texts and complete twelve-replicate manifests with the stand-in.

The implementer saw the outputs of (1) and (2). Those streams used the stand-in tokenizer's fingerprint, not the real
one, so they are **not** the streams the real tokenizer will produce, and nothing from them was retained or used. After
seeing (2) the implementer added limitation 9 (same-domain controls can share a header). To the implementer's account no
rubric constant, domain allocation, draw budget, outcome label or threshold was changed between the first generation and
the commit; the repository holds a single commit for that phase, so intermediate states cannot be checked from Git and
this statement rests on that account.

**Correct description of the timing of this study:** it is preregistered **before any REAL-tokenizer candidate
generation and before any new model probability exists**. It is NOT preregistered before all forms of candidate
generation. This sentence supersedes line 3-4 of the original preregistration.

Confirmed for the state of the repository when this amendment was prepared: no control was generated with the real
tokenizer; no `/tokenize` request was sent to a real `llama-server`; no `/completion` request was made; no
`llama-server` was started; no model probability for any robustness control exists. (The regression suite uses an
in-process mock server bound to localhost, and the offline stand-in tokenizer; neither involves a real model.)

## 3. Amended clauses

| # | Original clause (location) | Amendment | Effect on seed derivation |
|---|---|---|---|
| A1 | Manual review (prereg lines 95-97; protocol `selection.rubric.manual_review`; `replicates.py` RUBRIC) let a human reject a candidate and "advance to the next candidate of the same stream" | **Human review is annotation-only** (section 4). The clause is removed from the hashed rubric. | **Yes** (rubric hash changes) |
| A2 | Draw budget (prereg lines 65-68) was a documented default that `generate --max-draws` could override and the manifest did not record | The budget is **exactly 5,000 draws per replicate**; the CLI has no override; the manifest records it; verification rejects any other value (section 5) | No |
| A3 | Verification checked structure only (prereg section 6) | **Byte-identical deterministic regeneration** is required (section 6) | No |
| A4 | The anchor gate was procedural (prereg section 6) | The anchor result is **verifiable evidence**; generation refuses without a reproduced anchor (section 7) | No |
| A5 | The manifest was not bound to the controlling documents | Provenance binding (section 8) | No |
| A6 | Inaccurate statements (section 9) | Superseded as listed | No |
| A7 | Statistical statements (prereg sections 3 and 9) | Strengthened (section 10); no calculation changed | No |
| A8 | Seeds were derivable but not recorded | The twelve seeds, their control IDs and the expected tokenizer fingerprint are now preregistered in `protocol-1.1.json` | No (records the derivation) |
| A9 | "No change after any candidate has been generated" (prereg section 10) | Because stand-in generation occurred (section 2), this amendment is itself a disclosed deviation made BEFORE real-tokenizer generation. Future changes that touch the rubric, allocation, budget, labels or thresholds still require a new protocol version | n/a |

## 4. Annotation-only human review (A1)

For each accepted candidate a reviewer may record: confound present or absent; the exact words or phrases of concern
(each must occur in the control text); severity (`none`, `minor`, `moderate`, `serious`); a reason; the review date; a
non-identifying reviewer identifier. A review is bound to the manifest's selection hash.

A reviewer **cannot** replace, regenerate, reorder, advance a stream for, edit, or selectively discard any candidate of
the primary predefined set; a review containing any other field is rejected. If a reviewer finds serious answer-related
contamination they may issue `NO_GO_FOR_INFERENCE`. That blocks inference and requires a **new protocol version**; it
never replaces a candidate.

**Primary analysis:** always the complete set of accepted candidates in replicate order.
**Sensitivity analysis S1 (predefined, separately labelled, never primary):** the same analysis excluding accepted
candidates annotated `confound_present = true`. Fewer than 10 remaining means `INSUFFICIENT`. S1 is reported next to the
primary analysis and never in its place.

The review policy is **not** part of the hashed rubric and therefore not part of any seed.

## 5. Generation constraints (A2)

* The budget is 5,000 draws per replicate, every rejected draw included. The CLI exposes no `--max-draws`; scenario,
  offset, fixtures, C0, budget and protocol identity come only from the controlling documents.
* The manifest records, per replicate: index, domain, deterministic seed, the effective draw budget, draws actually
  consumed, structural rejection counts, every rubric rejection with its text and reasons, and the final status. At
  manifest level it records the canonical invocation parameters and the code revision. Generation refuses to run on a
  dirty working tree so that the code revision is exact.
* Verification rejects a manifest whose recorded settings disagree with this protocol.
* A replicate that exhausts its budget is `FAILED` and stays failed: no other domain, seed, replicate or text is
  substituted. Fewer than 10 accepted replicates means the later study reports `INSUFFICIENT`.

## 6. Deterministic regeneration (A3)

Verification does not merely check that supplied text is structurally valid. It regenerates the whole selection from the
recorded seed rule, domains, tokenizer identity and frozen rules, and requires the **canonical JSON of the regenerated
selection to be byte-identical** to the recorded one: the same candidate selection, rejection sequence, rejection
reasons, accepted text, control identifier, draw counts and failure states. A different structurally valid control is
never accepted as equivalent. If verification fails, the procedure stops.

## 7. Anchor as evidence (A4)

Before new controls are trusted, the replicate module's draw loop, run in frozen-compatible mode, must reproduce the
original C0 under the real tokenizer. The result is written as a deterministic evidence file (schema
`bleediff.replicate-anchor/1`; a failed anchor is recorded too). `generate` refuses to run without evidence showing the
anchor reproduced C0 under the live tokenizer, and the manifest embeds the evidence and its hash; verification re-runs
the anchor and requires the same evidence. **No live anchor test was performed for this amendment.**

## 8. Provenance binding (A5)

Every manifest records: protocol ID and revision; the original preregistration hash; the original protocol hash; the
amendment hash; the effective protocol hash; the frozen generator identity (algorithm, lexicon hash, `control.py`
hash); the rubric hash; the review-policy hash; the expected and live tokenizer fingerprint; the C0 identity; the fixed
draw budget; the anchor evidence; and the exact code revision. The hash chain is acyclic: this amendment cites only the
original documents; `protocol-1.1.json` cites this amendment; a manifest cites `protocol-1.1.json`.

## 9. Corrections of inaccurate statements (A6)

| Original statement | Superseding statement |
|---|---|
| "Written and committed BEFORE any tokenizer-based candidate generation" (prereg lines 3-4) | See section 2: preregistered before any **real-tokenizer** generation, after offline stand-in generation that the implementer observed. |
| "The frozen lexicon has only nine header phrases per domain" (prereg lines 169-171; protocol limitations) | **Wrong.** Only bakery and trail have nine headers; the others have seven. This skeleton needs a **three-word** header, so only 3-4 per domain are eligible, and after RUBRIC only 2-4 survive: bakery 4, trail 3, and 2 for each of orchestra, greenhouse, seaside, library, festival and workshop. Same-domain controls (bakery, festival, greenhouse, library each appear twice) will often share a whole header; controls are therefore less diverse than twelve independent texts. The figures are recomputed and checked by the protocol verifier. |
| "every statement in sections 4-8 was checked against the implemented code" (prereg lines 181-184) | **Overstated.** The self-review missed the two inaccuracies above and left the manual-review loophole, the unenforced budget and the unverified selection. The self-review was a partial check by the author, not a complete verification. |
| Section 8 explained the fold and the delivery mechanism but not the contrast | **Added:** `delta_contrast` is an arithmetic transform of the composite `M` and the clean reference `B_ref` (a composite log-odds shift). The underlying raw probability shift is already visible without taking logarithms, so the contrast adds no detection capability of its own. A large residual says the output distribution moved; it does not by itself show where or why. |
| C0 is "outside" the analysis and "cannot influence" selection (prereg lines 26-27, 62-63) | **Clarified:** C0's fixed **text** takes part in the duplicate and overlap checks (R5, R6), so it can change which candidates are accepted. C0's **observed probability result** (delta -0.0884) never enters selection. C0 is excluded from the primary 12-control analysis. |
| "R1-R3 prohibit only a handful of lexicon words ... does not starve any domain" (prereg lines 91-93) | True as stated, but it hides the header effect above. R4 additionally rejects only `9,350` and hour-9 clock times beyond what the frozen generator already bans. |
| "A larger effect ... is expected from lexical overlap alone" (prereg lines 157-159) | A **hypothesis**, not an established expectation. The design cannot separate lexical overlap from evidence content. |

## 10. Statistical honesty (A7)

Unchanged definitions (calculations are not amended): descriptive statistics, exact binomial interval, rank exceedance,
the t-based prediction bound, the five outcome labels in the same order, no PASS label, thresholds 0.4 and 0.1 imported
from the frozen code, the actual number of valid controls used in every statistic. Stated explicitly:

* The controls are **not automatically independent or exchangeable**. They are deterministically diversified samples from
  one constrained generator; later replicates are constructed to differ from earlier ones; same-domain controls can share
  a header.
* Rank statistics and the t-based prediction bound are **diagnostics, not calibrated significance claims**.
* The original +0.857-nat result was **observed before this robustness study was designed**; selecting `config-audit-v2`
  for follow-up was **post hoc**.
* No result of this study changes the frozen five-scenario **INCONCLUSIVE** verdict.
* A large contrastive residual does **not** demonstrate a real tenant-isolation vulnerability.
* No new threshold and no outcome-optimised selection rule is introduced.

The analysis module's output now carries these statements verbatim (wording only).

## 11. Effect on seed derivation

**Seed derivation changed.** The seed is a hash that includes the rubric hash. Removing the manual-review clause from the
hashed rubric (and clarifying its distinctness wording) changed the rubric hash from `0205528917...` to `ee30f09...`, so
all twelve seeds and control identifiers changed. No replicate was ever generated with the real tokenizer under the old
seeds, so no selected control changed and nothing was observed under either set. The mechanical rules R1-R6, the
allocation, the budget, the lexicon and the tokenizer-identity input are unchanged. The old seeds (first 16 hex digits,
for the record): 335bd1bc695658ca, 1de996316be8161c, e6b732dbcc156f36, e580341150bbe50b, f1101ed338210a55,
4e65c14f25aa1fec, e4412a089ba858e2, e50d936275fb9bdf, aff29f76002a376c, d6b0614f0de79092, 75dba151872540fc,
7ede8cb831de4f02. The amended seeds are preregistered in `protocol-1.1.json`:

| k | domain | seed (first 16 hex) |
|---|---|---|
| 0 | bakery | e1af0c9ba1d3b165 |
| 1 | festival | 21c188d89e9cf1ae |
| 2 | greenhouse | 91693a436be4ab73 |
| 3 | library | 852aa84e6523b953 |
| 4 | orchestra | de8c707ae9f377f5 |
| 5 | seaside | 4d118b528ea19e04 |
| 6 | trail | 4bc3e8cc89a566fe |
| 7 | workshop | 96fefaaef3791c8b |
| 8 | bakery | f765e15d89eddf20 |
| 9 | festival | d2d6b34f25c7fb12 |
| 10 | greenhouse | 2d8319274325f736 |
| 11 | library | d0334279bfc9cf9f |

## 12. What is NOT amended (preserved limitations)

The hypothesis and question; the frozen contrast equation, fold, normalisation, thresholds and 15-field trace schema;
the frozen `neutral-prose/1` implementation and lexicon; the `config-audit-v2` treatment text and C0; the historical
five-scenario experiment and all its evidence; the analysis definitions and label order. Known limitations remain as
declared (lexical overlap with option B, recency, technical-versus-ordinary register, post-hoc selection, one model, small
n, the residual record-keeping and inspection vocabulary in the frozen lexicon, limited header diversity). Open optional
items from the audit are not changed here: the `n_planned` default, the default of a missing status field, the rule for
non-identical repeat runs, the wording of the label `TREATMENT_PRECONDITION_FAILED`, and broader semantic exclusions
(which would need a new version).
