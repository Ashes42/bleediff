# Bleediff v0.4.0 release notes

**Status: exploratory research release. Not a security finding.** v0.4.0 makes the answer-position question measurable: Bleediff now
scores the *semantic* correct and competing answers wherever their letters sit, and ships a small offline evidence bundle that
reproduces the four-mapping experiment's numbers.

## What is new

* **Semantic answer roles** (`bleediff/roles.py`, `track --arms --correct L --incorrect L`, letters A to D): the correct-versus-competing
  margin `ln(p_correct / p_competing)`, the correct-versus-strongest-incorrect margin, movement toward the competing answer, overall
  preferred-answer flips, role-order crossings, and newly induced incorrect answers (only when the clean baseline was correct, split into
  "to the competing answer" and "to another wrong answer").
* **Answer-letter permutations**: the same semantic answers can be measured under any letter assignment; results are tracked by role, not letter.
* **Clearer labels**: the batch output, `leaderboard.csv` (five trailing columns added), and the Atlas (quantities legend, semantic-roles
  section, flag `AB_MARGIN_NOT_ROLE_MARGIN`) say which margin is which. The A/B margin `ln(p_A / p_B)` is described as a letters-only quantity.
* **Backward compatible**: v0.3.0 sessions, manifests and example datasets load unchanged; roles at A and B give bit-identical shifts.
* **`examples/four-mapping/`**: 48 saved sessions, frozen texts, candidate files, an Atlas manifest, SHA-256 list and `verify_summary.py`.

## Exploratory findings (four mappings, six questions, two models)

Four of the 24 possible letter permutations; Qwen2.5-1.5B and Llama 3.2 3B Instruct (Q4_K_M, CPU, llama.cpp b11963, raw prompts). One cell
is one question under one mapping (24 per model). Repeats were bit-identical and are reproducibility checks, not independent samples.

* **Llama 3B:** competing evidence moved the decision toward the competing answer in 24 of 24 cells, exceeded neutral text in 24 of 24, and
  repeating it added more shift in 23 of 24 (mean shift +3.99, +2.64, +2.52, +3.14 nats for the correct answer at A, B, C, D). Newly
  induced incorrect answers occurred only with the correct answer at A (5 of 18) or D (3 of 18).
* **Qwen 1.5B:** the effects reverse with the letter assignment (mean shift +1.18, -1.17, -0.70, +0.92 nats) and the shift is positive under
  all four mappings in 0 of 6 questions. The baseline p(correct) is 0.51, 0.23, 0.09, 0.59 for A, B, C, D, and competing evidence raised
  letter B (+0.13 to +0.20) and lowered letter D (-0.08 to -0.18) under every mapping. Newly induced incorrect answers in the five-treatment
  m1/m2 data: 9 under m1 and 1 under m2 (an earlier report of "8 + 1" was a miscount, corrected here).
* **Reading:** the evidence suggests semantic-answer sensitivity in the tested Llama configuration and strong answer-letter sensitivity in
  the tested Qwen configuration. Full tables: [`docs/FOUR_MAPPING_SUMMARY.md`](docs/FOUR_MAPPING_SUMMARY.md).

## Limitations (read before citing)

* **No universal model-family mechanism, no statistical test and no real tenant-isolation vulnerability was established.**
* Four mappings (not 24), six hand-written questions, two models, one quantization each, single wording per treatment, one machine and one
  llama.cpp build; no spread estimate beyond exact repeats.
* Letter position changes baseline confidence, so position-dependent flips are not necessarily a position-dependent evidence effect.
* m1 and m2 were measured in an earlier run (five treatments) and reused after a bit-identical compatibility control; that control's
  sessions are not bundled.
* The v0.2.0 and v0.3.0 results, limitations and the INCONCLUSIVE verdict of the original five-scenario experiment are unchanged.

## Reproducing (offline, no model)

```
python -m bleediff atlas --manifest examples/four-mapping/atlas-input.json --out runs/four-mapping
python examples/four-mapping/verify_summary.py runs/four-mapping/atlas.json
python -m unittest discover -s tests
```

The v0.3.0 example datasets reproduce as before (`examples/atlas/README.md`).

## Distribution

The release is the git tag `v0.4.0`. Model files, run directories and per-run sidecars are not distributed with the repository.
