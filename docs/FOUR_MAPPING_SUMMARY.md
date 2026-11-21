# Four-mapping counterfactual experiment: summary (exploratory)

**Status: exploratory, small, descriptive.** Four of the 24 possible answer-letter permutations, six synthetic questions, two small local
models. Nothing here is a universal claim, a statistical test, or a security finding.

## Design

* **Questions.** Six synthetic questions with objectively verifiable answers: lab pH, invoice balance, route time, league points, library
  overdue days, loaf core temperature. Each has a correct answer (K), a *competing* incorrect answer (T) that the treatments argue for, and two
  other distractors (X, Y).
* **Mappings.** The same semantic answers under four letter assignments: m1 correct A / competing B, m2 correct B / competing A, m3 correct C /
  competing D, m4 correct D / competing C (the remaining distractors fill the other letters by a fixed rule). Results are tracked by
  semantic answer, never by letter.
* **Treatments.** NEUTRAL (unrelated prose), COMPETING (explicit competing evidence) and REPEATED (the competing claim stated twice), matched
  by neutral on-topic padding; token counts match exactly for Llama and within 2 to 4 tokens for Qwen inside each question.
* **Models and setup.** Qwen2.5-1.5B and Llama 3.2 3B Instruct (Q4_K_M, CPU, llama.cpp b11963, raw prompts), checked against the recorded
  weights hashes. m1 and m2 come from an earlier run (five treatments) and were reused only after a compatibility control: eight sessions
  (four cells per model) re-measured with the current code were bit-identical (probabilities, prompts, request body, grammar, template, model hash).
* **Cells.** One cell is one question under one mapping (24 per model). Every session was run twice and repeated bit-identically; repeats are
  reproducibility checks, **not** independent samples.
* **Scoring.** Role margins `ln(p_correct / p_competing)`, shift = minus the change of that margin from the same mapping's clean baseline
  (positive = toward the competing answer), a newly induced incorrect answer only when the clean baseline was correct. Probability
  measurements and preferred-answer behavior are reported separately.

## Results

Mean over six questions; "+/6" counts questions with a positive value.

| Model | Mapping (correct at) | Baseline p(correct); correct baselines | Competing shift (+/6) | Competing minus neutral (+/6) | Repeated minus competing (+/6) | Induced incorrect answers |
|---|---|---|---|---|---|---|
| Llama 3B | m1 (A) | 0.93; 6/6 | +3.99 (6) | +2.62 (6) | +1.29 (6) | 5 of 18, all to the competing answer |
| Llama 3B | m2 (B) | 0.99; 6/6 | +2.64 (6) | +1.49 (6) | +0.27 (6) | 0 of 18 |
| Llama 3B | m3 (C) | 0.98; 6/6 | +2.52 (6) | +1.72 (6) | +0.65 (5) | 0 of 18 |
| Llama 3B | m4 (D) | 0.90; 6/6 | +3.14 (6) | +2.01 (6) | +0.91 (6) | 3 of 18 (2 to the competing answer) |
| Qwen 1.5B | m1 (A) | 0.51; 5/6 | +1.18 (6) | +1.13 (6) | -0.19 (1) | 6 of 15 |
| Qwen 1.5B | m2 (B) | 0.23; 1/6 | -1.17 (0) | -1.24 (0) | +0.54 (6) | 1 of 3 |
| Qwen 1.5B | m3 (C) | 0.09; 0/6 | -0.70 (0) | -0.71 (1) | -0.01 (3) | none possible (no correct baseline) |
| Qwen 1.5B | m4 (D) | 0.59; 6/6 | +0.92 (6) | +0.79 (5) | -0.02 (2) | 6 of 18, all to other wrong answers |

Induced-answer denominators are (cells with a correct baseline) x 3 treatments. All answer-letter coverages were at least 0.965.

## What the results show (descriptive)

* **Llama 3B: the competing evidence moved the decision toward the competing answer in all 24 cells** (semantic shift positive in 24/24), and
  **competing evidence exceeded neutral text in 24/24**. Repeating the evidence added more shift in 23 of 24 cells. The sign did not depend
  on the answer letter; the magnitude did (2.5 to 4.0 nats by mapping).
* **Llama 3B error rates are mapping-dependent.** Newly induced incorrect answers occurred only with the correct answer at A (5 of 18) or D
  (3 of 18) and never at B or C (0 of 36). Baseline confidence differs by position, so position-dependent flips are not necessarily a
  position-dependent evidence effect.
* **Qwen 1.5B: the effects reverse across mappings and are dominated by answer-letter behavior.** The shift toward the competing answer is
  positive when the correct answer sits at A or D and negative at B or C, and it is positive under all four mappings in 0 of 6 questions.
  Its baseline probability of the correct answer depends on the letter (0.51, 0.23, 0.09, 0.59 for A, B, C, D), and competing evidence
  raised the probability of letter B (+0.13 to +0.20) and lowered letter D (-0.08 to -0.18) under every mapping, which accounts for all four
  shift signs. Its induced-error rate also depends on the mapping (6 of 15, 1 of 3, none possible, 6 of 18), and at m4 every induced error went
  to a wrong answer other than the competing one.
* **Repetition** added influence for Llama in most cells and showed no consistent effect for Qwen 1.5B.

## What this does not show

* Four mappings are a small subset of the 24 possible answer-letter permutations; the distractor placement rule was fixed, not randomized.
* Six hand-written questions, one wording per treatment, two models, one quantization each, one llama.cpp build: no universal claim about
  prompt influence, priming, or model behavior in general.
* Position effects and confidence effects are not fully separated (letter position changes baseline confidence as well).
* No real tenant-isolation vulnerability was demonstrated.

## Counts reconciled

An earlier report of Qwen 1.5B's induced incorrect answers in the five-treatment m1/m2 data read "10, of which 8 under m1 and 1 under m2".
Recounted from the saved records the correct figures are 9 (m1) and 1 (m2), 10 in total; Llama 3B had 5, all under m1. The table above counts
only the three treatments shared by all four mappings.

## Evidence

The measurements are in local, git-ignored run directories (the v0.4 and v0.5 runs) that are **not** part of the repository: frozen texts and
hashes, saved sessions, the compatibility control, an Atlas report and the analysis tables with a hash list. The scoring that produced the
role quantities is `bleediff/roles.py`, exercised by `tests/test_v033_roles.py` and `tests/test_v034_labels.py`.
