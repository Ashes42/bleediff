# Bleediff v0.2.0 release notes

**Status: exploratory research release. Not a security finding.** Bleediff is a zero-dependency Python CLI that measures how
added context changes a local GGUF model's A/B/C/D answer probabilities by folding three requests and comparing the
composite with a clean baseline. v0.2.0 packages the measurement toolkit together with the results of one scenario
(`config-audit-v2`) on two models.

## Results (delta_contrast in nats; raw argmax in brackets)

| | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---|---|
| Original B | +0.857 (A) | +0.654 (B) |
| Paraphrase | +0.713 (A) | +0.479 (B) |
| Polarity reversed | -0.183 (A) | -0.668 (A) |
| 12 neutral controls | -0.241 to +0.061 | -0.698 to -0.673 |
| Controls within ±0.1 | 5 of 12 | 0 of 12 |

* The original treatment exceeded all 12 neutral controls on both models; the paraphrase kept most of the effect; reversing
  the polarity removed it.
* On 1.5B the raw answer stayed A everywhere; on 7B the treatment flipped it to B.
* Repeated measurements were bit-identical in the tested configurations.
* The two polarity cues interact: their effects are not strictly additive.

## What changed since 0.1.6
* `research/` drivers promoted from untracked scripts; a summary crash for probe-only runs is fixed.
* Tracked, sanitised evidence and a results file with hashes (`research/data/`).
* README rewritten; the previous README is `docs/TECHNICAL.md`.
* New tests for request plans, saved-evidence equations, analysis and an end-to-end mock-server run.
* No change to any equation, threshold, trace schema, validity gate, control generator, protocol or historical evidence.

## Limitations (read before citing)
* The simulated displacement is ordinary text concatenation at the model interface: the model receives the same bytes
  either way. No real tenant-isolation vulnerability was demonstrated.
* The fold is algebraically recoverable when requests 1 and 2 are unchanged and a clean baseline is known.
* The historical ±0.1 control tolerance failed often; neutral-control distributions need not be centered at zero.
* Lexical overlap alone does not explain the pattern, but semantic interpretation is not uniquely established.
* The original five-scenario experiment remains INCONCLUSIVE.
* One model family, one quantization per model, one post-hoc-selected scenario, selected prompts, single probe texts, one
  machine and one llama.cpp build. Do not generalize.

## Reproducing
See the README. You supply the GGUF files and `llama-server`; models, run directories and per-run sidecars are not part
of the repository.

## Publication
This release has been prepared and tagged locally. It has **not** been pushed or published.
