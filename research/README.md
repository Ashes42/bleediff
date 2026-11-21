# research/

Supported research drivers for Bleediff v0.2.0 (exploratory). They use the frozen measurement code in `bleediff/` unchanged.

* `run_arms.py` / `arms.py`: run the five frozen treatment-type texts (`data/frozen-arms.json`) and the 12 preserved neutral
  controls (`data/neutral-controls-replicates-manifest.json`) twice through the frozen fold, with a fresh calibration, and
  write 15-field traces, provenance sidecars and `analysis.json`. Use `--baseline reuse:<baseline.json>` to accept an
  earlier baseline only if a fresh clean fold reproduces it exactly. Use `--reference-analysis` for a probe-only run (no
  controls of its own).
* `measure_fragments.py`: tokenizer-only comparison of a candidate window text with the original (bytes, words, fragment
  tokens, request-3 prompt tokens, skeleton). Use it before freezing a text; never choose wording from model output.
* `data/`: frozen texts, the controls manifest, sanitised traces (pass 1 of each real run; pass 2 was byte-identical) and
  `results-v0.2.0.json` with hashes of the original (untracked) evidence.

```
llama-server -m <model.gguf> --host 127.0.0.1 --port 8099 -c 4096 -ngl 0 --no-webui
python research/run_arms.py --server http://127.0.0.1:8099 --controls research/data/neutral-controls-replicates-manifest.json --out runs/my-run
```

The controls were tokenizer-matched under the Qwen2.5 tokenizer. For another tokenizer `analysis.json` reports
`controls_tokenizer_fingerprint_matches_server: false`: exact token parity is then NOT guaranteed and must be checked.
