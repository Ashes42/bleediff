"""Offline cost ESTIMATE for a hosted answer-only pilot (stdlib only; reads the prompt file, never contacts any API).

    python examples/hosted-comparison/estimate_pilot_cost.py --prompts examples/gemini-answer-only/prompts.json --passes 2 \
        --probe-chars 560 --probe-input-tokens 228 --max-output-tokens 1024 \
        --price claude-haiku-5-5 0.10 0.50 --price claude-sonnet-5-5 2.00 10.00

For every model it prints three input-token views (calibrated, calibrated with a safety margin, the UTF-8-byte upper bound that the live spending guard uses) and the cost for several
output-token scenarios, up to the `--max-output-tokens` cap (the guard's per-request output reserve). All numbers are ESTIMATES from the prices you pass and a single calibration probe;
they are not provider billing totals. Output scenarios other than the cap depend on how much the model thinks, which a probe of one request cannot establish.
"""

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--probe-chars", type=float, required=True, help="characters of the prompt used for the calibration request")
    ap.add_argument("--probe-input-tokens", type=float, required=True, help="input tokens the provider reported for that request")
    ap.add_argument("--max-output-tokens", type=int, default=1024)
    ap.add_argument("--margin", type=float, default=1.5, help="safety factor on the calibrated input tokens (default 1.5: other prompts tokenize differently, e.g. random letters or punctuation)")
    ap.add_argument("--price", nargs=3, action="append", metavar=("MODEL", "IN_PER_M", "OUT_PER_M"), required=True)
    a = ap.parse_args(argv)
    with open(a.prompts, encoding="utf-8") as fh:
        items = json.load(fh)["items"]
    n = len(items) * a.passes
    chars = sum(len(i["prompt"]) for i in items) * a.passes
    upper = sum(len(i["prompt"].encode("utf-8")) + 64 for i in items) * a.passes
    ratio = a.probe_input_tokens / a.probe_chars
    cal = chars * ratio
    print(f"{len(items)} prompts x {a.passes} pass(es) = {n} requests; {chars} prompt characters")
    print(f"input tokens: calibrated {cal:,.0f} (ratio {ratio:.3f} tokens/char from the probe); with margin x{a.margin:g} {cal * a.margin:,.0f}; guard upper bound (1 token/byte + 64) {upper:,.0f}")
    scen = [("answer only (5 output tokens)", 5), ("light thinking (100)", 100), ("moderate thinking (300)", 300), (f"cap ({a.max_output_tokens})", a.max_output_tokens)]
    print()
    print(f"{'model':<22}{'input $/M':>10}{'out $/M':>9} | " + " | ".join(f"{s[0]:<30}" for s in scen))
    for name, pin, pout in a.price:
        pin, pout = float(pin) / 1e6, float(pout) / 1e6
        cells = [f"${cal * a.margin * pin + n * t * pout:8.3f}  (input ${cal * a.margin * pin:.3f})".ljust(30) for _s, t in scen]
        print(f"{name:<22}{float(pin) * 1e6:>10.2f}{float(pout) * 1e6:>9.2f} | " + " | ".join(cells))
    print()
    print("worst case the live guard allows per request (prompt upper bound + the output cap):")
    for name, pin, pout in a.price:
        pin, pout = float(pin) / 1e6, float(pout) / 1e6
        worst = max((len(i["prompt"].encode("utf-8")) + 64) * pin + a.max_output_tokens * pout for i in items)
        print(f"  {name}: ${worst:.5f} per request; at most ${worst * n:.3f} for the whole run if every request hit the cap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
