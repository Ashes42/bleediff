"""Research 002 compatibility demonstration for ``bleediff behavior`` (offline: no key, no network, no model).

Reads the PUBLISHED Research 002 evidence (the sanitized run records and the frozen analysis output) and checks that the generic behavioral-telemetry layer reproduces the
published condition-level and paired figures without any Research 002-specific code. Nothing in the published package is modified or recomputed by this script: it is only read.

    python examples/behavior/r002_compat_check.py PATH_TO_bleediff-research-002-publication

Exit code 0 = every figure matches; 1 = a mismatch (printed); 2 = the package was not found. Prices below are the Research 002 list prices (USD per 1M tokens), recovered from the
saved records, where they reproduce every usage-based cost exactly.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from bleediff import behavior as B  # noqa: E402

PRICES = (0.1, 0.5)


def main(pkg: str) -> int:
    ro = os.path.join(pkg, "research_artifacts", "run_output")
    pub_path = os.path.join(pkg, "research_artifacts", "analysis_output", "analysis_results.json")
    if not (os.path.isdir(ro) and os.path.isfile(pub_path)):
        print(f"not a Research 002 publication package: {pkg}")
        return 2
    with open(pub_path, encoding="utf-8") as fh:
        pub = json.load(fh)
    srcs = [B.load_evidence(os.path.join(ro, s), prices=PRICES, cost_from_usage=True) for s in ("sessionA", "sessionB")]
    rep = B.build_report(srcs, unit_field="question", select_unit=r"^F[0-9]+D$", contrasts=[("CONTRA", "CONSIST")])
    n_obs = sum(len(s["observations"]) for s in srcs)
    print(f"ingested {n_obs} records from {len(srcs)} sessions; ingest problems: {sum(len(s['problems']) for s in srcs)}; "
          f"selected {rep['selection']['observations_selected']} derivation records ({rep['selection']['unit_field']} units)")
    bad = 0
    per = pub["secondary"]["per_condition"]
    print(f"\n{'condition':10s} {'measure':22s} {'telemetry layer':>16s} {'published':>16s}  match")
    for cond, g in rep["scopes"]["pooled"]["conditions"].items():
        ua = g["unit_averaged"]
        rows = [("thinking_activation", ua["thinking_activation"]["mean_of_unit_means"], per[cond]["T"]["family_mean"]),
                ("correct", ua["correct"]["mean_of_unit_means"], per[cond]["correct"]["family_mean"]),
                ("format failure", 1 - ua["format_compliant"]["mean_of_unit_means"], per[cond]["format_failure"]["family_mean"]),
                ("thinking_tokens", ua["thinking_tokens"]["mean_of_unit_means"], per[cond]["thinking"]["family_mean"]),
                ("cost_usd (from usage)", ua["cost_usd"]["mean_of_unit_means"], per[cond]["cost"]["family_mean"])]
        for name, mine, theirs in rows:
            ok = abs(mine - theirs) <= 1e-8 * max(1.0, abs(theirs)) if "cost" not in name else abs(mine - theirs) <= 1e-11
            bad += not ok
            print(f"{cond:10s} {name:22s} {mine:16.9g} {theirs:16.9g}  {'ok' if ok else 'MISMATCH'}")
    c = rep["scopes"]["pooled"]["contrasts"][0]["measures"]
    pc = pub["secondary"]["contrasts"]["CONTRA-CONSIST"]
    print(f"\npaired contrast CONTRA - CONSIST over {rep['scopes']['pooled']['contrasts'][0]['pairs']} pairs (same session, cell and pass), unit-averaged over 24 families:")
    for name, key in (("thinking_activation", "T"), ("correct", "correct"), ("thinking_tokens", "thinking"), ("cost_usd", "cost")):
        mine, theirs = c[name]["unit_averaged_mean_difference"], pc[key]["mean"]
        ok = abs(mine - theirs) <= 1e-8 * max(1.0, abs(theirs)) if name != "cost_usd" else abs(mine - theirs) <= 1e-11
        bad += not ok
        print(f"  {name:22s} {mine:16.9g} {theirs:16.9g}  {'ok' if ok else 'MISMATCH'}")
    d = c["thinking_activation"]["discordance"]
    print(f"  thinking_activation discordance: {d}")
    print("\nThe published results are only READ; no interval or test is computed here. The published study's classification is unchanged: " + ", ".join(pub["decisions"]["categories"]) + ".")
    print("RESULT:", "every figure matches" if not bad else f"{bad} MISMATCH(ES)")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
