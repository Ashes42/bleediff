"""Offline verification of the documented four-mapping findings (stdlib only; reads files, never contacts a model).

    python -m bleediff atlas --manifest examples/four-mapping/atlas-input.json --out runs/four-mapping
    python examples/four-mapping/verify_summary.py runs/four-mapping/atlas.json

The script (1) checks the SHA-256 values in examples/four-mapping/SHA256SUMS.txt against the bundled data files and (2) recomputes every
number quoted in docs/FOUR_MAPPING_SUMMARY.md, the README and RELEASE_NOTES_v0.4.0.md from the Atlas report built over those files, printing
PASS or FAIL for each statement. It exits 1 if any statement fails.

Definitions (semantic roles, from the saved probabilities): the *shift* of a treatment is the movement of ln(p_correct / p_competing) toward the
competing answer from the same mapping's clean baseline, in nats. A cell is one question under one mapping. Each cell is one deterministic
measurement; the sessions were run twice and repeated bit-identically, which is a reproducibility check, not an independent sample.
"""

import hashlib
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAPPINGS = ["m1", "m2", "m3", "m4"]
THREE = {"neutral", "claim", "repeated_claim"}          # treatments shared by all four mappings: neutral, competing, repeated competing evidence
CORRECT_AT = {"m1": "A", "m2": "B", "m3": "C", "m4": "D"}
results = []


def check(name, ok, observed, documented):
    results.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + f"{name}: observed {observed} | documented {documented}")


def verify_hashes() -> None:
    path = os.path.join(HERE, "SHA256SUMS.txt")
    bad, n = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            digest, rel = line.rstrip("\n").split("  ", 1)
            n += 1
            full = os.path.join(HERE, rel)
            if not os.path.isfile(full):
                bad.append(rel + " (missing)")
                continue
            with open(full, "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() != digest:
                    bad.append(rel)
    check("bundle files match SHA256SUMS.txt", not bad and n > 0, f"{n} files, {len(bad)} differ {bad[:3]}", "all files unchanged")


def main(atlas_path: str) -> int:
    verify_hashes()
    with open(atlas_path, encoding="utf-8") as fh:
        atlas = json.load(fh)
    check("all sources validated by the Atlas", all(s["ok"] for s in atlas["sources"]), f"{sum(s['ok'] for s in atlas['sources'])}/{len(atlas['sources'])}", "48/48")
    cells = {}
    for c in atlas["cells"]:
        q, mp = c["task_id"].rsplit("-", 1)
        cells[(c["model_id"], q, mp)] = c
    questions = sorted({k[1] for k in cells})

    def cat(c, name):
        return [k for k in c["candidates"] if k["category"] == name and k["outcome"] != "unscored"][0]

    def shift(m, q, mp, name):
        return cat(cells[(m, q, mp)], name)["shift_toward_incorrect"]

    # ---- Llama 3B ----
    llama = [(q, mp) for q in questions for mp in MAPPINGS]
    pos = sum(shift("llama3b", q, mp, "claim") > 0 for q, mp in llama)
    check("Llama 3B: positive semantic competing-answer shift", pos == 24 and len(llama) == 24, f"{pos}/{len(llama)} cells", "24/24")
    gt = sum(shift("llama3b", q, mp, "claim") > shift("llama3b", q, mp, "neutral") for q, mp in llama)
    check("Llama 3B: competing evidence exceeds neutral text", gt == 24, f"{gt}/{len(llama)}", "24/24")
    rep = sum(shift("llama3b", q, mp, "repeated_claim") > shift("llama3b", q, mp, "claim") for q, mp in llama)
    check("Llama 3B: repeated competing evidence exceeds a single statement", rep == 23, f"{rep}/{len(llama)}", "23/24")
    means = {mp: statistics.fmean(shift("llama3b", q, mp, "claim") for q in questions) for mp in MAPPINGS}
    check("Llama 3B: mean competing shift by mapping (nats)", [round(means[m], 2) for m in MAPPINGS] == [3.99, 2.64, 2.52, 3.14], [round(means[m], 2) for m in MAPPINGS], "3.99, 2.64, 2.52, 3.14")
    # ---- Qwen 1.5B: letter behaviour ----
    pk = {mp: statistics.fmean(cells[("qwen1.5b", q, mp)]["baseline"]["roles"]["p_correct"] for q in questions) for mp in MAPPINGS}
    check("Qwen 1.5B: baseline p(correct) when the correct answer is at A, B, C, D", [round(pk[m], 2) for m in MAPPINGS] == [0.51, 0.23, 0.09, 0.59], [round(pk[m], 2) for m in MAPPINGS], "0.51, 0.23, 0.09, 0.59")
    dp = {mp: {l: statistics.fmean(cat(cells[("qwen1.5b", q, mp)], "claim")["probabilities"][l] - cells[("qwen1.5b", q, mp)]["baseline"]["probabilities"][l] for q in questions) for l in "ABCD"} for mp in MAPPINGS}
    b_ok = all(0.125 <= dp[m]["B"] <= 0.205 for m in MAPPINGS)
    d_ok = all(-0.185 <= dp[m]["D"] <= -0.075 for m in MAPPINGS)
    check("Qwen 1.5B: competing evidence raises letter B in every mapping", b_ok, [round(dp[m]["B"], 2) for m in MAPPINGS], "+0.13 to +0.20")
    check("Qwen 1.5B: competing evidence lowers letter D in every mapping", d_ok, [round(dp[m]["D"], 2) for m in MAPPINGS], "-0.08 to -0.18")
    all4 = sum(all(shift("qwen1.5b", q, mp, "claim") > 0 for mp in MAPPINGS) for q in questions)
    check("Qwen 1.5B: shift toward the competing answer positive under all four mappings", all4 == 0, f"{all4}/6 questions", "0/6")
    signs = {mp: statistics.fmean(shift("qwen1.5b", q, mp, "claim") for q in questions) > 0 for mp in MAPPINGS}
    check("Qwen 1.5B: mean shift positive when the correct answer is at A or D, negative at B or C", signs == {"m1": True, "m2": False, "m3": False, "m4": True}, {m: round(statistics.fmean(shift("qwen1.5b", q, m, "claim") for q in questions), 2) for m in MAPPINGS}, "+ - - +")

    # ---- newly induced incorrect answers (only when the baseline was correct) ----
    def induced(m, mp, cats=None):
        out = []
        for q in questions:
            c = cells[(m, q, mp)]
            for k in c["candidates"]:
                if k["outcome"] == "induced_incorrect" and (cats is None or k["category"] in cats):
                    out.append((q, k["category"], k["semantic"]["induced_to"]))
        possible = sum(1 for q in questions if cells[(m, q, mp)]["baseline"]["correct"]) * (len(cats) if cats else 5)
        return out, possible

    q1, _ = induced("qwen1.5b", "m1")
    q2, _ = induced("qwen1.5b", "m2")
    check("Qwen 1.5B: newly induced incorrect answers over the five treatments, m1 and m2", (len(q1), len(q2)) == (9, 1), f"{len(q1)} under m1 + {len(q2)} under m2 = {len(q1) + len(q2)}", "9 + 1 = 10")
    l1, _ = induced("llama3b", "m1")
    l2, _ = induced("llama3b", "m2")
    check("Llama 3B: newly induced incorrect answers over the five treatments, m1 and m2", (len(l1), len(l2)) == (5, 0), f"{len(l1)} under m1, {len(l2)} under m2", "5 under m1, 0 under m2")
    exp = {("qwen1.5b", "m1"): (6, 15), ("qwen1.5b", "m2"): (1, 3), ("qwen1.5b", "m3"): (0, 0), ("qwen1.5b", "m4"): (6, 18),
           ("llama3b", "m1"): (5, 18), ("llama3b", "m2"): (0, 18), ("llama3b", "m3"): (0, 18), ("llama3b", "m4"): (3, 18)}
    got = {}
    for m in ("qwen1.5b", "llama3b"):
        for mp in MAPPINGS:
            lst, poss = induced(m, mp, THREE)
            got[(m, mp)] = (len(lst), poss)
    check("induced incorrect answers over the three shared treatments (induced, possible) for every model and mapping", got == exp, {f"{m[:5]}-{mp}": v for (m, mp), v in got.items()}, exp)
    m4, _ = induced("qwen1.5b", "m4", THREE)
    check("Qwen 1.5B m4: every induced error goes to a wrong answer other than the competing one", len(m4) == 6 and all(t == "other_incorrect" for _q, _c, t in m4), f"{len(m4)} induced, to other: {sum(t == 'other_incorrect' for _q, _c, t in m4)}", "6, all other")
    wrong_cells = sorted((mp, sum(1 for q in questions if not cells[("qwen1.5b", q, mp)]["baseline"]["correct"])) for mp in MAPPINGS)
    check("Qwen 1.5B: clean baselines with a wrong preferred answer by mapping (m1..m4)", [n for _m, n in wrong_cells] == [1, 5, 6, 0], [n for _m, n in wrong_cells], "1, 5, 6, 0 of 6")
    lm = [mp for mp in MAPPINGS if any(k["outcome"] == "induced_incorrect" for q in questions for k in cells[("llama3b", q, mp)]["candidates"])]
    check("Llama 3B: induced errors occur only with the correct answer at A or D", lm == ["m1", "m4"], lm, "m1 (A) and m4 (D) only")
    cov = min(min(k["coverage"] for k in c["candidates"] if k.get("coverage") is not None) for c in cells.values())
    check("answer-letter coverage of every candidate is at least 0.965", cov >= 0.965, round(cov, 3), ">= 0.965")
    print()
    print("ALL DOCUMENTED STATEMENTS VERIFIED" if all(results) else "VERIFICATION FAILED: correct the documentation (never the measurements)")
    return 0 if all(results) else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
