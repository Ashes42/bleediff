"""Answer-mention contrasts from an Atlas report (stdlib only; reads atlas.json, never contacts a model).

    python -m bleediff atlas --manifest examples/atlas/slot-swap/atlas-input.json --out runs/atlas-slot-swap
    python examples/atlas/mention_contrasts.py runs/atlas-slot-swap/atlas.json

Shift = movement toward the incorrect answer B, in nats, from that model's own clean baseline (positive = toward B).
For one 2x2 (neither / A only / B only / both):
    B_ONLY - NEITHER, A_ONLY - NEITHER, B_ONLY - A_ONLY, BOTH - A_ONLY and the interaction (BOTH - A_ONLY) - (B_ONLY - NEITHER).
The "answer-mention" dataset has A in slot 1 and B in slot 2. The "slot-swap" dataset re-measures that arrangement and the swapped
one (B in slot 1, A in slot 2), so the B effect can be compared between positions. These are observed effects of single wordings,
not evidence of a mechanism.
"""

import json
import sys

ARRANGEMENTS = {   # name -> (neither, a_only, b_only, both)
    "A slot 1, B slot 2": ("neither", "a_only", "b_only", "both"),
    "A slot 1, B slot 2 (re-measured)": ("neither", "a_only_slot1", "b_only_slot2", "both_orig"),
    "B slot 1, A slot 2 (swapped)": ("neither", "a_only_slot2", "b_only_slot1", "both_swapped"),
}


def contrasts(sh: dict, n: str, a: str, b: str, both: str) -> dict:
    return {"B_ONLY-NEITHER": sh[b] - sh[n], "A_ONLY-NEITHER": sh[a] - sh[n], "B_ONLY-A_ONLY": sh[b] - sh[a], "BOTH-A_ONLY": sh[both] - sh[a],
            "interaction": (sh[both] - sh[a]) - (sh[b] - sh[n])}


def main(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        atlas = json.load(fh)
    found = False
    for c in atlas["cells"]:
        sh = {}
        for k in c["candidates"]:
            if k.get("outcome") != "unscored":
                sh[k["category"]] = k["shift_toward_incorrect"]
        for name, cats in ARRANGEMENTS.items():
            if all(x in sh for x in cats):
                found = True
                r = contrasts(sh, *cats)
                flips = [k["candidate_id"] for k in c["candidates"] if k["category"] in cats and k.get("outcome") == "induced_incorrect"]
                print(f"{c['task_id']:<20}{c['model_id']:<10}{name:<34}" + " ".join(f"{k}={v:+.2f}" for k, v in r.items()) + (f"  induced-incorrect: {','.join(flips)}" if flips else ""))
    if not found:
        print("no complete mention 2x2 (categories neither / a_only / b_only / both) in this report")
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
