"""Tokenizer-ONLY comparison of candidate window texts against the original treatment window (bytes, words, fragment tokens,
request-3 prompt tokens, skeleton). It never calls /completion. Use it BEFORE freezing a probe text; do not use model output
to choose wording.

    python research/measure_fragments.py --server http://127.0.0.1:8099 < candidates.txt      # one fragment per line
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bleediff import control as C  # noqa: E402
from bleediff.arena import load_fixtures, plan_requests  # noqa: E402
from bleediff.client import LlamaClient  # noqa: E402
from research import arms  # noqa: E402


def measure(fragment: str, sc, tokenize) -> dict:
    trailing = " ".join(sc.adjacent.split()[arms.OFFSET:])
    probe = dataclasses.replace(sc, adjacent=fragment + " " + trailing)
    plan = plan_requests(probe, "treatment", arms.OFFSET, arms.UNIT)
    try:
        window = plan.bleed_fragment.decode("ascii")
        prompt = bytes(plan.prompts[2]).decode("ascii")
    finally:
        plan.dispose()
    skel = C.parse_skeleton(fragment).to_json()
    return {"window_is_fragment": window == fragment, "bytes": len(fragment.encode()), "words": len(fragment.split()),
            "fragment_tokens": len(tokenize("\n" + fragment)), "request3_prompt_tokens": len(tokenize(prompt)),
            "skeleton_clauses": skel["clauses"], "header_len": skel["header_len"]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--server", required=True)
    a = p.parse_args(argv)
    sc = {s.id: s for s in load_fixtures(arms.FIXTURES_PATH).scenarios}[arms.SCENARIO_ID]
    client = LlamaClient(a.server, timeout=120)
    tok = lambda t: client.tokenize(t, cache=False)  # noqa: E731
    original = " ".join(sc.adjacent.split()[:arms.OFFSET])
    ref = measure(original, sc, tok)
    print(json.dumps({"original": ref}))
    for line in sys.stdin:
        line = line.rstrip("\n")
        if line.strip():
            m = measure(line, sc, tok)
            same = {k: m[k] == ref[k] for k in ("bytes", "words", "fragment_tokens", "request3_prompt_tokens", "skeleton_clauses", "header_len")}
            print(json.dumps({"text": line, **m, "matches_original": same}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
