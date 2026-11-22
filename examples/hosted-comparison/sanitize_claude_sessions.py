"""Maintainer tool: copy two saved Claude ANSWER_ONLY sessions into the public bundle (never overwrites).

Usage: python examples/hosted-comparison/sanitize_claude_sessions.py HAIKU_SESSION SONNET_SESSION OUT_DIR

Each source session (session.json, prompts.json, records/*.json) is copied to OUT_DIR/haiku55 and OUT_DIR/sonnet55. Changes made to the records, and only these:
  * opaque thinking signatures, redacted thinking data and any returned reasoning text -> placeholders (bleediff.hosted.scrub_private_reasoning)
  * the provider message id -> a pseudonymous 'msg-<sha256 prefix>' (unique, not linkable to an account)
Answers, strict statuses, usage, cost estimates, stop reasons and refusal categories are untouched, so replay output is identical to the raw evidence's.
"""
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bleediff.hosted import scrub_private_reasoning  # noqa: E402


def copy_session(src: str, dst: str) -> int:
    os.makedirs(os.path.join(dst, "records"), exist_ok=False)
    for name in ("session.json", "prompts.json"):
        shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))
    n = 0
    for fn in sorted(os.listdir(os.path.join(src, "records"))):
        rec = json.load(open(os.path.join(src, "records", fn), encoding="utf-8"))
        resp = rec.get("response")
        if isinstance(resp, dict):
            rec["response"] = scrub_private_reasoning(resp)
            if isinstance(resp.get("id"), str):
                rec["response"]["id"] = "msg-" + hashlib.sha256(resp["id"].encode()).hexdigest()[:16]
        with open(os.path.join(dst, "records", fn), "x", encoding="utf-8", newline="\n") as fh:
            json.dump(rec, fh, indent=1)
            fh.write("\n")
        n += 1
    return n


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    haiku, sonnet, out = sys.argv[1:]
    for label, src in (("haiku55", haiku), ("sonnet55", sonnet)):
        print(label, copy_session(src, os.path.join(out, label)), "records")
