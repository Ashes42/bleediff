"""Enumerate every raw hosted request record in the git-ignored exploratory runs (v0.10-v0.15) and reconcile counts by unique request.

Usage: python research/reconcile_evidence.py RUNS_DIR OUT_DIR

Reads only ``bleediff.hosted-answer/1`` records (``<run>/**/records/*.json``). Writes (never overwrites):
  OUT_DIR/raw-record-index.csv     one row per unique raw request record: run, model, cell, condition label, prompt hash, STRICT status/outcome, refusal category, tokens,
                                   the secondary final-line extraction (FLR-v1, a separately labelled post-hoc rule), and a pseudonymous record id (provider message ids are NOT written)
  OUT_DIR/reconciliation.json      totals per run and per (model, cell, prompt hash)

Strict classification is the one stored in each record by the original parser (exact single letter A-D; provider refusal = blocked; anything else invalid/empty). It is never changed here.

SECONDARY EXTRACTION RULE FLR-v1, Final-Line Recovery (post-hoc, diagnostic only): take the last non-empty line of the visible text, strip surrounding whitespace and markdown emphasis characters, and accept it if it is
exactly one letter A-D. It was defined after the Sonnet 5.5 replies had been read (v0.11 "Diagnostic B"), credits replies the prompt forbids, and is reported in its own column. FLR-v1 is
unrelated to the experimental filler identifier S1 used in v0.13-v0.15.
"""

from __future__ import annotations

import collections
import csv
import glob
import hashlib
import json
import os
import re
import sys

CHOICES = "ABCD"


def visible_text(response) -> str:
    content = response.get("content") if isinstance(response, dict) else None
    if not isinstance(content, list):
        return ""
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str))


def flr_v1_letter(response):
    """Secondary extraction rule FLR-v1 (post-hoc): last non-empty line is exactly one letter A-D once markdown emphasis is stripped. None otherwise."""
    lines = [ln.strip() for ln in visible_text(response).splitlines() if ln.strip()]
    if not lines:
        return None
    last = re.sub(r"[\s*_`~]+", "", lines[-1])
    return last if last in tuple(CHOICES) else None


def outcome(rec: dict) -> str:
    """Strict outcome from the stored record (the original classification)."""
    if rec.get("block_reason") == "refusal" or rec.get("status") == "blocked":
        return "refusal"
    if rec.get("status") == "ok":
        a = rec.get("answer")
        return "correct" if a == rec.get("correct_letter") else "competing" if a == rec.get("competing_letter") else "other_incorrect"
    return {"invalid": "invalid", "empty": "empty"}.get(rec.get("status"), "malformed")


def flr_v1_outcome_of(rec: dict) -> str:
    if outcome(rec) == "refusal":
        return "refusal"
    a = flr_v1_letter(rec.get("response"))
    return "none" if a is None else "correct" if a == rec.get("correct_letter") else "competing" if a == rec.get("competing_letter") else "other_incorrect"


def load(runs_dir: str):
    rows = []
    for f in sorted(glob.glob(os.path.join(runs_dir, "v0.1[0-5]*", "**", "records", "*.json"), recursive=True)):
        rec = json.load(open(f, encoding="utf-8"))
        if not isinstance(rec, dict) or rec.get("schema") != "bleediff.hosted-answer/1":
            continue
        rel = os.path.relpath(f, runs_dir).replace(os.sep, "/")
        resp = rec.get("response") or {}
        usage = rec.get("usage") or {}
        rid = resp.get("id") or f"noid:{rel}"
        rows.append({"run": rel.split("/")[0].split("-")[0], "stage": rel.split("/")[1] if rel.count("/") > 2 else "", "model": rec.get("model"), "cell": rec.get("cell"),
                     "condition": rec.get("condition"), "pass": rec.get("pass"), "prompt_sha256": rec.get("prompt_sha256"), "strict_status": rec.get("status"), "strict_outcome": outcome(rec),
                     "stop_reason": resp.get("stop_reason"), "refusal_category": (resp.get("stop_details") or {}).get("category"), "input_tokens": usage.get("promptTokenCount"),
                     "output_tokens": (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0), "flr_v1_outcome": flr_v1_outcome_of(rec),
                     "record_id": "rec-" + hashlib.sha256(rid.encode()).hexdigest()[:12], "_rid": rid})
    return rows


def main(argv) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    runs_dir, out_dir = argv[1:]
    rows = load(runs_dir)
    ids = collections.Counter(r["_rid"] for r in rows)
    dup = sorted(k for k, v in ids.items() if v > 1)
    if dup:
        print(f"ERROR: {len(dup)} provider message id(s) appear in more than one record: possible double-counting, refusing to write")
        return 1
    os.makedirs(out_dir, exist_ok=False)
    cols = ["run", "stage", "model", "cell", "condition", "pass", "prompt_sha256", "strict_status", "strict_outcome", "stop_reason", "refusal_category", "input_tokens", "output_tokens",
            "flr_v1_outcome", "record_id"]
    with open(os.path.join(out_dir, "raw-record-index.csv"), "x", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["run"], r["stage"], r["model"], r["cell"], str(r["condition"]), r["pass"] or 0)))
    by_run = collections.Counter(r["run"] for r in rows)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["model"], r["cell"], r["prompt_sha256"])].append(r)
    summary = {"unique_raw_records": len(rows), "by_run": dict(by_run), "duplicate_message_ids": 0, "groups": []}
    for (model, cell, sha), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], -len(kv[1]))):
        summary["groups"].append({"model": model, "cell": cell, "prompt_sha256": sha, "records": len(rs), "strict_outcomes": dict(collections.Counter(r["strict_outcome"] for r in rs)),
                                  "by_run": dict(collections.Counter(r["run"] for r in rs)), "condition_labels": sorted({f"{r['run']}:{r['condition']}" for r in rs})})
    with open(os.path.join(out_dir, "reconciliation.json"), "x", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=1)
    print(f"{len(rows)} unique raw request records; no message id repeats; by run {dict(sorted(by_run.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
