"""Offline check of the hosted comparison bundle (no key, no network, no model). Usage: python examples/hosted-comparison/verify_comparison.py

1. Replays all five hosted sessions and compares the strict table in README.md (correct/competing/other/no-answer per condition and model, and non-answer kinds).
2. Verifies SHA256SUMS.txt.
3. Scans every file of the bundle for secrets, opaque thinking signatures, reasoning text, local paths, e-mail addresses and raw provider message ids.
Exit 0 only if everything holds.
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bleediff import answers as A  # noqa: E402

SESSIONS = {"Haiku 5.5": "haiku55", "Sonnet 5.5": "sonnet55", "Gemini 3.8 Flash": "../gemini-answer-only/flash", "Gemini 4 Carbon": "../gemini-answer-only/carbon",
            "GPT-4.1 mini": "../openai-answer-only/gpt41mini"}
CONDS = ("BASELINE", "PROSE", "RANDOM_AB", "RANDOM_OTHER", "COMPETING")
EXPECTED = {   # correct/competing/other/no-answer, strict primary
    "BASELINE": ["32/0/0/0"] * 5,
    "PROSE": ["31/0/0/1", "32/0/0/0", "32/0/0/0", "32/0/0/0", "32/0/0/0"],
    "RANDOM_AB": ["32/0/0/0", "20/0/0/12", "32/0/0/0", "32/0/0/0", "32/0/0/0"],
    "RANDOM_OTHER": ["20/0/0/12", "16/0/0/16", "32/0/0/0", "32/0/0/0", "32/0/0/0"],
    "COMPETING": ["9/23/0/0", "0/0/0/32", "0/32/0/0", "4/28/0/0", "2/30/0/0"]}
NON_ANSWERS = {"Haiku 5.5": {"ok": 147, "blocked": 12, "invalid": 1}, "Sonnet 5.5": {"ok": 100, "invalid": 60}}
PATTERNS = {"api key": re.compile(r"sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{20,}"), "bearer": re.compile(r"Bearer\s+[A-Za-z0-9._-]{16,}"),
            "thinking signature": re.compile(r'"(signature|thoughtSignature|thought_signature)":\s*"(?!\[REDACTED:)[^"]+"'),
            "reasoning text": re.compile(r'"type":\s*"thinking",\s*"thinking":\s*"(?!\[REDACTED:)[^"]+"'),
            "redacted_thinking data": re.compile(r'"type":\s*"redacted_thinking",\s*"data":\s*"(?!\[REDACTED:)[^"]+"'),
            "local path": re.compile(r"[A-Za-z]:\\\\|/Us" r"ers/|/home/[a-z]"), "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}"),
            "raw message id": re.compile(r'"id":\s*"(msg_|chatcmpl-)[A-Za-z0-9]+"|"responseId":\s*"(?!resp-)[^"]+"')}


def table():
    out, problems = {}, []
    for label, rel in SESSIONS.items():
        ses = A.read_session(os.path.join(HERE, rel))
        problems += [f"{label}: {p}" for p in ses["problems"]]
        an = A.analyze(ses["records"], ses["prompts"])
        out[label] = {c: "{correct}/{competing}/{other_incorrect}/{non_answers}".format(**an["by_condition"][c]) for c in CONDS}
        if label in NON_ANSWERS and an["status_counts"] != NON_ANSWERS[label]:
            problems.append(f"{label}: status counts {an['status_counts']} != {NON_ANSWERS[label]}")
    for c in CONDS:
        got = [out[m][c] for m in SESSIONS]
        if got != EXPECTED[c]:
            problems.append(f"{c}: {got} != {EXPECTED[c]}")
    return problems


def sums():
    problems, listed = [], {}
    with open(os.path.join(HERE, "SHA256SUMS.txt"), encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                h, p = ln.rstrip("\n").split("  ", 1)
                listed[p] = h
    files = [os.path.relpath(os.path.join(d, f), HERE).replace(os.sep, "/") for d, _, fs in os.walk(HERE) for f in fs if "__pycache__" not in d and f != "README.md" and f != "SHA256SUMS.txt"]
    for f in sorted(files):
        if f not in listed:
            problems.append(f"not listed in SHA256SUMS.txt: {f}")
        elif hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest() != listed[f]:
            problems.append(f"hash mismatch: {f}")
    problems += [f"listed but missing: {p}" for p in listed if p not in files]
    return problems


def scan():
    problems = []
    for d, _, fs in os.walk(HERE):
        if "__pycache__" in d:
            continue
        for f in fs:
            if f.endswith((".py",)):
                continue                                       # the verifier itself names the patterns
            text = open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
            for name, rx in PATTERNS.items():
                if rx.search(text):
                    problems.append(f"{name} in {os.path.relpath(os.path.join(d, f), HERE)}")
    return problems


if __name__ == "__main__":
    probs = table() + sums() + scan()
    for p in probs:
        print("PROBLEM:", p)
    print("OK: strict table, checksums and public-safety scan all pass" if not probs else f"{len(probs)} problem(s)")
    sys.exit(1 if probs else 0)
