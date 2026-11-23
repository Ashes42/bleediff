"""v0.5.0 release checks, all offline: version and documentation, the hosted comparison bundle (replay, checksums, public-safety scan), the reconciled evidence index, the
secondary extraction rule FLR-v1, and the duplicate-record guard of research/reconcile_evidence.py. No network, key or model."""

from __future__ import annotations

import collections
import csv
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bleediff  # noqa: E402
from research import reconcile_evidence as RE  # noqa: E402

CMP = os.path.join(ROOT, "examples", "hosted-comparison")
INDEX = os.path.join(ROOT, "docs", "evidence", "raw-index-v0.10-v0.15")


def read(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as fh:
        return fh.read()


class TestVersionAndDocs(unittest.TestCase):
    def test_version_changelog_release_notes_and_readme_agree(self):
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
            self.assertEqual(tomllib.load(fh)["project"]["version"], "0.6.0")
        self.assertEqual(bleediff.__version__, "0.6.0")
        cl = read("CHANGELOG.md")
        self.assertLess(cl.index("## [0.5.0]"), cl.index("## [0.4.0]"))
        self.assertIn("**0.6.0**", read("README.md"))
        self.assertIn("RELEASE_NOTES_v0.5.0.md", read("README.md"))
        self.assertIn("Not a security finding", read("RELEASE_NOTES_v0.5.0.md"))

    def test_docs_keep_the_four_distinctions_and_no_overclaiming(self):
        readme, guide = " ".join(read("README.md").split()), " ".join(read("docs", "HOSTED_MODELS.md").split())
        for text in (readme, guide):
            for needle in ("DIRECT_PROBABILITY", "ANSWER_ONLY", "refusal", "incorrect", "strict", "secondary", "estimate", "billing"):
                self.assertIn(needle.lower(), text.lower(), needle)
        for name in ("README.md", os.path.join("docs", "CASE_NOTE_HOSTED_REFUSALS_AND_FORMAT.md"), os.path.join("docs", "HOSTED_MODELS.md"), os.path.join("docs", "EVIDENCE_RECONCILIATION.md")):
            for bad in ("discovered a vulnerability", "we found a vulnerability", "jailbreak works", "bypasses the safeguard", "safety bypass"):
                self.assertNotIn(bad, read(name).lower(), (name, bad))
        self.assertIn("**not** a jailbreak, an attack technique or a security vulnerability", read("docs", "CASE_NOTE_HOSTED_REFUSALS_AND_FORMAT.md"))

    def test_extraction_rule_is_named_flr_v1_and_s1_stays_the_filler_identifier(self):
        for name in ("README.md", "RELEASE_NOTES_v0.5.0.md", "CHANGELOG.md", os.path.join("docs", "HOSTED_MODELS.md"), os.path.join("docs", "EVIDENCE_RECONCILIATION.md"),
                     os.path.join("docs", "CASE_NOTE_HOSTED_REFUSALS_AND_FORMAT.md"), os.path.join("examples", "hosted-comparison", "README.md")):
            text = read(name)
            for stale in ("rule S1", "Rule S1", "rule (S1)", "s1_final_line"):
                self.assertNotIn(stale, text, (name, stale))
        t = read("docs", "EVIDENCE_RECONCILIATION.md")
        self.assertIn("FLR-v1", t)
        self.assertIn("Final-Line Recovery", t)
        self.assertIn("experimental filler identifier S1", t)
        self.assertIn("flr_v1_outcome", open(os.path.join(INDEX, "raw-record-index.csv"), encoding="utf-8").readline())

    def test_budget_documentation_does_not_promise_a_billing_limit(self):
        t = " ".join((read("README.md") + read("docs", "HOSTED_MODELS.md")).split())
        self.assertIn("not a guaranteed billing limit", t)
        self.assertIn("one more full worst-case attempt", t)
        self.assertNotIn("guaranteed to cap your bill", t)

    def test_each_provider_has_usage_documentation(self):
        t = read("README.md") + read("docs", "HOSTED_MODELS.md")
        for needle in ("bleediff gemini", "bleediff openai", "bleediff anthropic", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "--replay", "--budget"):
            self.assertIn(needle, t, needle)


class TestPublicBundles(unittest.TestCase):
    def test_hosted_comparison_verifies_offline(self):
        p = subprocess.run([sys.executable, os.path.join(CMP, "verify_comparison.py")], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("OK", p.stdout)

    def test_no_public_record_holds_signatures_reasoning_or_raw_ids(self):
        bad = re.compile(r'"(signature|thoughtSignature|thought_signature)":\s*"(?!\[REDACTED:)[^"]+"|"id":\s*"(msg_|chatcmpl-)[A-Za-z0-9]+"|"responseId":\s*"(?!resp-)[^"]+"'
                         r'|"type":\s*"thinking",\s*"thinking":\s*"(?!\[REDACTED:)[^"]+"')
        files = glob.glob(os.path.join(ROOT, "examples", "*answer-only", "*", "records", "*.json")) + glob.glob(os.path.join(CMP, "*", "records", "*.json"))
        self.assertGreaterEqual(len(files), 800)
        for f in files:
            with open(f, encoding="utf-8") as fh:
                self.assertIsNone(bad.search(fh.read()), f)

    def test_claude_bundle_replays_with_the_documented_failure_modes(self):
        from bleediff import answers as AN
        h = AN.read_session(os.path.join(CMP, "haiku55"))
        s = AN.read_session(os.path.join(CMP, "sonnet55"))
        self.assertEqual((h["problems"], s["problems"]), ([], []))
        ah, as_ = AN.analyze(h["records"], h["prompts"]), AN.analyze(s["records"], s["prompts"])
        self.assertEqual(ah["status_counts"], {"ok": 147, "blocked": 12, "invalid": 1})
        self.assertEqual(as_["status_counts"], {"ok": 100, "invalid": 60})
        refusals = [r for r in h["records"] if r["status"] == "blocked"]
        self.assertEqual({(r["condition"], r["response"]["stop_details"]["category"]) for r in refusals}, {("RANDOM_OTHER", "bio")})
        self.assertEqual(as_["by_condition"]["COMPETING"]["non_answers"], 32)       # strict: Sonnet's COMPETING row has no answers


class TestEvidenceIndex(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(INDEX, "raw-record-index.csv"), encoding="utf-8", newline="") as fh:
            self.rows = list(csv.DictReader(fh))

    def test_index_has_522_unique_records_and_no_provider_ids(self):
        self.assertEqual(len(self.rows), 522)
        self.assertEqual(len({r["record_id"] for r in self.rows}), 522)
        self.assertEqual(dict(collections.Counter(r["run"] for r in self.rows)), {"v0.10": 161, "v0.11": 161, "v0.12": 80, "v0.13": 72, "v0.14": 44, "v0.15": 4})
        self.assertNotIn("msg_", open(os.path.join(INDEX, "raw-record-index.csv"), encoding="utf-8").read())

    def test_q2_m1_counts_by_unique_request(self):
        c = collections.defaultdict(collections.Counter)
        for r in self.rows:
            if r["cell"] == "q2-invoice-m1" and "haiku" in r["model"]:
                c[r["prompt_sha256"][:8]][r["strict_outcome"]] += 1
        self.assertEqual(c["46f315e3"], {"refusal": 10})          # F0 / RANDOM_OTHER
        self.assertEqual(c["a60f9ede"], {"correct": 7})           # S1 (filler)
        self.assertEqual(c["27d0813d"], {"refusal": 4})           # S1g1: 1 stage-1 draw + 3 confirmation repeats
        sonnet = [r for r in self.rows if r["cell"] == "q2-invoice-m1" and r["prompt_sha256"].startswith("46f315e3") and "sonnet" in r["model"]]
        self.assertEqual([r["strict_outcome"] for r in sonnet], ["correct", "correct"])    # same prompt text, different model: never pool by prompt alone

    def test_refusals_are_not_counted_as_incorrect_and_strict_columns_are_unchanged(self):
        refusals = [r for r in self.rows if r["strict_outcome"] == "refusal"]
        self.assertEqual(len(refusals), 82)
        self.assertEqual({(r["stop_reason"], r["refusal_category"], r["output_tokens"]) for r in refusals}, {("refusal", "bio", "0")})
        sonnet = collections.Counter((r["strict_outcome"], r["flr_v1_outcome"]) for r in self.rows if "sonnet" in r["model"] and r["run"] == "v0.11" and r["stage"] == "pilot")
        self.assertEqual(dict(sonnet), {("correct", "correct"): 100, ("invalid", "competing"): 30, ("invalid", "correct"): 30})
        invalid = sum(v for (a, _), v in sonnet.items() if a == "invalid")
        self.assertEqual(invalid, 60)                              # the strict status is never rewritten by FLR-v1


class TestReconcileTool(unittest.TestCase):
    def test_secondary_rule_flr_v1(self):
        r = lambda t: {"content": [{"type": "text", "text": t}]}
        self.assertEqual(RE.flr_v1_letter(r("The evidence says so.\n\n**B**")), "B")
        self.assertEqual(RE.flr_v1_letter(r("reasoning\nB")), "B")
        self.assertIsNone(RE.flr_v1_letter(r("B is right because\nit holds.")))
        self.assertIsNone(RE.flr_v1_letter(r("")))
        self.assertIsNone(RE.flr_v1_letter({"content": [{"type": "thinking", "thinking": "B"}]}))   # thinking is never visible text

    def test_duplicate_message_ids_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            for run in ("v0.10-a", "v0.12-b"):
                os.makedirs(os.path.join(d, run, "records"))
                rec = {"schema": "bleediff.hosted-answer/1", "model": "m", "cell": "c", "condition": "X", "pass": 1, "prompt_sha256": "0" * 64, "status": "ok", "answer": "A", "correct_letter": "A",
                       "competing_letter": "B", "usage": {}, "response": {"id": "msg_same", "content": [{"type": "text", "text": "A"}], "stop_reason": "end_turn"}}
                with open(os.path.join(d, run, "records", "r.json"), "w", encoding="utf-8") as fh:
                    json.dump(rec, fh)
            out = os.path.join(d, "out")
            self.assertEqual(RE.main(["x", d, out]), 1)
            self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
