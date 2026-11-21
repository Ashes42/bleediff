"""v0.4.0 release checks: the four-mapping evidence bundle reproduces the documented numbers offline (no model, no network),
its hashes and hygiene hold, and the docs state the findings with their limits."""

from __future__ import annotations

import atexit
import hashlib
import io
import json
import os
import re
import runpy
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRST_NAME = "".join(chr(c) for c in (65, 115, 104, 108, 101, 121))   # the maintainer's first name, built at run time so no new literal enters history
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from tests.test_bleediff import run_cli  # noqa: E402

BUNDLE = os.path.join(ROOT, "examples", "four-mapping")
PERSONAL = re.compile(r"(@[A-Za-z0-9-]+\.(com|org|net|io)|[A-Za-z]:\\\\|[A-Za-z]:\\|/Users/|/home/|AppData|\.cache|huggingface|api[_-]?key\s*[:=]|password\s*[:=])", re.I)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def bundle_files():
    for r, _d, fs in os.walk(BUNDLE):
        for f in fs:
            yield os.path.join(r, f)


_CACHE: dict = {}


def build_atlas():
    if not _CACHE:
        d = tempfile.mkdtemp(prefix="bleediff-fm-test-")
        atexit.register(shutil.rmtree, d, True)
        out = os.path.join(d, "report")
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network access attempted")):
            code, text = run_cli("atlas", "--manifest", os.path.join(BUNDLE, "atlas-input.json"), "--out", out)
        _CACHE.update(code=code, text=text, out=out)
    return _CACHE


class TestBundle(unittest.TestCase):
    def test_sha256sums_cover_exactly_the_data_files(self):
        listed = {}
        for line in read(os.path.join(BUNDLE, "SHA256SUMS.txt")).splitlines():
            digest, rel = line.split("  ", 1)
            listed[rel] = digest
        found = {os.path.relpath(p, BUNDLE).replace("\\", "/") for p in bundle_files()} - {"SHA256SUMS.txt", "README.md", "verify_summary.py"}
        self.assertEqual(set(listed), found)
        self.assertEqual(len(listed), 161)
        for rel, digest in listed.items():
            with open(os.path.join(BUNDLE, rel), "rb") as fh:
                self.assertEqual(hashlib.sha256(fh.read()).hexdigest(), digest, rel)

    def test_bundle_is_small_lf_text_without_personal_or_local_data_or_models(self):
        files = list(bundle_files())
        self.assertLess(sum(os.path.getsize(p) for p in files), 2 * 1024 * 1024)
        for p in files:
            with open(p, "rb") as fh:
                data = fh.read()
            self.assertNotIn(b"\r\n", data, p)
            text = data.decode("utf-8")
            self.assertIsNone(PERSONAL.search(text), p)
            self.assertNotIn(FIRST_NAME, text, p)
        self.assertFalse([p for p in files if p.endswith((".gguf", ".safetensors", ".log"))])
        self.assertIn("examples/four-mapping/** -text", read(os.path.join(ROOT, ".gitattributes")))

    def test_atlas_validates_all_sources_offline(self):
        r = build_atlas()
        self.assertEqual(r["code"], 0, r["text"])
        a = json.loads(read(os.path.join(r["out"], "atlas.json")))
        self.assertEqual(len(a["sources"]), 48)
        self.assertTrue(all(s["ok"] for s in a["sources"]))
        self.assertEqual(len(a["cells"]), 48)
        self.assertEqual(sum(len(c["candidates"]) for c in a["cells"]), 192)

    def test_verify_summary_passes_on_the_bundle(self):
        r = build_atlas()
        buf = io.StringIO()
        with redirect_stdout(buf), mock.patch.object(sys, "argv", ["verify_summary.py", os.path.join(r["out"], "atlas.json")]):
            with self.assertRaises(SystemExit) as cm:
                runpy.run_path(os.path.join(BUNDLE, "verify_summary.py"), run_name="__main__")
        out = buf.getvalue()
        self.assertEqual(cm.exception.code, 0, out)
        self.assertNotIn("FAIL", out)
        self.assertGreaterEqual(out.count("PASS"), 17)
        self.assertIn("ALL DOCUMENTED STATEMENTS VERIFIED", out)

    def test_verify_summary_fails_when_a_measurement_is_tampered_with(self):
        r = build_atlas()
        a = json.loads(read(os.path.join(r["out"], "atlas.json")))
        for c in a["cells"]:
            if c["model_id"] == "llama3b":
                for k in c["candidates"]:
                    if k["category"] == "claim":
                        k["shift_toward_incorrect"] = -1.0
        d = tempfile.mkdtemp(prefix="bleediff-fm-tamper-")
        atexit.register(shutil.rmtree, d, True)
        p = os.path.join(d, "atlas.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(a, fh)
        buf = io.StringIO()
        with redirect_stdout(buf), mock.patch.object(sys, "argv", ["verify_summary.py", p]):
            with self.assertRaises(SystemExit) as cm:
                runpy.run_path(os.path.join(BUNDLE, "verify_summary.py"), run_name="__main__")
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("FAIL", buf.getvalue())


class TestDocs(unittest.TestCase):
    def test_release_notes_and_readme_state_the_findings_with_their_limits(self):
        notes = " ".join(read(os.path.join(ROOT, "RELEASE_NOTES_v0.4.0.md")).split())
        for needle in ("Not a security finding", "24 of 24", "23 of 24", "9 under m1 and 1 under m2", "semantic-answer sensitivity in the tested Llama configuration",
                       "answer-letter sensitivity in the tested Qwen configuration", "No universal model-family mechanism", "not independent samples", "Four mappings (not 24)",
                       "verify_summary.py"):
            self.assertIn(needle, notes, needle)
        readme = " ".join(read(os.path.join(ROOT, "README.md")).split())
        for needle in ("examples/four-mapping/verify_summary.py", "semantic-answer sensitivity", "answer-letter sensitivity", "model-family mechanism", "tenant-isolation vulnerability"):
            self.assertIn(needle, readme, needle)
        summary = read(os.path.join(ROOT, "docs", "FOUR_MAPPING_SUMMARY.md"))
        self.assertIn("examples/four-mapping", summary)
        self.assertNotIn("are **not** part of the repository", summary)

    def test_documented_reproduction_commands_exist(self):
        cmd = "python -m bleediff atlas --manifest examples/four-mapping/atlas-input.json --out runs/four-mapping"
        for name in ("README.md", os.path.join("examples", "four-mapping", "README.md"), "RELEASE_NOTES_v0.4.0.md", os.path.join("docs", "FOUR_MAPPING_SUMMARY.md")):
            self.assertIn(cmd, read(os.path.join(ROOT, name)), name)


if __name__ == "__main__":
    unittest.main()
