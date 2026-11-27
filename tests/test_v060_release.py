"""v0.6.0 release checks, all offline: version and documentation agree, the release notes keep new and existing features apart and make no live-validation claim, the experiment
authoring workflow (global and per-question specs, hosted loader) works from a fresh install layout, and the tag-time files are present. No network, key or model."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bleediff  # noqa: E402
from bleediff import answers as AN  # noqa: E402
from bleediff import experiment as EX  # noqa: E402


def read(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as fh:
        return fh.read()


def cli(*args):
    p = subprocess.run([sys.executable, "-m", "bleediff", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    return p.returncode, p.stdout + p.stderr


class TestVersionAndDocs(unittest.TestCase):
    def test_version_changelog_release_notes_and_readme_agree(self):
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
            self.assertEqual(tomllib.load(fh)["project"]["version"], "0.7.0")
        self.assertEqual(bleediff.__version__, "0.7.0")
        cl = read("CHANGELOG.md")
        self.assertLess(cl.index("## [0.6.0]"), cl.index("## [0.5.0]"))
        self.assertLess(cl.index("## [0.5.0]"), cl.index("## [0.4.0]"))
        for stale in ("no version bump", "Unreleased", "Pending for the next release", "after 0.5.0"):
            self.assertNotIn(stale, cl, stale)
        readme = read("README.md")
        self.assertIn("**0.7.0**", readme)
        self.assertIn("bleediff 0.7.0", readme)
        self.assertIn("RELEASE_NOTES_v0.6.0.md", readme)
        self.assertNotIn("bleediff 0.5.0", readme)
        self.assertTrue(read("RELEASE_NOTES_v0.6.0.md").startswith("# Bleediff v0.6.0 release notes"))

    def test_cli_reports_the_version(self):
        code, text = cli("--version")
        self.assertEqual(code, 0)
        self.assertEqual(text.strip(), "bleediff 0.7.0")

    def test_release_notes_separate_new_from_existing_and_do_not_overclaim(self):
        notes = " ".join(read("RELEASE_NOTES_v0.6.0.md").split())
        for needle in ("## What is new in v0.6.0", "## What already existed", "exercised offline only", "has not been validated through a live provider experiment",
                       "Not a security finding", "Strict classification is unchanged", "FLR-v2", "byte-identical", "per-question", "not a provider bill", "DEMONSTRATION"):
            self.assertIn(needle.lower(), notes.lower(), needle)
        for bad in ("discovered a vulnerability", "we found a vulnerability", "jailbreak works", "bypasses the safeguard", "validated through live", "proven"):
            self.assertNotIn(bad, notes.lower().replace("has not been validated through a live", ""), bad)
        # historical release notes are untouched in spirit: v0.5.0 still describes itself as v0.5.0
        self.assertTrue(read("RELEASE_NOTES_v0.5.0.md").startswith("# Bleediff v0.5.0 release notes"))

    def test_release_files_carry_no_private_paths_or_personal_addresses(self):
        for name in ("RELEASE_NOTES_v0.6.0.md", "CHANGELOG.md", os.path.join("examples", "experiment-multi-question", "experiment.json")):
            text = read(name)
            for bad in ("C:\\Users", "/Users/", "@gmail.com", "sk-ant-", "api_key"):
                self.assertNotIn(bad, text, (name, bad))


class TestAuthoringWorkflowFromTheRepository(unittest.TestCase):
    def test_starter_and_per_question_example_validate_write_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            code, text = cli("experiment", "init", os.path.join(d, "start"))
            self.assertEqual(code, 0, text)
            out = os.path.join(d, "start", "prompts.json")
            code, text = cli("experiment", "validate", os.path.join(d, "start", "experiment.json"), "--write-prompts", out, "--provider", "openai", "--price-in", "1", "--price-out", "5")
            self.assertEqual(code, 0, text)
            self.assertEqual(len(AN.load_prompts(out)["items"]), 12)
            mq = os.path.join(ROOT, "examples", "experiment-multi-question", "experiment.json")
            out2 = os.path.join(d, "mq.json")
            code, text = cli("experiment", "validate", mq, "--write-prompts", out2, "--provider", "anthropic", "--price-in", "1", "--price-out", "5")
            self.assertEqual(code, 0, text)
            self.assertNotIn("WARNING", text)
            loaded = AN.load_prompts(out2)
            self.assertEqual(len(loaded["items"]), 36)
            self.assertEqual(loaded["conditions"], ["BASELINE", "TREATMENT_COMPETING_CLAIM", "CONTROL_NEUTRAL"])

    def test_the_documented_example_spec_is_per_question_and_the_starter_is_global(self):
        with open(os.path.join(ROOT, "examples", "experiment-multi-question", "experiment.json"), encoding="utf-8") as fh:
            spec = json.load(fh)
        self.assertTrue(EX.per_question_mode(spec))
        self.assertFalse(EX.per_question_mode(EX.STARTER))
        self.assertEqual(EX.check_spec(spec), [])
        self.assertEqual(EX.check_spec(EX.STARTER), [])


if __name__ == "__main__":
    unittest.main()
