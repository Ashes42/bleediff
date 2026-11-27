"""v0.7.0 release checks, all offline: version and documentation agree, the release notes name the 13 measurements and the five states and make no live-validation claim, the published Research 002
repository is linked prominently, the package has zero third-party dependencies, and the experimental-design engine (a development milestone kept on its own branch) is NOT part of this release.
No network, key or model."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tomllib
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bleediff  # noqa: E402
from bleediff import telemetry as T  # noqa: E402

R002 = "https://github.com/Ashes42/bleediff-research-002"


def read(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as fh:
        return fh.read()


def cli(*args):
    p = subprocess.run([sys.executable, "-m", "bleediff", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    return p.returncode, p.stdout + p.stderr


class TestVersionAndDocs(unittest.TestCase):
    def test_version_changelog_release_notes_and_readme_agree(self):
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
            proj = tomllib.load(fh)["project"]
        self.assertEqual(proj["version"], "0.7.0")
        self.assertEqual(proj["dependencies"], [])
        self.assertEqual(bleediff.__version__, "0.7.0")
        cl = read("CHANGELOG.md")
        self.assertLess(cl.index("## [0.7.0]"), cl.index("## [0.6.0]"))
        for stale in ("no version bump", "Unreleased", "feature branch", "Pending for the next release", "Phase 1", "Phase 2"):
            self.assertNotIn(stale, cl, stale)
        readme = read("README.md")
        for needle in ("**0.7.0**", "bleediff 0.7.0", "RELEASE_NOTES_v0.7.0.md", "What is in v0.7.0", "bleediff behavior"):
            self.assertIn(needle, readme, needle)
        for stale in ("feature branch", "bleediff 0.6.0", "Phase 1"):
            self.assertNotIn(stale, readme, stale)
        self.assertTrue(read("RELEASE_NOTES_v0.7.0.md").startswith("# Bleediff v0.7.0 release notes"))
        self.assertNotIn("feature branch", read("docs", "BEHAVIORAL_TELEMETRY.md").split("\n")[2])

    def test_cli_reports_the_version_and_the_behavior_command(self):
        code, text = cli("--version")
        self.assertEqual((code, text.strip()), (0, "bleediff 0.7.0"))
        code, text = cli("behavior", "report", "--help")
        self.assertEqual(code, 0)
        self.assertIn("--cost-from-usage", text)
        code, text = cli("--help")
        self.assertIn("behavior", text)

    def test_the_published_research_002_repository_is_linked_prominently(self):
        readme = read("README.md")
        self.assertIn(R002, readme)
        self.assertLess(readme.index(R002), readme.index("## What is in v0.7.0"))              # before the feature description: prominent
        self.assertLess(readme.index(R002), readme.index("## Install"))
        for doc in ("RELEASE_NOTES_v0.7.0.md", os.path.join("docs", "BEHAVIORAL_TELEMETRY.md"), "CHANGELOG.md"):
            self.assertIn(R002, read(doc), doc)

    def test_release_notes_name_every_measurement_and_state_and_do_not_overclaim(self):
        notes = " ".join(read("RELEASE_NOTES_v0.7.0.md").split())
        for name in T.MEASURES:
            self.assertIn("`" + name + "`", notes, name)
        self.assertEqual(len(T.MEASURES), 13)
        self.assertEqual(T.STATES, ("OBSERVED", "MISSING", "UNSUPPORTED", "INCONSISTENT", "INVALID"))
        for state in T.STATES:
            self.assertIn("**" + state + "**", notes, state)
        for needle in ("## What is new in v0.7.0", "## What already existed", "## Research 002 compatibility demonstration", "## Compatibility", "## Limitations", "exercised offline only",
                       "has not been validated through a new live provider experiment", "Not a security finding", "never converted to zero", "no composite behavioral score", "zero third-party dependencies",
                       "not a provider bill"):
            self.assertIn(needle.lower(), notes.lower(), needle)
        low = notes.lower()
        for bad in ("discovered a vulnerability", "we found a vulnerability", "jailbreak works", "bypasses the safeguard", "demonstrates that models", "design engine"):
            self.assertNotIn(bad, low, bad)
        self.assertIsNone(re.search(r"proven", low))

    def test_release_files_carry_no_private_paths_or_personal_addresses(self):
        for name in ("RELEASE_NOTES_v0.7.0.md", "CHANGELOG.md", "README.md", os.path.join("docs", "BEHAVIORAL_TELEMETRY.md"), os.path.join("docs", "ARCHITECTURE_V07_BEHAVIOR.md"),
                     os.path.join("examples", "behavior", "r002_compat_check.py")):
            text = read(name)
            for bad in ("C:\\Users", "/Users/", "@gmail.com", "sk-ant-", "api_key", "Ashley", "vanloo"):
                self.assertNotIn(bad, text, (name, bad))


class TestZeroDependencies(unittest.TestCase):
    def test_every_import_in_the_package_is_standard_library_or_local(self):
        std = set(sys.stdlib_module_names)
        offenders = []
        for d in ("bleediff", "research"):
            for f in sorted(os.listdir(os.path.join(ROOT, d))):
                if not f.endswith(".py"):
                    continue
                tree = ast.parse(read(d, f))
                for node in ast.walk(tree):
                    mods = []
                    if isinstance(node, ast.Import):
                        mods = [a.name.split(".")[0] for a in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        mods = [node.module.split(".")[0]]
                    offenders += [(d + "/" + f, m) for m in mods if m not in std and m not in ("bleediff", "research", "tests")]
        self.assertEqual(offenders, [])

    def test_the_new_modules_are_python_3_11_syntax(self):
        for mod in ("telemetry", "behavior"):
            ast.parse(read("bleediff", mod + ".py"), feature_version=(3, 11))


class TestPhase2IsNotInThisRelease(unittest.TestCase):
    """The experimental design engine stays on its own development branch; the public v0.7.0 keeps Bleediff small."""

    def test_no_design_engine_files(self):
        for rel in ("bleediff/design_spec.py", "bleediff/design_build.py", "bleediff/design_validate.py", "bleediff/design_contrasts.py", "bleediff/design_cli.py", "bleediff/counterbalance.py",
                    "bleediff/tokenparity.py", "bleediff/freeze.py", "bleediff/synthetic.py", "docs/EXPERIMENTAL_DESIGN.md", "docs/ARCHITECTURE_V07_DESIGN.md", "examples/conflict-locality",
                    "tests/test_v070_design.py"):
            self.assertFalse(os.path.exists(os.path.join(ROOT, rel)), rel)

    def test_no_design_commands_or_options(self):
        code, text = cli("experiment", "--help")
        self.assertEqual(code, 0)
        self.assertNotIn("design", text.lower())
        code, text = cli("behavior", "report", "--help")
        self.assertNotIn("--design", text)
        for doc in ("README.md", "CHANGELOG.md", "RELEASE_NOTES_v0.7.0.md", os.path.join("docs", "BEHAVIORAL_TELEMETRY.md")):
            self.assertIsNone(re.search(r"experimental design engine|experiment design|conflict.locality", read(doc), re.I), doc)


if __name__ == "__main__":
    unittest.main()
