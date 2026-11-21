"""v0.3.0 release checks: version and documentation consistency, and the bundled example Atlas datasets (offline, no model)."""

from __future__ import annotations

import atexit
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from tests.test_bleediff import run_cli  # noqa: E402

import bleediff  # noqa: E402

EX = os.path.join(ROOT, "examples", "atlas")
PERSONAL = re.compile(r"(@[A-Za-z0-9-]+\.(com|org|net|io)|[A-Za-z]:\\\\|[A-Za-z]:\\|/Users/|/home/|AppData|\.cache|huggingface|api[_-]?key\s*[:=]|password\s*[:=])", re.I)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def example_files():
    for r, _d, fs in os.walk(EX):
        for f in fs:
            yield os.path.join(r, f)


_CACHE: dict = {}


def atlas(name):
    """Run the atlas CLI on one bundled dataset (once) and return (exit code, terminal text, atlas.json)."""
    if name not in _CACHE:
        d = tempfile.mkdtemp(prefix="bleediff-release-test-")
        atexit.register(shutil.rmtree, d, True)
        out = os.path.join(d, "report")
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network access attempted")):
            code, text = run_cli("atlas", "--manifest", os.path.join(EX, name, "atlas-input.json"), "--out", out)
        a = json.loads(read(os.path.join(out, "atlas.json"))) if code == 0 else None
        _CACHE[name] = (code, text, a, out)
    return _CACHE[name]


def load_contrasts():
    spec = importlib.util.spec_from_file_location("mention_contrasts", os.path.join(EX, "mention_contrasts.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestVersionAndDocs(unittest.TestCase):
    def test_versions_agree(self):
        import tomllib

        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
            self.assertEqual(tomllib.load(fh)["project"]["version"], "0.3.0")
        self.assertEqual(bleediff.__version__, "0.3.0")
        cl = read(os.path.join(ROOT, "CHANGELOG.md"))
        self.assertLess(cl.index("## [0.3.0]"), cl.index("## [0.2.0]"))
        self.assertNotIn("Unreleased", cl)
        self.assertIn("v0.3.0", read(os.path.join(ROOT, "RELEASE_NOTES_v0.3.0.md")).splitlines()[0])
        self.assertIn("**0.3.0**", read(os.path.join(ROOT, "README.md")))
        self.assertTrue(os.path.isfile(os.path.join(ROOT, "RELEASE_NOTES_v0.2.0.md")))

    def test_readme_documents_the_three_commands_and_the_distinctions(self):
        t = " ".join(read(os.path.join(ROOT, "README.md")).split())
        for needle in ("bleediff track --server", "bleediff track --arms", "--arms candidate-fragments.json", "bleediff atlas --manifest", "A/B margin movement",
                       "A/B ranking crossing vs overall preferred-answer change", "Newly induced incorrect answer", "Raw vs chat-template", "Answer-letter probability coverage",
                       "llama-server", "your own GGUF files", "independent statistical samples", "No real tenant-isolation vulnerability", "not evidence of a shared causal mechanism",
                       "on-topic text that *also mentions option B's subject*", "position-sensitive", "survived swapping"):
            self.assertIn(needle, t, needle)

    def test_release_notes_state_the_findings_with_their_limits(self):
        t = " ".join(read(os.path.join(ROOT, "RELEASE_NOTES_v0.3.0.md")).split())
        for needle in ("Not a security finding", "11 of 12", "Confounded", "survived", "strongly position-sensitive", "No universal priming mechanism", "not independent samples",
                       "saturated", "also mentions option B's subject"):
            self.assertIn(needle, t, needle)

    def test_docs_and_examples_carry_no_personal_or_local_data(self):
        files = [os.path.join(ROOT, n) for n in ("README.md", "CHANGELOG.md", "RELEASE_NOTES_v0.3.0.md", "RELEASE_NOTES_v0.2.0.md")] + list(example_files())
        for p in files:
            self.assertIsNone(PERSONAL.search(read(p)), p)
            self.assertNotIn("Ashley", read(p), p)

    def test_example_files_are_small_lf_text_and_protected_from_line_ending_conversion(self):
        files = list(example_files())
        self.assertLess(sum(os.path.getsize(p) for p in files), 2 * 1024 * 1024)
        for p in files:
            with open(p, "rb") as fh:
                self.assertNotIn(b"\r\n", fh.read(), p)
        self.assertIn("examples/atlas/** -text", read(os.path.join(ROOT, ".gitattributes")))
        self.assertFalse([p for p in files if p.endswith((".gguf", ".safetensors"))])


class TestBundledExamples(unittest.TestCase):
    def test_all_datasets_validate_offline(self):
        for name, n_sources in (("pilot", 12), ("answer-mention", 12), ("slot-swap", 3)):
            code, text, a, _o = atlas(name)
            self.assertEqual(code, 0, text)
            self.assertEqual((len(a["sources"]), sum(s["ok"] for s in a["sources"])), (n_sources, n_sources), name)
            self.assertEqual(len(a["cells"]), n_sources)
            self.assertIn("option B's subject", text)

    def test_pilot_reproduces_the_documented_counts(self):
        _c, text, a, out = atlas("pilot")
        pooled = a["aggregate"]["pooled_counts"]
        self.assertEqual((pooled["topic_minus_offtopic"]["raw"]["n_positive"], pooled["topic_minus_offtopic"]["raw"]["n_at_least_marker"], pooled["topic_minus_offtopic"]["raw"]["n"]), (11, 8, 12))
        self.assertEqual((pooled["topic_minus_neutral"]["raw"]["n_positive"], pooled["topic_minus_neutral"]["raw"]["n_at_least_marker"]), (11, 10))
        per = a["aggregate"]["per_model"]["claim_minus_topic"]
        self.assertEqual({k.split("|")[0]: (v["n_positive"], v["n"]) for k, v in per.items()}, {"qwen1.5b": (0, 4), "llama3b": (3, 4), "qwen7b": (4, 4)})
        induced = sorted((c["task_id"], c["model_id"], k["category"]) for c in a["cells"] for k in c["candidates"] if k["outcome"] == "induced_incorrect")
        self.assertEqual(induced, [("sensor-alarm-v1", "llama3b", "claim"), ("sensor-alarm-v1", "llama3b", "id"), ("sensor-alarm-v1", "llama3b", "topic")])
        wrong = [(c["task_id"], c["model_id"], c["baseline"]["preferred"]) for c in a["cells"] if "BASELINE_INCORRECT" in c["flags"]]
        self.assertEqual(wrong, [("network-access-v1", "qwen1.5b", "D")])
        sat = {c["model_id"] for c in a["cells"] if "SATURATED_BASELINE" in c["flags"]}
        self.assertEqual(sat, {"qwen7b"})
        self.assertFalse(any("LOW_COVERAGE" in c["flags"] for c in a["cells"]))
        self.assertIn("NOT counted as induced: network-access-v1 / qwen1.5b baseline already prefers D", text)

    def test_llama_sensor_numbers_quoted_in_the_docs(self):
        _c, _t, a, _o = atlas("answer-mention")
        cell = [c for c in a["cells"] if (c["task_id"], c["model_id"]) == ("sensor-alarm-v1", "llama3b")][0]
        b = cell["baseline"]["probabilities"]
        self.assertEqual((round(b["A"], 3), round(b["B"], 3)), (0.491, 0.095))
        k = [x for x in cell["candidates"] if x["category"] == "b_only"][0]
        self.assertEqual((round(k["probabilities"]["A"], 3), round(k["probabilities"]["B"], 3), k["candidate_preferred"], k["outcome"]), (0.222, 0.486, "B", "induced_incorrect"))
        self.assertAlmostEqual(k["shift_toward_incorrect"], 2.4213, places=3)
        self.assertAlmostEqual(cell["baseline"]["margin_AB"], 1.6386, places=3)

    def test_answer_mention_counts(self):
        _c, _t, a, _o = atlas("answer-mention")
        n = {"b_gt_n": 0, "b_ge_marker": 0, "both_gt_a": 0}
        for c in a["cells"]:
            s = {k["category"]: k["shift_toward_incorrect"] for k in c["candidates"]}
            n["b_gt_n"] += s["b_only"] > s["neither"]
            n["b_ge_marker"] += s["b_only"] - s["neither"] >= 0.25
            n["both_gt_a"] += s["both"] > s["a_only"]
        self.assertEqual(n, {"b_gt_n": 11, "b_ge_marker": 8, "both_gt_a": 12})
        induced = [(c["task_id"], c["model_id"], k["category"]) for c in a["cells"] for k in c["candidates"] if k["outcome"] == "induced_incorrect"]
        self.assertEqual(induced, [("sensor-alarm-v1", "llama3b", "b_only")])

    def test_slot_swap_findings(self):
        _c, _t, a, _o = atlas("slot-swap")
        mc = load_contrasts()
        got = {}
        for c in a["cells"]:
            sh = {k["category"]: k["shift_toward_incorrect"] for k in c["candidates"]}
            for arr, cats in mc.ARRANGEMENTS.items():
                if all(x in sh for x in cats):
                    got[(c["model_id"], arr)] = mc.contrasts(sh, *cats)["B_ONLY-NEITHER"]
        r = lambda m, arr: round(got[(m, arr)], 2)
        slot2, slot1 = "A slot 1, B slot 2 (re-measured)", "B slot 1, A slot 2 (swapped)"
        self.assertEqual((r("llama3b", slot2), r("llama3b", slot1)), (1.88, 1.37))
        self.assertEqual((r("qwen1.5b", slot2), r("qwen1.5b", slot1)), (0.55, 0.35))
        self.assertEqual((r("qwen7b", slot2), r("qwen7b", slot1)), (0.21, -0.26))
        llama = [c for c in a["cells"] if c["model_id"] == "llama3b"][0]
        flips = {k["candidate_id"]: k["candidate_preferred"] for k in llama["candidates"] if k["outcome"] == "induced_incorrect"}
        self.assertEqual(set(flips), {"ORIG_B_ONLY", "B_ONLY_S", "BOTH_S"})
        swapped = [k for k in llama["candidates"] if k["candidate_id"] == "B_ONLY_S"][0]
        self.assertEqual((round(swapped["probabilities"]["A"], 3), round(swapped["probabilities"]["B"], 3)), (0.293, 0.386))
        self.assertFalse(any(k["outcome"] == "induced_incorrect" for c in a["cells"] if c["model_id"] != "llama3b" for k in c["candidates"]))

    def test_contrasts_helper_runs_on_a_report(self):
        _c, _t, _a, out = atlas("slot-swap")
        mc = load_contrasts()
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(mc.main(os.path.join(out, "atlas.json")), 0)
        self.assertIn("induced-incorrect: B_ONLY_S,BOTH_S", buf.getvalue())

    def test_documented_numbers_match_the_examples(self):
        readme = " ".join(read(os.path.join(ROOT, "README.md")).split())
        for needle in ("A 0.491 / B 0.095", "A 0.222 / B 0.486", "A 0.293 / B 0.386", "+1.88", "+1.37", "+0.55", "+0.35", "+0.21", "-0.26", "11 of 12", "8 of 12", "10 of 12"):
            self.assertIn(needle, readme, needle)


if __name__ == "__main__":
    unittest.main()
