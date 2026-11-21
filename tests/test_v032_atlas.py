"""``bleediff atlas``: aggregation over saved batch sessions. Sessions are produced by the real batch mode against a mock server
(SIMULATED probabilities chosen by marker words); the atlas itself reads only the saved files and never contacts a model."""

from __future__ import annotations

import atexit
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from tests.test_bleediff import run_cli  # noqa: E402
from tests.test_v030_track import ScriptedState  # noqa: E402
from tests.test_v031_batch import run_batch  # noqa: E402

from bleediff import atlas as AT  # noqa: E402
from bleediff.track import margin_ab  # noqa: E402

CANDS = [
    {"label": "claim-1", "text": "ALPHA explicit competing claim fragment for the mock."},
    {"label": "topic-1", "text": "BRAVO on-topic text that also mentions the option subject."},
    {"label": "off-1", "text": "CHARLIE off topic reporting fragment."},
    {"label": "neut-1", "text": "DELTA ordinary unrelated prose about gardens and bread."},
    {"label": "neut-2", "text": "DELTATWO a second ordinary unrelated prose fragment about rivers."},
    {"label": "id-1", "text": "ECHO matching identifier fragment TASK-7."},
]
CATMAP = {"claim-1": "claim", "topic-1": "topic", "off-1": "offtopic", "neut-1": "neutral", "neut-2": "neutral", "id-1": "id"}
GOOD = [("ALPHA", [0.20, 0.70, 0.05, 0.05]), ("BRAVO", [0.45, 0.40, 0.075, 0.075]), ("CHARLIE", [0.60, 0.30, 0.05, 0.05]),
        ("DELTATWO", [0.75, 0.15, 0.05, 0.05]), ("DELTA", [0.75, 0.15, 0.05, 0.05]), ("ECHO", [0.40, 0.45, 0.075, 0.075])]
GOOD_BASE = [0.70, 0.20, 0.05, 0.05]
# a task whose clean baseline already prefers D (wrong): A/B crossing without an overall flip, and recovery to A
WRONG = [("ALPHA", [0.10, 0.20, 0.05, 0.65]), ("BRAVO", [0.50, 0.20, 0.05, 0.25]), ("CHARLIE", [0.20, 0.10, 0.05, 0.65]),
         ("DELTATWO", [0.18, 0.12, 0.05, 0.65]), ("DELTA", [0.18, 0.12, 0.05, 0.65]), ("ECHO", [0.22, 0.08, 0.05, 0.65])]
WRONG_BASE = [0.20, 0.10, 0.05, 0.65]
SAT = [("ALPHA", [0.9900, 0.0090, 0.0005, 0.0005]), ("BRAVO", [0.9970, 0.0020, 0.0005, 0.0005]), ("CHARLIE", [0.9985, 0.0005, 0.0005, 0.0005]),
       ("DELTATWO", [0.9993, 0.0003, 0.0002, 0.0002]), ("DELTA", [0.9993, 0.0003, 0.0002, 0.0002]), ("ECHO", [0.9975, 0.0015, 0.0005, 0.0005])]
SAT_BASE = [0.9995, 0.0003, 0.0001, 0.0001]


def tmpdir():
    d = tempfile.mkdtemp(prefix="bleediff-atlas-test-")
    atexit.register(shutil.rmtree, d, True)
    return d


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def jload(path):
    return json.loads(read(path))


def csv_rows(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class World:
    """A temp directory holding batch sessions (made by the real batch mode against mock servers) and an atlas input manifest."""

    def __init__(self):
        self.dir = tmpdir()
        self.cands_path = os.path.join(self.dir, "cands.json")
        with open(self.cands_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"candidates": CANDS}, fh)
        self.sessions = []

    def session(self, task, model, rules, base, *, presentation="raw", mass=0.99, state=None, cands=None, sid=None, replicate_of=None,
                correct="A", incorrect="B", rel=None, evidence=True):
        rel = rel or f"results/{model}/{task}" + ("-rep" if replicate_of else "")
        out = os.path.join(self.dir, rel)
        path = self.cands_path
        if cands is not None:
            path = os.path.join(self.dir, f"cands-{model}-{task}.json")
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump({"candidates": cands}, fh)
        st = state or ScriptedState(rules, default=base, choice_mass=mass)
        code, text, _ = run_batch(st, path, out=out)
        assert code == 0, text
        s = {"id": sid or f"{task}/{model}/{presentation}" + ("/rep" if replicate_of else ""), "session_dir": rel, "task": {"id": task, "name": task.title()},
             "scenario_id": "config-audit-v2", "model": {"id": model, "name": model.upper()}, "presentation": presentation, "correct_answer": correct,
             "incorrect_answer": incorrect, "candidates": dict(CATMAP)}
        if evidence:
            s["evidence"] = [{"role": "candidate_file", "path": os.path.relpath(path, self.dir).replace("\\", "/"), "sha256": sha(path)}]
        if replicate_of:
            s["replicate_of"] = replicate_of
        self.sessions.append(s)
        return s

    def manifest(self, name="atlas-input.json", sessions=None, **extra):
        p = os.path.join(self.dir, name)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"schema": AT.INPUT_SCHEMA, "sessions": self.sessions if sessions is None else sessions, **extra}, fh)
        return p

    def run(self, *args, manifest=None, out=None):
        out = out or os.path.join(self.dir, "report-" + hashlib.md5(repr(args).encode()).hexdigest()[:6])
        code, text = run_cli("atlas", "--manifest", manifest or self.manifest(), "--out", out, *args)
        return code, text, out


def make_world():
    w = World()
    w.session("task-one", "m1", GOOD, GOOD_BASE)
    w.session("task-one", "m2", SAT, SAT_BASE)
    w.session("task-two", "m1", WRONG, WRONG_BASE)
    w.session("task-two", "m2", GOOD, GOOD_BASE)
    return w


class TestAtlas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = make_world()
        cls.w.session("task-one", "m1", GOOD, GOOD_BASE, replicate_of="task-one/m1/raw")
        cls.code, cls.text, cls.out = cls.w.run()
        cls.atlas = jload(os.path.join(cls.out, "atlas.json"))
        cls.cells = {(c["task_id"], c["model_id"]): c for c in cls.atlas["cells"]}

    def cand(self, task, model, cid):
        return [k for k in self.cells[(task, model)]["candidates"] if k["candidate_id"] == cid][0]

    def test_runs_and_writes_all_outputs(self):
        self.assertEqual(self.code, 0, self.text)
        for name in ("atlas.json", "candidates.csv", "category-summary.csv", "model-comparison.csv", "report.md"):
            self.assertTrue(os.path.isfile(os.path.join(self.out, name)), name)
        self.assertEqual(self.atlas["schema"], AT.ATLAS_SCHEMA)
        self.assertEqual(len(self.atlas["cells"]), 4)
        self.assertTrue(all(s["ok"] for s in self.atlas["sources"]))

    def test_no_absolute_local_paths_in_any_output(self):
        shown = "\n".join(x for x in self.text.splitlines() if not x.startswith("Wrote "))      # that line echoes the user's own --out argument
        blob = shown + "".join(read(os.path.join(self.out, n)) for n in os.listdir(self.out))
        self.assertNotIn(self.w.dir, blob)
        self.assertNotIn(ROOT, blob)

    def test_measurements_are_preserved_with_the_unchanged_margin_function(self):
        k = self.cand("task-one", "m1", "claim-1")
        for c, p in zip("ABCD", [0.20, 0.70, 0.05, 0.05]):
            self.assertAlmostEqual(k["probabilities"][c], p, places=9)
        self.assertAlmostEqual(k["margin_AB"], margin_ab([0.20, 0.70, 0.05, 0.05]), places=9)
        cell = self.cells[("task-one", "m1")]
        self.assertAlmostEqual(cell["baseline"]["margin_AB"], margin_ab(GOOD_BASE), places=9)
        self.assertAlmostEqual(k["shift_toward_incorrect"], margin_ab(GOOD_BASE) - margin_ab([0.20, 0.70, 0.05, 0.05]), places=9)
        self.assertEqual((k["candidate_preferred"], cell["baseline"]["preferred"]), ("B", "A"))
        self.assertIsNotNone(k["coverage"])
        self.assertGreater(k["prompt_tokens"], 0)
        self.assertEqual(len(k["prompt_sha256"]), 64)
        self.assertEqual(cell["correct_answer"], "A")

    def test_paired_differences_are_exact_shift_differences(self):
        cell = self.cells[("task-one", "m1")]
        sh = {c: margin_ab(GOOD_BASE) - margin_ab(v) for c, v in (("claim", [0.20, 0.70, 0.05, 0.05]), ("topic", [0.45, 0.40, 0.075, 0.075]),
                                                                    ("offtopic", [0.60, 0.30, 0.05, 0.05]), ("neutral", [0.75, 0.15, 0.05, 0.05]), ("id", [0.40, 0.45, 0.075, 0.075]))}
        cmp_ = cell["comparisons"]
        self.assertAlmostEqual(cmp_["topic_minus_offtopic"]["value"], sh["topic"] - sh["offtopic"], places=9)
        self.assertAlmostEqual(cmp_["topic_minus_neutral"]["value"], sh["topic"] - sh["neutral"], places=9)
        self.assertAlmostEqual(cmp_["claim_minus_topic"]["value"], sh["claim"] - sh["topic"], places=9)
        self.assertAlmostEqual(cmp_["id_minus_topic"]["value"], sh["id"] - sh["topic"], places=9)
        self.assertEqual(cmp_["topic_minus_neutral"]["n_b"], 2)                       # two neutral candidates are averaged, and n is recorded

    def test_outcomes_and_baseline_correctness(self):
        k = self.cand("task-one", "m1", "claim-1")
        self.assertEqual((k["outcome"], k["induced_to"], k["overall_flip"]), ("induced_incorrect", "designated_incorrect", True))
        self.assertEqual(self.cand("task-one", "m1", "topic-1")["outcome"], "stayed_correct")
        wrong = self.cells[("task-two", "m1")]
        self.assertEqual((wrong["baseline"]["preferred"], wrong["baseline"]["correct"]), ("D", False))
        self.assertIn("BASELINE_INCORRECT", wrong["flags"])
        self.assertEqual(self.cand("task-two", "m1", "topic-1")["outcome"], "recovered_to_correct")          # never "newly misled"
        self.assertEqual(self.cand("task-two", "m1", "claim-1")["outcome"], "baseline_incorrect_remained_incorrect")
        self.assertFalse(any(k["outcome"] == "induced_incorrect" for k in wrong["candidates"]))
        self.assertIn("NOT counted as induced: task-two / m1 baseline already prefers D", self.text)

    def test_an_ab_crossing_is_not_an_overall_flip(self):
        k = self.cand("task-two", "m1", "claim-1")             # A>B at baseline, B>A for the candidate, D stays the answer
        self.assertTrue(k["ab_crossed"])
        self.assertFalse(k["overall_flip"])
        self.assertIn("A/B crossings that are NOT overall answer flips: 1", self.text)

    def test_saturation_token_and_coverage_flags(self):
        self.assertIn("SATURATED_BASELINE", self.cells[("task-one", "m2")]["flags"])
        self.assertNotIn("SATURATED_BASELINE", self.cells[("task-one", "m1")]["flags"])
        self.assertIn("TOKEN_MISMATCH", self.cells[("task-one", "m1")]["flags"])               # the synthetic fragments differ in length
        self.assertTrue(any("TOKEN_MISMATCH" in k["flags"] for k in self.cells[("task-one", "m1")]["candidates"]))
        cmp_ = self.cells[("task-one", "m1")]["comparisons"]["topic_minus_offtopic"]
        self.assertIn("LENGTH_UNMATCHED", cmp_["flags"])

    def test_low_coverage_is_flagged(self):
        w = World()
        w.session("task-one", "m1", GOOD, GOOD_BASE, mass=0.5)
        code, _t, out = w.run()
        self.assertEqual(code, 0)
        a = jload(os.path.join(out, "atlas.json"))
        k = a["cells"][0]["candidates"][0]
        self.assertLess(k["coverage"], AT.LOW_COVERAGE)
        self.assertIn("LOW_COVERAGE", k["flags"])
        self.assertIn("LOW_COVERAGE", a["cells"][0]["flags"])

    def test_repeat_sessions_are_checked_not_counted(self):
        self.assertEqual(len(self.atlas["cells"]), 4)
        self.assertEqual([r["bit_identical"] for r in self.atlas["replicate_checks"]], [True])
        self.assertIn("1/1 repeat sessions bit-identical", self.text)

    def test_a_different_repeat_is_reported(self):
        w = make_world()
        w.session("task-one", "m1", SAT, SAT_BASE, replicate_of="task-one/m1/raw")
        code, text, out = w.run()
        self.assertEqual(code, 0)
        self.assertFalse(jload(os.path.join(out, "atlas.json"))["replicate_checks"][0]["bit_identical"])

    def test_candidates_csv_is_complete_and_ranked_within_each_cell(self):
        rows = csv_rows(os.path.join(self.out, "candidates.csv"))
        self.assertEqual(len(rows), 24)
        self.assertEqual(list(rows[0].keys()), AT.CANDIDATE_COLUMNS)
        cell = [r for r in rows if (r["task_id"], r["model_id"]) == ("task-one", "m1")]
        self.assertEqual(sorted(int(r["rank_in_cell"]) for r in cell), [1, 2, 3, 4, 5, 5])      # neutral twins tie
        self.assertEqual(cell[0]["candidate_id"], "claim-1")
        self.assertEqual(cell[0]["category"], "claim")
        for r in rows:
            self.assertAlmostEqual(sum(float(r[f"p_{c}"]) for c in "ABCD"), 1.0, places=6)
        self.assertEqual({r["task_id"] for r in rows if r["rank_in_cell"] == "1"}, {"task-one", "task-two"})
        self.assertFalse(any("global" in c for c in rows[0]))

    def test_category_summary_counts(self):
        rows = csv_rows(os.path.join(self.out, "category-summary.csv"))
        pooled = [r for r in rows if r["scope"] == "all_models_counts_only" and r["comparison"] == "topic_minus_neutral"][0]
        vals = [c["comparisons"]["topic_minus_neutral"]["value"] for c in self.atlas["cells"]]
        self.assertEqual(int(pooled["n"]), 4)
        self.assertEqual(int(pooled["n_positive"]), sum(v > 0 for v in vals))
        self.assertEqual(pooled["mean"], "")                                                      # magnitudes are never pooled across models
        cell_rows = [r for r in rows if r["scope"] == "cell"]
        self.assertEqual(len(cell_rows), 16)

    def test_model_comparison_matches_models_on_task_and_prompt(self):
        rows = csv_rows(os.path.join(self.out, "model-comparison.csv"))
        r = [x for x in rows if x["task_id"] == "task-one" and x["metric"] == "topic_minus_neutral"][0]
        self.assertEqual(r["compatible"], "True")
        self.assertEqual(r["n_models"], "2")
        self.assertIn(r["same_sign_all_models"], ("True", "False"))
        self.assertIn("value__m1", r)
        self.assertIn("rank_in_cell__m2", r)
        self.assertIn("compare signs", r["note"])

    def test_model_comparison_refuses_different_candidate_texts(self):
        w = World()
        w.session("task-one", "m1", GOOD, GOOD_BASE)
        other = [dict(c) for c in CANDS]
        other[1]["text"] = "BRAVO a different on-topic text for the second model."
        w.session("task-one", "m2", GOOD, GOOD_BASE, cands=other)
        code, _t, out = w.run()
        self.assertEqual(code, 0)
        rows = csv_rows(os.path.join(out, "model-comparison.csv"))
        r = [x for x in rows if x["metric"] == "topic_minus_neutral"][0]
        self.assertEqual((r["compatible"], r["same_sign_all_models"]), ("False", ""))
        self.assertIn("candidate texts differ", r["incompatibility"])

    def test_presentations_are_never_combined(self):
        w = World()
        w.session("task-one", "m1", GOOD, GOOD_BASE)
        w.session("task-one", "m1", GOOD, GOOD_BASE, presentation="native", rel="results/m1/task-one-native")
        code, text, out = w.run()
        self.assertEqual(code, 0)
        a = jload(os.path.join(out, "atlas.json"))
        self.assertTrue(any("several prompt presentations" in x for x in a["warnings"]))
        rows = csv_rows(os.path.join(out, "model-comparison.csv"))
        self.assertEqual({(r["presentation"], r["n_models"]) for r in rows if r["metric"] == "topic_minus_neutral"}, {("raw", "1"), ("native", "1")})
        pooled = [r for r in csv_rows(os.path.join(out, "category-summary.csv")) if r["scope"] == "all_models_counts_only" and r["comparison"] == "topic_minus_neutral"]
        self.assertEqual({r["presentation"] for r in pooled}, {"raw", "native"})

    def test_the_topic_limitation_is_visible_everywhere(self):
        needle = "option B's subject"
        self.assertIn(needle, self.text)
        self.assertIn(needle, read(os.path.join(self.out, "report.md")))
        self.assertIn(needle, " ".join(self.atlas["limitations"]))
        self.assertIn(needle, self.atlas["category_labels"]["topic"])
        self.assertTrue(all(needle in r["note"] for r in csv_rows(os.path.join(self.out, "category-summary.csv")) if r["comparison"].startswith("topic")))

    def test_no_global_ranking_and_no_causal_claims(self):
        report = read(os.path.join(self.out, "report.md")).lower()
        self.assertNotIn("most dangerous", report)
        self.assertIn("no global ranking", report)
        self.assertIn("do not establish a mechanism", report)
        self.assertFalse(any(k in self.atlas for k in ("global_ranking", "most_dangerous")))

    def test_sources_are_not_modified(self):
        before = {os.path.join(r, f): sha(os.path.join(r, f)) for r, _d, fs in os.walk(os.path.join(self.w.dir, "results")) for f in fs}
        self.w.run(out=os.path.join(self.w.dir, "report-again"))
        after = {p: sha(p) for p in before}
        self.assertEqual(before, after)

    def test_it_reads_saved_results_and_never_contacts_a_model(self):
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network access attempted")):
            code, text, _o = self.w.run(out=os.path.join(self.w.dir, "report-offline"))
        self.assertEqual(code, 0, text)


class TestFilters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = make_world()

    def test_filter_by_model_task_category_and_outcome(self):
        code, text, out = self.w.run("--model", "m1", "--category", "claim", "--outcome", "induced_incorrect")
        self.assertEqual(code, 0, text)
        rows = csv_rows(os.path.join(out, "candidates.csv"))
        self.assertEqual([(r["task_id"], r["model_id"], r["category"], r["outcome"]) for r in rows], [("task-one", "m1", "claim", "induced_incorrect")])
        a = jload(os.path.join(out, "atlas.json"))
        self.assertEqual(len(a["cells"]), 4)                                                      # atlas.json always keeps the complete data
        self.assertEqual(a["filters_applied_to_csv_report_and_terminal"]["model"], ["m1"])
        self.assertIn("Filters: model=m1", text)
        cols = csv_rows(os.path.join(out, "model-comparison.csv"))[0].keys()
        self.assertIn("value__m1", cols)
        self.assertNotIn("value__m2", cols)

    def test_task_filter(self):
        code, _t, out = self.w.run("--task", "task-two")
        self.assertEqual({r["task_id"] for r in csv_rows(os.path.join(out, "candidates.csv"))}, {"task-two"})

    def test_baseline_incorrect_outcomes_can_be_listed(self):
        _c, _t, out = self.w.run("--outcome", "baseline_incorrect_remained_incorrect")
        rows = csv_rows(os.path.join(out, "candidates.csv"))
        self.assertTrue(rows)
        self.assertTrue(all(r["baseline_correct"] == "False" for r in rows))

    def test_bad_filters_are_refused(self):
        for args in (("--model", "nope"), ("--task", "nope"), ("--outcome", "nope")):
            code, text, out = self.w.run(*args)
            self.assertEqual(code, 2, args)
            self.assertFalse(os.path.exists(out))


class TestValidation(unittest.TestCase):
    def fresh(self):
        w = World()
        w.session("task-one", "m1", GOOD, GOOD_BASE)
        w.session("task-two", "m1", GOOD, GOOD_BASE)
        return w

    def test_tampered_history_is_refused(self):
        w = self.fresh()
        with open(os.path.join(w.dir, "results/m1/task-one/history.jsonl"), "ab") as fh:
            fh.write(b"\n")
        code, text, out = w.run()
        self.assertEqual(code, 2)
        self.assertIn("ATLAS_INPUT_INVALID", text)
        self.assertIn("does not match the hash recorded", text)
        self.assertFalse(os.path.exists(out))

    def test_keep_going_excludes_invalid_sources_and_reports_them(self):
        w = self.fresh()
        with open(os.path.join(w.dir, "results/m1/task-one/history.jsonl"), "ab") as fh:
            fh.write(b"\n")
        code, text, out = w.run("--keep-going")
        self.assertEqual(code, 0, text)
        a = jload(os.path.join(out, "atlas.json"))
        self.assertEqual([c["task_id"] for c in a["cells"]], ["task-two"])
        self.assertEqual([s["ok"] for s in a["sources"]], [False, True])
        self.assertIn("EXCLUDED", read(os.path.join(out, "report.md")))

    def test_missing_session_and_missing_evidence(self):
        w = self.fresh()
        w.sessions[0]["session_dir"] = "results/does-not-exist"
        code, text, _o = w.run()
        self.assertEqual(code, 2)
        self.assertIn("missing", text)
        w = self.fresh()
        w.sessions[0]["evidence"][0]["path"] = "nope.json"
        code, text, _o = w.run()
        self.assertEqual(code, 2)
        self.assertIn("evidence file", text)

    def test_evidence_hash_and_role_mismatches(self):
        w = self.fresh()
        w.sessions[0]["evidence"][0]["sha256"] = "0" * 64
        code, text, _o = w.run()
        self.assertEqual(code, 2)
        self.assertIn("does not match its recorded sha256", text)
        w = self.fresh()
        other = os.path.join(w.dir, "other.json")
        with open(other, "w", encoding="utf-8") as fh:
            fh.write("{}")
        w.sessions[0]["evidence"] = [{"role": "candidate_file", "path": "other.json"}]
        code, text, _o = w.run()
        self.assertEqual(code, 2)
        self.assertIn("is not the candidate file the session was run with", text)

    def test_identity_mismatches(self):
        w = self.fresh()
        w.sessions[0]["incorrect_answer"] = "C"
        code, text, _o = w.run()
        self.assertEqual((code, "designated" in text), (2, True))
        w = self.fresh()
        w.sessions[0]["scenario_id"] = "other-scenario"
        code, text, _o = w.run()
        self.assertEqual((code, "scenario id" in text), (2, True))
        w = self.fresh()
        w.sessions[0]["model"]["model_hash"] = "f" * 64
        code, text, _o = w.run()
        self.assertEqual((code, "model hash" in text), (2, True))

    def test_malformed_manifests(self):
        w = self.fresh()
        bads = {"no sessions": {"schema": AT.INPUT_SCHEMA, "sessions": []}, "wrong schema": {"schema": "x", "sessions": w.sessions},
                "same letter": {"schema": AT.INPUT_SCHEMA, "sessions": [{**w.sessions[0], "incorrect_answer": "A"}]},
                "missing key": {"schema": AT.INPUT_SCHEMA, "sessions": [{k: v for k, v in w.sessions[0].items() if k != "presentation"}]},
                "duplicate cell": {"schema": AT.INPUT_SCHEMA, "sessions": [w.sessions[0], {**w.sessions[0], "id": "other-id"}]},
                "duplicate id": {"schema": AT.INPUT_SCHEMA, "sessions": [w.sessions[0], {**w.sessions[1], "id": w.sessions[0]["id"]}]},
                "bad replicate_of": {"schema": AT.INPUT_SCHEMA, "sessions": [{**w.sessions[0], "replicate_of": "missing"}]}}
        for name, doc in bads.items():
            p = os.path.join(w.dir, "bad.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            code, text, out = w.run(manifest=p, out=os.path.join(w.dir, "o-" + name.replace(" ", "-")))
            self.assertEqual(code, 2, name)
            self.assertFalse(os.path.exists(out), name)
        code, text, _o = w.run(manifest=os.path.join(w.dir, "missing.json"))
        self.assertEqual(code, 2)
        with open(os.path.join(w.dir, "broken.json"), "wb") as fh:
            fh.write(b"{nope")
        self.assertEqual(w.run(manifest=os.path.join(w.dir, "broken.json"))[0], 2)

    def test_existing_output_directory_is_never_overwritten(self):
        w = self.fresh()
        taken = os.path.join(w.dir, "taken")
        os.makedirs(taken)
        with open(os.path.join(taken, "keep.txt"), "w", encoding="utf-8") as fh:
            fh.write("mine")
        code, text, _o = w.run(out=taken)
        self.assertEqual((code, os.listdir(taken)), (2, ["keep.txt"]))
        self.assertIn("OUTPUT_EXISTS", text)

    def test_unknown_candidate_ids_and_uncategorized_candidates(self):
        w = self.fresh()
        w.sessions[0]["candidates"] = {**CATMAP, "ghost": "claim"}
        del w.sessions[0]["candidates"]["id-1"]
        code, text, out = w.run()
        self.assertEqual(code, 0, text)
        cell = [c for c in jload(os.path.join(out, "atlas.json"))["cells"] if c["task_id"] == "task-one"][0]
        self.assertTrue(any("ghost" in p for p in cell["problems"]))
        ident = [k for k in cell["candidates"] if k["candidate_id"] == "id-1"][0]
        self.assertEqual(ident["category"], "uncategorized")
        self.assertIn("UNCATEGORIZED", ident["flags"])
        self.assertIsNone(cell["comparisons"]["id_minus_topic"]["value"])                          # no comparison without the category

    def test_a_failed_candidate_is_unscored_and_excluded_from_comparisons(self):
        class Flaky(ScriptedState):
            def completion(self, payload):
                if "BRAVO" in payload["prompt"]:
                    self.requests.append(payload)
                    return {"content": "A"}
                return super().completion(payload)

        w = World()
        w.session("task-one", "m1", GOOD, GOOD_BASE, state=Flaky(GOOD, default=GOOD_BASE, choice_mass=0.99))
        code, text, out = w.run()
        self.assertEqual(code, 0, text)
        a = jload(os.path.join(out, "atlas.json"))
        topic = [k for k in a["cells"][0]["candidates"] if k["category"] == "topic"][0]
        self.assertEqual((topic["status"], topic["outcome"]), ("error", "unscored"))
        self.assertNotIn("probabilities", topic)
        self.assertIsNone(a["cells"][0]["comparisons"]["topic_minus_offtopic"]["value"])
        self.assertIn("missing or unscored", a["cells"][0]["comparisons"]["topic_minus_offtopic"]["reason"])
        rows = csv_rows(os.path.join(out, "candidates.csv"))
        self.assertIn("unscored", [r["outcome"] for r in rows])

    def test_saved_values_that_disagree_are_reported(self):
        w = self.fresh()
        p = os.path.join(w.dir, "results/m1/task-one/history.jsonl")
        lines = read(p).splitlines()
        rec = [json.loads(x) for x in lines]
        for r in rec:
            if r.get("type") == "measurement" and (r.get("candidate") or {}).get("label") == "claim-1":
                r["margin_AB"] += 0.5
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(json.dumps(r) for r in rec) + "\n")
        w.sessions[0].pop("evidence")
        man = os.path.join(w.dir, "results/m1/task-one/batch-manifest.json")
        doc = jload(man)
        doc["files"]["history.jsonl"] = sha(p)                                                     # a consistent re-record, as a careless tool might
        with open(man, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh)
        code, text, out = w.run()
        self.assertEqual(code, 0, text)
        k = [x for x in jload(os.path.join(out, "atlas.json"))["cells"][0]["candidates"] if x["candidate_id"] == "claim-1"][0]
        self.assertIn("SAVED_VALUE_MISMATCH", k["flags"])
        self.assertAlmostEqual(k["margin_AB"], margin_ab([0.20, 0.70, 0.05, 0.05]), places=9)       # the margin is re-derived, not trusted


class TestPureFunctions(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(AT.classify("A", "A", "A", "B"), ("stayed_correct", None))
        self.assertEqual(AT.classify("A", "B", "A", "B"), ("induced_incorrect", "designated_incorrect"))
        self.assertEqual(AT.classify("A", "D", "A", "B"), ("induced_incorrect", "other_incorrect"))
        self.assertEqual(AT.classify("D", "A", "A", "B"), ("recovered_to_correct", None))
        self.assertEqual(AT.classify("D", "B", "A", "B"), ("baseline_incorrect_remained_incorrect", None))

    def test_stats(self):
        s = AT._stats([0.5, -0.1, 0.25, 1.0], 0.25)
        self.assertEqual((s["n"], s["n_positive"], s["n_negative"], s["n_at_least_marker"]), (4, 3, 1, 3))
        self.assertTrue(math.isclose(s["mean"], 0.4125))
        self.assertEqual(AT._stats([], 0.25)["n"], 0)

    def test_effect_marker_option(self):
        w = make_world()
        code, text, out = w.run("--effect-marker", "5")
        self.assertEqual(code, 0)
        pooled = [r for r in csv_rows(os.path.join(out, "category-summary.csv")) if r["scope"] == "all_models_counts_only" and r["comparison"] == "topic_minus_neutral"][0]
        self.assertEqual(pooled["n_at_least_marker"], str(sum(1 for c in jload(os.path.join(out, "atlas.json"))["cells"] if (c["comparisons"]["topic_minus_neutral"]["value"] or 0) >= 5)))
        self.assertEqual(w.run("--effect-marker", "-1")[0], 2)


if __name__ == "__main__":
    unittest.main()
