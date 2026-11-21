"""Semantic answer roles: scoring the correct and the competing answers wherever their letters are (A-D permutations).
Probabilities are SIMULATED by a mock server (marker words in the prompt); nothing here is research evidence."""

from __future__ import annotations

import atexit
import csv
import itertools
import json
import math
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from tests.test_bleediff import run_cli  # noqa: E402
from tests.test_v030_track import FIXTURES, ScriptedState, serve_state  # noqa: E402
from tests.test_v031_batch import SYNTH, hist, run_batch  # noqa: E402

from bleediff import atlas as AT  # noqa: E402
from bleediff import batch as BT  # noqa: E402
from bleediff import roles as RL  # noqa: E402
from bleediff.track import MeasurementError, margin_ab  # noqa: E402

LET = "ABCD"
SEM_BASE = {"K": 0.60, "T": 0.05, "X": 0.25, "Y": 0.10}                          # semantic probabilities of the clean prompt
SEM_VARS = {"ALPHA": {"K": 0.20, "T": 0.55, "X": 0.15, "Y": 0.10},               # competing evidence: wins for the competing answer
            "BRAVO": {"K": 0.45, "T": 0.15, "X": 0.30, "Y": 0.10},
            "CHARLIE": {"K": 0.58, "T": 0.04, "X": 0.28, "Y": 0.10},
            "DELTA": {"K": 0.30, "T": 0.20, "X": 0.40, "Y": 0.10},               # a wrong answer other than the competing one wins
            "ECHO": {"K": 0.52, "T": 0.08, "X": 0.30, "Y": 0.10}}
MAPPINGS = {"m1": {"A": "K", "B": "T", "C": "X", "D": "Y"}, "m2": {"A": "T", "B": "K", "C": "Y", "D": "X"},
            "m3": {"A": "X", "B": "Y", "C": "K", "D": "T"}, "m4": {"A": "Y", "B": "X", "C": "T", "D": "K"}}


def letter_vector(sem: dict, mapping: dict) -> list:
    return [sem[mapping[c]] for c in LET]


def letters_of(mapping: dict) -> tuple:
    inv = {r: l for l, r in mapping.items()}
    return inv["K"], inv["T"]


def tmpdir():
    d = tempfile.mkdtemp(prefix="bleediff-roles-test-")
    atexit.register(shutil.rmtree, d, True)
    return d


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def jload(path):
    return json.loads(read(path))


def sem_state(mapping: dict, base=None, variants=None, **kw):
    rules = [(marker, letter_vector(v, mapping)) for marker, v in (variants or SEM_VARS).items()]
    return ScriptedState(rules, default=letter_vector(base or SEM_BASE, mapping), choice_mass=0.99, **kw)


def write_cands(d, name="cands.json"):
    p = os.path.join(d, name)
    cands = [{"label": f"{k.lower()}-1", "text": f"{k} synthetic fragment number one."} for k in SEM_VARS]
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"candidates": cands}, fh)
    return p


class TestRoleScoring(unittest.TestCase):
    def test_every_ordered_letter_pair_scores_the_stated_roles(self):
        probs = [0.40, 0.30, 0.20, 0.10]
        for correct, competing in itertools.permutations(LET, 2):
            s = RL.role_scores(probs, correct, competing)
            p = dict(zip(LET, probs))
            self.assertEqual((s["correct_letter"], s["competing_letter"]), (correct, competing))
            self.assertEqual(s["margin_correct_vs_competing"], math.log(p[correct]) - math.log(p[competing]))
            others = [c for c in LET if c != correct]
            strongest = max(others, key=lambda c: (p[c], -LET.index(c)))
            self.assertEqual(s["strongest_incorrect_letter"], strongest)
            self.assertEqual(s["margin_correct_vs_strongest"], math.log(p[correct]) - math.log(p[strongest]))
            self.assertEqual(s["roles_are_ab"], {correct, competing} == {"A", "B"})
            self.assertEqual(s["preferred_letter"], "A")
            self.assertEqual(s["preferred_is_correct"], correct == "A")

    def test_scores_are_invariant_under_all_24_letter_permutations(self):
        ref = None
        for perm in itertools.permutations(LET):                                   # perm[i] = the letter holding role "KTXY"[i]
            mapping = {perm[i]: "KTXY"[i] for i in range(4)}
            correct, competing = letters_of(mapping)
            s = RL.role_scores(letter_vector(SEM_BASE, mapping), correct, competing)
            key = (s["margin_correct_vs_competing"], s["margin_correct_vs_strongest"], s["p_correct"], s["p_competing"], s["p_strongest_incorrect"], s["preferred_is_correct"])
            ref = ref or key
            self.assertEqual(key, ref, mapping)

    def test_strongest_incorrect_can_be_the_competing_answer_or_another_and_ties_take_the_earliest_letter(self):
        s = RL.role_scores([0.1, 0.5, 0.3, 0.1], "C", "D")                          # B is the strongest incorrect, not the competing answer
        self.assertEqual((s["strongest_incorrect_letter"], s["strongest_incorrect_is_competing"]), ("B", False))
        self.assertLess(s["margin_correct_vs_strongest"], s["margin_correct_vs_competing"] + 10)
        s = RL.role_scores([0.1, 0.1, 0.5, 0.3], "C", "D")
        self.assertEqual((s["strongest_incorrect_letter"], s["strongest_incorrect_is_competing"]), ("D", True))
        s = RL.role_scores([0.25, 0.25, 0.4, 0.1], "C", "D")                        # tie A/B -> earliest letter
        self.assertEqual(s["strongest_incorrect_letter"], "A")

    def test_changes_flip_crossing_and_induced_errors_are_separate(self):
        base = RL.role_scores([0.05, 0.10, 0.70, 0.15], "C", "D")
        cand = RL.role_scores([0.05, 0.10, 0.25, 0.60], "C", "D")
        ch = RL.role_changes(base, cand)
        self.assertTrue(ch["newly_induced_incorrect"] and ch["overall_flip"] and ch["role_order_crossed"])
        self.assertEqual(ch["induced_to"], "competing")
        self.assertAlmostEqual(ch["movement_toward_competing"], -(cand["margin_correct_vs_competing"] - base["margin_correct_vs_competing"]), places=12)
        self.assertAlmostEqual(ch["change_margin_correct_vs_strongest"], cand["margin_correct_vs_strongest"] - base["margin_correct_vs_strongest"], places=12)
        other = RL.role_changes(base, RL.role_scores([0.50, 0.10, 0.30, 0.10], "C", "D"))
        self.assertEqual((other["newly_induced_incorrect"], other["induced_to"]), (True, "other_incorrect"))
        quiet = RL.role_changes(base, RL.role_scores([0.05, 0.10, 0.60, 0.25], "C", "D"))
        self.assertFalse(quiet["newly_induced_incorrect"] or quiet["overall_flip"] or quiet["role_order_crossed"])

    def test_a_role_crossing_is_not_an_overall_flip(self):
        base = RL.role_scores([0.20, 0.10, 0.05, 0.65], "A", "B")                  # D (wrong) leads; A above B
        cand = RL.role_scores([0.10, 0.20, 0.05, 0.65], "A", "B")                  # B above A, D still leads
        ch = RL.role_changes(base, cand)
        self.assertTrue(ch["role_order_crossed"])
        self.assertFalse(ch["overall_flip"])

    def test_no_induced_error_when_the_baseline_was_already_wrong(self):
        base = RL.role_scores([0.60, 0.10, 0.20, 0.10], "C", "D")                  # prefers A: wrong from the start
        for vec in ([0.10, 0.10, 0.20, 0.60], [0.10, 0.10, 0.70, 0.10], [0.10, 0.60, 0.20, 0.10]):
            ch = RL.role_changes(base, RL.role_scores(vec, "C", "D"))
            self.assertFalse(ch["baseline_correct"])
            self.assertFalse(ch["newly_induced_incorrect"], vec)
            self.assertIsNone(ch["induced_to"])

    def test_equals_the_tracker_ab_margin_when_the_roles_are_a_and_b(self):
        for b in ([0.7, 0.2, 0.05, 0.05], [0.3, 0.6, 0.05, 0.05]):
            for c in ([0.2, 0.7, 0.05, 0.05], [0.55, 0.3, 0.1, 0.05]):
                d_ab = margin_ab(c) - margin_ab(b)
                self.assertEqual(RL.role_changes(RL.role_scores(b, "A", "B"), RL.role_scores(c, "A", "B"))["movement_toward_competing"], -d_ab)
                self.assertEqual(RL.role_changes(RL.role_scores(b, "B", "A"), RL.role_scores(c, "B", "A"))["movement_toward_competing"], d_ab)

    def test_invalid_input_is_refused_never_imputed(self):
        for args in (([0.4, 0.3, 0.2, 0.1], "A", "A"), ([0.4, 0.3, 0.2, 0.1], "E", "A"), ([0.4, 0.3, 0.2, 0.1], "A", ""),
                     ([0.4, 0.3, 0.2], "A", "B"), ([0.5, 0.3, 0.3, 0.1], "A", "B"), ([0.0, 0.5, 0.3, 0.2], "A", "B"), ([0.5, 0.0, 0.3, 0.2], "A", "B"),
                     ([float("nan"), 0.5, 0.3, 0.2], "A", "B")):
            with self.assertRaises(MeasurementError, msg=args):
                RL.role_scores(*args)
        with self.assertRaises(MeasurementError):                                  # an undefined strongest-incorrect margin is refused too
            RL.role_scores([0.5, 0.5, 0.0, 0.0], "A", "C")
        with self.assertRaises(MeasurementError):
            RL.role_changes(RL.role_scores([0.4, 0.3, 0.2, 0.1], "A", "B"), RL.role_scores([0.4, 0.3, 0.2, 0.1], "B", "A"))


class TestBatchRoles(unittest.TestCase):
    def run_roles(self, mapping, extra, state=None, cands=None):
        d = tmpdir()
        code, text, out = run_batch(state or sem_state(mapping), cands or write_cands(d), extra=extra, out=os.path.join(d, "batch"))
        return code, text, out

    def test_roles_at_c_and_d_are_scored_semantically_not_as_ab(self):
        code, text, out = self.run_roles(MAPPINGS["m3"], ("--correct", "C", "--incorrect", "D"))
        self.assertEqual(code, 0, text)
        h = hist(out)
        sem = [r for r in h if r["type"] == "semantic_roles"]
        self.assertEqual([r["label"] for r in sem], ["baseline"] + ["candidate"] * 5)
        base = sem[0]["scores"]
        self.assertEqual((base["correct_letter"], base["competing_letter"], base["roles_are_ab"]), ("C", "D", False))
        self.assertAlmostEqual(base["margin_correct_vs_competing"], math.log(0.60 / 0.05), places=9)
        alpha = [r for r in sem if r.get("candidate", {}).get("label") == "alpha-1"][0]
        self.assertAlmostEqual(alpha["scores"]["margin_correct_vs_competing"], math.log(0.20 / 0.55), places=9)
        self.assertAlmostEqual(alpha["changes"]["movement_toward_competing"], math.log(0.60 / 0.05) - math.log(0.20 / 0.55), places=9)
        self.assertTrue(alpha["changes"]["newly_induced_incorrect"] and alpha["changes"]["overall_flip"])
        man = jload(os.path.join(out, "batch-manifest.json"))
        self.assertEqual((man["designated_correct_answer"], man["designated_incorrect_answer"], man["semantic_roles"]["roles_are_ab"]), ("C", "D", False))
        self.assertIn("not the A/B margin", man["ranking_definition"])
        self.assertEqual([r["label"] for r in man["ranking"]][0], "alpha-1")                # ranked by the role movement
        self.assertIn("Roles: correct = C, competing = D", text)
        self.assertIn("NOT the role margin", text)
        self.assertIn("role margin", text)

    def test_the_ab_margin_would_have_ranked_these_candidates_differently(self):
        """Candidates that change p_A/p_B hugely but not C/D must not outrank the one that moves the C/D roles."""
        variants = {"ALPHA": {"K": 0.20, "T": 0.55, "X": 0.15, "Y": 0.10}, "BRAVO": {"K": 0.58, "T": 0.04, "X": 0.28, "Y": 0.10}}
        mapping = MAPPINGS["m3"]                                                        # A=X, B=Y: ECHO below changes only the X/Y letters
        state = sem_state(mapping, variants={**variants, "ECHO": {"K": 0.60, "T": 0.05, "X": 0.02, "Y": 0.33}})
        d = tmpdir()
        p = os.path.join(d, "c.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump([{"label": "alpha", "text": "ALPHA one"}, {"label": "echo", "text": "ECHO one"}], fh)
        code, text, out = run_batch(state, p, extra=("--correct", "C", "--incorrect", "D"), out=os.path.join(d, "o"))
        self.assertEqual(code, 0, text)
        man = jload(os.path.join(out, "batch-manifest.json"))
        self.assertEqual([r["label"] for r in man["ranking"]], ["alpha", "echo"])
        echo = [r for r in man["ranking"] if r["label"] == "echo"][0]
        self.assertAlmostEqual(echo["movement_toward_incorrect_nats"], 0.0, places=9)       # C/D roles untouched
        self.assertGreater(abs(echo["margin_AB"] - man["baseline"]["margin_AB"]), 2.0)     # while the A/B margin moved a lot

    def test_c_or_d_roles_require_an_explicit_correct_letter(self):
        for extra in (("--incorrect", "D"), ("--incorrect", "C"), ("--correct", "D", "--incorrect", "D"), ("--correct", "A", "--incorrect", "A")):
            d = tmpdir()
            code, text, out = run_batch(sem_state(MAPPINGS["m1"]), write_cands(d), extra=extra, out=os.path.join(d, "x"))
            self.assertEqual(code, 2, extra)
            self.assertIn("BAD_ARGS", text)
            self.assertFalse(os.path.exists(out), extra)

    def test_v030_defaults_and_the_implied_correct_letter_are_unchanged(self):
        code, text, out = self.run_roles(MAPPINGS["m1"], ())
        self.assertEqual(code, 0, text)
        man = jload(os.path.join(out, "batch-manifest.json"))
        self.assertEqual((man["designated_correct_answer"], man["designated_incorrect_answer"], man["semantic_roles"]["roles_are_ab"]), ("A", "B", True))
        self.assertIn("baseline margin - candidate margin", man["ranking_definition"])
        self.assertNotIn("Roles: correct", text)
        with open(os.path.join(out, "leaderboard.csv"), encoding="utf-8", newline="") as fh:
            self.assertEqual(next(csv.reader(fh)), BT.CSV_COLUMNS)                           # the CSV format did not change
        code, text, out = self.run_roles(MAPPINGS["m2"], ("--incorrect", "A"))              # v0.3.0: --incorrect A implies correct B
        man = jload(os.path.join(out, "batch-manifest.json"))
        self.assertEqual((man["designated_correct_answer"], man["designated_incorrect_answer"]), ("B", "A"))
        top = man["ranking"][0]
        self.assertEqual(top["label"], "alpha-1")
        base = man["baseline"]["margin_AB"]
        self.assertAlmostEqual(top["movement_toward_incorrect_nats"], top["margin_AB"] - base, places=9)    # the old A/B formula for incorrect = A

    def test_an_undefined_role_margin_is_reported_not_imputed(self):
        mapping = MAPPINGS["m3"]
        state = sem_state(mapping, variants={"ALPHA": {"K": 0.5, "T": 0.0, "X": 0.3, "Y": 0.2}, "BRAVO": SEM_VARS["BRAVO"]})
        d = tmpdir()
        p = os.path.join(d, "c.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump([{"label": "zero-competing", "text": "ALPHA one"}, {"label": "fine", "text": "BRAVO one"}], fh)
        code, text, out = run_batch(state, p, extra=("--correct", "C", "--incorrect", "D"), out=os.path.join(d, "o"))
        self.assertEqual(code, 0, text)
        with open(os.path.join(out, "leaderboard.csv"), encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        bad = [r for r in rows if r["label"] == "zero-competing"][0]
        self.assertNotEqual(bad["status"], "ok")
        self.assertEqual(bad["margin_AB"], "")
        self.assertEqual([r["label"] for r in rows if r["status"] == "ok"], ["fine"])


class TestAtlasRoles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tmpdir()
        cls.cands = write_cands(cls.d)
        cls.sessions = []
        for mp, mapping in MAPPINGS.items():                                        # the same semantic probabilities under four letter mappings
            correct, competing = letters_of(mapping)
            for model in ("m1", "m2"):
                st = sem_state(mapping) if model == "m1" else sem_state(mapping, base={"K": 0.15, "T": 0.10, "X": 0.60, "Y": 0.15})     # model m2's baseline is wrong (prefers X)
                rel = f"results/{model}/{mp}"
                code, text, _o = run_batch(st, cls.cands, extra=("--correct", correct, "--incorrect", competing), out=os.path.join(cls.d, rel))
                assert code == 0, text
                cls.sessions.append({"id": f"{mp}/{model}/raw", "session_dir": rel, "task": {"id": mp, "name": mp}, "scenario_id": "config-audit-v2", "model": {"id": model, "name": model.upper()},
                                     "presentation": "raw", "correct_answer": correct, "incorrect_answer": competing,
                                     "candidates": {f"{k.lower()}-1": c for k, c in zip(SEM_VARS, ("claim", "topic", "offtopic", "neutral", "id"))},
                                     "evidence": [{"role": "candidate_file", "path": "cands.json"}]})
        cls.manifest = os.path.join(cls.d, "atlas-input.json")
        with open(cls.manifest, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"schema": AT.INPUT_SCHEMA, "sessions": cls.sessions}, fh)
        cls.out = os.path.join(cls.d, "report")
        cls.code, cls.text = run_cli("atlas", "--manifest", cls.manifest, "--out", cls.out)
        cls.atlas = jload(os.path.join(cls.out, "atlas.json"))
        cls.cells = {(c["task_id"], c["model_id"]): c for c in cls.atlas["cells"]}

    def test_runs_and_validates_c_and_d_roles(self):
        self.assertEqual(self.code, 0, self.text)
        self.assertTrue(all(s["ok"] for s in self.atlas["sources"]))
        self.assertEqual(len(self.atlas["cells"]), 8)

    def test_role_scores_are_identical_under_every_letter_mapping_while_the_ab_margin_is_not(self):
        base = {}
        for mp in MAPPINGS:
            c = self.cells[(mp, "m1")]
            k = [x for x in c["candidates"] if x["candidate_id"] == "alpha-1"][0]
            base[mp] = (round(c["baseline"]["margin_correct_vs_competing"], 9), round(k["shift_toward_incorrect"], 9), round(k["semantic"]["margin_correct_vs_competing"], 9),
                        k["semantic"]["newly_induced_incorrect"], k["outcome"])
        self.assertEqual(len(set(base.values())), 1, base)
        self.assertAlmostEqual(next(iter(base.values()))[1], math.log(0.60 / 0.05) - math.log(0.20 / 0.55), places=8)
        ab = {self.cells[(mp, "m1")]["baseline"]["margin_AB"] for mp in MAPPINGS}
        self.assertGreater(len({round(x, 6) for x in ab}), 1)                                 # the A/B margins differ: they are not role margins

    def test_flags_and_notes_prevent_reading_c_d_results_as_ab(self):
        for mp in ("m1", "m2"):
            self.assertNotIn("AB_MARGIN_NOT_ROLE_MARGIN", self.cells[(mp, "m1")]["flags"])      # roles at A/B: the A/B margin is (up to sign) the role margin
            self.assertEqual(self.cells[(mp, "m1")]["roles_are_ab"], True)
        for mp in ("m3", "m4"):
            c = self.cells[(mp, "m1")]
            self.assertIn("AB_MARGIN_NOT_ROLE_MARGIN", c["flags"])
            self.assertFalse(c["roles_are_ab"])
            self.assertTrue(all("AB_MARGIN_NOT_ROLE_MARGIN" in k["flags"] for k in c["candidates"]))
        self.assertIn("not at A and B", self.text)
        self.assertIn("role margins", self.text)

    def test_the_semantic_section_and_columns_do_not_assume_option_b(self):
        report = read(os.path.join(self.out, "report.md"))
        section = report[report.index("## Semantic answer roles"):report.index("## Task/model cells")]
        self.assertNotIn("option B", section)
        self.assertNotIn("B's subject", section)
        self.assertIn("competing answer's subject", section)
        self.assertIn("| m3 | m1 | C | D | False |", section)
        with open(os.path.join(self.out, "candidates.csv"), encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        r = [x for x in rows if x["task_id"] == "m3" and x["model_id"] == "m1" and x["candidate_id"] == "alpha-1"][0]
        self.assertEqual((r["correct_letter"], r["competing_letter"], r["strongest_incorrect_letter"]), ("C", "D", "D"))
        self.assertAlmostEqual(float(r["p_correct"]), 0.20, places=6)
        self.assertAlmostEqual(float(r["margin_correct_vs_competing"]), math.log(0.20 / 0.55), places=6)
        self.assertAlmostEqual(float(r["change_margin_correct_vs_strongest"]), math.log(0.20 / 0.55) - math.log(0.60 / 0.25), places=6)
        self.assertEqual(r["newly_induced_incorrect"], "True")
        self.assertEqual(list(rows[0].keys()), AT.CANDIDATE_COLUMNS)

    def test_induced_errors_require_a_correct_baseline(self):
        for mp in MAPPINGS:
            wrong = self.cells[(mp, "m2")]                                                  # this model's baseline prefers X (wrong)
            self.assertIn("BASELINE_INCORRECT", wrong["flags"])
            self.assertFalse(any(k["semantic"]["newly_induced_incorrect"] for k in wrong["candidates"]))
            self.assertFalse(any(k["outcome"] == "induced_incorrect" for k in wrong["candidates"]))
            right = self.cells[(mp, "m1")]
            self.assertEqual({k["candidate_id"] for k in right["candidates"] if k["semantic"]["newly_induced_incorrect"]}, {"alpha-1", "delta-1"})
            self.assertEqual({k["candidate_id"]: k["semantic"]["induced_to"] for k in right["candidates"] if k["semantic"]["newly_induced_incorrect"]},
                             {"alpha-1": "competing", "delta-1": "other_incorrect"})

    def test_the_strongest_incorrect_margin_is_reported_separately(self):
        c = self.cells[("m3", "m1")]
        k = [x for x in c["candidates"] if x["candidate_id"] == "delta-1"][0]                # X wins: the strongest incorrect is not the competing answer
        sm = k["semantic"]
        self.assertFalse(sm["strongest_incorrect_is_competing"])
        self.assertAlmostEqual(sm["margin_correct_vs_strongest"], math.log(0.30 / 0.40), places=9)
        self.assertNotAlmostEqual(sm["margin_correct_vs_strongest"], sm["margin_correct_vs_competing"], places=3)

    def test_a_recorded_correct_answer_that_disagrees_with_the_manifest_is_refused(self):
        bad = json.loads(json.dumps(self.sessions))
        bad[0]["correct_answer"] = "C" if bad[0]["incorrect_answer"] != "C" else "D"                # a valid but different letter
        p = os.path.join(self.d, "bad.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"schema": AT.INPUT_SCHEMA, "sessions": bad}, fh)
        code, text = run_cli("atlas", "--manifest", p, "--out", os.path.join(self.d, "bad-report"))
        self.assertEqual(code, 2)
        self.assertIn("as correct", text)

    def test_v030_sessions_without_semantic_records_give_the_old_numbers_exactly(self):
        """The published example sessions predate this feature; their shifts must equal the original A/B formula bit for bit."""
        ex = os.path.join(ROOT, "examples", "atlas", "pilot", "atlas-input.json")
        out = os.path.join(tmpdir(), "r")
        code, _t = run_cli("atlas", "--manifest", ex, "--out", out)
        self.assertEqual(code, 0)
        a = jload(os.path.join(out, "atlas.json"))
        for c in a["cells"]:
            self.assertTrue(c["roles_are_ab"])
            self.assertNotIn("AB_MARGIN_NOT_ROLE_MARGIN", c["flags"])
            for k in c["candidates"]:
                old = BT.movement_toward(k["margin_AB"] - c["baseline"]["margin_AB"], c["incorrect_answer"])
                self.assertEqual(k["shift_toward_incorrect"], old, (c["task_id"], k["candidate_id"]))


if __name__ == "__main__":
    unittest.main()
