"""Display and labelling corrections for semantic roles: which margin is which, newly induced answers versus flips, and the
generic reading of the historical 'topic' category. Probabilities are SIMULATED by a mock server."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from tests.test_bleediff import run_cli  # noqa: E402
from tests.test_v030_track import ScriptedState  # noqa: E402
from tests.test_v031_batch import run_batch  # noqa: E402
from tests.test_v033_roles import MAPPINGS, SEM_BASE, SEM_VARS, letter_vector, letters_of, sem_state, tmpdir, write_cands  # noqa: E402

from bleediff import atlas as AT  # noqa: E402
from bleediff import batch as BT  # noqa: E402

V030_COLUMNS = ["rank", "label", "status", "movement_toward_incorrect_nats", "margin_AB", "change_from_baseline_nats", "p_A", "p_B", "p_C", "p_D",
                "preferred_choice", "preferred_differs_from_baseline", "pairwise_AB_crossed", "fragment_chars", "fragment_bytes", "fragment_words",
                "fragment_tokens", "prompt_tokens", "prompt_tokens_minus_modal", "fragment_sha256", "error_reason"]
SEM_COLUMNS = ["margin_correct_vs_competing", "margin_correct_vs_strongest", "newly_induced_incorrect", "correct_letter", "competing_letter"]


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def csv_rows(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def run_roles(mapping, extra, base=None, variants=None):
    d = tmpdir()
    code, text, out = run_batch(sem_state(mapping, base=base, variants=variants), write_cands(d), extra=extra, out=os.path.join(d, "b"))
    assert code == 0, text
    return text, out


class TestBatchTerminalLabels(unittest.TestCase):
    def test_roles_at_c_and_d_never_print_an_unqualified_ab_margin(self):
        text, _out = run_roles(MAPPINGS["m3"], ("--correct", "C", "--incorrect", "D"))
        self.assertIsNone(re.search(r"^Margin A/B: ", text, re.M), "an unqualified 'Margin A/B:' line would read as the role margin")
        self.assertIn("Margin A/B (letters A and B only; NOT the role margin):", text)
        m = re.search(r"Role margin ln\(p_C / p_D\) \(correct C vs competing D\): ([+-]\d+\.\d+) nats", text)
        self.assertIsNotNone(m, text)
        self.assertAlmostEqual(float(m.group(1)), math.log(0.60 / 0.05), places=3)
        ab = float(re.search(r"NOT the role margin\): ([+-]\d+\.\d+) nats", text).group(1))
        self.assertNotAlmostEqual(ab, float(m.group(1)), places=2)                      # the two margins are different numbers

    def test_the_summary_names_the_role_margin_the_movement_and_the_flip_distinction(self):
        text, _out = run_roles(MAPPINGS["m4"], ("--correct", "D", "--incorrect", "C"))
        summary = text[text.index("RANKED BY MOVEMENT"):]
        self.assertIn("margin correct(D)/competing(C)", summary)
        self.assertIn("'margin' below is ln(p_correct / p_competing), the role margin", summary)
        self.assertIn("minus its change from the baseline (movement toward the competing answer)", summary)
        self.assertIn("The A/B margin is not shown", summary)
        self.assertIn("'flip' is any change of the preferred answer, a different quantity from a newly induced incorrect answer", summary)
        self.assertNotIn("Baseline: margin A/B", summary)

    def test_default_roles_keep_the_v030_baseline_line_and_the_v030_summary_header(self):
        text, _out = run_roles(MAPPINGS["m1"], ())
        self.assertIn("Margin A/B: +2.485 nats", text)                                   # unchanged (the A/B margin of the simulated baseline)
        self.assertNotIn("NOT the role margin", text)
        self.assertNotIn("Role margin", text)
        self.assertIn("Baseline: margin A/B +2.485 nats, preferred A", text)

    def test_newly_induced_answers_are_reported_apart_from_flips(self):
        text, _out = run_roles(MAPPINGS["m3"], ("--correct", "C", "--incorrect", "D"))
        line = [x for x in text.splitlines() if "Newly induced incorrect answers" in x][0]
        self.assertIn("(correct baseline, the candidate's preferred answer is wrong): 2", line)
        self.assertIn("alpha-1", line)
        self.assertIn("delta-1", line)
        text, _out = run_roles(MAPPINGS["m1"], ())                                      # also reported for the default roles
        self.assertIn("Newly induced incorrect answers (correct baseline, the candidate's preferred answer is wrong): 2", text)

    def test_a_wrong_baseline_is_never_reported_as_newly_induced(self):
        wrong_base = {"K": 0.15, "T": 0.10, "X": 0.60, "Y": 0.15}                        # prefers X: wrong from the start
        text, _out = run_roles(MAPPINGS["m3"], ("--correct", "C", "--incorrect", "D"), base=wrong_base)
        self.assertIn("Baseline already prefers a wrong answer (A): no change below is reported as a newly induced incorrect answer.", text)
        self.assertNotIn("Newly induced incorrect answers (correct baseline", text)


class TestLeaderboardCsvLabels(unittest.TestCase):
    def test_v030_columns_are_unchanged_and_the_semantic_columns_are_appended(self):
        self.assertEqual(BT.CSV_COLUMNS[:len(V030_COLUMNS)], V030_COLUMNS)
        self.assertEqual(BT.CSV_COLUMNS[len(V030_COLUMNS):], SEM_COLUMNS)

    def test_the_fixed_ab_margin_and_the_role_margins_are_separate_columns(self):
        _text, out = run_roles(MAPPINGS["m3"], ("--correct", "C", "--incorrect", "D"))
        rows = csv_rows(os.path.join(out, "leaderboard.csv"))
        self.assertEqual(list(rows[0].keys()), BT.CSV_COLUMNS)
        alpha = [r for r in rows if r["label"] == "alpha-1"][0]
        probs = SEM_VARS["ALPHA"]                                                        # ALPHA: K .20, T .55, X .15, Y .10 under m3 (A=X, B=Y, C=K, D=T)
        self.assertAlmostEqual(float(alpha["margin_AB"]), math.log(probs["X"] / probs["Y"]), places=6)               # fixed A/B margin: letters A and B
        self.assertAlmostEqual(float(alpha["margin_correct_vs_competing"]), math.log(probs["K"] / probs["T"]), places=6)
        self.assertAlmostEqual(float(alpha["margin_correct_vs_strongest"]), math.log(probs["K"] / probs["T"]), places=6)    # T is the strongest incorrect
        base = SEM_BASE
        self.assertAlmostEqual(float(alpha["movement_toward_incorrect_nats"]), -(math.log(probs["K"] / probs["T"]) - math.log(base["K"] / base["T"])), places=6)
        self.assertEqual((alpha["newly_induced_incorrect"], alpha["correct_letter"], alpha["competing_letter"]), ("yes", "C", "D"))
        delta = [r for r in rows if r["label"] == "delta-1"][0]                          # X wins: the strongest incorrect answer is not the competing one
        self.assertNotAlmostEqual(float(delta["margin_correct_vs_strongest"]), float(delta["margin_correct_vs_competing"]), places=3)
        quiet = [r for r in rows if r["label"] == "charlie-1"][0]
        self.assertEqual(quiet["newly_induced_incorrect"], "no")

    def test_default_roles_still_fill_the_semantic_columns_and_failed_rows_keep_only_the_letters(self):
        class Flaky(ScriptedState):
            def completion(self, payload):
                if "BRAVO" in payload["prompt"]:
                    self.requests.append(payload)
                    return {"content": "A"}
                return super().completion(payload)

        d = tmpdir()
        rules = [(k, letter_vector(v, MAPPINGS["m1"])) for k, v in SEM_VARS.items()]
        code, _t, out = run_batch(Flaky(rules, default=letter_vector(SEM_BASE, MAPPINGS["m1"]), choice_mass=0.99), write_cands(d), out=os.path.join(d, "b"))
        self.assertEqual(code, 0)
        rows = csv_rows(os.path.join(out, "leaderboard.csv"))
        ok = [r for r in rows if r["status"] == "ok"][0]
        self.assertEqual((ok["correct_letter"], ok["competing_letter"]), ("A", "B"))
        self.assertAlmostEqual(float(ok["margin_correct_vs_competing"]), float(ok["margin_AB"]), places=9)     # for A/B roles with correct A they coincide
        failed = [r for r in rows if r["status"] != "ok"][0]
        self.assertEqual((failed["margin_correct_vs_competing"], failed["margin_correct_vs_strongest"], failed["newly_induced_incorrect"]), ("", "", ""))
        self.assertEqual((failed["correct_letter"], failed["competing_letter"]), ("A", "B"))


class TestAtlasLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tmpdir()
        cands = write_cands(cls.d)
        mapping = MAPPINGS["m3"]
        correct, competing = letters_of(mapping)
        code, text, _o = run_batch(sem_state(mapping), cands, extra=("--correct", correct, "--incorrect", competing), out=os.path.join(cls.d, "results", "s"))
        assert code == 0, text
        sess = {"id": "m3/m1/raw", "session_dir": "results/s", "task": {"id": "m3", "name": "m3"}, "scenario_id": "config-audit-v2", "model": {"id": "m1", "name": "M1"}, "presentation": "raw",
                "correct_answer": correct, "incorrect_answer": competing, "candidates": {f"{k.lower()}-1": c for k, c in zip(SEM_VARS, ("claim", "topic", "offtopic", "neutral", "id"))},
                "evidence": [{"role": "candidate_file", "path": "cands.json"}]}
        cls.manifest = os.path.join(cls.d, "atlas-input.json")
        with open(cls.manifest, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"schema": AT.INPUT_SCHEMA, "sessions": [sess]}, fh)
        cls.out = os.path.join(cls.d, "report")
        cls.code, cls.text = run_cli("atlas", "--manifest", cls.manifest, "--out", cls.out)
        cls.report = read(os.path.join(cls.out, "report.md"))

    def test_runs(self):
        self.assertEqual(self.code, 0, self.text)

    def test_the_historical_topic_definition_is_kept_and_the_generic_reading_is_added(self):
        for blob in (AT.TOPIC_LIMITATION, AT.CATEGORY_LABELS["topic"], " ".join(AT.COMPARISON_NOTES.values()), self.text, self.report):
            self.assertIn("competing answer's subject", blob)
            self.assertIn("option B's subject in the original A/B experiments", blob)
        footnote = [x for x in self.text.splitlines() if x.strip().startswith("*topic")][0]
        self.assertEqual(footnote.strip(), "*topic = on-topic text that also mentions the competing answer's subject (option B's subject in the original A/B experiments).")
        self.assertNotIn("mentions option B's subject.", footnote)                       # no longer a bare B assumption
        self.assertIn("\\*On-topic text that also mentions the competing answer's subject (option B's subject in the original A/B experiments).", self.report)
        for r in csv_rows(os.path.join(self.out, "category-summary.csv")):
            if r["comparison"].startswith("topic") or r["comparison"] == "claim_minus_topic":
                self.assertIn("competing answer's subject", r["note"])

    def test_the_report_has_a_legend_that_separates_the_five_quantities(self):
        sec = self.report[self.report.index("## Quantities in this report"):self.report.index("## Sources")]
        for needle in ("Fixed A/B log-odds margin", "ln(p_A / p_B)", "`margin_AB`", "Correct-versus-competing margin", "`margin_correct_vs_competing`",
                       "Correct-versus-strongest-incorrect margin", "`margin_correct_vs_strongest`", "Semantic competing-answer movement", "`shift_toward_incorrect`",
                       "Overall flip, A/B crossing, role-order crossing", "`overall_flip`", "`ab_crossed`", "`role_order_crossed`", "Newly induced incorrect answer",
                       "`newly_induced_incorrect`", "a wrong baseline never counts", "AB_MARGIN_NOT_ROLE_MARGIN", "historical definition"):
            self.assertIn(needle, sec, needle)

    def test_the_candidates_csv_carries_each_quantity_in_its_own_column(self):
        rows = csv_rows(os.path.join(self.out, "candidates.csv"))
        r = [x for x in rows if x["candidate_id"] == "alpha-1"][0]
        for col in ("margin_AB", "margin_correct_vs_competing", "margin_correct_vs_strongest", "shift_toward_incorrect", "overall_flip", "ab_crossed", "role_order_crossed", "newly_induced_incorrect"):
            self.assertIn(col, r)
            self.assertNotEqual(r[col], "")
        self.assertNotEqual(r["margin_AB"], r["margin_correct_vs_competing"])
        self.assertAlmostEqual(float(r["shift_toward_incorrect"]), -float(r["change_margin_correct_vs_competing"]), places=9)
        self.assertEqual((r["newly_induced_incorrect"], r["overall_flip"]), ("True", "True"))
        self.assertIn("AB_MARGIN_NOT_ROLE_MARGIN", r["flags"])

    def test_the_terminal_says_which_margin_it_shows(self):
        self.assertIn("NOTE: in 1 cell(s) the correct and competing answers are not at A and B", self.text)
        self.assertIn("role margins ln(p_correct / p_competing)", self.text)
        self.assertIn("margin_AB in atlas.json", self.text)

    def test_the_semantic_section_still_avoids_the_option_b_assumption(self):
        sec = self.report[self.report.index("## Semantic answer roles"):self.report.index("## Task/model cells")]
        self.assertNotIn("option B", sec)

    def test_legacy_example_sessions_keep_their_historical_wording_and_numbers(self):
        out = os.path.join(tmpdir(), "r")
        code, text = run_cli("atlas", "--manifest", os.path.join(ROOT, "examples", "atlas", "pilot", "atlas-input.json"), "--out", out)
        self.assertEqual(code, 0)
        self.assertIn("option B's subject", text)
        self.assertNotIn("NOTE: in", text)                                               # all roles are at A/B: no role-margin note
        report = read(os.path.join(out, "report.md"))
        self.assertIn("## Quantities in this report", report)


class TestDocumentation(unittest.TestCase):
    PERSONAL = re.compile(r"(@[A-Za-z0-9-]+\.(com|org|net|io)|[A-Za-z]:\\\\|[A-Za-z]:\\|/Users/|/home/|AppData|\.cache|huggingface|api[_-]?key\s*[:=]|password\s*[:=])", re.I)

    def test_summary_states_the_requested_findings_and_limits(self):
        t = " ".join(read(os.path.join(ROOT, "docs", "FOUR_MAPPING_SUMMARY.md")).split())
        for needle in ("exploratory", "Four of the 24 possible answer-letter permutations", "all 24 cells", "positive in 24/24", "exceeded neutral text in 24/24",
                       "reverse across mappings and are dominated by answer-letter behavior", "mapping-dependent", "not** independent samples",
                       "Nothing here is a universal claim", "9 (m1) and 1 (m2), 10 in total", "No real tenant-isolation vulnerability was demonstrated"):
            self.assertIn(needle, t, needle)
        self.assertIsNone(self.PERSONAL.search(t))
        self.assertNotIn("Ashley", t)

    def test_readme_and_changelog_describe_semantic_roles(self):
        readme = " ".join(read(os.path.join(ROOT, "README.md")).split())
        for needle in ("--correct C --incorrect D", "semantic roles", "margin_AB", "docs/FOUR_MAPPING_SUMMARY.md", "only when the clean baseline was correct"):
            self.assertIn(needle, readme, needle)
        cl = read(os.path.join(ROOT, "CHANGELOG.md"))
        self.assertIn("Pending for the next release - semantic answer roles", cl)
        self.assertLess(cl.index("Pending for the next release"), cl.index("## [0.3.0]"))
        self.assertNotIn("Unreleased", cl)
        self.assertIsNone(self.PERSONAL.search(cl))


if __name__ == "__main__":
    unittest.main()
