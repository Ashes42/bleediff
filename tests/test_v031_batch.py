"""Batch mode of the decision tracker (``bleediff track --arms``). Every test uses a mock llama-server (SIMULATED
probabilities chosen by marker words in the prompt); none of it is research evidence."""

from __future__ import annotations

import atexit
import csv
import dataclasses
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
from tests.test_v030_track import FIXTURES, ScriptedState, edit, run_session, serve_state  # noqa: E402

from bleediff import batch as BT  # noqa: E402
from bleediff import track as T  # noqa: E402
from bleediff.client import LlamaClient  # noqa: E402
from bleediff.errors import Inconclusive, ServerUnreachable  # noqa: E402

SYNTH = os.path.join(ROOT, "tests", "data", "batch_candidates_synthetic.json")
BASE = [0.70, 0.20, 0.05, 0.05]
RULES = [("ALPHA", [0.30, 0.55, 0.10, 0.05]), ("ECHO", [0.20, 0.70, 0.05, 0.05]), ("BRAVO", [0.55, 0.35, 0.05, 0.05]),
         ("CHARLIE", [0.60, 0.30, 0.05, 0.05]), ("DELTA", [0.85, 0.10, 0.025, 0.025]), ("FOXTROT", [0.40, 0.40, 0.10, 0.10])]


def tmpdir():
    d = tempfile.mkdtemp(prefix="bleediff-batch-test-")
    atexit.register(shutil.rmtree, d, True)
    return d


def write_json(doc, d=None, name="cands.json"):
    p = os.path.join(d or tmpdir(), name)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh)
    return p


def write_raw(raw: bytes, d=None, name="cands.json"):
    p = os.path.join(d or tmpdir(), name)
    with open(p, "wb") as fh:
        fh.write(raw)
    return p


def run_batch(state, arms_path, *, extra=(), out=None):
    out_dir = out or os.path.join(tmpdir(), "batch")
    with serve_state(state) as url:
        code, text = run_cli("track", "--server", url, "--arms", arms_path, "--out", out_dir, "--fixtures", FIXTURES, "--timeout", "30", *extra)
    return code, text, out_dir


def hist(out_dir):
    with open(os.path.join(out_dir, "history.jsonl"), encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def leaderboard(out_dir):
    with open(os.path.join(out_dir, "leaderboard.csv"), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def jload(path):
    return json.loads(read(path))


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class TestCandidateFile(unittest.TestCase):
    def test_all_accepted_shapes(self):
        items = [{"label": "a", "text": "one"}, {"label": "b", "text": "two"}]
        shapes = [items, {"candidates": items}, {"arms": {"a": "one", "b": {"text": "two"}}}, {"fragments": {"a": {"text": "one"}, "b": "two"}}]
        for doc in shapes:
            got = BT.load_candidates(write_json(doc))
            self.assertEqual([(c.label, c.text) for c in got], [("a", "one"), ("b", "two")], doc)

    def test_text_is_normalised_like_the_interactive_file(self):
        c = BT.load_candidates(write_json([{"label": "a", "text": "  line one\r\nline two \n\n"}]))[0]
        self.assertEqual(c.text, "line one\nline two")

    def test_utf8_bom_and_non_ascii_text(self):
        raw = b"\xef\xbb\xbf" + json.dumps([{"label": "café", "text": "naïve ✓"}], ensure_ascii=False).encode("utf-8")
        c = BT.load_candidates(write_raw(raw))[0]
        self.assertEqual((c.label, c.text), ("café", "naïve ✓"))

    def test_file_level_problems_are_errors(self):
        bad = {
            "empty list": [], "empty candidates": {"candidates": []}, "scalar": 7, "wrong object": {"foo": 1},
            "missing label": [{"text": "x"}], "blank label": [{"label": "  ", "text": "x"}], "non-string text": [{"label": "a", "text": 5}],
            "duplicate label": [{"label": "a", "text": "x"}, {"label": "a", "text": "y"}], "control char in label": [{"label": "a\nb", "text": "x"}],
            "not an object": ["x"],
        }
        for name, doc in bad.items():
            with self.subTest(name), self.assertRaises(Inconclusive):
                BT.load_candidates(write_json(doc))
        with self.assertRaises(BT.CandidateFileError):
            BT.load_candidates(write_raw(b"{not json"))
        with self.assertRaises(BT.CandidateFileError):
            BT.load_candidates(write_raw(b"\xff\xfe\xfa"))
        with self.assertRaises(BT.CandidateFileError):
            BT.load_candidates(write_raw(b'[{"label": "a", "text": "x", "label": "b"}]'))            # duplicate JSON key
        with self.assertRaises(BT.CandidateFileError):
            BT.load_candidates(write_raw(b'[{"label": "a", "text": NaN}]'))
        with self.assertRaises(BT.CandidateFileError):
            BT.load_candidates(os.path.join(tmpdir(), "missing.json"))
        try:
            BT.load_candidates(write_json([]))
        except Inconclusive as e:
            self.assertEqual(e.reason, "EMPTY_CANDIDATE_LIST")

    def test_bad_individual_candidates_are_kept_as_invalid_not_scored(self):
        good = "fine text"
        doc = [{"label": "ok", "text": good, "sha256": hashlib.sha256(good.encode()).hexdigest()},
               {"label": "empty", "text": "  \n "}, {"label": "nul", "text": "a\u0000b"},
               {"label": "long", "text": "x" * (T.MAX_FRAGMENT_CHARS + 1)}, {"label": "hash", "text": "abc", "sha256": "0" * 64}]
        got = {c.label: c for c in BT.load_candidates(write_json(doc))}
        self.assertIsNone(got["ok"].problem)
        self.assertEqual(got["empty"].problem_reason, "FRAGMENT_EMPTY")
        self.assertEqual(got["nul"].problem_reason, "FRAGMENT_UNREADABLE")
        self.assertEqual(got["long"].problem_reason, "FRAGMENT_TOO_LONG")
        self.assertEqual(got["hash"].problem_reason, "FRAGMENT_HASH_MISMATCH")
        self.assertTrue(all(got[k].text is None for k in ("empty", "nul", "long", "hash")))


class TestRanking(unittest.TestCase):
    def row(self, i, movement):
        return {"index": i, "movement": movement}

    def test_orders_by_movement_and_shares_rank_on_ties(self):
        rows = [self.row(1, 0.5), self.row(2, 2.0), self.row(3, 0.5), self.row(4, -1.0), self.row(5, 0.5 + 1e-12)]
        ranked = BT.rank_rows(rows)
        self.assertEqual([r["index"] for r in ranked], [2, 1, 3, 5, 4])                  # ties keep file order
        self.assertEqual([r["rank"] for r in ranked], [1, 2, 2, 2, 5])                   # 1, 2=, 2=, 2=, 5
        self.assertEqual([r["tied"] for r in ranked], [False, True, True, True, False])

    def test_movement_direction_follows_the_designated_answer(self):
        self.assertEqual(BT.movement_toward(-1.5, "B"), 1.5)
        self.assertEqual(BT.movement_toward(-1.5, "A"), -1.5)
        self.assertEqual(BT.movement_toward(0.0, "B"), 0.0)

    def test_csv_formula_guard_and_modal(self):
        self.assertEqual(BT._csv_safe("=1+1"), "'=1+1")
        self.assertEqual(BT._csv_safe("-x"), "'-x")
        self.assertEqual(BT._csv_safe("plain"), "plain")
        self.assertEqual(BT._modal([245, 245, 242, 243, 243]), 243)                         # tie 245x2 vs 243x2 -> the smaller value
        self.assertIsNone(BT._modal([]))


class TestBatchRun(unittest.TestCase):
    def setUp(self):
        self.state = ScriptedState(RULES, default=BASE)
        self.code, self.text, self.out = run_batch(self.state, SYNTH)
        self.h = hist(self.out)

    def test_runs_and_exits_zero(self):
        self.assertEqual(self.code, 0, self.text)

    def test_baseline_is_measured_once_then_one_request_per_candidate(self):
        ms = [r for r in self.h if r["type"] == "measurement"]
        self.assertEqual([r["label"] for r in ms], ["baseline"] + ["candidate"] * 7)
        self.assertEqual(len(self.state.requests), 8)
        self.assertEqual(self.h[0]["type"], "session_start")
        self.assertEqual(self.h[-1]["type"], "session_end")
        self.assertEqual(self.h[-1]["reason"], "batch_complete")

    def test_provenance_and_configuration_are_recorded(self):
        start = self.h[0]
        self.assertEqual(start["measurement_type"], "single_prompt_pairwise_margin")
        self.assertIn("model_hash", start["model"])
        self.assertIn("tokenization", start)
        self.assertEqual(start["inference"]["n_probs"], 32)
        ev = [r for r in self.h if r["type"] == "event" and r["event"] == "batch_start"][0]
        self.assertEqual(ev["candidate_file"]["sha256"], sha(SYNTH))
        self.assertEqual(ev["designated_incorrect_answer"], "B")
        self.assertIn("OBSERVED CANDIDATE EFFECTS", ev["observed_effects_note"])

    def test_margins_probabilities_and_changes_are_the_unchanged_calculations(self):
        ms = {r["candidate"]["label"]: r for r in self.h if r["type"] == "measurement" and r["label"] == "candidate"}
        base = [r for r in self.h if r["type"] == "measurement" and r["label"] == "baseline"][0]
        self.assertAlmostEqual(base["margin_AB"], math.log(0.70 / 0.20), places=9)
        for label, vec in (("alpha-toward-B", [0.30, 0.55, 0.10, 0.05]), ("echo-overall-flip", [0.20, 0.70, 0.05, 0.05]), ("delta-toward-A", [0.85, 0.10, 0.025, 0.025])):
            r = ms[label]
            self.assertAlmostEqual(r["margin_AB"], T.margin_ab(vec), places=9)
            self.assertAlmostEqual(r["change_from_baseline"], T.margin_ab(vec) - T.margin_ab(BASE), places=9)
            self.assertAlmostEqual(r["change_from_previous"], r["change_from_baseline"], places=12)     # candidates are independent
            for c, p in zip("ABCD", vec):
                self.assertAlmostEqual(r["probabilities"][c], p, places=9)

    def test_overall_preferred_answer_and_difference_from_baseline(self):
        ms = {r["candidate"]["label"]: r for r in self.h if r["type"] == "measurement" and r["label"] == "candidate"}
        self.assertEqual(ms["echo-overall-flip"]["preferred_choice"], "B")
        self.assertTrue(ms["echo-overall-flip"]["choice_flipped_from_baseline"])
        self.assertTrue(ms["echo-overall-flip"]["pairwise_crossed_from_baseline"])
        self.assertEqual(ms["bravo-mild"]["preferred_choice"], "A")
        self.assertFalse(ms["bravo-mild"]["choice_flipped_from_baseline"])
        # a tied A/B pair is not a crossing and, with A winning the arg-max tie, not a flip
        self.assertEqual(ms["=formula-looking-label"]["margin_AB"], 0.0)
        self.assertFalse(ms["=formula-looking-label"]["pairwise_crossed_from_baseline"])

    def test_leaderboard_order_ranks_and_ties(self):
        rows = leaderboard(self.out)
        self.assertEqual([r["label"] for r in rows], ["echo-overall-flip", "alpha-toward-B", "'=formula-looking-label", "bravo-mild",
                                                      "charlie-tie-one", "charlie-tie-two", "delta-toward-A"])
        self.assertEqual([r["rank"] for r in rows], ["1", "2", "3", "4", "5", "5", "7"])
        moves = [float(r["movement_toward_incorrect_nats"]) for r in rows]
        self.assertEqual(moves, sorted(moves, reverse=True))
        self.assertAlmostEqual(moves[0], T.margin_ab(BASE) - T.margin_ab([0.20, 0.70, 0.05, 0.05]), places=6)
        self.assertLess(moves[-1], 0)                                                   # moved toward A: negative movement toward B

    def test_leaderboard_columns_distributions_and_length_fields(self):
        rows = {r["label"]: r for r in leaderboard(self.out)}
        r = rows["alpha-toward-B"]
        self.assertEqual(list(r.keys()), BT.CSV_COLUMNS)
        self.assertAlmostEqual(sum(float(r[f"p_{c}"]) for c in "ABCD"), 1.0, places=6)
        self.assertEqual(r["preferred_choice"], "B")
        self.assertEqual(r["preferred_differs_from_baseline"], "yes")
        text = "ALPHA candidate fragment that the mock server answers B-leaning."
        self.assertEqual(int(r["fragment_chars"]), len(text))
        self.assertEqual(int(r["fragment_bytes"]), len(text.encode()))
        self.assertEqual(int(r["fragment_words"]), len(text.split()))
        self.assertGreater(int(r["fragment_tokens"]), 0)
        self.assertGreater(int(r["prompt_tokens"]), int(r["fragment_tokens"]))
        self.assertEqual(r["fragment_sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(rows["bravo-mild"]["preferred_differs_from_baseline"], "no")

    def test_token_counts_are_in_history_and_baseline_is_shorter(self):
        ms = [r for r in self.h if r["type"] == "measurement"]
        self.assertEqual(ms[0]["tokens"]["fragment_tokens"], 0)
        for r in ms[1:]:
            self.assertGreater(r["tokens"]["prompt_tokens"], ms[0]["tokens"]["prompt_tokens"])

    def test_terminal_summary_is_ranked_labelled_and_makes_no_causal_claim(self):
        self.assertIn("RANKED BY MOVEMENT TOWARD B (observed candidate effects)", self.text)
        table = self.text[self.text.index("RANKED BY"):]                                # the ranked section, not the progress lines
        order = [table.index(k) for k in ("echo-overall-flip", "alpha-toward-B", "bravo-mild", "delta-toward-A")]
        self.assertEqual(order, sorted(order))
        self.assertIn("5=", self.text)                                                  # the tie marker
        self.assertIn("A>B", self.text)                                                 # preferred answer changed
        self.assertIn("A large shift does not show that a particular word caused it", self.text)
        self.assertNotIn("proves", self.text.lower())
        self.assertIn("LENGTH-UNMATCHED", self.text)                                    # the synthetic fragments differ in length
        self.assertNotIn(os.path.dirname(SYNTH), self.text)                             # no absolute paths of the candidate file

    def test_manifest_records_hashes_that_verify(self):
        man = jload(os.path.join(self.out, "batch-manifest.json"))
        self.assertEqual(man["schema"], BT.BATCH_SCHEMA)
        self.assertEqual(man["files"]["history.jsonl"], sha(os.path.join(self.out, "history.jsonl")))
        self.assertEqual(man["files"]["leaderboard.csv"], sha(os.path.join(self.out, "leaderboard.csv")))
        self.assertEqual(man["candidate_file"]["sha256"], sha(SYNTH))
        self.assertEqual(man["counts"], {"candidates": 7, "scored": 7, "failed_or_invalid_or_not_run": 0})
        self.assertEqual(man["ranking"][0]["label"], "echo-overall-flip")
        self.assertTrue(man["model"]["model_hash_unchanged_at_end"])
        self.assertIn("OBSERVED CANDIDATE EFFECTS", man["observed_effects_note"])
        self.assertNotIn(ROOT, json.dumps(man))                                         # no absolute local paths

    def test_candidate_text_is_not_stored_by_default(self):
        blob = read(os.path.join(self.out, "history.jsonl")) + read(os.path.join(self.out, "leaderboard.csv"))
        self.assertNotIn("candidate fragment", blob)
        self.assertFalse(os.path.exists(os.path.join(self.out, "prompts")))

    def test_identical_to_the_interactive_tracker_for_the_same_fragments(self):
        """The batch measurement of a fragment equals an interactive session's measurement of the same text: same prompt hash,
        same probabilities, same margin."""
        cands = BT.load_candidates(SYNTH)
        batch = {r["candidate"]["label"]: r for r in self.h if r["type"] == "measurement" and r["label"] == "candidate"}
        for c in cands[:3]:
            state = ScriptedState(RULES, default=BASE)
            _code, _t, hist2, _o, _ = run_session(state, [edit(c.text), "quit"])
            m = [r for r in hist2 if r["type"] == "measurement"][1]
            b = batch[c.label]
            self.assertEqual(m["prompt"]["sha256"], b["prompt"]["sha256"])
            self.assertEqual(m["probabilities"], b["probabilities"])
            self.assertEqual(m["margin_AB"], b["margin_AB"])
            self.assertEqual(m["change_from_baseline"], b["change_from_baseline"])


class TestOptions(unittest.TestCase):
    def test_incorrect_answer_a_reverses_the_ranking(self):
        code, text, out = run_batch(ScriptedState(RULES, default=BASE), SYNTH, extra=("--incorrect", "A"))
        self.assertEqual(code, 0)
        rows = leaderboard(out)
        self.assertEqual(rows[0]["label"], "delta-toward-A")
        self.assertEqual(rows[-1]["label"], "echo-overall-flip")
        self.assertIn("RANKED BY MOVEMENT TOWARD A", text)

    def test_retain_prompts_keeps_local_prompt_files(self):
        _c, _t, out = run_batch(ScriptedState(RULES, default=BASE), SYNTH, extra=("--retain-prompts",))
        names = os.listdir(os.path.join(out, "prompts"))
        self.assertEqual(len([n for n in names if n.endswith(".fragment.txt")]), 7)

    def test_batch_and_interactive_fragment_cannot_be_combined(self):
        d = tmpdir()
        code, text, out = run_batch(ScriptedState(RULES, default=BASE), SYNTH, extra=("--fragment", os.path.join(d, "f.txt")), out=os.path.join(d, "o"))
        self.assertEqual(code, 2)
        self.assertIn("ERROR [BAD_ARGS]", text)
        self.assertFalse(os.path.exists(out))

    def test_existing_output_directory_is_never_overwritten(self):
        d = tmpdir()
        os.makedirs(os.path.join(d, "taken"))
        with open(os.path.join(d, "taken", "keep.txt"), "w", encoding="utf-8") as fh:
            fh.write("mine")
        code, text, _ = run_batch(ScriptedState(RULES, default=BASE), SYNTH, out=os.path.join(d, "taken"))
        self.assertEqual(code, 2)
        self.assertIn("OUTPUT_EXISTS", text)
        self.assertEqual(os.listdir(os.path.join(d, "taken")), ["keep.txt"])

    def test_unknown_scenario_refused(self):
        code, text, out = run_batch(ScriptedState(RULES, default=BASE), SYNTH, extra=("--scenario", "nope"))
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out))

    def test_interactive_mode_still_works_without_arms(self):
        state = ScriptedState(RULES, default=BASE)
        code, text, h, _o, _ = run_session(state, [edit("ALPHA something"), "quit"])
        self.assertEqual(code, 0)
        self.assertIn("EDIT 01", text)
        self.assertEqual([r["label"] for r in h if r["type"] == "measurement"], ["baseline", "edit"])


class TestErrorHandling(unittest.TestCase):
    def test_empty_candidate_list_is_refused_and_creates_nothing(self):
        for doc in ([], {"candidates": []}):
            code, text, out = run_batch(ScriptedState(default=BASE), write_json(doc))
            self.assertEqual(code, 2)
            self.assertIn("EMPTY_CANDIDATE_LIST", text)
            self.assertFalse(os.path.exists(out))

    def test_invalid_file_is_refused_before_the_server_is_contacted(self):
        state = ScriptedState(default=BASE)
        code, text, out = run_batch(state, write_raw(b"{broken"))
        self.assertEqual(code, 2)
        self.assertIn("CANDIDATE_FILE_INVALID", text)
        self.assertFalse(os.path.exists(out))
        self.assertEqual(state.requests, [])

    def test_unreachable_server_creates_nothing(self):
        out = os.path.join(tmpdir(), "x")
        code, text = run_cli("track", "--server", "http://127.0.0.1:9", "--arms", SYNTH, "--out", out, "--fixtures", FIXTURES, "--timeout", "3")
        self.assertEqual(code, 2)
        self.assertIn("SERVER_UNREACHABLE", text)
        self.assertFalse(os.path.exists(out))

    def test_baseline_failure_exits_2_and_scores_no_candidate(self):
        state = ScriptedState(omit_probs=True)
        code, text, out = run_batch(state, SYNTH)
        self.assertEqual(code, 2)
        self.assertIn("ERROR [PROBABILITY_FIELDS_MISSING]", text)
        h = hist(out)
        self.assertEqual([r["type"] for r in h], ["session_start", "measurement", "session_end"])
        self.assertEqual(h[-1]["reason"], "baseline_failed")
        self.assertTrue(all("candidate fragment" not in r["prompt"] for r in state.requests))      # no candidate reached the model
        self.assertFalse(os.path.exists(os.path.join(out, "leaderboard.csv")))

    def test_a_failed_request_is_recorded_and_the_rest_are_ranked(self):
        class Flaky(ScriptedState):
            def completion(self, payload):
                if "BRAVO" in payload["prompt"]:
                    self.requests.append(payload)
                    return {"content": "A"}                       # no probabilities
                return super().completion(payload)

        code, text, out = run_batch(Flaky(RULES, default=BASE), SYNTH)
        self.assertEqual(code, 0)
        self.assertIn("FAILED [PROBABILITY_FIELDS_MISSING]", text)
        self.assertIn("NOT SCORED: bravo-mild", text)
        rows = leaderboard(out)
        self.assertEqual(len(rows), 7)
        bravo = [r for r in rows if r["label"] == "bravo-mild"][0]
        self.assertEqual((bravo["rank"], bravo["status"], bravo["error_reason"], bravo["margin_AB"]), ("", "error", "PROBABILITY_FIELDS_MISSING", ""))
        self.assertEqual(rows[-1]["label"], "bravo-mild")                                # unscored rows come last, with no imputed numbers
        self.assertEqual([r["rank"] for r in rows[:-1]], ["1", "2", "3", "4", "4", "6"])
        err = [r for r in hist(out) if r["type"] == "measurement" and r.get("candidate", {}).get("label") == "bravo-mild"][0]
        self.assertEqual(err["status"], "error")
        self.assertNotIn("probabilities", err)
        man = jload(os.path.join(out, "batch-manifest.json"))
        self.assertEqual(man["counts"]["failed_or_invalid_or_not_run"], 1)

    def test_invalid_probabilities_are_errors_never_zeros(self):
        rules = RULES + [("GOLF", [0.0, 0.9, 0.05, 0.05])]
        doc = [{"label": "good", "text": "ALPHA ok"}, {"label": "zero-A", "text": "GOLF text"}]
        code, text, out = run_batch(ScriptedState(rules, default=BASE), write_json(doc))
        self.assertEqual(code, 0)
        rows = {r["label"]: r for r in leaderboard(out)}
        self.assertEqual(rows["zero-A"]["status"], "error")
        self.assertEqual(rows["zero-A"]["margin_AB"], "")
        self.assertEqual(rows["good"]["rank"], "1")

    def test_invalid_candidates_are_reported_not_sent_to_the_model(self):
        doc = [{"label": "ok", "text": "ALPHA ok"}, {"label": "empty", "text": ""}, {"label": "bad-hash", "text": "ECHO", "sha256": "1" * 64}]
        state = ScriptedState(RULES, default=BASE)
        code, text, out = run_batch(state, write_json(doc))
        self.assertEqual(code, 0)
        self.assertEqual(len(state.requests), 2)                                        # baseline + the one valid candidate
        self.assertIn("INVALID", text)
        rows = {r["label"]: r for r in leaderboard(out)}
        self.assertEqual((rows["empty"]["status"], rows["empty"]["error_reason"]), ("invalid", "FRAGMENT_EMPTY"))
        self.assertEqual(rows["bad-hash"]["error_reason"], "FRAGMENT_HASH_MISMATCH")

    def test_all_candidates_invalid_exits_2(self):
        code, text, out = run_batch(ScriptedState(default=BASE), write_json([{"label": "e", "text": ""}]))
        self.assertEqual(code, 2)
        self.assertIn("NO_CANDIDATE_SCORED", text)
        self.assertTrue(os.path.exists(os.path.join(out, "leaderboard.csv")))           # the failures are still saved

    def test_the_run_stops_after_repeated_unreachable_server_failures(self):
        real = LlamaClient.choice_probs
        calls = {"n": 0}

        def flaky(self_, prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                return real(self_, prompt)                                              # the baseline
            raise ServerUnreachable("simulated outage")

        doc = [{"label": f"c{i}", "text": f"ALPHA {i}"} for i in range(6)]
        with mock.patch.object(LlamaClient, "choice_probs", flaky):
            code, text, out = run_batch(ScriptedState(RULES, default=BASE), write_json(doc))
        self.assertEqual(code, 2)
        self.assertEqual(calls["n"], 1 + BT.UNREACHABLE_LIMIT)                          # three attempts, then the rest are not run
        rows = leaderboard(out)
        self.assertEqual([r["status"] for r in rows], ["error"] * 3 + ["not_run"] * 3)
        self.assertEqual(rows[-1]["error_reason"], "RUN_STOPPED")

    def test_duplicate_text_under_two_labels_is_flagged(self):
        doc = [{"label": "one", "text": "ALPHA same"}, {"label": "two", "text": "ALPHA same"}]
        code, text, out = run_batch(ScriptedState(RULES, default=BASE), write_json(doc))
        self.assertEqual(code, 0)
        self.assertIn("identical text was scored more than once", text)
        rows = leaderboard(out)
        self.assertEqual([r["rank"] for r in rows], ["1", "1"])                         # identical text, identical movement: a tie

    def test_model_identity_change_during_the_run_is_reported(self):
        state = ScriptedState(RULES, default=BASE)
        real = LlamaClient.server_info
        n = {"i": 0}

        def drifting(self_):
            info = real(self_)
            n["i"] += 1
            if n["i"] >= 2:
                info = dataclasses.replace(info, model_hash="f" * 64)
            return info

        with mock.patch.object(LlamaClient, "server_info", drifting):
            code, text, out = run_batch(state, SYNTH)
        man = jload(os.path.join(out, "batch-manifest.json"))
        self.assertFalse(man["model"]["model_hash_unchanged_at_end"])
        self.assertIn("model identity changed", text)


class TestResearchUntouched(unittest.TestCase):
    def test_probability_extraction_is_the_unchanged_client(self):
        import inspect
        self.assertIn("choice_probs", inspect.getsource(T.TrackSession._measure_prompt))
        self.assertNotIn(".choice_probs(", inspect.getsource(BT))                          # batch mode never extracts probabilities itself


if __name__ == "__main__":
    unittest.main()
