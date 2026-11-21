"""v0.3 tests: ``bleediff track`` (the interactive decision-margin tracker).

No model and no network beyond a stdlib mock llama-server on 127.0.0.1 whose answer vectors depend on the prompt text, so
edits can move the probabilities in scripted ways. Numerical correctness is also checked against the SAVED Bleediff results
(research/data) without altering them.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

try:
    from test_bleediff import MockState, _Handler, run_cli  # type: ignore
except ImportError:  # pragma: no cover
    from tests.test_bleediff import MockState, _Handler, run_cli

from bleediff import track as T  # noqa: E402
from bleediff.arena import load_fixtures, plan_requests  # noqa: E402
from bleediff.fold import read_traces  # noqa: E402
from bleediff.meta import file_sha256  # noqa: E402

FIXTURES = os.path.join(ROOT, "fixtures", "semantic-v2", "contexts.json")
DATA = os.path.join(ROOT, "research", "data")


def _json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def scenario():
    return {s.id: s for s in load_fixtures(FIXTURES).scenarios}["config-audit-v2"]


class ScriptedState(MockState):
    """Answer vectors chosen by what the prompt contains: the first rule whose marker occurs in the prompt wins."""

    def __init__(self, rules=(), default=(0.70, 0.20, 0.05, 0.05), sequence=None, **kw):
        super().__init__(**kw)
        self.rules = list(rules)
        self.default = list(default)
        self.sequence = list(sequence) if sequence else None

    def vector_for(self, prompt):
        if self.sequence:
            return list(self.sequence.pop(0))
        for marker, vec in self.rules:
            if marker in prompt:
                return list(vec)
        return list(self.default)


@contextlib.contextmanager
def serve_state(state):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.state = state
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def history(path):
    with open(os.path.join(path, "history.jsonl"), "r", encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def measurements(path):
    return [r for r in history(path) if r["type"] == "measurement"]


def run_session(state, inputs, *, extra=(), out=None, fragment=None):
    """Drive a full ``bleediff track`` session. ``inputs`` items are strings (returned from input()) or callables(path) that
    edit the fragment file and return what input() should return."""
    d = tempfile.mkdtemp(prefix="bleediff-track-test-")           # kept until interpreter exit so the caller can read the files
    atexit.register(shutil.rmtree, d, True)
    out_dir = out or os.path.join(d, "session")
    frag = fragment or os.path.join(out_dir, "fragment.txt")
    it = iter(inputs)

    def fake_input(prompt=""):
        try:
            item = next(it)
        except StopIteration:
            raise EOFError
        return item(frag) if callable(item) else item

    with serve_state(state) as url, mock.patch("builtins.input", side_effect=fake_input):
        args = ["track", "--server", url, "--out", out_dir, "--fixtures", FIXTURES, "--timeout", "30", *extra]
        if fragment:
            args += ["--fragment", fragment]
        code, text = run_cli(*args)
    return code, text, history(out_dir) if os.path.isdir(out_dir) else None, out_dir, d


def edit(text):
    def _do(path):
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        return ""
    return _do


# --------------------------------------------------------------------------------------
class TestMarginMath(unittest.TestCase):
    def test_signed_margins(self):
        self.assertAlmostEqual(T.margin_ab([0.6, 0.3, 0.05, 0.05]), math.log(2), places=12)        # positive: A over B
        self.assertEqual(T.margin_ab([0.4, 0.4, 0.1, 0.1]), 0.0)                                       # zero: tied
        self.assertAlmostEqual(T.margin_ab([0.2, 0.6, 0.1, 0.1]), -math.log(3), places=12)            # negative: B over A
        self.assertEqual(T.pairwise_order(1.0), "A_over_B")
        self.assertEqual(T.pairwise_order(-1.0), "B_over_A")
        self.assertEqual(T.pairwise_order(0.0), "A_equals_B")
        self.assertAlmostEqual(T.margin_ab([0.2, 0.6, 0.1, 0.1]), -T.margin_ab([0.6, 0.2, 0.1, 0.1]), places=12)   # fixed labels

    def test_margin_ignores_c_and_d_but_the_preferred_choice_does_not(self):
        a = [0.5, 0.2, 0.2, 0.1]
        b = [0.3, 0.1, 0.55, 0.05]                                          # same sign margin, but C now wins
        self.assertGreater(T.margin_ab(a), 0)
        self.assertGreater(T.margin_ab(b), 0)
        c = T.compare({"margin": T.margin_ab(a), "preferred": "A"}, {"margin": T.margin_ab(a), "preferred": "A"},
                      {"margin": T.margin_ab(b), "preferred": "C"})
        self.assertTrue(c["choice_flipped_from_previous"])
        self.assertFalse(c["pairwise_crossed_from_previous"])             # an overall flip WITHOUT any A/B crossing

    def test_margin_changes_are_exact(self):
        base = {"margin": 1.25, "preferred": "A"}
        prev = {"margin": 0.40, "preferred": "A"}
        cur = {"margin": -0.10, "preferred": "A"}
        c = T.compare(prev, base, cur)
        self.assertAlmostEqual(c["change_from_baseline"], -1.35, places=12)
        self.assertAlmostEqual(c["change_from_previous"], -0.50, places=12)
        first = T.compare(None, base, cur)
        self.assertIsNone(first["change_from_previous"])
        self.assertIsNone(first["choice_flipped_from_previous"])

    def test_pairwise_crossing_is_not_an_overall_flip(self):
        # (a) crossing, C wins before and after: no overall flip
        before, after = [0.2, 0.1, 0.6, 0.1], [0.1, 0.2, 0.6, 0.1]
        pb, pa = T.margin_ab(before), T.margin_ab(after)
        c = T.compare({"margin": pb, "preferred": "C"}, {"margin": pb, "preferred": "C"}, {"margin": pa, "preferred": "C"})
        self.assertTrue(c["pairwise_crossed_from_previous"])
        self.assertFalse(c["choice_flipped_from_previous"])
        self.assertIn("overall preferred answer unchanged: C", T.describe_margin_change(pb, pa, preferred="C", prev_preferred="C"))
        # (b) crossing and a real flip A -> B
        pb, pa = T.margin_ab([0.5, 0.3, 0.1, 0.1]), T.margin_ab([0.3, 0.5, 0.1, 0.1])
        text = T.describe_margin_change(pb, pa, preferred="B", prev_preferred="A")
        self.assertIn("B now above A", text)
        self.assertIn("overall preferred answer changed: A -> B", text)
        # (c) crossing, but C takes over: reported as a change to C, not to B
        self.assertIn("A -> C", T.describe_margin_change(pb, pa, preferred="C", prev_preferred="A"))

    def test_status_text(self):
        d = T.describe_margin_change
        self.assertEqual(d(1.3, 0.45), "A/B margin narrowed")
        self.assertEqual(d(0.45, 1.3), "A/B margin widened")
        self.assertEqual(d(-0.2, -0.9), "A/B margin widened")
        self.assertEqual(d(-0.9, -0.2), "A/B margin narrowed")
        self.assertEqual(d(0.5, 0.5 + 1e-12), "A/B margin unchanged")
        self.assertEqual(d(0.5, 0.0), "A and B are now tied")
        self.assertEqual(d(0.0, -0.4), "A/B margin opened toward B")
        for banned in ("SAFE", "CRITICAL", "GUARANTEED"):
            self.assertNotIn(banned, d(1.0, -1.0, preferred="B", prev_preferred="A"))

    def test_missing_or_invalid_probabilities_are_errors_never_zeros(self):
        bad = {"too few": [0.5, 0.5], "none": None, "nan": [float("nan"), 0.5, 0.25, 0.25], "inf": [float("inf"), 0.0, 0.0, 0.0],
               "missing B": [0.5, None, 0.25, 0.25], "negative": [1.2, -0.2, 0.0, 0.0], "not summing": [0.5, 0.2, 0.1, 0.1],
               "zero A": [0.0, 0.5, 0.25, 0.25], "zero B": [0.5, 0.0, 0.25, 0.25], "bool": [True, 0.0, 0.0, 0.0]}
        for name, v in bad.items():
            self.assertTrue(T.validate_probs(v), name)
            with self.assertRaises(T.MeasurementError, msg=name):
                T.margin_ab(v)
        self.assertEqual(T.validate_probs([0.4, 0.3, 0.2, 0.1]), [])

    def test_matches_the_saved_bleediff_results_without_altering_them(self):
        """margin_AB change from the clean baseline equals minus the historical delta_contrast for every saved record."""
        res = _json(os.path.join(DATA, "results-v0.2.0.json"))
        for fname, model in (("evidence-qwen1.5b-12-controls-pass1.jsonl", "qwen2.5-1.5b-instruct-q4_k_m"),
                             ("evidence-qwen2.5-7b-pass1.jsonl", "qwen2.5-7b-instruct-q4_k_m")):
            b_ref = res["models"][model]["baseline"]["B_ref"]
            base_margin = T.margin_ab(b_ref)
            for rec in read_traces(os.path.join(DATA, fname)):
                vec = [rec["p_A"], rec["p_B"], rec["p_C"], rec["p_D"]]
                change = T.margin_ab(vec) - base_margin
                self.assertAlmostEqual(change, -rec["delta_contrast"], places=9, msg=(fname, rec["trial"]))
        b15 = res["models"]["qwen2.5-1.5b-instruct-q4_k_m"]
        self.assertAlmostEqual(T.margin_ab(b15["baseline"]["B_ref"]), 1.308, places=3)                 # +1.308 nats baseline
        treat = b15["measurements"]["treatment_original_B"]
        m = T.margin_ab([treat["p_A"], treat["p_B"], treat["p_C"], treat["p_D"]])
        self.assertAlmostEqual(m, 0.451, places=3)                                                        # +0.451 nats
        self.assertAlmostEqual(m - T.margin_ab(b15["baseline"]["B_ref"]), -0.857, places=3)               # -0.857 nats
        # a raw A -> B flip on the 7B model is a margin crossing AND an arg-max change there
        t7 = res["models"]["qwen2.5-7b-instruct-q4_k_m"]["measurements"]["treatment_original_B"]
        self.assertLess(T.margin_ab([t7["p_A"], t7["p_B"], t7["p_C"], t7["p_D"]]), 0)

    def test_the_research_code_was_not_touched(self):
        from bleediff import robustness as RB

        pins = _json(os.path.join(ROOT, "fixtures", "config-audit-robustness", "protocol-1.1.json"))["frozen"]["source_sha256_newline_normalised"]
        for rel, want in pins.items():
            self.assertEqual(RB.source_sha256(os.path.join(ROOT, *rel.split("/"))), want, rel)


# --------------------------------------------------------------------------------------
class TestPromptsAndFragments(unittest.TestCase):
    def test_prompts_equal_the_existing_planner_byte_for_byte(self):
        sc = scenario()
        clean = bytes(plan_requests(sc, "clean").prompts[2]).decode()
        self.assertEqual(T.build_prompt(sc, None), clean)
        arms = _json(os.path.join(DATA, "frozen-arms.json"))["arms"]
        self.assertEqual(len(arms), 5)
        for name, a in arms.items():
            import dataclasses

            probe = dataclasses.replace(sc, adjacent=a["text"] + " " + " ".join(sc.adjacent.split()[25:]))
            plan = plan_requests(probe, "treatment", 25, "words")
            try:
                want = bytes(plan.prompts[2]).decode()
            finally:
                plan.dispose()
            self.assertEqual(T.build_prompt(sc, a["text"]), want, name)
        self.assertTrue(clean.endswith("Answer:"))

    def test_non_ascii_fragments_are_supported_in_utf8(self):
        p = T.build_prompt(scenario(), "café ✓")
        self.assertIn("café ✓", p)

    def test_reading_the_fragment_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "f.txt")
            with open(f, "wb") as fh:
                fh.write(b"\xef\xbb\xbf  Line one\r\nLine two \r\n\r\n")
            self.assertEqual(T.read_fragment(f), "Line one\nLine two")                # BOM, CRLF and edge whitespace normalised
            with open(f, "wb") as fh:
                fh.write(b"abc\xff")
            with self.assertRaises(T.FragmentError):
                T.read_fragment(f)
            with open(f, "wb") as fh:
                fh.write(b"a\x00b")
            with self.assertRaises(T.FragmentError):
                T.read_fragment(f)
            with self.assertRaises(T.FragmentError):
                T.read_fragment(os.path.join(d, "missing.txt"))
            with open(f, "w", encoding="utf-8") as fh:
                fh.write("x" * (T.MAX_FRAGMENT_CHARS + 1))
            with self.assertRaises(T.FragmentError):
                T.read_fragment(f)


# --------------------------------------------------------------------------------------
class TestSessions(unittest.TestCase):
    def test_full_scripted_session_margins_changes_flips_and_crossings(self):
        rules = [("ALPHA", [0.45, 0.40, 0.10, 0.05]),      # A/B margin narrows, A still wins
                 ("BRAVO", [0.30, 0.40, 0.20, 0.10]),      # A/B crosses AND the answer flips A -> B
                 ("CHARLIE", [0.10, 0.20, 0.60, 0.10]),    # B over A still, but C wins: an overall flip B -> C, no crossing
                 ("DELTA", [0.20, 0.10, 0.60, 0.10])]      # A over B again (crossing) while C still wins: NO overall flip
        state = ScriptedState(rules)
        code, text, hist, out_dir, _ = run_session(state, [edit("ALPHA evidence"), edit("BRAVO evidence"), edit("CHARLIE evidence"),
                                                           edit("DELTA evidence"), "quit"])
        self.assertEqual(code, 0, text)
        ms = [r for r in hist if r["type"] == "measurement"]
        self.assertEqual([m["label"] for m in ms], ["baseline", "edit", "edit", "edit", "edit"])
        self.assertEqual([m["step"] for m in ms], [1, 2, 3, 4, 5])
        base = math.log(0.70 / 0.20)
        self.assertAlmostEqual(ms[0]["margin_AB"], base, places=9)
        self.assertEqual([m["preferred_choice"] for m in ms], ["A", "A", "B", "C", "C"])
        self.assertAlmostEqual(ms[1]["change_from_baseline"], math.log(0.45 / 0.40) - base, places=9)
        self.assertAlmostEqual(ms[2]["change_from_previous"], math.log(0.30 / 0.40) - math.log(0.45 / 0.40), places=9)
        # ALPHA: narrowed, no crossing, no flip
        self.assertFalse(ms[1]["pairwise_crossed_from_previous"] or ms[1]["choice_flipped_from_previous"])
        # BRAVO: crossing and a real flip
        self.assertTrue(ms[2]["pairwise_crossed_from_previous"] and ms[2]["choice_flipped_from_previous"])
        self.assertEqual(ms[2]["pairwise_order"], "B_over_A")
        # CHARLIE: a real flip without a new crossing
        self.assertTrue(ms[3]["choice_flipped_from_previous"])
        self.assertFalse(ms[3]["pairwise_crossed_from_previous"])
        # DELTA: a crossing without a flip
        self.assertTrue(ms[4]["pairwise_crossed_from_previous"])
        self.assertFalse(ms[4]["choice_flipped_from_previous"])
        self.assertTrue(ms[4]["choice_flipped_from_baseline"])                         # still different from the baseline's A
        for rec in ms:
            self.assertEqual(rec["status"], "ok")
            self.assertAlmostEqual(sum(rec["probabilities"].values()), 1.0, places=9)
        # the terminal view
        for needle in ("BLEEDIFF TRACK", "BASELINE", "EDIT 01", "EDIT 04", "Margin A/B:", "Change from baseline:", "Change from previous:",
                       "Preferred answer: A (unchanged)", "Preferred answer: B (changed from A)", "Preferred answer: C (changed from B)",
                       "Status: A/B margin narrowed", "A/B order crossed: B now above A", "overall preferred answer unchanged: C",
                       "single prompt", "not the three-request fold"):
            self.assertIn(needle, text, needle)
        for banned in ("SAFE", "CRITICAL", "GUARANTEED"):
            self.assertNotIn(banned, text)

    def test_each_measurement_is_one_unchanged_bleediff_request(self):
        state = ScriptedState([("ALPHA", [0.45, 0.40, 0.10, 0.05])])
        code, _text, hist, _out, _ = run_session(state, [edit("ALPHA"), "quit"])
        self.assertEqual(code, 0)
        self.assertEqual(len(state.requests), 2)                                       # baseline + one edit: one /completion each
        body = state.requests[0]
        self.assertFalse(body["post_sampling_probs"])
        self.assertEqual(body["temperature"], 1.0)                                      # no temperature=0 collapse, no new constraint
        self.assertEqual(body["n_predict"], 1)
        self.assertEqual((body["top_k"], body["top_p"], body["min_p"], body["seed"]), (0, 1.0, 0.0, 0))
        self.assertEqual(state.requests[0].keys(), state.requests[1].keys())
        self.assertEqual(state.requests[0]["grammar"], state.requests[1]["grammar"])
        start = hist[0]
        self.assertEqual(start["measurement_type"], "single_prompt_pairwise_margin")
        self.assertIn("NOT the three-request fold", start["measurement_note"])

    def test_unchanged_fragment_is_detected_and_not_rescored(self):
        state = ScriptedState([("ALPHA", [0.45, 0.40, 0.10, 0.05])])
        code, text, hist, _out, _ = run_session(state, [edit("ALPHA"), "", "", "quit"])
        self.assertEqual(code, 0)
        self.assertEqual(len([r for r in hist if r["type"] == "measurement"]), 2)       # baseline + ONE edit; the two Enters did nothing
        self.assertEqual(len(state.requests), 2)
        self.assertEqual(text.count("is unchanged since step 2; not rescored"), 2)
        self.assertEqual([r["event"] for r in hist if r["type"] == "event"], ["fragment_unchanged_not_rescored"] * 2)

    def test_empty_fragment_is_not_a_measurement(self):
        state = ScriptedState()
        code, text, hist, _out, _ = run_session(state, ["", "quit"])
        self.assertEqual(code, 0)
        self.assertIn("fragment file is empty", text)
        self.assertEqual(len(state.requests), 1)

    def test_repeated_identical_prompt_measurements(self):
        state = ScriptedState([("ALPHA", [0.45, 0.40, 0.10, 0.05])])
        code, text, hist, _out, _ = run_session(state, [edit("ALPHA"), "again", "again", "quit"])
        self.assertEqual(code, 0)
        ms = [r for r in hist if r["type"] == "measurement"]
        self.assertEqual([m["label"] for m in ms], ["baseline", "edit", "repeat", "repeat"])
        self.assertEqual(ms[2]["identical_prompt_as_step"], 2)
        self.assertTrue(ms[2]["probabilities_identical_to_that_step"])
        self.assertEqual(ms[2]["change_from_previous"], 0.0)
        self.assertTrue(ms[2]["forced_repeat"])
        self.assertIn("already measured at step 2; the probabilities are identical", text)
        self.assertIn("A/B margin unchanged", text)

    def test_a_nondeterministic_server_is_flagged_for_an_identical_prompt(self):
        state = ScriptedState(sequence=[[0.7, 0.2, 0.05, 0.05], [0.45, 0.4, 0.1, 0.05], [0.5, 0.35, 0.1, 0.05]])
        code, text, hist, _out, _ = run_session(state, [edit("ALPHA"), "again", "quit"])
        ms = [r for r in hist if r["type"] == "measurement"]
        self.assertFalse(ms[2]["probabilities_identical_to_that_step"])
        self.assertIn("DIFFERENT probabilities", text)

    def test_editing_the_file_between_enters_changes_the_prompt(self):
        state = ScriptedState()
        run_session(state, [edit("first text"), edit("second text"), "quit"])
        prompts = [r["prompt"] for r in state.requests]
        self.assertEqual(len(prompts), 3)
        self.assertNotIn("first text", prompts[0])
        self.assertTrue(prompts[1].rstrip().endswith("Answer:") and "first text" in prompts[1])
        self.assertIn("second text", prompts[2])
        self.assertNotIn("first text", prompts[2])
        sc = scenario()
        self.assertEqual(prompts[0], T.build_prompt(sc, None))
        self.assertEqual(prompts[2], T.build_prompt(sc, "second text"))

    def test_history_records_provenance_hashes_and_no_text_by_default(self):
        state = ScriptedState()
        secret = "PRIVATE-FRAGMENT-TEXT-12345"
        code, text, hist, out_dir, _ = run_session(state, [edit(secret), "quit"])
        with open(os.path.join(out_dir, "history.jsonl"), encoding="utf-8") as fh:
            raw = fh.read()
        self.assertNotIn(secret, raw)                                                   # only hashes by default
        self.assertNotIn("Reply with exactly one letter", raw)                          # no prompt text either
        self.assertFalse(os.path.exists(os.path.join(out_dir, "prompts")))
        self.assertEqual([r["type"] for r in hist], ["session_start", "measurement", "measurement", "session_end"])
        start, base, e1, end = hist
        self.assertEqual(start["schema"], "bleediff.track-history/1")
        self.assertEqual(start["choices"], ["A", "B", "C", "D"])
        self.assertEqual(start["scenario"]["id"], "config-audit-v2")
        self.assertEqual(start["scenario"]["fixtures_sha256"], file_sha256(FIXTURES))
        self.assertEqual(start["model"]["name"], "mock-model")
        self.assertRegex(start["model"]["model_hash"], r"^[0-9a-f]{64}$")
        self.assertIn("inference", start)
        self.assertEqual(start["inference"]["n_probs"], 32)
        self.assertEqual(start["tokenization"]["dropped"], [])
        self.assertEqual(start["prompt_retention"], "off: only hashes are recorded")
        self.assertEqual(e1["fragment"]["sha256"], hashlib.sha256(secret.encode()).hexdigest())
        self.assertEqual((e1["fragment"]["bytes"], e1["fragment"]["words"]), (len(secret), 1))
        self.assertRegex(e1["prompt"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertNotEqual(e1["prompt"]["sha256"], base["prompt"]["sha256"])
        self.assertFalse(base["fragment"]["present"])
        self.assertEqual(end["reason"], "quit")
        for r in hist:
            self.assertIn("timestamp_utc", r)
        self.assertNotIn(":\\", raw)                                                     # no local absolute paths
        self.assertNotIn("/Users/", raw)

    def test_retain_prompts_is_explicit_and_local(self):
        state = ScriptedState()
        code, text, hist, out_dir, _ = run_session(state, [edit("keep me"), "quit"], extra=["--retain-prompts"])
        self.assertEqual(code, 0)
        self.assertEqual(hist[0]["prompt_retention"], "explicit local files (--retain-prompts)")
        with open(os.path.join(out_dir, "prompts", "step-02.txt"), encoding="utf-8") as fh:
            self.assertIn("keep me", fh.read())
        with open(os.path.join(out_dir, "prompts", "step-02.fragment.txt"), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "keep me")

    def test_sessions_and_user_files_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            out_dir = os.path.join(d, "s")
            state = ScriptedState()
            code, _t, _h, _o, _ = run_session(state, [edit("one"), "quit"], out=out_dir)
            self.assertEqual(code, 0)
            before = {f: file_sha256(os.path.join(out_dir, f)) for f in os.listdir(out_dir)}
            code2, text2, hist2, _o2, _ = run_session(ScriptedState(), ["quit"], out=out_dir)
            self.assertEqual(code2, 2)
            self.assertIn("already exists", text2)
            self.assertEqual({f: file_sha256(os.path.join(out_dir, f)) for f in os.listdir(out_dir)}, before)
            # a user-supplied fragment file is read, never replaced
            mine = os.path.join(d, "mine.txt")
            with open(mine, "w", encoding="utf-8") as fh:
                fh.write("my text")
            code3, _t3, _h3, out3, _ = run_session(ScriptedState(), ["", "quit"], out=os.path.join(d, "s2"), fragment=mine)
            self.assertEqual(code3, 0)
            with open(mine, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "my text")
            self.assertEqual(len([r for r in history(out3) if r["type"] == "measurement"]), 2)   # the existing text was scored on Enter

    def test_unreachable_server_is_a_clear_error_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            out_dir = os.path.join(d, "never")
            code, text = run_cli("track", "--server", "http://127.0.0.1:9", "--out", out_dir, "--fixtures", FIXTURES, "--timeout", "3")
            self.assertEqual(code, 2)
            self.assertIn("ERROR [SERVER_UNREACHABLE]", text)
            self.assertFalse(os.path.exists(out_dir))

    def test_baseline_failure_is_recorded_and_exits_2(self):
        state = ScriptedState(omit_probs=True)
        code, text, hist, _o, _ = run_session(state, ["quit"])
        self.assertEqual(code, 2)
        self.assertIn("ERROR [PROBABILITY_FIELDS_MISSING]", text)
        self.assertEqual([r["type"] for r in hist], ["session_start", "measurement", "session_end"])
        self.assertEqual(hist[1]["status"], "error")
        self.assertNotIn("probabilities", hist[1])                                       # nothing imputed
        self.assertEqual(hist[2]["reason"], "baseline_failed")

    def test_a_server_failure_mid_session_is_recorded_and_the_session_recovers(self):
        state = ScriptedState([("ALPHA", [0.45, 0.40, 0.10, 0.05]), ("BRAVO", [0.30, 0.40, 0.20, 0.10])])

        def break_server(path):
            edit("ALPHA")(path)
            state.omit_probs = True
            return ""

        def heal(path):
            state.omit_probs = False
            return ""

        code, text, hist, _o, _ = run_session(state, [break_server, heal, edit("BRAVO"), "quit"])
        self.assertEqual(code, 0)
        ms = [r for r in hist if r["type"] == "measurement"]
        self.assertEqual([m["status"] for m in ms], ["ok", "error", "ok", "ok"])
        self.assertIn("ERROR [PROBABILITY_FIELDS_MISSING]", text)
        self.assertNotIn("probabilities", ms[1])
        # the failed attempt did not become a measurement: the retry of ALPHA is compared with the baseline, then BRAVO with ALPHA
        self.assertEqual(ms[2]["edit"], 1)
        self.assertAlmostEqual(ms[2]["change_from_previous"], ms[2]["change_from_baseline"], places=12)
        self.assertAlmostEqual(ms[3]["change_from_previous"], math.log(0.30 / 0.40) - math.log(0.45 / 0.40), places=9)
        self.assertIn("EDIT 01", text)
        self.assertIn("EDIT 02", text)

    def test_an_unreadable_fragment_is_an_error_not_a_measurement(self):
        state = ScriptedState()

        def bad_bytes(path):
            with open(path, "wb") as fh:
                fh.write(b"\xff\xfe\xfa")
            return ""

        code, text, hist, _o, _ = run_session(state, [bad_bytes, edit("ok now"), "quit"])
        self.assertEqual(code, 0)
        self.assertIn("ERROR [FRAGMENT_UNREADABLE]", text)
        ms = [r for r in hist if r["type"] == "measurement"]
        self.assertEqual([m["status"] for m in ms], ["ok", "error", "ok"])
        self.assertEqual(len(state.requests), 2)                                         # the bad file never reached the model

    def test_unknown_commands_and_end_of_input(self):
        state = ScriptedState()
        code, text, hist, _o, _ = run_session(state, ["what", "quit"])
        self.assertIn("Unknown command", text)
        code2, text2, hist2, _o2, _ = run_session(ScriptedState(), [])                   # input() raises EOFError immediately
        self.assertEqual(code2, 0)
        self.assertEqual(hist2[-1]["reason"], "end_of_input")

    def test_unknown_scenario_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with serve_state(ScriptedState()) as url:
                code, text = run_cli("track", "--server", url, "--out", os.path.join(d, "x"), "--fixtures", FIXTURES, "--scenario", "nope")
            self.assertEqual(code, 2)
            self.assertIn("unknown scenario", text)
            self.assertFalse(os.path.exists(os.path.join(d, "x")))

    def test_end_to_end_mock_session_through_the_cli(self):
        """One complete simulated session: baseline, the three frozen evidence texts of the demonstration, quit."""
        arms = _json(os.path.join(DATA, "frozen-arms.json"))["arms"]
        # mock vectors loosely shaped like the historical 1.5B behaviour (SIMULATED, not model output)
        rules = [(arms["B_original"]["text"], [0.5987, 0.3812, 0.0050, 0.0151]),
                 (arms["P_paraphrase"]["text"], [0.6314, 0.3480, 0.0052, 0.0154]),
                 (arms["N_polarity_reversed"]["text"], [0.8013, 0.1804, 0.0049, 0.0134])]
        state = ScriptedState(rules, default=[0.7688, 0.2078, 0.0054, 0.0180])
        code, text, hist, out_dir, _ = run_session(state, [edit(arms["B_original"]["text"]), edit(arms["P_paraphrase"]["text"]),
                                                           edit(arms["N_polarity_reversed"]["text"]), "quit"])
        self.assertEqual(code, 0, text)
        ms = [r for r in hist if r["type"] == "measurement"]
        self.assertEqual(len(ms), 4)
        self.assertAlmostEqual(ms[0]["margin_AB"], 1.308, places=3)
        self.assertAlmostEqual(ms[1]["margin_AB"], 0.451, places=3)
        self.assertAlmostEqual(ms[1]["change_from_baseline"], -0.857, places=3)
        self.assertAlmostEqual(ms[2]["change_from_previous"], math.log(0.6314 / 0.3480) - math.log(0.5987 / 0.3812), places=3)
        self.assertEqual({m["preferred_choice"] for m in ms}, {"A"})
        self.assertEqual([m["choice_flipped_from_baseline"] for m in ms[1:]], [False, False, False])
        self.assertIn("Margin A/B: +1.308 nats", text)
        self.assertIn("Margin A/B: +0.451 nats", text)
        self.assertIn("Change from baseline: -0.857 nats", text)
        self.assertEqual(hist[-1]["type"], "session_end")


if __name__ == "__main__":
    unittest.main()
