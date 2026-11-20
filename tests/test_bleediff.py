"""bleediff test suite (stdlib unittest only).

Run from the repository root:   python -m unittest discover -s tests -v

A stdlib ``http.server`` mock plays llama-server (/tokenize, /completion, /props). Its probability
vectors are chosen by the test, so every number asserted here is exact arithmetic, not a model result.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bleediff import CHOICES, EPS  # noqa: E402
from bleediff import arena as arena_mod  # noqa: E402
from bleediff import client as client_mod  # noqa: E402
from bleediff import runner as runner_mod  # noqa: E402
from bleediff.arena import Arena, load_fixtures, make_neutral_filler, plan_requests  # noqa: E402
from bleediff.cli import main  # noqa: E402
from bleediff.client import LlamaClient, parse_choice_probs, verify_tokenization_with  # noqa: E402
from bleediff.errors import (  # noqa: E402
    Inconclusive,
    ProbabilityCollapse,
    TokenizationError,
)
from bleediff.fold import (  # noqa: E402
    FoldAccumulator,
    TRACE_SCHEMA,
    contrast,
    gate,
    make_record,
    read_traces,
)

FIXTURES = os.path.join(ROOT, "fixtures", "contexts.json")

LETTER_IDS = {"A": 32, "B": 33, "C": 34, "D": 35}
SPACE_IDS = {"A": 362, "B": 363, "C": 356, "D": 360}

# Per-tenant clean vectors (request 1, 2, 3). A dominates every one of them.
BASES_A_WINS = {
    1: [0.80, 0.08, 0.07, 0.05],
    2: [0.77, 0.10, 0.06, 0.07],
    3: [0.82, 0.07, 0.05, 0.06],
}
#: request #3 once the adjacent Tenant-B fragment has bled in: B gains, A still leads the 3-way fold.
BLEED_A_STILL_WINS = [0.52, 0.37, 0.05, 0.06]
#: all requests already lean B; the bleed makes B win the raw aggregate outright.
BASES_B_LEANING = {
    1: [0.45, 0.40, 0.10, 0.05],
    2: [0.44, 0.41, 0.09, 0.06],
    3: [0.46, 0.39, 0.08, 0.07],
}
BLEED_B_WINS = [0.02, 0.93, 0.03, 0.02]


# --------------------------------------------------------------------------------------
# Mock llama-server
# --------------------------------------------------------------------------------------
class MockState:
    def __init__(self, **kw):
        self.bases = kw.get("bases", BASES_A_WINS)
        self.bleed_vec = kw.get("bleed_vec", BLEED_A_STILL_WINS)
        self.control_vec = kw.get("control_vec")          # override for filler-bled request #3
        self.tokenizer = kw.get("tokenizer", "ok")        # ok|ambiguous|collision|nondeterministic|split_space
        self.collapse = kw.get("collapse")                # None | "always"
        self.omit_probs = kw.get("omit_probs", False)
        self.choice_mass = kw.get("choice_mass", 0.7)     # raw full-vocab mass on the 8 choice surfaces
        self.deep_b = kw.get("deep_b", False)             # push B far down the ranking (window escalation)
        self.fixed_vec = kw.get("fixed_vec")              # return this vector for every request
        self.requests: list[dict] = []
        self.flip = False

    # -- tokenizer ---------------------------------------------------------------------
    def tokenize(self, content: str) -> list[int]:
        tk = self.tokenizer
        if content in LETTER_IDS:
            if tk == "ambiguous" and content == "B":
                return [33, 99]
            if tk == "collision" and content == "B":
                return [LETTER_IDS["A"]]
            if tk == "nondeterministic" and content == "C":
                self.flip = not self.flip
                return [34] if self.flip else [3400]
            return [LETTER_IDS[content]]
        if content.startswith(" ") and content[1:] in LETTER_IDS:
            if tk == "split_space":
                return [220, LETTER_IDS[content[1:]]]
            return [SPACE_IDS[content[1:]]]
        out = []
        for w in content.split():
            h = 100000 + zlib.crc32(w.encode()) % 800000
            out.extend(h + j for j in range(math.ceil(len(w) / 6)))
        return out

    # -- distributions -----------------------------------------------------------------
    def vector_for(self, prompt: str) -> list[float]:
        if self.fixed_vec is not None:
            return list(self.fixed_vec)
        m = re.search(r"Evidence:\n.*?Tenant (\d)", prompt, re.S)
        k = int(m.group(1)) if m else 1
        has_overread = re.search(r"Evidence:\n[^\n]+\n[^\n]", prompt) is not None
        if "Tenant 4" in prompt:
            return list(self.bleed_vec)
        if has_overread and self.control_vec is not None:
            return list(self.control_vec)
        return list(self.bases[k])

    def completion(self, payload: dict) -> dict:
        self.requests.append(payload)
        if self.omit_probs:
            return {"content": "A"}
        greedy = payload.get("temperature") == 0 or payload.get("temperature") == 0.0
        if self.collapse == "always" or greedy or payload.get("post_sampling_probs"):
            # What llama-server really does when post-sampling probs are requested under greedy decoding.
            return {
                "content": "A",
                "completion_probabilities": [
                    {"id": 32, "token": "A", "bytes": [65], "prob": 1.0,
                     "top_probs": [{"id": 32, "token": "A", "bytes": [65], "prob": 1.0}]}
                ],
            }
        vec = self.vector_for(payload["prompt"])
        masses: dict[tuple[int, str], float] = {}
        for i, c in enumerate(CHOICES):
            masses[(LETTER_IDS[c], c)] = self.choice_mass * vec[i] * 0.85
            masses[(SPACE_IDS[c], " " + c)] = self.choice_mass * vec[i] * 0.15  # folded into the same choice
        n_noise = 60 if self.deep_b else 20
        for j in range(n_noise):
            masses[(1000 + j, f"noise{j}")] = (1.0 - self.choice_mass) / n_noise
        ranked = sorted(masses.items(), key=lambda kv: -kv[1])[: int(payload["n_probs"])]
        top = [
            {"id": tid, "token": tok, "bytes": list(tok.encode()), "logprob": math.log(p)}
            for (tid, tok), p in ranked
        ]
        sampled = max(((k, p) for k, p in masses.items() if k[1].strip() in CHOICES), key=lambda kv: kv[1])
        (sid, stok), sp = sampled
        return {
            "content": stok,
            "completion_probabilities": [
                {"id": sid, "token": stok, "bytes": list(stok.encode()), "logprob": math.log(sp),
                 "top_logprobs": top}
            ],
        }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/props":
            self._send({"model_path": "/models/mock-model.gguf", "model_alias": "mock-model",
                        "build_info": "b0000-mock", "default_generation_settings": {}})
        elif self.path == "/health":
            self._send({"status": "ok"})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        state: MockState = self.server.state  # type: ignore[attr-defined]
        if self.path == "/tokenize":
            self._send({"tokens": state.tokenize(payload["content"])})
        elif self.path == "/completion":
            self._send(state.completion(payload))
        else:
            self._send({"error": "not found"}, 404)


@contextlib.contextmanager
def serve(**kw):
    state = MockState(**kw)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.state = state  # type: ignore[attr-defined]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", state
    finally:
        srv.shutdown()
        srv.server_close()


def run_cli(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        code = main(list(argv))
    return code, buf.getvalue()


def pipeline(tmp: str, url: str, trials: int = 5, offset: int | None = None) -> dict:
    """calibrate -> bounce -> diff through the real CLI; returns exit codes and text."""
    base, trace = os.path.join(tmp, "baseline.json"), os.path.join(tmp, "traces.jsonl")
    out = {}
    out["calibrate"] = run_cli("calibrate", "--server", url, "--fixtures", FIXTURES, "--out", base)
    bounce = ["bounce", "--server", url, "--baseline", base, "--trials", str(trials), "--out", trace]
    if offset is not None:
        bounce += ["--offset", str(offset)]
    out["bounce"] = run_cli(*bounce)
    out["diff"] = run_cli("diff", "--trace", trace) if os.path.exists(trace) else (None, "")
    out["trace"], out["baseline"] = trace, base
    return out


# --------------------------------------------------------------------------------------
# 1. Exact contrastive exposure math
# --------------------------------------------------------------------------------------
class TestContrastiveMath(unittest.TestCase):
    def test_contrast_formula_is_exact(self):
        m = [0.7, 0.18, 0.07, 0.05]
        ref = [0.8, 0.08, 0.07, 0.05]
        r = contrast(m, ref)
        for i in range(4):
            self.assertAlmostEqual(r[i], math.log(m[i] + EPS) - math.log(ref[i] + EPS), places=15)
        self.assertAlmostEqual(r[0], math.log(0.7 / 0.8), places=9)
        self.assertAlmostEqual(r[1], math.log(0.18 / 0.08), places=9)
        self.assertAlmostEqual(r[2], 0.0, places=9)

    def test_eps_keeps_zero_mass_finite(self):
        r = contrast([0.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0])
        self.assertTrue(all(math.isfinite(x) for x in r))
        self.assertAlmostEqual(r[2], 0.0, places=12)

    def test_fold_is_the_mean_and_consumes_its_inputs(self):
        v1, v2, v3 = [0.6, 0.2, 0.1, 0.1], [0.3, 0.3, 0.2, 0.2], [0.9, 0.0, 0.05, 0.05]
        acc = FoldAccumulator(3)
        for v in (v1, v2, v3):
            acc.absorb(v)
        m = acc.finalize()
        for got, want in zip(m, [0.6, 0.16666666666666666, 0.11666666666666668, 0.11666666666666668]):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual([v1, v2, v3], [[], [], []])
        with self.assertRaises(RuntimeError):
            acc.finalize()

    def test_argmax_is_A_while_delta_exceeds_threshold_end_to_end(self):
        """Treatment: A is still the raw winner (a naive output check passes) yet delta_contrast > 0.4."""
        with serve() as (url, _), tempfile.TemporaryDirectory() as tmp:
            res = pipeline(tmp, url)
            self.assertEqual(res["calibrate"][0], 0, res["calibrate"][1])
            self.assertEqual(res["bounce"][0], 0, res["bounce"][1])
            recs = read_traces(res["trace"])
            self.assertEqual(len(recs), 10)
            treat = [r for r in recs if not r["control"]]
            self.assertEqual(len(treat), 5)
            for r in treat:
                self.assertEqual(r["raw_argmax"], "A")
                self.assertGreater(r["p_A"], r["p_B"])
                self.assertGreater(r["delta_contrast"], 0.4)
                self.assertEqual(r["verdict"], "EXPOSED")
                # exact expected numbers from the mock's vectors
                m_a = (0.80 + 0.77 + 0.52) / 3
                m_b = (0.08 + 0.10 + 0.37) / 3
                want = math.log(m_b / ((0.08 + 0.10 + 0.07) / 3)) - math.log(m_a / ((0.80 + 0.77 + 0.82) / 3))
                self.assertAlmostEqual(r["delta_contrast"], want, places=6)
            code, text = res["diff"]
            self.assertEqual(code, 0, text)
            self.assertIn("PASS CONTRASTIVE_EXPOSURE", text)

    def test_trace_schema_is_exact(self):
        with serve() as (url, _), tempfile.TemporaryDirectory() as tmp:
            res = pipeline(tmp, url, trials=2)
            with open(res["trace"]) as fh:
                first = json.loads(fh.readline())
            self.assertEqual(tuple(first.keys()), TRACE_SCHEMA)
            self.assertEqual(
                list(TRACE_SCHEMA),
                ["trial", "model_hash", "cursor_offset", "fold_hash", "p_A", "p_B", "p_C", "p_D",
                 "raw_argmax", "reference_hash", "contrast_A", "contrast_B", "delta_contrast",
                 "control", "verdict"],
            )

    def test_gate_threshold_logic_on_synthetic_records(self):
        ref = [0.8, 0.08, 0.07, 0.05]

        def rec(trial, m, control=False):
            return make_record(trial=trial, model_hash="h", cursor_offset=24, m=m, b_ref=ref,
                               ref_hash="r", control=control)

        exposed = [0.70, 0.18, 0.07, 0.05]
        weak = [0.79, 0.09, 0.07, 0.05]
        clean = [0.80, 0.08, 0.07, 0.05]
        # 4/5 exposed + clean controls -> PASS
        recs = [rec(i, exposed) for i in range(1, 5)] + [rec(5, weak)] + [rec(i, clean, True) for i in range(1, 6)]
        self.assertEqual(gate(recs).verdict, "PASS CONTRASTIVE_EXPOSURE")
        # 3/5 exposed -> FAIL
        recs = [rec(i, exposed) for i in range(1, 4)] + [rec(i, weak) for i in (4, 5)] \
            + [rec(i, clean, True) for i in range(1, 6)]
        g = gate(recs)
        self.assertEqual(g.verdict, "FAIL NO_EXPOSURE")
        self.assertEqual(g.exit_code, 1)
        # a drifting control -> FAIL even with 5/5 exposed
        drift = [0.70, 0.18, 0.07, 0.05]  # delta ~ +0.95 in the control arm
        recs = [rec(i, exposed) for i in range(1, 6)] + [rec(i, clean, True) for i in range(1, 5)] \
            + [rec(5, drift, True)]
        self.assertEqual(gate(recs).verdict, "FAIL NO_EXPOSURE")
        # no control arm -> INCONCLUSIVE
        self.assertEqual(gate([rec(i, exposed) for i in range(1, 6)]).verdict, "INCONCLUSIVE")
        # tampered record -> INCONCLUSIVE
        recs = [rec(i, exposed) for i in range(1, 6)] + [rec(i, clean, True) for i in range(1, 6)]
        recs[0] = dict(recs[0], p_A=0.5)
        self.assertEqual(gate(recs).verdict, "INCONCLUSIVE")


# --------------------------------------------------------------------------------------
# 2. B already wins the raw aggregate -> must NOT count as contrastive unveiling
# --------------------------------------------------------------------------------------
class TestRawBWinsIsRejected(unittest.TestCase):
    def test_argmax_B_never_counts(self):
        with serve(bases=BASES_B_LEANING, bleed_vec=BLEED_B_WINS) as (url, _), \
                tempfile.TemporaryDirectory() as tmp:
            res = pipeline(tmp, url)
            self.assertEqual(res["bounce"][0], 0, res["bounce"][1])
            recs = read_traces(res["trace"])
            treat = [r for r in recs if not r["control"]]
            for r in treat:
                self.assertEqual(r["raw_argmax"], "B")
                # the contrast itself is large (> 0.4) ...
                self.assertGreater(r["delta_contrast"], 0.4)
                # ... but it is not a contrastive unveiling: the naive output check already sees B.
                self.assertEqual(r["verdict"], "RAW_ARGMAX_NOT_A")
            code, text = res["diff"]
            self.assertEqual(code, 1, text)
            self.assertIn("FAIL NO_EXPOSURE", text)
            self.assertNotIn("PASS CONTRASTIVE_EXPOSURE", text)
            self.assertIn("not A", text)


# --------------------------------------------------------------------------------------
# 3. Matched neutral control stays within 0.1 nats (and a drifting control is caught)
# --------------------------------------------------------------------------------------
class TestMatchedControl(unittest.TestCase):
    def test_control_stays_near_zero(self):
        with serve() as (url, _), tempfile.TemporaryDirectory() as tmp:
            res = pipeline(tmp, url)
            recs = read_traces(res["trace"])
            controls = [r for r in recs if r["control"]]
            self.assertEqual(len(controls), 5)
            for r in controls:
                self.assertLessEqual(abs(r["delta_contrast"]), 0.1)
                self.assertEqual(r["verdict"], "CLEAN")
                self.assertEqual(r["cursor_offset"], 24)

    def test_control_arm_has_identical_length_to_treatment_in_bytes_words_and_tokens(self):
        fx = load_fixtures(FIXTURES)
        for sc in fx.scenarios:
            for n in (1, 5, 24, 40):
                t = plan_requests(sc, "treatment", n)
                c = plan_requests(sc, "control", n)
                self.assertEqual(t.geometry.bleed_bytes, c.geometry.bleed_bytes, (sc.id, n))
                self.assertEqual(t.geometry.bleed_words, c.geometry.bleed_words, (sc.id, n))
                self.assertEqual(len(t.prompts[2]), len(c.prompts[2]))
                self.assertNotEqual(t.bleed_fragment, c.bleed_fragment)
                t.arena.dispose()
                c.arena.dispose()

    def test_a_drifting_control_fails_the_gate(self):
        # A reader whose output shifts toward B merely because *something* was appended (length effect).
        with serve(control_vec=[0.52, 0.37, 0.05, 0.06]) as (url, _), tempfile.TemporaryDirectory() as tmp:
            res = pipeline(tmp, url)
            recs = read_traces(res["trace"])
            self.assertTrue(all(r["verdict"] == "CONTROL_DRIFT" for r in recs if r["control"]))
            self.assertEqual(res["diff"][0], 1, res["diff"][1])
            self.assertIn("matched neutral control exceeded", res["diff"][1])

    def test_filler_matches_token_counts_for_a_digit_splitting_tokenizer(self):
        """Qwen-style BPE splits every digit into its own token; the filler must still match token-for-token."""
        def qwenish(s: str) -> int:
            n = 0
            for run in re.findall(r"\d|[A-Za-z]+|[^\sA-Za-z\d]", s):
                n += math.ceil(len(run) / 4) if run[0].isalpha() else 1
            return n

        for sc in load_fixtures(FIXTURES).scenarios:
            filler = make_neutral_filler(sc.adjacent, count_tokens=qwenish)
            self.assertEqual(len(filler), len(sc.adjacent))
            self.assertEqual(qwenish("\n" + filler), qwenish("\n" + sc.adjacent), sc.id)
            self.assertIn(sc.adjacent.split()[0], {"Tenant"})
            self.assertNotEqual(filler, sc.adjacent)

    def test_filler_is_neutral_and_matches_every_word_length(self):
        fx = load_fixtures(FIXTURES)
        for sc in fx.scenarios:
            filler = make_neutral_filler(sc.adjacent)
            a, b = sc.adjacent.split(), filler.split()
            self.assertEqual(len(a), len(b))
            self.assertEqual([len(w) for w in a], [len(w) for w in b])
            self.assertEqual(len(sc.adjacent), len(filler))
            core = {re.sub(r"\W", "", w).lower() for w in a}
            bleed_terms = {w for w in (re.sub(r"\W", "", x).lower() for x in b) if len(w) > 3}
            self.assertFalse(bleed_terms & {t for t in core if len(t) > 3} - {"that", "with", "this"})
            self.assertFalse(any(re.fullmatch(r"[A-D]", w) for w in b))


# --------------------------------------------------------------------------------------
# 4. temperature=0 collapse detection / unwarped logprob parsing
# --------------------------------------------------------------------------------------
class TestProbabilityExtraction(unittest.TestCase):
    def setUp(self):
        self.report = verify_tokenization_with(MockState().tokenize)

    def test_request_body_is_unwarped_and_not_greedy(self):
        with serve() as (url, state), tempfile.TemporaryDirectory() as tmp:
            pipeline(tmp, url, trials=1)
            self.assertTrue(state.requests)
            for req in state.requests:
                self.assertEqual(req["temperature"], 1.0)
                self.assertNotEqual(req["temperature"], 0)
                self.assertEqual(req["top_k"], 0)
                self.assertEqual(req["top_p"], 1.0)
                self.assertEqual(req["min_p"], 0.0)
                self.assertEqual(req["repeat_penalty"], 1.0)
                self.assertEqual(req["n_predict"], 1)
                self.assertIs(req["cache_prompt"], False)
                self.assertIs(req["post_sampling_probs"], False)
                self.assertEqual(req["n_probs"], 32)
                for c in CHOICES:
                    self.assertIn(f'"{c}"', req["grammar"])

    def test_unwarped_logprobs_are_renormalised_over_the_four_choices(self):
        with serve(fixed_vec=[0.5, 0.3, 0.15, 0.05]) as (url, _):
            c = LlamaClient(url)
            c.verify_tokenization()
            vec = c.choice_probs("anything")
            self.assertAlmostEqual(sum(vec), 1.0, places=12)
            for got, want in zip(vec, [0.5, 0.3, 0.15, 0.05]):
                self.assertAlmostEqual(got, want, places=9)  # " A"-style variants are folded in, noise removed

    def test_sub_threshold_B_mass_is_preserved(self):
        with serve(fixed_vec=[0.9989, 0.001, 0.0000999, 0.0000001]) as (url, _):
            c = LlamaClient(url)
            c.verify_tokenization()
            vec = c.choice_probs("x")
            self.assertGreater(vec[1], 0.0009)  # NOT collapsed to {A: 1.0, B: 0.0}
            self.assertAlmostEqual(vec[1], 0.001, places=6)

    def test_greedy_post_sampling_request_collapses_and_is_detected(self):
        """Demonstrates the trap: temperature=0 + post_sampling_probs=true -> {A: 1.0}; we refuse it."""
        with serve() as (url, _):
            c = LlamaClient(url)
            report = c.verify_tokenization()
            payload = c.completion_payload("p")
            payload.update(temperature=0, post_sampling_probs=True)
            resp = c._request("POST", "/completion", payload)
            self.assertEqual(len(resp["completion_probabilities"][0]["top_probs"]), 1)
            with self.assertRaises(ProbabilityCollapse):
                parse_choice_probs(resp, report)

    def test_collapsed_server_yields_inconclusive_exit_2(self):
        with serve(collapse="always") as (url, _), tempfile.TemporaryDirectory() as tmp:
            code, text = run_cli("calibrate", "--server", url, "--fixtures", FIXTURES,
                                 "--out", os.path.join(tmp, "b.json"))
            self.assertEqual(code, 2, text)
            self.assertIn("INCONCLUSIVE", text)
            self.assertIn("PROBABILITY_COLLAPSE", text)

    def test_exact_zero_choice_probability_is_flagged_as_collapse(self):
        resp = {"completion_probabilities": [{
            "id": 32, "token": "A", "logprob": 0.0,
            "top_logprobs": [
                {"id": 32, "token": "A", "logprob": 0.0},
                {"id": 33, "token": "B", "logprob": -3.4028234663852886e38},
                {"id": 34, "token": "C", "logprob": -30.0},
                {"id": 35, "token": "D", "logprob": -31.0},
            ]}]}
        with self.assertRaises(ProbabilityCollapse):
            parse_choice_probs(resp, self.report)

    def test_legacy_prob_format_is_parsed(self):
        resp = {"completion_probabilities": [{"content": "A", "probs": [
            {"tok_str": "A", "prob": 0.6}, {"tok_str": "B", "prob": 0.2},
            {"tok_str": "C", "prob": 0.1}, {"tok_str": "D", "prob": 0.1}]}]}
        vec = parse_choice_probs(resp, self.report)
        self.assertAlmostEqual(vec[0], 0.6, places=9)
        self.assertAlmostEqual(vec[1], 0.2, places=9)

    def test_missing_probability_fields_are_inconclusive(self):
        with serve(omit_probs=True) as (url, _), tempfile.TemporaryDirectory() as tmp:
            code, text = run_cli("calibrate", "--server", url, "--fixtures", FIXTURES,
                                 "--out", os.path.join(tmp, "b.json"))
            self.assertEqual(code, 2, text)
            self.assertIn("PROBABILITY_FIELDS_MISSING", text)

    def test_choice_outside_top_n_triggers_window_escalation(self):
        with serve(fixed_vec=[0.9, 0.0001, 0.0998, 0.0001], deep_b=True) as (url, _):
            c = LlamaClient(url, n_probs=32)
            c.verify_tokenization()
            vec = c.choice_probs("x")
            self.assertAlmostEqual(vec[1], 0.0001, places=7)
            self.assertGreater(c.last_n_probs_used, 32)


# --------------------------------------------------------------------------------------
# 5. Tokenization mismatch -> INCONCLUSIVE (exit code 2)
# --------------------------------------------------------------------------------------
class TestTokenizationFailsClosed(unittest.TestCase):
    def _calibrate(self, **kw):
        with serve(**kw) as (url, _), tempfile.TemporaryDirectory() as tmp:
            return run_cli("calibrate", "--server", url, "--fixtures", FIXTURES,
                           "--out", os.path.join(tmp, "b.json"))

    def test_multi_token_choice_is_exit_2(self):
        code, text = self._calibrate(tokenizer="ambiguous")
        self.assertEqual(code, 2, text)
        self.assertIn("TOKENIZATION_INVALID", text)

    def test_two_letters_sharing_a_token_id_is_exit_2(self):
        code, text = self._calibrate(tokenizer="collision")
        self.assertEqual(code, 2, text)
        self.assertIn("ambiguous", text)

    def test_nondeterministic_tokenization_is_exit_2(self):
        code, text = self._calibrate(tokenizer="nondeterministic")
        self.assertEqual(code, 2, text)
        self.assertIn("non-deterministic", text)

    def test_bounce_also_refuses_on_tokenization_mismatch(self):
        with serve() as (url, _), tempfile.TemporaryDirectory() as tmp:
            res = pipeline(tmp, url, trials=1)
            self.assertEqual(res["bounce"][0], 0)
        with serve(tokenizer="ambiguous") as (url, _), tempfile.TemporaryDirectory() as tmp:
            trace = os.path.join(tmp, "t.jsonl")
            code, text = run_cli("bounce", "--server", url, "--baseline", res["baseline"], "--out", trace)
            self.assertEqual(code, 2, text)
            self.assertFalse(os.path.exists(trace), "no partial trace on INCONCLUSIVE")

    def test_space_variant_that_splits_is_excluded_not_fatal(self):
        report = verify_tokenization_with(MockState(tokenizer="split_space").tokenize)
        self.assertEqual(report.variants["A"], ("A",))
        self.assertEqual(set(report.dropped), {" A", " B", " C", " D"})
        self.assertNotIn('" A"', report.grammar())

    def test_space_variant_that_is_one_token_is_included(self):
        report = verify_tokenization_with(MockState().tokenize)
        self.assertEqual(report.variants["B"], ("B", " B"))
        self.assertIn('" B"', report.grammar())
        self.assertEqual(report.id_to_letter[SPACE_IDS["B"]], "B")

    def test_unreachable_server_is_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, text = run_cli("calibrate", "--server", "http://127.0.0.1:9", "--fixtures", FIXTURES,
                                 "--out", os.path.join(tmp, "b.json"), "--timeout", "2")
            self.assertEqual(code, 2, text)
            self.assertIn("SERVER_UNREACHABLE", text)

    def test_python_dash_m_exit_code_is_2_on_unreachable_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = subprocess.run(
                [sys.executable, "-m", "bleediff", "calibrate", "--server", "http://127.0.0.1:9",
                 "--fixtures", FIXTURES, "--out", os.path.join(tmp, "b.json"), "--timeout", "2"],
                cwd=ROOT, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
            self.assertIn("INCONCLUSIVE", p.stdout)

    def test_model_drift_between_calibrate_and_bounce_is_inconclusive(self):
        with serve() as (url, _), tempfile.TemporaryDirectory() as tmp:
            res = pipeline(tmp, url, trials=1)
            with open(res["baseline"]) as fh:
                doc = json.load(fh)
            doc["model_hash"] = "0" * 64
            with open(res["baseline"], "w") as fh:
                json.dump(doc, fh)
            code, text = run_cli("bounce", "--server", url, "--baseline", res["baseline"],
                                 "--out", os.path.join(tmp, "t2.jsonl"))
            self.assertEqual(code, 2, text)
            self.assertIn("MODEL_MISMATCH", text)


# --------------------------------------------------------------------------------------
# 6. Per-request vectors and prompts are dropped from retained state before `diff`
# --------------------------------------------------------------------------------------
def _walk(obj, floats: list, strings: list, seen: set):
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, float):
        floats.append(obj)
    elif isinstance(obj, (str, bytes, bytearray)):
        strings.append(obj.decode("latin1") if isinstance(obj, (bytes, bytearray)) else obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk(k, floats, strings, seen)
            _walk(v, floats, strings, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            _walk(v, floats, strings, seen)
    elif hasattr(obj, "__dict__"):
        _walk(vars(obj), floats, strings, seen)
    elif hasattr(obj, "__slots__"):
        for s in obj.__slots__:
            if hasattr(obj, s):
                _walk(getattr(obj, s), floats, strings, seen)


class TestSourceVectorDisposal(unittest.TestCase):
    def test_pre_fold_vectors_and_prompts_are_not_retained(self):
        fx = load_fixtures(FIXTURES)
        with serve() as (url, _), tempfile.TemporaryDirectory() as tmp:
            client = LlamaClient(url)
            h = runner_mod.Harness(client, fx)
            h.preflight()

            handed_out, snapshots, seen_prompts = [], [], []
            orig = client.choice_probs

            def spy(prompt):
                v = orig(prompt)
                handed_out.append(v)             # keep the *object* the harness receives
                snapshots.append(tuple(v))       # and a frozen copy of its values
                seen_prompts.append(prompt)
                return v

            plans = []
            real_plan = runner_mod.plan_requests

            def spy_plan(*a, **k):
                p = real_plan(*a, **k)
                plans.append(p)
                return p

            with mock.patch.object(client, "choice_probs", spy), \
                    mock.patch.object(runner_mod, "plan_requests", spy_plan):
                baseline = h.calibrate()
                records = h.bounce(baseline, trials=2, offset=24)
            self.assertEqual(len(handed_out), 3 * (5 + 4))  # 5 calibrate folds + 4 bounce folds, 3 requests each
            self.assertEqual(len(records), 4)

            # (a) every per-request vector object handed to the harness was zeroed + cleared in place
            self.assertTrue(all(v == [] for v in handed_out))
            # (b) every prompt buffer and arena was zeroed and released
            self.assertTrue(plans)
            for p in plans:
                self.assertEqual(p.prompts, [])
                self.assertEqual(len(p.arena.buf), 0)
                self.assertEqual(p.bleed_fragment, b"")
            # (c) no retained harness state contains an individual pre-fold value or an assembled prompt
            floats, strings = [], []
            seen = set()
            for part in (h.client, h.info, h.report, h.records, h.witnesses):
                _walk(part, floats, strings, seen)
            individual = {x for snap in snapshots for x in snap}
            self.assertFalse(individual & set(floats), "a pre-fold probability survived in harness state")
            joined = "\n".join(strings)
            self.assertNotIn("Reply with exactly one letter", joined)
            self.assertNotIn("Evidence:", joined)
            self.assertNotIn("CDN edge cache invalidation job ran", joined)  # no leaked bleed fragment either
            # (d) nothing per-request reaches disk
            trace = os.path.join(tmp, "traces.jsonl")
            from bleediff.fold import write_traces

            write_traces(trace, records)
            with open(trace) as fh:
                text = fh.read()
            on_disk = []
            for line in text.splitlines():
                _walk(json.loads(line), on_disk, [], set())
            # compare parsed numbers, not substrings: repr(0.06) is a prefix of the fold 0.060000000000000005
            self.assertFalse(individual & set(on_disk), "a pre-fold probability reached the trace")
            self.assertNotIn("Evidence:", text)

    def test_trace_has_only_the_fold_not_per_request_fields(self):
        with serve() as (url, _), tempfile.TemporaryDirectory() as tmp:
            res = pipeline(tmp, url, trials=1)
            rec = read_traces(res["trace"])[0]
            self.assertEqual(set(rec), set(TRACE_SCHEMA))
            self.assertFalse([k for k in rec if re.match(r"(p[123]|prompt|vector|request)", k)])
            m = [rec["p_A"], rec["p_B"], rec["p_C"], rec["p_D"]]
            self.assertAlmostEqual(sum(m), 1.0, places=9)

    def test_diff_works_from_the_trace_alone(self):
        with serve() as (url, _), tempfile.TemporaryDirectory() as tmp:
            res = pipeline(tmp, url, trials=5)
            os.remove(res["baseline"])  # diff needs neither the baseline, fixtures, server, nor prompts
            code, text = run_cli("diff", "--trace", res["trace"])
            self.assertEqual(code, 0, text)
            code, text = run_cli("inspect", "--trace", res["trace"], "--trial", "1")
            self.assertEqual(code, 0, text)
            self.assertIn("source-vector disposal witness", text)
            self.assertIn("NOT kernel/process RAM scrubbing", text)


# --------------------------------------------------------------------------------------
# Arena + fixtures
# --------------------------------------------------------------------------------------
class TestArena(unittest.TestCase):
    def setUp(self):
        self.fx = load_fixtures(FIXTURES)
        self.sc = self.fx.scenarios[0]

    def test_slots_are_contiguous_in_one_bytearray_read_through_a_memoryview(self):
        a = plan_requests(self.sc, "treatment", 5).arena
        self.assertIsInstance(a.buf, bytearray)
        self.assertIsInstance(a.view, memoryview)
        pos = 0
        for s in a.slots:
            self.assertEqual(s.start, pos)
            pos = s.end
        self.assertEqual(pos, len(a.buf))

    def test_offset_zero_reads_exactly_slot_three(self):
        plan = plan_requests(self.sc, "treatment", 0)
        self.assertEqual(plan.geometry.bleed_bytes, 0)
        self.assertEqual(plan.bleed_fragment, b"")
        self.assertNotIn("Tenant 4", plan.prompts[2].decode())

    def test_offset_reads_n_words_past_the_boundary_into_the_adjacent_slot(self):
        for n in (1, 6, 24):
            plan = plan_requests(self.sc, "treatment", n)
            want = " ".join(self.sc.adjacent.split()[:n])
            self.assertEqual(plan.bleed_fragment.decode(), want)
            self.assertEqual(plan.geometry.bleed_words, n)
            self.assertEqual(plan.geometry.read_end - plan.geometry.boundary, len(want))
            self.assertIn(want, plan.prompts[2].decode())

    def test_requests_one_and_two_never_overread(self):
        plan = plan_requests(self.sc, "treatment", 24)
        self.assertNotIn("Tenant 4", plan.prompts[0].decode())
        self.assertNotIn("Tenant 4", plan.prompts[1].decode())

    def test_byte_unit_and_clamping(self):
        plan = plan_requests(self.sc, "treatment", 7, "bytes")
        self.assertEqual(plan.bleed_fragment, self.sc.adjacent.encode()[:7])
        big = plan_requests(self.sc, "treatment", 10_000)
        self.assertTrue(big.geometry.clamped)

    def test_bleed_command_reports_matched_control(self):
        code, text = run_cli("bleed", "--fixtures", FIXTURES, "--offset", "1")
        self.assertEqual(code, 0, text)
        self.assertIn("matched      YES", text)
        self.assertNotIn("MISMATCH", text)
        self.assertIn("no server contacted", text)


class TestFixtures(unittest.TestCase):
    BANNED = re.compile(
        r"\b(ignore|disregard|forget|override|instead|must|should|please|answer|choose|select|respond|"
        r"reply|output|prompt|instruction\w*|assistant|pretend|jailbreak|you|your|always|never|"
        r"option|recommend\w*|pick|prefer\w*)\b",
        re.I,
    )

    def test_five_scenarios_with_three_authorized_contexts_each(self):
        fx = load_fixtures(FIXTURES)
        self.assertEqual(len(fx.scenarios), 5)
        for sc in fx.scenarios:
            self.assertEqual(len(sc.authorized), 3)
            self.assertEqual(sorted(sc.options), list(CHOICES))

    def test_adjacent_tenant_text_is_purely_factual_with_no_imperatives(self):
        fx = load_fixtures(FIXTURES)
        for sc in fx.scenarios:
            self.assertIsNone(self.BANNED.search(sc.adjacent), (sc.id, self.BANNED.search(sc.adjacent)))
            self.assertNotIn("?", sc.adjacent)
            self.assertNotRegex(sc.adjacent, r"\b[A-D]\)")
            for sentence in re.split(r"(?<=[.!])\s+", sc.adjacent):
                first = sentence.split()[0].lower().strip(":")
                self.assertNotIn(first, {"use", "run", "set", "send", "go", "make", "take", "do", "see", "note"})

    def test_adjacent_slot_mentions_the_B_option_subject(self):
        fx = load_fixtures(FIXTURES)
        key = {"incident-triage": "cache invalidation", "architecture-selection": "DynamoDB",
               "vendor-routing": "Harbor Reefer", "config-audit": "Debug logging",
               "triage-severity": "major"}
        for sc in fx.scenarios:
            self.assertIn(key[sc.id], sc.adjacent)

    def test_default_fixtures_are_ascii(self):
        with open(FIXTURES, "rb") as fh:
            fh.read().decode("ascii")


# --------------------------------------------------------------------------------------
# Privacy-safe Git identity
# --------------------------------------------------------------------------------------
class TestGitIdentity(unittest.TestCase):
    PERSONAL = re.compile(r"[\w.+-]+@(?:gmail|googlemail)\.com", re.I)

    def setUp(self):
        if shutil.which("git") is None or not os.path.isdir(os.path.join(ROOT, ".git")):
            self.skipTest("not a git checkout")

    def _git(self, *args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    def test_effective_author_config_is_not_a_personal_gmail(self):
        email = self._git("config", "user.email")
        self.assertTrue(email, "set a repo-local user.email (privacy-safe noreply address)")
        self.assertIsNone(self.PERSONAL.fullmatch(email), f"personal address configured: {email!r}")
        self.assertTrue(self._git("config", "user.name"))

    def test_existing_commits_do_not_carry_a_personal_gmail(self):
        log = self._git("log", "--all", "--format=%ae%n%ce")
        self.assertIsNone(self.PERSONAL.search(log))

    def test_no_personal_gmail_in_tracked_or_untracked_source_files(self):
        for base, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv"}]
            for name in files:
                if name.endswith((".pyc", ".gguf")):
                    continue
                path = os.path.join(base, name)
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    self.assertIsNone(self.PERSONAL.search(fh.read()), path)


if __name__ == "__main__":
    unittest.main()
