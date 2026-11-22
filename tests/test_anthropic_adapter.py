"""Anthropic (Claude) ANSWER-ONLY adapter (bleediff.anthropic_messages on the shared bleediff.hosted engine), offline replay, the hosted-comparison pilot preparation, and the Atlas
separation of Claude records from direct-probability measurements. All HTTP is mocked: nothing contacts a network or needs a key. tests/data/anthropic-probe-haiku-5-5.json is a real
response (only the opaque thinking signature and the message id were replaced)."""

from __future__ import annotations

import copy
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from tests.test_bleediff import run_cli  # noqa: E402
from tests.test_gemini_adapter import (MAPS, FakeAPI, all_text, csv_rows, make_dataset, rb, read_json, rt, tmp, write_json)  # noqa: E402

from bleediff import answers as AN  # noqa: E402
from bleediff import anthropic_messages as CL  # noqa: E402
from bleediff import hosted as H  # noqa: E402

PROBE = read_json(os.path.join(ROOT, "tests", "data", "anthropic-probe-haiku-5-5.json"))["response"]
OBUNDLE = os.path.join(ROOT, "examples", "openai-answer-only")
GBUNDLE = os.path.join(ROOT, "examples", "gemini-answer-only")
CMP = os.path.join(ROOT, "examples", "hosted-comparison")
SECRET = "sk-ant-" + "api03-FAKEtestkey0123456789abcdefghijklmnopqrstu"          # shaped like a key, built at run time; not a credential


def claude(text, usage=(120, 1, None), model="claude-haiku-5-5", stop="end_turn", thinking_block=False):
    content = ([{"type": "thinking", "thinking": "", "signature": "sig"}] if thinking_block else []) + ([{"type": "text", "text": text}] if text is not None else [])
    r = {"id": "msg_test", "type": "message", "role": "assistant", "model": model, "content": content, "stop_reason": stop, "stop_sequence": None}
    if usage is not None:
        inp, out, thinking = usage
        u = {"input_tokens": inp, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": out}
        if thinking is not None:
            u["output_tokens_details"] = {"thinking_tokens": thinking}
        r["usage"] = u
    return r


def error_body(kind, message):
    return {"type": "error", "error": {"type": kind, "message": message}}


def correct_answerer(prompt, n):
    mp = prompt.split()[2]
    c, t = MAPS[mp]
    return 200, claude(t if "COMPETING" in prompt else c)


def run_live(handler=correct_answerer, *, model="claude-haiku-5-5", passes=1, budget=1.0, pin=1.0, pout=10.0, mappings=("m1", "m3"), extra=None, dataset=None):
    d = tmp()
    ds = dataset or make_dataset(os.path.join(d, "prompts.json"), mappings=mappings)
    api = FakeAPI(handler)
    out = os.path.join(d, "session")
    kw = dict(prompts_path=ds, model=model, out_dir=out, passes=passes, budget=budget, price_in_per_m=pin, price_out_per_m=pout, key=SECRET, post=api.post, sleep=api.sleep)
    kw.update(extra or {})
    return CL.run_session(**kw), api, out, ds


# ======================================================================================
class TestParsing(unittest.TestCase):
    def test_valid_answers_and_text_block_assembly(self):
        for l in "ABCD":
            p = AN.parse_anthropic_answer(claude(l))
            self.assertEqual((p["status"], p["answer"], p["finish_reason"]), ("ok", l, "end_turn"))
        self.assertEqual(AN.parse_anthropic_answer(claude("  C\n"))["answer"], "C")
        r = claude("B")
        r["content"].insert(0, {"type": "thinking", "thinking": "A", "signature": "s"})
        r["content"].append({"type": "redacted_thinking", "data": "x"})
        self.assertEqual(AN.parse_anthropic_answer(r)["answer"], "B")                           # thinking blocks never count as the answer
        self.assertEqual(AN.parse_anthropic_answer(claude("D", stop="max_tokens"))["answer"], "D")

    def test_the_real_haiku_probe_response_is_an_honest_empty_non_answer(self):
        p = AN.parse_anthropic_answer(PROBE)
        self.assertEqual((p["status"], p["answer"], p["finish_reason"]), ("empty", None, "max_tokens"))      # thinking ate the 8 tokens: no text block
        self.assertEqual(AN.detect_provider(PROBE), "anthropic")
        self.assertFalse(AN.has_logprobs(PROBE))

    def test_invalid_refusal_and_malformed(self):
        for text in ("a", "AB", "A.", "Answer: A", "E"):
            self.assertEqual(AN.parse_anthropic_answer(claude(text))["status"], "invalid", text)
        p = AN.parse_anthropic_answer(claude(None, stop="refusal"))
        self.assertEqual((p["status"], p["answer"], p["block_reason"]), ("blocked", None, "refusal"))
        self.assertEqual(AN.parse_anthropic_answer(claude("A", stop="refusal"))["status"], "blocked")
        self.assertEqual(AN.parse_anthropic_answer(claude(None))["status"], "empty")
        for bad in (None, "x", 5, [], {"content": "A"}, {"content": [5]}, {"type": "error", "error": {"type": "overloaded_error"}}):
            self.assertEqual(AN.parse_anthropic_answer(bad)["status"], "malformed", repr(bad))

    def test_usage_counts_thinking_tokens_once_and_maps_cache_conservatively(self):
        u = AN.usage_from_anthropic(PROBE)
        self.assertEqual(u, {"promptTokenCount": 228, "candidatesTokenCount": 0, "thoughtsTokenCount": 8, "totalTokenCount": 236})
        cost, basis, ok = H.billed_cost(u, 0.1e-6, 0.5e-6, 600, 1000)
        self.assertTrue(ok)
        self.assertAlmostEqual(cost, 228 * 0.1e-6 + 8 * 0.5e-6)                                 # 8 billed output tokens, not 16
        u = AN.usage_from_anthropic({"usage": {"input_tokens": 100, "cache_read_input_tokens": 50, "cache_creation_input_tokens": 10, "output_tokens": 7}})
        self.assertEqual((u["promptTokenCount"], u["candidatesTokenCount"], u["thoughtsTokenCount"], u["totalTokenCount"]), (170, 7, None, 177))
        self.assertEqual(AN.usage_from_anthropic(claude("A", usage=None)), {k: None for k in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount")})
        self.assertIsNone(AN.usage_from_anthropic({"usage": {"input_tokens": True, "output_tokens": -1}})["promptTokenCount"])


class TestRequestRules(unittest.TestCase):
    def test_minimal_body_has_no_sampling_seed_or_logprobs_parameters(self):
        b = CL.build_request("hi", "claude-haiku-5-5")
        self.assertEqual(sorted(b), ["max_tokens", "messages", "model"])
        self.assertEqual(b["max_tokens"], 1024)
        self.assertNotIn("logprobs", json.dumps(b).lower())
        self.assertNotIn("seed", b)

    def test_optional_parameters_and_their_rules(self):
        self.assertEqual(CL.build_request("hi", "m", effort="low")["output_config"], {"effort": "low"})
        self.assertEqual(CL.build_request("hi", "m", thinking="adaptive")["thinking"], {"type": "adaptive"})
        self.assertEqual(CL.build_request("hi", "m", thinking="disabled")["thinking"], {"type": "disabled"})
        self.assertEqual(CL.build_request("hi", "m", temperature=0.2)["temperature"], 0.2)
        self.assertEqual(CL.build_request("hi", "m", top_p=0.9)["top_p"], 0.9)
        self.assertEqual(CL.build_request("hi", "m", thinking_budget=2048, max_output_tokens=4096)["thinking"], {"type": "enabled", "budget_tokens": 2048})
        for kw in ({"temperature": 0.2, "top_p": 0.9}, {"effort": "extreme"}, {"thinking": "enabled"}, {"thinking": "adaptive", "thinking_budget": 2048},
                   {"thinking_budget": 512, "max_output_tokens": 4096}, {"thinking_budget": 2048, "max_output_tokens": 1024}, {"thinking_budget": 2048, "max_output_tokens": 4096, "temperature": 0.5}):
            with self.assertRaises(Exception, msg=str(kw)):
                CL.build_request("hi", "m", **kw)

    def test_unsupported_parameters_are_refused_before_any_request_or_file(self):
        d = tmp()
        ds = make_dataset(os.path.join(d, "p.json"))
        api = FakeAPI(correct_answerer)
        with self.assertRaises(Exception):
            CL.run_session(prompts_path=ds, model="claude-haiku-5-5", out_dir=os.path.join(d, "s"), passes=1, budget=1, price_in_per_m=1, price_out_per_m=1, key=SECRET, post=api.post,
                           temperature=0.3, top_p=0.5)
        self.assertEqual((api.calls, sorted(os.listdir(d))), (0, ["p.json"]))

    def test_missing_key_and_invalid_model(self):
        for env in ({}, {"ANTHROPIC_API_KEY": ""}, {"ANTHROPIC_API_KEY": "  "}, {"OPENAI_API_KEY": "x"}):
            with self.assertRaises(H.HostedError) as cm:
                CL.api_key(env)
            self.assertEqual(cm.exception.reason, "MISSING_API_KEY")
            self.assertIn("ANTHROPIC_API_KEY", str(cm.exception))
        for bad in ("", "a b", "../x", "claude?x", None):
            with self.assertRaises(Exception, msg=repr(bad)):
                CL.normalise_model(bad)

    def test_headers_carry_the_key_and_version_never_the_url(self):
        seen = {}

        class Resp:
            status = 200
            headers = {}

            def read(self, n=-1):
                return b"{}"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake(req, timeout=None):
            seen["url"], seen["h"] = req.full_url, {k.lower(): v for k, v in req.header_items()}
            return Resp()

        with mock.patch("urllib.request.urlopen", fake):
            H.http_post(CL.ANTHROPIC, CL.ANTHROPIC.endpoint("claude-haiku-5-5"), b"{}", SECRET, 5)
        self.assertEqual(seen["h"].get("x-api-key"), SECRET)
        self.assertEqual(seen["h"].get("anthropic-version"), "2023-06-01")
        self.assertNotIn(SECRET, seen["url"])
        self.assertEqual(seen["url"], "https://api.anthropic.com/v1/messages")

    def test_redaction_covers_anthropic_key_shapes(self):
        s = H.redact(f"bad key {SECRET} and x-api-key: abcdefghijklmnop and sk-ant-" + "api03-abcdefghijklmnopqrstuvwxyz", SECRET)
        for leak in (SECRET, "api03-abcdefghijklmnop"):
            self.assertNotIn(leak, s)
        self.assertNotIn(SECRET, str(H.HostedError(f"boom {SECRET}", key=SECRET)))


class TestLiveRunWithMockHttp(unittest.TestCase):
    def test_valid_run_records_labelled_claude_answer_only_evidence(self):
        res, api, out, ds = run_live(passes=2, extra={"effort": "low", "max_output_tokens": 256})
        self.assertEqual((res["status"], api.calls), ("complete", 12))
        s = read_json(os.path.join(out, "session.json"))
        self.assertEqual((s["measurement_type"], s["provider"], s["model"], s["logprobs_requested"]), ("ANSWER_ONLY", "anthropic", "claude-haiku-5-5", False))
        self.assertIn("no logprobs parameter", s["logprobs_note"])
        self.assertEqual(s["generation_config"], {"model": "claude-haiku-5-5", "max_tokens": 256, "output_config": {"effort": "low"}})
        for f in glob.glob(os.path.join(out, "records", "*.json")):
            r = read_json(f)
            self.assertEqual((r["schema"], r["measurement_type"], r["provider"], r["status"]), (AN.RECORD_SCHEMA, "ANSWER_ONLY", "anthropic", "ok"))
            self.assertNotIn("messages", r["request"]["parameters"])
            self.assertFalse(r["request"]["credentials_recorded"])
        self.assertTrue(all(b["max_tokens"] == 256 and b["output_config"] == {"effort": "low"} and "seed" not in b for b in api.bodies))
        self.assertNotIn(SECRET, all_text(out))
        ses = AN.read_session(out)
        self.assertEqual((ses["problems"], ses["warnings"]), ([], []))

    def test_saved_records_never_hold_thinking_signatures_or_reasoning_text(self):
        sig, reasoning, gsig = "SIGNATURE-" + "x" * 40, "private chain of thought", "GEMINI-SIG-" + "y" * 20

        def h(p, n):
            r = claude("A", usage=(120, 9, 8), thinking_block=True)
            r["content"][0].update({"thinking": reasoning, "signature": sig})
            r["content"].append({"type": "redacted_thinking", "data": sig + "-red"})
            r["candidates"] = [{"content": {"parts": [{"text": reasoning, "thought": True, "thoughtSignature": gsig}, {"text": "A", "thoughtSignature": gsig}]}}]
            return 200, r

        res, api, out, _ = run_live(h)
        text = all_text(out)
        for secret in (sig, reasoning, gsig):
            self.assertNotIn(secret, text)
        rec = read_json(sorted(glob.glob(os.path.join(out, "records", "*.json")))[0])
        self.assertEqual(rec["response"]["content"][0]["signature"], H.SIGNATURE_PLACEHOLDER)
        self.assertEqual(rec["response"]["content"][0]["thinking"], H.REASONING_PLACEHOLDER)
        self.assertEqual(rec["response"]["candidates"][0]["content"]["parts"][1]["text"], "A")      # visible answer text is kept
        self.assertEqual(H.scrub_private_reasoning(H.scrub_private_reasoning(rec["response"])), rec["response"])   # idempotent
        ses = AN.read_session(out)
        self.assertEqual(ses["problems"], [])
        self.assertTrue(all(r["status"] == "ok" and r["answer"] == "A" for r in ses["records"]))

    def test_http_errors_follow_the_anthropic_error_shapes(self):
        cases = ((401, "authentication_error", "AUTH_FAILED"), (403, "permission_error", "AUTH_FAILED"), (404, "not_found_error", "MODEL_NOT_AVAILABLE"),
                 (400, "invalid_request_error", "BAD_REQUEST"), (413, "request_too_large", "HTTP_ERROR"))
        for code, kind, reason in cases:
            res, api, out, _ = run_live(lambda p, n, c=code, k=kind: (c, error_body(k, f"nope {SECRET}")))
            self.assertEqual((res["status"], api.calls), ("aborted", 1), code)
            self.assertIn(reason, res["stop_reason"])
            self.assertNotIn(SECRET, all_text(out) + res["stop_reason"])

    def test_a_billing_failure_stops_at_once_without_retries(self):
        res, api, out, _ = run_live(lambda p, n: (400, error_body("invalid_request_error", "Your credit balance is too low to access the Anthropic API.")), extra={"max_retries": 5})
        self.assertEqual((res["status"], api.calls, api.sleeps), ("aborted", 1, []))
        self.assertIn("credit balance", res["stop_reason"])

    def test_overload_529_and_rate_limits_are_retried_within_bounds(self):
        self.assertIn(529, H.TRANSIENT_HTTP)
        res, api, out, _ = run_live(lambda p, n: (529, error_body("overloaded_error", "Overloaded")) if n <= 2 else correct_answerer(p, n), extra={"max_retries": 2})
        self.assertEqual((res["status"], api.sleeps[:2]), ("complete", [2, 4]))
        res, api, out, _ = run_live(lambda p, n: (429, error_body("rate_limit_error", "slow")), extra={"max_retries": 2})
        self.assertEqual((res["status"], api.calls), ("aborted", 3 * H.MAX_CONSECUTIVE_FAILURES))
        self.assertIn("RATE_LIMITED", res["stop_reason"])
        res, api, out, _ = run_live(lambda p, n: (200, H.TransportError("URLError")), extra={"max_retries": 1})
        self.assertEqual((res["status"], api.calls), ("aborted", 6))
        self.assertGreater(res["estimated_spend_usd"], 0)

    def test_thinking_only_replies_refusals_and_invalid_text_are_non_answers(self):
        def h(p, n):
            return 200, [PROBE, claude(None, stop="refusal"), claude(None), claude("maybe"), {"content": "oops", "usage": {"input_tokens": 5, "output_tokens": 1}}, claude("A")][(n - 1) % 6]

        res, api, out, _ = run_live(h)
        ses = AN.read_session(out)
        self.assertEqual(ses["problems"], [])
        self.assertEqual(sorted(r["status"] for r in ses["records"]), sorted(["empty", "blocked", "empty", "invalid", "malformed", "ok"]))
        a = AN.analyze(ses["records"], ses["prompts"])
        self.assertEqual((sum(t["non_answers"] for t in a["by_condition"].values()), sum(t["valid"] for t in a["by_condition"].values())), (5, 1))
        first = sorted(ses["records"], key=lambda r: r["name"])
        self.assertTrue(any(r["usage"].get("thoughtsTokenCount") == 8 for r in first))

    def test_semantic_roles_across_permutations(self):
        def h(p, n):
            c, t = MAPS[p.split()[2]]
            return 200, claude(t if "COMPETING" in p else c)

        res, api, out, _ = run_live(h, mappings=("m1", "m2", "m3", "m4"), passes=2)
        ses = AN.read_session(out)
        a = AN.analyze(ses["records"], ses["prompts"])
        self.assertEqual((a["by_condition"]["BASELINE"]["correct"], a["versus_baseline"]["COMPETING"]["induced_to_competing"]), (8, 8))
        self.assertEqual({r["answer"] for r in ses["records"] if r["condition"] == "BASELINE"}, {"A", "B", "C", "D"})
        self.assertEqual(a["pass_agreement"]["all"], {"prompts_with_two_or_more_valid_passes": 12, "prompts_all_passes_agree": 12})

    def test_cli_live_run_missing_key_and_no_secret_in_any_output(self):
        d = tmp()
        ds = make_dataset(os.path.join(d, "p.json"))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            code, text = run_cli("anthropic", "--prompts", ds, "--model", "claude-haiku-5-5", "--out", os.path.join(d, "s0"), "--budget", "1", "--price-in", "1", "--price-out", "1")
            self.assertEqual(code, 2)
            self.assertIn("ANTHROPIC_API_KEY", text)
            self.assertFalse(os.path.exists(os.path.join(d, "s0")))
        from bleediff.cli import build_parser
        api = FakeAPI(correct_answerer)
        out = io.StringIO()
        ns = build_parser().parse_args(["anthropic", "--prompts", ds, "--model", "claude-haiku-5-5", "--out", os.path.join(d, "s"), "--budget", "1.00", "--price-in", "0.1", "--price-out", "0.5", "--effort", "low"])
        self.assertEqual((ns.max_output_tokens, ns.temperature, ns.top_p, ns.effort), (1024, None, None, "low"))
        code = CL.cmd_anthropic(ns, out=out, post=api.post, sleep=api.sleep, env={"ANTHROPIC_API_KEY": SECRET})
        text = out.getvalue()
        self.assertEqual(code, 0, text)
        self.assertIn("ESTIMATE", text)
        self.assertNotIn(SECRET, text + all_text(d))
        self.assertTrue(all(b["output_config"] == {"effort": "low"} for b in api.bodies))


class TestBudget(unittest.TestCase):
    def test_reserve_is_max_tokens_and_thinking_tokens_are_billed_once(self):
        self.assertEqual(CL.ANTHROPIC.output_reserve(777, {"effort": "low"}), 777)
        res, api, out, _ = run_live(lambda p, n: (200, claude("A", usage=(100, 401, 400))), budget=1.0, pin=1.0, pout=10.0, extra={"max_output_tokens": 600})
        rec = read_json(sorted(glob.glob(os.path.join(out, "records", "*.json")))[0])
        self.assertAlmostEqual(rec["cost_usd"], 100 * 1e-6 + 401 * 1e-5)
        self.assertFalse(rec["cost_exceeded_preflight_bound"])

    def test_the_cap_is_enforced_with_the_max_tokens_reserve(self):
        res, api, out, _ = run_live(lambda p, n: (200, claude("A", usage=(90, 400, None))), budget=0.02, pin=1.0, pout=10.0, extra={"max_output_tokens": 1000})
        self.assertEqual((res["status"], api.calls), ("partial", 3))
        self.assertIn("BUDGET_EXHAUSTED", res["stop_reason"])
        self.assertLessEqual(res["estimated_spend_usd"], 0.02)
        res, api, out, _ = run_live(budget=0.0001, extra={"max_output_tokens": 1000})
        self.assertEqual((api.calls, res["status"]), (0, "partial"))

    def test_output_beyond_max_tokens_stops_the_run(self):
        res, api, out, _ = run_live(lambda p, n: (200, claude("A", usage=(50, 5000, None))), budget=5.0, pin=1.0, pout=10.0, extra={"max_output_tokens": 100})
        self.assertEqual((res["status"], api.calls), ("aborted", 1))
        self.assertIn("COST_BOUND_EXCEEDED", res["stop_reason"])
        self.assertTrue(read_json(glob.glob(os.path.join(out, "records", "*.json"))[0])["cost_exceeded_preflight_bound"])

    def test_missing_usage_fails_closed_and_is_flagged(self):
        res, api, out, _ = run_live(lambda p, n: (200, claude("A", usage=None)), budget=1.0)
        recs = AN.read_session(out)["records"]
        self.assertTrue(all(not r["usage_complete"] and r["cost_usd"] > 0 and r["cost_basis"].startswith("upper_bound") for r in recs))
        self.assertTrue(any("incomplete usage" in l for l in AN.report_lines(AN.analyze(recs, AN.read_session(out)["prompts"]))))


class TestReplayAndAtlasSeparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """A mock Claude session over the REAL frozen 80 prompts (answering the correct letter except under COMPETING), then an Atlas with local + Gemini + OpenAI + Claude."""
        cls.d = tmp()
        ds = AN.load_prompts(os.path.join(OBUNDLE, "prompts.json"))
        by_prompt = {i["prompt"]: i for i in ds["items"]}

        def h(prompt, n):
            it = by_prompt[prompt]
            return 200, claude(it["competing_letter"] if it["condition"] == "COMPETING" else it["correct_letter"], usage=(228, 1, None), model="claude-sonnet-5-5-test")

        api = FakeAPI(h)
        cls.session = os.path.join(cls.d, "claude")
        CL.run_session(prompts_path=os.path.join(OBUNDLE, "prompts.json"), model="claude-sonnet-5-5", out_dir=cls.session, passes=2, budget=2.0, price_in_per_m=2.0, price_out_per_m=10.0, key=SECRET,
                       post=api.post, sleep=api.sleep, max_output_tokens=64)
        cls.root = os.path.join(cls.d, "x")
        os.makedirs(cls.root)
        for name in ("examples/openai-answer-only", "examples/gemini-answer-only", "examples/four-mapping"):
            shutil.copytree(os.path.join(ROOT, name), os.path.join(cls.root, os.path.basename(name)))
        shutil.copytree(cls.session, os.path.join(cls.root, "openai-answer-only", "claude"))
        man = read_json(os.path.join(cls.root, "openai-answer-only", "atlas-input.json"))
        man["answer_only_sessions"].append({"id": "claude-sonnet-5-5/chat", "session_dir": "claude", "model": {"id": "claude-sonnet-5-5", "name": "Claude Sonnet 5.5"}})
        cls.manifest = os.path.join(cls.root, "openai-answer-only", "atlas-input-claude.json")
        write_json(cls.manifest, man)
        cls.out = os.path.join(cls.d, "atlas")
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network access attempted")):
            cls.code, cls.text = run_cli("atlas", "--manifest", cls.manifest, "--out", cls.out)
        cls.atlas = read_json(os.path.join(cls.out, "atlas.json"))

    def test_the_claude_session_replays_offline_through_every_hosted_command(self):
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch("urllib.request.urlopen", side_effect=AssertionError("network access attempted")):
            for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
                os.environ.pop(k, None)
            for cmd in ("anthropic", "openai", "gemini"):
                code, text = run_cli(cmd, "--replay", self.session)
                self.assertEqual(code, 0, text)
                self.assertIn("no key used, no network", text)
                self.assertIn("anthropic model claude-sonnet-5-5", text)
        ses = AN.read_session(self.session)
        a = AN.analyze(ses["records"], ses["prompts"])
        self.assertEqual((a["records"], a["by_condition"]["BASELINE"]["correct"], a["by_condition"]["COMPETING"]["competing"]), (160, 32, 32))

    def test_tampered_claude_evidence_is_detected(self):
        for tamper in ("answer", "provider", "logprobs", "measurement_type"):
            dst = os.path.join(self.d, "t-" + tamper)
            shutil.copytree(self.session, dst)
            f = sorted(glob.glob(os.path.join(dst, "records", "*.json")))[0]
            r = read_json(f)
            if tamper == "answer":
                r["answer"] = "B" if r["answer"] != "B" else "C"
            elif tamper == "provider":
                r["provider"] = "mystery"
            elif tamper == "logprobs":
                r["response"]["content"][0]["logprobs"] = 1
                r["probabilities"] = {"A": 1.0}
            elif tamper == "measurement_type":
                r["measurement_type"] = "DIRECT_PROBABILITY"
            write_json(f, r)
            code, text = run_cli("anthropic", "--replay", dst)
            self.assertEqual(code, 2, tamper)

    def test_four_kinds_of_evidence_stay_separate_in_the_atlas(self):
        self.assertEqual(self.code, 0, self.text)
        ao = self.atlas["answer_only"]
        self.assertEqual(sorted((s["provider"], s["model"]["id"]) for s in ao["sessions"]),
                         [("anthropic", "claude-sonnet-5-5"), ("gemini", "gemini-3.8-flash"), ("gemini", "gemini-4-carbon"), ("openai", "gpt-4.1-mini")])
        self.assertEqual({s["measurement_type"] for s in ao["sessions"]}, {"ANSWER_ONLY"})
        self.assertEqual({c["measurement_type"] for c in self.atlas["cells"]}, {"DIRECT_PROBABILITY"})
        self.assertEqual({c["model_id"] for c in self.atlas["cells"]}, {"qwen1.5b", "llama3b"})
        cand = csv_rows(os.path.join(self.out, "candidates.csv"))
        self.assertFalse({r["model_id"] for r in cand} & {"claude-sonnet-5-5", "gpt-4.1-mini", "gemini-3.8-flash"})
        for fn in ("model-comparison.csv", "category-summary.csv"):
            self.assertNotIn("claude", rt(os.path.join(self.out, fn)))
        rows = {(r["measurement_type"], r["source"], r["label"]): r for r in csv_rows(os.path.join(self.out, "answer-level-comparison.csv"))}
        r = rows[("ANSWER_ONLY", "claude-sonnet-5-5", "COMPETING")]
        self.assertEqual((r["n"], r["competing"], r["correct"]), ("32", "32", "0"))
        self.assertIn("returned answer", r["answer_basis"])
        report = rt(os.path.join(self.out, "report.md"))
        self.assertIn("provider anthropic", report)
        self.assertLess(report.index("## Task/model cells"), report.index("## Hosted answer-only results"))

    def test_no_probabilities_margins_or_ranks_exist_for_claude(self):
        text = json.dumps(self.atlas["answer_only"]).lower()
        for banned in ('"shift', '"margin', '"probabilit', '"rank', '"p_a"', '"coverage"', "log_odds", "logit", '"confidence', "leaderboard\":"):
            self.assertNotIn(banned, text, banned)
        for r in csv_rows(os.path.join(self.out, "answer-only.csv")):
            self.assertEqual(r["measurement_type"], "ANSWER_ONLY")

    def test_local_probability_results_are_unchanged_by_adding_claude(self):
        plain = read_json(self.manifest)
        del plain["answer_only_sessions"]
        write_json(os.path.join(self.root, "openai-answer-only", "plain.json"), plain)
        code, text = run_cli("atlas", "--manifest", os.path.join(self.root, "openai-answer-only", "plain.json"), "--out", os.path.join(self.d, "plain"))
        self.assertEqual(code, 0, text)
        base = read_json(os.path.join(self.d, "plain", "atlas.json"))
        strip = lambda cells: [{k: v for k, v in c.items() if k != "measurement_type"} for c in cells]
        self.assertEqual(strip(self.atlas["cells"]), strip(base["cells"]))
        for fn in ("candidates.csv", "category-summary.csv", "model-comparison.csv"):
            self.assertEqual(rb(os.path.join(self.out, fn)), rb(os.path.join(self.d, "plain", fn)), fn)


class TestComparativePilotPreparation(unittest.TestCase):
    def test_template_manifest_lists_the_five_hosted_models_plus_the_local_sessions(self):
        m = read_json(os.path.join(CMP, "atlas-input.json"))
        self.assertEqual(m["schema"], "bleediff.atlas-input/1")
        ids = [s["model"]["id"] for s in m["answer_only_sessions"]]
        self.assertEqual(ids, ["claude-haiku-5-5", "claude-sonnet-5-5", "gemini-3.8-flash", "gemini-4-carbon", "gpt-4.1-mini"])
        self.assertEqual({s["model"]["id"] for s in m["sessions"]}, {"qwen1.5b", "llama3b"})
        for s in m["answer_only_sessions"]:                                            # all five hosted sessions are saved in the repository
            self.assertTrue(os.path.isdir(os.path.normpath(os.path.join(CMP, s["session_dir"]))), s["session_dir"])

    def test_runbook_documents_commands_for_both_claude_models_and_the_estimate_caveat(self):
        t = " ".join(rt(os.path.join(CMP, "README.md")).split())
        for needle in ("claude-haiku-5-5", "claude-sonnet-5-5", "--budget", "--price-in 0.10", "--price-in 2.00", "--effort low", "estimate_pilot_cost.py", "estimate", "not provider billing totals",
                       "gemini-3.8-flash", "gemini-4-carbon", "gpt-4.1-mini", "ANSWER_ONLY", "DIRECT_PROBABILITY", "adaptive thinking"):
            self.assertIn(needle, t, needle)

    def test_cost_estimator_runs_offline_and_reports_the_worst_case(self):
        p = subprocess.run([sys.executable, "-B", os.path.join(CMP, "estimate_pilot_cost.py"), "--prompts", os.path.join(GBUNDLE, "prompts.json"), "--passes", "2", "--probe-chars", "560",
                            "--probe-input-tokens", "228", "--max-output-tokens", "1024", "--price", "claude-haiku-5-5", "0.10", "0.50", "--price", "claude-sonnet-5-5", "2.00", "10.00"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("160 requests", p.stdout)
        self.assertIn("claude-sonnet-5-5", p.stdout)
        self.assertIn("worst case", p.stdout)
        self.assertIn("ESTIMATES", rt(os.path.join(CMP, "estimate_pilot_cost.py")))


class TestHygiene(unittest.TestCase):
    def test_new_files_hold_no_secrets_or_local_paths(self):
        import re
        pat = re.compile(r"(sk-ant-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{10,}|/Users/|AppData|@[A-Za-z0-9-]+\.(com|org|net|io)|api[_-]?key\s*[:=])", re.I)
        files = [os.path.join(ROOT, "bleediff", "anthropic_messages.py"), os.path.join(ROOT, "tests", "data", "anthropic-probe-haiku-5-5.json")] + [f for f in glob.glob(os.path.join(CMP, "*")) if os.path.isfile(f)]
        for p in files:
            self.assertIsNone(pat.search(rt(p)), p)
            if not p.endswith(".py"):               # data and docs are protected from line-ending conversion (.gitattributes); source files follow the checkout's autocrlf
                self.assertNotIn(b"\r\n", rb(p), p)
        self.assertNotIn("Ashley", "".join(rt(p) for p in files))


if __name__ == "__main__":
    unittest.main()
