"""OpenAI ANSWER-ONLY adapter (bleediff.openai_chat on the shared bleediff.hosted engine), offline replay of the saved GPT-4.1 mini pilot, the shared budget rules of both hosted
adapters, and the three-source Atlas (local DIRECT_PROBABILITY + Gemini + OpenAI ANSWER_ONLY). All HTTP is mocked: nothing contacts a network or needs a key."""

from __future__ import annotations

import copy
import glob
import io
import json
import math
import os
import shutil
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from tests.test_bleediff import run_cli  # noqa: E402
from tests.test_gemini_adapter import (MAPS, FakeAPI, all_text, csv_rows, make_dataset, rb, read_json, rt, tmp, write_json)  # noqa: E402

from bleediff import answers as AN  # noqa: E402
from bleediff import gemini as GM  # noqa: E402
from bleediff import hosted as H  # noqa: E402
from bleediff import openai_chat as OA  # noqa: E402

BUNDLE = os.path.join(ROOT, "examples", "openai-answer-only")
GBUNDLE = os.path.join(ROOT, "examples", "gemini-answer-only")
SECRET = "sk-" + "proj-FAKEtestkey0123456789abcdefghijklmnop"          # shaped like a key, built at run time; not a credential


def oai(text, usage=(120, 1, None), model="gpt-4.1-mini-2025-04-14", finish="stop", refusal=None):
    r = {"id": "chatcmpl-test", "object": "chat.completion", "model": model,
         "choices": [{"index": 0, "message": {"role": "assistant", "content": text, "refusal": refusal}, "finish_reason": finish}]}
    if usage is not None:
        pt, ct, reasoning = usage
        u = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
        if reasoning is not None:
            u["completion_tokens_details"] = {"reasoning_tokens": reasoning}
        r["usage"] = u
    return r


def correct_answerer(prompt, n):
    mp = prompt.split()[2]
    c, t = MAPS[mp]
    return 200, oai(t if "COMPETING" in prompt else c)


def run_live(handler=correct_answerer, *, model="gpt-4.1-mini", passes=1, budget=1.0, pin=1.0, pout=10.0, mappings=("m1", "m3"), extra=None):
    d = tmp()
    ds = make_dataset(os.path.join(d, "prompts.json"), mappings=mappings)
    api = FakeAPI(handler)
    out = os.path.join(d, "session")
    kw = dict(prompts_path=ds, model=model, out_dir=out, passes=passes, budget=budget, price_in_per_m=pin, price_out_per_m=pout, key=SECRET, post=api.post, sleep=api.sleep)
    kw.update(extra or {})
    return OA.run_session(**kw), api, out, ds


# ======================================================================================
class TestParsing(unittest.TestCase):
    def test_valid_answers(self):
        for l in "ABCD":
            p = AN.parse_openai_answer(oai(l))
            self.assertEqual((p["status"], p["answer"]), ("ok", l))
        self.assertEqual(AN.parse_openai_answer(oai(" C\n"))["answer"], "C")
        self.assertEqual(AN.parse_openai_answer(oai("A", finish="length"))["answer"], "A")              # max_completion_tokens=1 ends with finish_reason length

    def test_not_exactly_one_letter_is_invalid(self):
        for text in ("a", "AB", "A.", "Answer: A", "E", "I think B"):
            self.assertEqual(AN.parse_openai_answer(oai(text))["status"], "invalid", text)

    def test_refusal_and_content_filter_are_blocked(self):
        p = AN.parse_openai_answer(oai(None, refusal="I can't help with that."))
        self.assertEqual((p["status"], p["answer"], p["block_reason"]), ("blocked", None, "refusal"))
        p = AN.parse_openai_answer(oai("A", finish="content_filter"))
        self.assertEqual((p["status"], p["answer"]), ("blocked", None))

    def test_empty_missing_and_malformed(self):
        self.assertEqual(AN.parse_openai_answer(oai(""))["status"], "empty")
        self.assertEqual(AN.parse_openai_answer(oai(None))["status"], "empty")
        self.assertEqual(AN.parse_openai_answer({"choices": []})["status"], "empty")
        for bad in (None, "x", 5, {"choices": "x"}, {"choices": [5]}, {"choices": [{"message": "A"}]}, {"choices": [{"message": {"content": 7}}]}, {"error": {"message": "x"}}):
            self.assertEqual(AN.parse_openai_answer(bad)["status"], "malformed" if bad != {"error": {"message": "x"}} else "malformed", repr(bad))

    def test_logprobs_in_a_response_are_ignored_by_the_parser(self):
        r = oai("B")
        r["choices"][0]["logprobs"] = {"content": [{"token": "A", "logprob": 0.0, "top_logprobs": [{"token": "A", "logprob": 0.0}]}]}
        self.assertEqual(AN.parse_openai_answer(r)["answer"], "B")                   # the returned message decides, never the logprobs
        self.assertTrue(AN.has_logprobs(r))

    def test_usage_mapping_counts_reasoning_tokens_once(self):
        u = AN.usage_from_openai(oai("A", usage=(100, 401, 400)))
        self.assertEqual(u, {"promptTokenCount": 100, "candidatesTokenCount": 1, "thoughtsTokenCount": 400, "totalTokenCount": 501})
        cost, basis, ok = H.billed_cost(u, 1e-6, 1e-5, 500, 600)
        self.assertTrue(ok)
        self.assertAlmostEqual(cost, 100 * 1e-6 + 401 * 1e-5)                         # 401 billed completion tokens, not 801
        self.assertEqual(AN.usage_from_openai(oai("A", usage=None)), {k: None for k in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount")})
        self.assertIsNone(AN.usage_from_openai({"usage": {"prompt_tokens": True, "completion_tokens": -3}})["promptTokenCount"])


class TestRequestRules(unittest.TestCase):
    def test_gpt4_family_gets_sampling_parameters_and_never_logprobs(self):
        b = OA.build_request("hi", "gpt-4.1-mini", seed=3, max_output_tokens=9)
        self.assertEqual({k: b[k] for k in ("temperature", "top_p", "seed", "max_completion_tokens", "n")}, {"temperature": 1.0, "top_p": 1.0, "seed": 3, "max_completion_tokens": 9, "n": 1})
        self.assertNotIn("logprobs", json.dumps(b).lower())
        self.assertNotIn("max_tokens", b)
        self.assertEqual(OA.build_request("hi", "gpt-4o", temperature=0.2)["temperature"], 0.2)

    def test_reasoning_family_gets_no_sampling_parameters(self):
        for m in ("gpt-5", "gpt-5.4-mini", "gpt-6.2-sol", "o3", "o4-mini"):
            self.assertEqual(OA.model_family(m), "reasoning", m)
            b = OA.build_request("hi", m, reasoning_effort="low")
            self.assertNotIn("temperature", b)
            self.assertNotIn("top_p", b)
            self.assertEqual(b["reasoning_effort"], "low")
            with self.assertRaises(Exception, msg=m):
                OA.build_request("hi", m, temperature=0.5)
            with self.assertRaises(Exception, msg=m):
                OA.build_request("hi", m, top_p=0.5)

    def test_reasoning_effort_is_refused_where_unsupported_or_invalid(self):
        with self.assertRaises(Exception):
            OA.build_request("hi", "gpt-4.1-mini", reasoning_effort="low")
        with self.assertRaises(Exception):
            OA.build_request("hi", "gpt-5", reasoning_effort="extreme")

    def test_unknown_families_get_provider_defaults_only(self):
        self.assertEqual(OA.model_family("some-new-model"), "other")
        b = OA.build_request("hi", "some-new-model")
        self.assertEqual(sorted(b), ["max_completion_tokens", "messages", "model", "n", "seed"])
        with self.assertRaises(Exception):
            OA.build_request("hi", "some-new-model", temperature=1.0)

    def test_unsupported_parameters_are_refused_before_any_request_or_file(self):
        d = tmp()
        ds = make_dataset(os.path.join(d, "p.json"))
        api = FakeAPI(correct_answerer)
        with self.assertRaises(Exception):
            OA.run_session(prompts_path=ds, model="gpt-5", out_dir=os.path.join(d, "s"), passes=1, budget=1, price_in_per_m=1, price_out_per_m=1, key=SECRET, post=api.post, temperature=0.3)
        self.assertEqual((api.calls, sorted(os.listdir(d))), (0, ["p.json"]))

    def test_missing_key_and_invalid_model(self):
        for env in ({}, {"OPENAI_API_KEY": ""}, {"OPENAI_API_KEY": "  "}, {"GEMINI_API_KEY": "x"}):
            with self.assertRaises(H.HostedError) as cm:
                OA.api_key(env)
            self.assertEqual(cm.exception.reason, "MISSING_API_KEY")
            self.assertIn("OPENAI_API_KEY", str(cm.exception))
        for bad in ("", "a b", "../x", "gpt?x", None):
            with self.assertRaises(Exception, msg=repr(bad)):
                OA.normalise_model(bad)

    def test_bearer_header_is_used_and_never_the_url_or_the_error_text(self):
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
            seen["url"], seen["h"] = req.full_url, dict(req.header_items())
            return Resp()

        with mock.patch("urllib.request.urlopen", fake):
            H.http_post(OA.OPENAI, OA.OPENAI.endpoint("gpt-4.1-mini"), b"{}", SECRET, 5)
        self.assertEqual(seen["h"].get("Authorization"), "Bearer " + SECRET)
        self.assertNotIn(SECRET, seen["url"])
        self.assertTrue(seen["url"].endswith("/v1/chat/completions"))

    def test_secret_redaction_covers_openai_key_shapes(self):
        s = H.redact(f"bad key {SECRET}; sk-" + "abcdefghijklmnopqrstuvwx" + " Authorization: Bearer abc.def.ghijklmnopqr and Bearer zzzzzzzzzzzzzzzz", SECRET)
        for leak in (SECRET, "sk-" + "abcdefghijklmnop", "abc.def.ghi", "zzzzzzzzzzzz"):
            self.assertNotIn(leak, s)
        self.assertNotIn(SECRET, str(H.HostedError(f"boom {SECRET}", key=SECRET)))


class TestLiveRunWithMockHttp(unittest.TestCase):
    def test_valid_run_records_labelled_openai_answer_only_evidence(self):
        res, api, out, ds = run_live(passes=2)
        self.assertEqual((res["status"], api.calls), ("complete", 12))
        s = read_json(os.path.join(out, "session.json"))
        self.assertEqual((s["measurement_type"], s["provider"], s["model"], s["logprobs_requested"]), ("ANSWER_ONLY", "openai", "gpt-4.1-mini", False))
        self.assertIn("ESTIMATE", s["budget"]["kind"])
        self.assertEqual(rb(os.path.join(out, "prompts.json")), rb(ds))
        for f in glob.glob(os.path.join(out, "records", "*.json")):
            r = read_json(f)
            self.assertEqual((r["schema"], r["measurement_type"], r["provider"], r["status"]), (AN.RECORD_SCHEMA, "ANSWER_ONLY", "openai", "ok"))
            self.assertNotIn("messages", r["request"]["parameters"])
            self.assertNotIn("logprobs", json.dumps(r["request"]).lower())
            self.assertFalse(r["request"]["credentials_recorded"])
            self.assertEqual(r["model_version"], "gpt-4.1-mini-2025-04-14")
        self.assertTrue(all(b["model"] == "gpt-4.1-mini" and "logprobs" not in json.dumps(b) for b in api.bodies))
        self.assertTrue(all(SECRET == k for k in api.keys))
        self.assertNotIn(SECRET, all_text(out))
        ses = AN.read_session(out)
        self.assertEqual((ses["problems"], ses["warnings"]), ([], []))

    def test_http_errors(self):
        for code, reason, calls in ((401, "AUTH_FAILED", 1), (403, "AUTH_FAILED", 1), (400, "BAD_REQUEST", 1), (404, "MODEL_NOT_AVAILABLE", 1)):
            res, api, out, _ = run_live(lambda p, n, c=code: (c, {"error": {"message": f"nope {SECRET}", "type": "invalid_request_error"}}))
            self.assertEqual((res["status"], api.calls), ("aborted", calls), code)
            self.assertIn(reason, res["stop_reason"])
            self.assertNotIn(SECRET, all_text(out) + res["stop_reason"])

    def test_rate_limits_are_bounded_and_quota_exhaustion_is_not_retried(self):
        res, api, out, _ = run_live(lambda p, n: (429, {"error": {"message": "slow down", "type": "requests", "code": "rate_limit_exceeded"}}), extra={"max_retries": 2})
        self.assertEqual((res["status"], api.calls), ("aborted", 3 * H.MAX_CONSECUTIVE_FAILURES))
        self.assertIn("RATE_LIMITED", res["stop_reason"])
        res, api, out, _ = run_live(lambda p, n: (429, {"error": {"message": "You exceeded your current quota", "type": "insufficient_quota", "code": "insufficient_quota"}}), extra={"max_retries": 5})
        self.assertEqual((res["status"], api.calls, api.sleeps), ("aborted", 1, []))
        self.assertIn("QUOTA_EXHAUSTED", res["stop_reason"])

    def test_transient_errors_recover_and_exhausted_ones_are_charged_conservatively(self):
        res, api, out, _ = run_live(lambda p, n: (500, {"error": {"message": "oops"}}) if n == 1 else correct_answerer(p, n), extra={"max_retries": 1})
        self.assertEqual(res["status"], "complete")
        res, api, out, _ = run_live(lambda p, n: (200, H.TransportError("URLError")), extra={"max_retries": 1})
        self.assertEqual((res["status"], api.calls), ("aborted", 6))
        self.assertGreater(res["estimated_spend_usd"], 0)

    def test_refusals_empty_and_invalid_replies_are_non_answers(self):
        def h(p, n):
            return 200, [oai(None, refusal="no"), oai(""), {"choices": "oops", "usage": {"prompt_tokens": 5, "completion_tokens": 1}}, oai("maybe"), oai("A", finish="content_filter"), oai("A")][(n - 1) % 6]

        res, api, out, _ = run_live(h)
        ses = AN.read_session(out)
        self.assertEqual(ses["problems"], [])
        self.assertEqual(sorted(r["status"] for r in ses["records"]), sorted(["blocked", "empty", "malformed", "invalid", "blocked", "ok"]))
        a = AN.analyze(ses["records"], ses["prompts"])
        self.assertEqual((sum(t["non_answers"] for t in a["by_condition"].values()), sum(t["valid"] for t in a["by_condition"].values())), (5, 1))

    def test_a_reasoning_model_request_has_no_sampling_parameters(self):
        res, api, out, _ = run_live(model="gpt-5-test", extra={"reasoning_effort": "low"})
        self.assertEqual(res["status"], "complete")
        for b in api.bodies:
            self.assertNotIn("temperature", b)
            self.assertNotIn("top_p", b)
            self.assertEqual(b["reasoning_effort"], "low")

    def test_semantic_roles_across_permutations(self):
        def h(p, n):
            c, t = MAPS[p.split()[2]]
            return 200, oai(t if "COMPETING" in p else c)

        res, api, out, _ = run_live(h, mappings=("m1", "m2", "m3", "m4"))
        ses = AN.read_session(out)
        a = AN.analyze(ses["records"], ses["prompts"])
        self.assertEqual((a["by_condition"]["BASELINE"]["correct"], a["versus_baseline"]["COMPETING"]["induced_to_competing"]), (4, 4))
        self.assertEqual({r["answer"] for r in ses["records"] if r["condition"] == "BASELINE"}, {"A", "B", "C", "D"})

    def test_cli_live_run_and_missing_key(self):
        d = tmp()
        ds = make_dataset(os.path.join(d, "p.json"))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            code, text = run_cli("openai", "--prompts", ds, "--model", "gpt-4.1-mini", "--out", os.path.join(d, "s0"), "--budget", "1", "--price-in", "1", "--price-out", "1")
            self.assertEqual(code, 2)
            self.assertIn("OPENAI_API_KEY", text)
            self.assertFalse(os.path.exists(os.path.join(d, "s0")))
        from bleediff.cli import build_parser
        api = FakeAPI(correct_answerer)
        out = io.StringIO()
        ns = build_parser().parse_args(["openai", "--prompts", ds, "--model", "gpt-4.1-mini", "--out", os.path.join(d, "s"), "--budget", "1", "--price-in", "0.4", "--price-out", "1.6", "--passes", "2"])
        code = OA.cmd_openai(ns, out=out, post=api.post, sleep=api.sleep, env={"OPENAI_API_KEY": SECRET})
        text = out.getvalue()
        self.assertEqual(code, 0, text)
        self.assertIn("ESTIMATE", text)
        self.assertIn("not a provider billing total", text)
        self.assertNotIn(SECRET, text + all_text(d))


class TestBudgetRulesOfBothAdapters(unittest.TestCase):
    def test_openai_reasoning_tokens_inside_completion_tokens_count_once_and_the_reserve_is_the_output_cap(self):
        res, api, out, _ = run_live(lambda p, n: (200, oai("A", usage=(100, 401, 400))), budget=1.0, pin=1.0, pout=10.0, extra={"max_output_tokens": 600})
        rec = read_json(sorted(glob.glob(os.path.join(out, "records", "*.json")))[0])
        self.assertAlmostEqual(rec["cost_usd"], 100 * 1e-6 + 401 * 1e-5)
        self.assertFalse(rec["cost_exceeded_preflight_bound"])
        self.assertEqual(OA.OPENAI.output_reserve(600, {}), 600)

    def test_openai_cap_is_enforced_with_the_max_completion_tokens_reserve(self):
        # reserve 1000 output tokens = $0.01 + prompt; each reply bills 100 + 400 tokens (~$0.0041)
        res, api, out, _ = run_live(lambda p, n: (200, oai("A", usage=(100, 400, None))), budget=0.02, pin=1.0, pout=10.0, extra={"max_output_tokens": 1000})
        self.assertEqual((res["status"], api.calls), ("partial", 3))
        self.assertIn("BUDGET_EXHAUSTED", res["stop_reason"])
        self.assertLessEqual(res["estimated_spend_usd"], 0.02)
        self.assertIn("estimate", res["stop_reason"])

    def test_a_response_beyond_its_admitted_bound_stops_the_run(self):
        for runner in ("openai", "gemini"):
            if runner == "openai":
                res, api, out, _ = run_live(lambda p, n: (200, oai("A", usage=(100, 5000, None))), budget=5.0, pin=1.0, pout=10.0, extra={"max_output_tokens": 100})
            else:
                from tests.test_gemini_adapter import reply, run_live as glive
                res, api, out, _ = glive(lambda p, n: (200, reply("A", usage=(100, 1, 9000))), budget=5.0, pin=1.0, pout=10.0, extra={"max_output_tokens": 100, "thinking_budget": 50})
            self.assertEqual((res["status"], api.calls), ("aborted", 1), runner)
            self.assertIn("COST_BOUND_EXCEEDED", res["stop_reason"])
            rec = read_json(glob.glob(os.path.join(out, "records", "*.json"))[0])
            self.assertTrue(rec["cost_exceeded_preflight_bound"])
            self.assertGreater(rec["cost_usd"], 0)

    def test_missing_usage_fails_closed_for_both_providers(self):
        res, api, out, _ = run_live(lambda p, n: (200, oai("A", usage=None)), budget=1.0)
        recs = AN.read_session(out)["records"]
        self.assertTrue(all(not r["usage_complete"] and r["cost_usd"] > 0 for r in recs))
        from tests.test_gemini_adapter import reply, run_live as glive
        res, api, out, _ = glive(lambda p, n: (200, reply("A", usage=None)), budget=1.0)
        recs = AN.read_session(out)["records"]
        self.assertTrue(all(not r["usage_complete"] and r["cost_usd"] > 0 for r in recs))

    def test_gemini_reserve_accounts_for_thinking(self):
        self.assertEqual(GM.GEMINI.output_reserve(512, {}), 1024)                         # no explicit budget: assume thinking could add another full cap
        self.assertEqual(GM.GEMINI.output_reserve(512, {"thinking_level": "low"}), 1024)
        self.assertEqual(GM.GEMINI.output_reserve(512, {"thinking_budget": 0}), 512)
        self.assertEqual(GM.GEMINI.output_reserve(512, {"thinking_budget": 300}), 812)

    def test_the_reserve_grows_with_observed_output_so_later_requests_stop_sooner(self):
        res, api, out, _ = run_live(lambda p, n: (200, oai("A", usage=(100, 700, None))), budget=0.03, pin=1.0, pout=10.0, extra={"max_output_tokens": 100})
        self.assertEqual(res["status"], "aborted")                                         # 700 > the admitted bound of 100 tokens
        res, api, out, _ = run_live(lambda p, n: (200, oai("A", usage=(50, 100, None))), budget=0.0125, pin=1.0, pout=10.0, extra={"max_output_tokens": 100})
        self.assertIn(res["status"], ("partial", "complete"))
        self.assertLessEqual(res["estimated_spend_usd"], 0.0125)

    def test_documentation_states_that_limits_are_estimates_not_billing_totals(self):
        for doc in (H.__doc__, OA.__doc__, GM.__doc__, rt(os.path.join(ROOT, "README.md"))):
            self.assertIn("estimate", " ".join(doc.split()).lower())
        from bleediff.cli import build_parser
        sub = build_parser()._subparsers._group_actions[0].choices
        for name in ("openai", "gemini"):
            self.assertIn("not a provider billing total", " ".join(sub[name].format_help().split()), name)


class TestReplayOfTheSavedOpenAIPilot(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)
        self.net = mock.patch("urllib.request.urlopen", side_effect=AssertionError("network access attempted"))
        self.net.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.net.stop)

    def test_bundle_replays_offline_and_reproduces_the_pilot_findings(self):
        code, text = run_cli("openai", "--replay", os.path.join(BUNDLE, "gpt41mini"))
        self.assertEqual(code, 0, text)
        self.assertIn("no key used, no network", text)
        ses = AN.read_session(os.path.join(BUNDLE, "gpt41mini"))
        self.assertEqual((ses["problems"], ses["warnings"]), ([], []))
        a = AN.analyze(ses["records"], ses["prompts"])
        self.assertEqual((a["records"], a["status_counts"]), (160, {"ok": 160}))
        bc = a["by_condition"]
        for cond in ("BASELINE", "PROSE", "RANDOM_AB", "RANDOM_OTHER"):
            self.assertEqual((bc[cond]["correct"], bc[cond]["valid"]), (32, 32), cond)              # correct in every cell and pass
        for cond in ("PROSE", "RANDOM_AB", "RANDOM_OTHER"):
            self.assertEqual((a["versus_baseline"][cond]["answer_changed"], a["versus_baseline"][cond]["induced_incorrect"]), (0, 0), cond)
        self.assertEqual((bc["COMPETING"]["competing"], bc["COMPETING"]["correct"], bc["COMPETING"]["other_incorrect"]), (30, 2, 0))
        held = sorted(c for c, v in a["cells"].items() if v["conditions"]["COMPETING"][0]["role"] == "correct")
        self.assertEqual(held, ["q3-route-m4"])
        self.assertEqual(a["pass_agreement"]["all"], {"prompts_with_two_or_more_valid_passes": 80, "prompts_all_passes_agree": 80})
        self.assertAlmostEqual(a["estimated_cost_usd"], 0.0104, places=3)

    def test_the_bundle_carries_no_logprobs_and_says_so(self):
        s = read_json(os.path.join(BUNDLE, "gpt41mini", "session.json"))
        self.assertIn("removed and are not used", s["logprobs_note"])
        for f in glob.glob(os.path.join(BUNDLE, "gpt41mini", "records", "*.json")):
            r = read_json(f)
            self.assertFalse(AN.has_logprobs(r["response"]), f)
            self.assertEqual(r["measurement_type"], "ANSWER_ONLY")
        self.assertEqual(rb(os.path.join(BUNDLE, "prompts.json")), rb(os.path.join(GBUNDLE, "prompts.json")))      # the same frozen prompts as the Gemini bundle

    def test_legacy_pilot_replays_identically_and_matches_the_logprob_argmax_when_the_local_pilot_exists(self):
        v7 = sorted(glob.glob(os.path.join(ROOT, "runs", "v0.7-frontier-pilot-*")))
        if not (v7 and os.path.isdir(os.path.join(v7[0], "responses"))):
            self.skipTest("the local git-ignored pilot evidence is not present")
        legacy = AN.read_session(os.path.join(v7[0], "responses"), os.path.join(v7[0], "frozen-requests.json"))
        self.assertEqual((legacy["kind"], legacy["problems"]), ("legacy", []))
        self.assertTrue(any("logprobs" in w for w in legacy["warnings"]))
        new = AN.read_session(os.path.join(BUNDLE, "gpt41mini"))
        la, na = AN.analyze(legacy["records"], legacy["prompts"]), AN.analyze(new["records"], new["prompts"])
        for k in ("by_condition", "versus_baseline", "by_condition_mapping", "pass_agreement", "cells", "usage"):
            self.assertEqual(la[k], na[k], k)
        # the earlier DIRECT-evidence reading (arg-max of the saved top-20 logprobs) agrees with the returned answer in every record: verified from the saved files
        agree = 0
        for r in legacy["records"]:
            top = r["response"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            mass = {}
            for t in top:
                if t["token"] in ("A", "B", "C", "D", " A", " B", " C", " D"):
                    mass[t["token"].strip()] = mass.get(t["token"].strip(), 0) + math.exp(t["logprob"])
            agree += bool(mass) and max(mass, key=mass.get) == r["answer"]
        self.assertEqual(agree, 160)

    def test_tampered_openai_evidence_is_detected(self):
        d = tmp()
        for tamper in ("answer", "logprobs", "provider", "dataset", "measurement_type"):
            dst = os.path.join(d, tamper)
            shutil.copytree(os.path.join(BUNDLE, "gpt41mini"), dst)
            shutil.copy(os.path.join(BUNDLE, "prompts.json"), os.path.join(d, "prompts.json"))
            f = sorted(glob.glob(os.path.join(dst, "records", "*.json")))[0]
            r = read_json(f)
            if tamper == "answer":
                r["answer"] = "B" if r["answer"] != "B" else "C"
            elif tamper == "logprobs":
                r["response"]["choices"][0]["logprobs"] = {"content": []}
                r["response"]["choices"][0]["logprobs"]["content"] = [{"token": "A", "logprob": 0.0, "top_logprobs": []}]
            elif tamper == "provider":
                r["provider"] = "mystery"
            elif tamper == "measurement_type":
                r["measurement_type"] = "DIRECT_PROBABILITY"
            elif tamper == "dataset":
                pr = read_json(os.path.join(d, "prompts.json"))
                pr["items"][0]["prompt"] += " changed"
                write_json(os.path.join(d, "prompts.json"), pr)
            write_json(f, r)
            code, text = run_cli("openai", "--replay", dst)
            self.assertEqual(code, 2, tamper)

    def test_the_same_engine_replays_both_providers(self):
        for cmd, path in (("openai", os.path.join(BUNDLE, "gpt41mini")), ("gemini", os.path.join(GBUNDLE, "flash")), ("openai", os.path.join(GBUNDLE, "carbon")), ("gemini", os.path.join(BUNDLE, "gpt41mini"))):
            code, text = run_cli(cmd, "--replay", path)
            self.assertEqual(code, 0, text)
            self.assertIn("ANSWER_ONLY", text)


class TestThreeSourceAtlas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tmp()
        cls.out = os.path.join(cls.d, "atlas")
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network access attempted")):
            cls.code, cls.text = run_cli("atlas", "--manifest", os.path.join(BUNDLE, "atlas-input.json"), "--out", cls.out)
        cls.atlas = read_json(os.path.join(cls.out, "atlas.json"))

    def test_local_gemini_and_openai_results_appear_together_with_separate_labels(self):
        self.assertEqual(self.code, 0, self.text)
        ao = self.atlas["answer_only"]
        self.assertEqual(sorted((s["provider"], s["model"]["id"]) for s in ao["sessions"]), [("gemini", "gemini-3.8-flash"), ("gemini", "gemini-4-carbon"), ("openai", "gpt-4.1-mini")])
        self.assertEqual({s["measurement_type"] for s in ao["sessions"]}, {"ANSWER_ONLY"})
        self.assertEqual({c["measurement_type"] for c in self.atlas["cells"]}, {"DIRECT_PROBABILITY"})
        self.assertEqual({c["model_id"] for c in self.atlas["cells"]}, {"qwen1.5b", "llama3b"})
        report = rt(os.path.join(self.out, "report.md"))
        for needle in ("DIRECT_PROBABILITY", "ANSWER_ONLY", "gpt-4.1-mini", "gemini-3.8-flash", "gemini-4-carbon", "provider openai", "provider gemini"):
            self.assertIn(needle, report, needle)
        self.assertLess(report.index("## Task/model cells"), report.index("## Hosted answer-only results"))

    def test_comparison_rows_keep_the_types_apart_and_use_the_saved_evidence(self):
        rows = csv_rows(os.path.join(self.out, "answer-level-comparison.csv"))
        by = {(r["measurement_type"], r["source"], r["label"]): r for r in rows}
        for src, comp, corr in (("gemini-3.8-flash", "32", "0"), ("gemini-4-carbon", "28", "4"), ("gpt-4.1-mini", "30", "2")):
            r = by[("ANSWER_ONLY", src, "COMPETING")]
            self.assertEqual((r["competing"], r["correct"], r["n"]), (comp, corr, "32"), src)
            self.assertIn("returned answer", r["answer_basis"])
        for src in ("qwen1.5b", "llama3b"):
            self.assertIn("probabilities", by[("DIRECT_PROBABILITY", src, "COMPETING")]["answer_basis"])
        self.assertEqual({r["measurement_type"] for r in rows}, {"ANSWER_ONLY", "DIRECT_PROBABILITY"})
        ao = csv_rows(os.path.join(self.out, "answer-only.csv"))
        self.assertEqual({r["model_id"] for r in ao}, {"gemini-3.8-flash", "gemini-4-carbon", "gpt-4.1-mini"})
        self.assertEqual({r["measurement_type"] for r in ao}, {"ANSWER_ONLY"})

    def test_no_artificial_margins_or_probabilities_for_any_hosted_model(self):
        text = json.dumps(self.atlas["answer_only"]).lower()
        for banned in ('"shift', '"margin', '"probabilit', '"rank', '"p_a"', '"coverage"', "log_odds", "logit", "leaderboard\":"):
            self.assertNotIn(banned, text, banned)
        cand = csv_rows(os.path.join(self.out, "candidates.csv"))
        self.assertFalse({r["model_id"] for r in cand} & {"gemini-3.8-flash", "gemini-4-carbon", "gpt-4.1-mini"})
        self.assertNotIn("gpt-4.1", rt(os.path.join(self.out, "model-comparison.csv")) + rt(os.path.join(self.out, "category-summary.csv")))

    def test_hosted_model_filter_selects_the_openai_session(self):
        d = tmp()
        code, text = run_cli("atlas", "--manifest", os.path.join(BUNDLE, "atlas-input.json"), "--out", os.path.join(d, "f"), "--model", "gpt-4.1-mini")
        self.assertEqual(code, 0, text)
        self.assertEqual({r["model_id"] for r in csv_rows(os.path.join(d, "f", "answer-only.csv"))}, {"gpt-4.1-mini"})

    def test_local_probability_results_are_unchanged_by_either_hosted_source(self):
        d = tmp()
        shutil.copytree(BUNDLE, os.path.join(d, "o"))
        shutil.copytree(GBUNDLE, os.path.join(d, "gemini-answer-only"))
        shutil.copytree(os.path.join(ROOT, "examples", "four-mapping"), os.path.join(d, "four-mapping"))
        plain = read_json(os.path.join(d, "o", "atlas-input.json"))
        del plain["answer_only_sessions"]
        write_json(os.path.join(d, "o", "plain.json"), plain)
        code, text = run_cli("atlas", "--manifest", os.path.join(d, "o", "plain.json"), "--out", os.path.join(d, "plain"))
        self.assertEqual(code, 0, text)
        base = read_json(os.path.join(d, "plain", "atlas.json"))
        strip = lambda cells: [{k: v for k, v in c.items() if k != "measurement_type"} for c in cells]
        self.assertEqual(strip(self.atlas["cells"]), strip(base["cells"]))
        self.assertEqual(self.atlas["aggregate"], base["aggregate"])
        for fn in ("candidates.csv", "category-summary.csv", "model-comparison.csv"):
            self.assertEqual(rb(os.path.join(self.out, fn)), rb(os.path.join(d, "plain", fn)), fn)

    def test_an_invalid_openai_session_fails_closed_unless_keep_going(self):
        d = tmp()
        shutil.copytree(BUNDLE, os.path.join(d, "o"))
        shutil.copytree(GBUNDLE, os.path.join(d, "gemini-answer-only"))
        shutil.copytree(os.path.join(ROOT, "examples", "four-mapping"), os.path.join(d, "four-mapping"))
        f = sorted(glob.glob(os.path.join(d, "o", "gpt41mini", "records", "*.json")))[0]
        r = read_json(f)
        r["answer"] = "B" if r["answer"] != "B" else "C"
        write_json(f, r)
        mp = os.path.join(d, "o", "atlas-input.json")
        code, text = run_cli("atlas", "--manifest", mp, "--out", os.path.join(d, "o1"))
        self.assertEqual(code, 2)
        self.assertIn("failed validation", text)
        code, text = run_cli("atlas", "--manifest", mp, "--out", os.path.join(d, "o2"), "--keep-going")
        self.assertEqual(code, 0, text)
        a = read_json(os.path.join(d, "o2", "atlas.json"))
        self.assertEqual(sorted(s["model"]["id"] for s in a["answer_only"]["sessions"]), ["gemini-3.8-flash", "gemini-4-carbon"])
        self.assertEqual(len(a["answer_only"]["excluded"]), 1)


class TestBundleHygiene(unittest.TestCase):
    def test_bundle_is_small_lf_text_without_secrets_or_local_data(self):
        import re
        files = [os.path.join(r, f) for r, _d, fs in os.walk(BUNDLE) for f in fs]
        self.assertLess(sum(os.path.getsize(p) for p in files), 2 * 1024 * 1024)
        pat = re.compile(r"(sk-[A-Za-z0-9_-]{10,}|AIza[0-9A-Za-z_-]{10,}|/Users/|/home/|AppData|huggingface|\.cache|@[A-Za-z0-9-]+\.(com|org|net|io)|api[_-]?key\s*[:=]|Bearer )", re.I)
        for p in files:
            data = rb(p)
            self.assertNotIn(b"\r\n", data, p)
            self.assertIsNone(pat.search(data.decode("utf-8")), p)
        self.assertIn("examples/openai-answer-only/** -text", rt(os.path.join(ROOT, ".gitattributes")))

    def test_sha256sums_cover_the_bundle(self):
        import hashlib
        listed = dict(reversed(l.split("  ", 1)) for l in rt(os.path.join(BUNDLE, "SHA256SUMS.txt")).splitlines())
        found = {os.path.relpath(os.path.join(r, f), BUNDLE).replace("\\", "/") for r, _d, fs in os.walk(BUNDLE) for f in fs} - {"SHA256SUMS.txt", "README.md"}
        self.assertEqual(set(listed), found)
        for rel, digest in listed.items():
            self.assertEqual(hashlib.sha256(rb(os.path.join(BUNDLE, rel))).hexdigest(), digest, rel)


if __name__ == "__main__":
    unittest.main()
