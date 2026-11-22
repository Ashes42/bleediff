"""Gemini ANSWER-ONLY adapter (bleediff.answers, bleediff.gemini, `bleediff gemini`) and its Atlas integration. All HTTP is mocked: no test contacts a
network or needs a key. The tracked example bundle (examples/gemini-answer-only, converted from the v0.8 pilot) supplies the real saved evidence."""

from __future__ import annotations

import atexit
import copy
import csv
import glob
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from tests.test_bleediff import run_cli  # noqa: E402

from bleediff import answers as AN  # noqa: E402
from bleediff import atlas as AT  # noqa: E402
from bleediff import gemini as GM  # noqa: E402

BUNDLE = os.path.join(ROOT, "examples", "gemini-answer-only")
SECRET = "AI" + "zaSyFAKE-test-key-0123456789abcdefghij"          # shaped like a key, built at run time; it is not a credential


def tmp():
    d = tempfile.mkdtemp(prefix="bleediff-gemini-test-")
    atexit.register(shutil.rmtree, d, True)
    return d


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")


def csv_rows(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def rb(path):
    with open(path, "rb") as fh:
        return fh.read()


def rt(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


MAPS = {"m1": ("A", "B"), "m2": ("B", "A"), "m3": ("C", "D"), "m4": ("D", "C")}


def make_dataset(path, mappings=("m1", "m3"), conditions=("BASELINE", "PROSE", "COMPETING")):
    items = []
    for mp in mappings:
        for cond in conditions:
            c, t = MAPS[mp]
            items.append({"cell": f"q1-{mp}", "question": "q1", "mapping": mp, "condition": cond, "prompt": f"Question q1 {mp} {cond}.\nAnswer:", "correct_letter": c, "competing_letter": t})
    write_json(path, {"schema": "bleediff.hosted-prompts/1", "baseline_condition": "BASELINE", "items": items})
    return path


def reply(text, usage=(120, 1, None), version="gemini-test-001", finish="STOP"):
    r = {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish}], "modelVersion": version}
    if usage is not None:
        pt, ct, th = usage
        u = {"promptTokenCount": pt, "candidatesTokenCount": ct, "totalTokenCount": pt + ct + (th or 0)}
        if th is not None:
            u["thoughtsTokenCount"] = th
        r["usageMetadata"] = u
    return r


class FakeAPI:
    """A mock transport. handler(prompt, call_no) -> (status, response-dict-or-bytes-or-Exception)."""

    def __init__(self, handler):
        self.handler, self.calls, self.urls, self.bodies, self.keys, self.sleeps = handler, 0, [], [], [], []

    def post(self, url, data, key, timeout):
        self.calls += 1
        body = json.loads(data.decode("utf-8"))
        self.urls.append(url)
        self.bodies.append(body)
        self.keys.append(key)
        prompt = body["contents"][0]["parts"][0]["text"] if "contents" in body else body["messages"][0]["content"]          # Gemini or OpenAI request shape
        status, payload = self.handler(prompt, self.calls)
        if isinstance(payload, Exception):
            raise payload
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        return status, {}, raw

    def sleep(self, s):
        self.sleeps.append(s)


def correct_answerer(prompt, n):
    """Answers the correct letter, except under COMPETING (the competing letter)."""
    cell = prompt.split()[2]
    c, t = MAPS[cell]
    return 200, reply(t if "COMPETING" in prompt else c)


def run_live(handler=correct_answerer, *, passes=1, budget=1.0, pin=1.0, pout=10.0, dataset=None, extra=None, mappings=("m1", "m3")):
    d = tmp()
    ds = dataset or make_dataset(os.path.join(d, "prompts.json"), mappings=mappings)
    api = FakeAPI(handler)
    out = os.path.join(d, "session")
    kw = dict(prompts_path=ds, model="gemini-test", out_dir=out, passes=passes, budget=budget, price_in_per_m=pin, price_out_per_m=pout, key=SECRET, post=api.post, sleep=api.sleep)
    kw.update(extra or {})
    res = GM.run_session(**kw)
    return res, api, out, ds


def all_text(directory):
    out = []
    for r, _d, fs in os.walk(directory):
        for f in fs:
            out.append(rt(os.path.join(r, f)))
    return "\n".join(out)


# ======================================================================================
class TestParsing(unittest.TestCase):
    def test_valid_letters_and_whitespace(self):
        for l in "ABCD":
            self.assertEqual(AN.parse_answer(reply(l))["answer"], l)
        p = AN.parse_answer(reply("  B\n"))
        self.assertEqual((p["status"], p["answer"]), ("ok", "B"))

    def test_not_exactly_one_letter_is_invalid_not_guessed(self):
        for text in ("a", "AB", "A.", "Answer: A", "E", "The answer is A", "1"):
            p = AN.parse_answer(reply(text))
            self.assertEqual((p["status"], p["answer"]), ("invalid", None), text)

    def test_thought_parts_are_ignored(self):
        r = reply("C")
        r["candidates"][0]["content"]["parts"].insert(0, {"text": "A", "thought": True})
        self.assertEqual(AN.parse_answer(r)["answer"], "C")

    def test_blocked_prompt_and_blocked_output(self):
        self.assertEqual(AN.parse_answer({"promptFeedback": {"blockReason": "SAFETY"}})["status"], "blocked")
        r = reply("A", finish="SAFETY")
        self.assertEqual((AN.parse_answer(r)["status"], AN.parse_answer(r)["answer"]), ("blocked", None))
        self.assertEqual(AN.parse_answer(reply("", finish="PROHIBITED_CONTENT"))["status"], "blocked")

    def test_empty_missing_and_malformed_responses(self):
        self.assertEqual(AN.parse_answer({"candidates": []})["status"], "empty")
        self.assertEqual(AN.parse_answer({})["status"], "empty")
        self.assertEqual(AN.parse_answer(reply(""))["status"], "empty")
        self.assertEqual(AN.parse_answer({"candidates": [{"finishReason": "STOP"}]})["status"], "empty")
        for bad in ("text", None, 5, [], {"candidates": "x"}, {"candidates": [5]}, {"candidates": [{"content": {"parts": "A"}}]}):
            self.assertIn(AN.parse_answer(bad)["status"], ("malformed", "empty"), bad)
        self.assertEqual(AN.parse_answer("text")["status"], "malformed")
        self.assertEqual(AN.parse_answer({"candidates": "x"})["status"], "malformed")

    def test_max_tokens_with_a_valid_letter_is_still_an_answer(self):
        self.assertEqual(AN.parse_answer(reply("D", finish="MAX_TOKENS"))["answer"], "D")

    def test_usage_extraction_ignores_non_integers(self):
        u = AN.usage_from({"usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": True, "thoughtsTokenCount": -1, "totalTokenCount": "9"}})
        self.assertEqual(u, {"promptTokenCount": 5, "candidatesTokenCount": None, "thoughtsTokenCount": None, "totalTokenCount": None})


class TestDataset(unittest.TestCase):
    def test_rejects_bad_datasets(self):
        d = tmp()
        good = read_json(make_dataset(os.path.join(d, "g.json")))
        cases = {}
        x = copy.deepcopy(good); x["items"].pop(0); cases["no baseline"] = x
        x = copy.deepcopy(good); x["items"][1]["condition"] = x["items"][2]["condition"]; cases["duplicate"] = x
        x = copy.deepcopy(good); x["items"][0]["prompt_sha256"] = "0" * 64; cases["bad hash"] = x
        x = copy.deepcopy(good); x["items"][0]["competing_letter"] = x["items"][0]["correct_letter"]; cases["same letters"] = x
        x = copy.deepcopy(good); x["items"][1]["correct_letter"] = "D"; cases["roles differ inside a cell"] = x
        x = copy.deepcopy(good); x["schema"] = "other"; cases["schema"] = x
        for name, doc in cases.items():
            write_json(os.path.join(d, "b.json"), doc)
            with self.assertRaises(AN.AnswersError, msg=name):
                AN.load_prompts(os.path.join(d, "b.json"))
        with self.assertRaises(AN.AnswersError):
            AN.load_prompts(os.path.join(d, "missing.json"))

    def test_accepts_the_v07_frozen_schema_and_hashes_prompts(self):
        ds = AN.load_prompts(os.path.join(BUNDLE, "prompts.json"))
        self.assertEqual((len(ds["items"]), len(ds["cells"]), ds["conditions"]), (80, 16, ["BASELINE", "PROSE", "RANDOM_AB", "RANDOM_OTHER", "COMPETING"]))


class TestRequestAndSecrets(unittest.TestCase):
    def test_request_never_asks_for_logprobs(self):
        body = GM.build_request("hi", thinking_level="low")
        self.assertNotIn("responseLogprobs", json.dumps(body))
        self.assertNotIn("logprobs", json.dumps(body).lower())
        self.assertEqual(body["generationConfig"]["thinkingConfig"], {"thinkingLevel": "low"})
        self.assertEqual(GM.build_request("hi", thinking_budget=0)["generationConfig"]["thinkingConfig"], {"thinkingBudget": 0})
        with self.assertRaises(Exception):
            GM.build_request("hi", thinking_level="low", thinking_budget=0)

    def test_invalid_model_names(self):
        for bad in ("", "a b", "../x", "m?key=1", "x/y", None, "m" * 200):
            with self.assertRaises(Exception, msg=repr(bad)):
                GM.normalise_model(bad)
        self.assertEqual(GM.normalise_model("models/gemini-3.8-flash"), "gemini-3.8-flash")

    def test_missing_api_key(self):
        for env in ({}, {"GEMINI_API_KEY": ""}, {"GEMINI_API_KEY": "   "}):
            with self.assertRaises(GM.GeminiError) as cm:
                GM.api_key(env)
            self.assertEqual(cm.exception.reason, "MISSING_API_KEY")
        self.assertEqual(GM.api_key({"GEMINI_API_KEY": "  k  "}), "k")

    def test_redaction(self):
        s = GM.redact(f"error for key {SECRET} and " + "AI" + "zaSyABCDEFGHIJKLMNOP" + " and x-goog-api-key: abc123 and ?key=zzz&a=1", SECRET)
        for leak in (SECRET, "AI" + "zaSyABCDEFGH", "abc123", "key=zzz"):
            self.assertNotIn(leak, s)
        self.assertIn("[REDACTED]", s)
        self.assertNotIn(SECRET, str(GM.GeminiError(f"boom {SECRET}", key=SECRET)))

    def test_transport_uses_a_header_not_the_url_and_hides_exception_text(self):
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

        def fake_urlopen(req, timeout=None):
            seen["url"], seen["headers"] = req.full_url, dict(req.header_items())
            return Resp()

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            GM._http_post(f"{GM.API_BASE}/models/m:generateContent", b"{}", SECRET, 5)
        self.assertNotIn(SECRET, seen["url"])
        self.assertEqual(seen["headers"].get("X-goog-api-key"), SECRET)
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError(f"cannot reach host with {SECRET}")):
            with self.assertRaises(GM.TransportError) as cm:
                GM._http_post("https://x", b"{}", SECRET, 5)
        self.assertNotIn(SECRET, str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)


class TestCost(unittest.TestCase):
    def test_usage_based_cost_counts_thinking_tokens(self):
        cost, basis, ok = GM.billed_cost({"promptTokenCount": 100, "candidatesTokenCount": 1, "thoughtsTokenCount": 99, "totalTokenCount": 200}, 1e-6, 1e-5, 500, 600)
        self.assertTrue(ok)
        self.assertAlmostEqual(cost, 100 * 1e-6 + 100 * 1e-5)
        self.assertEqual(basis, "usage_metadata")

    def test_tokens_only_visible_in_the_total_are_billed_as_output(self):
        cost, basis, ok = GM.billed_cost({"promptTokenCount": 100, "candidatesTokenCount": 5, "thoughtsTokenCount": None, "totalTokenCount": 130}, 1e-6, 1e-5, 500, 600)
        self.assertTrue(ok)
        self.assertAlmostEqual(cost, 100 * 1e-6 + 30 * 1e-5)
        self.assertEqual(basis, "usage_metadata_total_minus_prompt")

    def test_missing_or_inconsistent_usage_is_charged_at_the_worst_case(self):
        worst = 500 * 1e-6 + 600 * 1e-5
        for usage in ({}, {"promptTokenCount": 10}, {"candidatesTokenCount": 1}, {"promptTokenCount": 100, "candidatesTokenCount": 1, "totalTokenCount": 50}):
            cost, basis, ok = GM.billed_cost(usage, 1e-6, 1e-5, 500, 600)
            self.assertFalse(ok)
            self.assertAlmostEqual(cost, worst)
            self.assertTrue(basis.startswith("upper_bound"))


class TestHttpHandling(unittest.TestCase):
    def test_valid_run_writes_labelled_records_and_a_session(self):
        res, api, out, ds = run_live(passes=2)
        self.assertEqual((res["status"], api.calls), ("complete", 12))
        s = read_json(os.path.join(out, "session.json"))
        self.assertEqual((s["measurement_type"], s["provider"], s["status"], s["logprobs_requested"]), ("ANSWER_ONLY", "gemini", "complete", False))
        self.assertEqual(rb(os.path.join(out, "prompts.json")), rb(ds))
        recs = sorted(glob.glob(os.path.join(out, "records", "*.json")))
        self.assertEqual(len(recs), 12)
        for f in recs:
            r = read_json(f)
            self.assertEqual((r["schema"], r["measurement_type"], r["status"]), (AN.RECORD_SCHEMA, "ANSWER_ONLY", "ok"))
            self.assertFalse(r["request"]["credentials_recorded"])
            self.assertTrue(r["usage_complete"])
            self.assertNotIn("probabilities", r)
        self.assertTrue(all(SECRET == k for k in api.keys))
        self.assertTrue(all("logprobs" not in json.dumps(b).lower() for b in api.bodies))
        self.assertTrue(all(SECRET not in u for u in api.urls))
        self.assertNotIn(SECRET, all_text(out))

    def test_invalid_model_is_not_retried(self):
        d = tmp()
        api = FakeAPI(lambda p, n: (404, {"error": {"code": 404, "message": f"models/gemini-bad is not found; key {SECRET}", "status": "NOT_FOUND"}}))
        res = GM.run_session(prompts_path=make_dataset(os.path.join(d, "p.json")), model="gemini-bad", out_dir=os.path.join(d, "s"), passes=1, budget=1, price_in_per_m=1, price_out_per_m=1,
                             key=SECRET, post=api.post, sleep=api.sleep)
        self.assertEqual((api.calls, res["status"]), (1, "aborted"))
        self.assertIn("MODEL_NOT_AVAILABLE", res["stop_reason"])
        rec = read_json(glob.glob(os.path.join(d, "s", "records", "*.json"))[0])
        self.assertEqual((rec["status"], rec["error"]["reason"], rec["cost_usd"]), ("error", "MODEL_NOT_AVAILABLE", 0.0))
        self.assertNotIn(SECRET, all_text(os.path.join(d, "s")))

    def test_auth_and_bad_request_errors_stop_at_once(self):
        for code, reason in ((401, "AUTH_FAILED"), (403, "AUTH_FAILED"), (400, "BAD_REQUEST")):
            res, api, out, _ = run_live(lambda p, n, c=code: (c, {"error": {"message": "nope " + SECRET}}))
            self.assertEqual((api.calls, res["status"]), (1, "aborted"))
            self.assertIn(reason, res["stop_reason"])
            self.assertNotIn(SECRET, all_text(out) + res["stop_reason"])

    def test_rate_limit_is_retried_with_bounded_backoff_then_succeeds(self):
        def h(p, n):
            return (429, {"error": {"message": "slow down"}}) if n <= 2 else correct_answerer(p, n)

        res, api, out, _ = run_live(h, extra={"max_retries": 2})
        self.assertEqual(res["status"], "complete")
        self.assertEqual(api.sleeps[:2], [2, 4])
        self.assertEqual(read_json(sorted(glob.glob(os.path.join(out, "records", "*.json")))[0])["attempts"], 3)

    def test_retry_after_header_is_honoured_up_to_a_cap(self):
        calls = []

        def post(url, data, key, timeout):
            calls.append(1)
            return (429, {"Retry-After": "7"}, b"{}") if len(calls) == 1 else (200, {}, json.dumps(reply("A")).encode())

        sleeps = []
        r = GM.generate(post, "m", {}, SECRET, timeout=1, max_retries=2, sleep=sleeps.append)
        self.assertEqual((r["attempts"], sleeps), (2, [7]))

    def test_persistent_rate_limits_are_bounded_and_abort(self):
        res, api, out, _ = run_live(lambda p, n: (429, {"error": {"message": "quota"}}), extra={"max_retries": 2})
        self.assertEqual(res["status"], "aborted")
        self.assertEqual(api.calls, 3 * GM.MAX_CONSECUTIVE_FAILURES)                   # 3 attempts per request, then stop after 3 failed requests
        self.assertEqual(len(api.sleeps), 2 * GM.MAX_CONSECUTIVE_FAILURES)
        self.assertIn("RATE_LIMITED", res["stop_reason"])

    def test_transient_server_and_network_errors_are_retried_then_recorded_honestly(self):
        res, api, out, _ = run_live(lambda p, n: (503, {"error": {"message": "unavailable"}}) if n == 1 else correct_answerer(p, n), extra={"max_retries": 1})
        self.assertEqual(res["status"], "complete")
        res, api, out, _ = run_live(lambda p, n: (200, GM.TransportError("URLError")), extra={"max_retries": 1})
        self.assertEqual((res["status"], api.calls), ("aborted", 6))
        rec = read_json(sorted(glob.glob(os.path.join(out, "records", "*.json")))[0])
        self.assertEqual((rec["status"], rec["error"]["reason"]), ("error", "TRANSPORT_ERROR"))
        self.assertGreater(rec["cost_usd"], 0)                                          # a request that may have been processed is charged at the worst case
        self.assertGreater(res["estimated_spend_usd"], 0)

    def test_non_json_success_body_is_an_error(self):
        res, api, out, _ = run_live(lambda p, n: (200, b"<html>not json</html>"))
        self.assertEqual(res["status"], "aborted")
        self.assertIn("BAD_RESPONSE", res["stop_reason"])

    def test_blocked_empty_and_malformed_answers_are_recorded_as_non_answers(self):
        def h(p, n):
            kind = ["blocked", "empty", "malformed", "invalid", "ok", "ok"][(n - 1) % 6]
            return 200, {"blocked": {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 0}},
                         "empty": reply(""), "malformed": {"candidates": "oops", "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1}},
                         "invalid": reply("maybe"), "ok": correct_answerer(p, n)[1]}[kind]

        res, api, out, ds = run_live(h)
        self.assertEqual(res["status"], "complete")
        ses = AN.read_session(out)
        self.assertEqual(ses["problems"], [])
        self.assertEqual(sorted(r["status"] for r in ses["records"]), sorted(["blocked", "empty", "malformed", "invalid", "ok", "ok"]))
        a = AN.analyze(ses["records"], ses["prompts"])
        self.assertEqual(sum(t["non_answers"] for t in a["by_condition"].values()), 4)
        self.assertEqual(sum(t["valid"] for t in a["by_condition"].values()), 2)


class TestBudget(unittest.TestCase):
    def test_cap_is_enforced_before_a_request_can_exceed_it(self):
        # each reply bills 100 prompt + 401 output tokens (~$0.0041); the reserve for the next request then grows to 2 x 401 output tokens (~$0.0081)
        res, api, out, _ = run_live(lambda p, n: (200, reply("A", usage=(100, 1, 400))), budget=0.02, pin=1.0, pout=10.0)
        self.assertEqual(res["status"], "partial")
        self.assertIn("BUDGET_EXHAUSTED", res["stop_reason"])
        self.assertEqual(api.calls, 3)
        self.assertLessEqual(res["estimated_spend_usd"], 0.02)
        self.assertEqual(read_json(os.path.join(out, "session.json"))["status"], "partial")

    def test_a_cap_below_one_worst_case_request_sends_nothing(self):
        res, api, out, _ = run_live(budget=0.0001)
        self.assertEqual((api.calls, res["status"], res["requests_attempted"]), (0, "partial", 0))

    def test_missing_usage_metadata_is_charged_and_flagged_never_free(self):
        res, api, out, _ = run_live(lambda p, n: (200, reply("A" if "m1" in p or "BASELINE" in p else "C", usage=None)), budget=1.0)
        ses = AN.read_session(out)
        self.assertTrue(all(not r["usage_complete"] for r in ses["records"]))
        self.assertTrue(all(r["cost_usd"] > 0 and r["cost_basis"].startswith("upper_bound") for r in ses["records"]))
        a = AN.analyze(ses["records"], ses["prompts"])
        self.assertEqual(a["usage"]["records_with_incomplete_usage"], 6)
        self.assertTrue(any("incomplete usage" in l for l in AN.report_lines(a)))

    def test_incomplete_usage_exhausts_the_budget_sooner(self):
        full, _, _, _ = run_live(lambda p, n: correct_answerer(p, n), budget=0.02)
        poor, _, _, _ = run_live(lambda p, n: (200, reply("A", usage=None)), budget=0.02)
        self.assertEqual(full["status"], "complete")
        self.assertEqual(poor["status"], "partial")

    def test_thinking_tokens_are_billed_and_raise_the_reserve(self):
        res, api, out, _ = run_live(lambda p, n: (200, reply("A", usage=(100, 1, 400))), budget=1.0, pout=10.0)
        ses = AN.read_session(out)
        self.assertAlmostEqual(ses["records"][0]["cost_usd"], 100 * 1e-6 + 401 * 1e-5)

    def test_explicit_arguments_are_required_and_validated(self):
        d = tmp()
        ds = make_dataset(os.path.join(d, "p.json"))
        base = dict(prompts_path=ds, model="gemini-test", out_dir=os.path.join(d, "s"), passes=1, budget=1.0, price_in_per_m=1.0, price_out_per_m=1.0, key=SECRET)
        for change in ({"budget": 0}, {"budget": -1}, {"budget": float("nan")}, {"price_in_per_m": -1}, {"price_out_per_m": float("inf")}, {"passes": 0}, {"passes": 99},
                       {"max_retries": 9}, {"thinking_level": "extreme"}, {"max_output_tokens": 0}, {"thinking_level": "low", "thinking_budget": 0}, {"model": "bad model"}):
            api = FakeAPI(correct_answerer)
            with self.assertRaises(Exception, msg=str(change)):
                GM.run_session(**{**base, **change}, post=api.post)
            self.assertEqual(api.calls, 0)
        self.assertEqual(sorted(os.listdir(d)), ["p.json"])                              # nothing was created by the refused runs

    def test_output_directory_is_never_overwritten(self):
        d = tmp()
        ds = make_dataset(os.path.join(d, "p.json"))
        os.makedirs(os.path.join(d, "session"))
        api = FakeAPI(correct_answerer)
        with self.assertRaises(Exception):
            GM.run_session(prompts_path=ds, model="m", out_dir=os.path.join(d, "session"), passes=1, budget=1, price_in_per_m=1, price_out_per_m=1, key=SECRET, post=api.post)
        self.assertEqual(api.calls, 0)


class TestSemanticAnalysis(unittest.TestCase):
    """Correct-versus-competing tracking across answer-letter permutations; answers only, no probabilities."""

    def run_with(self, plan):
        def h(prompt, n):
            mp, cond = prompt.split()[2], prompt.split()[3].rstrip(".")
            return 200, reply(plan(mp, cond))

        res, api, out, ds = run_live(h, passes=1, mappings=("m1", "m2", "m3", "m4"), extra={})
        ses = AN.read_session(out)
        return AN.analyze(ses["records"], ses["prompts"]), ses

    def test_same_semantic_answer_at_different_letters(self):
        def plan(mp, cond):
            c, t = MAPS[mp]
            return t if cond == "COMPETING" else c
        a, ses = self.run_with(plan)
        self.assertEqual(a["by_condition"]["BASELINE"]["correct"], 4)
        self.assertEqual(a["by_condition"]["COMPETING"]["competing"], 4)
        self.assertEqual(a["versus_baseline"]["COMPETING"], {**a["versus_baseline"]["COMPETING"], "pairs": 4, "answer_changed": 4, "induced_incorrect": 4, "induced_to_competing": 4, "induced_to_other": 0})
        letters = {r["answer"] for r in ses["records"] if r["condition"] == "BASELINE"}
        self.assertEqual(letters, {"A", "B", "C", "D"})                                # different letters, one semantic role
        for mp in MAPS:
            self.assertEqual(a["by_condition_mapping"][f"BASELINE|{mp}"]["correct"], 1)

    def test_induced_other_recovered_and_flip_accounting(self):
        def plan(mp, cond):
            c, t = MAPS[mp]
            if mp == "m1":                                   # baseline wrong (competing); PROSE recovers; COMPETING stays wrong
                return {"BASELINE": t, "PROSE": c, "COMPETING": t}[cond]
            if mp == "m2":                                        # baseline correct; PROSE induces an "other incorrect" answer
                other = [x for x in "ABCD" if x not in (c, t)][0]
                return {"BASELINE": c, "PROSE": other, "COMPETING": c}[cond]
            return c
        a, _ = self.run_with(plan)
        p = a["versus_baseline"]["PROSE"]
        self.assertEqual((p["induced_incorrect"], p["induced_to_other"], p["induced_to_competing"], p["recovered_to_correct"], p["baseline_incorrect_pairs"], p["answer_changed"]), (1, 1, 0, 1, 1, 2))
        c = a["versus_baseline"]["COMPETING"]
        self.assertEqual((c["induced_incorrect"], c["answer_changed"]), (0, 0 + 0))      # a wrong baseline that stays wrong is neither a flip nor an induced error
        self.assertEqual(a["by_condition"]["BASELINE"]["competing"], 1)

    def test_pass_agreement_is_replicate_stability_only(self):
        d = tmp()
        ds = make_dataset(os.path.join(d, "p.json"), mappings=("m1",))
        api = FakeAPI(lambda p, n: (200, reply("A" if n <= 3 else ("A" if "PROSE" in p else "B"))))
        out = os.path.join(d, "s")
        GM.run_session(prompts_path=ds, model="m", out_dir=out, passes=2, budget=1, price_in_per_m=1, price_out_per_m=1, key=SECRET, post=api.post, sleep=api.sleep)
        ses = AN.read_session(out)
        pa = AN.analyze(ses["records"], ses["prompts"])["pass_agreement"]
        self.assertEqual((pa["all"]["prompts_with_two_or_more_valid_passes"], pa["all"]["prompts_all_passes_agree"]), (3, 1))
        self.assertIn("NOT a probability", pa["note"])

    def test_analysis_contains_no_probability_margin_or_confidence_quantities(self):
        a, ses = self.run_with(lambda mp, cond: MAPS[mp][0])
        bad = {"probabilities", "probability", "log_odds", "margin", "logprob", "confidence", "residual", "shift", "rank", "leaderboard"}

        def keys(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    yield str(k).lower()
                    yield from keys(v)
            elif isinstance(o, list):
                for v in o:
                    yield from keys(v)

        for k in keys(a):
            self.assertFalse(any(b in k for b in bad), k)
        for r in ses["records"]:
            self.assertEqual(r["measurement_type"], "ANSWER_ONLY")


class TestReplayOfTheSavedPilot(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("GEMINI_API_KEY", None)
        self.net = mock.patch("urllib.request.urlopen", side_effect=AssertionError("network access attempted"))
        self.net.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.net.stop)

    def test_bundle_replays_offline_and_reproduces_the_pilot_tables(self):
        for d, comp_correct, comp_competing, cost in (("flash", 0, 32, 0.0898), ("carbon", 4, 28, 0.0696)):
            code, text = run_cli("gemini", "--replay", os.path.join(BUNDLE, d))
            self.assertEqual(code, 0, text)
            self.assertIn("no key used, no network", text)
            ses = AN.read_session(os.path.join(BUNDLE, d))
            self.assertEqual((ses["problems"], ses["warnings"]), ([], []))
            a = AN.analyze(ses["records"], ses["prompts"])
            self.assertEqual(a["records"], 160)
            for cond in ("BASELINE", "PROSE", "RANDOM_AB", "RANDOM_OTHER"):
                self.assertEqual((a["by_condition"][cond]["correct"], a["by_condition"][cond]["valid"]), (32, 32), cond)
            for cond in ("PROSE", "RANDOM_AB", "RANDOM_OTHER"):
                self.assertEqual((a["versus_baseline"][cond]["answer_changed"], a["versus_baseline"][cond]["induced_incorrect"]), (0, 0), cond)
            self.assertEqual((a["by_condition"]["COMPETING"]["correct"], a["by_condition"]["COMPETING"]["competing"]), (comp_correct, comp_competing))
            self.assertEqual(a["versus_baseline"]["COMPETING"]["induced_to_competing"], comp_competing)
            self.assertEqual(a["pass_agreement"]["all"], {"prompts_with_two_or_more_valid_passes": 80, "prompts_all_passes_agree": 80})
            self.assertAlmostEqual(a["estimated_cost_usd"], cost, places=3)
            self.assertEqual(a["status_counts"], {"ok": 160})

    def test_carbon_resisted_competing_evidence_only_in_the_league_question_under_m1_m2(self):
        ses = AN.read_session(os.path.join(BUNDLE, "carbon"))
        a = AN.analyze(ses["records"], ses["prompts"])
        held = sorted(c for c, v in a["cells"].items() if v["conditions"]["COMPETING"][0]["role"] == "correct")
        self.assertEqual(held, ["q4-league-m1", "q4-league-m2"])

    def test_legacy_pilot_layout_replays_identically_when_the_local_pilot_exists(self):
        v7, v8 = sorted(glob.glob(os.path.join(ROOT, "runs", "v0.7-frontier-pilot-*"))), sorted(glob.glob(os.path.join(ROOT, "runs", "v0.8-gemini-pilot-*")))
        if not (v7 and v8 and os.path.isdir(os.path.join(v8[0], "responses", "g38flash"))):
            self.skipTest("the local git-ignored pilot evidence is not present")
        for tag, d in (("g38flash", "flash"), ("g4carbon", "carbon")):
            legacy = AN.read_session(os.path.join(v8[0], "responses", tag), os.path.join(v7[0], "frozen-requests.json"))
            self.assertEqual(legacy["kind"], "legacy")
            self.assertEqual(legacy["problems"], [])
            new = AN.read_session(os.path.join(BUNDLE, d))
            la, na = AN.analyze(legacy["records"], legacy["prompts"]), AN.analyze(new["records"], new["prompts"])
            for k in ("by_condition", "versus_baseline", "by_condition_mapping", "pass_agreement", "cells", "usage"):
                self.assertEqual(la[k], na[k], k)

    def test_tampered_evidence_is_detected(self):
        d = tmp()
        for tamper in ("answer", "prompt_hash", "logprobs", "measurement_type", "unknown_cell", "duplicate", "dataset"):
            dst = os.path.join(d, tamper)
            shutil.copytree(os.path.join(BUNDLE, "flash"), dst)
            shutil.copy(os.path.join(BUNDLE, "prompts.json"), os.path.join(d, "prompts.json"))
            f = sorted(glob.glob(os.path.join(dst, "records", "*.json")))[0]
            r = read_json(f)
            if tamper == "answer":
                r["answer"] = "A" if r["answer"] != "A" else "B"
            elif tamper == "prompt_hash":
                r["prompt_sha256"] = "0" * 64
            elif tamper == "logprobs":
                r["response"]["candidates"][0]["logprobsResult"] = {"topCandidates": []}
            elif tamper == "measurement_type":
                r["measurement_type"] = "DIRECT_PROBABILITY"
            elif tamper == "unknown_cell":
                r["cell"] = "zz"
            elif tamper == "dataset":
                pr = read_json(os.path.join(d, "prompts.json"))
                pr["items"][0]["prompt"] += " changed"
                write_json(os.path.join(d, "prompts.json"), pr)
            write_json(f, r)
            if tamper == "duplicate":
                r2 = read_json(sorted(glob.glob(os.path.join(dst, "records", "*.json")))[1])
                r2.update({"name": r["name"], "cell": r["cell"], "condition": r["condition"], "pass": r["pass"]})
                write_json(os.path.join(dst, "records", "zz-dup.json"), r2)
            try:
                ses = AN.read_session(dst)
                problems = ses["problems"]
            except AN.AnswersError:
                problems = ["rejected"]
            self.assertTrue(problems, tamper)
            code, text = run_cli("gemini", "--replay", dst)
            self.assertEqual(code, 2, tamper)

    def test_missing_records_make_the_session_incomplete_not_valid(self):
        d = tmp()
        dst = os.path.join(d, "s")
        shutil.copytree(os.path.join(BUNDLE, "flash"), dst)
        shutil.copy(os.path.join(BUNDLE, "prompts.json"), os.path.join(d, "prompts.json"))
        os.remove(sorted(glob.glob(os.path.join(dst, "records", "*.json")))[0])
        ses = AN.read_session(dst)
        self.assertTrue(any("incomplete" in w for w in ses["warnings"]))

    def test_json_out_is_new_and_never_overwritten(self):
        d = tmp()
        out = os.path.join(d, "a.json")
        code, text = run_cli("gemini", "--replay", os.path.join(BUNDLE, "flash"), "--json-out", out)
        self.assertEqual(code, 0, text)
        j = read_json(out)
        self.assertEqual(j["measurement_type"], "ANSWER_ONLY")
        code, text = run_cli("gemini", "--replay", os.path.join(BUNDLE, "flash"), "--json-out", out)
        self.assertEqual(code, 2)
        self.assertIn("already exists", text)

    def test_replay_rejects_live_options(self):
        code, text = run_cli("gemini", "--replay", os.path.join(BUNDLE, "flash"), "--budget", "1")
        self.assertEqual(code, 2)
        self.assertIn("BAD_ARGS", text)


class TestCli(unittest.TestCase):
    def test_live_run_needs_explicit_cap_prices_and_key(self):
        d = tmp()
        ds = make_dataset(os.path.join(d, "p.json"))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            code, text = run_cli("gemini", "--prompts", ds, "--model", "gemini-test", "--out", os.path.join(d, "s"))
            self.assertEqual(code, 2)
            self.assertIn("--budget", text)
            code, text = run_cli("gemini", "--prompts", ds, "--model", "gemini-test", "--out", os.path.join(d, "s"), "--budget", "1", "--price-in", "1", "--price-out", "1")
            self.assertEqual(code, 2)
            self.assertIn("MISSING_API_KEY", text)
            self.assertFalse(os.path.exists(os.path.join(d, "s")))                       # nothing was created without a key

    def test_cli_live_run_with_a_mock_transport_and_no_secret_in_any_output(self):
        d = tmp()
        ds = make_dataset(os.path.join(d, "p.json"))
        api = FakeAPI(correct_answerer)
        out = io.StringIO()
        args = ["gemini", "--prompts", ds, "--model", "models/gemini-test", "--out", os.path.join(d, "s"), "--budget", "1", "--price-in", "1", "--price-out", "10", "--passes", "2", "--json-out", os.path.join(d, "a.json")]
        from bleediff.cli import build_parser
        ns = build_parser().parse_args(args)
        code = GM.cmd_gemini(ns, out=out, post=api.post, sleep=api.sleep, env={"GEMINI_API_KEY": SECRET})
        text = out.getvalue()
        self.assertEqual(code, 0, text)
        self.assertIn("ANSWER_ONLY", text)
        self.assertIn("estimated spend", text)
        self.assertNotIn(SECRET, text + all_text(d))
        self.assertEqual(api.calls, 12)

    def test_unexpected_exceptions_never_echo_the_secret(self):
        d = tmp()
        ds = make_dataset(os.path.join(d, "p.json"))
        api = FakeAPI(lambda p, n: (200, RuntimeError(f"boom {SECRET}")))
        out = io.StringIO()
        from bleediff.cli import build_parser
        ns = build_parser().parse_args(["gemini", "--prompts", ds, "--model", "gemini-test", "--out", os.path.join(d, "s"), "--budget", "1", "--price-in", "1", "--price-out", "1"])
        code = GM.cmd_gemini(ns, out=out, post=api.post, sleep=api.sleep, env={"GEMINI_API_KEY": SECRET})
        self.assertEqual(code, 2)
        self.assertNotIn(SECRET, out.getvalue() + all_text(d))
        ses = read_json(os.path.join(d, "s", "session.json"))
        self.assertEqual(ses["status"], "aborted")

    def test_budget_stop_returns_exit_code_2(self):
        d = tmp()
        ds = make_dataset(os.path.join(d, "p.json"))
        out = io.StringIO()
        from bleediff.cli import build_parser
        ns = build_parser().parse_args(["gemini", "--prompts", ds, "--model", "gemini-test", "--out", os.path.join(d, "s"), "--budget", "0.0001", "--price-in", "1", "--price-out", "10"])
        code = GM.cmd_gemini(ns, out=out, post=FakeAPI(correct_answerer).post, env={"GEMINI_API_KEY": SECRET})
        self.assertEqual(code, 2)
        self.assertIn("BUDGET_EXHAUSTED", out.getvalue())


class TestAtlasIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tmp()
        cls.out = os.path.join(cls.d, "atlas")
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network access attempted")):
            cls.code, cls.text = run_cli("atlas", "--manifest", os.path.join(BUNDLE, "atlas-input.json"), "--out", cls.out)
        cls.atlas = read_json(os.path.join(cls.out, "atlas.json"))

    def test_atlas_succeeds_and_reports_both_measurement_types(self):
        self.assertEqual(self.code, 0, self.text)
        self.assertIn("DIRECT_PROBABILITY", self.text)
        self.assertIn("ANSWER_ONLY hosted sessions", self.text)
        for name in ("atlas.json", "candidates.csv", "category-summary.csv", "model-comparison.csv", "report.md", "answer-only.csv", "answer-level-comparison.csv"):
            self.assertTrue(os.path.isfile(os.path.join(self.out, name)), name)

    def test_every_local_cell_is_direct_probability_and_hosted_sessions_are_answer_only(self):
        self.assertEqual({c["measurement_type"] for c in self.atlas["cells"]}, {"DIRECT_PROBABILITY"})
        ao = self.atlas["answer_only"]
        self.assertEqual(ao["measurement_type"], "ANSWER_ONLY")
        self.assertEqual({s["measurement_type"] for s in ao["sessions"]}, {"ANSWER_ONLY"})
        self.assertEqual({s["model"]["id"] for s in ao["sessions"]}, {"gemini-3.8-flash", "gemini-4-carbon"})
        self.assertEqual(set(self.atlas["measurement_types"]), {"DIRECT_PROBABILITY", "ANSWER_ONLY"})

    def test_no_probability_margin_or_leaderboard_for_answer_only_data(self):
        text = json.dumps(self.atlas["answer_only"]).lower()
        for banned in ('"shift', '"margin', '"probabilit', '"rank', "leaderboard\":", '"p_a"', '"coverage"', "logodds", "log_odds"):
            self.assertNotIn(banned, text, banned)
        rows = csv_rows(os.path.join(self.out, "candidates.csv"))
        self.assertTrue(rows)
        self.assertFalse({r["model_id"] for r in rows} & {"gemini-3.8-flash", "gemini-4-carbon"})   # hosted models are not in the probability table
        for fn in ("model-comparison.csv", "category-summary.csv"):
            self.assertNotIn("gemini", rt(os.path.join(self.out, fn)))
        report = rt(os.path.join(self.out, "report.md"))
        self.assertIn("## Measurement types", report)
        self.assertIn("## Hosted answer-only results (ANSWER_ONLY)", report)
        self.assertIn("never enter the margin, shift, rank or paired-difference sections", " ".join(report.split()))
        self.assertLess(report.index("## Task/model cells"), report.index("## Hosted answer-only results"))

    def test_csvs_label_the_measurement_type_on_every_row(self):
        ao = csv_rows(os.path.join(self.out, "answer-only.csv"))
        self.assertTrue(ao)
        self.assertEqual({r["measurement_type"] for r in ao}, {"ANSWER_ONLY"})
        cmpr = csv_rows(os.path.join(self.out, "answer-level-comparison.csv"))
        self.assertEqual({r["measurement_type"] for r in cmpr}, {"ANSWER_ONLY", "DIRECT_PROBABILITY"})
        direct = [r for r in cmpr if r["measurement_type"] == "DIRECT_PROBABILITY"]
        self.assertEqual({r["source"] for r in direct}, {"qwen1.5b", "llama3b"})
        self.assertTrue(all("probabilities" in r["answer_basis"] for r in direct))
        self.assertTrue(all("returned answer" in r["answer_basis"] for r in cmpr if r["measurement_type"] == "ANSWER_ONLY"))
        comp = {(r["measurement_type"], r["source"]): r for r in cmpr if r["label"] == "COMPETING"}
        self.assertEqual((comp[("ANSWER_ONLY", "gemini-3.8-flash")]["competing"], comp[("ANSWER_ONLY", "gemini-4-carbon")]["competing"]), ("32", "28"))
        self.assertEqual(comp[("DIRECT_PROBABILITY", "llama3b")]["n"], "16")

    def test_existing_probability_calculations_are_unchanged_by_the_answer_only_block(self):
        d = tmp()
        shutil.copytree(BUNDLE, os.path.join(d, "g"))
        shutil.copytree(os.path.join(ROOT, "examples", "four-mapping"), os.path.join(d, "four-mapping"))
        plain = read_json(os.path.join(d, "g", "atlas-input.json"))
        del plain["answer_only_sessions"]
        write_json(os.path.join(d, "g", "plain.json"), plain)
        code, text = run_cli("atlas", "--manifest", os.path.join(d, "g", "plain.json"), "--out", os.path.join(d, "plain"))
        self.assertEqual(code, 0, text)
        base = read_json(os.path.join(d, "plain", "atlas.json"))
        self.assertNotIn("answer_only", base)
        strip = lambda cells: [{k: v for k, v in c.items() if k != "measurement_type"} for c in cells]
        self.assertEqual(strip(self.atlas["cells"]), strip(base["cells"]))
        self.assertEqual(self.atlas["aggregate"], base["aggregate"])
        self.assertEqual(self.atlas["model_comparison"], base["model_comparison"])
        for fn in ("candidates.csv", "category-summary.csv", "model-comparison.csv"):
            self.assertEqual(rb(os.path.join(self.out, fn)), rb(os.path.join(d, "plain", fn)), fn)
        self.assertEqual(sorted(os.listdir(os.path.join(d, "plain"))), ["atlas.json", "candidates.csv", "category-summary.csv", "model-comparison.csv", "report.md"])

    def test_invalid_hosted_session_fails_closed_unless_keep_going(self):
        d = tmp()
        shutil.copytree(BUNDLE, os.path.join(d, "g"))
        shutil.copytree(os.path.join(ROOT, "examples", "four-mapping"), os.path.join(d, "four-mapping"))
        f = sorted(glob.glob(os.path.join(d, "g", "flash", "records", "*.json")))[0]
        r = read_json(f)
        r["answer"] = "A" if r["answer"] != "A" else "B"
        write_json(f, r)
        mp = os.path.join(d, "g", "atlas-input.json")
        code, text = run_cli("atlas", "--manifest", mp, "--out", os.path.join(d, "o1"))
        self.assertEqual(code, 2)
        self.assertIn("failed validation", text)
        self.assertFalse(os.path.exists(os.path.join(d, "o1", "atlas.json")))
        code, text = run_cli("atlas", "--manifest", mp, "--out", os.path.join(d, "o2"), "--keep-going")
        self.assertEqual(code, 0, text)
        a = read_json(os.path.join(d, "o2", "atlas.json"))
        self.assertEqual([s["model"]["id"] for s in a["answer_only"]["sessions"]], ["gemini-4-carbon"])
        self.assertEqual(len(a["answer_only"]["excluded"]), 1)

    def test_malformed_answer_only_manifest_entries(self):
        d = tmp()
        shutil.copytree(BUNDLE, os.path.join(d, "g"))
        shutil.copytree(os.path.join(ROOT, "examples", "four-mapping"), os.path.join(d, "four-mapping"))
        man = read_json(os.path.join(d, "g", "atlas-input.json"))
        for bad in ([], [{"model": {"id": "x"}}], [{"session_dir": "flash"}], [man["answer_only_sessions"][0], man["answer_only_sessions"][0]], "x"):
            m = copy.deepcopy(man)
            m["answer_only_sessions"] = bad
            write_json(os.path.join(d, "g", "bad.json"), m)
            code, text = run_cli("atlas", "--manifest", os.path.join(d, "g", "bad.json"), "--out", os.path.join(d, "o-" + str(abs(hash(str(bad))))))
            self.assertEqual(code, 2, str(bad))

    def test_model_filter_applies_to_the_hosted_sections(self):
        d = tmp()
        code, text = run_cli("atlas", "--manifest", os.path.join(BUNDLE, "atlas-input.json"), "--out", os.path.join(d, "f"), "--model", "gemini-4-carbon")
        self.assertEqual(code, 0, text)
        rows = csv_rows(os.path.join(d, "f", "answer-only.csv"))
        self.assertEqual({r["model_id"] for r in rows}, {"gemini-4-carbon"})
        self.assertEqual(csv_rows(os.path.join(d, "f", "candidates.csv")), [])
        code, text = run_cli("atlas", "--manifest", os.path.join(BUNDLE, "atlas-input.json"), "--out", os.path.join(d, "g"), "--model", "llama3b")
        self.assertEqual(code, 0, text)
        self.assertEqual(csv_rows(os.path.join(d, "g", "answer-only.csv")), [])
        self.assertEqual(read_json(os.path.join(d, "g", "atlas.json"))["filters_applied_to_csv_report_and_terminal"]["model"], ["llama3b"])


class TestBundleHygiene(unittest.TestCase):
    def test_bundle_is_small_lf_text_with_no_secrets_or_local_data(self):
        files = [os.path.join(r, f) for r, _d, fs in os.walk(BUNDLE) for f in fs]
        self.assertLess(sum(os.path.getsize(p) for p in files), 2 * 1024 * 1024)
        import re
        pat = re.compile(r"(AIza[0-9A-Za-z_-]{10,}|/Users/|/home/|AppData|huggingface|\.cache|@[A-Za-z0-9-]+\.(com|org|net|io)|api[_-]?key\s*[:=])", re.I)
        for p in files:
            data = rb(p)
            self.assertNotIn(b"\r\n", data, p)
            self.assertIsNone(pat.search(data.decode("utf-8")), p)
        self.assertIn("examples/gemini-answer-only/** -text", rt(os.path.join(ROOT, ".gitattributes")))

    def test_sha256sums_cover_the_bundle(self):
        import hashlib
        listed = {}
        for line in rt(os.path.join(BUNDLE, "SHA256SUMS.txt")).splitlines():
            digest, rel = line.split("  ", 1)
            listed[rel] = digest
        found = {os.path.relpath(os.path.join(r, f), BUNDLE).replace("\\", "/") for r, _d, fs in os.walk(BUNDLE) for f in fs} - {"SHA256SUMS.txt", "README.md"}
        self.assertEqual(set(listed), found)
        for rel, digest in listed.items():
            self.assertEqual(hashlib.sha256(rb(os.path.join(BUNDLE, rel))).hexdigest(), digest, rel)


if __name__ == "__main__":
    unittest.main()
