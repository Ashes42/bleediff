"""Behavioral telemetry, v0.7.0 (bleediff.telemetry, bleediff.behavior, ``bleediff behavior``). All offline: no key, no network, no model.

Covers the five measurement states (legitimate zeros, missing, unsupported, inconsistent, invalid), the adapters, ingestion (including malformed evidence), condition summaries,
explicitly paired contrasts, the CLI, backward compatibility, and a real-world validation fixture: the published Research 002 evidence (skipped, with a stated reason, when the
published package is not next to this repository; point BLEEDIFF_R002_PUBLICATION at it otherwise). Research 002 files are only ever READ here.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import math
import os
import socket
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bleediff import answers as AN  # noqa: E402
from bleediff import behavior as B  # noqa: E402
from bleediff import telemetry as T  # noqa: E402
from bleediff.cli import main  # noqa: E402

with open(os.path.join(ROOT, "tests", "data", "anthropic-probe-haiku-5-5.json"), encoding="utf-8") as _fh:
    PROBE = json.load(_fh)["response"]     # a REAL Messages response (signature and id removed)
ABSENT = object()


# --------------------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------------------
def response(text="A", inp=100, out=3, thinking=0, stop="end_turn", block=None, details=ABSENT, usage=ABSENT, cache=(0, 0)):
    """An Anthropic Messages response. ``thinking`` None/ABSENT-style knobs let tests build every telemetry shape; ``block`` forces a thinking block (default: iff thinking > 0)."""
    if block is None:
        block = isinstance(thinking, int) and not isinstance(thinking, bool) and thinking > 0
    content = ([{"type": "thinking", "thinking": "", "signature": "[REDACTED:opaque-thinking-signature]"}] if block else []) + ([{"type": "text", "text": text}] if text is not None else [])
    r = {"model": "claude-haiku-5-5", "id": "msg_REDACTED", "type": "message", "role": "assistant", "content": content, "stop_reason": stop}
    if usage is ABSENT:
        u = {"input_tokens": inp, "cache_creation_input_tokens": cache[1], "cache_read_input_tokens": cache[0], "output_tokens": out}
        u["output_tokens_details"] = {"thinking_tokens": thinking} if details is ABSENT else details        # details=None gives an explicit null
        r["usage"] = u
    elif usage is not None:
        r["usage"] = usage
    return r


def record(cond="CONTRA", cell="F01D-v1-m1", pas=1, resp=ABSENT, session_fields=True, **kw):
    r = response() if resp is ABSENT else resp
    parsed = AN.parse_anthropic_answer(r) if isinstance(r, dict) else {"status": "error", "answer": None, "text": None, "finish_reason": None, "block_reason": None}
    rec = {"schema": AN.RECORD_SCHEMA, "measurement_type": "ANSWER_ONLY", "provider": "anthropic", "name": f"{cell}__{cond}__pass{pas}", "cell": cell, "question": cell.split("-")[0],
           "mapping": "m1", "condition": cond, "pass": pas, "prompt_sha256": hashlib.sha256(f"{cell}{cond}".encode()).hexdigest(), "correct_letter": "A", "competing_letter": "C",
           "model": "claude-haiku-5-5", "request": {"parameters": {"model": "claude-haiku-5-5", "max_tokens": 1024}}, "status": parsed["status"], "answer": parsed["answer"],
           "text": parsed["text"], "finish_reason": parsed["finish_reason"], "block_reason": parsed["block_reason"], "http_status": 200, "attempts": 1, "elapsed_s": 1.25,
           "usage": AN.usage_from_anthropic(r) if isinstance(r, dict) else {}, "usage_complete": True, "cost_usd": 3.0e-05, "cost_basis": "usage_metadata", "response": r}
    rec.update(kw)
    return rec


def obs(**kw):
    return T.observe_record(record(**kw), session="S")


def m(o, name):
    return o["measures"][name]


class Tmp:
    def __enter__(self):
        self._d = tempfile.TemporaryDirectory()
        return self._d.name

    def __exit__(self, *a):
        self._d.cleanup()


def write_session(d, records, name="sess", meta=True):
    sdir = os.path.join(d, name)
    os.makedirs(os.path.join(sdir, "records"))
    for r in records:
        with open(os.path.join(sdir, "records", r["name"] + ".json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(r, fh)
    if meta:
        with open(os.path.join(sdir, "session.json"), "w", encoding="utf-8") as fh:
            json.dump({"schema": AN.SESSION_SCHEMA, "measurement_type": "ANSWER_ONLY", "provider": "anthropic", "model": "claude-haiku-5-5", "run_id": name + "-run"}, fh)
    return sdir


def tree_hash(path):
    h = hashlib.sha256()
    for r, ds, fs in os.walk(path):
        ds.sort()
        for f in sorted(fs):
            p = os.path.join(r, f)
            with open(p, "rb") as fh:
                h.update(os.path.relpath(p, path).encode() + b"\0" + hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def run_main(*argv):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch("sys.stdout", out), mock.patch("sys.stderr", err):
        code = main(list(argv))
    return code, out.getvalue() + err.getvalue()


# --------------------------------------------------------------------------------------
# the five states
# --------------------------------------------------------------------------------------
class TestObservedZero(unittest.TestCase):
    def test_zero_thinking_tokens_is_observed_zero_not_missing(self):
        o = obs(resp=response(thinking=0))
        self.assertEqual((m(o, "thinking_tokens")["state"], m(o, "thinking_tokens")["value"]), (T.OBSERVED, 0))
        self.assertEqual((m(o, "thinking_activation")["state"], m(o, "thinking_activation")["value"]), (T.OBSERVED, False))
        self.assertEqual((m(o, "non_thinking_output_tokens")["state"], m(o, "non_thinking_output_tokens")["value"]), (T.OBSERVED, 3))
        self.assertEqual(m(o, "cache_read_tokens")["value"], 0)          # a reported zero cache count is also an observed zero
        self.assertEqual(T.validate_observation(o), [])

    def test_positive_thinking_activates(self):
        o = obs(resp=response(out=60, thinking=57))
        self.assertEqual((m(o, "thinking_activation")["value"], m(o, "non_thinking_output_tokens")["value"]), (True, 3))
        self.assertEqual(o["context"]["thinking_corroboration"], "agrees")

    def test_real_response_with_all_output_spent_on_thinking(self):
        rec = {"cell": "c", "condition": "X", "pass": 1, "provider": "anthropic", "response": PROBE, "correct_letter": "A", "competing_letter": "B", "status": "empty"}
        o = T.observe_record(rec)
        self.assertEqual([m(o, k)["value"] for k in ("input_tokens", "output_tokens", "thinking_tokens", "non_thinking_output_tokens")], [228, 8, 8, 0])
        self.assertEqual((o["answer_status"], m(o, "answer_outcome")["value"], m(o, "format_compliant")["value"], o["context"]["output_cap_hit"]), ("empty", T.NO_ANSWER, False, True))
        self.assertEqual(m(o, "latency_s")["state"], T.MISSING)           # this saved probe response carries no harness latency or cost
        self.assertEqual(m(o, "cost_usd")["state"], T.MISSING)


class TestMissingTelemetry(unittest.TestCase):
    def test_absent_thinking_telemetry_is_never_zero(self):
        for details, why in ((ABSENT, None), (None, "output_tokens_details_null"), ({}, "field_absent"), ({"thinking_tokens": None}, "field_null")):
            r = response(out=10)
            if details is ABSENT:
                del r["usage"]["output_tokens_details"]
                why = "output_tokens_details_absent"
            else:
                r["usage"]["output_tokens_details"] = details
            o = obs(resp=r, usage={})        # the harness stored no usage: cross-check not in play
            th = m(o, "thinking_tokens")
            self.assertEqual((th["state"], th["value"], th["reason"]), (T.MISSING, None, why))
            self.assertEqual((m(o, "thinking_activation")["state"], m(o, "thinking_activation")["value"]), (T.MISSING, None), details)
            self.assertEqual(m(o, "non_thinking_output_tokens")["state"], T.MISSING)
            self.assertEqual((m(o, "output_tokens")["state"], m(o, "output_tokens")["value"]), (T.OBSERVED, 10))     # the other fields stay observed
            self.assertEqual(T.validate_observation(o), [])

    def test_no_usage_and_no_response(self):
        o = obs(resp=response(usage=None), usage={})
        for k in ("input_tokens", "output_tokens", "thinking_tokens", "cache_read_tokens"):
            self.assertEqual((m(o, k)["state"], m(o, k)["reason"]), (T.MISSING, "usage_absent"), k)
        e = obs(resp=None, status="error", answer=None, usage={}, cost_usd=None, error={"reason": "HTTP_500"})
        self.assertEqual((e["answer_status"], m(e, "thinking_tokens")["state"], m(e, "thinking_tokens")["reason"]), ("error", T.MISSING, "response_absent"))
        self.assertEqual((m(e, "answer_outcome")["state"], m(e, "refusal")["state"], m(e, "cost_usd")["state"]), (T.MISSING, T.MISSING, T.MISSING))

    def test_missing_latency_and_scoring_key(self):
        rec = record()
        del rec["elapsed_s"]
        del rec["correct_letter"]
        o = T.observe_record(rec)
        self.assertEqual((m(o, "latency_s")["state"], m(o, "latency_s")["reason"]), (T.MISSING, "field_absent"))
        self.assertEqual((m(o, "answer_outcome")["state"], m(o, "answer_role")["state"], m(o, "answer_role")["reason"]), (T.OBSERVED, T.MISSING, "scoring_key_absent"))

    def test_legitimate_zero_latency_is_observed(self):
        self.assertEqual((m(obs(elapsed_s=0.0), "latency_s")["state"], m(obs(elapsed_s=0.0), "latency_s")["value"]), (T.OBSERVED, 0.0))


class TestInvalidTelemetry(unittest.TestCase):
    def test_wrong_typed_thinking_tokens_are_invalid_and_keep_the_reported_value(self):
        for bad, reason in (("5", "not_an_integer"), (True, "not_an_integer"), (2.5, "not_an_integer"), (-3, "negative"), ([1], "not_an_integer")):
            o = obs(resp=response(details={"thinking_tokens": bad}), usage={})
            th = m(o, "thinking_tokens")
            self.assertEqual((th["state"], th["value"], th["reason"]), (T.INVALID, None, reason), bad)
            self.assertEqual(th["reported"], T.safe(bad))
            self.assertEqual(m(o, "thinking_activation")["state"], T.INVALID)
            self.assertEqual(T.validate_observation(o), [])
            json.dumps(o, allow_nan=False)       # the observation stays strict-JSON serialisable

    def test_structure_of_the_wrong_type(self):
        self.assertEqual(m(obs(resp=response(details=[1, 2]), usage={}), "thinking_tokens")["reason"], "output_tokens_details_not_an_object")
        self.assertEqual((m(obs(resp=response(usage="x"), usage={}), "input_tokens")["state"], m(obs(resp=response(usage="x"), usage={}), "input_tokens")["reason"]), (T.INVALID, "usage_not_an_object"))
        o = obs(resp="not an object", usage={}, status="malformed")
        self.assertEqual((o["answer_status"], m(o, "output_tokens")["state"], m(o, "answer_outcome")["state"]), ("malformed", T.INVALID, T.INVALID))

    def test_invalid_latency_and_cost(self):
        for bad in (True, "1.2", -0.5, float("nan"), float("inf"), [1]):
            self.assertEqual(m(obs(elapsed_s=bad), "latency_s")["state"], T.INVALID, bad)
            self.assertEqual(m(obs(cost_usd=bad), "cost_usd")["state"], T.INVALID, bad)
        o = obs(elapsed_s=float("nan"))
        json.dumps(o, allow_nan=False)           # NaN is reported as a string, never emitted as a JSON NaN


class TestInconsistentTelemetry(unittest.TestCase):
    def test_thinking_above_output_is_inconsistent_for_both_and_for_the_derived_values(self):
        o = obs(resp=response(out=5, thinking=9), usage={})
        for k in ("thinking_tokens", "output_tokens", "thinking_activation", "non_thinking_output_tokens"):
            self.assertEqual((m(o, k)["state"], m(o, k)["value"]), (T.INCONSISTENT, None), k)
        self.assertEqual((m(o, "thinking_tokens")["reported"], m(o, "output_tokens")["reported"]), (9, 5))
        self.assertEqual(m(o, "input_tokens")["state"], T.OBSERVED)

    def test_output_above_max_tokens_is_inconsistent(self):
        o = obs(resp=response(out=2000, thinking=1900), usage={})
        self.assertEqual((m(o, "output_tokens")["state"], m(o, "output_tokens")["reason"]), (T.INCONSISTENT, "output_exceeds_max_tokens"))

    def test_stored_usage_that_disagrees_with_the_saved_response(self):
        r = response(out=40, thinking=37)
        stored = AN.usage_from_anthropic(r)
        stored["thoughtsTokenCount"] = 0
        o = obs(resp=r, usage=stored)
        self.assertEqual(m(o, "thinking_tokens")["state"], T.INCONSISTENT)
        self.assertIn("stored_usage_disagrees_with_response:thoughtsTokenCount", o["problems"])
        stored2 = AN.usage_from_anthropic(r)
        stored2["promptTokenCount"] = 1
        self.assertEqual(m(obs(resp=r, usage=stored2), "input_tokens")["state"], T.INCONSISTENT)

    def test_stored_answer_that_disagrees_with_the_saved_response(self):
        o = obs(status="ok", answer="B")                       # the response says A
        for k in ("answer_outcome", "answer_role", "format_compliant", "refusal"):
            self.assertEqual(m(o, k)["state"], T.INCONSISTENT, k)
        self.assertIn("stored_answer_disagrees_with_response", o["problems"])

    def test_block_and_counter_disagreement_is_flagged_not_excluded(self):
        o1 = obs(resp=response(out=60, thinking=57, block=False), usage={})
        self.assertEqual((o1["context"]["thinking_corroboration"], m(o1, "thinking_tokens")["state"]), ("counter_positive_no_thinking_block", T.OBSERVED))
        o2 = obs(resp=response(thinking=0, block=True), usage={})
        self.assertEqual((o2["context"]["thinking_corroboration"], m(o2, "thinking_activation")["value"]), ("counter_zero_with_thinking_block", False))


class TestUnsupportedTelemetry(unittest.TestCase):
    def _openai(self, reasoning=None):
        det = {"reasoning_tokens": reasoning} if reasoning is not None else None
        resp = {"choices": [{"finish_reason": "stop", "message": {"content": "A"}}], "usage": {"prompt_tokens": 90, "completion_tokens": 12, "total_tokens": 102, "completion_tokens_details": det}}
        return {"cell": "c", "condition": "X", "pass": 1, "provider": "openai", "response": resp, "correct_letter": "A", "competing_letter": "B", "model": "gpt-test"}

    def test_openai_reasoning_is_unsupported_but_answer_and_usage_are_observed(self):
        o = T.observe_record(self._openai(reasoning=7))
        self.assertEqual((m(o, "thinking_tokens")["state"], m(o, "thinking_activation")["state"], m(o, "non_thinking_output_tokens")["state"]), (T.UNSUPPORTED,) * 3)
        self.assertEqual((m(o, "input_tokens")["value"], m(o, "output_tokens")["value"], m(o, "answer_role")["value"], m(o, "format_compliant")["value"]), (90, 12, "correct", True))
        self.assertEqual(o["provider_fields"], {"usage.completion_tokens_details.reasoning_tokens": 7})     # the raw field is preserved, not interpreted
        self.assertEqual(T.validate_observation(o), [])
        self.assertEqual(m(T.observe_record(self._openai()), "thinking_tokens")["state"], T.UNSUPPORTED)      # even with no reasoning field: unsupported, not zero

    def test_gemini_is_answer_and_input_only(self):
        resp = {"candidates": [{"content": {"parts": [{"text": "A"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 77, "candidatesTokenCount": 1, "thoughtsTokenCount": 40, "totalTokenCount": 118}}
        o = T.observe_record({"cell": "c", "condition": "X", "pass": 1, "provider": "gemini", "response": resp, "correct_letter": "A", "competing_letter": "B"})
        self.assertEqual((m(o, "input_tokens")["value"], m(o, "output_tokens")["state"], m(o, "thinking_tokens")["state"], m(o, "answer_outcome")["value"]), (77, T.UNSUPPORTED, T.UNSUPPORTED, "A"))
        self.assertEqual(o["provider_fields"]["usageMetadata.thoughtsTokenCount"], 40)

    def test_unknown_provider_and_legacy_latency(self):
        o = T.observe_record({"cell": "c", "condition": "X", "pass": 1, "provider": "acme", "response": {"x": 1}})
        self.assertEqual({st["state"] for k, st in o["measures"].items() if k not in ("latency_s", "cost_usd")}, {T.UNSUPPORTED})
        self.assertEqual(o["adapter"]["id"], "none")
        leg = T.observe_record(record(), layout="legacy")
        self.assertEqual(m(leg, "latency_s")["state"], T.UNSUPPORTED)

    def test_adapters_only_where_parsers_exist(self):
        self.assertEqual(set(T.ADAPTERS), set(AN.PARSERS))
        self.assertEqual([k for k, a in T.ADAPTERS.items() if a.thinking], ["anthropic"])


class TestAnswerFormatRefusal(unittest.TestCase):
    def test_answer_roles_and_format(self):
        cases = [(response(text="A"), "A", "correct", True), (response(text="C"), "C", "competing", True), (response(text="D"), "D", "other_incorrect", True),
                 (response(text="The answer is A"), T.NO_ANSWER, T.NO_ANSWER, False), (response(text=None, out=8, thinking=8, stop="max_tokens"), T.NO_ANSWER, T.NO_ANSWER, False)]
        for r, outcome, role, fmt in cases:
            o = obs(resp=r, usage={})
            self.assertEqual((m(o, "answer_outcome")["value"], m(o, "answer_role")["value"], m(o, "format_compliant")["value"], m(o, "refusal")["value"]), (outcome, role, fmt, False))

    def test_refusal_is_a_status_not_a_format_failure_or_a_missing_value(self):
        o = obs(resp=response(text=None, out=0, thinking=0, stop="refusal"), usage={})
        self.assertEqual((o["answer_status"], m(o, "refusal")["value"], m(o, "answer_outcome")["value"], m(o, "thinking_tokens")["value"]), ("blocked", True, T.NO_ANSWER, 0))


class TestCost(unittest.TestCase):
    def test_recorded_cost_and_failed_attempt_flag(self):
        o = obs(cost_usd=1.5e-05, cost_basis="usage_metadata+1_failed_attempt(s)_at_worst_case_bound")
        self.assertEqual((m(o, "cost_usd")["state"], m(o, "cost_usd")["value"], m(o, "cost_usd")["includes_worst_case_charge_for_failed_attempts"]), (T.OBSERVED, 1.5e-05, True))

    def test_upper_bound_substitute_is_not_an_observed_spend(self):
        o = obs(cost_usd=0.5, cost_basis="upper_bound_missing_or_inconsistent_usage")
        c = m(o, "cost_usd")
        self.assertEqual((c["state"], c["value"], c["reported"], c["reason"]), (T.MISSING, None, 0.5, "usage_missing_harness_charged_upper_bound"))

    def test_prices_derive_cost_only_where_the_record_has_none(self):
        rec = record()
        del rec["cost_usd"]
        o = T.observe_record(rec, prices=(1.0, 5.0))
        self.assertAlmostEqual(m(o, "cost_usd")["value"], 100 * 1.0 / 1e6 + 3 * 5.0 / 1e6)
        self.assertEqual(T.observe_record(rec)["measures"]["cost_usd"]["state"], T.MISSING)
        self.assertEqual(T.observe_record(record(), prices=(1.0, 5.0))["measures"]["cost_usd"]["value"], 3.0e-05)       # a recorded cost always wins
        rec2 = record(resp=response(usage=None), usage={})
        del rec2["cost_usd"]
        self.assertEqual(T.observe_record(rec2, prices=(1.0, 5.0))["measures"]["cost_usd"]["state"], T.MISSING)         # unknown usage cannot be priced


class TestValidation(unittest.TestCase):
    def test_validate_observation_rules(self):
        good = obs()
        self.assertEqual(T.validate_observation(good), [])
        bad = copy.deepcopy(good)
        bad["measures"]["thinking_tokens"]["value"] = None
        self.assertTrue(any("OBSERVED with an invalid" in p for p in T.validate_observation(bad)))
        bad = copy.deepcopy(good)
        bad["measures"]["thinking_tokens"]["state"] = T.MISSING
        self.assertTrue(any("carries a value" in p for p in T.validate_observation(bad)))
        bad = copy.deepcopy(good)
        bad["measures"]["answer_role"]["value"] = "great"
        self.assertTrue(T.validate_observation(bad))
        self.assertTrue(T.validate_observation({"schema": "x"}))
        self.assertTrue(T.validate_observation(None))

    def test_no_absent_input_ever_becomes_a_number(self):
        shapes = [response(usage=None), response(details=None), response(details={}), response(details={"thinking_tokens": None}), response(usage="u"), "junk", None]
        for r in shapes:
            o = obs(resp=r, usage={}, status="error" if r is None else None)
            for k in ("thinking_tokens", "thinking_activation", "non_thinking_output_tokens"):
                self.assertIsNone(m(o, k)["value"], (k, r))
                self.assertNotEqual(m(o, k)["state"], T.OBSERVED)

    def test_observations_carry_no_probability_like_fields(self):
        blob = json.dumps(obs()).lower()
        for k in ("logprob", "probabilit", "confidence", "composite", "score"):
            self.assertNotIn(k, blob)

    def test_bad_records_are_rejected_with_a_reason(self):
        with self.assertRaises(T.TelemetryError) as e:
            T.observe_record([1, 2])
        self.assertEqual(e.exception.reason, "RECORD_NOT_AN_OBJECT")
        with self.assertRaises(T.TelemetryError) as e:
            T.observe_record({"cell": "c", "condition": "X"})
        self.assertEqual(e.exception.reason, "RECORD_IDENTITY_MISSING")


# --------------------------------------------------------------------------------------
# ingestion
# --------------------------------------------------------------------------------------
class TestIngestion(unittest.TestCase):
    def test_ingest_reads_only_and_records_provenance(self):
        with Tmp() as d:
            recs = [record(cond=c, cell=f"F0{i}D-v1-m1") for i in range(1, 4) for c in ("CONTRA", "CONSIST")]
            sdir = write_session(d, recs)
            before = tree_hash(sdir)
            src = B.load_evidence(sdir)
            self.assertEqual(tree_hash(sdir), before)                              # the evidence is untouched
            self.assertEqual((len(src["observations"]), src["problems"], src["kind"], src["session"]), (6, [], "session", "sess-run"))
            o = src["observations"][0]
            fp = os.path.join(sdir, "records", os.path.basename(o["evidence"]["file"]))
            self.assertEqual(o["evidence"]["sha256"], hashlib.sha256(open(fp, "rb").read()).hexdigest())
            self.assertNotIn(os.path.basename(d), json.dumps(src["observations"]))     # no absolute path leaks into an observation
            self.assertEqual(B.load_evidence(sdir)["digest"], src["digest"])      # deterministic

    def test_malformed_files_are_listed_not_dropped_and_do_not_stop_the_rest(self):
        with Tmp() as d:
            sdir = write_session(d, [record(cell="F01D-v1-m1"), record(cell="F02D-v1-m1")])
            rd = os.path.join(sdir, "records")
            open(os.path.join(rd, "broken.json"), "w").write("{not json")
            open(os.path.join(rd, "list.json"), "w").write("[1,2]")
            noid = record(cell="F03D-v1-m1")
            del noid["cell"]
            open(os.path.join(rd, "noid.json"), "w").write(json.dumps(noid))
            dup = record(cell="F01D-v1-m1")
            dup["name"] = "dup-of-first"
            open(os.path.join(rd, "dup.json"), "w").write(json.dumps(dup))
            src = B.load_evidence(sdir)
            self.assertEqual(len(src["observations"]), 2)
            self.assertEqual(sorted(p["code"] for p in src["problems"]), ["duplicate_record", "invalid_json", "record_identity_missing", "record_not_an_object"])
            self.assertEqual(src["files"], 6)

    def test_unusable_sources(self):
        with Tmp() as d:
            with self.assertRaises(B.BehaviorError):
                B.load_evidence(os.path.join(d, "nope"))
            os.makedirs(os.path.join(d, "empty"))
            with self.assertRaises(B.BehaviorError):
                B.load_evidence(os.path.join(d, "empty"))
            legacy = os.path.join(d, "legacy")
            os.makedirs(legacy)
            open(os.path.join(legacy, "c__X__pass1.json"), "w").write("{}")
            with self.assertRaises(B.BehaviorError) as e:       # the legacy layout carries no scoring key
                B.load_evidence(legacy)
            self.assertIn("--prompts", str(e.exception))

    def test_session_without_prompts_json_scores_from_the_record(self):
        with Tmp() as d:
            src = B.load_evidence(write_session(d, [record()]))
            self.assertEqual(m(src["observations"][0], "answer_role")["value"], "correct")

    def test_observation_file_roundtrip_and_overwrite_protection(self):
        with Tmp() as d:
            src = B.load_evidence(write_session(d, [record(cond=c) for c in ("CONTRA", "CONSIST")]))
            path = os.path.join(d, "obs.jsonl")
            self.assertEqual(B.write_observations(path, [src]), 2)
            back = B.read_observations(path)
            self.assertEqual(back["observations"], src["observations"])
            self.assertEqual(back["meta"]["schema"], B.OBSERVATIONS_FILE_SCHEMA)
            with self.assertRaises(B.BehaviorError):
                B.write_observations(path, [src])
            with open(path, "a") as fh:
                fh.write("garbage\n" + json.dumps({"schema": T.OBSERVATION_SCHEMA}) + "\n")
            again = B.read_observations(path)
            self.assertEqual(len(again["observations"]), 2)
            self.assertEqual(sorted(p["code"] for p in again["problems"]), ["invalid_json", "invalid_observation"])


# --------------------------------------------------------------------------------------
# summaries and paired contrasts
# --------------------------------------------------------------------------------------
def mixed_records():
    """CONTRA: thinking 0 / 50 / MISSING / INVALID.  CONSIST: all observed zero.  Units: three questions, one cell each, plus one unbalanced extra."""
    recs = [record(cond="CONTRA", cell="F01D-v1-m1", resp=response(out=3, thinking=0)),
            record(cond="CONTRA", cell="F02D-v1-m1", resp=response(out=53, thinking=50)),
            record(cond="CONTRA", cell="F03D-v1-m1", resp=response(details=None), usage={}),
            record(cond="CONTRA", cell="F04D-v1-m1", resp=response(details={"thinking_tokens": "x"}), usage={})]
    recs += [record(cond="CONSIST", cell=f"F0{i}D-v1-m1", resp=response(out=3, thinking=0)) for i in range(1, 5)]
    return recs


class TestSummaries(unittest.TestCase):
    def _report(self, recs, **kw):
        with Tmp() as d:
            return B.build_report([B.load_evidence(write_session(d, recs))], **kw)

    def test_excluded_records_are_not_counted_as_zero_in_rates_or_means(self):
        g = self._report(mixed_records())["scopes"]["pooled"]["conditions"]["CONTRA"]
        act = g["thinking"]["activation"]
        self.assertEqual((act["observed"], act["activated"], act["rate"], act["excluded"]), (2, 1, 0.5, 2))       # not 1/4
        self.assertEqual((g["thinking"]["thinking_tokens"]["n"], g["thinking"]["thinking_tokens"]["mean"]), (2, 25.0))   # not 50/4
        self.assertEqual(g["states"]["thinking_tokens"], {"OBSERVED": 2, "MISSING": 1, "UNSUPPORTED": 0, "INCONSISTENT": 0, "INVALID": 1})
        self.assertEqual(g["unit_averaged"]["thinking_activation"]["units_without_observed"], 2)
        self.assertEqual(self._report(mixed_records())["scopes"]["pooled"]["conditions"]["CONSIST"]["states"]["thinking_tokens"]["MISSING"], 0)

    def test_inconsistent_and_unsupported_records_never_enter_an_aggregate(self):
        recs = [record(cond="CONTRA", cell="F01D-v1-m1", resp=response(out=53, thinking=50)),
                record(cond="CONTRA", cell="F02D-v1-m1", resp=response(out=5, thinking=9), usage={}),            # thinking above output: INCONSISTENT
                record(cond="CONTRA", cell="F03D-v1-m1", resp=response(out=2000, thinking=1900), usage={})]      # output above max_tokens: INCONSISTENT output
        g = self._report(recs)["scopes"]["pooled"]["conditions"]["CONTRA"]
        self.assertEqual(g["states"]["thinking_tokens"]["INCONSISTENT"], 1)
        self.assertEqual(g["states"]["output_tokens"]["INCONSISTENT"], 2)
        self.assertEqual((g["thinking"]["thinking_tokens"]["n"], g["thinking"]["thinking_tokens"]["sum"]), (2, 1950.0))        # F02 (9) is excluded
        self.assertEqual(g["resources"]["output_tokens"]["n"], 1)
        self.assertEqual(g["integrity"]["records_with_problems"], 0)

    def test_all_missing_has_no_rate_at_all(self):
        recs = [record(cond="CONTRA", cell=f"F0{i}D-v1-m1", resp=response(details=None), usage={}) for i in range(1, 4)]
        g = self._report(recs)["scopes"]["pooled"]["conditions"]["CONTRA"]
        self.assertEqual((g["thinking"]["activation"]["rate"], g["thinking"]["thinking_tokens"]["mean"], g["thinking"]["thinking_tokens"]["n"]), (None, None, 0))

    def test_four_blocks_are_separate_and_there_is_no_composite(self):
        g = self._report(mixed_records())["scopes"]["pooled"]["conditions"]["CONTRA"]
        self.assertEqual(set(g) - {"units", "states", "unit_averaged", "integrity"}, {"answer_behavior", "thinking", "resources", "format_and_refusal"})
        blob = json.dumps(self._report(mixed_records())).lower()
        for bad in ("overall_score", "behavioral_score", "combined_score", "composite_score", "probabilit", "confidence"):
            self.assertNotIn(bad, blob)
        self.assertIn("no interval, test or composite score", self._report(mixed_records())["note"].lower())

    def test_repeated_prompts_are_counted_as_one_prompt_and_two_requests(self):
        with Tmp() as d:
            a = write_session(d, [record(cond="CONTRA", cell=f"F0{i}D-v1-m1") for i in range(1, 4)], name="A")
            b = write_session(d, [record(cond="CONTRA", cell=f"F0{i}D-v1-m1") for i in range(1, 4)], name="B")
            rep = B.build_report([B.load_evidence(a), B.load_evidence(b)])
            g = rep["scopes"]["pooled"]["conditions"]["CONTRA"]["units"]
            self.assertEqual((g["requests"], g["distinct_prompts"], g["distinct_units"]), (6, 3, 3))
            self.assertEqual(sorted(rep["scopes"]), ["pooled", "session:A-run", "session:B-run"])

    def test_a_session_scope_lists_only_the_conditions_it_contains(self):
        with Tmp() as d:
            a = write_session(d, [record(cond="X"), record(cond="Y")], name="A")
            b = write_session(d, [record(cond="X")], name="B")
            rep = B.build_report([B.load_evidence(a), B.load_evidence(b)], contrasts=[("X", "Y")])
        self.assertEqual(list(rep["scopes"]["session:B-run"]["conditions"]), ["X"])
        self.assertEqual(list(rep["scopes"]["pooled"]["conditions"]), ["X", "Y"])
        self.assertIn("NO PAIRS", rep["scopes"]["session:B-run"]["contrasts"][0]["verdict"])

    def test_unit_average_differs_from_pooled_when_units_are_unbalanced(self):
        recs = [record(cond="CONTRA", cell="F01D-v1-m1", resp=response(out=53, thinking=50)),
                record(cond="CONTRA", cell="F02D-v1-m1", resp=response(out=3, thinking=0)), record(cond="CONTRA", cell="F02D-v1-m2", resp=response(out=3, thinking=0)),
                record(cond="CONTRA", cell="F02D-v1-m3", resp=response(out=3, thinking=0))]
        g = self._report(recs)["scopes"]["pooled"]["conditions"]["CONTRA"]
        self.assertEqual(g["thinking"]["activation"]["rate"], 0.25)
        self.assertEqual(g["unit_averaged"]["thinking_activation"]["mean_of_unit_means"], 0.5)

    def test_cap_censoring_retries_and_latency_notes(self):
        recs = [record(cond="CONTRA", cell="F01D-v1-m1", resp=response(text=None, out=1024, thinking=1024, stop="max_tokens")), record(cond="CONTRA", cell="F02D-v1-m1", attempts=2)]
        g = self._report(recs)["scopes"]["pooled"]["conditions"]["CONTRA"]["resources"]
        self.assertEqual((g["output_cap"]["cap_reached"], g["records_with_retries"]), (1, 1))
        self.assertIn("includes retries", g["latency_s"]["note"])
        self.assertTrue(g["cost_usd"]["estimate"])

    def test_select_unit_and_bad_selection(self):
        recs = [record(cond="CONTRA", cell="F01D-v1-m1"), record(cond="CONTRA", cell="F01L-v1-m1")]
        self.assertEqual(self._report(recs, select_unit="^F01D$")["selection"]["observations_selected"], 1)
        with self.assertRaises(B.BehaviorError):
            self._report(recs, select_unit="^zzz$")


class TestPairedContrasts(unittest.TestCase):
    def test_pairs_use_both_records_observed_and_report_discordance(self):
        recs = mixed_records()
        with Tmp() as d:
            rep = B.build_report([B.load_evidence(write_session(d, recs))], contrasts=[("CONTRA", "CONSIST")])
        c = rep["scopes"]["pooled"]["contrasts"][0]
        self.assertEqual((c["pairs"], c["CONTRA_only"], c["CONSIST_only"]), (4, 0, 0))
        t = c["measures"]["thinking_activation"]
        self.assertEqual((t["pairs_used"], t["pooled_mean_difference"]), (2, 0.5))             # the MISSING and INVALID CONTRA records cannot form a pair
        self.assertEqual(t["pairs_excluded"], {"CONTRA:MISSING / CONSIST:OBSERVED": 1, "CONTRA:INVALID / CONSIST:OBSERVED": 1})
        self.assertEqual(t["discordance"], {"CONTRA=1,CONSIST=1": 0, "CONTRA=1,CONSIST=0": 1, "CONTRA=0,CONSIST=1": 0, "CONTRA=0,CONSIST=0": 1})
        self.assertEqual(c["measures"]["input_tokens"]["pairs_used"], 4)                       # other measurements pair independently

    def test_conditions_that_share_no_prompt_cell_are_not_contrasted(self):
        recs = [record(cond="CONTRA", cell="F01D-v1-m1"), record(cond="CONSIST", cell="F02D-v1-m1")]
        with Tmp() as d:
            rep = B.build_report([B.load_evidence(write_session(d, recs))], contrasts=[("CONTRA", "CONSIST")])
        c = rep["scopes"]["pooled"]["contrasts"][0]
        self.assertEqual((c["pairs"], c["CONTRA_only"], c["CONSIST_only"]), (0, 1, 1))
        self.assertIn("NO PAIRS", c["verdict"])
        self.assertNotIn("thinking_activation", c["measures"])

    def test_pairs_never_cross_sessions(self):
        with Tmp() as d:
            a = write_session(d, [record(cond="CONTRA")], name="A")
            b = write_session(d, [record(cond="CONSIST")], name="B")
            rep = B.build_report([B.load_evidence(a), B.load_evidence(b)], contrasts=[("CONTRA", "CONSIST")])
        self.assertEqual(rep["scopes"]["pooled"]["contrasts"][0]["pairs"], 0)

    def test_baseline_option_and_unknown_condition(self):
        recs = [record(cond=c, cell="F01D-v1-m1") for c in ("BASE", "X", "Y")]
        with Tmp() as d:
            src = B.load_evidence(write_session(d, recs))
            rep = B.build_report([src], baseline="BASE")
            self.assertEqual([c["conditions"] for c in rep["scopes"]["pooled"]["contrasts"]], [["X", "BASE"], ["Y", "BASE"]])
            with self.assertRaises(B.BehaviorError):
                B.build_report([src], contrasts=[("X", "NOPE")])

    def test_answer_outcome_pairs(self):
        recs = [record(cond="CONTRA", cell="F01D-v1-m1", resp=response(text="C")), record(cond="CONSIST", cell="F01D-v1-m1", resp=response(text="A")),
                record(cond="CONTRA", cell="F02D-v1-m1", resp=response(text="A")), record(cond="CONSIST", cell="F02D-v1-m1", resp=response(text="A"))]
        with Tmp() as d:
            rep = B.build_report([B.load_evidence(write_session(d, recs))], contrasts=[("CONTRA", "CONSIST")])
        c = rep["scopes"]["pooled"]["contrasts"][0]
        self.assertEqual(c["answer_outcome_pairs"], {"different_outcome": 1, "same_outcome": 1})
        self.assertEqual(c["measures"]["correct"]["discordance"]["CONTRA=0,CONSIST=1"], 1)


# --------------------------------------------------------------------------------------
# CLI, offline guarantees, compatibility
# --------------------------------------------------------------------------------------
class TestCLI(unittest.TestCase):
    def test_report_ingest_and_output_protection(self):
        with Tmp() as d:
            sdir = write_session(d, mixed_records())
            code, text = run_main("behavior", "report", sdir, "--contrast", "CONTRA", "CONSIST", "--json-out", os.path.join(d, "r.json"))
            self.assertEqual(code, 0, text)
            self.assertIn("OFFLINE", text)
            self.assertIn("PAIRED CONTRAST CONTRA - CONSIST", text)
            self.assertEqual(json.load(open(os.path.join(d, "r.json")))["schema"], B.REPORT_SCHEMA)
            code, text = run_main("behavior", "report", sdir, "--json-out", os.path.join(d, "r.json"))
            self.assertEqual(code, 2)
            self.assertIn("never overwritten", text)
            code, text = run_main("behavior", "ingest", sdir, "--out", os.path.join(d, "o.jsonl"))
            self.assertEqual(code, 0, text)
            code, text = run_main("behavior", "report", os.path.join(d, "o.jsonl"), "--baseline", "CONSIST")       # reports read the observation file as well
            self.assertEqual(code, 0, text)
            self.assertIn("PAIRED CONTRAST CONTRA - CONSIST", text)

    def test_errors_exit_two_and_strict_flags_ingest_problems(self):
        with Tmp() as d:
            code, text = run_main("behavior", "report", os.path.join(d, "missing"))
            self.assertEqual(code, 2)
            sdir = write_session(d, [record()])
            open(os.path.join(sdir, "records", "bad.json"), "w").write("{")
            self.assertEqual(run_main("behavior", "report", sdir)[0], 0)
            code, text = run_main("behavior", "report", sdir, "--strict")
            self.assertEqual(code, 2)
            self.assertIn("invalid_json", text)
            self.assertEqual(run_main("behavior", "report", sdir, "--price-in", "1")[0], 2)       # one price alone is refused

    def test_text_report_shows_every_non_observed_state(self):
        with Tmp() as d:
            code, text = run_main("behavior", "report", write_session(d, mixed_records()))
        self.assertIn("NOT-OBSERVED STATES", text)
        self.assertIn('"MISSING": 1', text)
        self.assertIn('"INVALID": 1', text)

    def test_no_network_no_key_no_model(self):
        def boom(*a, **k):
            raise AssertionError("network access attempted")
        with Tmp() as d, mock.patch.object(socket.socket, "connect", boom), mock.patch.object(socket, "getaddrinfo", boom), mock.patch.dict(os.environ, {}, clear=False):
            for k in [k for k in os.environ if "API_KEY" in k]:
                os.environ.pop(k)
            self.assertEqual(run_main("behavior", "report", write_session(d, mixed_records()))[0], 0)

    def test_existing_commands_and_version_are_unchanged(self):
        import bleediff
        self.assertEqual(bleediff.__version__, "0.7.0")
        from bleediff.cli import build_parser
        names = set(next(a for a in build_parser()._actions if getattr(a, "choices", None) and "calibrate" in a.choices).choices)
        self.assertTrue({"calibrate", "bleed", "bounce", "diff", "inspect", "track", "gemini", "openai", "anthropic", "atlas", "experiment", "behavior"} <= names)
        # the v0.6 replay reader and the new reader see the same saved session and agree on every answer status
        with Tmp() as d:
            recs = mixed_records()
            ds = os.path.join(d, "prompts.json")
            items = [{"cell": r["cell"], "question": r["question"], "mapping": "m1", "condition": r["condition"], "prompt": "p " + r["name"], "correct_letter": "A", "competing_letter": "C"} for r in recs]
            json.dump({"schema": "bleediff.hosted-prompts/1", "baseline_condition": "CONSIST", "items": items}, open(ds, "w"))
            for r in recs:
                r["prompt_sha256"] = AN.sha256_bytes(("p " + r["name"]).encode("utf-8"))      # the v0.6 reader verifies the prompt hash
            sdir = write_session(d, recs)
            old_session = AN.read_session(sdir, ds)
            self.assertEqual(old_session["problems"], [])
            new = {o["identity"]["name"]: o["answer_status"] for o in B.load_evidence(sdir)["observations"]}
            self.assertEqual(new, {r["name"]: r["status"] for r in old_session["records"]})
            self.assertEqual(B.load_evidence(sdir, prompts_path=ds)["digest"], B.load_evidence(sdir)["digest"])

    def test_new_modules_are_python_3_11_syntax(self):
        for mod in ("telemetry.py", "behavior.py"):
            src = open(os.path.join(ROOT, "bleediff", mod), encoding="utf-8").read()
            ast.parse(src, feature_version=(3, 11))
            for imp in ("numpy", "scipy", "pandas", "requests", "yaml"):
                self.assertNotIn("import " + imp, src)


# --------------------------------------------------------------------------------------
# real-world validation fixture: the published Research 002 evidence (read only)
# --------------------------------------------------------------------------------------
def _r002_root():
    cand = [os.environ.get("BLEEDIFF_R002_PUBLICATION"), os.path.join(os.path.dirname(ROOT), "bleediff-research-002-publication")]
    for c in cand:
        if c and os.path.isdir(os.path.join(c, "research_artifacts", "run_output", "sessionA", "records")):
            return c
    return None


R002 = _r002_root()


@unittest.skipIf(R002 is None, "the published Research 002 package was not found next to this repository (set BLEEDIFF_R002_PUBLICATION to its directory)")
class TestResearch002Compatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ro = os.path.join(R002, "research_artifacts", "run_output")
        cls.before = tree_hash(cls.ro)
        cls.srcs = [B.load_evidence(os.path.join(cls.ro, "sessionA")), B.load_evidence(os.path.join(cls.ro, "sessionB"))]
        with open(os.path.join(R002, "research_artifacts", "analysis_output", "analysis_results.json"), encoding="utf-8") as fh:
            cls.pub = json.load(fh)
        cls.rep = B.build_report(cls.srcs, unit_field="question", select_unit=r"^F[0-9]+D$", contrasts=[("CONTRA", "CONSIST")])
        # the list prices of the Research 002 run (USD per 1M tokens, recovered from the saved records: they reproduce every usage-based cost exactly)
        cls.priced = [B.load_evidence(os.path.join(cls.ro, s), prices=(0.1, 0.5), cost_from_usage=True) for s in ("sessionA", "sessionB")]
        cls.rep_priced = B.build_report(cls.priced, unit_field="question", select_unit=r"^F[0-9]+D$", contrasts=[("CONTRA", "CONSIST")])

    @classmethod
    def tearDownClass(cls):
        assert tree_hash(cls.ro) == cls.before, "the Research 002 evidence changed while it was being read"

    def test_every_record_is_ingested_complete_and_consistent(self):
        obs = [o for s in self.srcs for o in s["observations"]]
        self.assertEqual((len(obs), [s["problems"] for s in self.srcs]), (2560, [[], []]))
        for o in obs:
            self.assertEqual(T.validate_observation(o), [])
            self.assertEqual(o["problems"], [])
            self.assertEqual({m(o, k)["state"] for k in T.MEASURES}, {T.OBSERVED}, o["identity"]["name"])
        self.assertEqual(sum(1 for o in obs if m(o, "thinking_activation")["value"]), 290)             # the 290 thinking blocks of the published record set
        self.assertEqual({o["context"]["thinking_corroboration"] for o in obs}, {"agrees"})
        self.assertEqual(sum(1 for o in obs if o["context"]["attempts"] == 2), 2)                     # the two retried requests
        self.assertEqual(sum(1 for o in obs if m(o, "cost_usd").get("includes_worst_case_charge_for_failed_attempts")), 2)
        self.assertEqual({m(o, "answer_role")["state"] for o in obs}, {T.OBSERVED})

    def test_condition_level_values_equal_the_published_results(self):
        per = self.pub["secondary"]["per_condition"]
        conds = self.rep["scopes"]["pooled"]["conditions"]
        for cond, g in conds.items():
            ua, ex = g["unit_averaged"], per[cond]
            self.assertEqual(g["units"]["distinct_units"], 24)
            self.assertAlmostEqual(ua["thinking_activation"]["mean_of_unit_means"], ex["T"]["family_mean"], places=9, msg=cond)
            self.assertAlmostEqual(ua["correct"]["mean_of_unit_means"], ex["correct"]["family_mean"], places=9, msg=cond)
            self.assertAlmostEqual(1 - ua["format_compliant"]["mean_of_unit_means"], ex["format_failure"]["family_mean"], places=9, msg=cond)
            self.assertAlmostEqual(ua["refusal"]["mean_of_unit_means"], ex["refusal"]["family_mean"], places=9, msg=cond)
            self.assertAlmostEqual(ua["thinking_tokens"]["mean_of_unit_means"], ex["thinking"]["family_mean"], places=8, msg=cond)
            pc = self.rep_priced["scopes"]["pooled"]["conditions"][cond]["unit_averaged"]["cost_usd"]["mean_of_unit_means"]       # priced from usage, as the published analysis does
            self.assertAlmostEqual(pc, ex["cost"]["family_mean"], delta=1e-11, msg=cond)

    def test_counts_equal_the_published_counts(self):
        cnt = self.pub["secondary"]["counts_by_condition"]
        for cond, g in self.rep["scopes"]["pooled"]["conditions"].items():
            a, e = g["answer_behavior"], cnt[cond]
            self.assertEqual((g["units"]["requests"], a["correct"], a["competing"], g["format_and_refusal"]["refusal"]["refused"]), (e["records"], e["strict_correct"], e["competing_selected"], e["refusals"]), cond)
            self.assertEqual(g["format_and_refusal"]["format_compliant"]["observed"] - g["format_and_refusal"]["format_compliant"]["compliant"], e["format_failures"], cond)
            self.assertEqual(g["resources"]["output_cap"]["cap_reached"], e["output_cap"], cond)
            self.assertEqual(g["units"]["distinct_prompts"] * 2, g["units"]["requests"], "each prompt was requested in two sessions: repeated measurement, not 384 independent prompts")

    def test_paired_contrast_equals_the_published_contrast(self):
        c = self.rep["scopes"]["pooled"]["contrasts"][0]
        pub = self.pub["secondary"]["contrasts"]["CONTRA-CONSIST"]
        self.assertEqual((c["pairs"], c["CONTRA_only"], c["CONSIST_only"]), (384, 0, 0))
        ms = c["measures"]
        self.assertAlmostEqual(ms["thinking_activation"]["unit_averaged_mean_difference"], pub["T"]["mean"], places=9)
        self.assertAlmostEqual(ms["correct"]["unit_averaged_mean_difference"], pub["correct"]["mean"], places=9)
        pm = self.rep_priced["scopes"]["pooled"]["contrasts"][0]["measures"]["cost_usd"]["unit_averaged_mean_difference"]
        self.assertAlmostEqual(pm, pub["cost"]["mean"], delta=1e-11)
        self.assertAlmostEqual(ms["thinking_tokens"]["unit_averaged_mean_difference"], pub["thinking"]["mean"], places=8)
        self.assertNotIn("ci95", json.dumps(c))                                                         # no inference is computed

    def test_published_results_file_is_not_modified_and_numbers_stand(self):
        self.assertAlmostEqual(self.pub["secondary"]["contrasts"]["CONTRA-CONSIST"]["T"]["mean"], 0.46875)
        self.assertEqual(self.pub["decisions"]["categories"], ["tier 2"])

    def test_recorded_cost_differs_from_usage_cost_only_by_the_two_failed_attempt_charges(self):
        rec_total = sum(self.rep["scopes"]["pooled"]["conditions"][c]["resources"]["cost_usd"]["sum"] for c in self.rep["conditions"])
        use_total = sum(self.rep_priced["scopes"]["pooled"]["conditions"][c]["resources"]["cost_usd"]["sum"] for c in self.rep_priced["conditions"])
        flagged = sum(self.rep["scopes"]["pooled"]["conditions"][c]["resources"]["cost_usd"]["records_with_failed_attempt_charge"] for c in self.rep["conditions"])
        self.assertEqual(flagged, 1)                                  # one of the two retried requests is a derivation record (the other is a lookup twin outside the selection)
        self.assertGreater(rec_total, use_total)
        self.assertAlmostEqual(rec_total - use_total, 0.0006011 - 2.77e-05, delta=1e-9)       # exactly that record's worst-case charge for its failed attempt

    def test_resource_figures_stay_separate_from_thinking_figures(self):
        g = self.rep["scopes"]["pooled"]["conditions"]["CONTRA"]
        self.assertEqual(g["resources"]["input_tokens"]["n"], 384)
        self.assertEqual(set(g["thinking"]), {"activation", "thinking_tokens", "thinking_tokens_among_activated", "corroboration"})


if __name__ == "__main__":
    unittest.main()
