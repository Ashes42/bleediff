"""`bleediff experiment` (offline authoring of an ANSWER_ONLY experiment). No test contacts a network or needs a key."""

from __future__ import annotations

import copy
import json
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from tests.test_anthropic_adapter import SECRET, claude  # noqa: E402
from tests.test_bleediff import run_cli  # noqa: E402
from tests.test_gemini_adapter import FakeAPI, rb, tmp, write_json  # noqa: E402

from bleediff import answers as AN  # noqa: E402
from bleediff import anthropic_messages as CL  # noqa: E402
from bleediff import experiment as EX  # noqa: E402
from bleediff import gemini as GM  # noqa: E402
from bleediff import hosted as H  # noqa: E402
from bleediff import openai_chat as OA  # noqa: E402


def starter_file(d, mutate=None):
    spec = copy.deepcopy(EX.STARTER)
    if mutate:
        mutate(spec)
    path = os.path.join(d, "experiment.json")
    write_json(path, spec)
    return path


class TestSpec(unittest.TestCase):
    def test_starter_is_valid_and_has_no_warnings(self):
        self.assertEqual(EX.check_spec(EX.STARTER), [])
        self.assertEqual(EX.warnings(EX.STARTER), [])

    def test_each_defect_is_reported(self):
        cases = {
            "schema": lambda s: s.update(schema="x"),
            "id": lambda s: s.update(id="Bad Id"),
            "no questions": lambda s: s.update(questions=[]),
            "dup option": lambda s: s["questions"][0]["options"].update(competing="mercury"),
            "missing role": lambda s: s["questions"][0]["options"].pop("distractor_2"),
            "empty evidence": lambda s: s["questions"][0].update(evidence=" "),
            "baseline listed": lambda s: s["conditions"].update(BASELINE="x"),
            "lowercase condition": lambda s: s["conditions"].update(bad="x"),
            "empty fragment": lambda s: s["conditions"].update(CONTROL_NEUTRAL=""),
            "mapping not a permutation": lambda s: s["mappings"]["m1"].update(B="correct"),
            "mapping missing letter": lambda s: s["mappings"]["m1"].pop("D"),
            "duplicate mappings": lambda s: s["mappings"].update(m5=dict(s["mappings"]["m1"])),
        }
        for name, f in cases.items():
            s = copy.deepcopy(EX.STARTER)
            f(s)
            self.assertTrue(EX.check_spec(s), name)
        self.assertEqual(EX.check_spec([]), ["the spec must be a JSON object"])

    def test_warnings_are_worded_as_heuristics_about_characters(self):
        s = copy.deepcopy(EX.STARTER)
        s["conditions"]["CONTROL_NEUTRAL"] = "Mercury is mentioned here, " + "x" * 400
        w = " ".join(EX.warnings(s))
        self.assertIn("probably not neutral (text match only)", w)
        self.assertIn("character counts differ", w)
        self.assertIn("token counts may still differ", w)
        s = copy.deepcopy(EX.STARTER)
        s["conditions"] = {"TREATMENT": "Venus is closest."}
        w = " ".join(EX.warnings(s))
        self.assertIn("name heuristic only", w)
        self.assertIn("does not make a fragment neutral", w)
        s = copy.deepcopy(EX.STARTER)
        s["mappings"] = {"m1": s["mappings"]["m1"]}
        self.assertIn("one letter only", " ".join(EX.warnings(s)))

    def test_a_misnamed_control_passes_the_heuristic_which_is_why_the_output_says_so(self):
        s = copy.deepcopy(EX.STARTER)
        s["conditions"] = {"CONTROL_ANYTHING": "Venus is the closest planet to the Sun, many say.", "TREATMENT": "Venus is the closest planet to the Sun, many say."}
        self.assertTrue([w for w in EX.warnings(s) if "probably not neutral" in w])       # text match catches this one,
        s["conditions"]["CONTROL_ANYTHING"] = "Many say the planet nearest the Sun is the second one out."
        self.assertEqual(EX.warnings(s), [])                                                # but a paraphrase is not caught: a clean result proves nothing
        code, text = run_cli("experiment", "validate", starter_file(tmp(), lambda x: x.update(conditions=s["conditions"])))
        self.assertIn("No warning does not prove that a control is neutral", text)
        self.assertIn("not tokenizer parity", text)


class TestPrompts(unittest.TestCase):
    def test_dataset_loads_with_the_runners_loader_and_roles_follow_the_mapping(self):
        d = tmp()
        path = os.path.join(d, "prompts.json")
        write_json(path, EX.build_prompts(EX.STARTER))
        loaded = AN.load_prompts(path)
        self.assertEqual(len(loaded["items"]), 12)
        self.assertEqual(loaded["conditions"], ["BASELINE", "TREATMENT_COMPETING_CLAIM", "CONTROL_NEUTRAL"])
        for it in loaded["items"]:
            lines = {ln[0]: ln[3:] for ln in it["prompt"].split("\n") if len(ln) > 3 and ln[1] == ")"}
            self.assertEqual(lines[it["correct_letter"]], "Mercury")
            self.assertEqual(lines[it["competing_letter"]], "Venus")
        pairs = {(i["mapping"], i["correct_letter"], i["competing_letter"]) for i in loaded["items"]}
        self.assertEqual(pairs, {("m1", "A", "B"), ("m2", "B", "A"), ("m3", "C", "D"), ("m4", "D", "C")})

    def test_baseline_has_no_fragment_and_conditions_differ_only_by_it(self):
        items = {(i["mapping"], i["condition"]): i for i in EX.build_prompts(EX.STARTER)["items"]}
        base = items[("m1", "BASELINE")]
        self.assertIsNone(base["fragment_sha256"])
        for cond, frag in EX.STARTER["conditions"].items():
            self.assertEqual(items[("m1", cond)]["prompt"], base["prompt"].replace("\n\nReply with", f"\n{frag}\n\nReply with"))

    def test_generation_is_deterministic(self):
        self.assertEqual(json.dumps(EX.build_prompts(EX.STARTER)), json.dumps(EX.build_prompts(copy.deepcopy(EX.STARTER))))

    def test_classification_wording(self):
        small = EX.classification(EX.STARTER)
        self.assertTrue(small.startswith("DEMONSTRATION"))
        self.assertIn("not that any context caused any change", small)
        big = copy.deepcopy(EX.STARTER)
        big["questions"] = [dict(EX.STARTER["questions"][0], id=f"q{i}") for i in range(30)]
        text = EX.classification(big)
        for needle in ("illustrative convention", "not a power analysis", "Rewording one question 30 times does not create 30 independent questions", "its own evidence and its own controls"):
            self.assertIn(needle, text)


class TestCli(unittest.TestCase):
    def test_init_validate_and_write_prompts_offline(self):
        out = os.path.join(tmp(), "exp")
        code, text = run_cli("experiment", "init", out)
        self.assertEqual(code, 0, text)
        self.assertEqual(run_cli("experiment", "init", out)[0], 2)                      # never overwrites
        spec, prompts = os.path.join(out, "experiment.json"), os.path.join(out, "prompts.json")
        code, text = run_cli("experiment", "validate", spec, "--write-prompts", prompts, "--passes", "2", "--provider", "anthropic", "--price-in", "1", "--price-out", "5")
        self.assertEqual(code, 0, text)
        for needle in ("VALID", "12 prompts", "m3: A=distractor_1", "24 requests", "Budget estimate for anthropic", "ADMIT", "not a provider billing limit", "DEMONSTRATION", "FLR-v2"):
            self.assertIn(needle, text)
        self.assertEqual(AN.load_prompts(prompts)["schema"], "bleediff.hosted-prompts/1")
        self.assertEqual(run_cli("experiment", "validate", spec, "--write-prompts", prompts)[0], 2)      # an existing file is not replaced

    def test_invalid_spec_is_inconclusive_and_writes_nothing(self):
        d = tmp()
        path = starter_file(d, lambda s: s["conditions"].update(BASELINE="x"))
        out = os.path.join(d, "p.json")
        code, text = run_cli("experiment", "validate", path, "--write-prompts", out)
        self.assertEqual(code, 2)
        self.assertIn("EXPERIMENT_SPEC_INVALID", text)
        self.assertFalse(os.path.exists(out))
        self.assertEqual(run_cli("experiment", "validate", os.path.join(d, "missing.json"))[0], 2)

    def test_half_an_estimate_request_is_refused(self):
        for extra in (["--price-in", "1"], ["--provider", "openai"], ["--provider", "openai", "--price-in", "1"], ["--price-in", "1", "--price-out", "5"]):
            code, text = run_cli("experiment", "validate", starter_file(tmp()), *extra)
            self.assertEqual((code, "BAD_ARGS" in text), (2, True), extra)

    def test_no_estimate_without_price_options(self):
        code, text = run_cli("experiment", "validate", starter_file(tmp()))
        self.assertEqual(code, 0)
        self.assertNotIn("Budget estimate", text)
        self.assertNotIn("$-", text)

    def test_provider_changes_the_figures_and_the_estimate_is_labelled_an_estimate(self):
        spec = starter_file(tmp())
        outs = {p: run_cli("experiment", "validate", spec, "--provider", p, "--price-in", "1", "--price-out", "5")[1] for p in ("anthropic", "openai", "gemini")}
        self.assertIn("at most $0.1356", outs["anthropic"])
        self.assertIn("at most $0.0742", outs["openai"])
        self.assertIn("at most $0.1356", outs["gemini"])
        self.assertIn("reserve 1024 output tokens per request (the cap plus the provider's thinking allowance)", outs["gemini"])
        self.assertIn("Gemini figures assume no --thinking-budget", outs["gemini"].replace("These Gemini", "Gemini"))
        self.assertNotIn("thinking-budget", outs["openai"])
        for t in outs.values():
            self.assertIn("not a provider billing limit or a guarantee that a run completes", t)
            self.assertNotIn("Suggested", t)


class TestArgumentRanges(unittest.TestCase):
    def test_out_of_range_values_are_bad_args_and_write_nothing(self):
        d = tmp()
        spec = starter_file(d)
        base = ["experiment", "validate", spec, "--provider", "anthropic", "--price-in", "1", "--price-out", "5"]
        bad = {"--passes": ("0", "-3", "21", "999"), "--max-output-tokens": ("0", "-1", "65537"), "--max-retries": ("-1", "6")}
        n = 0
        for flag, values in bad.items():
            for v in values:
                out = os.path.join(d, f"p{n}.json")
                code, text = run_cli(*base, flag, v, "--write-prompts", out)
                self.assertEqual(code, 2, (flag, v))
                self.assertIn("BAD_ARGS", text)
                self.assertNotIn("$-", text)
                self.assertFalse(os.path.exists(out), (flag, v))
                n += 1

    def test_the_limits_are_accepted_and_match_the_runner(self):
        self.assertEqual((H.PASSES_RANGE, H.MAX_OUTPUT_TOKENS_RANGE, H.MAX_RETRIES_RANGE), ((1, 20), (1, 65536), (0, 5)))
        spec = starter_file(tmp())
        for flag, v in (("--passes", "1"), ("--passes", "20"), ("--max-output-tokens", "1"), ("--max-output-tokens", "65536"), ("--max-retries", "0"), ("--max-retries", "5")):
            code, text = run_cli("experiment", "validate", spec, "--provider", "gemini", "--price-in", "1", "--price-out", "5", flag, v)
            self.assertEqual(code, 0, (flag, v, text))
        for bad in ("nan", "inf", "-1"):
            self.assertEqual(run_cli("experiment", "validate", spec, "--provider", "openai", "--price-in", bad, "--price-out", "5")[0], 2, bad)

    def test_the_runner_rejects_the_same_values(self):
        d = tmp()
        ds = os.path.join(d, "p.json")
        write_json(ds, EX.build_prompts(EX.STARTER))
        for kw in ({"passes": 0}, {"passes": 21}, {"max_output_tokens": 0}, {"max_output_tokens": 65537}, {"max_retries": 6}):
            with self.assertRaises(Exception) as cm:
                H.run_session(CL.ANTHROPIC, prompts_path=ds, model="claude-haiku-5-5", out_dir=os.path.join(d, "s"), budget=1, price_in_per_m=1, price_out_per_m=1, key=SECRET,
                              **{"passes": 1, **kw})
            self.assertEqual(getattr(cm.exception, "reason", None), "BAD_ARGS", kw)


class TestFilesystemErrors(unittest.TestCase):
    def test_missing_parent_directory(self):
        d = tmp()
        out = os.path.join(d, "nodir", "p.json")
        code, text = run_cli("experiment", "validate", starter_file(d), "--write-prompts", out)
        self.assertEqual(code, 2)
        self.assertIn("OUTPUT_UNWRITABLE", text)
        self.assertNotIn("Traceback", text)
        self.assertFalse(os.path.exists(os.path.dirname(out)))

    def test_parent_is_a_file_and_permission_denied(self):
        d = tmp()
        spec = starter_file(d)
        blocker = os.path.join(d, "afile")
        with open(blocker, "w") as fh:
            fh.write("x")
        self.assertIn("OUTPUT_UNWRITABLE", run_cli("experiment", "validate", spec, "--write-prompts", os.path.join(blocker, "p.json"))[1])
        out = os.path.join(d, "p.json")
        real_open = open

        def deny(path, *a, **k):
            if os.path.abspath(path) == os.path.abspath(out):
                raise PermissionError(13, "Permission denied")
            return real_open(path, *a, **k)
        with mock.patch("builtins.open", side_effect=deny):
            code, text = run_cli("experiment", "validate", spec, "--write-prompts", out)
        self.assertEqual(code, 2)
        self.assertIn("OUTPUT_UNWRITABLE", text)
        self.assertFalse(os.path.exists(out))

    def test_existing_files_and_directories_are_never_overwritten(self):
        d = tmp()
        spec = starter_file(d)
        out = os.path.join(d, "p.json")
        with open(out, "w") as fh:
            fh.write("precious")
        code, text = run_cli("experiment", "validate", spec, "--write-prompts", out)
        self.assertEqual((code, "OUTPUT_EXISTS" in text), (2, True))
        self.assertEqual(rb(out), b"precious")
        before = rb(spec)
        self.assertEqual(run_cli("experiment", "validate", spec, "--write-prompts", spec)[0], 2)          # the spec itself
        self.assertEqual(rb(spec), before)
        self.assertEqual(run_cli("experiment", "validate", spec, "--write-prompts", d)[0], 2)             # a directory
        code, text = run_cli("experiment", "init", d)
        self.assertEqual((code, "OUTPUT_EXISTS" in text), (2, True))

    def test_init_reports_an_unwritable_location(self):
        d = tmp()
        blocker = os.path.join(d, "afile")
        with open(blocker, "w") as fh:
            fh.write("x")
        code, text = run_cli("experiment", "init", os.path.join(blocker, "sub"))
        self.assertEqual((code, "OUTPUT_UNWRITABLE" in text), (2, True))


class TestProviderBudget(unittest.TestCase):
    """The estimate uses the hosted engine's own reserve/admission functions. These tests check it against literal per-provider rules and against the real runner (mock transport)."""
    PIN, POUT = 1.0, 5000.0                    # a large output price so that rounding to cents cannot hide the differences

    def setUp(self):
        self.ds = EX.build_prompts(EX.STARTER)
        self.uppers = [len(i["prompt"].encode("utf-8")) + 64 for i in self.ds["items"]]

    def test_provider_reserve_rules_and_defaults(self):
        self.assertEqual((H.DEFAULT_MAX_OUTPUT_TOKENS, H.DEFAULT_MAX_RETRIES), ({"anthropic": 1024, "openai": 512, "gemini": 512}, 2))
        for name, cap, mult in (("anthropic", 1024, 1), ("openai", 512, 1), ("gemini", 512, 2)):
            e = EX.estimate(self.ds, name, 2, cap, self.PIN, self.POUT)
            bound = cap * mult * self.POUT / 1e6
            literal = sum(self.uppers) * 2 * self.PIN / 1e6 + 24 * bound
            self.assertEqual(e["reserve_tokens"], cap * mult, name)                                      # Gemini adds a thinking allowance of one more cap
            self.assertAlmostEqual(e["success_ceiling_usd"], literal, places=9, msg=name)
            self.assertAlmostEqual(e["admit_fixed_usd"], literal, places=9)
            self.assertAlmostEqual(e["admit_grown_usd"], literal + bound, places=9)                      # only the last request needs one more bound
            self.assertLessEqual(e["admit_fixed_usd"], e["admit_grown_usd"])
            self.assertGreater(e["admit_retries_usd"], 2 * e["admit_grown_usd"])                          # 2 retries on every request cost about 3x
            self.assertLessEqual(e["admit_retries_usd"], 3 * (sum(self.uppers) * 2 * self.PIN / 1e6 + 24 * 2 * bound) + 1e-9)    # failed attempts are charged at the doubled bound

    def test_budgets_are_positive_whole_cents_and_never_below_the_figures(self):
        for name in EX.PROVIDERS:
            for prices in ((0.0, 0.0), (0.1, 0.5), (3.0, 15.0)):
                e = EX.estimate(self.ds, name, 1, 64, *prices)
                for k in ("fixed", "grown", "retries"):
                    self.assertGreaterEqual(e["budget_" + k], 0.01)
                    self.assertGreaterEqual(e["budget_" + k] + 1e-9, e["admit_" + k + "_usd"])
                    self.assertAlmostEqual(e["budget_" + k] * 100, round(e["budget_" + k] * 100), places=6)

    def _run(self, provider, budget, replies, max_retries=0):
        d = tmp()
        ds = os.path.join(d, "p.json")
        write_json(ds, self.ds)
        by_prompt = {i["prompt"]: i for i in self.ds["items"]}
        api = FakeAPI(lambda prompt, n: replies(by_prompt[prompt], prompt, n))
        res = H.run_session(provider, prompts_path=ds, model="claude-haiku-5-5", out_dir=os.path.join(d, "s"), passes=1, budget=budget, price_in_per_m=self.PIN,
                            price_out_per_m=self.POUT, key=SECRET, post=api.post, max_output_tokens=1024, max_retries=max_retries, sleep=lambda s: None)
        return res, api

    def test_first_request_admission_matches_the_literal_provider_rule(self):
        ok = lambda item, prompt, n: (200, claude(item["correct_letter"]))
        for provider, mult in ((CL.ANTHROPIC, 1), (OA.OPENAI, 1), (GM.GEMINI, 2)):
            first = (self.uppers[0] * self.PIN + 1024 * mult * self.POUT) / 1e6
            res, api = self._run(provider, first - 1e-7, ok)
            self.assertEqual((api.calls, res["status"]), (0, "partial"), provider.name)
            if provider is CL.ANTHROPIC:                      # the mock replies use the Anthropic shape
                res, api = self._run(provider, first + 1e-7, ok)
                self.assertGreaterEqual(api.calls, 1)

    def test_figures_two_and_three_against_the_real_runner_when_replies_fill_the_cap(self):
        full = lambda item, prompt, n: (200, claude(item["correct_letter"], usage=(len(prompt.encode("utf-8")) + 64, 1024, None)))
        e = EX.estimate(self.ds, "anthropic", 1, 1024, self.PIN, self.POUT)
        res, api = self._run(CL.ANTHROPIC, e["budget_grown"], full)
        self.assertEqual((res["status"], api.calls), ("complete", 12), res)
        res, api = self._run(CL.ANTHROPIC, e["budget_fixed"], full)                                       # figure 2 alone is not enough once replies fill the reserve
        self.assertEqual(res["status"], "partial")
        self.assertLess(api.calls, 12)
        short = lambda item, prompt, n: (200, claude(item["correct_letter"]))
        res, api = self._run(CL.ANTHROPIC, e["budget_fixed"], short)                                      # with short replies it is enough
        self.assertEqual((res["status"], api.calls), ("complete", 12))

    def test_figure_four_against_the_real_runner_with_one_retry_per_request(self):
        def flaky(item, prompt, n):
            if n % 2 == 1:
                return 500, {"type": "error", "error": {"type": "api_error", "message": "boom"}}
            return 200, claude(item["correct_letter"], usage=(len(prompt.encode("utf-8")) + 64, 1024, None))
        e = EX.estimate(self.ds, "anthropic", 1, 1024, self.PIN, self.POUT, max_retries=1)
        res, api = self._run(CL.ANTHROPIC, e["budget_retries"], flaky, max_retries=1)
        self.assertEqual((res["status"], api.calls), ("complete", 24), res)
        res, api = self._run(CL.ANTHROPIC, e["budget_grown"], flaky, max_retries=1)                       # the no-retry figure does not cover the same outage
        self.assertNotEqual(res["status"], "complete")                                                    # stopped early (retries are refused or consecutive failures abort)
        self.assertLess(api.calls, 24)


class TestEndToEndWithMockedApi(unittest.TestCase):
    """Generated prompts run through the real hosted engine (mock transport) and replay offline. No key, no network."""

    def test_run_and_replay(self):
        d = tmp()
        prompts = os.path.join(d, "prompts.json")
        ds = EX.build_prompts(EX.STARTER)
        write_json(prompts, ds)
        by_prompt = {i["prompt"]: i for i in ds["items"]}

        def handler(prompt, n):
            it = by_prompt[prompt]
            if it["condition"] == "TREATMENT_COMPETING_CLAIM":
                return 200, claude(f"The answer is {it['competing_letter']}\n{it['competing_letter']}")      # format-invalid: only the labelled diagnostic recovers it
            return 200, claude(it["correct_letter"])

        api = FakeAPI(handler)
        out = os.path.join(d, "session")
        CL.run_session(prompts_path=prompts, model="claude-haiku-5-5", out_dir=out, passes=1, budget=1.0, price_in_per_m=1, price_out_per_m=5, key=SECRET, post=api.post)
        self.assertEqual(api.calls, 12)
        code, replay = run_cli("anthropic", "--replay", out)
        self.assertEqual(code, 0, replay)
        for needle in ("OFFLINE REPLAY", "BASELINE", "TREATMENT_COMPETING_CLAIM", "CONTROL_NEUTRAL", "DIAGNOSTIC"):
            self.assertIn(needle, replay)


if __name__ == "__main__":
    unittest.main()
