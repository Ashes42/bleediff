"""`bleediff experiment` (offline authoring of an ANSWER_ONLY experiment). No test contacts a network or needs a key."""

from __future__ import annotations

import copy
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from tests.test_anthropic_adapter import SECRET, claude  # noqa: E402
from tests.test_bleediff import run_cli  # noqa: E402
from tests.test_gemini_adapter import FakeAPI, tmp, write_json  # noqa: E402

from bleediff import answers as AN  # noqa: E402
from bleediff import anthropic_messages as CL  # noqa: E402
from bleediff import experiment as EX  # noqa: E402


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

    def test_warnings(self):
        s = copy.deepcopy(EX.STARTER)
        s["conditions"]["CONTROL_NEUTRAL"] = "Mercury is mentioned here, " + "x" * 400
        w = " ".join(EX.warnings(s))
        self.assertIn("not neutral", w)
        self.assertIn("lengths differ", w)
        s = copy.deepcopy(EX.STARTER)
        s["conditions"] = {"TREATMENT": "Venus is closest."}
        self.assertIn("CONTROL", " ".join(EX.warnings(s)))
        s = copy.deepcopy(EX.STARTER)
        s["mappings"] = {"m1": s["mappings"]["m1"]}
        self.assertIn("one letter only", " ".join(EX.warnings(s)))


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

    def test_estimate_matches_the_guard_formula(self):
        ds = EX.build_prompts(EX.STARTER)
        e = EX.estimate(ds, 2, 1024, 1.0, 5.0)
        inp = sum(len(i["prompt"].encode()) + 64 for i in ds["items"]) * 2
        self.assertEqual((e["requests"], e["input_tokens_upper"], e["output_tokens_reserved"]), (24, inp, 24 * 1024))
        self.assertAlmostEqual(e["worst_case_usd"], inp / 1e6 + 24 * 1024 * 5 / 1e6)
        self.assertGreaterEqual(e["suggested_budget_usd"], e["worst_case_usd"])

    def test_classification_never_claims_significance(self):
        self.assertTrue(EX.classification(EX.STARTER).startswith("DEMONSTRATION"))
        big = copy.deepcopy(EX.STARTER)
        big["questions"] = [dict(EX.STARTER["questions"][0], id=f"q{i}") for i in range(30)]
        self.assertIn("not a power analysis", EX.classification(big))


class TestCli(unittest.TestCase):
    def test_init_validate_and_write_prompts_offline(self):
        out = os.path.join(tmp(), "exp")
        code, text = run_cli("experiment", "init", out)
        self.assertEqual(code, 0, text)
        self.assertEqual(run_cli("experiment", "init", out)[0], 2)                      # never overwrites
        spec, prompts = os.path.join(out, "experiment.json"), os.path.join(out, "prompts.json")
        code, text = run_cli("experiment", "validate", spec, "--write-prompts", prompts, "--passes", "2", "--price-in", "1", "--price-out", "5")
        self.assertEqual(code, 0, text)
        for needle in ("VALID", "12 prompts", "m3: A=distractor_1", "24 requests", "Suggested --budget", "DEMONSTRATION", "not a bill", "FLR-v2"):
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

    def test_half_a_price_pair_is_refused(self):
        self.assertEqual(run_cli("experiment", "validate", starter_file(tmp()), "--price-in", "1")[0], 2)


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
