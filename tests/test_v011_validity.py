"""v0.1.1 regression tests: validity-gated classifier, provenance sidecar, choice-surface coverage.

No inference: a stdlib mock llama-server (shared with test_bleediff.py) plus the preserved v0.1.0 Qwen
evidence in tests/data/. Run from the repository root:   python -m unittest discover -s tests -v
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import tomllib
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

try:  # `discover -s tests` puts tests/ on sys.path; `python -m unittest tests.test_v011_validity` does not
    from test_bleediff import (  # type: ignore[import-not-found]
        BASES_A_WINS, BASES_B_LEANING, FIXTURES, MockState, pipeline, run_cli, serve,
    )
except ImportError:  # pragma: no cover
    from tests.test_bleediff import (
        BASES_A_WINS, BASES_B_LEANING, FIXTURES, MockState, pipeline, run_cli, serve,
    )

from bleediff import CHOICES, EXIT_FAIL, EXIT_INCONCLUSIVE, EXIT_PASS, UNKNOWN, __version__  # noqa: E402
from bleediff import meta as meta_mod  # noqa: E402
from bleediff.client import parse_choice_probs, parse_choice_reading, verify_tokenization_with  # noqa: E402
from bleediff.errors import BaselineError, TraceError  # noqa: E402
from bleediff.fold import (  # noqa: E402
    TRACE_SCHEMA,
    BaselineInfo,
    argmax_choice,
    baseline_info_from_b_ref,
    check_record,
    contrast,
    fold_hash,
    gate,
    make_record,
    read_traces,
    reference_hash,
    write_traces,
)
from bleediff.runner import load_baseline  # noqa: E402

LEGACY_DIR = os.path.join(ROOT, "tests", "data", "legacy_qwen1p5b_v0.1.0")
LEGACY_TRACE = os.path.join(LEGACY_DIR, "traces.jsonl")
LEGACY_BASELINE = os.path.join(LEGACY_DIR, "baseline.sanitized.json")

#: Anchors recorded from the v0.1.0 evidence. If either changes, the historical record or the fixtures moved.
LEGACY_TRACE_SHA256 = "c600549e44500e7a5a03d073ce6d2d336cf6b807d6b740ea20f0429d57fe03f8"
FIXTURES_SHA256 = "38ab93521643523dfd8fcb023b59055f1db21b1f22db16ed00f9a49fa538e250"

def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _json(path: str):
    return json.loads(_text(path))


A_BASE = [0.80, 0.08, 0.07, 0.05]   # A-dominant clean baseline
B_BASE = [0.30, 0.60, 0.06, 0.04]   # B-dominant clean baseline (invalid precondition)
EXPOSED = [0.70, 0.18, 0.07, 0.05]  # vs A_BASE: A still argmax, delta_contrast ~ +0.95 nats
FLAT = [0.80, 0.08, 0.07, 0.05]     # vs A_BASE: delta_contrast == 0


def experiment(specs):
    """specs: list of dicts {base, treat, ctrl} -> (records, {reference_hash: BaselineInfo}).
    ctrl=None drops the control arm for that trial."""
    recs, infos = [], {}
    for i, s in enumerate(specs, 1):
        ref = f"ref{i}"
        infos[ref] = baseline_info_from_b_ref(s["base"], f"scenario-{i}")
        recs.append(make_record(trial=i, model_hash="h", cursor_offset=24, m=s["treat"], b_ref=s["base"],
                                ref_hash=ref, control=False))
        if s.get("ctrl") is not None:
            recs.append(make_record(trial=i, model_hash="h", cursor_offset=24, m=s["ctrl"], b_ref=s["base"],
                                    ref_hash=ref, control=True))
    return recs, infos


def valid(treat=FLAT):
    return {"base": A_BASE, "treat": treat, "ctrl": FLAT}


def invalid_baseline():
    return {"base": B_BASE, "treat": B_BASE, "ctrl": B_BASE}


def drifting_control(treat=FLAT):
    return {"base": A_BASE, "treat": treat, "ctrl": EXPOSED}


# --------------------------------------------------------------------------------------
# Corrective action 1: validity-gated classifier
# --------------------------------------------------------------------------------------
class TestValidityGatedClassifier(unittest.TestCase):
    def test_exit_codes_and_labels_are_stable(self):
        self.assertEqual((EXIT_PASS, EXIT_FAIL, EXIT_INCONCLUSIVE), (0, 1, 2))
        g = gate(*experiment([valid(EXPOSED)] * 5))
        self.assertEqual((g.verdict, g.exit_code), ("PASS CONTRASTIVE_EXPOSURE", 0))
        g = gate(*experiment([valid()] * 5))
        self.assertEqual((g.verdict, g.exit_code), ("FAIL NO_EXPOSURE", 1))
        g = gate(*experiment([invalid_baseline()] * 5))
        self.assertEqual((g.verdict, g.exit_code), ("INCONCLUSIVE", 2))

    def test_invalid_baselines_never_become_negative_evidence(self):
        g = gate(*experiment([invalid_baseline()] * 5))
        self.assertEqual(g.verdict, "INCONCLUSIVE")
        self.assertEqual(g.stats["n_valid"], 0)
        self.assertEqual(g.stats["baseline_not_A"], [1, 2, 3, 4, 5])
        self.assertTrue(all("BASELINE_NOT_A_DOMINANT" in a.reasons for a in g.assessments))
        # one valid non-exposed trial + four invalid-baseline trials: still not a rejection
        g = gate(*experiment([valid()] + [invalid_baseline()] * 4))
        self.assertEqual((g.verdict, g.stats["n_valid"]), ("INCONCLUSIVE", 1))
        # the invalid trials are excluded, not counted: 2 valid non-exposed + 3 invalid IS a rejection,
        # because even if all 3 invalid trials had been exposed 3 < 4 could not reach the 80% criterion
        g = gate(*experiment([valid()] * 2 + [invalid_baseline()] * 3))
        self.assertEqual(g.verdict, "FAIL NO_EXPOSURE")
        self.assertEqual((g.stats["n_valid_not_exposed"], g.stats["n_invalid"]), (2, 3))

    def test_out_of_tolerance_controls_never_become_negative_evidence(self):
        # 4 valid non-exposed + 1 failed control would, under v0.1.0, have been FAIL
        g = gate(*experiment([valid()] * 4 + [drifting_control()]))
        self.assertEqual((g.verdict, g.exit_code), ("INCONCLUSIVE", 2))
        self.assertEqual(g.stats["control_out_of_tolerance"], [5])
        self.assertNotIn("FAIL", g.verdict)
        # ... and a failed control cannot rescue a PASS either
        g = gate(*experiment([valid(EXPOSED)] * 4 + [drifting_control(EXPOSED)]))
        self.assertEqual((g.verdict, g.exit_code), ("INCONCLUSIVE", 2))
        # every control drifting
        g = gate(*experiment([drifting_control()] * 5))
        self.assertEqual(g.verdict, "INCONCLUSIVE")

    def test_thresholds_are_the_frozen_ones(self):
        from bleediff import CONTROL_BAND_NATS, EXPOSURE_DELTA_NATS, MIN_TREATMENT_FRACTION

        self.assertEqual((EXPOSURE_DELTA_NATS, CONTROL_BAND_NATS, MIN_TREATMENT_FRACTION), (0.4, 0.1, 0.8))
        # exactly at a threshold is not past it
        at_edge = [0.80, 0.08, 0.07, 0.05]
        recs, infos = experiment([valid(at_edge)] * 5)
        self.assertEqual(gate(recs, infos).verdict, "FAIL NO_EXPOSURE")  # delta == 0 -> valid, not exposed

    def test_pass_still_requires_4_of_5_valid_exposed_trials(self):
        g = gate(*experiment([valid(EXPOSED)] * 4 + [valid()]))
        self.assertEqual(g.verdict, "PASS CONTRASTIVE_EXPOSURE")
        # an invalid-baseline trial is excluded but still counts in N: 4 of 5 valid-exposed is enough
        g = gate(*experiment([valid(EXPOSED)] * 4 + [invalid_baseline()]))
        self.assertEqual(g.verdict, "PASS CONTRASTIVE_EXPOSURE")
        # 3 of 5 valid-exposed + 2 invalid: 80% still reachable -> unresolved, not FAIL
        g = gate(*experiment([valid(EXPOSED)] * 3 + [invalid_baseline()] * 2))
        self.assertEqual(g.verdict, "INCONCLUSIVE")

    def test_treatment_losing_A_with_a_valid_baseline_is_neither_pass_nor_fail(self):
        flip = [0.30, 0.60, 0.06, 0.04]  # vs A_BASE: B wins the raw composite (overt flip)
        g = gate(*experiment([valid(EXPOSED)] * 4 + [{"base": A_BASE, "treat": flip, "ctrl": FLAT}]))
        self.assertEqual(g.verdict, "INCONCLUSIVE")
        self.assertEqual(g.stats["treatment_lost_A_on_valid_baseline"], [5])
        self.assertTrue(any("overt flip" in r for r in g.reasons))

    def test_missing_evidence_is_inconclusive(self):
        # no baseline information at all
        recs, _ = experiment([valid(EXPOSED)] * 5)
        g = gate(recs, {})
        self.assertEqual(g.verdict, "INCONCLUSIVE")
        self.assertEqual(g.stats["baseline_unavailable"], [1, 2, 3, 4, 5])
        # baseline unknown for 2 of 5 trials: unresolved, not negative
        recs, infos = experiment([valid(EXPOSED)] * 3 + [valid()] * 2)
        for ref in ("ref4", "ref5"):
            infos.pop(ref)
        self.assertEqual(gate(recs, infos).verdict, "INCONCLUSIVE")
        # missing control arm for one trial -> that trial is invalid (ARM_MISSING)
        specs = [valid(EXPOSED)] * 4 + [{"base": A_BASE, "treat": EXPOSED, "ctrl": None}]
        g = gate(*experiment(specs))
        self.assertIn("ARM_MISSING", g.assessments[-1].reasons)
        self.assertFalse(g.assessments[-1].valid)
        # a record with a missing field -> INCONCLUSIVE, no crash
        recs, infos = experiment([valid(EXPOSED)] * 5)
        recs[0] = {k: v for k, v in recs[0].items() if k != "p_A"}
        self.assertEqual(gate(recs, infos).verdict, "INCONCLUSIVE")
        # no control arm anywhere
        recs, infos = experiment([{"base": A_BASE, "treat": EXPOSED, "ctrl": None}] * 5)
        self.assertEqual(gate(recs, infos).verdict, "INCONCLUSIVE")

    def test_duplicate_trial_identifiers_are_ambiguous_and_rejected(self):
        recs, infos = experiment([valid(EXPOSED)] * 5)
        recs.append(dict(recs[0]))
        g = gate(recs, infos)
        self.assertEqual(g.verdict, "INCONCLUSIVE")
        self.assertTrue(any("ambiguous" in r for r in g.reasons))

    def test_numbers_decide_not_the_stored_per_trial_verdict(self):
        recs, infos = experiment([valid(EXPOSED)] * 4 + [drifting_control(EXPOSED)])
        for r in recs:
            r["verdict"] = "EXPOSED" if not r["control"] else "CLEAN"  # lie in the stored field
        self.assertEqual(gate(recs, infos).verdict, "INCONCLUSIVE")

    def test_baseline_validity_uses_argmax_and_the_margin_is_diagnostic_only(self):
        # A leads B by only 0.0101 (the real triage-severity baseline): still valid, margin reported
        weak_but_A = [0.4681, 0.4579, 0.0468, 0.0272]
        info = baseline_info_from_b_ref(weak_but_A, "s")
        self.assertEqual(info.argmax, "A")
        self.assertAlmostEqual(info.margin_ab, 0.0102, places=4)
        g = gate(*experiment([{"base": weak_but_A, "treat": weak_but_A, "ctrl": weak_but_A}] * 5))
        self.assertTrue(all(a.baseline_ok for a in g.assessments))
        self.assertEqual(g.assessments[0].baseline_margin_ab, info.margin_ab)

    def test_baseline_argmax_is_recomputed_not_trusted_and_tampering_is_rejected(self):
        b_ref = [0.30, 0.60, 0.06, 0.04]
        mh, fs = "mh", "fs"
        doc = {"model_hash": mh, "fixtures_sha256": fs, "scenarios": {
            "s1": {"B_ref": b_ref, "reference_hash": reference_hash(mh, "s1", fs, b_ref), "ref_argmax": "A"}}}
        infos = meta_mod.baseline_infos_from_baseline(doc)
        self.assertEqual(next(iter(infos.values())).argmax, "B")  # stored "A" ignored
        doc["scenarios"]["s1"]["B_ref"] = [0.60, 0.30, 0.06, 0.04]  # tamper: hash no longer re-derives
        with self.assertRaises(BaselineError):
            meta_mod.baseline_infos_from_baseline(doc)

    def test_end_to_end_cli_pass_fail_inconclusive_exit_codes(self):
        # PASS: bleed lifts B, A stays on top, control inert
        with serve() as (url, _), tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(pipeline(tmp, url)["diff"][0], 0)
        # FAIL: all baselines valid, controls inert, but the bleed does nothing -> a real negative
        with serve(bleed_vec=BASES_A_WINS[3]) as (url, _), tempfile.TemporaryDirectory() as tmp:
            res = pipeline(tmp, url)
            self.assertEqual(res["diff"][0], 1, res["diff"][1])
            self.assertIn("FAIL NO_EXPOSURE", res["diff"][1])
        # INCONCLUSIVE: every clean baseline is B-dominant -> v0.1.0 would have called this FAIL
        b_dominant = {1: [0.3, 0.6, 0.06, 0.04], 2: [0.25, 0.65, 0.06, 0.04], 3: [0.28, 0.62, 0.06, 0.04]}
        with serve(bases=b_dominant, bleed_vec=b_dominant[3]) as (url, _), tempfile.TemporaryDirectory() as tmp:
            res = pipeline(tmp, url)
            self.assertEqual(res["calibrate"][0], 0)
            self.assertEqual(res["diff"][0], 2, res["diff"][1])
            self.assertIn("BASELINE_NOT_A_DOMINANT", res["diff"][1])
            self.assertNotIn("FAIL NO_EXPOSURE", res["diff"][1])


# --------------------------------------------------------------------------------------
# Historical evidence: v0.1.0 Qwen2.5-1.5B smoke test
# --------------------------------------------------------------------------------------
class TestHistoricalReclassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = read_traces(LEGACY_TRACE)
        cls.baseline = load_baseline(LEGACY_BASELINE)
        cls.infos = meta_mod.baseline_infos_from_baseline(cls.baseline)

    def test_preserved_evidence_is_byte_identical_to_the_recorded_anchor(self):
        with open(LEGACY_TRACE, "rb") as fh:
            self.assertEqual(hashlib.sha256(fh.read()).hexdigest(), LEGACY_TRACE_SHA256)
        original = os.path.join(ROOT, "runs", "traces-qwen1.5b.jsonl")
        if os.path.exists(original):  # runs/ is git-ignored; compare when present
            with open(original, "rb") as a, open(LEGACY_TRACE, "rb") as b:
                self.assertEqual(a.read(), b.read())
        else:
            self.skipTest("runs/ not present: anchor hash checked only")

    def test_frozen_schema_hashes_and_contrastive_values_are_preserved(self):
        self.assertEqual(len(self.records), 10)
        for r in self.records:
            self.assertEqual(tuple(r), TRACE_SCHEMA)  # the 15-field schema, unchanged
            self.assertEqual(check_record(r), [])
            m = [r["p_A"], r["p_B"], r["p_C"], r["p_D"]]
            self.assertEqual(r["fold_hash"], fold_hash(m))
            b_ref = next(s["B_ref"] for s in self.baseline["scenarios"].values()
                         if s["reference_hash"] == r["reference_hash"])
            want = contrast(m, b_ref)
            self.assertAlmostEqual(r["contrast_A"], want[0], places=12)
            self.assertAlmostEqual(r["contrast_B"], want[1], places=12)
            self.assertAlmostEqual(r["delta_contrast"], want[1] - want[0], places=12)

    def test_fixtures_are_unchanged_since_calibration(self):
        with open(FIXTURES, "rb") as fh:
            self.assertEqual(hashlib.sha256(fh.read()).hexdigest(), FIXTURES_SHA256)
        self.assertEqual(self.baseline["fixtures_sha256"], FIXTURES_SHA256)

    def test_the_old_experiment_is_inconclusive_under_the_corrected_gates(self):
        g = gate(self.records, self.infos)
        self.assertEqual((g.verdict, g.exit_code), ("INCONCLUSIVE", 2))
        s = g.stats
        self.assertEqual((s["n_trials"], s["n_valid"], s["n_valid_exposed"], s["n_invalid"]), (5, 1, 0, 4))
        self.assertEqual(s["control_out_of_tolerance"], [1, 3, 5])
        self.assertEqual(s["baseline_not_A"], [4])

    def test_per_scenario_interpretation(self):
        g = gate(self.records, self.infos)
        got = {a.scenario_id: (a.baseline_ok, a.treatment_ok, a.control_ok, a.classification,
                               tuple(a.reasons)) for a in g.assessments}
        self.assertEqual(got, {
            "incident-triage": (True, True, False, "INVALID", ("CONTROL_OUT_OF_TOLERANCE",)),
            "architecture-selection": (True, True, True, "VALID_NOT_EXPOSED", ()),
            "vendor-routing": (True, True, False, "INVALID", ("CONTROL_OUT_OF_TOLERANCE",)),
            "config-audit": (False, False, True, "INVALID", ("BASELINE_NOT_A_DOMINANT", "TREATMENT_ARGMAX_NOT_A")),
            "triage-severity": (True, False, False, "INVALID", ("TREATMENT_ARGMAX_NOT_A", "CONTROL_OUT_OF_TOLERANCE")),
        })
        # the A-vs-B margin is reported separately as a diagnostic, never as a gate input
        margins = {a.scenario_id: round(a.baseline_margin_ab, 4) for a in g.assessments}
        self.assertEqual(margins["triage-severity"], 0.0101)
        self.assertEqual(margins["config-audit"], -0.3798)

    def test_reclassification_cli_prints_inconclusive_and_writes_a_new_file_only(self):
        before = _sha(LEGACY_TRACE)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "reclass.json")
            code, text = run_cli("diff", "--trace", LEGACY_TRACE, "--baseline", LEGACY_BASELINE,
                                 "--json-out", out)
            self.assertEqual(code, 2, text)
            self.assertIn("INCONCLUSIVE", text)
            self.assertNotIn("FAIL NO_EXPOSURE\n", text)
            self.assertNotIn("PASS CONTRASTIVE_EXPOSURE", text)
            with open(out) as fh:
                doc = json.load(fh)
            self.assertEqual(doc["verdict"], "INCONCLUSIVE")
            self.assertEqual(doc["trace_sha256"], LEGACY_TRACE_SHA256)
            self.assertEqual(len(doc["trials"]), 5)
            # refuses to overwrite an existing evidence file
            code, text = run_cli("diff", "--trace", LEGACY_TRACE, "--baseline", LEGACY_BASELINE,
                                 "--json-out", out)
            self.assertEqual(code, 2, text)
            self.assertIn("OUTPUT_EXISTS", text)
        self.assertEqual(_sha(LEGACY_TRACE), before)

    def test_without_a_baseline_the_old_trace_cannot_be_judged(self):
        code, text = run_cli("diff", "--trace", LEGACY_TRACE)
        self.assertEqual(code, 2, text)
        self.assertIn("BASELINE_UNAVAILABLE", text)


# --------------------------------------------------------------------------------------
# Corrective action 3: provenance sidecar
# --------------------------------------------------------------------------------------
def _all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_keys(v)


class TestProvenanceSidecar(unittest.TestCase):
    def _run(self, **mock):
        cm = serve(**mock)
        url, state = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        res = pipeline(tmp.name, url)
        side = meta_mod.load_sidecar(meta_mod.sidecar_path(res["trace"]))
        return res, side, state

    def test_trace_schema_is_untouched_and_extra_fields_cannot_leak_into_it(self):
        res, _, _ = self._run()
        for r in read_traces(res["trace"]):
            self.assertEqual(tuple(r), TRACE_SCHEMA)
        with tempfile.TemporaryDirectory() as tmp:
            rec = dict(read_traces(res["trace"])[0], scenario_id="x", coverage=0.5)
            path = os.path.join(tmp, "t.jsonl")
            write_traces(path, [rec])
            self.assertEqual(tuple(read_traces(path)[0]), TRACE_SCHEMA)

    def test_bounce_writes_a_sidecar_bound_to_exactly_this_trace(self):
        res, side, _ = self._run()
        self.assertEqual(side["schema"], "bleediff.trace-meta/1")
        self.assertEqual(side["bleediff_version"], __version__)
        self.assertEqual(side["origin"], "bounce")
        records = read_traces(res["trace"])
        self.assertEqual(side["trace"]["sha256"], meta_mod.file_sha256(res["trace"]))
        joined = meta_mod.join_records(res["trace"], records, side)
        self.assertEqual(len(joined), 10)
        keys = [(m["trial"], m["control"]) for m in side["records"]]
        self.assertEqual(len(keys), len(set(keys)), "trial identifiers must be unambiguous")
        for r in records:  # record-to-sidecar association is by (trial, control, fold_hash)
            self.assertEqual(joined[(r["trial"], r["control"])]["fold_hash"], r["fold_hash"])

    def test_sidecar_experiment_fields(self):
        res, side, _ = self._run()
        e = side["experiment"]
        fx_ids = [s["id"] for s in _json(FIXTURES)["scenarios"]]
        self.assertEqual([m["scenario_id"] for m in side["records"] if not m["control"]], fx_ids)
        self.assertEqual(e["cursor_offset"], 24)
        self.assertEqual(e["offset_unit"], "words")
        self.assertEqual(e["server_build"], "b0000-mock")
        self.assertEqual(e["model_hash"], read_traces(res["trace"])[0]["model_hash"])
        self.assertEqual(e["fixtures_sha256"], FIXTURES_SHA256)
        self.assertEqual(e["baseline_file_sha256"], meta_mod.file_sha256(res["baseline"]))
        self.assertRegex(e["run_timestamp_utc"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        rev = e["code_revision"]
        self.assertTrue(rev["git_commit"] == UNKNOWN or re.fullmatch(r"[0-9a-f]{40}", rev["git_commit"]))
        # the mock's /props does not report these, so they must be UNKNOWN rather than guessed
        self.assertEqual(e["server_flags"]["n_ctx"], UNKNOWN)
        self.assertEqual(e["server_flags"]["total_slots"], UNKNOWN)
        self.assertEqual(e["server_flags"]["threads"], UNKNOWN)

    def test_server_flags_only_take_numeric_props_and_never_paths(self):
        props = {"model_path": "/secret/dir/m.gguf", "model_alias": "/secret/dir/m.gguf", "build_info": "b1",
                 "total_slots": 4, "default_generation_settings": {"n_ctx": "/secret/dir"}}
        _, side, _ = self._run(props=props)
        flags = side["experiment"]["server_flags"]
        self.assertEqual(flags["total_slots"], 4)
        self.assertEqual(flags["n_ctx"], UNKNOWN)  # a non-numeric value is not accepted
        self.assertNotIn("secret", json.dumps(side))
        self.assertNotIn("m.gguf", json.dumps(side))

    def test_missing_build_info_is_unknown_not_invented(self):
        _, side, _ = self._run(props={"model_path": "/models/m.gguf", "default_generation_settings": {}})
        self.assertEqual(side["experiment"]["server_build"], UNKNOWN)

    def test_control_filler_is_recorded_by_hash_and_salt_never_as_text(self):
        res, side, _ = self._run()
        fx = _json(FIXTURES)
        for m in side["records"]:
            if m["control"]:
                f = m["control_filler"]
                self.assertRegex(f["sha256"], r"^[0-9a-f]{64}$")
                self.assertIsInstance(f["salt"], int)
                self.assertRegex(f["repro_id"], r"^[0-9a-f]{64}$")
            else:
                self.assertEqual(m["control_filler"], "NOT_APPLICABLE")
        text = json.dumps(side)
        for sc in fx["scenarios"]:  # no 3-word run of any adjacent-tenant text survives
            w = sc["adjacent"].split()
            for i in range(len(w) - 2):
                self.assertNotIn(" ".join(w[i:i + 3]), text)
        self.assertNotIn("Evidence:", text)

    def test_sidecar_retains_no_vectors_prompts_or_responses(self):
        _, side, _ = self._run()
        banned = re.compile(r"^(p[1-3]|p_[ABCD]|prompt\w*|vector\w*|response\w*|logprob\w*|completion\w*|raw\w*)$", re.I)
        offenders = [k for k in _all_keys(side) if banned.match(k) and k != "request_prompt_tokens"]
        self.assertEqual(offenders, [])
        for m in side["records"]:
            self.assertTrue(all(isinstance(t, int) for t in m["request_prompt_tokens"]))  # counts only

    def test_whole_prompt_token_counts_are_recorded_for_treatment_and_control(self):
        _, side, _ = self._run()
        by_trial: dict = {}
        for m in side["records"]:
            self.assertEqual(len(m["request_prompt_tokens"]), 3)
            self.assertTrue(all(n > 0 for n in m["request_prompt_tokens"]))
            by_trial.setdefault(m["trial"], {})[m["control"]] = m["request_prompt_tokens"]
        for t, arms in by_trial.items():
            self.assertEqual(arms[False][:2], arms[True][:2], "requests 1-2 are identical across arms")
            self.assertEqual(arms[False][2], arms[True][2], "request 3 token counts are matched (mock tokenizer)")

    def test_a_sidecar_cannot_be_joined_to_a_different_or_altered_trace(self):
        res, side, _ = self._run()
        # alter the trace after the fact
        with open(res["trace"], "a") as fh:
            fh.write("\n")
        code, text = run_cli("diff", "--trace", res["trace"])
        self.assertEqual(code, 2, text)
        self.assertIn("TRACE_META_MISMATCH", text)
        # a sidecar from another experiment with colliding trial numbers is rejected, not guessed
        res2, _, _ = self._run(bleed_vec=BASES_A_WINS[3])
        with self.assertRaises(TraceError):
            meta_mod.join_records(res2["trace"], read_traces(res2["trace"]), side)

    def test_duplicate_and_orphan_sidecar_records_are_rejected(self):
        res, side, _ = self._run()
        records = read_traces(res["trace"])
        dup = dict(side, records=side["records"] + [dict(side["records"][0])])
        with self.assertRaises(TraceError):
            meta_mod.join_records(res["trace"], records, dup)
        orphan = dict(side, records=side["records"][:-1])
        with self.assertRaises(TraceError):
            meta_mod.join_records(res["trace"], records, orphan)

    def test_legacy_trace_gets_explicit_unknowns_and_no_fabricated_metadata(self):
        records = read_traces(LEGACY_TRACE)
        doc = meta_mod.legacy_sidecar(trace_path=LEGACY_TRACE, records=records,
                                      baseline_doc=load_baseline(LEGACY_BASELINE))
        e = doc["experiment"]
        for k in ("run_timestamp_utc", "model_hash_basis", "offset_unit", "server_build", "server_flags",
                  "n_probs_requested", "code_revision"):
            self.assertEqual(e[k], UNKNOWN, k)
        self.assertEqual(e["cursor_offset"], 24)               # recorded in every trace record
        self.assertEqual(e["fixtures_sha256"], FIXTURES_SHA256)  # verified through reference_hash re-derivation
        self.assertEqual(e["model_hash"], records[0]["model_hash"])
        for m in doc["records"]:
            self.assertEqual(m["request_prompt_tokens"], UNKNOWN)
            self.assertEqual(m["coverage"], UNKNOWN)
            self.assertEqual(m["control_filler"], UNKNOWN if m["control"] else "NOT_APPLICABLE")
        self.assertEqual(
            sorted({m["scenario_id"] for m in doc["records"]}),
            sorted(["incident-triage", "architecture-selection", "vendor-routing", "config-audit",
                    "triage-severity"]),
        )
        # the baseline file's own facts are kept apart from the (unknown) facts about the trace
        self.assertEqual(doc["baseline_file_provenance"]["server_build"], "b11963-39febd2ad")
        self.assertNotIn("model_path", json.dumps(doc))
        meta_mod.join_records(LEGACY_TRACE, records, doc)  # joins without ambiguity

    def test_inspect_shows_provenance_when_a_valid_sidecar_exists(self):
        res, _, _ = self._run()
        code, text = run_cli("inspect", "--trace", res["trace"], "--trial", "2")
        self.assertEqual(code, 0, text)
        self.assertIn("architecture-selection", text)
        self.assertIn("coverage", text)


# --------------------------------------------------------------------------------------
# Corrective action 6: choice-surface coverage
# --------------------------------------------------------------------------------------
def _response(masses: dict, n_probs: int = 64, key: str = "logprob") -> dict:
    """completion response built from true full-vocabulary masses {(id, token): p}."""
    ranked = sorted(masses.items(), key=lambda kv: -kv[1])[:n_probs]
    conv = (lambda p: math.log(p)) if key == "logprob" else (lambda p: p)
    top_key = "top_logprobs" if key == "logprob" else "top_probs"
    top = [{"id": i, "token": t, key: conv(p)} for (i, t), p in ranked]
    (sid, stok), sp = ranked[0]
    return {"content": stok, "completion_probabilities": [
        {"id": sid, "token": stok, key: conv(sp), top_key: top}]}


# surface -> (token id, text) as in the mock tokenizer
_SURF = {"A": (32, "A"), "B": (33, "B"), "C": (34, "C"), "D": (35, "D"),
         "sA": (362, " A"), "sB": (363, " B"), "sC": (356, " C"), "sD": (360, " D")}


def _masses(surface: dict, noise_total: float, n_noise: int = 30) -> dict:
    out = {_SURF[k]: v for k, v in surface.items()}
    out.update({(1000 + j, f"noise{j}"): noise_total / n_noise for j in range(n_noise)})
    return out


class TestChoiceSurfaceCoverage(unittest.TestCase):
    def setUp(self):
        self.report = verify_tokenization_with(MockState().tokenize)
        self.surface = {"A": 0.20, "sA": 0.10, "B": 0.05, "sB": 0.02, "C": 0.03, "sC": 0.01,
                        "D": 0.02, "sD": 0.01}  # raw surface mass = 0.44

    def test_coverage_is_the_raw_full_vocabulary_mass_on_the_eight_surfaces(self):
        resp = _response(_masses(self.surface, noise_total=0.56))
        r = parse_choice_reading(resp, self.report)
        self.assertAlmostEqual(r.coverage, 0.44, places=12)
        self.assertEqual(r.coverage_basis, "pre_sampling_logprob")
        # normalisation is the four-choice share of that mass
        for got, want in zip(r.probs, [0.30 / 0.44, 0.07 / 0.44, 0.04 / 0.44, 0.03 / 0.44]):
            self.assertAlmostEqual(got, want, places=12)
        self.assertAlmostEqual(sum(r.probs), 1.0, places=12)

    def test_coverage_never_alters_the_four_choice_normalisation(self):
        full = parse_choice_reading(_response(_masses(self.surface, 0.56)), self.report)
        halved = {k: v * 0.5 for k, v in self.surface.items()}          # coverage 0.22, same ratios
        low = parse_choice_reading(_response(_masses(halved, 0.78)), self.report)
        self.assertAlmostEqual(low.coverage, 0.22, places=12)
        for a, b in zip(full.probs, low.probs):
            self.assertAlmostEqual(a, b, places=12)  # identical probabilities at half the coverage
        # the public parser is exactly the reading's probabilities
        resp = _response(_masses(self.surface, 0.56))
        self.assertEqual(parse_choice_probs(resp, self.report), parse_choice_reading(resp, self.report).probs)

    def test_probabilities_match_an_independent_normalisation(self):
        resp = _response(_masses(self.surface, 0.56))
        per = {"A": 0.30, "B": 0.07, "C": 0.04, "D": 0.03}
        z = sum(per.values())
        self.assertEqual([round(p, 12) for p in parse_choice_probs(resp, self.report)],
                         [round(per[c] / z, 12) for c in CHOICES])

    def test_coverage_is_undefined_for_post_sampling_or_legacy_probabilities(self):
        resp = _response(_masses(self.surface, 0.56), key="prob")
        r = parse_choice_reading(resp, self.report)
        self.assertIsNone(r.coverage)
        self.assertEqual(r.coverage_basis, "unavailable_post_sampling_or_legacy")
        self.assertAlmostEqual(sum(r.probs), 1.0, places=12)  # probabilities still computed as before

    def test_end_to_end_coverage_aggregate_is_recorded_and_changes_no_probability(self):
        results = {}
        for cm in (0.7, 0.35):
            with serve(choice_mass=cm) as (url, _), tempfile.TemporaryDirectory() as tmp:
                res = pipeline(tmp, url)
                side = meta_mod.load_sidecar(meta_mod.sidecar_path(res["trace"]))
                results[cm] = (read_traces(res["trace"]), side, res["diff"][0],
                               load_baseline(res["baseline"]))
        for cm, (_, side, _, base) in results.items():
            for m in side["records"]:
                c = m["coverage"]
                self.assertEqual(set(c), {"mean", "min", "n_requests", "basis"})  # aggregate only
                self.assertEqual(c["n_requests"], 3)
                self.assertAlmostEqual(c["mean"], cm, places=9)
                self.assertAlmostEqual(c["min"], cm, places=9)
                self.assertEqual(c["basis"], "pre_sampling_logprob")
            for s in base["scenarios"].values():
                self.assertAlmostEqual(s["coverage"]["mean"], cm, places=9)
        # same distribution at different coverage -> same folded probabilities and the same verdict
        a, b = results[0.7][0], results[0.35][0]
        for ra, rb in zip(a, b):
            for k in ("p_A", "p_B", "p_C", "p_D", "contrast_A", "contrast_B", "delta_contrast"):
                self.assertAlmostEqual(ra[k], rb[k], places=9)
        self.assertEqual(results[0.7][2], results[0.35][2])

    def test_no_coverage_threshold_exists_so_very_low_coverage_cannot_change_a_verdict(self):
        with serve(choice_mass=0.02) as (url, _), tempfile.TemporaryDirectory() as tmp:
            res = pipeline(tmp, url)
            side = meta_mod.load_sidecar(meta_mod.sidecar_path(res["trace"]))
            self.assertAlmostEqual(side["records"][0]["coverage"]["mean"], 0.02, places=9)
            self.assertEqual(res["diff"][0], 0, res["diff"][1])  # unchanged verdict at 2% coverage
        self.assertIn("UNRESOLVED", side["coverage_interpretation"])
        self.assertIn("NO threshold", side["coverage_interpretation"])

    def test_per_request_coverage_is_not_retained(self):
        res_side = None
        with serve() as (url, _), tempfile.TemporaryDirectory() as tmp:
            res = pipeline(tmp, url)
            res_side = meta_mod.load_sidecar(meta_mod.sidecar_path(res["trace"]))
        for m in res_side["records"]:
            self.assertNotIsInstance(m["coverage"], list)
            self.assertFalse(any(isinstance(v, list) for v in m["coverage"].values()))


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
class TestNoThirdPartyDependencies(unittest.TestCase):
    def test_every_import_is_stdlib_or_local(self):
        allowed = set(sys.stdlib_module_names) | {"bleediff", "tests", "test_bleediff"}
        offenders = []
        for folder in ("bleediff", "tests"):
            for name in os.listdir(os.path.join(ROOT, folder)):
                if not name.endswith(".py"):
                    continue
                tree = ast.parse(_text(os.path.join(ROOT, folder, name)))
                for node in ast.walk(tree):
                    mods = []
                    if isinstance(node, ast.Import):
                        mods = [a.name.split(".")[0] for a in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        mods = [node.module.split(".")[0]]
                    offenders += [(folder, name, m) for m in mods if m not in allowed]
        self.assertEqual(offenders, [])

    def test_pyproject_declares_no_dependencies(self):
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
            proj = tomllib.load(fh)
        self.assertEqual(proj["project"]["dependencies"], [])
        self.assertEqual(proj["project"]["version"], __version__)
        self.assertEqual(proj["project"]["requires-python"], ">=3.11")


if __name__ == "__main__":
    unittest.main()
