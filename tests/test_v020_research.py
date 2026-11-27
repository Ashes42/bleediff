"""v0.2.0 tests: the promoted research drivers, the tracked evidence they were checked against, and the documented findings.

No model and no network beyond a stdlib mock llama-server on 127.0.0.1. The saved evidence in research/data/ are copies of the
traces of real runs (Qwen2.5-1.5B and Qwen2.5-7B); these tests recompute the frozen equations from them and check that the
driver's planning and analysis reproduce the documented numbers. They do not rerun any experiment.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import unittest
import contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

try:
    from test_bleediff import serve  # type: ignore
except ImportError:  # pragma: no cover
    from tests.test_bleediff import serve

from bleediff import __version__  # noqa: E402
from bleediff import robustness as RB  # noqa: E402
from bleediff.arena import load_fixtures, plan_requests  # noqa: E402
from bleediff.fold import TRACE_SCHEMA, contrast, fold_hash, read_traces, reference_hash  # noqa: E402
from bleediff.meta import file_sha256, join_records, load_sidecar, sidecar_path  # noqa: E402
from research import arms  # noqa: E402
from research import measure_fragments as MF  # noqa: E402
from research import run_arms  # noqa: E402

DATA = os.path.join(ROOT, "research", "data")
FROZEN = os.path.join(DATA, "frozen-arms.json")
CONTROLS = os.path.join(DATA, "neutral-controls-replicates-manifest.json")
RESULTS = os.path.join(DATA, "results-v0.2.0.json")
FIXTURES = os.path.join(ROOT, "fixtures", "semantic-v2", "contexts.json")
M15, M7 = "qwen2.5-1.5b-instruct-q4_k_m", "qwen2.5-7b-instruct-q4_k_m"


def _json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def scenario():
    return {s.id: s for s in load_fixtures(FIXTURES).scenarios}[arms.SCENARIO_ID]


def results():
    return _json(RESULTS)


NAMES_7B = list(arms.ARM_ORDER) + [f"control-{k:02d}" for k in range(12)]
NAMES_15B = ["B_original"] + [f"control-{k:02d}" for k in range(12)]


# --------------------------------------------------------------------------------------
class TestRequestPlans(unittest.TestCase):
    def setUp(self):
        self.sc = scenario()
        self.txt = arms.load_frozen_arms(FROZEN)
        self.controls = arms.load_controls(CONTROLS)
        self.arms = arms.build_arms(self.sc, self.txt, self.controls)

    def test_five_frozen_texts_each_170_bytes_25_words(self):
        self.assertEqual(list(self.txt), list(arms.ARM_ORDER))
        for name, t in self.txt.items():
            self.assertEqual((len(t.encode()), len(t.split())), (170, 25), name)
            self.assertTrue(t.startswith("Tenant 4 compliance scan finding: Control SEC-21 "), name)
        original = " ".join(self.sc.adjacent.split()[:25])
        self.assertEqual(self.txt["B_original"], original)                    # the frozen config-audit-v2 treatment, unedited

    def test_arm_order_and_count(self):
        self.assertEqual([a.name for a in self.arms], NAMES_7B)
        self.assertEqual([a.kind for a in self.arms], ["treatment"] * 5 + ["control"] * 12)

    def test_arms_differ_only_in_the_window(self):
        base = arms.plan_for(self.arms[0])
        base_prompts = [bytes(p) for p in base.prompts]
        base.dispose()
        for a in self.arms[1:]:
            plan = arms.plan_for(a)
            try:
                got = [bytes(p) for p in plan.prompts]
                self.assertEqual(len(got), 3, a.name)
                self.assertEqual(got[:2], base_prompts[:2], a.name)               # requests 1 and 2 are identical in every arm
                self.assertNotEqual(got[2], base_prompts[2], a.name)
                self.assertEqual(plan.geometry.bleed_words, 25, a.name)
                self.assertEqual(plan.geometry.bleed_bytes, 170, a.name)
                text = a.control_entry["fragment_text"] if a.kind == "control" else self.txt[a.name]
                self.assertEqual(plan.bleed_fragment.decode(), text, a.name)
                head, tail = base_prompts[2].split(self.txt["B_original"].encode())
                self.assertEqual(got[2], head + text.encode() + tail, a.name)     # byte-identical except the window text
            finally:
                plan.dispose()

    def test_controls_come_from_the_verified_manifest(self):
        ids = [e["control_id"][:12] for _k, _d, e in self.controls]
        self.assertEqual(len(set(ids)), 12)
        self.assertEqual([k for k, _d, _e in self.controls], list(range(12)))
        for _k, _d, e in self.controls:
            self.assertEqual((len(e["fragment_text"].encode()), len(e["fragment_text"].split())), (170, 25))
            self.assertEqual(e["token_parity"], "VERIFIED")
        self.assertEqual(_sha(CONTROLS), "01bce5b24075cbbf" + _sha(CONTROLS)[16:])

    def test_a_modified_or_misplaced_window_is_refused(self):
        too_short = " ".join(self.txt["P_paraphrase"].split()[:24])
        with self.assertRaises(ValueError):
            arms.scenario_variants(self.sc, {"X": too_short})                    # 24 words: the 25-word window would run into the trailer
        with tempfile.TemporaryDirectory() as d:
            doc = _json(FROZEN)
            doc["arms"]["P_paraphrase"]["text"] += " tampered"
            path = os.path.join(d, "f.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            with self.assertRaises(ValueError):
                arms.load_frozen_arms(path)                                       # recorded hash no longer matches

    def test_the_manifest_must_hold_twelve_accepted_controls(self):
        with tempfile.TemporaryDirectory() as d:
            doc = _json(CONTROLS)
            doc["selection"]["replicates"][4]["status"] = "FAILED"
            path = os.path.join(d, "m.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            with self.assertRaises(ValueError):
                arms.load_controls(path)


# --------------------------------------------------------------------------------------
def _records(name):
    return read_traces(os.path.join(DATA, name))


class TestSavedEvidenceAndAnalysis(unittest.TestCase):
    def test_every_saved_trace_has_the_frozen_schema_and_reproduces_the_equations(self):
        res = results()
        files = {"evidence-qwen1.5b-12-controls-pass1.jsonl": M15, "evidence-qwen1.5b-lexical-probe-pass1.jsonl": M15,
                 "evidence-qwen1.5b-cue-probe-pass1.jsonl": M15, "evidence-qwen2.5-7b-pass1.jsonl": M7}
        for fname, model in files.items():
            m = res["models"][model]
            b_ref = m["baseline"]["B_ref"]
            for rec in _records(fname):
                self.assertEqual(tuple(rec), TRACE_SCHEMA)
                vec = [rec["p_A"], rec["p_B"], rec["p_C"], rec["p_D"]]
                rc = contrast(vec, b_ref)
                self.assertAlmostEqual(rc[1] - rc[0], rec["delta_contrast"], places=12, msg=fname)
                self.assertEqual(rec["fold_hash"], fold_hash(vec), fname)
                self.assertEqual(rec["reference_hash"], m["baseline"]["reference_hash"], fname)
                self.assertEqual(rec["raw_argmax"], "ABCD"[max(range(4), key=lambda i: (vec[i], -i))])
                self.assertAlmostEqual(sum(vec), 1.0, places=9)
            sc_ref = reference_hash(rec["model_hash"], arms.SCENARIO_ID, "b733148e71510a0b30ed5366e470befe72353d05b4a34408eaf45f3cec0d3ceb", b_ref)
            self.assertEqual(sc_ref, m["baseline"]["reference_hash"], fname)       # the baseline identity recomputes from the data

    def test_the_two_passes_of_every_experiment_were_byte_identical(self):
        self.assertTrue(all(results()["repeatability_pass1_equals_pass2_bytes"].values()))
        self.assertEqual(len(results()["repeatability_pass1_equals_pass2_bytes"]), 4)

    def test_summarize_reproduces_the_1p5b_numbers(self):
        recs = _records("evidence-qwen1.5b-12-controls-pass1.jsonl")
        s = arms.summarize(NAMES_15B, recs, recs)
        cr = s["control_range"]
        self.assertAlmostEqual(cr["min"], -0.2409, places=4)
        self.assertAlmostEqual(cr["max"], 0.0607, places=4)
        self.assertAlmostEqual(cr["median"], -0.1154, places=4)
        self.assertEqual(cr["n_within_0.1"], 5)
        b = s["arms"]["B_original"]
        self.assertAlmostEqual(b["delta_contrast"], 0.8569, places=4)
        self.assertEqual((b["raw_argmax"], b["label"]), ("A", "HIGH"))
        self.assertEqual(b["position_vs_neutral_controls"], "above the neutral-control range")
        self.assertTrue(s["repeatability_bit_identical"])
        self.assertIsNone(s["cue_table"])                                            # not all four cue arms exist in this run

    def test_summarize_reproduces_the_7b_numbers_including_the_raw_flip(self):
        recs = _records("evidence-qwen2.5-7b-pass1.jsonl")
        s = arms.summarize(NAMES_7B, recs, recs)
        a, cr = s["arms"], s["control_range"]
        self.assertAlmostEqual(cr["min"], -0.6977, places=4)
        self.assertAlmostEqual(cr["max"], -0.6727, places=4)
        self.assertEqual(cr["n_within_0.1"], 0)
        expect = {"B_original": (0.6542, "B", "HIGH"), "P_paraphrase": (0.4793, "B", "HIGH"), "N_polarity_reversed": (-0.6683, "A", "INTERMEDIATE"),
                  "CueA_failed_but_disabled": (0.4004, "B", "HIGH"), "CueB_passed_but_enabled": (0.2609, "B", "INTERMEDIATE")}
        for name, (d, am, lab) in expect.items():
            self.assertAlmostEqual(a[name]["delta_contrast"], d, places=4, msg=name)
            self.assertEqual((a[name]["raw_argmax"], a[name]["label"]), (am, lab), name)
        self.assertAlmostEqual(a["P_paraphrase"]["difference_from_original_B"], 0.4793 - 0.6542, places=4)

    def test_probe_only_runs_use_the_supplied_reference_and_no_longer_crash(self):
        """FIX (v0.2.0): a run without B_original/controls (the cue probes) failed at its summary step in the untracked driver."""
        recs = _records("evidence-qwen1.5b-cue-probe-pass1.jsonl")
        names = ["CueA_failed_but_disabled", "CueB_passed_but_enabled"]
        ref = results()["models"][M15]["measurements"]
        rng = {"min": ref["control_stats"]["min"], "max": ref["control_stats"]["max"], "median": ref["control_stats"]["median"]}
        s = arms.summarize(names, recs, recs, reference_range=rng, reference_original_delta=ref["treatment_original_B"]["delta_contrast"])
        self.assertAlmostEqual(s["arms"][names[0]]["delta_contrast"], 0.4093, places=4)
        self.assertAlmostEqual(s["arms"][names[1]]["delta_contrast"], 0.7258, places=4)
        self.assertAlmostEqual(s["arms"][names[0]]["difference_from_original_B"], 0.4093 - 0.8569, places=4)
        self.assertEqual([s["arms"][n]["label"] for n in names], ["HIGH", "HIGH"])
        self.assertIsNone(s["cue_table"])
        self.assertIn("not measured in this run", s["control_range"]["source"])
        s2 = arms.summarize(names, recs, recs)                                      # and with no reference at all it still does not crash
        self.assertIsNone(s2["control_range"])
        self.assertIsNone(s2["arms"][names[0]]["label"])
        self.assertIsNone(s2["arms"][names[0]]["difference_from_original_B"])

    def test_cue_interactions_are_not_additive(self):
        for model, want in ((M15, -0.4610), (M7, -0.6754)):
            ct = results()["models"][model]["measurements"]["cue_table"]
            self.assertAlmostEqual(ct["interaction_contrast"], want, places=4)
            self.assertGreater(abs(ct["interaction_contrast"]), 0.4)                # additive cues would give 0
        recs = _records("evidence-qwen2.5-7b-pass1.jsonl")
        s = arms.summarize(NAMES_7B, recs, recs)
        self.assertAlmostEqual(s["cue_table"]["interaction_contrast"], -0.6754, places=4)
        self.assertTrue(arms.cue_table({"B_original": 1.0}) is None)
        self.assertEqual(arms.cue_table({"B_original": 1, "CueA_failed_but_disabled": 1, "CueB_passed_but_enabled": 1, "N_polarity_reversed": 1})["interaction_contrast"], 0)

    def test_frozen_robustness_analysis_labels_on_both_models(self):
        res = results()
        for model, names, fname, label in ((M15, NAMES_15B, "evidence-qwen1.5b-12-controls-pass1.jsonl", "SEPARATED_FROM_PLACEBO_EXPLORATORY"),
                                           (M7, NAMES_7B, "evidence-qwen2.5-7b-pass1.jsonl", "TREATMENT_PRECONDITION_FAILED")):
            recs = _records(fname)
            ctrls = arms.load_controls(CONTROLS)
            obs = arms.robustness_observations(ctrls, names, recs, res["models"][model]["baseline"]["argmax"])
            out = RB.analyze(obs)
            self.assertEqual(out["label"], label, model)
            self.assertEqual(out["descriptive"]["n_valid"], 12)
            self.assertEqual(out["inferential_diagnostics"]["rank_exceedance"]["controls_at_or_above_treatment_signed"], 0, model)
            self.assertIn("No label here is a PASS.", out["interpretation_limits"])

    def test_documented_findings_hold_in_the_saved_results(self):
        res = results()
        m15, m7 = res["models"][M15]["measurements"], res["models"][M7]["measurements"]
        for m in (m15, m7):
            cs = [c["delta_contrast"] for c in m["controls"]]
            self.assertGreater(m["treatment_original_B"]["delta_contrast"], max(cs))        # exceeds all 12 neutral controls
            self.assertGreater(m["paraphrase"]["delta_contrast"], max(cs))
        self.assertEqual({m15[k]["raw_argmax"] for k in ("treatment_original_B", "paraphrase", "polarity_reversed", "cue_failed_but_disabled", "cue_passed_but_enabled")} |
                         {c["raw_argmax"] for c in m15["controls"]}, {"A"})                    # 1.5B: A stays the raw argmax everywhere
        self.assertEqual(m7["treatment_original_B"]["raw_argmax"], "B")                          # 7B: the raw answer flips to B
        self.assertEqual(m7["polarity_reversed"]["raw_argmax"], "A")
        self.assertEqual(res["models"][M15]["baseline"]["argmax"], "A")
        self.assertEqual(res["models"][M7]["baseline"]["argmax"], "A")
        self.assertLess(res["models"][M7]["baseline"]["ab_margin"], 0.12)                        # a thin baseline margin
        self.assertEqual((m15["control_stats"]["n_within_0.1"], m7["control_stats"]["n_within_0.1"]), (5, 0))   # the +-0.1 tolerance failed often
        self.assertLess(m7["control_stats"]["max"], -0.6)                                        # the 7B neutral distribution is far from zero
        self.assertEqual(res["five_scenario_semantic_v2_experiment"]["verdict"], "INCONCLUSIVE")
        self.assertIn("EXPLORATORY", res["status"])

    def test_tracked_data_matches_the_recorded_hashes(self):
        res = results()
        for name, sha in res["tracked_copies_sha256"].items():
            self.assertEqual(_sha(os.path.join(DATA, name)), sha, name)


# --------------------------------------------------------------------------------------
class TestDriverEndToEnd(unittest.TestCase):
    """A full driver run against the stdlib mock server: schema, sidecars, repeatability and refusal to overwrite."""

    def _run(self, out, **kw):
        with serve() as (url, _state), contextlib.redirect_stdout(io.StringIO()):
            return arms.run(server=url, out_dir=out, frozen_path=FROZEN, controls_path=CONTROLS, fixtures_path=FIXTURES, timeout=60, **kw)

    def test_full_run_writes_schema_valid_bound_traces_and_an_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "run")
            res = self._run(out)
            self.assertTrue(res["repeatability_bit_identical"])
            for p in (1, 2):
                tr = os.path.join(out, f"traces-pass{p}.jsonl")
                recs = read_traces(tr)
                self.assertEqual(len(recs), 17)                                         # 5 treatment-type arms + 12 controls
                self.assertEqual([r["control"] for r in recs], [False] * 5 + [True] * 12)
                self.assertTrue(all(tuple(r) == TRAILING_SCHEMA for r in recs))
                join_records(tr, recs, load_sidecar(sidecar_path(tr)))                   # sidecar bound to the trace by sha256
                side = load_sidecar(sidecar_path(tr))
                self.assertEqual(side["experiment"]["control_algorithm"], "neutral-prose-replicates/1")
                self.assertEqual(side["experiment"]["control_manifest_sha256"], file_sha256(CONTROLS))
            self.assertEqual(sorted(res["arms"]), sorted(arms.ARM_ORDER))
            self.assertIn("robustness_analysis_original_B_vs_controls", res)
            self.assertIn("cue_table", res)
            self.assertIn("interaction_contrast", res["cue_table"])
            self.assertEqual(res["controls_tokenizer_fingerprint_matches_server"], False)   # the mock tokenizer is not Qwen's: flagged, not hidden
            self.assertEqual(res["baseline"]["argmax"], "A")
            self.assertEqual(read_traces(os.path.join(out, "traces-pass1.jsonl")), read_traces(os.path.join(out, "traces-pass2.jsonl")))
            on_disk = _json(os.path.join(out, "analysis.json"))
            self.assertEqual(on_disk["driver"], "research/arms.py")
            self.assertEqual(on_disk["bleediff_version"], __version__)

    def test_it_refuses_to_write_into_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):
                self._run(d)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_arms.main(["--server", "http://127.0.0.1:9", "--out", d]), 2)

    def test_a_reused_baseline_is_accepted_only_when_a_fresh_fold_reproduces_it(self):
        with tempfile.TemporaryDirectory() as d:
            first = os.path.join(d, "a")
            self._run(first)
            res = self._run(os.path.join(d, "b"), baseline="reuse:" + os.path.join(first, "baseline.json"))
            self.assertEqual(res["baseline"]["mode"], "reuse")
            doc = _json(os.path.join(first, "baseline.json"))
            doc["scenarios"][arms.SCENARIO_ID]["B_ref"][0] += 0.01
            bad = os.path.join(d, "bad.json")
            with open(bad, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            with self.assertRaises(RuntimeError):
                self._run(os.path.join(d, "c"), baseline="reuse:" + bad)

    def test_a_probe_only_run_works_with_a_reference_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            ref = os.path.join(d, "ref.json")
            with open(ref, "w", encoding="utf-8") as fh:
                json.dump({"passes": {"pass1": {"descriptive": {"signed_delta": {"min": -0.2, "max": 0.06, "median": -0.1},
                                                                "treatment": {"delta_contrast": 0.85}}}}}, fh)
            with serve() as (url, _s), contextlib.redirect_stdout(io.StringIO()):
                res = arms.run(server=url, out_dir=os.path.join(d, "probe"), frozen_path=FROZEN, controls_path=None, fixtures_path=FIXTURES,
                               reference_analysis=ref, timeout=60)
            self.assertIn("not measured in this run", res["control_range"]["source"])
            self.assertEqual(len(read_traces(os.path.join(d, "probe", "traces-pass1.jsonl"))), 5)


TRAILING_SCHEMA = TRACE_SCHEMA


class TestMeasureFragments(unittest.TestCase):
    def test_measure_reports_counts_and_flags_a_mismatch(self):
        sc = scenario()
        tok = lambda t: [sum(map(ord, w)) % 7 for w in t.split()]             # a trivial deterministic tokenizer
        orig = " ".join(sc.adjacent.split()[:25])
        m = MF.measure(orig, sc, tok)
        self.assertEqual((m["bytes"], m["words"]), (170, 25))
        self.assertTrue(m["window_is_fragment"])
        self.assertEqual(m["skeleton_clauses"], [[14, "."], [6, "."]])
        self.assertEqual(m["header_len"], 3)
        other = arms.load_frozen_arms(FROZEN)["P_paraphrase"]
        self.assertEqual(MF.measure(other, sc, tok)["skeleton_clauses"], m["skeleton_clauses"])
        shorter = " ".join(other.split()[:24])
        self.assertFalse(MF.measure(shorter, sc, tok)["window_is_fragment"])      # 24 words: the window overruns into the trailer


# --------------------------------------------------------------------------------------
class TestReleaseHygiene(unittest.TestCase):
    def test_research_code_is_standard_library_only(self):
        allowed = set(sys.stdlib_module_names) | {"bleediff", "research", "tests", "test_bleediff"}
        for name in ("arms.py", "run_arms.py", "measure_fragments.py", "__init__.py"):
            tree = ast.parse(_read(os.path.join(ROOT, "research", name)))
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mods = [node.module.split(".")[0]]
                self.assertFalse([m for m in mods if m not in allowed], (name, mods))

    def test_the_driver_adds_no_equation_or_threshold(self):
        src = _read(os.path.join(ROOT, "research", "arms.py"))
        self.assertNotIn("math.log", src)
        self.assertNotIn("EPS", src)
        self.assertEqual(arms.EXPOSURE_THRESHOLD, 0.4)
        import bleediff
        self.assertEqual(arms.EXPOSURE_THRESHOLD, bleediff.EXPOSURE_DELTA_NATS)

    def test_tracked_text_files_carry_no_personal_or_local_data(self):
        pat = re.compile(r"(@[A-Za-z0-9-]+\.(com|org|net|io)|[A-Za-z]:\\\\|[A-Za-z]:\\|/Users/|/home/|AppData|api[_-]?key\s*[:=]|password\s*[:=])", re.I)
        files = [os.path.join(ROOT, "README.md"), os.path.join(ROOT, "CHANGELOG.md"), os.path.join(ROOT, "RELEASE_NOTES_v0.2.0.md"),
                 os.path.join(ROOT, "docs", "TECHNICAL.md")]
        files += [os.path.join(DATA, f) for f in os.listdir(DATA)]
        files += [os.path.join(ROOT, "research", f) for f in os.listdir(os.path.join(ROOT, "research")) if f.endswith((".py", ".md"))]
        for path in files:
            self.assertTrue(os.path.exists(path), path)
            self.assertIsNone(pat.search(_read(path)), path)

    def test_version_files_agree(self):
        import tomllib

        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
            self.assertEqual(tomllib.load(fh)["project"]["version"], __version__)
        self.assertEqual(__version__, "0.7.0")                                  # the package version moved on; the v0.2.0 evidence did not
        self.assertEqual(_json(RESULTS)["bleediff_release"], "0.2.0")
        self.assertIn("## [0.2.0]", _read(os.path.join(ROOT, "CHANGELOG.md")))
        self.assertIn("## [0.3.0]", _read(os.path.join(ROOT, "CHANGELOG.md")))
        self.assertIn("## [0.4.0]", _read(os.path.join(ROOT, "CHANGELOG.md")))

    def test_gitignore_keeps_models_and_private_evidence_out(self):
        gi = _read(os.path.join(ROOT, ".gitignore"))
        for needle in ("runs/", "*.gguf", "*.jsonl.meta.json", "!research/data/**"):
            self.assertIn(needle, gi)

    def test_readme_quotes_the_measured_numbers_and_the_required_disclosures(self):
        t = " ".join(_read(os.path.join(ROOT, "README.md")).split())
        res = results()
        for model in (M15, M7):
            m = res["models"][model]["measurements"]
            for key in ("treatment_original_B", "paraphrase", "polarity_reversed"):
                self.assertIn(f"{m[key]['delta_contrast']:+.3f}", t, (model, key))
            self.assertIn(f"{m['control_stats']['min']:+.3f}", t)
            self.assertIn(f"{m['control_stats']['max']:+.3f}", t)
        for needle in ("ordinary text concatenation", "algebraically recoverable", "±0.1", "need not be centered at zero", "INCONCLUSIVE",
                       "No real tenant-isolation vulnerability", "one model family", "not strictly additive", "bit-identical",
                       "Lexical overlap alone is insufficient", "not uniquely established"):
            self.assertIn(needle, t, needle)


if __name__ == "__main__":
    unittest.main()
