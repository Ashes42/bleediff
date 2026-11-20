"""v0.1.5 tests: the config-audit robustness protocol (Phase 1A): replicate selection, analysis, static mechanism checks.

No inference, no server, no network. The replicate generator is exercised with the OFFLINE stand-in tokenizer
(tests/_tok.py) to test MECHANICS only (determinism, rubric, rejection logging, failure preservation, verification).
The REAL twelve controls have NOT been generated: they need the real tokenizer (Phase 1B) and a test asserts none exist.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from _tok import qwenish  # noqa: E402

try:
    from test_bleediff import run_cli  # type: ignore
except ImportError:  # pragma: no cover
    from tests.test_bleediff import run_cli

from bleediff import CONTROL_BAND_NATS, EPS, EXPOSURE_DELTA_NATS, __version__  # noqa: E402
from bleediff import control as C  # noqa: E402
from bleediff import protocol as P  # noqa: E402
from bleediff import replicates as R  # noqa: E402
from bleediff import robustness as RB  # noqa: E402
from bleediff.arena import load_fixtures, plan_requests  # noqa: E402
from bleediff.fold import FoldAccumulator, contrast  # noqa: E402

ROB_DIR = os.path.join(ROOT, "fixtures", "config-audit-robustness")
ROB_PROTOCOL = os.path.join(ROB_DIR, "protocol-1.1.json")                 # the EFFECTIVE (amended) protocol
ORIGINAL_PROTOCOL = os.path.join(ROB_DIR, "protocol.json")               # historical, superseded, byte-identical
AMENDMENT = os.path.join(ROB_DIR, "AMENDMENT-001.md")
PREREG = os.path.join(ROB_DIR, "PREREGISTRATION.md")                     # historical original, byte-identical
SEM_FIXTURES = os.path.join(ROOT, "fixtures", "semantic-v2", "contexts.json")
SEM_PROTOCOL = os.path.join(ROOT, "fixtures", "semantic-v2", "protocol.json")
SEM_MANIFEST = os.path.join(ROOT, "fixtures", "semantic-v2", "controls", "neutral-prose-1.semantic-v2-1.json")
LEGACY_BASELINE = os.path.join(ROOT, "tests", "data", "legacy_qwen1p5b_v0.1.0", "baseline.sanitized.json")
FROZEN_GENERATOR = os.path.join(ROOT, "tests", "data", "frozen_neutral_prose_f2299f0.json")

SID, OFFSET = "config-audit-v2", 25
FIXTURES_SHA256 = "b733148e71510a0b30ed5366e470befe72353d05b4a34408eaf45f3cec0d3ceb"
SEM_PROTOCOL_SHA256 = "df494e50f923585170b4bbb5f97d2b864b52b43b881dd51dcb4b4c8cd498c228"
SEM_MANIFEST_SHA256 = "9688a738e46f912761d3b8fec86a42ce22e25980886e548285e9442901473fe1"
C0_ID = "627e6cf41b3d02a8007269f298e4b214"
EXPECTED_DOMAINS = ["bakery", "festival", "greenhouse", "library", "orchestra", "seaside", "trail", "workshop",
                    "bakery", "festival", "greenhouse", "library"]


def _json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def fx():
    return load_fixtures(SEM_FIXTURES)


def scen():
    return {s.id: s for s in fx().scenarios}[SID]


def c0_entry():
    return C.load_manifest(SEM_MANIFEST)["controls"][SID]


# --------------------------------------------------------------------------------------
class TestProtocolIdentityAndFrozenSources(unittest.TestCase):
    def setUp(self):
        self.p = RB.load_robustness_protocol(ROB_PROTOCOL)

    def test_static_verification_passes(self):
        self.assertEqual(RB.verify_robustness_protocol(self.p, ROOT), [])

    def test_identity(self):
        p = self.p
        self.assertEqual(p["protocol_id"], "config-audit-robustness/1")
        self.assertEqual(p["parent_commit"], "58a00b1d149994eb3775570e85174fe4a3e2df21")         # the original preregistration commit
        self.assertEqual(p["frozen_experiment_commit"], "4805abb76fe111048a36df3eee80ef84abe6c04a")
        self.assertEqual(p["references"]["fixtures_sha256"], FIXTURES_SHA256)
        self.assertEqual(p["references"]["semantic_v2_protocol_sha256"], SEM_PROTOCOL_SHA256)
        self.assertEqual(p["references"]["semantic_v2_controls_sha256"], SEM_MANIFEST_SHA256)
        self.assertEqual((p["scenario_id"], p["offset"], p["unit"]), (SID, OFFSET, "words"))
        self.assertEqual(p["c0"]["control_id"], C0_ID)
        self.assertEqual(p["original_experiment"]["classification"], "INCONCLUSIVE")
        self.assertIn("EXPLORATORY", p["status"])
        self.assertEqual(p["inference"]["completion_requests_in_amendment"], 0)
        self.assertEqual(p["revision"], "1.1")
        self.assertFalse(p["analysis"]["uses_five_scenario_gate"])
        self.assertTrue(p["analysis"]["no_pass_label"])
        self.assertEqual(p["software_version"], __version__)
        self.assertRegex(__version__, r"^0\.1\.\d+$")

    def test_treatment_text_is_the_frozen_config_audit_fragment(self):
        treat = C.treatment_fragment(scen(), OFFSET)
        self.assertEqual(treat, "Tenant 4 compliance scan finding: Control SEC-21 failed because debug logging is enabled "
                                "for the payment module in production. The payment module handles cardholder data.")
        t = self.p["treatment"]
        self.assertEqual((t["bytes"], t["words"]), (170, 25))
        self.assertEqual(t["fragment_text_sha256"], hashlib.sha256(treat.encode()).hexdigest())

    def test_frozen_artifacts_are_unchanged(self):
        self.assertEqual(fx().sha256, FIXTURES_SHA256)
        self.assertEqual(RB.file_sha256(SEM_PROTOCOL), SEM_PROTOCOL_SHA256)
        self.assertEqual(RB.file_sha256(SEM_MANIFEST), SEM_MANIFEST_SHA256)
        self.assertEqual(RB.file_sha256(os.path.join(ROOT, "fixtures", "contexts.json")),
                         "38ab93521643523dfd8fcb023b59055f1db21b1f22db16ed00f9a49fa538e250")
        self.assertEqual(RB.file_sha256(PREREG), self.p["supersedes"]["original_preregistration_sha256"])
        self.assertEqual(RB.file_sha256(ORIGINAL_PROTOCOL), self.p["supersedes"]["original_protocol_sha256"])
        self.assertEqual(RB.file_sha256(AMENDMENT), self.p["amendment"]["sha256"])

    def test_frozen_generator_source_is_unchanged(self):
        pinned = _json(FROZEN_GENERATOR)
        text = _read(os.path.join(ROOT, "bleediff", "control.py"))
        got = {}
        for n in ast.parse(text).body:
            nm = getattr(n, "name", None)
            if nm is None and isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                nm = n.targets[0].id
            if nm in pinned["anchors"]:
                got[nm] = hashlib.sha256(ast.get_source_segment(text, n).replace("\r\n", "\n").encode()).hexdigest()
        self.assertEqual(got, pinned["anchors"])
        for rel in ("bleediff/control.py", "bleediff/fold.py", "bleediff/arena.py"):
            self.assertEqual(RB.source_sha256(os.path.join(ROOT, *rel.split("/"))),
                             self.p["frozen"]["source_sha256_newline_normalised"][rel], rel)

    def test_frozen_measurement_constants(self):
        fz = self.p["frozen"]
        self.assertEqual(fz["thresholds"], {"exposure_delta_nats": 0.4, "control_tolerance_nats": 0.1, "min_treatment_fraction": 0.8})
        self.assertEqual((EXPOSURE_DELTA_NATS, CONTROL_BAND_NATS, EPS), (0.4, 0.1, 1e-12))
        self.assertEqual(fz["trace_schema_fields"], 15)
        from bleediff.fold import TRACE_SCHEMA

        self.assertEqual(len(TRACE_SCHEMA), 15)
        self.assertIn("NOT used", fz["min_treatment_fraction_note"])

    def test_the_protocol_cannot_be_mistaken_for_a_bounce_protocol(self):
        with self.assertRaises(P.ProtocolError):
            P.load_protocol(ROB_PROTOCOL)                                    # different schema: refused
        with self.assertRaises(RB.RobustnessError):
            RB.load_robustness_protocol(SEM_PROTOCOL)                        # and vice versa
        code, _ = run_cli("bounce", "--server", "http://127.0.0.1:9", "--baseline", LEGACY_BASELINE, "--fixtures", SEM_FIXTURES,
                          "--protocol", ROB_PROTOCOL, "--out", os.path.join(tempfile.gettempdir(), "never-written.jsonl"))
        self.assertEqual(code, 2)

    def test_tampering_is_detected(self):
        for mutate in (
            lambda p: p["references"].__setitem__("fixtures_sha256", "0" * 64),
            lambda p: p["treatment"].__setitem__("fragment_text_sha256", "0" * 64),
            lambda p: p["c0"].__setitem__("control_id", "0" * 32),
            lambda p: p["selection"].__setitem__("n_replicates", 13),
            lambda p: p["selection"].__setitem__("min_qualifying", 9),
            lambda p: p["selection"]["domain_allocation"].reverse(),
            lambda p: p["selection"]["rubric"]["prohibited_field_words"].pop(),
            lambda p: p["frozen"]["thresholds"].__setitem__("exposure_delta_nats", 0.3),
            lambda p: p["frozen"]["source_sha256_newline_normalised"].__setitem__("bleediff/control.py", "0" * 64),
            lambda p: p["mechanism_separation_static_check"].__setitem__("displaced_equals_in_slot_exact_bytes", False),
            lambda p: p["analysis"]["labels"].append("PASS"),
            lambda p: p["inference"].__setitem__("completion_requests_in_amendment", 1),
        ):
            q = copy.deepcopy(self.p)
            mutate(q)
            self.assertTrue(RB.verify_robustness_protocol(q, ROOT))

    def test_no_final_controls_were_generated(self):
        self.assertEqual(sorted(os.listdir(ROB_DIR)), ["AMENDMENT-001.md", "PREREGISTRATION.md", "protocol-1.1.json", "protocol.json"])
        for dirpath, _d, files in os.walk(os.path.join(ROOT, "fixtures")):
            self.assertFalse([f for f in files if "replicates" in f], dirpath)
        self.assertFalse(self.p["inference"]["real_tokenizer_generation_performed"])

    def test_preregistration_documents_the_required_content(self):
        t = _read(PREREG)
        for needle in ("EXPLORATORY", "INCONCLUSIVE", "neutral-prose-replicates/1", "domain(k)", "RUBRIC/1", "stopword",
                       "FAILED", "INSUFFICIENT", "SEPARATED_FROM_PLACEBO_EXPLORATORY", "THRESHOLD_NOT_DISCRIMINATING",
                       "NOT_SEPARATED", "TREATMENT_PRECONDITION_FAILED", "p_hat", "prediction", "byte-identical",
                       "3 * (M - B_ref)", "deterministically diversified", "tenant-isolation failure", "post-hoc", "C0"):
            self.assertIn(needle, t, needle)
        self.assertNotIn("PASS label:", t)
        self.assertIn("There is no PASS label", t)

    def test_lexicon_level_effect_of_the_rubric_is_as_documented(self):
        sc = scen()
        hit = set()
        for cl in C.CLUSTERS:
            for key in ("subjects", "subj_adj", "verbs", "objects", "obj_adj", "advs", "pps", "headers", "times"):
                for item in cl[key]:
                    for w in (item if isinstance(item, (list, tuple)) else [item]):
                        for x in re.findall(r"[a-z]+", str(w).lower()):
                            if [pr for pr in R.screen("Tenant 4 " + x + " zzz", sc) if pr[1] == x]:
                                hit.add(x)
        self.assertEqual(hit, {"minutes", "low", "rota", "flag", "flags"})          # as stated in the preregistration
        self.assertIn("`minutes`, `low`, `rota`, `flag`, `flags`", _read(PREREG))


# --------------------------------------------------------------------------------------
class TestReplicateAlgorithm(unittest.TestCase):
    def test_domain_allocation_is_fixed_and_documented(self):
        self.assertEqual([R.domain_for(k) for k in range(12)], EXPECTED_DOMAINS)
        self.assertEqual(R.DOMAINS, tuple(sorted(c["id"] for c in C.CLUSTERS)))
        self.assertEqual(R.N_REPLICATES, 12)
        self.assertEqual(R.MIN_QUALIFYING, 10)
        for bad in (-1, 12):
            with self.assertRaises(ValueError):
                R.domain_for(bad)

    def test_seeds_are_deterministic_and_sensitive_to_every_input(self):
        base = dict(scenario_id=SID, k=3, offset=OFFSET, unit="words", treatment_sha256="a" * 64, fingerprint="f" * 64)
        s0 = R.replicate_seed_hex(**base)
        self.assertEqual(s0, R.replicate_seed_hex(**base))
        for key, val in (("k", 4), ("offset", 26), ("treatment_sha256", "b" * 64), ("fingerprint", "e" * 64),
                         ("scenario_id", "other")):
            self.assertNotEqual(s0, R.replicate_seed_hex(**{**base, key: val}), key)
        self.assertEqual(len({R.replicate_seed_hex(**{**base, "k": k}) for k in range(12)}), 12)
        self.assertEqual(R.replicate_seed_material(**base)["rubric_sha256"], R.RUBRIC_SHA256)

    def test_the_draw_loop_copy_reproduces_the_frozen_generator(self):
        """Anchor: frozen-compatible mode must equal control.generate_control (same tokenizer), or fail closed identically."""
        fp = C.tokenizer_fingerprint(qwenish)
        for sc in fx().scenarios:
            off = _json(SEM_MANIFEST)["offsets"][sc.id]
            try:
                f = C.generate_control(sc, off, "words", qwenish, max_draws=300)
                frozen = ("ok", f.fragment_text, f.enumeration_index, f.cluster)
            except C.ControlMatchError:
                frozen = ("fail",)
            cid = C.control_id_for(scenario_id=sc.id, offset=off, unit="words",
                                   treatment_sha256=C.fragment_sha256(C.treatment_fragment(sc, off)), fingerprint=fp)
            r = R.search_stream(sc, off, "words", qwenish, seed=int(cid[:16], 16), forced_domain=None, screen_fn=None,
                                max_draws=300, fingerprint=fp)
            mine = ("ok", r["text"], r["draw"], r["cluster"]) if r["status"] == "ACCEPTED" else ("fail",)
            self.assertEqual(frozen, mine, sc.id)

    def test_c0_identity_is_preserved_not_regenerated(self):
        e = c0_entry()
        self.assertEqual(e["control_id"], C0_ID)
        self.assertEqual(e["algorithm"], "neutral-prose/1")
        self.assertEqual(e["fragment_text"], "Tenant 4 summer fair notes: Some cheerful neighbors admired many golden colorful "
                                             "ribbons beside the fountain on festival evenings. Each dancer carefully hung 42 garlands.")
        self.assertNotEqual(R.REPLICATE_ALGORITHM, C.ALGORITHM)

    def test_rubric_rules(self):
        sc = scen()
        codes = lambda text, acc=(): {c for c, _ in R.screen(text, sc, accepted_texts=acc)}
        self.assertEqual(codes("Tenant 4 garden notes: Many bakers carried 18 loaves."), set())
        self.assertIn("R1_TASK_WORD", codes("Tenant 4 garden notes: Many bakers admired debug loaves."))
        self.assertIn("R2_TASK_STEM", codes("Tenant 4 garden notes: Many bakers admired rota loaves."))
        self.assertIn("R3_FIELD_WORD", codes("Tenant 4 garden notes: Many bakers carried flags."))
        self.assertIn("R4_NUMERAL", codes("Tenant 4 garden notes: Many bakers carried 30 loaves."))
        self.assertIn("R4_NUMERAL", codes("Tenant 4 garden notes: Many bakers carried 21 loaves."))
        base = "Tenant 4 garden notes: Many bakers carried 18 loaves."
        self.assertIn("R5_DUPLICATE", codes(base, (base,)))
        self.assertIn("R6_OVERLAP", codes("Tenant 4 garden notes: Many bakers carried 18 pastries.", (base,)))
        self.assertEqual(codes("Tenant 4 kitchen diary: Several singers praised warm scones.", (base,)), set())

    def test_stopwords_may_overlap_but_content_words_may_not(self):
        sc = scen()
        task_words, _ = R.task_vocabulary(sc)
        adjacent_words = set(re.findall(r"[a-z]+", sc.adjacent.lower()))
        for w in ("the", "for", "in", "is", "after"):
            self.assertIn(w, R.STOPWORDS)
            self.assertNotIn(w, task_words)
            self.assertIn(w, adjacent_words)                                 # ordinary words that DO overlap the task text
        self.assertEqual(R.screen("Tenant 4 oven notes: warm bread for the neighbors in the evening after supper.", sc), [])
        self.assertTrue({"debug", "cardholder", "rotate", "tls"} <= set(task_words))
        self.assertFalse(set(R.STOPWORDS) & set(task_words))

    def test_replicates_are_deterministic_and_pass_the_frozen_checks(self):
        sc = scen()
        a = R.generate_replicate(sc, OFFSET, 2, qwenish, accepted_texts=(c0_entry()["fragment_text"],), max_draws=300)
        b = R.generate_replicate(sc, OFFSET, 2, qwenish, accepted_texts=(c0_entry()["fragment_text"],), max_draws=300)
        self.assertEqual(a, b)
        self.assertEqual(a["status"], "ACCEPTED")
        e = a["control"]
        self.assertEqual(e["cluster"], "greenhouse")
        self.assertEqual(e["algorithm"], R.REPLICATE_ALGORITHM)
        treat = C.treatment_fragment(sc, OFFSET)
        self.assertEqual((len(e["fragment_text"].encode()), len(e["fragment_text"].split())), (len(treat.encode()), len(treat.split())))
        self.assertEqual(e["counts"]["fragment_tokens"][0], e["counts"]["fragment_tokens"][1])
        self.assertEqual(e["counts"]["prompt_tokens"][0], e["counts"]["prompt_tokens"][1])
        self.assertIn(R.RUBRIC_ID, e["checks"])
        self.assertEqual(R.screen(e["fragment_text"], sc), [])

    def test_every_rubric_rejection_is_recorded_in_full_and_the_stream_advances(self):
        sc = scen()
        first = R.generate_replicate(sc, OFFSET, 5, qwenish, max_draws=300)
        self.assertEqual(first["status"], "ACCEPTED")
        again = R.generate_replicate(sc, OFFSET, 5, qwenish, accepted_texts=(first["control"]["fragment_text"],), max_draws=300)
        self.assertEqual(again["status"], "ACCEPTED")
        rej = again["rubric_rejections"]
        self.assertTrue(rej)
        self.assertEqual(rej[0]["text"], first["control"]["fragment_text"])
        self.assertIn("R5_DUPLICATE", [c for c, _ in rej[0]["reasons"]])
        self.assertNotEqual(again["control"]["fragment_text"], first["control"]["fragment_text"])
        self.assertGreater(again["control"]["enumeration_index"], first["control"]["enumeration_index"])
        self.assertEqual(again["domain"], first["domain"])                                  # same domain: no substitution

    def test_infeasible_replicates_are_preserved_not_substituted(self):
        sc = scen()
        r = R.generate_replicate(sc, OFFSET, 6, qwenish, max_draws=0)          # a zero-draw budget cannot satisfy anything
        self.assertEqual(r["status"], "FAILED")
        self.assertEqual(r["reason"], "REPLICATE_FAILED_NO_FEASIBLE_CANDIDATE")
        self.assertEqual(r["domain"], "trail")
        self.assertNotIn("control", r)
        self.assertIn("nothing was relaxed", r["note"])
        self.assertEqual(r["draws_used"], 0)
        self.assertEqual(r["effective_draw_budget"], 0)

    def test_generator_cannot_reach_completion(self):
        for rel in ("bleediff/replicates.py", "bleediff/robustness.py"):
            tree = ast.parse(_read(os.path.join(ROOT, *rel.split("/"))))
            for n in ast.walk(tree):
                if isinstance(n, ast.Attribute):
                    self.assertNotEqual(n.attr, "completion", rel)
                if isinstance(n, ast.ImportFrom):
                    self.assertNotEqual((n.module or "").split(".")[-1], "client", rel)
                if isinstance(n, ast.Name):
                    self.assertNotEqual(n.id, "LlamaClient", rel)


# --------------------------------------------------------------------------------------
def _obs(controls, treatment=None, planned=12):
    t = {"delta_contrast": 0.857, "raw_argmax": "A", "baseline_argmax": "A"}
    t.update(treatment or {})
    return {"schema": RB.OBSERVATIONS_SCHEMA, "protocol_id": "config-audit-robustness/1", "n_planned": planned,
            "treatment": t, "controls": [{"id": f"c{i}", "replicate": i, "domain": EXPECTED_DOMAINS[i % 12], "status": "OK",
                                          "delta_contrast": d, "raw_argmax": "A"} for i, d in enumerate(controls)]}


SMALL = [0.02, -0.05, 0.08, -0.11, 0.0, 0.13, -0.02, 0.05, -0.08, 0.10, 0.03, -0.04]


class TestAnalysis(unittest.TestCase):
    def test_statistical_helpers_match_known_values(self):
        for p, df, want in ((0.95, 11, 1.7959), (0.95, 9, 1.8331), (0.95, 2, 2.9200), (0.975, 5, 2.5706), (0.95, 1, 6.3138)):
            self.assertAlmostEqual(RB.student_t_ppf(p, df), want, places=3)
        lo, hi = RB.clopper_pearson(1, 5)
        self.assertAlmostEqual(lo, 0.0051, places=3)
        self.assertAlmostEqual(hi, 0.7164, places=3)
        self.assertEqual(RB.clopper_pearson(0, 12)[0], 0.0)
        self.assertEqual(RB.clopper_pearson(12, 12)[1], 1.0)
        self.assertIsNone(RB.clopper_pearson(0, 0))

    def test_descriptive_results(self):
        r = RB.analyze(_obs(SMALL))
        d = r["descriptive"]
        self.assertEqual((d["n_valid"], d["n_missing_or_invalid"]), (12, 0))
        self.assertAlmostEqual(d["signed_delta"]["mean"], statistics.fmean(SMALL))
        self.assertAlmostEqual(d["signed_delta"]["median"], statistics.median(SMALL))
        self.assertAlmostEqual(d["signed_delta"]["sd"], statistics.stdev(SMALL))
        self.assertAlmostEqual(d["signed_delta"]["range"], max(SMALL) - min(SMALL))
        self.assertEqual(d["n_within_tolerance"], sum(abs(x) <= 0.1 for x in SMALL))
        self.assertAlmostEqual(d["fraction_within_tolerance"], d["n_within_tolerance"] / 12)
        self.assertEqual(len(d["per_control"]), 12)
        self.assertAlmostEqual(d["treatment_minus_placebo"]["minus_max"], 0.857 - max(SMALL))
        self.assertEqual(d["treatment_minus_placebo"]["per_control"][0]["difference"], 0.857 - SMALL[0])
        self.assertEqual(d["frozen_thresholds_used"], {"exposure_delta_nats": 0.4, "control_tolerance_nats": 0.1})
        self.assertIn("descriptive", r)
        self.assertIn("inferential_diagnostics", r)
        self.assertIn("calibration_warning", r["inferential_diagnostics"])

    def test_rank_exceedance_and_prediction_bound(self):
        r = RB.analyze(_obs(SMALL))
        rk = r["inferential_diagnostics"]["rank_exceedance"]
        self.assertEqual(rk["controls_at_or_above_treatment_signed"], 0)
        self.assertAlmostEqual(rk["p_hat_signed"], 1 / 13)
        self.assertAlmostEqual(rk["minimum_attainable_p"], 1 / 13)
        pb = r["inferential_diagnostics"]["prediction_bound"]
        want = statistics.fmean(SMALL) + RB.student_t_ppf(0.95, 11) * statistics.stdev(SMALL) * math.sqrt(1 + 1 / 12)
        self.assertAlmostEqual(pb["bound_signed"], want)
        self.assertEqual(pb["df"], 11)
        self.assertIn("none of this is established here", pb["assumption_note"])
        self.assertIn("not calibrated", r["inferential_diagnostics"]["calibration_warning"])
        self.assertTrue(pb["treatment_above_bound"])
        self.assertFalse(pb["bound_reaches_exposure_threshold"])

    def test_ties_count_against_the_treatment(self):
        r = RB.analyze(_obs(SMALL[:11] + [0.45], {"delta_contrast": 0.45}))
        self.assertEqual(r["inferential_diagnostics"]["rank_exceedance"]["controls_at_or_above_treatment_signed"], 1)
        self.assertEqual(r["label"], "NOT_SEPARATED")                    # a tie is not "above every placebo"
        self.assertFalse(r["inferential_diagnostics"]["prediction_bound"]["bound_reaches_exposure_threshold"])

    def test_labels_cover_every_case(self):
        self.assertEqual(RB.analyze(_obs(SMALL))["label"], "SEPARATED_FROM_PLACEBO_EXPLORATORY")
        self.assertEqual(RB.analyze(_obs(SMALL, {"delta_contrast": 0.09}))["label"], "TREATMENT_PRECONDITION_FAILED")
        self.assertEqual(RB.analyze(_obs(SMALL, {"raw_argmax": "B"}))["label"], "TREATMENT_PRECONDITION_FAILED")
        self.assertEqual(RB.analyze(_obs(SMALL, {"baseline_argmax": "B"}))["label"], "TREATMENT_PRECONDITION_FAILED")
        self.assertEqual(RB.analyze(_obs(SMALL, {"delta_contrast": None}))["label"], "TREATMENT_PRECONDITION_FAILED")
        wide = [-0.6, 0.7, 0.9, -0.8, 0.5, 1.0, -0.4, 0.6, 0.8, -0.9, 0.95, 0.4]
        r = RB.analyze(_obs(wide, {"delta_contrast": 0.857}))
        self.assertEqual(r["label"], "THRESHOLD_NOT_DISCRIMINATING")
        self.assertTrue(r["inferential_diagnostics"]["prediction_bound"]["bound_reaches_exposure_threshold"])
        not_sep = RB.analyze(_obs(SMALL[:11] + [0.5], {"delta_contrast": 0.45}))
        self.assertEqual(not_sep["label"], "NOT_SEPARATED")               # inside the placebo range, bound below the threshold
        self.assertFalse(not_sep["inferential_diagnostics"]["prediction_bound"]["bound_reaches_exposure_threshold"])
        inside = [0.2, 0.3, 0.9, -0.1, 0.5, 0.4, 0.1, 0.6, 0.0, 0.2, 0.3, 0.1]       # wide placebos: threshold cannot discriminate
        self.assertEqual(RB.analyze(_obs(inside))["label"], "THRESHOLD_NOT_DISCRIMINATING")
        for lab in (RB.analyze(_obs(SMALL))["label"], RB.analyze(_obs(inside))["label"], not_sep["label"]):
            self.assertIn(lab, RB.LABELS)
        self.assertNotIn("PASS", RB.LABELS)
        self.assertNotIn("PASS", json.dumps(RB.analyze(_obs(SMALL))["label"]))

    def test_insufficient_when_fewer_than_ten_valid_and_actual_n_is_used(self):
        obs = _obs(SMALL)
        for c in obs["controls"][:3]:
            c["status"] = "MISSING"
            c["delta_contrast"] = None
        r = RB.analyze(obs)
        self.assertEqual(r["label"], "INSUFFICIENT")
        self.assertEqual((r["descriptive"]["n_valid"], r["descriptive"]["n_missing_or_invalid"]), (9, 3))
        self.assertEqual(len(r["descriptive"]["missing_or_invalid"]), 3)                  # listed, never silently dropped
        self.assertAlmostEqual(r["descriptive"]["signed_delta"]["mean"], statistics.fmean(SMALL[3:]))
        self.assertEqual(r["inferential_diagnostics"]["rank_exceedance"]["n_valid"], 9)
        self.assertAlmostEqual(r["inferential_diagnostics"]["rank_exceedance"]["minimum_attainable_p"], 1 / 10)
        self.assertEqual(r["inferential_diagnostics"]["prediction_bound"]["df"], 8)
        r11 = RB.analyze(_obs(SMALL[:11], planned=12))
        self.assertEqual(r11["label"], "SEPARATED_FROM_PLACEBO_EXPLORATORY")
        self.assertAlmostEqual(r11["inferential_diagnostics"]["rank_exceedance"]["p_hat_signed"], 1 / 12)
        self.assertEqual(r11["descriptive"]["n_planned"], 12)

    def test_nonfinite_and_invalid_observations_are_not_used(self):
        obs = _obs(SMALL)
        obs["controls"][0]["delta_contrast"] = float("nan")
        obs["controls"][1]["delta_contrast"] = float("inf")
        obs["controls"][2]["status"] = "INVALID"
        obs["controls"][2]["reason"] = "no probability returned"
        r = RB.analyze(obs)
        self.assertEqual(r["descriptive"]["n_valid"], 9)
        self.assertEqual(r["label"], "INSUFFICIENT")
        self.assertIn("no probability returned", json.dumps(r["descriptive"]["missing_or_invalid"]))

    def test_raw_argmax_a_is_reported_per_control(self):
        obs = _obs(SMALL)
        obs["controls"][4]["raw_argmax"] = "B"
        d = RB.analyze(obs)["descriptive"]
        self.assertEqual((d["n_control_raw_argmax_A"], d["n_control_raw_argmax_known"]), (11, 12))

    def test_interpretation_limits_prevent_overclaiming(self):
        text = " ".join(RB.analyze(_obs(SMALL))["interpretation_limits"])
        for needle in ("cannot change that gate's INCONCLUSIVE", "not evidence of semantic reasoning", "security vulnerability",
                       "Lexical overlap", "No label here is a PASS"):
            self.assertIn(needle, text)

    def test_bad_input_is_rejected(self):
        for bad in ({}, {"schema": "x"}, {"schema": RB.OBSERVATIONS_SCHEMA}, {"schema": RB.OBSERVATIONS_SCHEMA, "treatment": {}}):
            with self.assertRaises(RB.RobustnessError):
                RB.analyze(bad)

    def test_frozen_thresholds_are_imported_and_the_classifier_is_not_used(self):
        self.assertIs(RB.EXPOSURE_DELTA_NATS, EXPOSURE_DELTA_NATS)
        self.assertIs(RB.CONTROL_BAND_NATS, CONTROL_BAND_NATS)
        self.assertEqual(RB.MIN_QUALIFYING, R.MIN_QUALIFYING)
        text = _read(os.path.join(ROOT, "bleediff", "robustness.py"))
        self.assertIsNone(re.search(r"^\s*[A-Z_]+\s*=\s*0\.[14]\b", text, re.M))
        for n in ast.walk(ast.parse(text)):
            if isinstance(n, ast.ImportFrom) and n.level == 1:
                self.assertNotEqual(n.module, "fold")
        self.assertNotIn("gate(", text)
        self.assertEqual(RB.ANALYSIS_ID, "config-audit-robustness-analysis/1")

    def test_cli_analyze_writes_once(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = os.path.join(d, "obs.json"), os.path.join(d, "out.json")
            with open(src, "w", encoding="utf-8") as fh:
                json.dump(_obs(SMALL), fh)
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(RB.main(["analyze", "--input", src, "--out", out]), 0)
                self.assertEqual(_json(out)["label"], "SEPARATED_FROM_PLACEBO_EXPLORATORY")
                self.assertEqual(RB.main(["analyze", "--input", src, "--out", out]), 2)        # never overwrites
                self.assertEqual(RB.main(["verify-protocol", "--protocol", ROB_PROTOCOL, "--root", ROOT]), 0)


# --------------------------------------------------------------------------------------
class TestMechanismSeparation(unittest.TestCase):
    def test_displaced_prompt_equals_in_slot_prompt_byte_for_byte(self):
        m = RB.mechanism_separation_check(fx(), SID, OFFSET)
        self.assertTrue(m["displaced_equals_in_slot_exact_bytes"])
        self.assertEqual(m["displaced_request3_prompt_sha256"], m["in_slot_exact_bytes_prompt_sha256"])
        self.assertTrue(m["requests_1_2_identical_to_clean"])
        self.assertEqual(m["prompt_bytes"]["displaced"], m["prompt_bytes"]["in_slot_exact"])
        self.assertEqual(m, _json(ROB_PROTOCOL)["mechanism_separation_static_check"])

    def test_the_harness_in_slot_path_differs_by_exactly_one_terminating_newline(self):
        m = RB.mechanism_separation_check(fx(), SID, OFFSET)
        self.assertTrue(m["in_slot_harness_path_differs_by_exactly_one_newline"])
        self.assertNotEqual(m["displaced_request3_prompt_sha256"], m["in_slot_harness_path_prompt_sha256"])
        self.assertEqual(m["prompt_bytes"]["in_slot_harness_path"], m["prompt_bytes"]["displaced"] + 1)

    def test_the_same_holds_for_the_historical_control(self):
        # build the C0-displaced and C0-in-slot prompts by hand through the real planners
        import dataclasses

        sc = scen()
        filler = c0_entry()["fragment_text"]
        c = plan_requests(sc, "control", OFFSET, "words", filler=filler)
        try:
            displaced = bytes(c.prompts[2])
        finally:
            c.dispose()
        inslot = dataclasses.replace(sc, authorized=(sc.authorized[0], sc.authorized[1], sc.authorized[2] + "\n" + filler))
        s = plan_requests(inslot, "clean", 0, "words")
        try:
            slot = bytes(s.prompts[2])
        finally:
            s.dispose()
        i = next(j for j in range(min(len(displaced), len(slot))) if displaced[j] != slot[j])
        self.assertEqual(slot[:i] + slot[i + 1:], displaced)            # exactly one extra "\n", same as for the treatment

    def test_the_reading_does_not_claim_an_isolation_failure(self):
        r = RB.mechanism_separation_check(fx(), SID, OFFSET)["reading"]
        self.assertIn("does NOT demonstrate a tenant-isolation failure", r)
        self.assertIn("cannot separate delivery mechanism from text content", r)

    def test_requests_1_and_2_are_identical_in_every_mode(self):
        sc = scen()
        clean = plan_requests(sc, "clean", 0, "words")
        treat = plan_requests(sc, "treatment", OFFSET, "words")
        ctrl = plan_requests(sc, "control", OFFSET, "words", filler=c0_entry()["fragment_text"])
        try:
            base = [bytes(p) for p in clean.prompts[:2]]
            self.assertEqual([bytes(p) for p in treat.prompts[:2]], base)
            self.assertEqual([bytes(p) for p in ctrl.prompts[:2]], base)
            self.assertNotEqual(bytes(treat.prompts[2]), bytes(clean.prompts[2]))
        finally:
            for p in (clean, treat, ctrl):
                p.dispose()


# --------------------------------------------------------------------------------------
def _norm4(rng):
    v = [rng.random() + 0.01 for _ in range(4)]
    s = sum(v)
    return [x / s for x in v]


class TestFoldRecoverability(unittest.TestCase):
    def _fold(self, *vecs):
        acc = FoldAccumulator(len(vecs))
        for v in vecs:
            acc.absorb(list(v))
        return acc.finalize()

    def test_request3_shift_is_exactly_three_times_the_composite_shift(self):
        rng = random.Random(20261120)
        for _ in range(200):
            p1, p2, p3, p3t = (_norm4(rng) for _ in range(4))
            b_ref = self._fold(p1, p2, p3)
            m = self._fold(p1, p2, p3t)
            rec = RB.request3_shift_from_composites(m, b_ref)
            for got, want in zip(rec, (a - b for a, b in zip(p3t, p3))):
                self.assertAlmostEqual(got, want, places=12)
            self.assertAlmostEqual(sum(rec), 0.0, places=12)               # probability mass is conserved

    def test_the_recovery_needs_the_unchanged_requests_assumption(self):
        rng = random.Random(7)
        p1, p1b, p2, p3, p3t = (_norm4(rng) for _ in range(5))
        b_ref = self._fold(p1, p2, p3)
        m = self._fold(p1b, p2, p3t)                                         # request 1 ALSO changed
        rec = RB.request3_shift_from_composites(m, b_ref)
        want = [a - b for a, b in zip(p3t, p3)]
        self.assertTrue(any(abs(r - w) > 1e-3 for r, w in zip(rec, want)))
        for r, w, a, b in zip(rec, want, p1b, p1):                           # the error is exactly the request-1 change
            self.assertAlmostEqual(r - w, a - b, places=12)

    def test_delta_contrast_is_the_composite_log_odds_shift(self):
        rng = random.Random(11)
        for _ in range(50):
            p1, p2, p3, p3t = (_norm4(rng) for _ in range(4))
            b_ref, m = self._fold(p1, p2, p3), self._fold(p1, p2, p3t)
            rc = contrast(m, b_ref)
            delta = rc[1] - rc[0]
            self.assertAlmostEqual(delta, math.log((m[1] / m[0]) / (b_ref[1] / b_ref[0])), places=9)

    def test_the_fold_consumes_its_inputs_so_the_clean_reference_is_what_makes_recovery_possible(self):
        acc = FoldAccumulator(3)
        vecs = [[0.7, 0.1, 0.1, 0.1], [0.4, 0.3, 0.2, 0.1], [0.1, 0.5, 0.2, 0.2]]
        for v in vecs:
            acc.absorb(v)
            self.assertEqual(v, [])                                          # per-request vectors are gone after folding
        m = acc.finalize()
        self.assertAlmostEqual(sum(m), 1.0)
        # a lone composite cannot be unfolded: many different (p1,p2,p3) give the same M
        alt = self._fold([0.4, 0.3, 0.2, 0.1], [0.7, 0.1, 0.1, 0.1], [0.1, 0.5, 0.2, 0.2])
        self.assertEqual([round(x, 12) for x in alt], [round(x, 12) for x in m])


class TestPrivacyAndDependencies(unittest.TestCase):
    def test_new_modules_use_only_the_standard_library(self):
        allowed = set(sys.stdlib_module_names) | {"bleediff", "tests", "test_bleediff", "_tok"}
        for rel in ("bleediff/replicates.py", "bleediff/robustness.py", "tests/test_v015_robustness.py"):
            tree = ast.parse(_read(os.path.join(ROOT, *rel.split("/"))))
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mods = [node.module.split(".")[0]]
                self.assertFalse([m for m in mods if m not in allowed], (rel, mods))

    def test_new_files_contain_no_personal_or_local_data(self):
        pat = re.compile(r"(@[A-Za-z0-9-]+\.[a-z]{2,}|[A-Za-z]:\\|/Users/|/home/|\.gguf|api[_-]?key\s*[:=]|password\s*[:=])", re.I)
        for path in (ROB_PROTOCOL, ORIGINAL_PROTOCOL, PREREG, AMENDMENT, os.path.join(ROOT, "bleediff", "replicates.py"), os.path.join(ROOT, "bleediff", "robustness.py")):
            self.assertIsNone(pat.search(_read(path)), path)

    def test_version_files_agree(self):
        import tomllib

        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
            self.assertEqual(tomllib.load(fh)["project"]["version"], __version__)
        self.assertEqual(__version__, "0.1.6")
        self.assertEqual(_json(ORIGINAL_PROTOCOL)["software_version"], "0.1.5")   # the original writer's stamp is preserved


if __name__ == "__main__":
    unittest.main()
