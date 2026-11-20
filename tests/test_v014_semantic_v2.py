"""v0.1.4 tests: the frozen semantic-v2/1 experimental protocol.

No inference and no network beyond a stdlib mock server on 127.0.0.1. The REAL controls in
fixtures/semantic-v2/controls/ were generated and verified tokenize-only against the real Qwen2.5 tokenizer; this
file checks them statically (everything that does not need a tokenizer) and exercises the generator against the
offline stand-in tokenizer to prove determinism and fail-closed behaviour. Nothing here shows that the fixtures are
semantically valid: that needs human review and, later, a measured baseline.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from _tok import qwenish  # noqa: E402

try:
    from test_bleediff import MockState, run_cli, serve  # type: ignore
except ImportError:  # pragma: no cover
    from tests.test_bleediff import MockState, run_cli, serve

from bleediff import UNKNOWN, __version__  # noqa: E402
from bleediff import control as C  # noqa: E402
from bleediff import protocol as P  # noqa: E402
from bleediff.arena import load_fixtures  # noqa: E402
from bleediff.control import ControlMatchError  # noqa: E402
from bleediff.fold import TRACE_SCHEMA  # noqa: E402

SEM = os.path.join(ROOT, "fixtures", "semantic-v2")
SEM_FIXTURES = os.path.join(SEM, "contexts.json")
SEM_PROTOCOL = os.path.join(SEM, "protocol.json")
SEM_MANIFEST = os.path.join(SEM, "controls", "neutral-prose-1.semantic-v2-1.json")
APPROVED = os.path.join(ROOT, "tests", "data", "semantic_v2_approved_design.json")
LEGACY_BASELINE = os.path.join(ROOT, "tests", "data", "legacy_qwen1p5b_v0.1.0", "baseline.sanitized.json")
FROZEN_GENERATOR = os.path.join(ROOT, "tests", "data", "frozen_neutral_prose_f2299f0.json")
FROZEN_FOLD = os.path.join(ROOT, "tests", "data", "frozen_anchors_6a03565.json")

SEM_FIXTURES_SHA256 = "b733148e71510a0b30ed5366e470befe72353d05b4a34408eaf45f3cec0d3ceb"
APPROVED_SHA256 = "57514b9a7567d792e940d01821a84ff204ab737d7f37c9446881ad0e68ec3f30"
#: historical artefacts that must stay byte-identical (window-repair/1 and the v0.1.2 controls)
HISTORICAL = {
    "fixtures/contexts.json": "38ab93521643523dfd8fcb023b59055f1db21b1f22db16ed00f9a49fa538e250",
    "fixtures/controls/neutral-prose-1.json": "7266bf7e687f1829b0fd18a26c484d7d8fe0c359b1e223fe909dcba8a50b5145",
    "fixtures/controls/neutral-prose-1.window-repair-1.json": "b4b729f5",
}
OFFSETS = {"incident-root-cause-v2": 31, "datastore-v2": 37, "vendor-routing-v2": 33, "config-audit-v2": 25,
           "triage-severity-v2": 39}
BYTES = {"incident-root-cause-v2": 184, "datastore-v2": 257, "vendor-routing-v2": 202, "config-audit-v2": 170,
         "triage-severity-v2": 250}


def _json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def fixtures():
    return load_fixtures(SEM_FIXTURES)


def approved():
    return {r["id"]: r for r in _json(APPROVED)["scenarios"]}


# --------------------------------------------------------------------------------------
class TestFixtureFidelity(unittest.TestCase):
    def test_hashes_are_pinned(self):
        self.assertEqual(_sha(SEM_FIXTURES), SEM_FIXTURES_SHA256)
        self.assertEqual(fixtures().sha256, SEM_FIXTURES_SHA256)
        self.assertEqual(_sha(APPROVED), APPROVED_SHA256)

    def test_fixtures_reproduce_the_approved_text_exactly(self):
        ap = approved()
        fx = fixtures()
        self.assertEqual([s.id for s in fx.scenarios], list(OFFSETS))
        self.assertEqual(sorted(ap), sorted(OFFSETS))
        for sc in fx.scenarios:
            a = ap[sc.id]
            self.assertEqual(sc.question, a["question"], sc.id)
            self.assertEqual(sc.options, a["options"], sc.id)
            self.assertEqual(list(sc.authorized), a["authorized"], sc.id)
            self.assertEqual(len(sc.authorized), 3)
            self.assertEqual(sc.adjacent, a["displaced_window"] + " " + a["unread_trailing_sentence"], sc.id)

    def test_deliberate_b_c_swap_in_the_datastore_scenario(self):
        sc = {s.id: s for s in fixtures().scenarios}["datastore-v2"]
        self.assertEqual(sc.options["B"], "Cassandra multi-datacenter cluster")
        self.assertEqual(sc.options["C"], "DynamoDB with on-demand capacity")
        self.assertIn("Cassandra", sc.adjacent)
        self.assertIn("PostgreSQL", sc.options["A"])

    def test_fixtures_require_the_protocol_and_cannot_be_used_bare(self):
        self.assertEqual(fixtures().requires_protocol, "semantic-v2/1")
        code, text = run_cli("bleed", "--fixtures", SEM_FIXTURES)
        self.assertEqual(code, 2, text)

    def test_historical_fixtures_do_not_require_a_protocol(self):
        self.assertIsNone(load_fixtures(os.path.join(ROOT, "fixtures", "contexts.json")).requires_protocol)


# --------------------------------------------------------------------------------------
class TestRuleSWindows(unittest.TestCase):
    def test_per_scenario_offsets_and_bytes_match_the_design(self):
        ap = approved()
        for sc in fixtures().scenarios:
            sel = P.select_window_rule_s(sc, ap[sc.id]["support_set"])
            self.assertEqual(sel["offset"], OFFSETS[sc.id], sc.id)
            self.assertEqual(sel["bytes"], BYTES[sc.id], sc.id)
            self.assertEqual(sel["words"], OFFSETS[sc.id])
            self.assertEqual((sel["offset"], sel["words"], sel["bytes"]),
                             (OFFSETS[sc.id], ap[sc.id]["window_words"], ap[sc.id]["window_bytes"]))

    def test_window_is_sentence_complete_and_contains_the_whole_support_set(self):
        ap = approved()
        for sc in fixtures().scenarios:
            frag = C.treatment_fragment(sc, OFFSETS[sc.id], "words")      # the real extraction function
            self.assertTrue(frag.endswith("."), sc.id)
            self.assertEqual(frag, ap[sc.id]["displaced_window"], sc.id)
            for phrase in ap[sc.id]["support_set"]:
                self.assertIn(phrase.lower(), frag.lower(), (sc.id, phrase))

    def test_b_cue_phrases_are_inside_the_support_set_and_the_window(self):
        for sid, a in approved().items():
            self.assertTrue(a["b_cue_phrases"], sid)
            for p in a["b_cue_phrases"]:
                self.assertIn(p, a["support_set"], sid)
                self.assertIn(p.lower(), a["displaced_window"].lower(), sid)

    def test_no_support_term_lives_in_the_unread_trailing_sentence(self):
        for sc in fixtures().scenarios:
            sel = P.select_window_rule_s(sc, approved()[sc.id]["support_set"])
            self.assertEqual(sel["support_terms_in_trailing"], [], sc.id)
            self.assertEqual(sel["trailing_sentence"], approved()[sc.id]["unread_trailing_sentence"])

    def test_the_rule_is_minimal_every_smaller_sentence_end_fails(self):
        for sc in fixtures().scenarios:
            support = approved()[sc.id]["support_set"]
            sel = P.select_window_rule_s(sc, support)
            self.assertTrue(sel["rejected_smaller_ends"], sc.id)
            for n in sel["rejected_smaller_ends"]:
                frag = C.treatment_fragment(sc, n, "words").lower()
                self.assertFalse(all(p.lower() in frag for p in support), (sc.id, n))

    def test_rule_s_fails_closed_when_the_support_set_cannot_be_satisfied(self):
        sc = fixtures().scenarios[0]
        with self.assertRaises(P.ProtocolError):
            P.select_window_rule_s(sc, ["a phrase that occurs nowhere in the slot"])
        with self.assertRaises(P.ProtocolError):
            P.select_window_rule_s(sc, approved()[sc.id]["support_set"], unit="bytes")


# --------------------------------------------------------------------------------------
class TestControls(unittest.TestCase):
    def test_every_control_is_present_and_parity_is_recorded(self):
        m = C.load_manifest(SEM_MANIFEST)
        fx = fixtures()
        self.assertEqual(m["schema"], C.MANIFEST_SCHEMA_V2)
        self.assertEqual(m["fixtures_sha256"], fx.sha256)
        self.assertEqual(m["offsets"], OFFSETS)
        self.assertEqual(m["carried_over"]["scenarios"], [])
        self.assertEqual(sorted(m["generated"]), sorted(OFFSETS))
        for sc in fx.scenarios:
            e = m["controls"][sc.id]
            treat = C.treatment_fragment(sc, OFFSETS[sc.id], "words")
            self.assertEqual(e["offset"], OFFSETS[sc.id])
            self.assertEqual(e["treatment_fragment_sha256"], C.fragment_sha256(treat))
            self.assertEqual(C.fragment_sha256(e["fragment_text"]), e["fragment_sha256"])
            self.assertEqual(len(e["fragment_text"].encode()), len(treat.encode()), sc.id)   # exact byte parity
            self.assertEqual(len(e["fragment_text"].split()), len(treat.split()), sc.id)      # exact word parity
            self.assertEqual(e["token_parity"], "VERIFIED")
            for key in ("bytes", "words", "fragment_tokens", "prompt_tokens"):       # exact parity, incl. real tokenizer
                self.assertEqual(e["counts"][key][0], e["counts"][key][1], (sc.id, key))
            self.assertEqual(e["counts"]["bytes"][0], len(treat.encode()))
            self.assertEqual(e["counts"]["words"][0], len(treat.split()))
            self.assertEqual(e["algorithm"], "neutral-prose/1")
            self.assertEqual(e["tokenizer_fingerprint"], m["tokenizer_fingerprint"])
            self.assertEqual(e["lexicon_sha256"], C.LEXICON_SHA256)

    def test_stored_controls_pass_every_tokenizer_free_check(self):
        m = C.load_manifest(SEM_MANIFEST)
        for sc in fixtures().scenarios:
            e = m["controls"][sc.id]
            treat = C.treatment_fragment(sc, OFFSETS[sc.id], "words")
            sk = C.parse_skeleton(treat)
            vocab, runs = C.scenario_vocabulary(sc)
            words = e["fragment_text"].split()
            self.assertEqual(C.lexical_problems(words, sk, vocab, runs), [], sc.id)
            self.assertEqual(C.skeleton_problems(words, sk), [], sc.id)
            self.assertEqual(C.recognize_fragment(words, sk, e["cluster"]), [], sc.id)

    def test_boundary_and_insertion_geometry_are_identical(self):
        from bleediff.arena import plan_requests

        m = C.load_manifest(SEM_MANIFEST)
        for sc in fixtures().scenarios:
            off = OFFSETS[sc.id]
            t = plan_requests(sc, "treatment", off, "words")
            c = plan_requests(sc, "control", off, "words", filler=m["controls"][sc.id]["fragment_text"])
            self.assertEqual((t.geometry.bleed_bytes, t.geometry.bleed_words, t.geometry.boundary),
                             (c.geometry.bleed_bytes, c.geometry.bleed_words, c.geometry.boundary), sc.id)
            self.assertEqual(len(t.prompts[2]), len(c.prompts[2]), sc.id)
            self.assertEqual([bytes(x) for x in t.prompts[:2]], [bytes(x) for x in c.prompts[:2]], sc.id)      # requests 1 and 2 are identical

    def test_controls_never_contain_the_treatment_evidence(self):
        ap = approved()
        m = C.load_manifest(SEM_MANIFEST)
        for sid, a in ap.items():
            text = m["controls"][sid]["fragment_text"].lower()
            for phrase in a["support_set"]:
                self.assertNotIn(phrase.lower(), text, sid)
            for opt in a["options"].values():
                self.assertNotIn(opt.lower(), text, sid)

    def test_generation_is_deterministic_and_fails_closed_without_the_real_tokenizer(self):
        fx = fixtures()
        for sc in fx.scenarios:
            outcomes = []
            for _ in range(2):
                try:
                    outcomes.append(("ok", C.generate_control(sc, OFFSETS[sc.id], "words", qwenish, max_draws=400).to_json()))
                except ControlMatchError as e:
                    outcomes.append(("fail", e.reason))
            self.assertEqual(outcomes[0], outcomes[1], sc.id)                 # same inputs, same answer, never random
            if outcomes[0][0] == "fail":
                self.assertEqual(outcomes[0][1], "CONTROL_MATCH_FAILED", sc.id)

    def test_control_id_is_reproducible_from_its_inputs(self):
        m = C.load_manifest(SEM_MANIFEST)
        for sc in fixtures().scenarios:
            e = m["controls"][sc.id]
            cid = C.control_id_for(scenario_id=sc.id, offset=OFFSETS[sc.id], unit="words",
                                   treatment_sha256=e["treatment_fragment_sha256"], fingerprint=m["tokenizer_fingerprint"])
            self.assertEqual(cid, e["control_id"], sc.id)

    def test_the_generator_algorithm_is_the_frozen_one(self):
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
        self.assertEqual(C.ALGORITHM, "neutral-prose/1")
        self.assertEqual(C.LEXICON_SHA256, "f91e691f054f431d7ea772e99f90238d354a4b7dd991a7dcd333b2fcafd03ba5")


# --------------------------------------------------------------------------------------
class TestProtocolManifest(unittest.TestCase):
    def setUp(self):
        self.p = P.load_protocol(SEM_PROTOCOL)

    def test_identity_fields(self):
        p = self.p
        self.assertEqual(p["protocol_id"], "semantic-v2/1")
        self.assertEqual(p["schema"], "bleediff.protocol/1")
        self.assertEqual(p["fixtures_sha256"], SEM_FIXTURES_SHA256)
        self.assertEqual(p["scenario_offsets"], OFFSETS)
        self.assertEqual(p["window_rule"], P.RULE_S)
        self.assertEqual(p["control_algorithm"], "neutral-prose/1")
        self.assertEqual(p["design_reference"]["approved_design_sha256"], APPROVED_SHA256)
        self.assertRegex(p["tokenizer_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertEqual(p["code_revision"]["git_commit"], "8bc3db9ab7a3a78a5342434fc768f49c8cd14031")
        self.assertEqual(p["parent_protocol"]["id"], "window-repair/1")

    def test_hashes_bind_the_actual_files(self):
        self.assertEqual(self.p["controls_manifest_sha256"], _sha(SEM_MANIFEST))
        m = C.load_manifest(SEM_MANIFEST)
        self.assertEqual(self.p["tokenizer_fingerprint"], m["tokenizer_fingerprint"])
        self.assertEqual(self.p["lexicon_sha256"], m["lexicon_sha256"])
        self.assertEqual(m["generator"]["protocol_id"], self.p["protocol_id"])

    def test_three_identities_are_distinct_and_recorded(self):
        m = C.load_manifest(SEM_MANIFEST)
        g = m["generator"]
        self.assertEqual(g, {"software_version": __version__, "control_algorithm": "neutral-prose/1",
                             "protocol_id": "semantic-v2/1"})
        self.assertEqual(len({g["software_version"], g["control_algorithm"], g["protocol_id"]}), 3)
        self.assertEqual(set(self.p["version_semantics"]), {"protocol_id", "control_algorithm", "software_version"})

    def test_per_scenario_metadata(self):
        ap = approved()
        for sid, meta in self.p["scenarios"].items():
            self.assertEqual(meta["window_words"], OFFSETS[sid])
            self.assertEqual(meta["window_bytes"], BYTES[sid])
            self.assertEqual(meta["support_set"], ap[sid]["support_set"])
            self.assertEqual(meta["b_cue_phrases"], ap[sid]["b_cue_phrases"])
            self.assertEqual(meta["option_B"], ap[sid]["options"]["B"])
            self.assertEqual(meta["option_A"], ap[sid]["options"]["A"])
            self.assertRegex(meta["support_set_sha256"], r"^[0-9a-f]{64}$")

    def test_static_verification_passes(self):
        self.assertEqual(P.verify_protocol_static(self.p, fixtures()), [])

    def test_static_verification_rejects_a_tampered_protocol(self):
        fx = fixtures()
        for mutate, expect in (
            (lambda p: p["scenario_offsets"].__setitem__("datastore-v2", 38), "offset"),
            (lambda p: p["scenarios"]["datastore-v2"].__setitem__("window_bytes", 999), "bytes"),
            (lambda p: p["scenarios"]["vendor-routing-v2"].__setitem__("fragment_sha256", "0" * 64), "fragment_sha256"),
            (lambda p: p["scenarios"]["config-audit-v2"].__setitem__("trailing_sentence_sha256", "0" * 64), "trailing"),
            (lambda p: p.__setitem__("window_rule", "something else"), "Rule S"),
            (lambda p: p["scenarios"]["incident-root-cause-v2"]["b_cue_phrases"].append("not in support set"), "support set"),
        ):
            p = copy.deepcopy(self.p)
            mutate(p)
            problems = P.verify_protocol_static(p, fx)
            self.assertTrue(problems, expect)

    def test_protocol_for_other_fixtures_is_refused(self):
        p = copy.deepcopy(self.p)
        p["fixtures_sha256"] = "0" * 64
        with self.assertRaises(P.ProtocolError):
            P.scenario_offsets(p, fixtures())

    def test_manifest_hash_mismatch_is_refused_by_bounce(self):
        # a doctored controls manifest must be refused before any probability is requested
        with tempfile.TemporaryDirectory() as d:
            doc = _json(SEM_MANIFEST)
            doc["controls"]["datastore-v2"]["fragment_text"] += " "
            alt = os.path.join(d, "alt.json")
            with open(alt, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(doc, fh, indent=2)
            baseline = self._valid_baseline(d)
            with serve() as (url, state):
                code, text = run_cli("bounce", "--server", url, "--baseline", baseline, "--fixtures", SEM_FIXTURES,
                                     "--protocol", SEM_PROTOCOL, "--controls-manifest", alt,
                                     "--trials", "1", "--out", os.path.join(d, "t.jsonl"))
                self.assertEqual(code, 2, text)
                self.assertEqual([r for r in state.requests], [])           # no /completion reached the server

    def _valid_baseline(self, d: str) -> str:
        """A minimal baseline that carries the semantic protocol identity (structure only; no measurements are claimed)."""
        b = _json(LEGACY_BASELINE)
        b["fixtures_sha256"] = SEM_FIXTURES_SHA256
        b["protocol"] = {"protocol_id": "semantic-v2/1", "protocol_file_sha256": _sha(SEM_PROTOCOL)}
        path = os.path.join(d, "baseline.json")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(b, fh)
        return path


# --------------------------------------------------------------------------------------
class TestHistoricalIsolation(unittest.TestCase):
    def test_historical_files_are_byte_identical(self):
        for rel, sha in HISTORICAL.items():
            self.assertTrue(_sha(os.path.join(ROOT, *rel.split("/"))).startswith(sha), rel)

    def test_window_repair_protocol_is_untouched_and_distinct(self):
        old = P.load_protocol(os.path.join(ROOT, "fixtures", "protocols", "window-repair-1.json"))
        new = P.load_protocol(SEM_PROTOCOL)
        self.assertEqual(old["protocol_id"], "window-repair/1")
        self.assertNotEqual(old["protocol_id"], new["protocol_id"])
        self.assertNotEqual(old["fixtures_sha256"], new["fixtures_sha256"])
        self.assertNotIn("scenarios", old)
        self.assertNotEqual((old.get("baseline_policy") or {}).get("policy"), P.BASELINE_POLICY_BOUND)
        self.assertEqual(old["parent_protocol"]["id"], "uniform-24-words (v0.1.0, v0.1.1, v0.1.2)")

    def test_window_repair_protocol_still_loads_against_its_own_fixtures(self):
        fx = load_fixtures(os.path.join(ROOT, "fixtures", "contexts.json"))
        old = P.load_protocol(os.path.join(ROOT, "fixtures", "protocols", "window-repair-1.json"))
        offs = P.scenario_offsets(old, fx)
        self.assertEqual(offs["triage-severity"], 45)
        with self.assertRaises(P.ProtocolError):                    # and cannot be bound to the semantic fixtures
            P.scenario_offsets(old, fixtures())

    def test_scenario_ids_do_not_collide_with_historical_ids(self):
        old_ids = {s.id for s in load_fixtures(os.path.join(ROOT, "fixtures", "contexts.json")).scenarios}
        self.assertFalse(old_ids & {s.id for s in fixtures().scenarios})

    def test_no_baseline_was_generated_for_semantic_v2(self):
        for dirpath, _dirs, files in os.walk(SEM):
            for f in files:
                self.assertFalse(f.startswith("baseline"), f)
        self.assertIn("NOT been run", P.load_protocol(SEM_PROTOCOL)["baseline_policy"]["historical_baselines"])


# --------------------------------------------------------------------------------------
class TestBaselineRejection(unittest.TestCase):
    def setUp(self):
        self.fx = fixtures()
        self.p = P.load_protocol(SEM_PROTOCOL)
        self.good = {"fixtures_sha256": SEM_FIXTURES_SHA256,
                     "protocol": {"protocol_id": "semantic-v2/1", "protocol_file_sha256": _sha(SEM_PROTOCOL)}}

    def _refused(self, baseline, protocol=None, path=SEM_PROTOCOL):
        with self.assertRaises(P.ProtocolError) as cm:
            P.check_baseline_identity(baseline, protocol or self.p, path, self.fx)
        self.assertEqual(cm.exception.reason, "BASELINE_PROTOCOL_MISMATCH")

    def test_matching_identity_is_accepted(self):
        P.check_baseline_identity(self.good, self.p, SEM_PROTOCOL, self.fx)

    def test_a_historical_baseline_is_rejected_for_fixture_mismatch(self):
        self._refused(_json(LEGACY_BASELINE))

    def test_a_baseline_without_a_protocol_identity_is_rejected(self):
        b = dict(self.good)
        del b["protocol"]
        self._refused(b)
        b["protocol"] = "NONE"
        self._refused(b)

    def test_another_protocol_id_is_rejected(self):
        b = copy.deepcopy(self.good)
        b["protocol"]["protocol_id"] = "window-repair/1"
        self._refused(b)

    def test_a_changed_protocol_file_is_rejected(self):
        b = copy.deepcopy(self.good)
        b["protocol"]["protocol_file_sha256"] = "0" * 64
        self._refused(b)
        self._refused(self.good, path=None)

    def test_relabelling_does_not_help_when_fixtures_differ(self):
        b = _json(LEGACY_BASELINE)
        b["protocol"] = self.good["protocol"]                       # a legacy baseline wearing the new label
        self._refused(b)

    def test_bounce_refuses_a_historical_baseline_before_any_server_contact(self):
        with tempfile.TemporaryDirectory() as d, serve() as (url, state):
            code, text = run_cli("bounce", "--server", url, "--baseline", LEGACY_BASELINE, "--fixtures", SEM_FIXTURES,
                                 "--protocol", SEM_PROTOCOL, "--trials", "1", "--out", os.path.join(d, "t.jsonl"))
            self.assertEqual(code, 2, text)
            self.assertIn("BASELINE_PROTOCOL_MISMATCH", text + "")
            self.assertEqual(state.requests, [])
            self.assertFalse(os.path.exists(os.path.join(d, "t.jsonl")))

    def test_bounce_refuses_a_same_fixtures_baseline_without_identity(self):
        with tempfile.TemporaryDirectory() as d, serve() as (url, state):
            b = _json(LEGACY_BASELINE)
            b["fixtures_sha256"] = SEM_FIXTURES_SHA256
            path = os.path.join(d, "b.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(b, fh)
            code, text = run_cli("bounce", "--server", url, "--baseline", path, "--fixtures", SEM_FIXTURES,
                                 "--protocol", SEM_PROTOCOL, "--trials", "1", "--out", os.path.join(d, "t.jsonl"))
            self.assertEqual(code, 2, text)
            self.assertEqual(state.requests, [])

    def test_calibrate_requires_the_protocol_for_semantic_fixtures_and_does_not_contact_the_server(self):
        with tempfile.TemporaryDirectory() as d, serve() as (url, state):
            code, text = run_cli("calibrate", "--server", url, "--fixtures", SEM_FIXTURES, "--out", os.path.join(d, "b.json"))
            self.assertEqual(code, 2, text)
            self.assertEqual(state.requests, [])
            self.assertFalse(os.path.exists(os.path.join(d, "b.json")))

    def test_calibrate_with_protocol_stamps_the_identity_and_never_calls_completion_on_a_wrong_protocol(self):
        # the mock server's tokenizer is not Qwen's, so only the identity stamp is checked here (structure, not measurement)
        with tempfile.TemporaryDirectory() as d, serve() as (url, _state):
            out = os.path.join(d, "b.json")
            code, text = run_cli("calibrate", "--server", url, "--fixtures", SEM_FIXTURES, "--protocol", SEM_PROTOCOL, "--out", out)
            self.assertEqual(code, 0, text)
            b = _json(out)
            self.assertEqual(b["protocol"], {"protocol_id": "semantic-v2/1", "protocol_file_sha256": _sha(SEM_PROTOCOL)})
            self.assertEqual(b["fixtures_sha256"], SEM_FIXTURES_SHA256)
            P.check_baseline_identity(b, self.p, SEM_PROTOCOL, self.fx)

    def test_historical_calibration_without_protocol_still_records_none(self):
        with tempfile.TemporaryDirectory() as d, serve() as (url, _state):
            out = os.path.join(d, "b.json")
            code, text = run_cli("calibrate", "--server", url, "--fixtures", os.path.join(ROOT, "fixtures", "contexts.json"),
                                 "--out", out)
            self.assertEqual(code, 0, text)
            self.assertEqual(_json(out)["protocol"], "NONE")


# --------------------------------------------------------------------------------------
class TestFrozenMechanicsPreserved(unittest.TestCase):
    def test_fold_and_contrast_source_is_unchanged(self):
        anchors = _json(FROZEN_FOLD)["anchors"]
        text = _read(os.path.join(ROOT, "bleediff", "fold.py"))
        got = {}
        for n in ast.parse(text).body:
            nm = getattr(n, "name", None)
            if nm is None and isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                nm = n.targets[0].id
            if nm in anchors["bleediff/fold.py"]:
                got[nm] = hashlib.sha256(ast.get_source_segment(text, n).replace("\r\n", "\n").encode()).hexdigest()
        self.assertEqual(got, anchors["bleediff/fold.py"])

    def test_thresholds_normalisation_and_schema(self):
        from bleediff import CONTROL_BAND_NATS, EPS, EXPOSURE_DELTA_NATS, MIN_TREATMENT_FRACTION

        self.assertEqual((EXPOSURE_DELTA_NATS, CONTROL_BAND_NATS, MIN_TREATMENT_FRACTION, EPS), (0.4, 0.1, 0.8, 1e-12))
        self.assertEqual(len(TRACE_SCHEMA), 15)
        frozen = P.load_protocol(SEM_PROTOCOL)["frozen"]
        self.assertEqual(frozen["thresholds"], {"exposure_delta_nats": 0.4, "control_tolerance_nats": 0.1,
                                                "min_treatment_fraction": 0.8})
        self.assertEqual(frozen["requests"], 3)
        self.assertIn("15 fields", frozen["trace_schema"])
        self.assertEqual(frozen["contrast_equation"], "R_c = log(M_c+eps) - log(B_ref_c+eps), eps=1e-12")

    def test_fold_still_discards_the_vectors_and_is_the_mean(self):
        from bleediff.fold import FoldAccumulator

        acc = FoldAccumulator(3)
        vecs = ([0.7, 0.1, 0.1, 0.1], [0.4, 0.3, 0.2, 0.1], [0.1, 0.5, 0.2, 0.2])
        for v in vecs:
            acc.absorb(v)
            self.assertEqual(v, [])                              # the caller's vector is consumed and scrubbed
        m = acc.finalize()
        self.assertEqual([round(x, 9) for x in m], [round(x, 9) for x in (0.4, 0.3, 0.16666666666666666, 0.13333333333333334)])
        with self.assertRaises(RuntimeError):
            acc.finalize()                                       # irreversible: a second fold is refused

    def test_selected_options_are_the_frozen_four_choices(self):
        for sc in fixtures().scenarios:
            self.assertEqual(sorted(sc.options), ["A", "B", "C", "D"])

    def test_the_semantic_trace_schema_is_the_historical_one(self):
        old = _json(os.path.join(ROOT, "tests", "data", "legacy_qwen1p5b_v0.1.0", "baseline.sanitized.json"))
        self.assertTrue(old)        # legacy evidence is preserved and not reinterpreted
        with open(os.path.join(ROOT, "tests", "data", "legacy_qwen1p5b_v0.1.0", "traces.jsonl"), encoding="utf-8") as fh:
            first = json.loads(fh.readline())
        self.assertEqual(list(first), list(TRACE_SCHEMA))


# --------------------------------------------------------------------------------------
class TestFailClosedAndIsolation(unittest.TestCase):
    def test_protocol_with_wrong_scenario_set_is_refused(self):
        p = _json(SEM_PROTOCOL)
        del p["scenario_offsets"]["triage-severity-v2"]
        with self.assertRaises(P.ProtocolError):
            P.scenario_offsets(p, fixtures())

    def test_non_positive_offsets_are_refused(self):
        p = _json(SEM_PROTOCOL)
        p["scenario_offsets"]["datastore-v2"] = 0
        with self.assertRaises(P.ProtocolError):
            P.scenario_offsets(p, fixtures())

    def test_requires_protocol_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            doc = _json(SEM_FIXTURES)
            doc["requires_protocol"] = "some-other/9"
            path = os.path.join(d, "c.json")
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(doc, fh)
            p = _json(SEM_PROTOCOL)
            p["fixtures_sha256"] = load_fixtures(path).sha256       # make hash agree so only the id can fail
            ppath = os.path.join(d, "p.json")
            with open(ppath, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(p, fh)
            code, text = run_cli("bleed", "--fixtures", path, "--protocol", ppath)
            self.assertEqual(code, 2, text)

    def test_unreadable_or_foreign_protocol_files_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(P.ProtocolError):
                P.load_protocol(os.path.join(d, "missing.json"))
            bad = os.path.join(d, "bad.json")
            with open(bad, "w", encoding="utf-8") as fh:
                json.dump({"schema": "something/9"}, fh)
            with self.assertRaises(P.ProtocolError):
                P.load_protocol(bad)

    def test_the_tokenizer_only_helper_never_posts_completion(self):
        text = _read(os.path.join(ROOT, "bleediff", "control.py"))
        self.assertNotIn(".completion(", text)
        self.assertEqual(text.count("LlamaClient("), 1, "only the tokenizer helper may build a client")
        self.assertEqual(text.count("client.tokenize("), 1)

    def test_bleed_preview_with_protocol_reports_exact_parity(self):
        code, text = run_cli("bleed", "--fixtures", SEM_FIXTURES, "--protocol", SEM_PROTOCOL)
        self.assertEqual(code, 0, text)
        self.assertEqual(text.count("matched      YES"), 5)
        for sid, n in BYTES.items():
            self.assertIn(f"+{n} bytes / {OFFSETS[sid]} words", text)


class TestVersioningAndDependencies(unittest.TestCase):
    def test_version_is_consistent(self):
        import tomllib

        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
            self.assertEqual(tomllib.load(fh)["project"]["version"], __version__)
        self.assertEqual(__version__, "0.1.4")
        self.assertEqual(_json(SEM_PROTOCOL)["generator"]["software_version"], __version__)

    def test_no_third_party_imports(self):
        allowed = set(sys.stdlib_module_names) | {"bleediff", "tests", "test_bleediff", "_tok"}
        for rel in ("bleediff/protocol.py", "bleediff/cli.py", "bleediff/runner.py", "bleediff/arena.py",
                    "tests/test_v014_semantic_v2.py"):
            tree = ast.parse(_read(os.path.join(ROOT, *rel.split("/"))))
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mods = [node.module.split(".")[0]]
                self.assertFalse([m for m in mods if m not in allowed], (rel, mods))
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
            import tomllib

            self.assertEqual(tomllib.load(fh)["project"].get("dependencies", []), [])

    def test_the_new_fixtures_and_protocol_contain_no_personal_or_local_data(self):
        pat = re.compile(r"(@[A-Za-z0-9-]+\.[a-z]{2,}|[A-Za-z]:\\|/Users/|/home/|\.gguf|api[_-]?key|password)", re.I)
        for path in (SEM_FIXTURES, SEM_PROTOCOL, SEM_MANIFEST, APPROVED):
            text = _read(path)
            self.assertIsNone(pat.search(text), path)


if __name__ == "__main__":
    unittest.main()
