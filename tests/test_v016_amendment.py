"""v0.1.6 tests: AMENDMENT-001 to the config-audit robustness preregistration (selection integrity lock).

No inference, no server, no real tokenizer and no real controls. Generation mechanics are exercised with the OFFLINE
stand-in tokenizer (tests/_tok.py) and a SYNTHETIC C0 produced by the frozen generator under that stand-in: the REAL
C0 cannot be reproduced by the stand-in (a test asserts that the anchor gate trips), which is exactly why the live
anchor is a Phase 1B step.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import inspect
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from _tok import qwenish  # noqa: E402

from bleediff import CONTROL_BAND_NATS, EXPOSURE_DELTA_NATS, __version__  # noqa: E402
from bleediff import control as C  # noqa: E402
from bleediff import replicates as R  # noqa: E402
from bleediff import robustness as RB  # noqa: E402
from bleediff.arena import load_fixtures  # noqa: E402

ROB_DIR = os.path.join(ROOT, "fixtures", "config-audit-robustness")
EFFECTIVE = os.path.join(ROB_DIR, "protocol-1.1.json")
ORIGINAL_PROTOCOL = os.path.join(ROB_DIR, "protocol.json")
ORIGINAL_PREREG = os.path.join(ROB_DIR, "PREREGISTRATION.md")
AMENDMENT = os.path.join(ROB_DIR, "AMENDMENT-001.md")
SEM_MANIFEST = os.path.join(ROOT, "fixtures", "semantic-v2", "controls", "neutral-prose-1.semantic-v2-1.json")
SID, OFFSET = "config-audit-v2", 25

# independent literals (not read from the code under test)
ORIG_PREREG_SHA = "74c118cc9c14c892ee5915fe20e83229e558783031e64d872db62e82e2ba437b"
ORIG_PROTOCOL_SHA = "d9cc96c6cd3ebcabd2bc9e2cc16eb43dbce9de368d4ebf0ff1883b74c81aac4c"
ORIG_RUBRIC_SHA = "020552891747c7161e3277d8e8dc9a5e4690f3902da7c100410a016030110264"
C0_ID = "627e6cf41b3d02a8007269f298e4b214"
EXPECTED_DOMAINS = ["bakery", "festival", "greenhouse", "library", "orchestra", "seaside", "trail", "workshop",
                    "bakery", "festival", "greenhouse", "library"]
REV = {"git_commit": "a" * 40, "dirty": False}


def _json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def fx():
    return load_fixtures(os.path.join(ROOT, "fixtures", "semantic-v2", "contexts.json"))


def scen():
    return {s.id: s for s in fx().scenarios}[SID]


_W: dict = {}


def world() -> dict:
    """A synthetic C0, matching controlling documents and anchor under the stand-in tokenizer, and one built manifest."""
    if not _W:
        sc = scen()
        rec = C.generate_control(sc, OFFSET, "words", qwenish, max_draws=300).to_json()
        fp = C.tokenizer_fingerprint(qwenish)
        real = RB.load_controlling_documents(ROOT)
        ctl = {**real, "tokenizer_fingerprint_expected": fp,
               "c0": {"control_id": rec["control_id"], "fragment_sha256": rec["fragment_sha256"]}}
        anchor = R.run_anchor(sc, OFFSET, qwenish, rec)
        m = R.build_replicate_manifest(fx(), SID, OFFSET, qwenish, c0_entry=rec, c0_manifest_sha256="b" * 64,
                                       controlling=ctl, anchor=anchor, code_revision=REV)
        _W.update(sc=sc, rec=rec, fp=fp, real=real, ctl=ctl, anchor=anchor, m=m)
    return _W


def verify(m, ctl=None, rec=None):
    w = world()
    return R.verify_replicate_manifest(m, fx(), qwenish, c0_entry=rec or w["rec"], c0_manifest_sha256="b" * 64,
                                       controlling=ctl or w["ctl"])


def accepted_texts(m):
    return [r["control"]["fragment_text"] for r in m["selection"]["replicates"] if r["status"] == "ACCEPTED"]


# --------------------------------------------------------------------------------------
class TestDrawBudgetIsLocked(unittest.TestCase):
    def test_budget_is_exactly_5000_everywhere(self):
        self.assertEqual(R.PROTOCOL_DRAW_BUDGET, 5000)
        self.assertEqual(R.DEFAULT_MAX_DRAWS, 5000)
        self.assertEqual(C.DEFAULT_MAX_DRAWS, 5000)
        p = _json(EFFECTIVE)
        self.assertEqual(p["selection"]["draw_budget_per_replicate"], 5000)
        self.assertIs(p["selection"]["draw_budget_overridable"], False)
        m = world()["m"]
        self.assertEqual(m["selection"]["effective_draw_budget"], 5000)
        self.assertTrue(all(r["effective_draw_budget"] == 5000 for r in m["selection"]["replicates"]))
        self.assertEqual(m["provenance"]["generation"]["invocation"]["parameters"]["draw_budget_per_replicate"], 5000)

    def test_the_cli_has_no_way_to_override_the_budget_or_the_experiment(self):
        base = {"generate": ["--anchor-evidence", "a", "--out", "o"], "anchor": ["--out", "o"], "verify": ["--manifest", "m"]}
        for cmd, extra in base.items():
            for flag in (["--max-draws", "10"], ["--scenario", "x"], ["--offset", "30"], ["--fixtures", "f"],
                         ["--protocol-id", "p"], ["--c0-manifest", "c"], ["--budget", "1"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                    R.main([cmd, "--server", "http://127.0.0.1:9", *extra, *flag])
                self.assertEqual(cm.exception.code, 2, (cmd, flag))
        src = inspect.getsource(R.main)
        for needle in ("max-draws", "max_draws", "--scenario", "--offset"):
            self.assertNotIn(needle, src)

    def test_library_generation_with_another_budget_is_refused(self):
        w = world()
        with self.assertRaises(C.ControlMatchError) as cm:
            R.build_replicate_manifest(fx(), SID, OFFSET, qwenish, c0_entry=w["rec"], c0_manifest_sha256="b" * 64,
                                       controlling=w["ctl"], anchor=w["anchor"], code_revision=REV, max_draws=50)
        self.assertEqual(cm.exception.reason, "CONTROL_PROVENANCE_INVALID")
        self.assertIn("draw budget", str(cm.exception))

    def test_a_manifest_recording_another_budget_fails_verification(self):
        m = copy.deepcopy(world()["m"])
        self.assertEqual(verify(m), [])
        m["selection"]["effective_draw_budget"] = 50
        self.assertTrue([p for p in verify(m) if "draw budget" in p])
        m2 = copy.deepcopy(world()["m"])
        m2["selection"]["replicates"][3]["effective_draw_budget"] = 77
        self.assertTrue([p for p in verify(m2) if "draw budget" in p])
        m3 = copy.deepcopy(world()["m"])
        m3["provenance"]["generation"]["invocation"]["parameters"]["draw_budget_per_replicate"] = 10
        self.assertTrue([p for p in verify(m3) if "budget" in p])

    def test_a_controlling_document_with_another_budget_is_refused(self):
        ctl = {**world()["ctl"], "draw_budget": 10}
        self.assertTrue(R._check_controlling(ctl, world()["fp"], SID, OFFSET, "words", world()["rec"]))

    def test_the_budget_changes_nothing_silently_zero_budget_gives_recorded_failures(self):
        sel = R.build_replicate_selection(fx(), SID, OFFSET, qwenish, c0_entry=world()["rec"], c0_manifest_sha256="b" * 64,
                                          max_draws=0)
        self.assertEqual(sel["effective_draw_budget"], 0)                 # recorded, never hidden
        self.assertEqual(sel["summary"]["n_accepted"], 0)


# --------------------------------------------------------------------------------------
class TestRegenerationVerification(unittest.TestCase):
    def test_a_correct_manifest_verifies_by_byte_identical_regeneration(self):
        w = world()
        self.assertEqual(verify(w["m"]), [])
        again = R.build_replicate_selection(fx(), SID, OFFSET, qwenish, c0_entry=w["rec"], c0_manifest_sha256="b" * 64)
        self.assertEqual(R.canonical(again), R.canonical(w["m"]["selection"]))

    def test_verification_needs_a_tokenizer(self):
        w = world()
        self.assertTrue(R.verify_replicate_manifest(w["m"], fx(), None, c0_entry=w["rec"], c0_manifest_sha256="b" * 64,
                                                    controlling=w["ctl"]))

    def test_a_substituted_but_structurally_valid_control_is_not_accepted(self):
        w = world()
        m = copy.deepcopy(w["m"])
        reps = m["selection"]["replicates"]
        k = 11                                                     # last replicate: nothing after it depends on its text
        self.assertEqual(reps[k]["status"], "ACCEPTED")
        before = [w["rec"]["fragment_text"]] + [r["control"]["fragment_text"] for r in reps[:k] if r["status"] == "ACCEPTED"]
        alt = R.search_stream(w["sc"], OFFSET, "words", qwenish, seed=int(reps[k]["seed_hex16"], 16) ^ 0xABCDEF,
                              forced_domain=R.domain_for(k), screen_fn=lambda t: R.screen(t, w["sc"], accepted_texts=tuple(before)),
                              max_draws=300, fingerprint=w["fp"])
        self.assertEqual(alt["status"], "ACCEPTED")
        self.assertNotEqual(alt["text"], reps[k]["control"]["fragment_text"])
        e = reps[k]["control"]
        e.update(fragment_text=alt["text"], fragment_sha256=C.fragment_sha256(alt["text"]), enumeration_index=alt["draw"],
                 counts={a: list(b) for a, b in alt["counts"].items()})
        problems = verify(m)
        # the substitute satisfies every STRUCTURAL check: no problem is attributed to the replicate itself ...
        self.assertEqual([p for p in problems if p.startswith(f"replicate {k}:") and "differs from regeneration" not in p], [])
        # ... and it is rejected only because the selection is not what the recorded seed deterministically yields
        self.assertTrue([p for p in problems if "NOT byte-identical" in p])
        self.assertTrue([p for p in problems if f"replicate {k} differs from regeneration" in p])

    def test_regeneration_reproduces_accepted_and_rejected_candidates(self):
        w = world()
        reps = w["m"]["selection"]["replicates"]
        self.assertTrue([r for r in reps if r["rubric_rejections"]], "the stand-in run should contain a rubric rejection")
        again = R.build_replicate_selection(fx(), SID, OFFSET, qwenish, c0_entry=w["rec"], c0_manifest_sha256="b" * 64)["replicates"]
        for a, b in zip(reps, again):
            self.assertEqual(a["rubric_rejections"], b["rubric_rejections"])               # same rejected texts and reasons, same order
            self.assertEqual(a["structural_rejections"], b["structural_rejections"])
            self.assertEqual(a["draws_used"], b["draws_used"])
            self.assertEqual(a["control_id"], b["control_id"])
            self.assertEqual(a.get("control"), b.get("control"))
            self.assertEqual(a["status"], b["status"])
        # every recorded rejection really fails the rubric against the accepted set at that time
        accepted = [w["rec"]["fragment_text"]]
        for r in reps:
            for rej in r["rubric_rejections"]:
                self.assertTrue(R.screen(rej["text"], w["sc"], accepted_texts=tuple(accepted)))
            if r["status"] == "ACCEPTED":
                accepted.append(r["control"]["fragment_text"])

    def test_tampered_rejections_counts_and_identifiers_are_detected(self):
        m0 = world()["m"]
        k = next(r["replicate"] for r in m0["selection"]["replicates"] if r["rubric_rejections"])
        for mutate in (
            lambda d: d["selection"]["replicates"][k]["rubric_rejections"].pop(),
            lambda d: d["selection"]["replicates"][k]["rubric_rejections"][0]["reasons"].append(["R3_FIELD_WORD", "x"]),
            lambda d: d["selection"]["replicates"][k].__setitem__("draws_used", d["selection"]["replicates"][k]["draws_used"] + 1),
            lambda d: d["selection"]["replicates"][2]["structural_rejections"].__setitem__("lexical", 99),
            lambda d: d["selection"]["replicates"][5].__setitem__("control_id", "0" * 32),
            lambda d: d["selection"]["replicates"][4]["control"].__setitem__("enumeration_index", 3999),
        ):
            d = copy.deepcopy(m0)
            mutate(d)
            self.assertTrue(verify(d))

    def test_domain_and_seed_substitutions_are_detected(self):
        m0 = world()["m"]
        for mutate in (
            lambda d: d["selection"]["replicates"][0].__setitem__("domain", "workshop"),
            lambda d: d["selection"]["replicates"][1].__setitem__("seed_hex16", "0" * 16),
            lambda d: d["selection"]["replicates"].reverse(),
            lambda d: d["selection"]["replicates"].pop(),
            lambda d: d["selection"].__setitem__("rubric_sha256", "0" * 64),
            lambda d: d["selection"]["summary"].__setitem__("n_accepted", 3),
        ):
            d = copy.deepcopy(m0)
            mutate(d)
            self.assertTrue(verify(d))

    def test_a_replicate_cannot_be_turned_into_a_failure_or_a_failure_into_a_success(self):
        m = copy.deepcopy(world()["m"])
        rep = m["selection"]["replicates"][2]
        rep.pop("control")
        rep.update(status="FAILED", reason="REPLICATE_FAILED_NO_FEASIBLE_CANDIDATE", note="n", draws_used=5000)
        m["selection"]["summary"].update(n_accepted=11, n_failed=1, qualifies_for_analysis=True)
        self.assertTrue([p for p in verify(m) if "NOT byte-identical" in p])      # recorded failure state != regenerated state

    def test_failed_domains_remain_failed_and_are_never_substituted(self):
        sel = R.build_replicate_selection(fx(), SID, OFFSET, qwenish, c0_entry=world()["rec"], c0_manifest_sha256="b" * 64,
                                          max_draws=0)
        self.assertEqual([r["status"] for r in sel["replicates"]], ["FAILED"] * 12)
        self.assertEqual([r["domain"] for r in sel["replicates"]], EXPECTED_DOMAINS)
        self.assertTrue(all("control" not in r and r["reason"] == "REPLICATE_FAILED_NO_FEASIBLE_CANDIDATE" for r in sel["replicates"]))
        self.assertTrue(all("no other domain was substituted" in r["note"] for r in sel["replicates"]))
        self.assertFalse(sel["summary"]["qualifies_for_analysis"])
        self.assertIn("INSUFFICIENT", sel["summary"]["if_not_qualifying"])
        again = R.build_replicate_selection(fx(), SID, OFFSET, qwenish, c0_entry=world()["rec"], c0_manifest_sha256="b" * 64,
                                            max_draws=0)
        self.assertEqual(R.canonical(sel), R.canonical(again))                   # failure is itself deterministic

    def test_a_failed_replicate_that_did_not_use_the_whole_budget_is_rejected(self):
        m = copy.deepcopy(world()["m"])
        rep = m["selection"]["replicates"][7]
        rep.pop("control")
        rep.update(status="FAILED", reason="X", note="n", draws_used=12)
        self.assertTrue([p for p in verify(m) if "whole budget" in p])


# --------------------------------------------------------------------------------------
class TestProvenanceBinding(unittest.TestCase):
    def test_controlling_documents_bind_the_exact_files(self):
        ctl = RB.load_controlling_documents(ROOT)
        self.assertEqual(ctl["protocol_id"], "config-audit-robustness/1")
        self.assertEqual(ctl["protocol_revision"], "1.1")
        self.assertEqual(ctl["original_commit"], "58a00b1d149994eb3775570e85174fe4a3e2df21")
        self.assertEqual(ctl["original_preregistration_sha256"], ORIG_PREREG_SHA)
        self.assertEqual(ctl["original_protocol_sha256"], ORIG_PROTOCOL_SHA)
        self.assertEqual(ctl["amendment_sha256"], _sha(AMENDMENT))
        self.assertEqual(ctl["effective_protocol_sha256"], _sha(EFFECTIVE))
        self.assertEqual(ctl["rubric_sha256"], R.RUBRIC_SHA256)
        self.assertNotEqual(ctl["rubric_sha256"], ORIG_RUBRIC_SHA)
        self.assertEqual(ctl["draw_budget"], 5000)
        self.assertEqual(ctl["c0"]["control_id"], C0_ID)
        self.assertEqual(ctl["frozen_generator"]["lexicon_sha256"], C.LEXICON_SHA256)
        self.assertEqual(ctl["tokenizer_fingerprint_expected"], _json(os.path.join(ROOT, "fixtures", "semantic-v2", "protocol.json"))["tokenizer_fingerprint"])
        self.assertEqual(ctl["frozen_generator"]["control_py_sha256_newline_normalised"],
                         _json(ORIGINAL_PROTOCOL)["frozen"]["source_sha256_newline_normalised"]["bleediff/control.py"])

    def test_the_hash_chain_is_acyclic(self):
        eff_sha, am_sha = _sha(EFFECTIVE), _sha(AMENDMENT)
        self.assertNotIn(eff_sha, _read(AMENDMENT))                  # the amendment does not cite the effective protocol
        self.assertNotIn(eff_sha, _read(EFFECTIVE))                  # a file never cites its own hash
        self.assertNotIn(am_sha, _read(AMENDMENT))
        self.assertIn(am_sha, _read(EFFECTIVE))                      # effective protocol -> amendment -> originals
        self.assertIn(ORIG_PREREG_SHA, _read(AMENDMENT))
        self.assertIn(ORIG_PROTOCOL_SHA, _read(AMENDMENT))
        self.assertNotIn(eff_sha, _read(ORIGINAL_PROTOCOL))

    def test_the_manifest_records_every_required_identity(self):
        m = world()["m"]
        pr = m["provenance"]
        self.assertEqual(pr["generator"]["protocol_id"], "config-audit-robustness/1")
        self.assertEqual(pr["generator"]["protocol_revision"], "1.1")
        self.assertEqual(pr["generator"]["replicate_algorithm"], "neutral-prose-replicates/1")
        for key in ("original_preregistration_sha256", "original_protocol_sha256", "amendment_sha256", "effective_protocol_sha256",
                    "rubric_sha256", "frozen_generator", "c0", "draw_budget", "tokenizer_fingerprint_expected"):
            self.assertIn(key, pr["controlling_documents"], key)
        self.assertEqual(pr["anchor"]["evidence_sha256"], R.anchor_evidence_sha256(pr["anchor"]["evidence"]))
        self.assertEqual(pr["generation"]["code_revision"], REV)
        params = pr["generation"]["invocation"]["parameters"]
        for key in ("scenario_id", "offset", "unit", "draw_budget_per_replicate", "fixtures_sha256", "c0_manifest_sha256",
                    "anchor_evidence_sha256", "protocol_id", "protocol_revision", "effective_protocol_sha256"):
            self.assertIn(key, params, key)
        self.assertEqual(m["selection"]["tokenizer_fingerprint"], world()["fp"])
        self.assertEqual(m["selection"]["rubric_sha256"], R.RUBRIC_SHA256)
        self.assertEqual(m["selection"]["c0"]["control_id"], world()["rec"]["control_id"])
        self.assertEqual(pr["independence_statement"], R.INDEPENDENCE_STATEMENT)
        self.assertIn("not statistically independent or exchangeable", pr["independence_statement"].replace("do not make them ", "not "))

    def test_missing_or_invalid_provenance_fails_closed(self):
        m0 = world()["m"]
        for name, mutate in (
            ("no provenance", lambda d: d.pop("provenance")),
            ("no controlling documents", lambda d: d["provenance"].pop("controlling_documents")),
            ("controlling edited", lambda d: d["provenance"]["controlling_documents"].__setitem__("amendment_sha256", "0" * 64)),
            ("rubric hash edited", lambda d: d["provenance"]["controlling_documents"].__setitem__("rubric_sha256", "0" * 64)),
            ("no anchor", lambda d: d["provenance"].pop("anchor")),
            ("anchor not reproduced", lambda d: d["provenance"]["anchor"]["evidence"].__setitem__("reproduces_c0", False)),
            ("anchor other tokenizer", lambda d: d["provenance"]["anchor"]["evidence"].__setitem__("tokenizer_fingerprint", "0" * 64)),
            ("anchor hash", lambda d: d["provenance"]["anchor"].__setitem__("evidence_sha256", "0" * 64)),
            ("no invocation", lambda d: d["provenance"]["generation"].pop("invocation")),
            ("dirty code", lambda d: d["provenance"]["generation"].__setitem__("code_revision", {"git_commit": "a" * 40, "dirty": True})),
            ("unknown code", lambda d: d["provenance"]["generation"].__setitem__("code_revision", {"git_commit": "UNKNOWN", "dirty": "UNKNOWN"})),
            ("no code", lambda d: d["provenance"]["generation"].pop("code_revision")),
            ("revision", lambda d: d["provenance"]["generator"].__setitem__("protocol_revision", "1.0")),
            ("independence", lambda d: d["provenance"].__setitem__("independence_statement", "these are independent")),
            ("schema", lambda d: d.__setitem__("schema", "bleediff.controls/2")),
        ):
            d = copy.deepcopy(m0)
            try:
                mutate(d)
            except KeyError:
                pass
            self.assertTrue(verify(d), name)

    def test_generation_refuses_incomplete_or_inconsistent_inputs(self):
        w = world()
        kw = dict(c0_entry=w["rec"], c0_manifest_sha256="b" * 64, code_revision=REV)
        bad = {
            "no anchor": dict(controlling=w["ctl"], anchor=None),
            "failed anchor": dict(controlling=w["ctl"], anchor={**w["anchor"], "reproduces_c0": False}),
            "anchor other tokenizer": dict(controlling=w["ctl"], anchor={**w["anchor"], "tokenizer_fingerprint": "0" * 64}),
            "incomplete controlling": dict(controlling={k: v for k, v in w["ctl"].items() if k != "amendment_sha256"}, anchor=w["anchor"]),
            "wrong expected tokenizer": dict(controlling={**w["ctl"], "tokenizer_fingerprint_expected": "0" * 64}, anchor=w["anchor"]),
            "wrong rubric": dict(controlling={**w["ctl"], "rubric_sha256": "0" * 64}, anchor=w["anchor"]),
            "wrong C0": dict(controlling={**w["ctl"], "c0": {"control_id": "0" * 32, "fragment_sha256": "0" * 64}}, anchor=w["anchor"]),
        }
        for name, extra in bad.items():
            with self.assertRaises(C.ControlMatchError, msg=name) as cm:
                R.build_replicate_manifest(fx(), SID, OFFSET, qwenish, **kw, **extra)
            self.assertEqual(cm.exception.reason, "CONTROL_PROVENANCE_INVALID", name)

    def test_the_real_c0_cannot_be_reproduced_by_the_standin_so_the_anchor_gate_trips(self):
        real_c0 = C.load_manifest(SEM_MANIFEST)["controls"][SID]
        ev = R.run_anchor(scen(), OFFSET, qwenish, real_c0)
        self.assertFalse(ev["reproduces_c0"])                          # a different tokenizer identity: recorded, not hidden
        self.assertEqual(ev["schema"], R.ANCHOR_SCHEMA)
        self.assertEqual(ev["c0"]["control_id"], C0_ID)
        self.assertEqual(R.anchor_evidence_sha256(ev), R.anchor_evidence_sha256(copy.deepcopy(ev)))
        w = world()
        with self.assertRaises(C.ControlMatchError):
            R.build_replicate_manifest(fx(), SID, OFFSET, qwenish, c0_entry=w["rec"], c0_manifest_sha256="b" * 64,
                                       controlling=w["ctl"], anchor=ev, code_revision=REV)

    def test_the_synthetic_anchor_is_deterministic_and_reproduces(self):
        w = world()
        self.assertTrue(w["anchor"]["reproduces_c0"])
        self.assertEqual(R.run_anchor(w["sc"], OFFSET, qwenish, w["rec"]), w["anchor"])
        self.assertNotIn("time", " ".join(w["anchor"]).lower())

    def test_the_live_anchor_was_not_run_in_the_amendment(self):
        p = _json(EFFECTIVE)
        self.assertIs(p["anchor_policy"]["live_anchor_performed_in_amendment"], False)
        self.assertIs(p["anchor_policy"]["required_before_generation"], True)
        self.assertIs(p["inference"]["real_tokenizer_generation_performed"], False)
        self.assertEqual((p["inference"]["completion_requests_in_amendment"], p["inference"]["tokenize_requests_in_amendment"]), (0, 0))
        self.assertEqual(sorted(os.listdir(ROB_DIR)), ["AMENDMENT-001.md", "PREREGISTRATION.md", "protocol-1.1.json", "protocol.json"])


# --------------------------------------------------------------------------------------
def _copy_root(dst: str) -> None:
    for rel in ("fixtures",):
        shutil.copytree(os.path.join(ROOT, rel), os.path.join(dst, rel))
    os.makedirs(os.path.join(dst, "tests", "data"))
    for f in ("frozen_neutral_prose_f2299f0.json", "frozen_anchors_6a03565.json", "semantic_v2_approved_design.json"):
        shutil.copy(os.path.join(ROOT, "tests", "data", f), os.path.join(dst, "tests", "data", f))
    os.makedirs(os.path.join(dst, "bleediff"))
    for f in ("control.py", "fold.py", "arena.py"):
        shutil.copy(os.path.join(ROOT, "bleediff", f), os.path.join(dst, "bleediff", f))


class TestProtocolAndAmendmentIdentity(unittest.TestCase):
    def test_the_effective_protocol_verifies_and_the_original_is_refused(self):
        p = RB.load_robustness_protocol(EFFECTIVE)
        self.assertEqual(RB.verify_robustness_protocol(p, ROOT), [])
        original = RB.load_robustness_protocol(ORIGINAL_PROTOCOL)
        self.assertTrue(RB.verify_robustness_protocol(original, ROOT))       # superseded: cannot control generation
        self.assertEqual(RB.verify_original_artifacts(ROOT), [])

    def test_a_copy_of_the_repository_files_verifies(self):
        with tempfile.TemporaryDirectory() as d:
            _copy_root(d)
            self.assertEqual(RB.verify_robustness_protocol(RB.load_robustness_protocol(os.path.join(d, "fixtures", "config-audit-robustness", "protocol-1.1.json")), d), [])
            self.assertEqual(RB.load_controlling_documents(d)["effective_protocol_sha256"], _sha(EFFECTIVE))

    def test_changing_the_amendment_or_the_originals_is_detected(self):
        for rel, mark in (("AMENDMENT-001.md", "amendment"), ("PREREGISTRATION.md", "original"), ("protocol.json", "original")):
            with tempfile.TemporaryDirectory() as d:
                _copy_root(d)
                path = os.path.join(d, "fixtures", "config-audit-robustness", rel)
                with open(path, "ab") as fh:
                    fh.write(b"\n tampered \n")
                p = RB.load_robustness_protocol(os.path.join(d, "fixtures", "config-audit-robustness", "protocol-1.1.json"))
                probs = RB.verify_robustness_protocol(p, d)
                self.assertTrue([x for x in probs if mark in x], (rel, probs))
                with self.assertRaises(RB.RobustnessError):
                    RB.load_controlling_documents(d)                           # fails closed

    def test_a_missing_amendment_or_effective_protocol_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            _copy_root(d)
            os.remove(os.path.join(d, "fixtures", "config-audit-robustness", "AMENDMENT-001.md"))
            with self.assertRaises(RB.RobustnessError):
                RB.load_controlling_documents(d)
        with tempfile.TemporaryDirectory() as d:
            _copy_root(d)
            os.remove(os.path.join(d, "fixtures", "config-audit-robustness", "protocol-1.1.json"))
            with self.assertRaises(RB.RobustnessError):
                RB.load_controlling_documents(d)

    def test_tampering_with_the_effective_protocol_is_detected(self):
        p0 = RB.load_robustness_protocol(EFFECTIVE)
        for mutate in (
            lambda p: p["amendment"].__setitem__("sha256", "0" * 64),
            lambda p: p["supersedes"].__setitem__("original_commit", "0" * 40),
            lambda p: p["selection"]["preregistered_seeds"][0].__setitem__("seed_hex16", "0" * 16),
            lambda p: p["selection"].__setitem__("draw_budget_per_replicate", 4999),
            lambda p: p["selection"].__setitem__("draw_budget_overridable", True),
            lambda p: p["selection"].__setitem__("expected_tokenizer_fingerprint", "0" * 64),
            lambda p: p["selection"]["header_pool_facts"]["per_domain"]["festival"].__setitem__("surviving_rubric", 7),
            lambda p: p["review_policy"].__setitem__("sha256", "0" * 64),
            lambda p: p["review_policy"]["policy"].__setitem__("mode", "REPLACE"),
            lambda p: p["anchor_policy"].__setitem__("required_before_generation", False),
            lambda p: p["verification_policy"].__setitem__("regeneration", "structural"),
            lambda p: p["statistical_honesty"].pop(),
            lambda p: p["inference"].__setitem__("real_tokenizer_generation_performed", True),
            lambda p: p.__setitem__("revision", "1.0"),
        ):
            q = copy.deepcopy(p0)
            mutate(q)
            self.assertTrue(RB.verify_robustness_protocol(q, ROOT))

    def test_the_amendment_discloses_what_it_must(self):
        t = " ".join(_read(AMENDMENT).split())                        # whitespace-normalised: the markdown wraps lines
        for needle in ("58a00b1d149994eb3775570e85174fe4a3e2df21", ORIG_PREREG_SHA, ORIG_PROTOCOL_SHA, ORIG_RUBRIC_SHA,
                       "offline stand-in tokenizer", "inaccurate", "NOT preregistered before all forms of candidate generation",
                       "Seed derivation changed", "annotation-only", "exactly 5,000", "byte-identical", "No live anchor test",
                       "Wrong", "Overstated", "not automatically independent or exchangeable", "post hoc", "INCONCLUSIVE",
                       "does **not** demonstrate a real tenant-isolation", "C0's fixed **text**", "observed before this robustness study",
                       "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"):
            self.assertIn(needle, t, needle)
        self.assertNotIn("before any tokenizer-based candidate generation".upper(), t)

    def test_the_header_correction_is_verifiable(self):
        facts = R.header_pool_facts(scen(), OFFSET)
        self.assertEqual(facts["header_words_required_by_skeleton"], 3)
        per = facts["per_domain"]
        self.assertEqual({k: v["all_headers"] for k, v in per.items()},
                         {"bakery": 9, "trail": 9, "orchestra": 7, "greenhouse": 7, "seaside": 7, "library": 7, "festival": 7, "workshop": 7})
        self.assertEqual({k: v["eligible_for_skeleton"] for k, v in per.items()},
                         {"bakery": 4, "trail": 4, "orchestra": 3, "greenhouse": 3, "seaside": 3, "library": 3, "festival": 3, "workshop": 3})
        self.assertEqual({k: v["surviving_rubric"] for k, v in per.items()},
                         {"bakery": 4, "trail": 3, "orchestra": 2, "greenhouse": 2, "seaside": 2, "library": 2, "festival": 2, "workshop": 2})
        self.assertEqual(_json(EFFECTIVE)["selection"]["header_pool_facts"], facts)

    def test_seed_derivation_change_is_exactly_the_rubric_change(self):
        self.assertNotEqual(R.RUBRIC_SHA256, ORIG_RUBRIC_SHA)
        self.assertNotIn("manual_review", R.RUBRIC)
        self.assertNotIn("advances to the next candidate", json.dumps(R.RUBRIC))
        old_seeds = ["335bd1bc695658ca", "1de996316be8161c", "e6b732dbcc156f36", "e580341150bbe50b", "f1101ed338210a55",
                     "4e65c14f25aa1fec", "e4412a089ba858e2", "e50d936275fb9bdf", "aff29f76002a376c", "d6b0614f0de79092",
                     "75dba151872540fc", "7ede8cb831de4f02"]
        new = [s["seed_hex16"] for s in _json(EFFECTIVE)["selection"]["preregistered_seeds"]]
        self.assertEqual(len(set(new)), 12)
        self.assertFalse(set(new) & set(old_seeds))
        amendment = _read(AMENDMENT)
        self.assertIn(R.RUBRIC_SHA256, amendment)                       # the amendment states the true amended rubric hash
        self.assertTrue(all(sd in amendment for sd in new))             # ... and the true amended seeds
        self.assertTrue(all(sd in amendment for sd in old_seeds))       # ... and the superseded ones, for the record
        # the seed depends on the rubric hash and on nothing in the governance documents
        base = dict(scenario_id=SID, k=0, offset=OFFSET, unit="words", treatment_sha256="a" * 64, fingerprint="f" * 64)
        self.assertEqual(R.replicate_seed_material(**base)["rubric_sha256"], R.RUBRIC_SHA256)
        self.assertNotIn("amendment", json.dumps(R.replicate_seed_material(**base)))
        # the mechanical rules R1-R6 and the allocation are unchanged
        self.assertEqual([R.domain_for(k) for k in range(12)], EXPECTED_DOMAINS)
        self.assertEqual(R.JACCARD_MAX, 0.5)
        self.assertEqual(set(R.RUBRIC["rejection_codes"]), {"R1_TASK_WORD", "R2_TASK_STEM", "R3_FIELD_WORD", "R4_NUMERAL", "R5_DUPLICATE", "R6_OVERLAP"})
        self.assertEqual(R.RUBRIC["stopwords"], _json(ORIGINAL_PROTOCOL)["selection"]["rubric"]["stopwords"])
        self.assertEqual(R.RUBRIC["prohibited_field_words"], _json(ORIGINAL_PROTOCOL)["selection"]["rubric"]["prohibited_field_words"])


# --------------------------------------------------------------------------------------
def _review(m, annotations=None, **over):
    base = {"schema": R.REVIEW_SCHEMA, "selection_sha256": R.selection_sha256(m), "reviewer_id": "rev-01",
            "review_date": "2026-11-21", "annotations": annotations if annotations is not None else [],
            "verdict": "NO_ACTION", "verdict_reason": ""}
    base.update(over)
    return base


def _annotation(m, k=0, phrase=None, confound=True, severity="moderate"):
    text = next(r["control"]["fragment_text"] for r in m["selection"]["replicates"] if r["replicate"] == k)
    return {"replicate": k, "confound_present": confound, "concern_phrases": [phrase or text.split()[-2]] if confound else [],
            "severity": severity if confound else "none", "reason": "example reason", "review_date": "2026-11-21", "reviewer_id": "rev-01"}


def _obs(m, deltas=None):
    ks = [r["replicate"] for r in m["selection"]["replicates"] if r["status"] == "ACCEPTED"]
    deltas = deltas or [0.02, -0.05, 0.08, -0.11, 0.0, 0.13, -0.02, 0.05, -0.08, 0.10, 0.03, -0.04]
    return {"schema": RB.OBSERVATIONS_SCHEMA, "protocol_id": "x", "n_planned": 12,
            "treatment": {"delta_contrast": 0.857, "raw_argmax": "A", "baseline_argmax": "A"},
            "controls": [{"id": f"c{k}", "replicate": k, "domain": EXPECTED_DOMAINS[k], "status": "OK", "delta_contrast": deltas[i],
                          "raw_argmax": "A"} for i, k in enumerate(ks)]}


class TestHumanReviewIsAnnotationOnly(unittest.TestCase):
    def test_a_valid_review_never_changes_the_primary_set_or_the_manifest(self):
        m = copy.deepcopy(world()["m"])
        before = R.canonical(m)
        rev = _review(m, [_annotation(m, 0), _annotation(m, 3, confound=False)])
        self.assertEqual(R.validate_review(rev, m), [])
        fx_ = R.review_effects(rev, m)
        self.assertEqual(R.canonical(m), before)                                     # the manifest is untouched
        accepted = [r["replicate"] for r in m["selection"]["replicates"] if r["status"] == "ACCEPTED"]
        self.assertEqual(fx_["primary_set"], accepted)                                # the complete primary set, in order
        self.assertIs(fx_["primary_set_preserved"], True)
        self.assertEqual(fx_["sensitivity_S1_excluding_annotated"], [k for k in accepted if k != 0])
        self.assertIn("not the primary", fx_["sensitivity_label"])
        self.assertFalse(fx_["inference_blocked"])
        self.assertEqual(verify(m), [])                                               # and it still verifies

    def test_any_field_that_could_replace_regenerate_reorder_or_discard_is_rejected(self):
        m = copy.deepcopy(world()["m"])
        for key in ("replacement_text", "replace_with", "regenerate", "discard", "order", "reorder", "advance_stream", "new_control", "exclude_from_primary"):
            rev = _review(m, [_annotation(m, 0)])
            rev[key] = "x"
            self.assertTrue([p for p in R.validate_review(rev, m) if "forbidden" in p], key)
            rev2 = _review(m, [{**_annotation(m, 0), key: "x"}])
            self.assertTrue([p for p in R.validate_review(rev2, m) if "forbidden" in p], key)
            with self.assertRaises(C.ControlMatchError):
                R.review_effects(rev2, m)

    def test_annotations_are_validated(self):
        m = copy.deepcopy(world()["m"])
        ann = _annotation(m, 0)
        cases = {
            "unaccepted replicate": [{**ann, "replicate": 99}],
            "phrase not in text": [{**ann, "concern_phrases": ["zebra crossing"]}],
            "confound without phrase": [{**ann, "concern_phrases": []}],
            "bad severity": [{**ann, "severity": "catastrophic"}],
            "no confound with severity": [{**ann, "confound_present": False, "concern_phrases": [], "severity": "serious"}],
            "no reason": [{**ann, "reason": " "}],
            "bad date": [{**ann, "review_date": "21/11/2026"}],
            "identifying reviewer": [{**ann, "reviewer_id": "someone@example.com"}],
            "duplicate": [ann, ann],
            "not an object": ["x"],
        }
        for name, anns in cases.items():
            self.assertTrue(R.validate_review(_review(m, anns), m), name)
        self.assertTrue(R.validate_review(_review(m, [ann], selection_sha256="0" * 64), m))        # bound to another manifest
        self.assertTrue(R.validate_review(_review(m, [ann], reviewer_id="a b"), m))
        self.assertTrue(R.validate_review(_review(m, [ann], verdict="REPLACE"), m))
        self.assertTrue(R.validate_review(_review(m, [ann], verdict="NO_GO_FOR_INFERENCE", verdict_reason=""), m))

    def test_a_failed_replicate_cannot_be_annotated_into_existence(self):
        m = copy.deepcopy(world()["m"])
        rep = m["selection"]["replicates"][2]
        rep.pop("control")
        rep.update(status="FAILED", reason="X", note="n", draws_used=5000)
        rev = _review(m, [{**_annotation(world()["m"], 2)}])
        self.assertTrue([p for p in R.validate_review(rev, m) if "not an accepted candidate" in p])

    def test_a_no_go_verdict_blocks_inference_and_replaces_nothing(self):
        m = copy.deepcopy(world()["m"])
        rev = _review(m, [_annotation(m, 1, severity="serious")], verdict="NO_GO_FOR_INFERENCE", verdict_reason="answer-related contamination")
        fx_ = R.review_effects(rev, m)
        self.assertTrue(fx_["inference_blocked"])
        self.assertIn("new protocol version", fx_["inference_blocked_note"])
        self.assertEqual(fx_["primary_set"], [r["replicate"] for r in m["selection"]["replicates"] if r["status"] == "ACCEPTED"])
        self.assertEqual(R.canonical(m["selection"]), R.canonical(world()["m"]["selection"]))

    def test_the_review_code_cannot_reach_the_generator(self):
        for fn in (R.validate_review, R.review_effects):
            src = inspect.getsource(fn)
            for banned in ("generate_replicate", "search_stream", "build_replicate", "write_manifest", "random", "advance"):
                self.assertNotIn(banned, src.replace("advance a stream", ""), (fn.__name__, banned))
        self.assertEqual(R.REVIEW_POLICY["mode"], "ANNOTATION_ONLY")
        self.assertIn("replace a candidate", R.REVIEW_POLICY["may_not"])
        self.assertNotIn("advances to the next candidate", json.dumps(R.RUBRIC))

    def test_sensitivity_analysis_is_separate_labelled_and_never_replaces_the_primary(self):
        m = copy.deepcopy(world()["m"])
        obs = _obs(m)
        primary_alone = RB.analyze(obs)
        rev = _review(m, [_annotation(m, 0), _annotation(m, 5)])
        out = RB.analyze_with_review(obs, R.review_effects(rev, m))
        self.assertEqual(out["primary"], primary_alone)                                # identical to the unreviewed analysis
        self.assertIs(out["primary_set_preserved"], True)
        sens = out["sensitivity_S1_excluding_annotated"]
        self.assertEqual(sens["analysis_role"], "SENSITIVITY_S1_NOT_PRIMARY")
        self.assertEqual(sens["descriptive"]["n_valid"], 10)                           # actual n, 12 minus 2
        self.assertEqual(out["primary"]["descriptive"]["n_valid"], 12)
        rev3 = _review(m, [_annotation(m, 0), _annotation(m, 1), _annotation(m, 2)])
        out3 = RB.analyze_with_review(obs, R.review_effects(rev3, m))
        self.assertEqual(out3["sensitivity_S1_excluding_annotated"]["label"], "INSUFFICIENT")       # 9 < 10
        self.assertNotEqual(out3["primary"]["label"], "INSUFFICIENT")                                 # primary unaffected


# --------------------------------------------------------------------------------------
class TestFrozenAndHistoricalPreserved(unittest.TestCase):
    def test_original_preregistration_and_protocol_are_byte_identical(self):
        self.assertEqual(_sha(ORIGINAL_PREREG), ORIG_PREREG_SHA)
        self.assertEqual(_sha(ORIGINAL_PROTOCOL), ORIG_PROTOCOL_SHA)
        self.assertIn("BEFORE any tokenizer-based candidate generation", _read(ORIGINAL_PREREG))    # the mistake is preserved
        self.assertIn("nine header phrases per domain", _read(ORIGINAL_PREREG))
        self.assertEqual(_json(ORIGINAL_PROTOCOL)["selection"]["rubric_sha256"], ORIG_RUBRIC_SHA)

    def test_c0_and_the_frozen_generator_are_unchanged(self):
        e = C.load_manifest(SEM_MANIFEST)["controls"][SID]
        self.assertEqual(e["control_id"], C0_ID)
        self.assertEqual(e["fragment_text"], "Tenant 4 summer fair notes: Some cheerful neighbors admired many golden colorful "
                                             "ribbons beside the fountain on festival evenings. Each dancer carefully hung 42 garlands.")
        self.assertEqual(_sha(SEM_MANIFEST), "9688a738e46f912761d3b8fec86a42ce22e25980886e548285e9442901473fe1")
        pinned = _json(os.path.join(ROOT, "tests", "data", "frozen_neutral_prose_f2299f0.json"))
        text = _read(os.path.join(ROOT, "bleediff", "control.py"))
        got = {}
        for n in ast.parse(text).body:
            nm = getattr(n, "name", None)
            if nm is None and isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                nm = n.targets[0].id
            if nm in pinned["anchors"]:
                got[nm] = hashlib.sha256(ast.get_source_segment(text, n).replace("\r\n", "\n").encode()).hexdigest()
        self.assertEqual(got, pinned["anchors"])
        pins = _json(ORIGINAL_PROTOCOL)["frozen"]["source_sha256_newline_normalised"]          # recorded BEFORE the amendment
        for rel, want in pins.items():
            self.assertEqual(RB.source_sha256(os.path.join(ROOT, *rel.split("/"))), want, rel)
        self.assertEqual(C.LEXICON_SHA256, "f91e691f054f431d7ea772e99f90238d354a4b7dd991a7dcd333b2fcafd03ba5")
        self.assertEqual(C.ALGORITHM, "neutral-prose/1")
        self.assertEqual(fx().sha256, "b733148e71510a0b30ed5366e470befe72353d05b4a34408eaf45f3cec0d3ceb")

    def test_measurement_statistics_and_gates_are_unchanged(self):
        from bleediff.fold import TRACE_SCHEMA

        self.assertEqual((EXPOSURE_DELTA_NATS, CONTROL_BAND_NATS), (0.4, 0.1))
        self.assertEqual(len(TRACE_SCHEMA), 15)
        # pinned BEFORE the amendment (golden values of the analysis, computed by the Phase 1A code)
        deltas = [0.02, -0.05, 0.08, -0.11, 0.0, 0.13, -0.02, 0.05, -0.08, 0.10, 0.03, -0.04]
        r = RB.analyze(_obs(world()["m"], deltas))
        d, g = r["descriptive"], r["inferential_diagnostics"]
        self.assertEqual(r["label"], "SEPARATED_FROM_PLACEBO_EXPLORATORY")
        self.assertEqual((d["n_valid"], d["n_within_tolerance"]), (12, 10))
        self.assertAlmostEqual(d["signed_delta"]["mean"], 0.009166666666666668, places=15)
        self.assertAlmostEqual(d["signed_delta"]["median"], 0.01, places=15)
        self.assertAlmostEqual(d["signed_delta"]["sd"], 0.0732937220842938, places=13)
        self.assertAlmostEqual(d["signed_delta"]["range"], 0.24, places=15)
        self.assertAlmostEqual(d["fraction_within_tolerance_ci95_exact"][0], 0.5158622513140328, places=12)
        self.assertAlmostEqual(d["fraction_within_tolerance_ci95_exact"][1], 0.9791374745399075, places=12)
        self.assertAlmostEqual(g["rank_exceedance"]["p_hat_signed"], 0.07692307692307693, places=15)
        self.assertAlmostEqual(g["prediction_bound"]["t_quantile"], 1.7958848187040428, places=12)
        self.assertAlmostEqual(g["prediction_bound"]["bound_signed"], 0.1461684781008976, places=12)
        self.assertAlmostEqual(d["treatment_minus_placebo"]["minus_max"], 0.727, places=12)
        self.assertEqual(RB.ANALYSIS_ID, "config-audit-robustness-analysis/1")
        self.assertEqual(list(RB.LABELS), ["INSUFFICIENT", "TREATMENT_PRECONDITION_FAILED", "SEPARATED_FROM_PLACEBO_EXPLORATORY",
                                           "THRESHOLD_NOT_DISCRIMINATING", "NOT_SEPARATED"])
        o, e = _json(ORIGINAL_PROTOCOL)["analysis"], _json(EFFECTIVE)["analysis"]
        self.assertEqual({k: v for k, v in e.items() if k != "wording_only_change"}, o)             # definitions identical
        self.assertEqual(_json(ORIGINAL_PROTOCOL)["frozen"], _json(EFFECTIVE)["frozen"])
        # the amendment only ADDED wording to the output
        self.assertTrue(set(RB.STATISTICAL_HONESTY) <= set(r["interpretation_limits"]))
        self.assertIn("No label here is a PASS.", r["interpretation_limits"])

    def test_statistical_honesty_statements_are_present_in_every_output(self):
        text = " ".join(RB.analyze(_obs(world()["m"]))["interpretation_limits"])
        for needle in ("not automatically independent or exchangeable", "diagnostics, not calibrated significance claims",
                       "observed BEFORE this robustness study was designed", "post hoc", "five-scenario INCONCLUSIVE",
                       "does not demonstrate a real tenant-isolation vulnerability", "arithmetic transform"):
            self.assertIn(needle, text, needle)

    def test_all_historical_evidence_is_intact(self):
        for rel, sha in (("fixtures/contexts.json", "38ab93521643523dfd8fcb023b59055f1db21b1f22db16ed00f9a49fa538e250"),
                         ("fixtures/controls/neutral-prose-1.json", "7266bf7e687f1829b0fd18a26c484d7d8fe0c359b1e223fe909dcba8a50b5145"),
                         ("fixtures/semantic-v2/protocol.json", "df494e50f923585170b4bbb5f97d2b864b52b43b881dd51dcb4b4c8cd498c228"),
                         ("fixtures/semantic-v2/contexts.json", "b733148e71510a0b30ed5366e470befe72353d05b4a34408eaf45f3cec0d3ceb"),
                         ("tests/data/semantic_v2_approved_design.json", "57514b9a7567d792e940d01821a84ff204ab737d7f37c9446881ad0e68ec3f30")):
            self.assertEqual(_sha(os.path.join(ROOT, *rel.split("/"))), sha, rel)
        self.assertTrue(_sha(os.path.join(ROOT, "fixtures", "controls", "neutral-prose-1.window-repair-1.json")).startswith("b4b729f5"))
        runs = os.path.join(ROOT, "runs")                                              # git-ignored local evidence, when present
        if os.path.isdir(runs):
            for name in os.listdir(runs):
                if name.startswith("v0.1.4-semantic-v2-qwen1.5b-exploratory-"):
                    base = os.path.join(runs, name)
                    self.assertEqual(_sha(os.path.join(base, "traces-semantic-v2.jsonl"))[:16], "6bea7539da8a48eb")
                    self.assertEqual(_sha(os.path.join(base, "baseline-semantic-v2.json"))[:16], "3f122b462f8d3424")
            for rel, pre in (("traces-qwen1.5b.jsonl", "c600549e44500e7a"), ("baseline-qwen1.5b.json", "e2482f6b7870edda")):
                if os.path.exists(os.path.join(runs, rel)):
                    self.assertEqual(_sha(os.path.join(runs, rel))[:16], pre, rel)

    def test_new_files_are_standard_library_only_and_free_of_personal_data(self):
        allowed = set(sys.stdlib_module_names) | {"bleediff", "tests", "test_bleediff", "_tok"}
        for rel in ("bleediff/replicates.py", "bleediff/robustness.py", "tests/test_v016_amendment.py"):
            for node in ast.walk(ast.parse(_read(os.path.join(ROOT, *rel.split("/"))))):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mods = [node.module.split(".")[0]]
                self.assertFalse([m for m in mods if m not in allowed], (rel, mods))
        pat = re.compile(r"(@[A-Za-z0-9-]+\.[a-z]{2,}|[A-Za-z]:\\|/Users/|/home/|\.gguf|api[_-]?key\s*[:=]|password\s*[:=])", re.I)
        for path in (AMENDMENT, EFFECTIVE):
            self.assertIsNone(pat.search(_read(path)), path)
        self.assertEqual(__version__, "0.1.6")
        for n in ast.walk(ast.parse(_read(os.path.join(ROOT, "bleediff", "replicates.py")))):
            if isinstance(n, ast.Attribute):
                self.assertNotEqual(n.attr, "completion")


if __name__ == "__main__":
    unittest.main()
