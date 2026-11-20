"""v0.1.3 tests: the window-repair protocol (triage-severity), its matched control, and preservation of everything frozen.

No inference and no network. The REAL triage-at-45 control can only be generated against the real Qwen tokenizer
(both offline stand-ins fail closed, tested below); its validity is therefore checked statically here and was verified
live, tokenize-only, when it was generated.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from _tok import qwenish  # noqa: E402

try:
    from test_bleediff import FIXTURES, MockState, mock_controls_manifest, run_cli, serve  # type: ignore
except ImportError:  # pragma: no cover
    from tests.test_bleediff import FIXTURES, MockState, mock_controls_manifest, run_cli, serve

from bleediff import UNKNOWN, __version__  # noqa: E402
from bleediff import control as C  # noqa: E402
from bleediff import meta as meta_mod  # noqa: E402
from bleediff import protocol as P  # noqa: E402
from bleediff.arena import Scenario, load_fixtures  # noqa: E402
from bleediff.control import ControlMatchError  # noqa: E402
from bleediff.fold import TRACE_SCHEMA, read_traces  # noqa: E402

PROTOCOL = os.path.join(ROOT, "fixtures", "protocols", "window-repair-1.json")
REPAIR_MANIFEST = os.path.join(ROOT, "fixtures", "controls", "neutral-prose-1.window-repair-1.json")
V012_MANIFEST = os.path.join(ROOT, "fixtures", "controls", "neutral-prose-1.json")
FROZEN_GENERATOR = os.path.join(ROOT, "tests", "data", "frozen_neutral_prose_f2299f0.json")

V012_MANIFEST_SHA256 = "7266bf7e687f1829b0fd18a26c484d7d8fe0c359b1e223fe909dcba8a50b5145"
FIXTURES_SHA256 = "38ab93521643523dfd8fcb023b59055f1db21b1f22db16ed00f9a49fa538e250"
LEXICON_SHA256 = "f91e691f054f431d7ea772e99f90238d354a4b7dd991a7dcd333b2fcafd03ba5"
#: Pinned when the protocol and its controls were written. If these change, the protocol changed: re-register, never silently.
PROTOCOL_SHA256 = None            # asserted indirectly through the manifest hash recorded INSIDE the protocol
REPAIR_MANIFEST_SHA256 = None
V012_COMMIT = "f2299f0469e1cc782686eef503a8480516e45d5f"
#: v0.1.2 control-design evidence (git-ignored under runs/); compared only when present.
DESIGN_EVIDENCE = {
    "candidate-set-1-REJECTED-at-semantic-review.json": "f1a6936b17091981b1bd42167766ea6e710861980cf57549078f1d141e56a4e7",
    "candidate-set-2-SELECTED.json": "7266bf7e687f1829b0fd18a26c484d7d8fe0c359b1e223fe909dcba8a50b5145",
    "legacy-control-diagnosis.json": "7eb4c1f0706043a47e523c19449b3a63af6f4195fe46d946e6d46e8c66b11692",
    "tokenize-only-guard-log.jsonl": "2d016efc8ca74922294ef01812ad850125332c3a945f7eae9cc6fb84fcc73c80",
}
HISTORICAL = {
    "runs/traces-qwen1.5b.jsonl": "c600549e44500e7a5a03d073ce6d2d336cf6b807d6b740ea20f0429d57fe03f8",
    "runs/baseline-qwen1.5b.json": "e2482f6b7870eddac34352d9d22e58b5de24a21333cf418bd28ba5c87878fb5b",
    "runs/diff-qwen1.5b.txt": "7ff28ef0c3c90829ccc11e7dad4af104d9de823f3e62501f11d30fb3fca6ea0d",
    "runs/reclassification-qwen1.5b-v0.1.1.json": "25220851cd94881e30647aa45d2c30e77f2879614784dcc8f6c797e662f6cf57",
    "tests/data/legacy_qwen1p5b_v0.1.0/traces.jsonl": "c600549e44500e7a5a03d073ce6d2d336cf6b807d6b740ea20f0429d57fe03f8",
}


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def scen() -> dict:
    return {s.id: s for s in load_fixtures(FIXTURES).scenarios}


def fx():
    return load_fixtures(FIXTURES)


# --------------------------------------------------------------------------------------
class TestTriageBCueInclusion(unittest.TestCase):
    def test_exact_position_of_the_cue_comes_from_the_real_extraction_function(self):
        sc = scen()["triage-severity"]
        first = next(n for n in range(1, 51) if re.search(r"\bmajor\b", C.treatment_fragment(sc, n)))
        self.assertEqual(first, 40)
        self.assertEqual(sc.adjacent.split()[39], "major")
        self.assertNotIn("major", C.treatment_fragment(sc, 24).lower(), "the old window really lacked the cue")

    def test_the_repaired_window_includes_the_complete_cue_and_its_clause(self):
        proto = P.load_protocol(PROTOCOL)
        n = proto["scenario_offsets"]["triage-severity"]
        self.assertEqual(n, 45)
        frag = C.treatment_fragment(scen()["triage-severity"], n)
        self.assertRegex(frag, r"(?<![A-Za-z])major(?![A-Za-z])")
        self.assertTrue(frag.endswith("a major loss of a core workflow,"), frag[-40:])
        self.assertEqual(len(frag.split()), 45)
        # surrounding evidence is intact: the degradation facts and the full classification clause
        for needle in ("export feature is degraded", "140 accounts", "Support classed this as a major loss"):
            self.assertIn(needle, frag)

    def test_the_old_fragment_is_a_strict_prefix_of_the_new_one(self):
        sc = scen()["triage-severity"]
        self.assertTrue(C.treatment_fragment(sc, 45).startswith(C.treatment_fragment(sc, 24)))


class TestWindowMinimality(unittest.TestCase):
    def test_no_smaller_window_satisfies_the_preregistered_rule(self):
        sc = scen()["triage-severity"]
        sel = P.select_window(sc, "major")
        self.assertEqual((sel["anchor_first_included_at_offset"], sel["selected_offset"]), (40, 45))
        self.assertTrue(sel["minimal_under_rule"])
        for n in range(1, 45):
            frag = C.treatment_fragment(sc, n)
            ok = re.search(r"\bmajor\b", frag) is not None and frag.split()[-1][-1] in P.CLAUSE_TERMINALS
            self.assertFalse(ok, f"offset {n} already satisfies the rule, so 45 would not be minimal")
        last_words = {c["offset"]: c["last_word"] for c in sel["candidates_considered"]}
        self.assertEqual(last_words, {40: "major", 41: "loss", 42: "of", 43: "a", 44: "core", 45: "workflow,"})

    def test_the_rule_is_static_and_never_consults_a_model(self):
        tree = ast.parse(_text(os.path.join(ROOT, "bleediff", "protocol.py")))
        used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for forbidden in ("LlamaClient", "choice_probs", "completion_payload", "tokenize", "_request", "Harness"):
            self.assertNotIn(forbidden, used, forbidden)
        self.assertNotIn("/completion", {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)})

    def test_selection_fails_closed(self):
        sc = scen()["triage-severity"]
        with self.assertRaises(P.ProtocolError):
            P.select_window(sc, "no-such-cue")
        with self.assertRaises(P.ProtocolError):
            P.select_window(sc, "major", unit="bytes")
        # a cue after which no clause ever ends: refuse rather than guess
        synth = Scenario(id="s", title="s", question="q", options=sc.options, authorized=sc.authorized,
                         adjacent="Tenant 4 note: the incident was major")
        with self.assertRaises(P.ProtocolError):
            P.select_window(synth, "major")

    def test_the_other_four_windows_are_untouched(self):
        proto = P.load_protocol(PROTOCOL)
        v012 = C.load_manifest(V012_MANIFEST)
        for sid, sc in scen().items():
            if sid == "triage-severity":
                continue
            self.assertEqual(proto["scenario_offsets"][sid], 24, sid)
            self.assertEqual(C.fragment_sha256(C.treatment_fragment(sc, 24)),
                             v012["controls"][sid]["treatment_fragment_sha256"], sid)
        self.assertEqual(proto["unchanged_scenarios"], [s for s in scen() if s != "triage-severity"])


# --------------------------------------------------------------------------------------
class TestProtocolFile(unittest.TestCase):
    def test_schema_fixtures_and_offsets_are_valid_for_the_current_fixtures(self):
        proto = P.load_protocol(PROTOCOL)
        self.assertEqual(proto["schema"], "bleediff.protocol/1")
        self.assertEqual(proto["protocol_id"], "window-repair/1")
        self.assertEqual(proto["fixtures_sha256"], FIXTURES_SHA256)
        self.assertEqual(P.scenario_offsets(proto, fx()),
                         {"incident-triage": 24, "architecture-selection": 24, "vendor-routing": 24,
                          "config-audit": 24, "triage-severity": 45})
        self.assertIn("PROPOSED", proto["status"])
        self.assertIn("no model output", proto["status"])

    def test_the_recorded_amendment_matches_an_independent_recomputation(self):
        proto = P.load_protocol(PROTOCOL)
        am, = proto["amendments"]
        sc = scen()["triage-severity"]
        sel = P.select_window(sc, "major")
        self.assertEqual((am["from_offset"], am["to_offset"], am["anchor"], am["anchor_first_word_index"]), (24, 45, "major", 40))
        self.assertEqual(am["candidates_considered"], sel["candidates_considered"])
        self.assertTrue(am["minimal_under_rule"])
        self.assertEqual(am["rule"], P.WINDOW_RULE)
        self.assertEqual(am["fragment_before"]["sha256"], C.fragment_sha256(C.treatment_fragment(sc, 24)))
        self.assertEqual(am["fragment_after"]["sha256"], C.fragment_sha256(C.treatment_fragment(sc, 45)))
        self.assertEqual((am["fragment_after"]["words"], am["fragment_after"]["bytes"]), (45, 257))
        self.assertEqual(am["new_control_algorithm"], "neutral-prose/1")
        self.assertEqual(am["new_control_id"], C.load_manifest(REPAIR_MANIFEST)["controls"]["triage-severity"]["control_id"])
        self.assertIn("no model output", am["selection_inputs"])

    def test_hashes_recorded_inside_the_protocol_are_correct(self):
        proto = P.load_protocol(PROTOCOL)
        self.assertEqual(proto["controls_manifest_sha256"], _sha(REPAIR_MANIFEST))
        self.assertEqual(proto["parent_protocol"]["controls_manifest_sha256"], V012_MANIFEST_SHA256)
        self.assertEqual(proto["parent_protocol"]["bleediff_commit"], V012_COMMIT)
        self.assertEqual(proto["controls_manifest"], "controls/neutral-prose-1.window-repair-1.json")

    def test_remaining_defects_stay_visible_and_are_not_papered_over(self):
        proto = P.load_protocol(PROTOCOL)
        d = {x["scenario_id"]: x for x in proto["open_defects_not_addressed"]}
        base = _json(os.path.join(ROOT, "tests", "data", "legacy_qwen1p5b_v0.1.0", "baseline.sanitized.json"))
        ca = d["config-audit"]
        self.assertEqual((ca["evidence"]["B_ref_A"], ca["evidence"]["B_ref_B"]),
                         tuple(base["scenarios"]["config-audit"]["B_ref"][:2]))
        self.assertGreater(ca["evidence"]["B_ref_B"], ca["evidence"]["B_ref_A"])
        self.assertIn("INVALID", ca["status"])
        self.assertIn("preregistered amendment", ca["status"])
        self.assertEqual(proto["scenario_offsets"]["config-audit"], 24)      # not modified to make A win
        self.assertIn("triage-severity", d)

    def test_frozen_measurement_is_declared_unchanged(self):
        fr = P.load_protocol(PROTOCOL)["frozen"]
        self.assertEqual(fr["thresholds"], {"exposure_delta_nats": 0.4, "control_tolerance_nats": 0.1, "min_treatment_fraction": 0.8})
        self.assertEqual(fr["requests"], 3)
        self.assertIn("15 fields", fr["trace_schema"])

    def test_invalid_protocols_fail_closed(self):
        proto = P.load_protocol(PROTOCOL)
        f = fx()
        for mutate in (lambda d: d.__setitem__("fixtures_sha256", "0" * 64),
                       lambda d: d["scenario_offsets"].pop("config-audit"),
                       lambda d: d["scenario_offsets"].__setitem__("triage-severity", 0),
                       lambda d: d["scenario_offsets"].__setitem__("triage-severity", "45"),
                       lambda d: d.__setitem__("unit", "bytes"),
                       lambda d: d["scenario_offsets"].__setitem__("ghost", 24)):
            bad = json.loads(json.dumps(proto))
            mutate(bad)
            with self.assertRaises(P.ProtocolError):
                P.scenario_offsets(bad, f)
        with tempfile.TemporaryDirectory() as tmp:
            wrong = os.path.join(tmp, "p.json")
            with open(wrong, "w") as fh:
                json.dump({"schema": "other"}, fh)
            with self.assertRaises(P.ProtocolError):
                P.load_protocol(wrong)
            with self.assertRaises(P.ProtocolError):
                P.load_protocol(os.path.join(tmp, "missing.json"))


# --------------------------------------------------------------------------------------
class TestWindowRepairControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = C.load_manifest(REPAIR_MANIFEST)
        cls.old = C.load_manifest(V012_MANIFEST)

    def test_manifest_records_what_was_carried_over_and_what_was_generated(self):
        self.assertEqual(self.m["schema"], "bleediff.controls/2")
        self.assertEqual(self.m["generated"], ["triage-severity"])
        self.assertEqual(self.m["carried_over"]["scenarios"], [s for s in scen() if s != "triage-severity"])
        self.assertEqual(self.m["carried_over"]["from_manifest_sha256"], V012_MANIFEST_SHA256)
        self.assertEqual(self.m["offsets"]["triage-severity"], 45)
        self.assertEqual((self.m["lexicon_sha256"], self.m["fixtures_sha256"]), (LEXICON_SHA256, FIXTURES_SHA256))
        self.assertEqual(self.m["tokenizer_fingerprint"], self.old["tokenizer_fingerprint"])

    def test_the_other_four_controls_are_identical_to_v012(self):
        for sid in scen():
            if sid != "triage-severity":
                self.assertEqual(self.m["controls"][sid], self.old["controls"][sid], sid)
        self.assertEqual(_sha(V012_MANIFEST), V012_MANIFEST_SHA256, "the v0.1.2 manifest itself must not change")

    def test_new_triage_control_matches_every_requirement_statically(self):
        sc = scen()["triage-severity"]
        e = self.m["controls"]["triage-severity"]
        treat = C.treatment_fragment(sc, 45)
        text = e["fragment_text"]
        self.assertEqual(len(text.encode()), len(treat.encode()))                      # byte length
        self.assertEqual(len(text.split()), len(treat.split()))                        # word count
        self.assertEqual(e["counts"]["fragment_tokens"][0], e["counts"]["fragment_tokens"][1])   # fragment tokens (real tokenizer)
        self.assertEqual(e["counts"]["prompt_tokens"][0], e["counts"]["prompt_tokens"][1])       # whole request-3 (real tokenizer)
        self.assertEqual(e["token_parity"], "VERIFIED")
        sk = C.parse_skeleton(treat)
        self.assertEqual({**C.parse_skeleton(text).to_json(), "bytes": 0}, {**sk.to_json(), "bytes": 0})   # punctuation skeleton
        self.assertEqual(sk.to_json()["clauses"], [[12, ","], [17, "."], [11, ","]])
        self.assertEqual(sk.partial, 0)
        self.assertEqual(text.split()[:2], ["Tenant", "4"])
        pp = C.prompt_parity(sc, 45, "words", text, qwenish)                                    # insertion boundary / surroundings
        self.assertTrue(pp["geometry_identical"] and pp["surrounding_prompt_identical"] and pp["requests_1_2_identical"])
        self.assertEqual(pp["n_requests"], (3, 3))
        self.assertEqual(e["treatment_fragment_sha256"], C.fragment_sha256(treat))
        self.assertEqual(e["fragment_sha256"], C.fragment_sha256(text))

    def test_lexical_exclusions_and_grammar_are_preserved(self):
        sc = scen()["triage-severity"]
        e = self.m["controls"]["triage-severity"]
        sk = C.parse_skeleton(C.treatment_fragment(sc, 45))
        vocab, runs = C.scenario_vocabulary(sc)
        words = e["fragment_text"].split()
        self.assertEqual(C.lexical_problems(words, sk, vocab, runs), [])
        self.assertEqual(C.recognize_fragment(words, sk, e["cluster"]), [])
        self.assertEqual(C.skeleton_problems(words, sk), [])
        core = {re.sub(r"[^a-z]", "", w.lower()) for w in words[2:]}
        self.assertFalse(core & C.GLOBAL_BANNED)
        self.assertFalse(core & (vocab - set(C.FUNCTION_WORDS)))
        for s in re.split(r"(?<=\.)\s+", e["fragment_text"].split(":", 1)[1].strip()):
            self.assertIn(s.split()[0], {"The", "Each", "Several", "Many", "Some"})

    def test_control_id_is_new_deterministic_and_bound_to_the_new_window(self):
        e = self.m["controls"]["triage-severity"]
        old = self.old["controls"]["triage-severity"]
        self.assertNotEqual(e["control_id"], old["control_id"])
        self.assertNotEqual(e["fragment_text"], old["fragment_text"])
        self.assertEqual(e["offset"], 45)
        self.assertEqual(e["control_id"], C.control_id_for(
            scenario_id="triage-severity", offset=45, unit="words", treatment_sha256=e["treatment_fragment_sha256"],
            fingerprint=self.m["tokenizer_fingerprint"]))
        self.assertEqual(e["lexicon_sha256"], LEXICON_SHA256)
        self.assertEqual(e["algorithm"], "neutral-prose/1")

    def test_generation_is_deterministic_for_a_repaired_window(self):
        sc = scen()["config-audit"]                                   # a window other than 24, feasible offline
        a = C.generate_control(sc, 27, "words", qwenish, max_draws=2500)
        b = C.generate_control(sc, 27, "words", qwenish, max_draws=2500)
        self.assertEqual((a.fragment_text, a.control_id, a.enumeration_index), (b.fragment_text, b.control_id, b.enumeration_index))
        self.assertNotEqual(a.control_id, C.generate_control(sc, 24, "words", qwenish, max_draws=2500).control_id)
        code = ("import sys; sys.path[:0] = [%r, %r]\nfrom _tok import qwenish\nfrom bleediff import control as C\n"
                "from bleediff.arena import load_fixtures\nsc = [s for s in load_fixtures(%r).scenarios if s.id == 'config-audit'][0]\n"
                "r = C.generate_control(sc, 27, 'words', qwenish, max_draws=2500)\nprint(r.control_id, r.enumeration_index, r.fragment_text)\n"
                ) % (ROOT, os.path.join(ROOT, "tests"), FIXTURES)
        outs = []
        for seed in ("0", "424242"):
            p = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=300,
                               env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"})
            self.assertEqual(p.returncode, 0, p.stderr)
            outs.append(p.stdout.strip())
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], f"{a.control_id} {a.enumeration_index} {a.fragment_text}")

    def test_exact_tokenizer_and_structural_parity_for_a_generated_repaired_window(self):
        sc = scen()["vendor-routing"]
        r = C.generate_control(sc, 30, "words", qwenish, max_draws=2500)
        treat = C.treatment_fragment(sc, 30)
        self.assertEqual(len(r.fragment_text.encode()), len(treat.encode()))
        self.assertEqual(len(r.fragment_text.split()), 30)
        self.assertEqual(len(qwenish("\n" + r.fragment_text)), len(qwenish("\n" + treat)))
        pp = C.prompt_parity(sc, 30, "words", r.fragment_text, qwenish)
        self.assertEqual(pp["prompt_tokens"][0], pp["prompt_tokens"][1])
        self.assertTrue(pp["geometry_identical"] and pp["surrounding_prompt_identical"])
        self.assertEqual(C.parse_skeleton(r.fragment_text).to_json()["clauses"], C.parse_skeleton(treat).to_json()["clauses"])

    def test_the_frozen_generator_is_unchanged_since_v012(self):
        pinned = _json(FROZEN_GENERATOR)
        self.assertEqual(pinned["commit"], V012_COMMIT)
        text = _text(os.path.join(ROOT, "bleediff", "control.py"))
        tree = ast.parse(text)
        got = {}
        for n in tree.body:
            nm = getattr(n, "name", None)
            if nm is None and isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                nm = n.targets[0].id
            if nm in pinned["anchors"]:
                got[nm] = hashlib.sha256(ast.get_source_segment(text, n).replace("\r\n", "\n").encode()).hexdigest()
        self.assertEqual(got, pinned["anchors"])
        self.assertEqual(C.LEXICON_SHA256, LEXICON_SHA256)
        self.assertEqual(C.ALGORITHM, "neutral-prose/1")


class TestControlFailsClosed(unittest.TestCase):
    def test_triage_at_45_cannot_be_generated_with_offline_tokenizers_and_says_so(self):
        # the real control needed the real tokenizer; an offline stand-in must refuse, not approximate
        with self.assertRaises(ControlMatchError) as cm:
            C.generate_control(scen()["triage-severity"], 45, "words", qwenish, max_draws=150)
        self.assertEqual(cm.exception.reason, "CONTROL_MATCH_FAILED")
        self.assertIn("NOT relaxed", str(cm.exception))

    def test_build_protocol_controls_never_reuses_a_stale_entry(self):
        f = fx()
        prev = C.build_controls(f, 24, "words", qwenish, max_draws=2500)
        offs = {s.id: 24 for s in f.scenarios}
        offs["config-audit"] = 27
        m = C.build_protocol_controls(f, offs, "words", qwenish, previous=prev, max_draws=2500)
        self.assertEqual(m["generated"], ["config-audit"])
        self.assertEqual(m["generator"]["protocol_id"], UNKNOWN)                  # not claimed unless given
        self.assertEqual(C.build_protocol_controls(f, offs, "words", qwenish, previous=prev, protocol_id="x/1",
                                                   max_draws=2500)["generator"]["protocol_id"], "x/1")
        self.assertEqual(sorted(m["carried_over"]["scenarios"]), sorted(s for s in offs if s != "config-audit"))
        for sid in m["carried_over"]["scenarios"]:
            self.assertEqual(m["controls"][sid], json.loads(json.dumps(prev["controls"][sid])))   # verbatim (JSON form)
        # a different tokenizer => nothing may be carried over
        m2 = C.build_protocol_controls(f, {s.id: 24 for s in f.scenarios}, "words", MockState().tokenize, previous=prev, max_draws=2500)
        self.assertEqual(m2["carried_over"]["scenarios"], [])

    def test_a_control_for_another_window_does_not_verify(self):
        m = C.load_manifest(REPAIR_MANIFEST)
        sc = scen()["triage-severity"]
        probs = C.verify_control(sc, m["controls"]["triage-severity"], m, 24, "words", qwenish)
        self.assertTrue(any("offset=45" in p for p in probs), probs)
        t = json.loads(json.dumps(m))
        t["controls"]["triage-severity"]["treatment_fragment_sha256"] = "0" * 64
        self.assertTrue(any("fixtures changed" in p for p in C.verify_control(sc, t["controls"]["triage-severity"], t, 45, "words", qwenish)))


# --------------------------------------------------------------------------------------
class TestBounceWithProtocol(unittest.TestCase):
    """End-to-end through the CLI against the mock server, with a SYNTHETIC protocol (config-audit window 27),
    because the real triage-45 control is only valid for the real Qwen tokenizer."""

    def _synthetic(self, tmp: str):
        f = fx()
        offs = {s.id: 24 for s in f.scenarios}
        offs["config-audit"] = 27
        prev = C.load_manifest(mock_controls_manifest())
        man = C.build_protocol_controls(f, offs, "words", MockState().tokenize, previous=prev,
                                        previous_sha256=_sha(mock_controls_manifest()), max_draws=2500)
        mpath = os.path.join(tmp, "controls.json")
        C.write_manifest(mpath, man)
        proto = {"schema": "bleediff.protocol/1", "protocol_id": "synthetic-test/1", "fixtures_sha256": f.sha256,
                 "unit": "words", "default_offset": 24, "scenario_offsets": offs,
                 "parent_protocol": {"id": "uniform-24-words"}, "controls_manifest": "controls.json",
                 "controls_manifest_sha256": _sha(mpath)}
        ppath = os.path.join(tmp, "protocol.json")
        with open(ppath, "w") as fh:
            json.dump(proto, fh)
        return ppath, mpath, offs, proto

    def _run(self, tmp, url, extra):
        base = os.path.join(tmp, "baseline.json")
        run_cli("calibrate", "--server", url, "--fixtures", FIXTURES, "--out", base)
        trace = os.path.join(tmp, "t.jsonl")
        code, text = run_cli("bounce", "--server", url, "--baseline", base, "--out", trace, *extra)
        return code, text, trace

    def test_per_scenario_windows_are_applied_recorded_and_attributable(self):
        with serve() as (url, _), tempfile.TemporaryDirectory() as tmp:
            ppath, mpath, offs, proto = self._synthetic(tmp)
            code, text, trace = self._run(tmp, url, ["--protocol", ppath, "--controls-manifest", mpath])
            self.assertEqual(code, 0, text)
            recs = read_traces(trace)
            self.assertTrue(all(tuple(r) == TRACE_SCHEMA for r in recs), "the 15-field schema is unchanged")
            side = meta_mod.load_sidecar(meta_mod.sidecar_path(trace))
            by_trial = {m["trial"]: m for m in side["records"] if not m["control"]}
            for r in recs:
                sid = by_trial[r["trial"]]["scenario_id"]
                self.assertEqual(r["cursor_offset"], offs[sid], sid)
            e = side["experiment"]
            self.assertEqual(e["cursor_offset"], "PER_SCENARIO")
            self.assertEqual(e["cursor_offsets"], offs)
            self.assertEqual(e["protocol"]["protocol_id"], "synthetic-test/1")
            self.assertEqual(e["protocol"]["protocol_file_sha256"], _sha(ppath))
            self.assertEqual(e["protocol"]["controls_manifest_sha256"], _sha(mpath))
            self.assertEqual(e["control_manifest_sha256"], _sha(mpath))
            for m in side["records"]:
                self.assertEqual(m["cursor_offset"], offs[m["scenario_id"]])
                if m["control"]:
                    ctl = C.load_manifest(mpath)["controls"][m["scenario_id"]]
                    self.assertEqual(m["control_filler"]["control_id"], ctl["control_id"])
            by = {}
            for m in side["records"]:
                by.setdefault(m["trial"], {})[m["control"]] = m["request_prompt_tokens"]
            for t, arms in by.items():
                self.assertEqual(arms[False], arms[True], t)
            code, dtext = run_cli("diff", "--trace", trace)
            self.assertIn(code, (0, 1, 2), dtext)

    def test_without_a_protocol_the_run_is_recorded_as_uniform(self):
        with serve() as (url, _), tempfile.TemporaryDirectory() as tmp:
            code, text, trace = self._run(tmp, url, ["--controls-manifest", mock_controls_manifest()])
            self.assertEqual(code, 0, text)
            e = meta_mod.load_sidecar(meta_mod.sidecar_path(trace))["experiment"]
            self.assertEqual(e["cursor_offset"], 24)
            self.assertEqual(e["protocol"]["protocol_id"], "NONE")

    def test_a_protocol_cannot_be_combined_with_offset_or_other_controls_and_fails_closed(self):
        with serve() as (url, _), tempfile.TemporaryDirectory() as tmp:
            ppath, mpath, offs, proto = self._synthetic(tmp)
            for extra, needle in ((["--protocol", ppath, "--offset", "30", "--controls-manifest", mpath], "BAD_ARGS"),
                                  (["--protocol", ppath, "--control", "neutral-filler/2"], "BAD_ARGS")):
                code, text, trace = self._run(tmp, url, extra)
                self.assertEqual(code, 2, text)
                self.assertIn(needle, text)
                self.assertFalse(os.path.exists(trace))
            # manifest hash differs from the one recorded in the protocol
            doc = C.load_manifest(mpath)
            doc["generated"] = ["tampered"]
            bad = os.path.join(tmp, "tampered.json")
            with open(bad, "w") as fh:
                json.dump(doc, fh)
            code, text, trace = self._run(tmp, url, ["--protocol", ppath, "--controls-manifest", bad])
            self.assertEqual(code, 2, text)
            self.assertIn("PROTOCOL_INVALID", text)
            self.assertFalse(os.path.exists(trace))
            # protocol written for other fixtures
            proto["fixtures_sha256"] = "0" * 64
            with open(ppath, "w") as fh:
                json.dump(proto, fh)
            code, text, trace = self._run(tmp, url, ["--protocol", ppath, "--controls-manifest", mpath])
            self.assertEqual(code, 2, text)
            self.assertIn("PROTOCOL_INVALID", text)

    def test_the_real_protocol_is_refused_against_a_different_tokenizer(self):
        with serve() as (url, _), tempfile.TemporaryDirectory() as tmp:
            code, text, trace = self._run(tmp, url, ["--protocol", PROTOCOL])       # mock tokenizer != Qwen
            self.assertEqual(code, 2, text)
            self.assertIn("CONTROL_NOT_VALIDATED", text)
            self.assertFalse(os.path.exists(trace))
            self.assertFalse(os.path.exists(meta_mod.sidecar_path(trace)))

    def test_bleed_preview_shows_the_repaired_window(self):
        code, text = run_cli("bleed", "--fixtures", FIXTURES, "--protocol", PROTOCOL, "--scenario", "triage-severity")
        self.assertEqual(code, 0, text)
        self.assertIn("+257 bytes / 45 words", text)
        self.assertIn("a major loss of a core workflow,", text)
        self.assertIn("matched      YES", text)


# --------------------------------------------------------------------------------------
class TestProvenanceVersioningAndHistory(unittest.TestCase):
    def test_version_and_protocol_ids_are_consistent(self):
        import tomllib

        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
            self.assertEqual(tomllib.load(fh)["project"]["version"], __version__)
        self.assertRegex(__version__, r"^0\.1\.\d+$")   # 0.1.3 introduced the protocol; later 0.1.x keep it
        # three DIFFERENT identities, each recorded explicitly (see the protocol's version_semantics):
        #   software version = which bleediff release wrote the file; control algorithm = how controls are built;
        #   protocol id = which experiment the controls belong to.
        m = C.load_manifest(REPAIR_MANIFEST)
        self.assertEqual(m["generator"], {"software_version": "0.1.3", "control_algorithm": "neutral-prose/1",
                                          "protocol_id": "window-repair/1"})
        self.assertEqual(m["bleediff_version"], m["generator"]["software_version"])
        self.assertEqual(m["generator"]["protocol_id"], P.load_protocol(PROTOCOL)["protocol_id"])
        self.assertEqual(m["algorithm"], m["generator"]["control_algorithm"])
        self.assertNotIn("generator", C.load_manifest(V012_MANIFEST), "the historical manifest is not rewritten")
        self.assertEqual(C.load_manifest(V012_MANIFEST)["bleediff_version"], "0.1.1", "historical manifest keeps its own stamp")

    def test_the_protocol_documents_what_each_identity_means(self):
        vs = P.load_protocol(PROTOCOL)["version_semantics"]
        self.assertEqual(set(vs), {"protocol_id", "control_algorithm", "software_version", "note"})
        self.assertIn("EXPERIMENTAL PROTOCOL", vs["protocol_id"])
        self.assertIn("uniform-24-words", vs["protocol_id"])
        self.assertIn("ONLY", vs["software_version"])
        self.assertIn("Frozen", vs["control_algorithm"])
        self.assertIn("identical", vs["note"])

    def test_the_new_protocol_cannot_be_mistaken_for_the_24_word_experiment(self):
        proto = P.load_protocol(PROTOCOL)
        self.assertNotEqual(set(proto["scenario_offsets"].values()), {24})                     # offsets differ from uniform-24
        self.assertEqual(proto["parent_protocol"]["id"], "uniform-24-words (v0.1.0, v0.1.1, v0.1.2)")
        self.assertNotEqual(proto["controls_manifest_sha256"], V012_MANIFEST_SHA256)           # different controls file
        self.assertNotEqual(C.load_manifest(REPAIR_MANIFEST)["controls"]["triage-severity"]["control_id"],
                            C.load_manifest(V012_MANIFEST)["controls"]["triage-severity"]["control_id"])

    def test_legacy_sidecar_views_do_not_claim_a_protocol(self):
        base = _json(os.path.join(ROOT, "tests", "data", "legacy_qwen1p5b_v0.1.0", "baseline.sanitized.json"))
        trace = os.path.join(ROOT, "tests", "data", "legacy_qwen1p5b_v0.1.0", "traces.jsonl")
        doc = meta_mod.legacy_sidecar(trace_path=trace, records=read_traces(trace), baseline_doc=base)
        self.assertEqual(doc["experiment"]["protocol"], UNKNOWN)
        self.assertEqual(doc["experiment"]["control_algorithm"], UNKNOWN)

    def test_historical_and_design_evidence_is_unchanged(self):
        for rel, sha in HISTORICAL.items():
            path = os.path.join(ROOT, *rel.split("/"))
            if os.path.exists(path):
                self.assertEqual(_sha(path), sha, rel)
        d = os.path.join(ROOT, "runs", "v0.1.2-control-design")
        for name, sha in DESIGN_EVIDENCE.items():
            path = os.path.join(d, name)
            if os.path.exists(path):
                self.assertEqual(_sha(path), sha, name)
        self.assertEqual(_sha(V012_MANIFEST), V012_MANIFEST_SHA256)
        self.assertEqual(_sha(FIXTURES), FIXTURES_SHA256)

    def test_fixtures_were_not_edited_to_repair_the_window(self):
        sc = scen()["triage-severity"]
        self.assertEqual(sc.adjacent.split()[39], "major")                 # the cue still sits at word 40
        self.assertEqual(len(sc.adjacent.split()), 50)

    def test_equations_thresholds_schema_and_normalisation_are_unchanged(self):
        # the existing anchor test (v0.1.2) covers source segments; restate the numeric facts here
        from bleediff import CONTROL_BAND_NATS, EXPOSURE_DELTA_NATS, MIN_TREATMENT_FRACTION, EPS

        self.assertEqual((EXPOSURE_DELTA_NATS, CONTROL_BAND_NATS, MIN_TREATMENT_FRACTION, EPS), (0.4, 0.1, 0.8, 1e-12))
        self.assertEqual(len(TRACE_SCHEMA), 15)
        anchors = _json(os.path.join(ROOT, "tests", "data", "frozen_anchors_6a03565.json"))["anchors"]
        text = _text(os.path.join(ROOT, "bleediff", "fold.py"))
        got = {}
        for n in ast.parse(text).body:
            nm = getattr(n, "name", None)
            if nm is None and isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                nm = n.targets[0].id
            if nm in anchors["bleediff/fold.py"]:
                got[nm] = hashlib.sha256(ast.get_source_segment(text, n).replace("\r\n", "\n").encode()).hexdigest()
        self.assertEqual(got, anchors["bleediff/fold.py"])

    def test_no_third_party_imports_in_new_modules(self):
        allowed = set(sys.stdlib_module_names) | {"bleediff", "tests", "test_bleediff", "_tok"}
        for rel in ("bleediff/protocol.py", "tests/test_v013_protocol.py"):
            tree = ast.parse(_text(os.path.join(ROOT, rel)))
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mods = [node.module.split(".")[0]]
                self.assertFalse([m for m in mods if m not in allowed], (rel, mods))


if __name__ == "__main__":
    unittest.main()
