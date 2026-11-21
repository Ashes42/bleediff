"""v0.1.2 tests: the neutral-prose/1 control, fragment integrity, and preservation of everything frozen.

No inference and no network: a stdlib mock server plus a deterministic Qwen-style tokenizer (tests/_tok.py).
Passing these structural tests does NOT establish semantic neutrality; that needs the separate qualitative review.
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
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from _tok import qwenish  # noqa: E402

try:
    from test_bleediff import FIXTURES, MockState, mock_controls_manifest, pipeline, run_cli, serve  # type: ignore
except ImportError:  # pragma: no cover
    from tests.test_bleediff import FIXTURES, MockState, mock_controls_manifest, pipeline, run_cli, serve

from bleediff import UNKNOWN, __version__  # noqa: E402
from bleediff import control as C  # noqa: E402
from bleediff import meta as meta_mod  # noqa: E402
from bleediff.arena import load_fixtures, plan_requests  # noqa: E402
from bleediff.control import ControlMatchError  # noqa: E402

PRESERVED = os.path.join(ROOT, "fixtures", "controls", "neutral-prose-1.json")
ANCHORS = os.path.join(ROOT, "tests", "data", "frozen_anchors_6a03565.json")
#: The preserved selection. If this changes, the controls changed: re-review and re-register, never silently.
PRESERVED_MANIFEST_SHA256 = "7266bf7e687f1829b0fd18a26c484d7d8fe0c359b1e223fe909dcba8a50b5145"
FIXTURES_SHA256 = "38ab93521643523dfd8fcb023b59055f1db21b1f22db16ed00f9a49fa538e250"

#: The historical (v0.1.0/v0.1.1) control fragment for incident-triage, reconstructed with the real tokenizer and
#: verified against the sha256 recorded in the repeat run's sidecar. Used only as a NEGATIVE example.
LEGACY_FRAGMENT = ("During 1 many throughout season: Yes End this light understandbu low two by 50:62 out thanpo "
                   "34,936 also should day general. Use paper key")

# The test suite's OWN notion of an answer cue (deliberately not imported from the module under test)
CUE_WORDS = {"yes", "no", "true", "false", "correct", "wrong", "option", "choice", "choose", "select", "answer",
             "best", "better", "worst", "should", "must", "major", "minor", "critical", "severe", "urgent",
             "first", "second", "third", "fourth", "one", "two", "three", "four", "alpha", "beta", "gamma",
             "delta", "apple", "banana", "cat", "dog", "severity", "sev", "use", "set", "get", "end", "key",
             "map", "please", "ignore", "recommend", "prefer", "pick", "important"}
SCALE_WORDS = {"small", "tiny", "large", "big", "huge", "heavy", "light", "thin", "low", "high", "tall", "short",
               "long", "wide", "narrow", "slow", "slowly", "quick", "quickly", "fast", "steady", "steadily"}
SENTENCE_STARTERS = {"The", "Each", "Several", "Many", "Some"}

_GEN: dict = {}


def generated() -> dict:
    """neutral-prose/1 controls for all five scenarios against the deterministic test tokenizer (cached)."""
    if not _GEN:
        fx = load_fixtures(FIXTURES)
        for sc in fx.scenarios:
            _GEN[sc.id] = C.generate_control(sc, 24, "words", qwenish, max_draws=3000)
    return _GEN


def scen():
    return {sc.id: sc for sc in load_fixtures(FIXTURES).scenarios}


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _body_sentences(text: str) -> list[str]:
    body = text.split(":", 1)[1].strip()
    return [s for s in re.split(r"(?<=\.)\s+", body) if s]


# --------------------------------------------------------------------------------------
class TestDeterministicGeneration(unittest.TestCase):
    def test_same_inputs_give_the_identical_control(self):
        sc = scen()["vendor-routing"]
        a = C.generate_control(sc, 24, "words", qwenish, max_draws=3000)
        b = C.generate_control(sc, 24, "words", qwenish, max_draws=3000)
        self.assertEqual(a.fragment_text, b.fragment_text)
        self.assertEqual((a.control_id, a.enumeration_index, a.cluster), (b.control_id, b.enumeration_index, b.cluster))
        self.assertEqual(a.fragment_text, generated()["vendor-routing"].fragment_text)

    def test_generation_is_independent_of_python_hash_randomisation(self):
        code = ("import sys; sys.path[:0] = [%r, %r]\n"
                "from _tok import qwenish\n"
                "from bleediff import control as C\n"
                "from bleediff.arena import load_fixtures\n"
                "sc = [s for s in load_fixtures(%r).scenarios if s.id == 'config-audit'][0]\n"
                "r = C.generate_control(sc, 24, 'words', qwenish, max_draws=3000)\n"
                "print(r.control_id, r.enumeration_index, r.fragment_text)\n") % (ROOT, os.path.join(ROOT, "tests"), FIXTURES)
        outs = []
        for seed in ("0", "98765"):
            p = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=300,
                               env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"})
            self.assertEqual(p.returncode, 0, p.stderr)
            outs.append(p.stdout.strip())
        self.assertEqual(outs[0], outs[1])
        g = generated()["config-audit"]
        self.assertEqual(outs[0], f"{g.control_id} {g.enumeration_index} {g.fragment_text}")

    def test_control_id_binds_algorithm_scenario_offset_fragment_tokenizer_and_lexicon(self):
        base = dict(scenario_id="s", offset=24, unit="words", treatment_sha256="t" * 64, fingerprint="f" * 64)
        cid = C.control_id_for(**base)
        self.assertRegex(cid, r"^[0-9a-f]{32}$")
        for k, v in (("scenario_id", "s2"), ("offset", 12), ("treatment_sha256", "u" * 64), ("fingerprint", "g" * 64)):
            self.assertNotEqual(cid, C.control_id_for(**{**base, k: v}), k)
        with mock.patch.object(C, "LEXICON_SHA256", "0" * 64):
            self.assertNotEqual(cid, C.control_id_for(**base), "a lexicon change must change every control_id")
        self.assertTrue(C.ALGORITHM.endswith("/1"))
        self.assertNotEqual(C.ALGORITHM, "neutral-filler/2")

    def test_selection_rule_never_consults_a_model(self):
        import inspect

        tree = ast.parse(inspect.getsource(C))
        used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for forbidden in ("choice_probs", "completion_payload", "parse_choice_reading", "parse_choice_probs", "_request",
                          "last_coverage", "post_sampling_probs"):
            self.assertNotIn(forbidden, used, forbidden)
        strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        self.assertNotIn("/completion", strings)
        # the only server access is the tokenize-only wrapper
        self.assertEqual(sum(1 for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "tokenize"
                             and isinstance(n.value, ast.Name) and n.value.id == "client"), 1)


# --------------------------------------------------------------------------------------
class TestStructuralParity(unittest.TestCase):
    def test_byte_word_and_fragment_token_parity_for_every_scenario(self):
        for sid, sc in scen().items():
            r = generated()[sid]
            treat = C.treatment_fragment(sc, 24)
            self.assertEqual(len(r.fragment_text.encode()), len(treat.encode()), sid)
            self.assertEqual(len(r.fragment_text.split()), len(treat.split()), sid)
            self.assertEqual(len(qwenish("\n" + r.fragment_text)), len(qwenish("\n" + treat)), sid)
            self.assertEqual(r.token_parity, "VERIFIED")
            self.assertEqual(r.fragment_sha256, C.fragment_sha256(r.fragment_text))

    def test_whole_third_request_token_parity_and_identical_surroundings(self):
        for sid, sc in scen().items():
            r = generated()[sid]
            pp = C.prompt_parity(sc, 24, "words", r.fragment_text, qwenish)
            self.assertEqual(pp["prompt_tokens"][0], pp["prompt_tokens"][1], sid)
            self.assertTrue(pp["requests_1_2_identical"], sid)
            self.assertEqual(pp["n_requests"], (3, 3), sid)
            self.assertTrue(pp["geometry_identical"], sid)
            self.assertTrue(pp["surrounding_prompt_identical"], sid)

    def test_identical_insertion_boundary_and_displacement(self):
        for sid, sc in scen().items():
            r = generated()[sid]
            t = plan_requests(sc, "treatment", 24)
            c = plan_requests(sc, "control", 24, "words", filler=r.fragment_text)
            try:
                self.assertEqual((t.geometry.boundary, t.geometry.read_end, t.geometry.bleed_bytes, t.geometry.bleed_words),
                                 (c.geometry.boundary, c.geometry.read_end, c.geometry.bleed_bytes, c.geometry.bleed_words))
                self.assertFalse(c.geometry.clamped)
                self.assertEqual(c.bleed_fragment.decode(), r.fragment_text)
                # the fragment sits at the same offset inside request #3 in both arms
                self.assertEqual(bytes(t.prompts[2]).find(t.bleed_fragment), bytes(c.prompts[2]).find(c.bleed_fragment))
            finally:
                t.dispose()
                c.dispose()

    def test_skeleton_is_mirrored(self):
        for sid, sc in scen().items():
            r = generated()[sid]
            tsk = C.parse_skeleton(C.treatment_fragment(sc, 24))
            csk = C.parse_skeleton(r.fragment_text)
            self.assertEqual({**tsk.to_json(), "bytes": 0}, {**csk.to_json(), "bytes": 0}, sid)
            self.assertEqual(r.fragment_text.split()[:2], ["Tenant", "4"], "shared fixed prefix, identical in both arms")


# --------------------------------------------------------------------------------------
class TestNoAnswerCues(unittest.TestCase):
    def _all_texts(self):
        out = {sid: r.fragment_text for sid, r in generated().items()}
        with open(PRESERVED, encoding="utf-8") as fh:
            m = json.load(fh)
        out.update({f"preserved:{sid}": e["fragment_text"] for sid, e in m["controls"].items()})
        return out

    def test_no_cue_scale_or_label_words_no_lone_letters_no_option_digits(self):
        for name, text in self._all_texts().items():
            words = [re.sub(r"^\W+|\W+$", "", w).lower() for w in text.split()[2:]]   # after the shared prefix
            self.assertFalse({w for w in words} & CUE_WORDS, (name, {w for w in words} & CUE_WORDS))
            self.assertFalse({w for w in words} & SCALE_WORDS, (name, {w for w in words} & SCALE_WORDS))
            self.assertFalse([w for w in text.split()[2:] if re.fullmatch(r"[A-D]", w.strip(".,:"))], name)
            self.assertIsNone(re.search(r"\b[A-D]\)", text), name)
            for run in re.findall(r"\d+", " ".join(text.split()[2:])):
                self.assertFalse(len(run) == 1 and run in "1234", (name, run))

    def test_no_word_of_the_scenario_text_is_reused(self):
        for sid, sc in scen().items():
            blob = " ".join([sc.question, *sc.options.values(), *sc.authorized, sc.adjacent]).lower()
            vocab = set(re.findall(r"[a-z]+", blob)) - set(C.FUNCTION_WORDS)
            for name in (sid, f"preserved:{sid}"):
                text = self._all_texts()[name]
                used = {w for w in re.findall(r"[a-z]+", text.lower()[len("tenant 4"):])}
                self.assertFalse(used & vocab, (name, used & vocab))
            runs = set(re.findall(r"\d+", blob))
            for run in re.findall(r"\d+", self._all_texts()[sid][len("Tenant 4"):]):
                self.assertFalse(len(run) >= 2 and run in runs, (sid, run))

    def test_every_sentence_is_declarative_and_starts_with_a_determiner(self):
        for name, text in self._all_texts().items():
            for s in _body_sentences(text):
                self.assertIn(s.split()[0], SENTENCE_STARTERS, (name, s))

    def test_no_repeated_content_words_or_shared_stems(self):
        for name, text in self._all_texts().items():
            words = [re.sub(r"[^a-z]", "", w.lower()) for w in text.split()[2:]]
            content = [w for w in words if w and w not in C.FUNCTION_WORDS]
            self.assertEqual(len(content), len(set(content)), name)
            stems = [w[:5] for w in content if len(w) >= 5]
            self.assertEqual(len(stems), len(set(stems)), (name, stems))


# --------------------------------------------------------------------------------------
class TestSyntacticCoherenceChecks(unittest.TestCase):
    def test_generated_and_preserved_fragments_parse_against_the_grammar(self):
        with open(PRESERVED, encoding="utf-8") as fh:
            m = json.load(fh)
        for sid, sc in scen().items():
            for text, cluster in ((generated()[sid].fragment_text, generated()[sid].cluster),
                                  (m["controls"][sid]["fragment_text"], m["controls"][sid]["cluster"])):
                sk = C.parse_skeleton(C.treatment_fragment(sc, 24))
                self.assertEqual(C.recognize_fragment(text.split(), sk, cluster), [], (sid, text))

    def test_the_checks_have_teeth_the_legacy_filler_fails_them(self):
        sc = scen()["incident-triage"]
        sk = C.parse_skeleton(C.treatment_fragment(sc, 24))
        words = LEGACY_FRAGMENT.split()
        self.assertEqual(len(words), sk.words)
        vocab, runs = C.scenario_vocabulary(sc)
        lex = C.lexical_problems(words, sk, vocab, runs)
        self.assertTrue(any("yes" in p for p in lex), lex)
        self.assertTrue(any("'use'" in p for p in lex), lex)
        self.assertTrue(any("banned" in p for p in lex), lex)
        self.assertTrue(C.recognize_fragment(words, sk, "bakery"), "word salad must not parse")

    def test_scrambling_or_corrupting_a_valid_control_is_detected(self):
        sc = scen()["triage-severity"]
        r = generated()["triage-severity"]
        sk = C.parse_skeleton(C.treatment_fragment(sc, 24))
        words = r.fragment_text.split()
        shuffled = words[:2] + list(reversed(words[2:]))
        self.assertTrue(C.recognize_fragment(shuffled, sk, r.cluster) or C.skeleton_problems(shuffled, sk))
        swapped = list(words)
        swapped[8] = "purged"                                  # a word that belongs to no cluster
        self.assertTrue(C.recognize_fragment(swapped, sk, r.cluster))
        self.assertTrue(C.recognize_fragment(words, sk, "no-such-cluster"))
        imperative = list(words)
        imperative[next(i for i, w in enumerate(words) if w.endswith(":")) + 1] = "Use"   # sentence now starts with a verb
        self.assertTrue(C.recognize_fragment(imperative, sk, r.cluster))

    def test_independent_recognizer_accepts_every_solver_output_across_many_draws(self):
        # property check: draw many candidate structures with different seeds and make sure grammar and solver agree
        sc = scen()["incident-triage"]
        sk = C.parse_skeleton(C.treatment_fragment(sc, 24))
        vocab, runs = C.scenario_vocabulary(sc)
        counter = C.TokenCounter(qwenish)
        tgt_b = sk.bytes - (sk.words - 1) - sum(len(w) for w in sk.prefix)
        tgt_t = len(qwenish("\n" + C.treatment_fragment(sc, 24))) - counter.additive(list(sk.prefix))
        accepted = 0
        cache: dict = {}
        for seed in range(300):
            rng = C.Rng(seed * 7919 + 1)
            cl = rng.choice(C.CLUSTERS)
            specs = C.build_slots(rng, cl, sk)
            if specs is None:
                continue
            groups = []
            for sp in specs:
                key = (cl["id"], sp[4], sp[1], sp[2], sp[3])
                if key not in cache:
                    cache[key] = C._signature_groups(sp, counter, vocab, runs)
                groups.append(cache[key])
            solved = C.solve_slots(rng, groups, tgt_b, tgt_t)
            if solved is None:
                continue
            words = list(sk.prefix) + [w for part in solved for w in part]
            self.assertEqual(len(" ".join(words).encode()), sk.bytes)
            self.assertEqual(C.recognize_fragment(words, sk, cl["id"]), [], " ".join(words))
            accepted += 1
        self.assertGreater(accepted, 4)


# --------------------------------------------------------------------------------------
class TestFailsClosed(unittest.TestCase):
    def test_a_non_additive_tokenizer_is_refused(self):
        sc = scen()["incident-triage"]

        def nonadditive(text: str) -> list:                    # the full fragment costs one token more than its words
            base = qwenish(text)
            return base + [0] if text.startswith("\n") and len(text) > 40 else base

        with self.assertRaises(ControlMatchError) as cm:
            C.generate_control(sc, 24, "words", nonadditive, max_draws=25)
        self.assertEqual(cm.exception.reason, "CONTROL_TOKENIZER_NOT_ADDITIVE")

    def test_an_unreachable_token_budget_fails_closed_instead_of_relaxing(self):
        sc = scen()["incident-triage"]
        heavy_words = ("CDN", "invalidation", "purged", "regions", "operations")

        def heavy(text: str) -> list:                          # additive, but the treatment words cost 40 tokens each
            return qwenish(text) + [7] * (40 * sum(text.count(w) for w in heavy_words))

        with self.assertRaises(ControlMatchError) as cm:
            C.generate_control(sc, 24, "words", heavy, max_draws=25)
        self.assertEqual(cm.exception.reason, "CONTROL_MATCH_FAILED")
        self.assertIn("NOT relaxed", str(cm.exception))

    def test_exhausted_search_reports_match_failure_and_never_relaxes(self):
        sc = scen()["architecture-selection"]
        with mock.patch.object(C, "build_slots", return_value=None):
            with self.assertRaises(ControlMatchError) as cm:
                C.generate_control(sc, 24, "words", qwenish, max_draws=10)
        self.assertEqual(cm.exception.reason, "CONTROL_MATCH_FAILED")
        self.assertIn("NOT relaxed", str(cm.exception))

    def test_unsupported_unit_skeleton_and_exit_code(self):
        sc = scen()["incident-triage"]
        with self.assertRaises(ControlMatchError) as cm:
            C.generate_control(sc, 24, "bytes", qwenish)
        self.assertEqual(cm.exception.reason, "CONTROL_UNSUPPORTED_UNIT")
        with self.assertRaises(ControlMatchError):
            C.parse_skeleton("too short")
        self.assertIsInstance(cm.exception, C.Inconclusive)  # => exit code 2 everywhere

    def test_static_mode_without_a_tokenizer_is_pending_not_validated(self):
        r = C.generate_control(scen()["vendor-routing"], 24, "words", None, max_draws=3000)
        self.assertEqual(r.token_parity, "PENDING")
        self.assertNotIn("fragment_tokens", r.checks)
        self.assertNotIn("prompt_tokens", r.checks)
        self.assertEqual(r.tokenizer_fingerprint, UNKNOWN)
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = C.main(["static", "--fixtures", FIXTURES])
        self.assertEqual(code, 0)
        self.assertIn("PENDING", out.getvalue())


# --------------------------------------------------------------------------------------
class TestReproducibilityMetadataAndDisposal(unittest.TestCase):
    def test_record_carries_everything_needed_to_reproduce_it(self):
        r = generated()["config-audit"]
        sc = scen()["config-audit"]
        self.assertEqual(r.algorithm, "neutral-prose/1")
        self.assertEqual(r.lexicon_sha256, C.LEXICON_SHA256)
        self.assertEqual(r.tokenizer_fingerprint, C.tokenizer_fingerprint(qwenish))
        self.assertEqual(r.treatment_fragment_sha256, C.fragment_sha256(C.treatment_fragment(sc, 24)))
        self.assertEqual(r.control_id, C.control_id_for(scenario_id=sc.id, offset=24, unit="words",
                                                        treatment_sha256=r.treatment_fragment_sha256,
                                                        fingerprint=r.tokenizer_fingerprint))
        self.assertIsInstance(r.enumeration_index, int)
        self.assertIn(r.cluster, {c["id"] for c in C.CLUSTERS})
        self.assertEqual(r.counts["fragment_tokens"][0], r.counts["fragment_tokens"][1])
        self.assertEqual(r.counts["prompt_tokens"][0], r.counts["prompt_tokens"][1])

    def test_manifest_round_trip_refuses_overwrite_and_wrong_schema(self):
        fx = load_fixtures(FIXTURES)
        m = C.build_controls(fx, 24, "words", qwenish, max_draws=3000)
        self.assertEqual(m["schema"], "bleediff.controls/1")
        self.assertEqual({k for k in m["controls"]}, {s.id for s in fx.scenarios})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            C.write_manifest(path, m)
            self.assertEqual(C.load_manifest(path)["controls"].keys(), m["controls"].keys())
            with self.assertRaises(Exception):
                C.write_manifest(path, m)                      # never overwrites a preserved selection
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w") as fh:
                json.dump({"schema": "other"}, fh)
            with self.assertRaises(ControlMatchError):
                C.load_manifest(bad)

    def test_generation_retains_no_source_buffers(self):
        sc = scen()["triage-severity"]
        plans = []
        real = C.plan_requests

        def spy(*a, **k):
            p = real(*a, **k)
            plans.append(p)
            return p

        with mock.patch.object(C, "plan_requests", spy):
            C.generate_control(sc, 24, "words", qwenish, max_draws=3000)
        self.assertTrue(plans)
        for p in plans:
            self.assertEqual(p.prompts, [])
            self.assertEqual(len(p.arena.buf), 0)
            self.assertEqual(p.bleed_fragment, b"")

    def test_preserved_manifest_holds_no_treatment_text_and_no_vectors(self):
        with open(PRESERVED, encoding="utf-8") as fh:
            text = fh.read()
        for sc in load_fixtures(FIXTURES).scenarios:
            w = sc.adjacent.split()
            for i in range(len(w) - 2):
                self.assertNotIn(" ".join(w[i:i + 3]), text)
        self.assertNotIn("Evidence:", text)
        self.assertNotRegex(text, r'"(p_[ABCD]|logprobs?|vectors?|prompts?|responses?)"\s*:')


# --------------------------------------------------------------------------------------
class TestPreservedControls(unittest.TestCase):
    def test_the_preserved_selection_is_pinned(self):
        self.assertEqual(_sha(PRESERVED), PRESERVED_MANIFEST_SHA256)

    def test_static_validity_of_the_preserved_controls_against_the_current_fixtures(self):
        m = C.load_manifest(PRESERVED)
        self.assertEqual(m["lexicon_sha256"], C.LEXICON_SHA256, "lexicon changed since the controls were selected")
        self.assertEqual(m["fixtures_sha256"], FIXTURES_SHA256)
        self.assertEqual((m["offset"], m["unit"]), (24, "words"))
        for sid, sc in scen().items():
            e = m["controls"][sid]
            treat = C.treatment_fragment(sc, 24)
            self.assertEqual(e["token_parity"], "VERIFIED")
            self.assertEqual(e["treatment_fragment_sha256"], C.fragment_sha256(treat))
            self.assertEqual(e["fragment_sha256"], C.fragment_sha256(e["fragment_text"]))
            self.assertEqual(len(e["fragment_text"].encode()), len(treat.encode()))
            self.assertEqual(len(e["fragment_text"].split()), len(treat.split()))
            self.assertEqual(e["counts"]["fragment_tokens"][0], e["counts"]["fragment_tokens"][1])
            self.assertEqual(e["counts"]["prompt_tokens"][0], e["counts"]["prompt_tokens"][1])
            self.assertEqual(e["control_id"], C.control_id_for(
                scenario_id=sid, offset=24, unit="words", treatment_sha256=e["treatment_fragment_sha256"],
                fingerprint=m["tokenizer_fingerprint"]))
            sk = C.parse_skeleton(treat)
            vocab, runs = C.scenario_vocabulary(sc)
            self.assertEqual(C.lexical_problems(e["fragment_text"].split(), sk, vocab, runs), [], sid)
            self.assertEqual(C.recognize_fragment(e["fragment_text"].split(), sk, e["cluster"]), [], sid)

    def test_a_tampered_or_stale_control_does_not_verify(self):
        m = C.load_manifest(PRESERVED)
        sid = "incident-triage"
        sc = scen()[sid]
        # fingerprint differs from the mock tokenizer => refused ("validated against a different tokenizer")
        probs = C.verify_control(sc, m["controls"][sid], m, 24, "words", MockState().tokenize)
        self.assertTrue(any("tokenizer differs" in p for p in probs), probs)
        # wrong offset
        self.assertTrue(any("offset" in p for p in C.verify_control(sc, m["controls"][sid], m, 12, "words", qwenish)))
        # stale lexicon
        with mock.patch.object(C, "LEXICON_SHA256", "0" * 64):
            self.assertTrue(any("lexicon" in p for p in C.verify_control(sc, m["controls"][sid], m, 24, "words", qwenish)))
        # tampered text
        t = json.loads(json.dumps(m))
        t["controls"][sid]["fragment_text"] = t["controls"][sid]["fragment_text"].replace("Tenant 4", "Tenant 5")
        self.assertTrue(C.verify_control(sc, t["controls"][sid], t, 24, "words", qwenish))
        # changed fixture (different treatment fragment) is detected
        t2 = json.loads(json.dumps(m))
        t2["controls"][sid]["treatment_fragment_sha256"] = "0" * 64
        self.assertTrue(any("fixtures changed" in p for p in C.verify_control(sc, t2["controls"][sid], t2, 24, "words", qwenish)))


# --------------------------------------------------------------------------------------
class TestBounceUsesPreservedControls(unittest.TestCase):
    def _run(self, **kw):
        cm = serve()
        url, state = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        res = pipeline(tmp.name, url, **kw)
        side = meta_mod.load_sidecar(meta_mod.sidecar_path(res["trace"])) if os.path.exists(res["trace"]) else None
        return res, side, tmp.name

    def test_default_control_is_neutral_prose_and_is_recorded_by_hash_only(self):
        res, side, _ = self._run()
        self.assertEqual(res["bounce"][0], 0, res["bounce"][1])
        e = side["experiment"]
        self.assertEqual(e["control_algorithm"], "neutral-prose/1")
        self.assertEqual(e["control_manifest_sha256"], _sha(mock_controls_manifest()))
        manifest = C.load_manifest(mock_controls_manifest())
        blob = json.dumps(side)
        for m in side["records"]:
            if m["control"]:
                f = m["control_filler"]
                entry = manifest["controls"][m["scenario_id"]]
                self.assertEqual(f["algorithm"], "neutral-prose/1")
                self.assertEqual(f["sha256"], entry["fragment_sha256"])
                self.assertEqual(f["control_id"], entry["control_id"])
                self.assertEqual(f["repro_id"], entry["control_id"])
                self.assertEqual(f["enumeration_index"], entry["enumeration_index"])
                self.assertEqual(f["salt"], "NOT_APPLICABLE")
                self.assertNotIn(entry["fragment_text"], blob, "raw control text must not be copied into the sidecar")
                self.assertNotIn(" ".join(entry["fragment_text"].split()[2:7]), blob)
            else:
                self.assertEqual(m["control_filler"], "NOT_APPLICABLE")

    def test_control_arm_prompts_have_the_same_token_count_as_treatment(self):
        _, side, _ = self._run()
        by = {}
        for m in side["records"]:
            by.setdefault(m["trial"], {})[m["control"]] = m["request_prompt_tokens"]
        for t, arms in by.items():
            self.assertEqual(arms[False], arms[True], t)

    def test_a_manifest_for_another_tokenizer_is_refused_and_writes_nothing(self):
        with serve() as (url, _), tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "baseline.json")
            run_cli("calibrate", "--server", url, "--fixtures", FIXTURES, "--out", base)
            trace = os.path.join(tmp, "t.jsonl")
            code, text = run_cli("bounce", "--server", url, "--baseline", base, "--out", trace,
                                 "--controls-manifest", PRESERVED)           # Qwen-validated; the mock tokenizer differs
            self.assertEqual(code, 2, text)
            self.assertIn("CONTROL_NOT_VALIDATED", text)
            self.assertFalse(os.path.exists(trace))
            self.assertFalse(os.path.exists(meta_mod.sidecar_path(trace)))

    def test_missing_manifest_wrong_offset_and_tampering_are_inconclusive(self):
        with serve() as (url, _), tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "baseline.json")
            run_cli("calibrate", "--server", url, "--fixtures", FIXTURES, "--out", base)
            trace = os.path.join(tmp, "t.jsonl")
            code, text = run_cli("bounce", "--server", url, "--baseline", base, "--out", trace,
                                 "--controls-manifest", os.path.join(tmp, "nope.json"))
            self.assertEqual((code, "CONTROL_MANIFEST_INVALID" in text), (2, True), text)
            code, text = run_cli("bounce", "--server", url, "--baseline", base, "--out", trace, "--offset", "12",
                                 "--controls-manifest", mock_controls_manifest())
            self.assertEqual(code, 2, text)
            self.assertIn("offset=24", text)
            doc = C.load_manifest(mock_controls_manifest())
            sid = next(iter(doc["controls"]))
            doc["controls"][sid]["fragment_text"] = doc["controls"][sid]["fragment_text"].replace("Tenant 4", "Tenant 7")
            bad = os.path.join(tmp, "tampered.json")
            with open(bad, "w") as fh:
                json.dump(doc, fh)
            code, text = run_cli("bounce", "--server", url, "--baseline", base, "--out", trace,
                                 "--controls-manifest", bad)
            self.assertEqual(code, 2, text)
            self.assertIn("CONTROL_NOT_VALIDATED", text)
            self.assertFalse(os.path.exists(trace))

    def test_historical_filler_is_only_reachable_by_naming_it(self):
        res, side, _ = self._run(control="neutral-filler/2")
        self.assertEqual(res["bounce"][0], 0, res["bounce"][1])
        self.assertEqual(side["experiment"]["control_algorithm"], "neutral-filler/2")
        self.assertEqual(side["experiment"]["control_manifest_sha256"], "NOT_APPLICABLE")
        with self.assertRaises(SystemExit) as ex:
            run_cli("bounce", "--control", "made-up/9")
        self.assertEqual(ex.exception.code, 2)   # argparse rejects an unknown algorithm

    def test_bleed_preview_shows_the_preserved_control_and_matched_geometry(self):
        code, text = run_cli("bleed", "--fixtures", FIXTURES, "--offset", "24")
        self.assertEqual(code, 0, text)
        self.assertIn("neutral-prose/1 (preserved manifest)", text)
        self.assertNotIn("MISMATCH", text)
        code, text = run_cli("bleed", "--fixtures", FIXTURES, "--offset", "7")
        self.assertIn("neutral-filler/2 (historical", text)

    def test_legacy_sidecars_stay_unknown_for_the_new_fields(self):
        with open(os.path.join(ROOT, "tests", "data", "legacy_qwen1p5b_v0.1.0", "baseline.sanitized.json"), encoding="utf-8") as fh:
            base = json.load(fh)
        trace = os.path.join(ROOT, "tests", "data", "legacy_qwen1p5b_v0.1.0", "traces.jsonl")
        from bleediff.fold import read_traces

        doc = meta_mod.legacy_sidecar(trace_path=trace, records=read_traces(trace), baseline_doc=base)
        self.assertEqual(doc["experiment"]["control_algorithm"], UNKNOWN)
        self.assertEqual(doc["experiment"]["control_manifest_sha256"], UNKNOWN)


# --------------------------------------------------------------------------------------
class TestFragmentIntegrityAudit(unittest.TestCase):
    def test_displaced_fragments_and_their_B_evidence(self):
        rep = {r["scenario_id"]: r for r in C.fragment_integrity(load_fixtures(FIXTURES), 24)}
        for sid in ("incident-triage", "architecture-selection", "vendor-routing", "config-audit"):
            self.assertTrue(rep[sid]["all_anchors_present"], sid)
        t = rep["triage-severity"]
        self.assertFalse(t["all_anchors_present"])
        self.assertEqual(t["anchors"], ["major"])
        self.assertEqual(t["anchor_first_word_index_in_slot"]["major"], 40)   # 16 words beyond the 24-word window

    def test_the_known_defect_is_reported_and_the_fixtures_were_not_touched_to_hide_it(self):
        self.assertEqual(_sha(FIXTURES), FIXTURES_SHA256)
        self.assertEqual(C.DEFAULT_MANIFEST, os.path.join("fixtures", "controls", "neutral-prose-1.json"))


# --------------------------------------------------------------------------------------
def _norm(t: str) -> str:
    return t.replace("\r\n", "\n")


def _segments(path: str, names: set) -> dict:
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        text = fh.read()
    out = {}
    for n in ast.parse(text).body:
        nm = getattr(n, "name", None)
        if nm is None and isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            nm = n.targets[0].id
        if nm in names:
            out[nm] = hashlib.sha256(_norm(ast.get_source_segment(text, n)).encode()).hexdigest()
    return out


class TestFrozenMeasurementIsUntouched(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(ANCHORS, encoding="utf-8") as fh:
            cls.anchors = json.load(fh)["anchors"]

    def test_equations_fold_schema_thresholds_gate_and_arena_match_commit_6a03565(self):
        for path, expected in self.anchors.items():
            if path == "client-normalisation":
                continue
            got = _segments(path, set(expected))
            self.assertEqual(got, expected, f"a frozen definition in {path} changed")

    def test_four_choice_normalisation_lines_are_unchanged(self):
        with open(os.path.join(ROOT, "bleediff", "client.py"), encoding="utf-8") as fh:
            lines = [l.split("#")[0].rstrip() for l in fh.read().splitlines()
                     if "masses = [_logsumexp" in l or "z = _logsumexp(masses)" in l or "probs = [math.exp(m - z)" in l]
        self.assertEqual(hashlib.sha256("\n".join(lines).encode()).hexdigest(), self.anchors["client-normalisation"])

    def test_the_historical_filler_algorithm_is_preserved_byte_for_byte(self):
        from bleediff.arena import FILLER_ALGORITHM

        self.assertEqual(FILLER_ALGORITHM, "neutral-filler/2")
        self.assertIn("make_neutral_filler", self.anchors["bleediff/arena.py"])

    def test_thresholds_and_trace_schema(self):
        from bleediff import CONTROL_BAND_NATS, EXIT_FAIL, EXIT_INCONCLUSIVE, EXIT_PASS, EXPOSURE_DELTA_NATS, MIN_TREATMENT_FRACTION
        from bleediff.fold import TRACE_SCHEMA

        self.assertEqual((EXPOSURE_DELTA_NATS, CONTROL_BAND_NATS, MIN_TREATMENT_FRACTION), (0.4, 0.1, 0.8))
        self.assertEqual((EXIT_PASS, EXIT_FAIL, EXIT_INCONCLUSIVE), (0, 1, 2))
        self.assertEqual(len(TRACE_SCHEMA), 15)

    def test_original_fixtures_and_historical_evidence_are_unchanged(self):
        self.assertEqual(_sha(FIXTURES), FIXTURES_SHA256)
        legacy = os.path.join(ROOT, "tests", "data", "legacy_qwen1p5b_v0.1.0", "traces.jsonl")
        self.assertEqual(_sha(legacy), "c600549e44500e7a5a03d073ce6d2d336cf6b807d6b740ea20f0429d57fe03f8")
        orig = os.path.join(ROOT, "runs", "traces-qwen1.5b.jsonl")
        if os.path.exists(orig):
            self.assertEqual(_sha(orig), "c600549e44500e7a5a03d073ce6d2d336cf6b807d6b740ea20f0429d57fe03f8")

    def test_version_is_bumped_consistently(self):
        import tomllib

        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
            self.assertEqual(tomllib.load(fh)["project"]["version"], __version__)
        self.assertRegex(__version__, r"^0\.\d+\.\d+$")   # 0.1.2 introduced the control; later 0.1.x releases keep it

    def test_no_third_party_imports_in_the_new_module_or_tests(self):
        allowed = set(sys.stdlib_module_names) | {"bleediff", "tests", "test_bleediff", "_tok"}
        for rel in ("bleediff/control.py", "tests/test_v012_control.py", "tests/_tok.py"):
            with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mods = [node.module.split(".")[0]]
                self.assertFalse([m for m in mods if m not in allowed], (rel, mods))


if __name__ == "__main__":
    unittest.main()
