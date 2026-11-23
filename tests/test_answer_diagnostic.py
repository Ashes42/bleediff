"""Optional final-letter diagnostic (FLR-v2) and the five-way answer-only reporting split. Offline; reads only the saved bundles; never changes strict classifications."""
import copy
import os
import unittest


from bleediff import answers as A
from research import reconcile_evidence as RE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "examples", "hosted-comparison")


def anth(text, stop="end_turn"):
    return {"type": "message", "stop_reason": stop, "content": [{"type": "text", "text": text}] if text is not None else []}


def rec(response, provider="anthropic"):
    p = {"anthropic": A.parse_anthropic_answer, "openai": A.parse_openai_answer, "gemini": A.parse_answer}[provider](response)
    return {"provider": provider, "response": response, **p}


class ExtractFinalLetter(unittest.TestCase):
    def letter(self, text):
        return A.extract_final_letter(text)

    def test_markdown_wrapped_and_last_line_letters(self):
        for text in ("**B**", "Reasoning here.\n\n**B**", "_B_", "`B`", "Some prose\nmore prose\n  B  \n\n", "~~B~~"):
            self.assertEqual((self.letter(text)["status"], self.letter(text)["letter"]), ("extracted", "B"), text)

    def test_ambiguous_or_missing_are_rejected(self):
        for text in ("", "   \n \n", "A or B", "The answer is B", "B.", "I choose (B)", "AB", "Final: B\nI am not sure", "E", None, 7):
            r = self.letter(text)
            self.assertEqual((r["status"], r["letter"]), ("none", None), repr(text))

    def test_contradictory_letters_are_rejected_not_resolved(self):
        for text in ("A\nB", "A\nsome prose\nB", "**A**\n\n**B**", "B\nprose\nA"):
            r = self.letter(text)
            self.assertEqual((r["status"], r["letter"]), ("conflict", None), text)
        self.assertEqual(self.letter("B\nprose\nB")["letter"], "B")      # a repeat of the same letter is not a conflict

    def test_deterministic_and_labelled(self):
        self.assertEqual(self.letter("x\n**C**"), self.letter("x\n**C**"))
        self.assertEqual(self.letter("x\n**C**")["rule"], "FLR-v2")


class RuleVersioning(unittest.TestCase):
    def test_v2_differs_from_historical_v1_only_on_conflicts(self):
        def resp(t):
            return {"type": "message", "stop_reason": "end_turn", "content": [{"type": "text", "text": t}]}
        for t in ("**B**", "prose\n\nB", "A\nprose\nA", "A or B", "", "B."):      # no conflicting standalone letter: identical
            self.assertEqual(RE.flr_v1_letter(resp(t)), A.extract_final_letter(t)["letter"], repr(t))
        for t in ("A\nB", "**A**\nprose\n**B**"):      # historical FLR-v1 silently takes the last line; FLR-v2 refuses
            self.assertEqual(RE.flr_v1_letter(resp(t)), "B")
            r = A.extract_final_letter(t)
            self.assertEqual((r["letter"], r["status"], r["rule"]), (None, "conflict", "FLR-v2"))


class DiagnosticClass(unittest.TestCase):
    def test_strict_valid_is_untouched(self):
        r = rec(anth("A"))
        self.assertEqual((r["status"], r["answer"], A.diagnostic_class(r)["class"]), ("ok", "A", "strict_valid"))

    def test_refusal_is_never_reinterpreted(self):
        r = rec(anth("Sorry, I can't help.\n\n**B**", stop="refusal"))
        self.assertEqual(r["status"], "blocked")
        self.assertEqual(A.diagnostic_class(r), {"class": "refusal", "flr_letter": None})
        self.assertEqual(rec(anth("", stop="refusal"))["status"], "blocked")

    def test_format_invalid_with_one_final_letter(self):
        r = rec(anth("The invoice was paid.\n\n**C**"))
        self.assertEqual((r["status"], r["answer"]), ("invalid", None))          # strict status preserved
        self.assertEqual(A.diagnostic_class(r), {"class": "format_recovered", "flr_letter": "C"})
        self.assertEqual(r["status"], "invalid")

    def test_conflict_empty_malformed_and_prose_only(self):
        self.assertEqual(A.diagnostic_class(rec(anth("A\n\nB")))["class"], "ambiguous")
        self.assertEqual(A.diagnostic_class(rec(anth("It depends, A or B.")))["class"], "missing")
        self.assertEqual(A.diagnostic_class(rec(anth("")))["class"], "missing")          # empty
        self.assertEqual(A.diagnostic_class(rec(anth(None)))["class"], "missing")        # no text blocks
        self.assertEqual(A.diagnostic_class({"status": "malformed", "response": "x", "provider": "anthropic"})["class"], "missing")

    def test_visible_text_ignores_thinking_and_covers_other_providers(self):
        resp = {"type": "message", "stop_reason": "end_turn", "content": [{"type": "thinking", "thinking": "A"}, {"type": "text", "text": "words\nD"}]}
        self.assertEqual(A.extract_final_letter(A.visible_text(resp))["letter"], "D")
        oa = {"choices": [{"finish_reason": "stop", "message": {"content": "why\n**B**"}}]}
        self.assertEqual(A.diagnostic_class(rec(oa, "openai")), {"class": "format_recovered", "flr_letter": "B"})
        ge = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "hidden\nA", "thought": True}, {"text": "prose\nC"}]}}]}
        self.assertEqual(A.diagnostic_class(rec(ge, "gemini")), {"class": "format_recovered", "flr_letter": "C"})


class SavedPilots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.an = {}
        for name in ("haiku55", "sonnet55"):
            ses = A.read_session(os.path.join(BUNDLE, name))
            assert not ses["problems"], ses["problems"]
            cls.an[name] = (ses, A.analyze(ses["records"], ses["prompts"]))

    def test_haiku_twelve_random_other_refusals(self):
        t = self.an["haiku55"][1]["by_condition"]["RANDOM_OTHER"]
        self.assertEqual((t["refusals"], t["non_answers"], t["format_invalid_recoverable"], t["valid"]), (12, 12, 0, 20))

    def test_sonnet_sixty_strict_failures_and_competing_diagnostic(self):
        bc = self.an["sonnet55"][1]["by_condition"]
        self.assertEqual(sum(t["non_answers"] for t in bc.values()), 60)
        c = bc["COMPETING"]
        self.assertEqual((c["valid"], c["non_answers"], c["refusals"], c["ambiguous_or_missing"]), (0, 32, 0, 0))     # strict view unchanged
        self.assertEqual((c["format_invalid_recoverable"], c["flr_correct"], c["flr_competing"], c["flr_other_incorrect"]), (32, 2, 30, 0))

    def test_strict_columns_and_status_counts_are_unchanged(self):
        for name, (ses, a) in self.an.items():
            for t in a["by_condition"].values():
                self.assertEqual(t["valid"] + t["non_answers"], t["records"])
                self.assertEqual(t["refusals"] + t["format_invalid_recoverable"] + t["ambiguous_or_missing"], t["non_answers"])
        self.assertEqual(self.an["sonnet55"][1]["status_counts"], {"ok": 100, "invalid": 60})
        self.assertEqual(self.an["haiku55"][1]["status_counts"], {"ok": 147, "blocked": 12, "invalid": 1})

    def test_diagnostic_does_not_mutate_records_or_invent_numbers(self):
        ses = self.an["sonnet55"][0]
        before = copy.deepcopy(ses["records"])
        for r in ses["records"]:
            A.diagnostic_class(r)
        self.assertEqual(before, ses["records"])
        for k in self.an["sonnet55"][1]:
            self.assertNotIn("probab", k.lower().replace("probability fields", ""))

    def test_reports_show_the_split(self):
        ses, a = self.an["sonnet55"]
        text = "\n".join(A.report_lines(a, "x"))
        self.assertIn("FLR-v2", text)
        self.assertIn("correct 2 / competing 30 / other 0", text)
        block = {"measurement_type": A.ANSWER_ONLY, "note": A.NOTE, "excluded": [], "comparison": [], "comparison_note": "",
                 "sessions": [{"id": "s", "model": {"id": "m"}, "provider": "anthropic", "session_dir": "d", "kind": "session", "warnings": [], "analysis": a}]}
        md = "\n".join(A.atlas_markdown(block))
        self.assertIn("Explicit refusals / blocks", md)
        self.assertIn("| COMPETING | 32 | 0 | 32 | 0 | 2 | 30 | 0 |", md)
        rows = A.answer_only_rows(block["sessions"] and [{"id": "s", "model": {"id": "m"}, "analysis": a}])
        comp = [r for r in rows if r["condition"] == "COMPETING" and r["mapping"] == "all"][0]
        self.assertEqual((comp["non_answers"], comp["flr_v2_correct"], comp["flr_v2_competing"]), (32, 2, 30))


if __name__ == "__main__":
    unittest.main()
