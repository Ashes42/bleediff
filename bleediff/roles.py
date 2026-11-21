"""Semantic answer roles for single-prompt measurements (v0.3.x, exploratory).

An answer-choice permutation moves the same semantic answers to different letters. The A/B margin ``ln(p_A / p_B)`` of the
tracker is a statement about the letters A and B; it describes the correct and the competing answer only when those two answers
happen to occupy A and B. This module scores the SEMANTIC roles instead, wherever their letters are:

* ``correct``   the letter currently holding the correct semantic answer,
* ``competing`` the letter currently holding the competing (designated incorrect) semantic answer.

Everything is derived from the saved normalised A-D probabilities with plain logarithms; nothing is imputed and no model is
contacted. The definitions are:

    margin_correct_vs_competing  = ln(p_correct) - ln(p_competing)
    margin_correct_vs_strongest  = ln(p_correct) - ln(p_strongest_incorrect)   (the strongest of ALL incorrect letters; ties
                                                                                 go to the earliest letter)
    movement_toward_competing    = -(change of margin_correct_vs_competing from the clean baseline)

When the roles are (A correct, B competing) this equals the tracker's existing ``-(change of margin_AB)``; with (B correct,
A competing) it equals ``+(change of margin_AB)``. A *newly induced incorrect answer* requires a CORRECT baseline: a model whose
clean baseline already prefers a wrong answer is never reported as newly misled.
"""

from __future__ import annotations

import math

from . import CHOICES
from .fold import argmax_choice
from .track import MeasurementError, validate_probs

ROLE_NOTE = ("Role margins compare the correct and the competing semantic answers wherever their letters are. The A/B margin "
             "(margin_AB) is a statement about the letters A and B and equals a role margin only when the two roles occupy A and B.")


def validate_roles(correct: str, competing: str) -> None:
    for name, letter in (("correct", correct), ("competing", competing)):
        if letter not in CHOICES:
            raise MeasurementError(f"the {name} answer letter {letter!r} is not one of {list(CHOICES)}")
    if correct == competing:
        raise MeasurementError("the correct and the competing answers must occupy different letters")


def roles_are_ab(correct: str, competing: str) -> bool:
    """True when the role margin is the (possibly sign-flipped) A/B margin."""
    return {correct, competing} == {"A", "B"}


def role_scores(probs, correct: str, competing: str) -> dict:
    """Role view of one normalised A-D probability vector. Raises MeasurementError for invalid input or an undefined margin."""
    validate_roles(correct, competing)
    problems = validate_probs(probs)
    if problems:
        raise MeasurementError("; ".join(problems))
    p = dict(zip(CHOICES, probs))
    incorrect = [c for c in CHOICES if c != correct]
    strongest = max(incorrect, key=lambda c: (p[c], -CHOICES.index(c)))          # ties -> earliest letter
    for name, letter in (("correct", correct), ("competing", competing), ("strongest incorrect", strongest)):
        if not p[letter] > 0:
            raise MeasurementError(f"p({name}) is not positive, so the role margin is undefined")
    preferred = argmax_choice(probs)
    return {"correct_letter": correct, "competing_letter": competing, "p_correct": p[correct], "p_competing": p[competing],
            "strongest_incorrect_letter": strongest, "p_strongest_incorrect": p[strongest], "strongest_incorrect_is_competing": strongest == competing,
            "margin_correct_vs_competing": math.log(p[correct]) - math.log(p[competing]),
            "margin_correct_vs_strongest": math.log(p[correct]) - math.log(p[strongest]),
            "preferred_letter": preferred, "preferred_is_correct": preferred == correct, "preferred_is_competing": preferred == competing,
            "roles_are_ab": roles_are_ab(correct, competing)}


def role_changes(base: dict, cand: dict) -> dict:
    """Changes of a candidate's role scores from the corresponding clean baseline (both from ``role_scores`` with the same roles)."""
    if (base["correct_letter"], base["competing_letter"]) != (cand["correct_letter"], cand["competing_letter"]):
        raise MeasurementError("a candidate must be compared with a baseline that used the same roles")
    d_cc = cand["margin_correct_vs_competing"] - base["margin_correct_vs_competing"]
    d_cs = cand["margin_correct_vs_strongest"] - base["margin_correct_vs_strongest"]
    base_ok = base["preferred_is_correct"]
    induced = base_ok and not cand["preferred_is_correct"]
    return {"baseline_correct": base_ok, "candidate_correct": cand["preferred_is_correct"],
            "change_margin_correct_vs_competing": d_cc, "change_margin_correct_vs_strongest": d_cs, "movement_toward_competing": -d_cc,
            "overall_flip": cand["preferred_letter"] != base["preferred_letter"],
            "role_order_crossed": base["margin_correct_vs_competing"] * cand["margin_correct_vs_competing"] < 0,
            "newly_induced_incorrect": induced,
            "induced_to": (None if not induced else ("competing" if cand["preferred_is_competing"] else "other_incorrect"))}
