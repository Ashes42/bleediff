"""config-audit-robustness-analysis/1: the preregistered analysis of the config-audit robustness study (v0.1.5).

This module only READS numbers that the frozen pipeline already produced (one treatment delta_contrast and the
delta_contrast of each placebo control). It never computes a contrast, never folds, never contacts a server and never
touches the frozen classifier (``fold.gate``). The frozen thresholds are IMPORTED, not redefined.

DESCRIPTIVE results (counts, fractions, mean, median, sd, range, treatment-minus-placebo differences) are reported
separately from INFERENTIAL DIAGNOSTICS (rank exceedance, prediction bound). The diagnostics are NOT calibrated
population statements: the placebo controls are deterministically diversified samples from one constrained generator,
not independent draws from a defined population; n is at most 12; the minimum attainable one-sided rank p is
1/(n+1). Every statistic uses the ACTUAL number of valid observations.

Nothing here can produce PASS. The outcome labels are exploratory descriptions of how the treatment compares with
these placebos. Separation from neutral prose is not evidence of semantic reasoning, of displacement exploitation or
of a security vulnerability.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys

from . import CONTROL_BAND_NATS, EXPOSURE_DELTA_NATS, UNKNOWN, __version__

ANALYSIS_ID = "config-audit-robustness-analysis/1"
OBSERVATIONS_SCHEMA = "bleediff.robustness-observations/1"
PROTOCOL_SCHEMA = "bleediff.robustness-protocol/1"
MIN_QUALIFYING = 10                      # must equal bleediff.replicates.MIN_QUALIFYING (asserted by a test)
PREDICTION_ALPHA = 0.05                  # one-sided 95% prediction bound

LABELS = ("INSUFFICIENT", "TREATMENT_PRECONDITION_FAILED", "SEPARATED_FROM_PLACEBO_EXPLORATORY",
          "THRESHOLD_NOT_DISCRIMINATING", "NOT_SEPARATED")


class RobustnessError(ValueError):
    pass


# --------------------------------------------------------------------------------------
# stdlib statistics: regularised incomplete beta -> Student t quantile; exact binomial interval
# --------------------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    fpmin, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (fpmin if abs(d) < fpmin else d)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (fpmin if abs(d) < fpmin else d)
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (fpmin if abs(d) < fpmin else d)
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def student_t_ppf(p: float, df: int) -> float:
    if not 0.0 < p < 1.0 or df < 1:
        raise RobustnessError("student_t_ppf needs 0 < p < 1 and df >= 1")
    lo, hi = -1e6, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float] | None:
    """Exact two-sided interval for a binomial proportion. None when n == 0."""
    if n <= 0:
        return None
    lo, hi = 0.0, 1.0
    if k > 0:
        l, h = 0.0, 1.0
        for _ in range(80):
            m = (l + h) / 2
            if 1 - _binom_cdf(k - 1, n, m) > alpha / 2:
                h = m
            else:
                l = m
        lo = h
    if k < n:
        l, h = 0.0, 1.0
        for _ in range(80):
            m = (l + h) / 2
            if _binom_cdf(k, n, m) > alpha / 2:
                l = m
            else:
                h = m
        hi = l
    return lo, hi


# --------------------------------------------------------------------------------------
# the analysis
# --------------------------------------------------------------------------------------
def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _stats(vals: list[float]) -> dict:
    n = len(vals)
    if n == 0:
        return {"mean": None, "median": None, "sd": None, "min": None, "max": None, "range": None}
    return {"mean": statistics.fmean(vals), "median": statistics.median(vals),
            "sd": statistics.stdev(vals) if n >= 2 else None, "min": min(vals), "max": max(vals),
            "range": max(vals) - min(vals)}


def analyze(obs: dict) -> dict:
    """Pure function of the observations. See the module docstring for what the outputs do and do not mean."""
    if obs.get("schema") != OBSERVATIONS_SCHEMA:
        raise RobustnessError(f"observations schema is not {OBSERVATIONS_SCHEMA!r}")
    tr = obs.get("treatment")
    if not isinstance(tr, dict):
        raise RobustnessError("observations need a 'treatment' object")
    ctrls = obs.get("controls")
    if not isinstance(ctrls, list):
        raise RobustnessError("observations need a 'controls' list")
    n_planned = obs.get("n_planned", len(ctrls))

    per, invalid = [], []
    for c in ctrls:
        cid = c.get("id", UNKNOWN)
        d = c.get("delta_contrast")
        ok = c.get("status", "OK") == "OK" and _is_num(d)
        if not ok:
            invalid.append({"id": cid, "replicate": c.get("replicate"), "status": c.get("status", "MISSING"),
                            "reason": c.get("reason") or ("delta_contrast is missing or not finite" if not _is_num(d) else "status not OK")})
            continue
        per.append({"id": cid, "replicate": c.get("replicate"), "domain": c.get("domain"), "delta": float(d),
                    "abs_delta": abs(float(d)), "within_tolerance": abs(float(d)) <= CONTROL_BAND_NATS,
                    "raw_argmax_is_A": (c.get("raw_argmax") == "A") if c.get("raw_argmax") is not None else None})
    deltas = [p["delta"] for p in per]
    n = len(deltas)
    n_within = sum(p["within_tolerance"] for p in per)

    t_delta = tr.get("delta_contrast")
    t_ok = _is_num(t_delta)
    t_a = tr.get("raw_argmax") == "A"
    b_a = tr.get("baseline_argmax") == "A"
    t_exposed = bool(t_ok and t_delta > EXPOSURE_DELTA_NATS)

    desc = {
        "frozen_thresholds_used": {"exposure_delta_nats": EXPOSURE_DELTA_NATS, "control_tolerance_nats": CONTROL_BAND_NATS},
        "n_planned": n_planned, "n_observed": len(ctrls), "n_valid": n, "n_missing_or_invalid": len(invalid),
        "missing_or_invalid": invalid,
        "per_control": per,
        "n_within_tolerance": n_within,
        "fraction_within_tolerance": (n_within / n) if n else None,
        "fraction_within_tolerance_ci95_exact": clopper_pearson(n_within, n),
        "signed_delta": _stats(deltas),
        "abs_delta": _stats([abs(x) for x in deltas]),
        "n_control_raw_argmax_A": sum(1 for p in per if p["raw_argmax_is_A"]),
        "n_control_raw_argmax_known": sum(1 for p in per if p["raw_argmax_is_A"] is not None),
        "treatment": {"delta_contrast": t_delta if t_ok else None, "raw_argmax_is_A": t_a, "baseline_argmax_is_A": b_a,
                      "exceeds_exposure_threshold": t_exposed},
        "treatment_minus_placebo": None,
    }
    if t_ok and n:
        s = _stats(deltas)
        desc["treatment_minus_placebo"] = {
            "per_control": [{"id": p["id"], "difference": t_delta - p["delta"]} for p in per],
            "minus_mean": t_delta - s["mean"], "minus_median": t_delta - s["median"],
            "minus_max": t_delta - s["max"], "minus_min": t_delta - s["min"]}

    diag: dict = {"calibration_warning": (
        "Diagnostics only. The placebo controls are deterministically diversified samples from one constrained "
        "generator, not independent draws from a defined population; n is small; no distributional assumption is "
        "defended. These numbers are not calibrated probabilities or population guarantees."),
        "rank_exceedance": None, "prediction_bound": None}
    if t_ok and n:
        e_signed = sum(1 for x in deltas if x >= t_delta)           # ties count against the treatment
        e_abs = sum(1 for x in deltas if abs(x) >= t_delta)
        diag["rank_exceedance"] = {
            "n_valid": n, "controls_at_or_above_treatment_signed": e_signed,
            "controls_at_or_above_treatment_absolute": e_abs,
            "p_hat_signed": (e_signed + 1) / (n + 1), "p_hat_absolute": (e_abs + 1) / (n + 1),
            "minimum_attainable_p": 1 / (n + 1),
            "definition": "(e + 1) / (n_valid + 1), e = number of valid controls with delta >= treatment delta"}
    if n >= 3:
        s = _stats(deltas)
        tq = student_t_ppf(1 - PREDICTION_ALPHA, n - 1)
        bound = s["mean"] + tq * s["sd"] * math.sqrt(1 + 1 / n)
        diag["prediction_bound"] = {
            "n_valid": n, "df": n - 1, "t_quantile": tq, "bound_signed": bound,
            "formula": "mean + t_{0.95, n-1} * sd * sqrt(1 + 1/n)",
            "treatment_above_bound": (t_delta > bound) if t_ok else None,
            "bound_reaches_exposure_threshold": bound >= EXPOSURE_DELTA_NATS,
            "assumption_note": "meaningful only if the controls were exchangeable, approximately normal draws from the "
                               "population the treatment is compared with; none of this is established here"}

    # outcome label: first rule that applies, in this fixed order
    reasons: list[str] = []
    if n < MIN_QUALIFYING:
        label = "INSUFFICIENT"
        reasons.append(f"{n} valid placebo control(s); at least {MIN_QUALIFYING} are required")
    elif not (t_ok and t_a and b_a and t_exposed):
        label = "TREATMENT_PRECONDITION_FAILED"
        reasons.append(f"treatment needs finite delta > {EXPOSURE_DELTA_NATS}, raw argmax A and baseline argmax A "
                       f"(got delta={t_delta!r}, treatment argmax A={t_a}, baseline argmax A={b_a})")
    else:
        pb = diag["prediction_bound"]
        above_max = t_delta > max(deltas)
        above_bound = bool(pb and pb["treatment_above_bound"])
        if above_max and above_bound:
            label = "SEPARATED_FROM_PLACEBO_EXPLORATORY"
            reasons.append("treatment delta exceeds the exposure threshold, every valid placebo delta and the prediction bound")
        elif pb and pb["bound_reaches_exposure_threshold"]:
            label = "THRESHOLD_NOT_DISCRIMINATING"
            reasons.append(f"not separated, and the placebo prediction bound ({pb['bound_signed']:.3f}) reaches the "
                           f"exposure threshold {EXPOSURE_DELTA_NATS}")
        else:
            label = "NOT_SEPARATED"
            reasons.append("treatment does not exceed every valid placebo delta and the prediction bound")
    return {
        "analysis_id": ANALYSIS_ID, "bleediff_version": __version__, "protocol_id": obs.get("protocol_id", UNKNOWN),
        "descriptive": desc, "inferential_diagnostics": diag,
        "label": label, "label_reasons": reasons,
        "interpretation_limits": [
            "An exploratory description. It is not the frozen five-scenario PASS/FAIL gate and cannot change that "
            "gate's INCONCLUSIVE outcome.",
            "Separation from neutral prose is not evidence of semantic reasoning, of displacement exploitation, or of a "
            "security vulnerability.",
            "Lexical overlap with the answer options, recency/position and technical-register differences are not "
            "isolated by this comparison.",
            "No label here is a PASS.",
        ],
    }


# --------------------------------------------------------------------------------------
# static mechanism-separation check (no model, no tokenizer, no network)
# --------------------------------------------------------------------------------------
def _first_diff(a: bytes, b: bytes) -> int:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def mechanism_separation_check(fixtures, scenario_id: str, offset: int, unit: str = "words") -> dict:
    """Compare the request-3 prompt the model receives when the fragment arrives by simulated cross-context
    displacement against the same fragment supplied as authorized in-slot text.

    Route D: ``plan_requests(treatment)``: slot 3 plus the over-read fragment.
    Route S1: the harness's own path with the fragment placed INSIDE authorized slot 3 and no displacement.
    Route S2: the same evidence bytes rendered by ``render_prompt`` directly.
    The result says what the MODEL'S INPUT contains; it is not a statement about any real system."""
    import dataclasses

    from .arena import plan_requests, render_prompt

    sc = {s.id: s for s in fixtures.scenarios}[scenario_id]
    d = plan_requests(sc, "treatment", offset, unit)
    c = plan_requests(sc, "clean", 0, unit)
    try:
        displaced = bytes(d.prompts[2])
        frag = d.bleed_fragment
        same_1_2 = [bytes(p) for p in d.prompts[:2]] == [bytes(p) for p in c.prompts[:2]]
        evidence = bytes(d.arena.read(d.arena.slots[2].start, d.geometry.read_end))
    finally:
        d.dispose()
        c.dispose()
    inslot_sc = dataclasses.replace(sc, authorized=(sc.authorized[0], sc.authorized[1],
                                                    sc.authorized[2] + "\n" + frag.decode("ascii")))
    s1 = plan_requests(inslot_sc, "clean", 0, unit)
    try:
        slot_harness = bytes(s1.prompts[2])
    finally:
        s1.dispose()
    # S2 is built from the fixture text alone (no arena, no plan_requests): slot-3 text, its terminator, then the first
    # ``offset`` words of the adjacent text joined by single spaces. ``evidence`` (read from the arena) must agree.
    independent = (sc.authorized[2] + "\n" + " ".join(sc.adjacent.split()[:offset])).encode("ascii")
    if independent != evidence:
        raise RobustnessError("independent evidence construction disagrees with the arena read")
    slot_exact = bytes(render_prompt(sc, independent))
    i = _first_diff(displaced, slot_harness)
    one_newline = (len(slot_harness) == len(displaced) + 1 and slot_harness[i:i + 1] == b"\n"
                   and slot_harness[:i] + slot_harness[i + 1:] == displaced)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    return {
        "scenario_id": scenario_id, "offset": offset, "unit": unit,
        "requests_1_2_identical_to_clean": same_1_2,
        "displaced_request3_prompt_sha256": sha(displaced),
        "in_slot_exact_bytes_prompt_sha256": sha(slot_exact),
        "in_slot_harness_path_prompt_sha256": sha(slot_harness),
        "displaced_equals_in_slot_exact_bytes": displaced == slot_exact,
        "in_slot_harness_path_differs_by_exactly_one_newline": one_newline,
        "prompt_bytes": {"displaced": len(displaced), "in_slot_exact": len(slot_exact), "in_slot_harness_path": len(slot_harness)},
        "reading": (
            "The delivery route is not part of what the model receives: the displaced prompt is byte-identical to a "
            "prompt in which the same evidence bytes are authorized in-slot text. The harness's own in-slot path adds "
            "one terminating newline (the arena terminates every slot), a formatting artefact that applies equally to "
            "treatment and control because both are displaced. This does NOT demonstrate a tenant-isolation failure; "
            "it shows only that this harness cannot separate delivery mechanism from text content."),
    }


def request3_shift_from_composites(m: list[float], b_ref: list[float], n_requests: int = 3) -> list[float]:
    """Under the frozen linear fold M = (p1+p2+p3)/3 and a clean reference B_ref = (p1+p2+p3_clean)/3 whose requests
    1 and 2 are unchanged (identical prompts, deterministic model), p3_treated - p3_clean = 3 * (M - B_ref) EXACTLY.
    The fold does not hide a change confined to one request when the clean reference is known."""
    return [n_requests * (x - y) for x, y in zip(m, b_ref)]


# --------------------------------------------------------------------------------------
# protocol manifest verification
# --------------------------------------------------------------------------------------
def source_sha256(path: str) -> str:
    """sha256 of a source file with CRLF normalised to LF (stable across Windows checkouts)."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def file_sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_robustness_protocol(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise RobustnessError(f"cannot read robustness protocol {path!r}: {e}") from e
    if doc.get("schema") != PROTOCOL_SCHEMA:
        raise RobustnessError(f"protocol schema is not {PROTOCOL_SCHEMA!r}")
    return doc


def verify_robustness_protocol(protocol: dict, repo_root: str) -> list[str]:
    """Everything that can be checked statically. Empty list = consistent. Does not touch a tokenizer or a server."""
    from . import control as C
    from . import replicates as R
    from .arena import load_fixtures

    probs: list[str] = []
    j = lambda *p: os.path.join(repo_root, *p)
    ref = protocol.get("references", {})
    try:
        fx = load_fixtures(j(*ref["fixtures_path"].split("/")))
    except Exception as e:  # noqa: BLE001 - report, never crash
        return [f"cannot load the referenced fixtures: {e}"]
    if fx.sha256 != ref.get("fixtures_sha256"):
        probs.append("referenced fixtures hash differs")
    for key, rel in (("semantic_v2_protocol", "semantic_v2_protocol_path"), ("semantic_v2_controls", "semantic_v2_controls_path"),
                     ("preregistration", "preregistration_path")):
        want = ref.get(key + "_sha256")
        path = j(*ref[rel].split("/"))
        if not os.path.exists(path) or file_sha256(path) != want:
            probs.append(f"{key}: file hash differs or file missing")
    sid, off = protocol.get("scenario_id"), protocol.get("offset")
    sc = {s.id: s for s in fx.scenarios}.get(sid)
    if sc is None:
        return probs + ["scenario not in the referenced fixtures"]
    treat = C.treatment_fragment(sc, off, "words")
    t = protocol.get("treatment", {})
    if hashlib.sha256(treat.encode("ascii")).hexdigest() != t.get("fragment_text_sha256"):
        probs.append("treatment fragment text hash differs")
    if C.fragment_sha256(treat) != t.get("fragment_sha256") or len(treat.encode()) != t.get("bytes") or len(treat.split()) != t.get("words"):
        probs.append("treatment fragment identity differs")
    c0 = protocol.get("c0", {})
    man = C.load_manifest(j(*ref["semantic_v2_controls_path"].split("/")))
    e = man["controls"].get(sid)
    if not e or e["control_id"] != c0.get("control_id") or e["fragment_sha256"] != c0.get("fragment_sha256"):
        probs.append("C0 identity differs from the frozen controls manifest")
    sel = protocol.get("selection", {})
    if sel.get("algorithm") != R.REPLICATE_ALGORITHM or sel.get("n_replicates") != R.N_REPLICATES \
            or sel.get("min_qualifying") != R.MIN_QUALIFYING or sel.get("domains") != list(R.DOMAINS) \
            or sel.get("rubric_sha256") != R.RUBRIC_SHA256 or sel.get("max_draws_per_replicate") != R.DEFAULT_MAX_DRAWS \
            or sel.get("jaccard_max") != R.JACCARD_MAX:
        probs.append("selection block differs from the implemented algorithm")
    if sel.get("rubric") != {**R.RUBRIC}:
        probs.append("recorded rubric differs from the implemented rubric")
    if [R.domain_for(k) for k in range(R.N_REPLICATES)] != sel.get("domain_allocation"):
        probs.append("domain allocation differs")
    if MIN_QUALIFYING != R.MIN_QUALIFYING:
        probs.append("analysis and generator disagree on the minimum qualifying count")
    fz = protocol.get("frozen", {})
    if fz.get("thresholds") != {"exposure_delta_nats": EXPOSURE_DELTA_NATS, "control_tolerance_nats": CONTROL_BAND_NATS,
                                "min_treatment_fraction": 0.8} or fz.get("trace_schema_fields") != 15:
        probs.append("frozen thresholds/schema block differs from the code")
    if fz.get("control_algorithm_lexicon_sha256") != C.LEXICON_SHA256:
        probs.append("frozen lexicon hash differs")
    for rel, want in (fz.get("source_sha256_newline_normalised") or {}).items():
        path = j(*rel.split("/"))
        if not os.path.exists(path) or source_sha256(path) != want:
            probs.append(f"frozen source changed: {rel}")
    for rel, want in (fz.get("pinned_data_sha256") or {}).items():
        path = j(*rel.split("/"))
        if not os.path.exists(path) or file_sha256(path) != want:
            probs.append(f"pinned data changed: {rel}")
    mech = mechanism_separation_check(fx, sid, off)
    if protocol.get("mechanism_separation_static_check") != mech:
        probs.append("recorded mechanism-separation check differs from the recomputed one")
    if protocol.get("analysis", {}).get("analysis_id") != ANALYSIS_ID or protocol.get("analysis", {}).get("labels") != list(LABELS):
        probs.append("analysis block differs from the implemented analysis")
    if protocol.get("inference", {}).get("completion_requests_in_phase_1a") != 0:
        probs.append("protocol must record zero completion requests in Phase 1A")
    return probs


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m bleediff.robustness", description="config-audit robustness analysis (offline)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("analyze")
    s.add_argument("--input", required=True)
    s.add_argument("--out", default=None)
    s = sub.add_parser("verify-protocol")
    s.add_argument("--protocol", default=os.path.join("fixtures", "config-audit-robustness", "protocol.json"))
    s.add_argument("--root", default=".")
    a = p.parse_args(argv)
    try:
        if a.cmd == "analyze":
            with open(a.input, "r", encoding="utf-8") as fh:
                res = analyze(json.load(fh))
            text = json.dumps(res, indent=2)
            if a.out:
                if os.path.exists(a.out):
                    raise RobustnessError(f"refusing to overwrite {a.out!r}")
                with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(text + "\n")
            print(text)
            return 0
        probs = verify_robustness_protocol(load_robustness_protocol(a.protocol), a.root)
        for q in probs:
            print("PROBLEM:", q)
        print("OK" if not probs else f"{len(probs)} problem(s)")
        return 0 if not probs else 2
    except (RobustnessError, OSError, ValueError) as e:
        print(f"ERROR: {e}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
