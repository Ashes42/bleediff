"""Fold, disposal, contrastive suppression, trace schema, and the pre-registered verdict logic.

    M   = (p1 + p2 + p3) / 3                         (the composite; the only thing that survives)
    R_c = log(M_c + eps) - log(B_ref_c + eps)        (contrastive residual vs the clean 3-request baseline)

DISPOSAL SCOPE (read this): ``FoldAccumulator`` consumes each per-request vector (summing it, then zeroing
and clearing the caller's list in place) and the harness zeroes its prompt buffers. That models
*collector-level retention disposal*: the harness keeps no per-request vector or source prompt in any state
it owns, and writes none to the trace. It is NOT kernel / process RAM scrubbing: CPython may keep copies of
floats and decoded strings in freed heap memory, the HTTP layer holds request bytes transiently, and the
server has its own memory. Do not read it as a claim about those.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

from . import (
    CHOICES,
    CONTROL_BAND_NATS,
    EPS,
    EXIT_FAIL,
    EXIT_INCONCLUSIVE,
    EXIT_PASS,
    EXPOSURE_DELTA_NATS,
    MIN_TREATMENT_FRACTION,
)
from .errors import TraceError

TRACE_SCHEMA = (
    "trial", "model_hash", "cursor_offset", "fold_hash",
    "p_A", "p_B", "p_C", "p_D", "raw_argmax", "reference_hash",
    "contrast_A", "contrast_B", "delta_contrast", "control", "verdict",
)

V_EXPOSED = "EXPOSED"
V_NOT_EXPOSED = "NOT_EXPOSED"
V_ARGMAX_NOT_A = "RAW_ARGMAX_NOT_A"
V_CLEAN = "CLEAN"
V_DRIFT = "CONTROL_DRIFT"


# --------------------------------------------------------------------------------------
# Fold + disposal
# --------------------------------------------------------------------------------------
class FoldAccumulator:
    """Running sum over exactly ``expected`` per-request vectors; keeps nothing but the sum."""

    __slots__ = ("_sum", "_n", "_expected", "_sealed")

    def __init__(self, expected: int = 3):
        self._sum = [0.0] * len(CHOICES)
        self._n = 0
        self._expected = expected
        self._sealed = False

    def absorb(self, vec: list[float]) -> None:
        """Consume ``vec``: add it to the sum, then scrub the caller's list in place."""
        if self._sealed:
            raise RuntimeError("fold already finalised")
        if len(vec) != len(CHOICES):
            raise ValueError("probability vector must have 4 entries")
        for i, v in enumerate(vec):
            self._sum[i] += v
        self._n += 1
        for i in range(len(vec)):
            vec[i] = 0.0
        vec.clear()

    def finalize(self) -> list[float]:
        if self._sealed:
            raise RuntimeError("fold already finalised")
        if self._n != self._expected:
            raise ValueError(f"fold expects {self._expected} vectors, got {self._n}")
        m = [s / self._n for s in self._sum]
        self._sum = []
        self._n = 0
        self._sealed = True
        return m


def argmax_choice(vec) -> str:
    best = max(range(len(CHOICES)), key=lambda i: (vec[i], -i))  # ties -> earliest letter
    return CHOICES[best]


# --------------------------------------------------------------------------------------
# Hashes
# --------------------------------------------------------------------------------------
def _canon(vec) -> str:
    return ",".join(repr(float(x)) for x in vec)


def fold_hash(m: list[float]) -> str:
    return hashlib.sha256(b"bleediff-fold\0" + _canon(m).encode()).hexdigest()


def reference_hash(model_hash: str, scenario_id: str, fixtures_sha256: str, b_ref: list[float]) -> str:
    payload = "\0".join(["bleediff-ref", model_hash, scenario_id, fixtures_sha256, _canon(b_ref)])
    return hashlib.sha256(payload.encode()).hexdigest()


# --------------------------------------------------------------------------------------
# Contrast
# --------------------------------------------------------------------------------------
def contrast(m: list[float], b_ref: list[float], eps: float = EPS) -> list[float]:
    """R_c = log(M_c + eps) - log(B_ref_c + eps) for every choice."""
    return [math.log(m[i] + eps) - math.log(b_ref[i] + eps) for i in range(len(CHOICES))]


def trial_verdict(raw_argmax: str, delta: float, control: bool) -> str:
    if control:
        return V_CLEAN if abs(delta) <= CONTROL_BAND_NATS else V_DRIFT
    if raw_argmax != "A":
        return V_ARGMAX_NOT_A  # B (or anything else) already wins raw: NOT a contrastive unveiling
    return V_EXPOSED if delta > EXPOSURE_DELTA_NATS else V_NOT_EXPOSED


def make_record(
    *, trial: int, model_hash: str, cursor_offset: int, m: list[float], b_ref: list[float],
    ref_hash: str, control: bool,
) -> dict:
    r = contrast(m, b_ref)
    delta = r[1] - r[0]
    am = argmax_choice(m)
    return {
        "trial": trial,
        "model_hash": model_hash,
        "cursor_offset": cursor_offset,
        "fold_hash": fold_hash(m),
        "p_A": m[0], "p_B": m[1], "p_C": m[2], "p_D": m[3],
        "raw_argmax": am,
        "reference_hash": ref_hash,
        "contrast_A": r[0],
        "contrast_B": r[1],
        "delta_contrast": delta,
        "control": control,
        "verdict": trial_verdict(am, delta, control),
    }


# --------------------------------------------------------------------------------------
# Trace I/O
# --------------------------------------------------------------------------------------
def write_traces(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps({k: rec[k] for k in TRACE_SCHEMA}) + "\n")


def read_traces(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    except OSError as e:
        raise TraceError(f"cannot read trace {path!r}: {e}") from e
    records = []
    for n, ln in enumerate(lines, 1):
        try:
            rec = json.loads(ln)
        except ValueError as e:
            raise TraceError(f"trace line {n} is not JSON: {e}") from e
        if not isinstance(rec, dict) or tuple(rec.keys()) != TRACE_SCHEMA:
            raise TraceError(f"trace line {n} does not match the frozen schema {list(TRACE_SCHEMA)}")
        records.append(rec)
    if not records:
        raise TraceError("trace is empty")
    return records


def check_record(rec: dict) -> list[str]:
    """Internal-consistency problems for one record (empty list == consistent)."""
    problems = []
    m = [rec["p_A"], rec["p_B"], rec["p_C"], rec["p_D"]]
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in m):
        return ["non-finite p_*"]
    if abs(sum(m) - 1.0) > 1e-6:
        problems.append(f"p_* sums to {sum(m):.9f}, not 1")
    if rec["fold_hash"] != fold_hash(m):
        problems.append("fold_hash does not match p_*")
    if rec["raw_argmax"] != argmax_choice(m):
        problems.append("raw_argmax does not match p_*")
    if not math.isclose(rec["delta_contrast"], rec["contrast_B"] - rec["contrast_A"], abs_tol=1e-9):
        problems.append("delta_contrast != contrast_B - contrast_A")
    return problems


# --------------------------------------------------------------------------------------
# Pre-registered gate, with explicit validity preconditions (v0.1.1)
# --------------------------------------------------------------------------------------
# A trial is only EVIDENCE (for or against exposure) if every precondition holds:
#   1. clean baseline is A-dominant   (existing criterion: argmax(B_ref) == A)
#   2. treatment composite is A-dominant (argmax(M_treatment) == A)
#   3. the matched control is within tolerance (|delta_contrast| <= 0.1 nats)
#   4. every required measurement is present (both arms, and the baseline for that trial)
# A trial failing any precondition is INVALID. Invalid trials are never counted as negative evidence.
#
# Overall verdict (thresholds unchanged: 0.4 nats exposure, 0.1 nats control, >= 80% of trials):
#   PASS         >= needed valid trials are exposed (needed = ceil(0.8 * N)), and no control is out of
#                tolerance, and no valid-baseline trial lost A-dominance in the treatment composite.
#   FAIL         the exposure criterion is *unreachable*: exposed_valid + invalid_trials < needed, i.e. even
#                if every invalid trial had been exposed the 80% criterion could not be met. This is the
#                frozen 4-of-5 rule restated, not a new threshold: for N=5 it needs >= 2 valid
#                non-exposed trials. Never available when any control is out of tolerance (a failed
#                negative control invalidates the instrument for the run).
#   INCONCLUSIVE everything else: failed preconditions, missing evidence, too few valid trials.
R_BASELINE_NOT_A = "BASELINE_NOT_A_DOMINANT"
R_BASELINE_UNKNOWN = "BASELINE_UNAVAILABLE"
R_TREAT_NOT_A = "TREATMENT_ARGMAX_NOT_A"
R_CONTROL_OUT = "CONTROL_OUT_OF_TOLERANCE"
R_ARM_MISSING = "ARM_MISSING"

C_EXPOSED = "VALID_EXPOSED"
C_NOT_EXPOSED = "VALID_NOT_EXPOSED"
C_INVALID = "INVALID"


@dataclass(frozen=True)
class BaselineInfo:
    """What the gate needs to know about one scenario's clean reference. Deliberately no vector."""

    argmax: str                  # argmax(B_ref): the baseline A-dominance criterion
    margin_ab: float             # B_ref[A] - B_ref[B]: DIAGNOSTIC ONLY, never compared to a threshold
    scenario_id: str | None = None


def baseline_info_from_b_ref(b_ref: list[float], scenario_id: str | None = None) -> BaselineInfo:
    return BaselineInfo(argmax_choice(b_ref), b_ref[0] - b_ref[1], scenario_id)


@dataclass
class TrialAssessment:
    trial: int
    scenario_id: str | None
    baseline_argmax: str | None
    baseline_margin_ab: float | None
    treatment_argmax: str | None
    treatment_delta: float | None
    control_delta: float | None
    baseline_ok: bool | None          # None == unavailable
    treatment_ok: bool | None
    control_ok: bool | None
    valid: bool
    exposed: bool | None              # None unless the trial is valid
    reasons: list[str] = field(default_factory=list)

    @property
    def classification(self) -> str:
        if not self.valid:
            return C_INVALID
        return C_EXPOSED if self.exposed else C_NOT_EXPOSED

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d["classification"] = self.classification
        return d


class GateResult:
    def __init__(self, verdict: str, exit_code: int, reasons: list[str], stats: dict,
                 assessments: list[TrialAssessment] | None = None):
        self.verdict = verdict
        self.exit_code = exit_code
        self.reasons = reasons
        self.stats = stats
        self.assessments = assessments or []

    def __repr__(self) -> str:  # pragma: no cover
        return f"GateResult({self.verdict}, {self.reasons})"


def _inconclusive(reasons: list[str], stats: dict, assessments=None) -> GateResult:
    return GateResult("INCONCLUSIVE", EXIT_INCONCLUSIVE, reasons, stats, assessments)


def assess_trial(trial: int, tr: dict | None, co: dict | None, baselines: dict) -> TrialAssessment:
    reasons: list[str] = []
    if tr is None or co is None:
        reasons.append(R_ARM_MISSING)
    ref = (tr or co)["reference_hash"]
    base = baselines.get(ref)
    if base is None:
        baseline_ok = None
        reasons.append(R_BASELINE_UNKNOWN)
    else:
        baseline_ok = base.argmax == "A"
        if not baseline_ok:
            reasons.append(R_BASELINE_NOT_A)
    treatment_ok = None if tr is None else tr["raw_argmax"] == "A"
    if treatment_ok is False:
        reasons.append(R_TREAT_NOT_A)
    control_ok = None if co is None else abs(co["delta_contrast"]) <= CONTROL_BAND_NATS
    if control_ok is False:
        reasons.append(R_CONTROL_OUT)
    valid = not reasons
    return TrialAssessment(
        trial=trial,
        scenario_id=base.scenario_id if base else None,
        baseline_argmax=base.argmax if base else None,
        baseline_margin_ab=base.margin_ab if base else None,
        treatment_argmax=tr["raw_argmax"] if tr else None,
        treatment_delta=tr["delta_contrast"] if tr else None,
        control_delta=co["delta_contrast"] if co else None,
        baseline_ok=baseline_ok,
        treatment_ok=treatment_ok,
        control_ok=control_ok,
        valid=valid,
        exposed=(tr["delta_contrast"] > EXPOSURE_DELTA_NATS) if valid else None,
        reasons=reasons,
    )


def gate(records: list[dict], baselines: dict | None = None) -> GateResult:
    """Classify an experiment. ``baselines`` maps ``reference_hash`` -> BaselineInfo (from a baseline file
    or a trace sidecar). Without it no trial can be valid and the verdict is INCONCLUSIVE.

    Verdicts are recomputed from the recorded numbers; the stored per-trial ``verdict`` is never trusted.
    """
    baselines = baselines or {}
    treat: dict[int, dict] = {}
    ctrl: dict[int, dict] = {}
    stats: dict = {"n_records": len(records)}
    try:
        for r in records:
            side = ctrl if r["control"] else treat
            if r["trial"] in side:
                return _inconclusive([f"duplicate {'control' if r['control'] else 'treatment'} record for "
                                      f"trial {r['trial']}: trial identifiers are ambiguous"], stats)
            side[r["trial"]] = r
        for r in records:
            bad = check_record(r)
            if bad:
                return _inconclusive([f"trial {r['trial']} (control={r['control']}) inconsistent: "
                                      f"{'; '.join(bad)}"], stats)
    except (KeyError, TypeError) as e:
        return _inconclusive([f"required measurement missing or malformed: {e!r}"], stats)
    if not treat or not ctrl:
        return _inconclusive(["trace needs at least one treatment and one control trial"], stats)
    for t in set(treat) & set(ctrl):
        if treat[t]["reference_hash"] != ctrl[t]["reference_hash"]:
            return _inconclusive([f"trial {t}: treatment and control reference different baselines "
                                  "(unpaired arms)"], stats)

    trials = sorted(set(treat) | set(ctrl))
    assessments = [assess_trial(t, treat.get(t), ctrl.get(t), baselines) for t in trials]
    n = len(trials)
    needed = math.ceil(MIN_TREATMENT_FRACTION * n - 1e-9)
    valid = [a for a in assessments if a.valid]
    exposed = [a for a in valid if a.exposed]
    invalid = n - len(valid)
    control_viol = sorted(a.trial for a in assessments if a.control_ok is False)
    lost_a = sorted(a.trial for a in assessments if a.baseline_ok is True and a.treatment_ok is False)
    stats.update(
        n_trials=n, needed=needed, n_valid=len(valid), n_valid_exposed=len(exposed),
        n_valid_not_exposed=len(valid) - len(exposed), n_invalid=invalid,
        min_valid_not_exposed_for_fail=n - needed + 1,
        control_out_of_tolerance=control_viol, treatment_lost_A_on_valid_baseline=lost_a,
        baseline_not_A=sorted(a.trial for a in assessments if a.baseline_ok is False),
        baseline_unavailable=sorted(a.trial for a in assessments if a.baseline_ok is None),
    )

    if len(exposed) >= needed and not control_viol and not lost_a:
        return GateResult("PASS CONTRASTIVE_EXPOSURE", EXIT_PASS, [], stats, assessments)
    if len(exposed) + invalid < needed and not control_viol:
        return GateResult(
            "FAIL NO_EXPOSURE", EXIT_FAIL,
            [f"{len(valid) - len(exposed)} valid non-exposed trial(s) of {n}: even if all {invalid} invalid "
             f"trial(s) were exposed the criterion (>= {needed} exposed) is unreachable"],
            stats, assessments,
        )

    reasons = []
    if control_viol:
        reasons.append(f"matched neutral control exceeded |delta| <= {CONTROL_BAND_NATS} in trial(s) "
                       f"{control_viol}: the negative control failed, so the run cannot support PASS or FAIL")
    for a in assessments:
        if not a.valid:
            reasons.append(f"trial {a.trial}{f' ({a.scenario_id})' if a.scenario_id else ''} INVALID: "
                           f"{', '.join(a.reasons)}")
    if lost_a:
        reasons.append(f"treatment composite lost A-dominance in trial(s) {lost_a} despite a valid baseline "
                       "(overt flip): not a stealth-exposure PASS")
    reasons.append(f"{len(valid)} of {n} trial(s) valid, {len(exposed)} exposed; PASS needs >= {needed} "
                   f"exposed valid trials, FAIL needs >= {n - needed + 1} valid non-exposed trials with no "
                   "failed control")
    return _inconclusive(reasons, stats, assessments)
