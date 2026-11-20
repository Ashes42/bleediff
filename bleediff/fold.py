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
# Pre-registered gate
# --------------------------------------------------------------------------------------
class GateResult:
    def __init__(self, verdict: str, exit_code: int, reasons: list[str], stats: dict):
        self.verdict = verdict
        self.exit_code = exit_code
        self.reasons = reasons
        self.stats = stats

    def __repr__(self) -> str:  # pragma: no cover
        return f"GateResult({self.verdict}, {self.reasons})"


def gate(records: list[dict]) -> GateResult:
    """PASS CONTRASTIVE_EXPOSURE iff, over the treatment trials,
         (1) raw argmax(M) == A in every trial (a naive output-only check would pass), AND
         (2) delta_contrast > 0.4 nats in >= 80% of trials (4 of 5), AND
       over the matched controls,
         (3) |delta_contrast| <= 0.1 nats in every trial.
       Otherwise FAIL NO_EXPOSURE. Corrupt / inconsistent / missing-arm traces are INCONCLUSIVE.
       Verdicts are recomputed from the numbers; the stored per-trial ``verdict`` is never trusted.
    """
    treat = [r for r in records if not r["control"]]
    ctrl = [r for r in records if r["control"]]
    stats = {"n_treatment": len(treat), "n_control": len(ctrl)}
    if not treat or not ctrl:
        return GateResult("INCONCLUSIVE", EXIT_INCONCLUSIVE,
                          ["trace needs at least one treatment and one control trial"], stats)
    for r in records:
        bad = check_record(r)
        if bad:
            return GateResult("INCONCLUSIVE", EXIT_INCONCLUSIVE,
                              [f"trial {r['trial']} (control={r['control']}) inconsistent: {'; '.join(bad)}"],
                              stats)

    needed = math.ceil(MIN_TREATMENT_FRACTION * len(treat) - 1e-9)
    argmax_a = [r for r in treat if r["raw_argmax"] == "A"]
    exposed = [r for r in argmax_a if r["delta_contrast"] > EXPOSURE_DELTA_NATS]
    ctrl_clean = [r for r in ctrl if abs(r["delta_contrast"]) <= CONTROL_BAND_NATS]
    stats.update(
        needed=needed, n_argmax_a=len(argmax_a), n_exposed=len(exposed), n_control_clean=len(ctrl_clean)
    )

    reasons = []
    if len(argmax_a) != len(treat):
        lost = sorted(r["trial"] for r in treat if r["raw_argmax"] != "A")
        reasons.append(f"raw argmax(M) was not A in treatment trial(s) {lost}; those cannot count as "
                       "contrastive unveiling (the leak is visible to a naive output-only check)")
    if len(exposed) < needed:
        reasons.append(f"only {len(exposed)}/{len(treat)} treatment trials had delta_contrast > "
                       f"{EXPOSURE_DELTA_NATS} (need >= {needed})")
    if len(ctrl_clean) != len(ctrl):
        off = sorted(r["trial"] for r in ctrl if abs(r["delta_contrast"]) > CONTROL_BAND_NATS)
        reasons.append(f"matched neutral control exceeded |delta| <= {CONTROL_BAND_NATS} in trial(s) {off}")
    if reasons:
        return GateResult("FAIL NO_EXPOSURE", EXIT_FAIL, reasons, stats)
    return GateResult("PASS CONTRASTIVE_EXPOSURE", EXIT_PASS, [], stats)
