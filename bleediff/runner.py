"""Orchestration: calibrate (clean baseline) and bounce (treatment + matched control folds)."""

from __future__ import annotations

import datetime as _dt
import json

from . import CHOICES, UNKNOWN, __version__
from .arena import (
    FILLER_ALGORITHM,
    Fixtures,
    RequestPlan,
    make_neutral_filler,
    plan_requests,
    scrub,
)
from .client import LlamaClient, ServerInfo, TokenizationReport
from .control import ALGORITHM as PROSE_ALGORITHM
from .control import ControlMatchError, verify_control
from .errors import BaselineError, Inconclusive
from .fold import FoldAccumulator, argmax_choice, make_record, reference_hash
from .meta import filler_hash, filler_repro_id

FILLER_SALTS = 12  # attempts to find a neutral filler whose token count matches the treatment fragment


def load_baseline(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise BaselineError(f"cannot read baseline {path!r}: {e}") from e
    for key in ("model_hash", "fixtures_sha256", "scenarios"):
        if key not in doc:
            raise BaselineError(f"baseline is missing {key!r}")
    return doc


class Harness:
    """Holds ONLY: the client, fixtures, server info, final records, disposal witnesses, and per-fold
    *aggregate* diagnostics (``meta_records``: scenario id, token counts, coverage mean/min, filler hash).

    It never stores a per-request probability vector, a per-request coverage value, a source prompt, or the
    raw control-filler text. Those exist only inside ``fold()`` / ``bounce()`` and are zeroed, cleared or
    dropped before they return.
    """

    def __init__(self, client: LlamaClient, fixtures: Fixtures):
        self.client = client
        self.fixtures = fixtures
        self.info: ServerInfo | None = None
        self.report: TokenizationReport | None = None
        self.records: list[dict] = []
        self.witnesses: list[dict] = []
        self.meta_records: list[dict] = []
        self._last_diag: dict | None = None
        self.control_algorithm: str = UNKNOWN          # recorded in the sidecar
        self.control_manifest_sha256: str = UNKNOWN
        self.cursor_offsets: dict | None = None         # per-scenario windows when a protocol is applied
        self.protocol_info: dict | None = None          # set by the CLI: ids and hashes only

    # -- setup ---------------------------------------------------------------------------
    def preflight(self) -> ServerInfo:
        self.report = self.client.verify_tokenization()  # fail closed on ambiguity
        self.info = self.client.server_info()
        return self.info

    # -- the fold ------------------------------------------------------------------------
    def fold(self, plan: RequestPlan, measure_tokens: bool = False) -> list[float]:
        """Issue the 3 requests, fold to M = mean(p1, p2, p3), dispose of the sources.

        Side output (``self._last_diag``, aggregates only): whole-prompt token counts (when requested) and
        the choice-surface coverage mean/min over the three requests. Per-request coverage values live in a
        local list that is cleared before return.
        """
        acc = FoldAccumulator(expected=3)
        zeroed = 0
        coverages: list = []
        tokens: list = []
        basis = UNKNOWN
        for buf in plan.prompts:
            prompt = buf.decode("ascii")
            if measure_tokens:
                tokens.append(len(self.client.tokenize(prompt, cache=False)))
            vec = self.client.choice_probs(prompt)
            coverages.append(self.client.last_coverage)
            basis = self.client.last_coverage_basis
            self.client.last_coverage = None  # transient: do not leave a per-request value on the client
            del prompt
            scrub(buf)
            zeroed += 1
            acc.absorb(vec)  # sums, then zeroes + clears `vec` in place
        if coverages and all(isinstance(c, float) for c in coverages):
            cov = {"mean": sum(coverages) / len(coverages), "min": min(coverages),
                   "n_requests": len(coverages), "basis": basis}
        else:
            cov = {"mean": UNKNOWN, "min": UNKNOWN, "n_requests": len(coverages), "basis": basis}
        coverages.clear()
        self._last_diag = {"request_prompt_tokens": tokens if measure_tokens else UNKNOWN, "coverage": cov}
        plan.dispose()
        m = acc.finalize()
        self.witnesses.append(
            {
                "vectors_absorbed_and_zeroed": 3,
                "prompt_buffers_zeroed": zeroed,
                "arena_zeroed": True,
                "scope": "collector-level retention disposal, not kernel RAM scrubbing",
            }
        )
        return m

    # -- calibrate -----------------------------------------------------------------------
    def calibrate(self, fixtures_path_for_record: str | None = None, protocol_info: dict | None = None) -> dict:
        info = self.info or self.preflight()
        scenarios = {}
        warnings = []
        for sc in self.fixtures.scenarios:
            b_ref = self.fold(plan_requests(sc, "clean"))
            ref_am = argmax_choice(b_ref)
            if ref_am != "A":
                warnings.append(
                    f"scenario {sc.id!r}: clean baseline argmax is {ref_am}, not A: trials for this "
                    "scenario will be INVALID (baseline precondition), not negative evidence"
                )
            scenarios[sc.id] = {
                "B_ref": b_ref,
                "reference_hash": reference_hash(info.model_hash, sc.id, self.fixtures.sha256, b_ref),
                "ref_argmax": ref_am,
                # diagnostics only (no threshold is applied to either):
                "ab_margin": b_ref[0] - b_ref[1],
                "coverage": self._last_diag["coverage"] if self._last_diag else UNKNOWN,
            }
        return {
            "bleediff_version": __version__,
            "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "server": info.to_json(),
            "model_hash": info.model_hash,
            "tokenization": self.report.to_json() if self.report else None,
            "n_probs_used": self.client.last_n_probs_used,
            "fixtures_path": fixtures_path_for_record or self.fixtures.path,
            "fixtures_sha256": self.fixtures.sha256,
            # identity of the experiment this calibration belongs to; checked by protocol.check_baseline_identity
            "protocol": ({"protocol_id": protocol_info["protocol_id"],
                          "protocol_file_sha256": protocol_info["protocol_file_sha256"]} if protocol_info else "NONE"),
            "choices": list(CHOICES),
            "scenarios": scenarios,
            "warnings": warnings,
        }

    # -- bounce --------------------------------------------------------------------------
    def _matched_filler(self, adjacent: str, fragment_words: int, unit: str, offset: int, sc) -> tuple[str, int]:
        """Neutral filler whose bleed fragment matches the treatment fragment in bytes AND model tokens.

        Returns ``(filler_text, salt)``; the caller keeps only the salt and the text's SHA-256."""
        count = lambda s: len(self.client.tokenize(s))  # noqa: E731
        treat = plan_requests(sc, "treatment", offset, unit)
        want_bytes = treat.bleed_fragment
        # cache=False: never retain a source fragment as a cache key in harness/client state.
        want_tokens = len(self.client.tokenize("\n" + want_bytes.decode("ascii"), cache=False))
        treat.dispose()
        last = None
        for salt in range(FILLER_SALTS):
            filler = make_neutral_filler(adjacent, count_tokens=count, salt=salt)
            probe = plan_requests(sc, "control", offset, unit, filler=filler)
            frag = probe.bleed_fragment
            probe.dispose()
            got_tokens = len(self.client.tokenize("\n" + frag.decode("ascii"), cache=False))
            last = (len(frag), len(want_bytes), got_tokens, want_tokens)
            if len(frag) == len(want_bytes) and got_tokens == want_tokens:
                return filler, salt
        raise Inconclusive(
            "could not build a neutral control matching the treatment fragment in bytes and tokens "
            f"(bytes {last[0]} vs {last[1]}, tokens {last[2]} vs {last[3]}) for scenario {sc.id!r}",
            reason="CONTROL_LENGTH_MISMATCH",
        )

    def bounce(self, baseline: dict, trials: int, offset: int, unit: str = "words", *,
               control: str = PROSE_ALGORITHM, manifest: dict | None = None,
               manifest_sha256: str | None = None, offsets: dict | None = None) -> list[dict]:
        """Run treatment + matched-control folds. ``control`` selects the control construction:

        * ``neutral-prose/1`` (default, v0.1.2): the PRESERVED fluent controls from ``manifest``; each is re-verified
          against the live tokenizer, the current fixtures and the current lexicon before any request is sent.
        * ``neutral-filler/2``: the historical v0.1.0/v0.1.1 word-substitution filler, kept only so old evidence
          stays reproducible. Its results are NOT comparable with ``neutral-prose/1`` results.

        ``offsets`` (scenario id -> words) applies a protocol amendment: each scenario uses its own displacement
        window and records that window in its trace records' ``cursor_offset``. Without it, ``offset`` is uniform.
        """
        info = self.info or self.preflight()
        if baseline["model_hash"] != info.model_hash:
            raise Inconclusive(
                f"model_hash drift: baseline {baseline['model_hash'][:12]}... vs server {info.model_hash[:12]}...",
                reason="MODEL_MISMATCH",
            )
        if baseline["fixtures_sha256"] != self.fixtures.sha256:
            raise BaselineError("fixtures changed since calibration (sha256 mismatch); re-run calibrate")
        if control not in (PROSE_ALGORITHM, FILLER_ALGORITHM):
            raise Inconclusive(f"unknown control algorithm {control!r}", reason="BAD_ARGS")
        if offsets is not None and sorted(offsets) != sorted(s.id for s in self.fixtures.scenarios):
            raise Inconclusive("offsets must name exactly the fixture scenarios", reason="BAD_ARGS")

        def off_for(sc) -> int:
            return offsets[sc.id] if offsets is not None else offset

        self.cursor_offsets = dict(offsets) if offsets is not None else None
        self.control_algorithm = control
        self.control_manifest_sha256 = (manifest_sha256 or UNKNOWN) if control == PROSE_ALGORITHM else "NOT_APPLICABLE"
        if control == PROSE_ALGORITHM:
            if manifest is None:
                raise ControlMatchError("neutral-prose/1 needs a preserved controls manifest", reason="CONTROL_MANIFEST_INVALID")
            tok = lambda s: self.client.tokenize(s, cache=False)  # noqa: E731
            for sc in self.fixtures.scenarios[:max(1, min(trials, len(self.fixtures.scenarios)))]:
                entry = manifest["controls"].get(sc.id)
                problems = (["no preserved control for this scenario"] if entry is None
                            else verify_control(sc, entry, manifest, off_for(sc), unit, tok))
                if problems:
                    raise ControlMatchError(
                        f"preserved control for {sc.id!r} does not validate: {'; '.join(problems)}",
                        reason="CONTROL_NOT_VALIDATED")
        scs = self.fixtures.scenarios
        records: list[dict] = []
        for t in range(1, trials + 1):
            sc = scs[(t - 1) % len(scs)]
            ref = baseline["scenarios"].get(sc.id)
            if ref is None:
                raise BaselineError(f"baseline has no reference for scenario {sc.id!r}")
            b_ref = [float(x) for x in ref["B_ref"]]
            off = off_for(sc)
            for control_arm in (False, True):
                filler_meta: object = "NOT_APPLICABLE"
                if control_arm and control == PROSE_ALGORITHM:
                    entry = manifest["controls"][sc.id]
                    plan = plan_requests(sc, "control", off, unit, filler=entry["fragment_text"])
                    filler_meta = {
                        "sha256": entry["fragment_sha256"], "algorithm": PROSE_ALGORITHM,
                        "control_id": entry["control_id"], "enumeration_index": entry["enumeration_index"],
                        "salt": "NOT_APPLICABLE", "repro_id": entry["control_id"],
                        "token_parity": entry["token_parity"],
                    }
                elif control_arm:
                    filler, salt = self._matched_filler(sc.adjacent, 0, unit, off, sc)
                    plan = plan_requests(sc, "control", off, unit, filler=filler)
                    filler_meta = {
                        "sha256": filler_hash(filler),
                        "salt": salt,
                        "repro_id": filler_repro_id(
                            model_hash=info.model_hash, fixtures_sha256=self.fixtures.sha256,
                            scenario_id=sc.id, cursor_offset=off, unit=unit, salt=salt,
                            algorithm=FILLER_ALGORITHM),
                        "algorithm": FILLER_ALGORITHM,
                    }
                    del filler  # the raw filler text is never retained, only its hash
                else:
                    plan = plan_requests(sc, "treatment", off, unit)
                m = self.fold(plan, measure_tokens=True)
                rec = make_record(
                    trial=t, model_hash=info.model_hash, cursor_offset=off, m=m, b_ref=b_ref,
                    ref_hash=ref["reference_hash"], control=control_arm,
                )
                records.append(rec)
                diag = self._last_diag or {}
                self.meta_records.append({
                    "trial": t, "control": control_arm, "fold_hash": rec["fold_hash"],
                    "scenario_id": sc.id, "cursor_offset": off,
                    "request_prompt_tokens": diag.get("request_prompt_tokens", UNKNOWN),
                    "coverage": diag.get("coverage", UNKNOWN),
                    "control_filler": filler_meta,
                })
        self.records = records
        return records
