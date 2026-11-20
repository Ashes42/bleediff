"""Orchestration: calibrate (clean baseline) and bounce (treatment + matched control folds)."""

from __future__ import annotations

import datetime as _dt
import json

from . import CHOICES, __version__
from .arena import (
    Fixtures,
    RequestPlan,
    make_neutral_filler,
    plan_requests,
    scrub,
)
from .client import LlamaClient, ServerInfo, TokenizationReport
from .errors import BaselineError, Inconclusive
from .fold import FoldAccumulator, argmax_choice, make_record, reference_hash

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
    """Holds ONLY: the client, fixtures, server info, final records, and disposal witnesses.

    It never stores a per-request probability vector or a source prompt. Those exist only inside
    ``fold()`` and are zeroed/cleared before it returns.
    """

    def __init__(self, client: LlamaClient, fixtures: Fixtures):
        self.client = client
        self.fixtures = fixtures
        self.info: ServerInfo | None = None
        self.report: TokenizationReport | None = None
        self.records: list[dict] = []
        self.witnesses: list[dict] = []

    # -- setup ---------------------------------------------------------------------------
    def preflight(self) -> ServerInfo:
        self.report = self.client.verify_tokenization()  # fail closed on ambiguity
        self.info = self.client.server_info()
        return self.info

    # -- the fold ------------------------------------------------------------------------
    def fold(self, plan: RequestPlan) -> list[float]:
        """Issue the 3 requests, fold to M = mean(p1, p2, p3), dispose of the sources."""
        acc = FoldAccumulator(expected=3)
        zeroed = 0
        for buf in plan.prompts:
            vec = self.client.choice_probs(buf.decode("ascii"))
            scrub(buf)
            zeroed += 1
            acc.absorb(vec)  # sums, then zeroes + clears `vec` in place
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
    def calibrate(self, fixtures_path_for_record: str | None = None) -> dict:
        info = self.info or self.preflight()
        scenarios = {}
        warnings = []
        for sc in self.fixtures.scenarios:
            b_ref = self.fold(plan_requests(sc, "clean"))
            ref_am = argmax_choice(b_ref)
            if ref_am != "A":
                warnings.append(
                    f"scenario {sc.id!r}: clean baseline argmax is {ref_am}, not A (fixture does not "
                    "make A dominate on this model)"
                )
            scenarios[sc.id] = {
                "B_ref": b_ref,
                "reference_hash": reference_hash(info.model_hash, sc.id, self.fixtures.sha256, b_ref),
                "ref_argmax": ref_am,
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
            "choices": list(CHOICES),
            "scenarios": scenarios,
            "warnings": warnings,
        }

    # -- bounce --------------------------------------------------------------------------
    def _matched_filler(self, adjacent: str, fragment_words: int, unit: str, offset: int, sc) -> str:
        """Neutral filler whose bleed fragment matches the treatment fragment in bytes AND model tokens."""
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
                return filler
        raise Inconclusive(
            "could not build a neutral control matching the treatment fragment in bytes and tokens "
            f"(bytes {last[0]} vs {last[1]}, tokens {last[2]} vs {last[3]}) for scenario {sc.id!r}",
            reason="CONTROL_LENGTH_MISMATCH",
        )

    def bounce(self, baseline: dict, trials: int, offset: int, unit: str = "words") -> list[dict]:
        info = self.info or self.preflight()
        if baseline["model_hash"] != info.model_hash:
            raise Inconclusive(
                f"model_hash drift: baseline {baseline['model_hash'][:12]}... vs server {info.model_hash[:12]}...",
                reason="MODEL_MISMATCH",
            )
        if baseline["fixtures_sha256"] != self.fixtures.sha256:
            raise BaselineError("fixtures changed since calibration (sha256 mismatch); re-run calibrate")
        scs = self.fixtures.scenarios
        records: list[dict] = []
        for t in range(1, trials + 1):
            sc = scs[(t - 1) % len(scs)]
            ref = baseline["scenarios"].get(sc.id)
            if ref is None:
                raise BaselineError(f"baseline has no reference for scenario {sc.id!r}")
            b_ref = [float(x) for x in ref["B_ref"]]
            for control in (False, True):
                if control:
                    filler = self._matched_filler(sc.adjacent, 0, unit, offset, sc)
                    plan = plan_requests(sc, "control", offset, unit, filler=filler)
                else:
                    plan = plan_requests(sc, "treatment", offset, unit)
                m = self.fold(plan)
                records.append(
                    make_record(
                        trial=t, model_hash=info.model_hash, cursor_offset=offset, m=m, b_ref=b_ref,
                        ref_hash=ref["reference_hash"], control=control,
                    )
                )
        self.records = records
        return records
