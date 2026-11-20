"""Versioned experiment protocols: per-scenario displacement windows (schema ``bleediff.protocol/1``).

A protocol amends ONLY the displacement window (``cursor_offset``) of named scenarios. It never changes the
fixtures, the three-request architecture, the contrast equation, the fold, the four-choice normalisation, the
thresholds or the 15-field trace schema. Historical results stay attributable to the protocol they were
produced under: the v0.1.0/v0.1.1/v0.1.2 runs are the implicit uniform-24-word protocol; a run made with
``bounce --protocol`` records the protocol id and file hash in its provenance sidecar.

WINDOW-SELECTION RULE (pre-registered here; inputs are the fixture text and extraction mechanics ONLY)
------------------------------------------------------------------------------------------------------
For a scenario whose intended B cue (an *anchor*) lies outside the current window, choose
    N = the smallest number of words such that the fragment the real extraction function returns for N
        (a) contains the complete anchor as a whole word, AND
        (b) ends on a word carrying clause-terminal punctuation ('.', ',', ';', ':'),
        so that the clause which carries the cue is complete and its meaning is preserved.
Condition (a) alone would stop on the cue word itself (e.g. "...as a major"), a dangling modifier. The rule is
evaluated with ``control.treatment_fragment``, i.e. through ``arena.plan_requests``, so it is the extraction
mechanics, not a re-implementation of them, that decides. No model output is read or produced.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

from .arena import Fixtures, Scenario, load_fixtures
from .control import treatment_fragment
from .errors import FixtureError, Inconclusive

PROTOCOL_SCHEMA = "bleediff.protocol/1"
CLAUSE_TERMINALS = ".,;:"
WINDOW_RULE = ("smallest N whose extracted fragment (a) contains the complete anchor as a whole word and "
               "(b) ends on a word carrying clause-terminal punctuation ('.', ',', ';', ':')")


class ProtocolError(Inconclusive):
    reason = "PROTOCOL_INVALID"


def _whole_word(anchor: str, text: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(anchor) + r"(?![A-Za-z0-9])", text, re.I) is not None


def select_window(sc: Scenario, anchor: str, unit: str = "words") -> dict:
    """Apply WINDOW_RULE to one scenario. Returns every candidate that was considered, so the choice can be audited."""
    if unit != "words":
        raise ProtocolError("window selection is defined for unit='words' only")
    total = len(sc.adjacent.split())
    frags = {n: treatment_fragment(sc, n, unit) for n in range(1, total + 1)}   # the real extraction function
    including = [n for n, f in frags.items() if _whole_word(anchor, f)]
    if not including:
        raise ProtocolError(f"anchor {anchor!r} does not occur in the adjacent slot of {sc.id!r}")
    first = min(including)
    ok = [n for n in including if frags[n].split()[-1][-1] in CLAUSE_TERMINALS]
    if not ok:
        raise ProtocolError(f"no window of {sc.id!r} ends a clause after the anchor {anchor!r}")
    selected = min(ok)
    considered = [{"offset": n, "last_word": frags[n].split()[-1], "contains_anchor": True,
                   "ends_on_clause_terminal": frags[n].split()[-1][-1] in CLAUSE_TERMINALS}
                  for n in range(first, selected + 1)]
    return {"scenario_id": sc.id, "anchor": anchor, "unit": unit, "rule": WINDOW_RULE,
            "slot_words": total, "anchor_first_included_at_offset": first, "selected_offset": selected,
            "candidates_considered": considered,
            "fragment_sha256": hashlib.sha256(frags[selected].encode("ascii")).hexdigest(),
            "minimal_under_rule": all(n not in ok for n in range(1, selected))}


def file_sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_protocol(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise ProtocolError(f"cannot read protocol {path!r}: {e}") from e
    if doc.get("schema") != PROTOCOL_SCHEMA:
        raise ProtocolError(f"protocol schema is not {PROTOCOL_SCHEMA!r}")
    return doc


def scenario_offsets(protocol: dict, fixtures: Fixtures) -> dict[str, int]:
    """Validate a protocol against the CURRENT fixtures and return {scenario_id: offset}. Fails closed."""
    if protocol.get("fixtures_sha256") != fixtures.sha256:
        raise ProtocolError("protocol was written for different fixtures (sha256 mismatch)")
    if protocol.get("unit") != "words":
        raise ProtocolError("only unit='words' protocols are supported")
    offsets = protocol.get("scenario_offsets")
    ids = [s.id for s in fixtures.scenarios]
    if not isinstance(offsets, dict) or sorted(offsets) != sorted(ids):
        raise ProtocolError("scenario_offsets must name exactly the fixture scenarios")
    for sid, n in offsets.items():
        if not isinstance(n, int) or n < 1:
            raise ProtocolError(f"offset for {sid!r} must be a positive integer")
    return dict(offsets)


def protocol_summary(protocol: dict, path: str) -> dict:
    """What a sidecar records about the protocol (ids and hashes; no fragment text)."""
    return {"protocol_id": protocol.get("protocol_id"), "protocol_file_sha256": file_sha256(path),
            "parent_protocol": (protocol.get("parent_protocol") or {}).get("id"),
            "scenario_offsets": protocol.get("scenario_offsets"),
            "controls_manifest_sha256": protocol.get("controls_manifest_sha256")}
