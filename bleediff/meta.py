"""Versioned provenance sidecar for a trace (``<trace>.meta.json``, schema ``bleediff.trace-meta/1``).

The 15-field trace schema is frozen and untouched. Everything that makes a trace independently
interpretable but does not belong in that schema lives here instead.

Rules
-----
* Nothing is guessed. Anything unavailable is the explicit string ``UNKNOWN`` (see ``bleediff.UNKNOWN``).
* Nothing here undermines collector-level disposal: no per-request probability vectors, no raw prompts, no
  raw model responses, no raw control-filler text (only its SHA-256 and a salt/reproducibility id) and no
  per-request coverage values (only a per-fold aggregate).
* The sidecar is bound to exactly one trace by ``trace.sha256`` and to its records by
  ``(trial, control, fold_hash)``; any mismatch, duplicate, or orphan record is an error, never a guess.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

from . import UNKNOWN, __version__
from .errors import BaselineError, TraceError
from .fold import BaselineInfo, baseline_info_from_b_ref, reference_hash

SIDECAR_SCHEMA = "bleediff.trace-meta/1"
SIDECAR_SUFFIX = ".meta.json"
NOT_APPLICABLE = "NOT_APPLICABLE"


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def sidecar_path(trace_path: str) -> str:
    return trace_path + SIDECAR_SUFFIX


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def code_revision() -> dict:
    """Git commit of the running checkout plus a dirty flag; UNKNOWN when not a git checkout / no commit."""
    root = Path(__file__).resolve().parent.parent
    unknown = {"git_commit": UNKNOWN, "dirty": UNKNOWN}
    if not (root / ".git").exists():
        return unknown
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        if head.returncode != 0 or len(head.stdout.strip()) < 40:
            return unknown
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=10)
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else UNKNOWN
        return {"git_commit": head.stdout.strip(), "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return unknown


def filler_hash(text: str) -> str:
    """SHA-256 of the control filler text. The text itself is never retained."""
    return hashlib.sha256(b"bleediff-filler\0" + text.encode("ascii")).hexdigest()


def filler_repro_id(*, model_hash: str, fixtures_sha256: str, scenario_id: str, cursor_offset: int,
                    unit: str, salt: int, algorithm: str) -> str:
    """Identifier from which the filler can be regenerated with the same code + tokenizer."""
    payload = json.dumps(
        {"model_hash": model_hash, "fixtures_sha256": fixtures_sha256, "scenario_id": scenario_id,
         "cursor_offset": cursor_offset, "unit": unit, "salt": salt, "algorithm": algorithm},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _baseline_entry(info: BaselineInfo) -> dict:
    return {"scenario_id": info.scenario_id, "argmax": info.argmax, "margin_ab": info.margin_ab,
            "margin_ab_basis": "B_ref[A] - B_ref[B] (diagnostic only; no threshold is applied)"}


# --------------------------------------------------------------------------------------
# baselines (for the validity gate)
# --------------------------------------------------------------------------------------
def baseline_infos_from_baseline(doc: dict) -> dict[str, BaselineInfo]:
    """reference_hash -> BaselineInfo from a baseline.json. The argmax is RECOMPUTED from ``B_ref`` (the
    stored ``ref_argmax`` is not trusted) and each stored reference_hash is re-derived, so a baseline that
    does not match its own hashes is rejected instead of silently used."""
    out: dict[str, BaselineInfo] = {}
    try:
        for sid, s in doc["scenarios"].items():
            b_ref = [float(x) for x in s["B_ref"]]
            want = reference_hash(doc["model_hash"], sid, doc["fixtures_sha256"], b_ref)
            if want != s["reference_hash"]:
                raise BaselineError(f"baseline scenario {sid!r}: reference_hash does not re-derive from "
                                    "model_hash + fixtures_sha256 + B_ref")
            out[s["reference_hash"]] = baseline_info_from_b_ref(b_ref, sid)
    except (KeyError, TypeError, ValueError) as e:
        raise BaselineError(f"baseline is malformed: {e!r}") from e
    return out


def baseline_infos_from_sidecar(doc: dict) -> dict[str, BaselineInfo]:
    out: dict[str, BaselineInfo] = {}
    for ref, b in (doc.get("baselines") or {}).items():
        if b.get("argmax") in (None, UNKNOWN) or not isinstance(b.get("margin_ab"), (int, float)):
            continue  # unknown stays unknown: the gate will report BASELINE_UNAVAILABLE
        out[ref] = BaselineInfo(b["argmax"], float(b["margin_ab"]), b.get("scenario_id"))
    return out


def merge_baseline_infos(a: dict[str, BaselineInfo], b: dict[str, BaselineInfo]) -> dict[str, BaselineInfo]:
    merged = dict(a)
    for ref, info in b.items():
        if ref in merged and (merged[ref].argmax != info.argmax
                              or abs(merged[ref].margin_ab - info.margin_ab) > 1e-9):
            raise TraceError(f"baseline sources disagree for reference_hash {ref[:12]}...")
        merged.setdefault(ref, info)
    return merged


# --------------------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------------------
def build_sidecar(*, trace_path: str, records: list[dict], harness, baseline_doc: dict,
                  baseline_file_sha256: str | None, cursor_offset: int, unit: str) -> dict:
    """Sidecar for a freshly produced trace. ``harness`` supplies per-fold diagnostics (aggregates only)."""
    info = harness.info
    infos = baseline_infos_from_baseline(baseline_doc)
    offs = {mr["cursor_offset"] for mr in harness.meta_records} or {cursor_offset}
    by_key = {(m["trial"], m["control"]): m for m in harness.meta_records}
    recs = []
    for r in records:
        m = by_key.get((r["trial"], r["control"]))
        if m is None:
            raise TraceError(f"no fold diagnostics for trial {r['trial']} control={r['control']}")
        recs.append({
            "trial": r["trial"],
            "control": r["control"],
            "fold_hash": r["fold_hash"],
            "scenario_id": m["scenario_id"],
            "cursor_offset": m.get("cursor_offset", cursor_offset),
            "request_prompt_tokens": m["request_prompt_tokens"],
            "coverage": m["coverage"],
            "control_filler": m["control_filler"] if r["control"] else NOT_APPLICABLE,
        })
    return {
        "schema": SIDECAR_SCHEMA,
        "bleediff_version": __version__,
        "origin": "bounce",
        "trace": {"sha256": file_sha256(trace_path), "n_records": len(records)},
        "experiment": {
            "run_timestamp_utc": utc_now(),
            "model_hash": info.model_hash,
            "model_hash_basis": info.model_hash_basis,
            "fixtures_sha256": harness.fixtures.sha256,
            "baseline_file_sha256": baseline_file_sha256 or UNKNOWN,
            "cursor_offset": (next(iter(offs)) if len(offs) == 1 else "PER_SCENARIO"),
            "cursor_offsets": {mr["scenario_id"]: mr["cursor_offset"] for mr in harness.meta_records},
            "offset_unit": unit,
            "protocol": getattr(harness, "protocol_info", None) or {
                "protocol_id": "NONE", "note": "no protocol file: uniform --offset for every scenario"},
            "server_build": info.build_info or UNKNOWN,
            "server_flags": info.public_flags(),
            "n_probs_requested": harness.client.n_probs,
            "code_revision": code_revision(),
            "control_algorithm": getattr(harness, "control_algorithm", UNKNOWN),
            "control_manifest_sha256": getattr(harness, "control_manifest_sha256", UNKNOWN),
        },
        "baselines": {ref: _baseline_entry(i) for ref, i in infos.items()},
        "coverage_interpretation": (
            "Coverage = original full-vocabulary probability mass on the eight candidate surfaces before "
            "renormalisation. NO threshold is defined: low-coverage interpretation is UNRESOLVED until a "
            "threshold can be justified independently. Only a per-fold aggregate is retained."),
        "records": recs,
    }


def legacy_sidecar(*, trace_path: str, records: list[dict], baseline_doc: dict,
                   baseline_file_sha256: str | None = None) -> dict:
    """In-memory sidecar view for a trace produced before sidecars existed (e.g. the v0.1.0 Qwen run).

    Only facts that are *verifiable from the artifacts* are filled in. The scenario of each record comes from
    joining ``reference_hash`` against a baseline whose hashes were re-derived (a join, not an inference from
    trial order). Everything else is UNKNOWN: unit, server flags, code revision, timestamp, filler, token
    counts, coverage.
    """
    infos = baseline_infos_from_baseline(baseline_doc)
    model_hashes = {r["model_hash"] for r in records}
    offsets = {r["cursor_offset"] for r in records}
    server = baseline_doc.get("server") or {}
    recs = []
    for r in records:
        info = infos.get(r["reference_hash"])
        recs.append({
            "trial": r["trial"], "control": r["control"], "fold_hash": r["fold_hash"],
            "scenario_id": info.scenario_id if info else UNKNOWN,
            "request_prompt_tokens": UNKNOWN, "coverage": UNKNOWN,
            "control_filler": UNKNOWN if r["control"] else NOT_APPLICABLE,
        })
    return {
        "schema": SIDECAR_SCHEMA,
        "bleediff_version": __version__,
        "origin": "legacy-reconstruction",
        "trace": {"sha256": file_sha256(trace_path), "n_records": len(records)},
        "experiment": {
            "run_timestamp_utc": UNKNOWN,
            "model_hash": next(iter(model_hashes)) if len(model_hashes) == 1 else UNKNOWN,
            "model_hash_basis": UNKNOWN,
            "fixtures_sha256": baseline_doc.get("fixtures_sha256", UNKNOWN),
            "baseline_file_sha256": baseline_file_sha256 or UNKNOWN,
            "cursor_offset": next(iter(offsets)) if len(offsets) == 1 else UNKNOWN,
            "offset_unit": UNKNOWN,
            "server_build": UNKNOWN,
            "server_flags": UNKNOWN,
            "n_probs_requested": UNKNOWN,
            "code_revision": UNKNOWN,
            "control_algorithm": UNKNOWN,
            "control_manifest_sha256": UNKNOWN,
            "protocol": UNKNOWN,
        },
        "baseline_file_provenance": {
            "note": "facts about the baseline file only; the trace itself recorded none of these",
            "created_utc": baseline_doc.get("created_utc", UNKNOWN),
            "server_build": server.get("build_info", UNKNOWN),
            "bleediff_version": baseline_doc.get("bleediff_version", UNKNOWN),
        },
        "baselines": {ref: _baseline_entry(i) for ref, i in infos.items()},
        "records": recs,
    }


# --------------------------------------------------------------------------------------
# reading / joining
# --------------------------------------------------------------------------------------
def write_sidecar(path: str, doc: dict, *, overwrite: bool = False) -> None:
    if os.path.exists(path) and not overwrite:
        raise TraceError(f"refusing to overwrite existing {path!r}", reason="OUTPUT_EXISTS")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")


def load_sidecar(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise TraceError(f"cannot read sidecar {path!r}: {e}") from e
    if not isinstance(doc, dict) or doc.get("schema") != SIDECAR_SCHEMA:
        raise TraceError(f"sidecar schema is not {SIDECAR_SCHEMA!r}", reason="TRACE_META_INVALID")
    return doc


def join_records(trace_path: str, records: list[dict], doc: dict) -> dict[tuple, dict]:
    """Bind a sidecar to its trace. Returns {(trial, control): sidecar_record}; raises on any ambiguity."""
    want = (doc.get("trace") or {}).get("sha256")
    got = file_sha256(trace_path)
    if want != got:
        raise TraceError("sidecar does not belong to this trace (trace.sha256 mismatch)",
                         reason="TRACE_META_MISMATCH")
    side: dict[tuple, dict] = {}
    for m in doc.get("records", []):
        key = (m.get("trial"), m.get("control"))
        if key in side:
            raise TraceError(f"sidecar has duplicate records for trial={key[0]} control={key[1]}",
                             reason="TRACE_META_MISMATCH")
        side[key] = m
    seen = set()
    for r in records:
        key = (r["trial"], r["control"])
        if key in seen:
            raise TraceError(f"trace has duplicate records for trial={key[0]} control={key[1]}",
                             reason="TRACE_META_MISMATCH")
        seen.add(key)
        m = side.get(key)
        if m is None or m.get("fold_hash") != r["fold_hash"]:
            raise TraceError(f"sidecar record for trial={key[0]} control={key[1]} is missing or has a "
                             "different fold_hash", reason="TRACE_META_MISMATCH")
    if set(side) - seen:
        raise TraceError("sidecar has records that are not in the trace", reason="TRACE_META_MISMATCH")
    return side
