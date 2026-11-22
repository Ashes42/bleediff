"""CLI: calibrate | bleed | bounce | diff | inspect (the frozen research lexicon), track (decision-margin tracker), atlas and gemini (answer-only hosted runs).

Exit codes: 0 = PASS / ok, 1 = FAIL NO_EXPOSURE (or bleed preview mismatch), 2 = INCONCLUSIVE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from . import (
    CONTROL_BAND_NATS,
    EXIT_FAIL,
    EXIT_INCONCLUSIVE,
    EXIT_PASS,
    EXPOSURE_DELTA_NATS,
    __version__,
)
from .arena import FILLER_ALGORITHM, load_fixtures, plan_requests
from .control import ALGORITHM as PROSE_ALGORITHM
from .control import load_manifest
from .protocol import file_sha256 as _file_sha256
from .protocol import (
    ProtocolError,
    check_baseline_identity,
    load_protocol,
    protocol_summary,
    scenario_offsets,
    verify_protocol_static,
)
from .client import DEFAULT_N_PROBS, LlamaClient
from .errors import Inconclusive, TraceError
from .fold import TRACE_SCHEMA, check_record, gate, read_traces, write_traces
from .meta import (
    baseline_infos_from_baseline,
    baseline_infos_from_sidecar,
    build_sidecar,
    file_sha256,
    join_records,
    legacy_sidecar,
    load_sidecar,
    merge_baseline_infos,
    sidecar_path,
    write_sidecar,
)
from .runner import Harness, load_baseline
from .track import DEFAULT_FIXTURES as TRACK_FIXTURES
from .track import DEFAULT_SCENARIO as TRACK_SCENARIO
from .track import cmd_track
from .atlas import cmd_atlas
from .gemini import THINKING_LEVELS, cmd_gemini

DEFAULT_SERVER = "http://127.0.0.1:8080"
DEFAULT_FIXTURES = os.path.join("fixtures", "contexts.json")


# --------------------------------------------------------------------------------------
def _fmt(x: float, w: int = 8, p: int = 4) -> str:
    return f"{x:{w}.{p}f}"


def _inconclusive(e: Inconclusive) -> int:
    print(f"INCONCLUSIVE [{e.reason}]: {e}")
    return EXIT_INCONCLUSIVE


# -- calibrate -----------------------------------------------------------------------------
def _bind_protocol(a, fixtures, path: str | None):
    """Load + validate a protocol against the fixtures BEFORE any server contact. Returns (protocol, offsets)."""
    if fixtures.requires_protocol and not path:
        raise Inconclusive(f"these fixtures may only be used under protocol {fixtures.requires_protocol!r}: pass --protocol",
                           reason="BAD_ARGS")
    if not path:
        return None, None
    protocol = load_protocol(path)
    offsets = scenario_offsets(protocol, fixtures)                 # fixtures hash + exact scenario coverage
    if fixtures.requires_protocol and fixtures.requires_protocol != protocol.get("protocol_id"):
        raise ProtocolError(f"fixtures require protocol {fixtures.requires_protocol!r}, not {protocol.get('protocol_id')!r}")
    if "scenarios" in protocol:                                    # semantic protocols carry per-scenario static metadata
        problems = verify_protocol_static(protocol, fixtures)
        if problems:
            raise ProtocolError("protocol fails static verification: " + "; ".join(problems))
    return protocol, offsets


def cmd_calibrate(a) -> int:
    fixtures = load_fixtures(a.fixtures)
    protocol, _ = _bind_protocol(a, fixtures, a.protocol)
    client = LlamaClient(a.server, timeout=a.timeout, n_probs=a.n_probs)
    h = Harness(client, fixtures)
    info = h.preflight()
    print(f"server      {info.base_url}")
    print(f"model       {info.model_alias or info.model_path}  (hash basis: {info.model_hash_basis})")
    print(f"model_hash  {info.model_hash}")
    print(f"tokens      {h.report.to_json()['id_to_letter']}  dropped={list(h.report.dropped)}")
    base = h.calibrate(protocol_info=protocol_summary(protocol, a.protocol) if protocol else None)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(base, fh, indent=2)
        fh.write("\n")
    print(f"\nclean 3-request baseline B_ref (3 per-request vectors per scenario folded and discarded)")
    print(f"{'scenario':<22}{'B_ref[A]':>9}{'B_ref[B]':>9}{'B_ref[C]':>9}{'B_ref[D]':>9}  argmax  A-B margin  "
          f"coverage(mean/min)  reference_hash")
    for sid, s in base["scenarios"].items():
        r = s["B_ref"]
        cov = s["coverage"]
        cov_txt = (f"{cov['mean']:.3f}/{cov['min']:.3f}" if isinstance(cov, dict)
                   and isinstance(cov.get("mean"), float) else "unknown")
        print(f"{sid:<22}{_fmt(r[0],9)}{_fmt(r[1],9)}{_fmt(r[2],9)}{_fmt(r[3],9)}  {s['ref_argmax']:^6}  "
              f"{s['ab_margin']:>+9.4f}  {cov_txt:>18}  {s['reference_hash'][:16]}")
    print("(A-B margin and coverage are diagnostics only; no threshold is applied to either.)")
    for w in base["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    print(f"\nwrote {a.out}")
    return EXIT_PASS


# -- bleed (offline) -----------------------------------------------------------------------
def cmd_bleed(a) -> int:
    fixtures = load_fixtures(a.fixtures)
    ok = True
    protocol, offs = _bind_protocol(a, fixtures, a.protocol)
    mpath = a.controls_manifest or (os.path.join(os.path.dirname(fixtures.path) or ".", protocol["controls_manifest"])
                                    if protocol else os.path.join(os.path.dirname(fixtures.path) or ".", "controls",
                                                                  "neutral-prose-1.json"))
    manifest = None
    if os.path.exists(mpath):
        try:
            manifest = load_manifest(mpath)
        except Inconclusive:
            manifest = None
    use_prose = bool(manifest) and (bool(protocol) or (manifest.get("offset"), manifest.get("unit")) == (a.offset, a.unit))
    which = (f"{PROSE_ALGORITHM} (preserved manifest)" if use_prose else
             f"{FILLER_ALGORITHM} (historical tokenizer-free preview: no preserved controls for this offset)")
    print(f"cursor_offset={offs if offs else a.offset} unit={a.unit}   (offline preview: no server contacted, no model state touched)")
    print(f"control: {which}\n")
    for sc in fixtures.scenarios:
        if a.scenario and sc.id != a.scenario:
            continue
        so = offs[sc.id] if offs else a.offset
        t = plan_requests(sc, "treatment", so, a.unit)
        entry = (manifest or {}).get("controls", {}).get(sc.id) if use_prose else None
        c = plan_requests(sc, "control", so, a.unit, filler=entry["fragment_text"] if entry else None)
        tg, cg = t.geometry, c.geometry
        slots = ", ".join(f"{s.name}[{s.start}:{s.end}]" for s in t.arena.slots)
        match = (tg.bleed_bytes == cg.bleed_bytes and tg.bleed_words == cg.bleed_words
                 and tg.boundary == cg.boundary and (bool(entry) or len(t.arena.buf) == len(c.arena.buf)))
        ok &= match
        print(f"== {sc.id} - {sc.title}")
        print(f"   arena        {len(t.arena.buf)} bytes contiguous | {slots}")
        print(f"   boundary     byte {tg.boundary} (end of slot #3 == start of adjacent slot)")
        print(f"   treatment    cursor reads to byte {tg.read_end}: +{tg.bleed_bytes} bytes / {tg.bleed_words} words"
              f"{'  [CLAMPED to adjacent slot]' if tg.clamped else ''}")
        print(f"                {t.bleed_fragment.decode('ascii').strip()!r}")
        print(f"   control      cursor reads to byte {cg.read_end}: +{cg.bleed_bytes} bytes / {cg.bleed_words} words")
        print(f"                {c.bleed_fragment.decode('ascii').strip()!r}")
        print(f"   matched      {'YES' if match else 'NO  <-- MISMATCH'} "
              f"(bytes {tg.bleed_bytes}={cg.bleed_bytes}, words {tg.bleed_words}={cg.bleed_words}; "
              "model-token parity is checked live in `bounce`)")
        r3 = [bytes(p) for p in t.prompts][2]
        r3c = [bytes(p) for p in c.prompts][2]
        print(f"   request #3   prompt bytes treatment={len(r3)} control={len(r3c)} "
              f"sha256 {hashlib.sha256(r3).hexdigest()[:12]} vs {hashlib.sha256(r3c).hexdigest()[:12]}\n")
        t.dispose()
        c.dispose()
    print("ARENA PREVIEW:", "matched control OK" if ok else "MISMATCH")
    return EXIT_PASS if ok else EXIT_FAIL


# -- bounce --------------------------------------------------------------------------------
def _resolve_fixtures(a, baseline: dict, baseline_path: str) -> str:
    if a.fixtures:
        return a.fixtures
    p = baseline.get("fixtures_path") or DEFAULT_FIXTURES
    if not os.path.isabs(p) and not os.path.exists(p):
        alt = os.path.join(os.path.dirname(os.path.abspath(baseline_path)), p)
        if os.path.exists(alt):
            return alt
    return p


def cmd_bounce(a) -> int:
    baseline = load_baseline(a.baseline)
    fixtures = load_fixtures(_resolve_fixtures(a, baseline, a.baseline))
    protocol, offsets = None, None
    if a.protocol:
        if a.offset is not None:
            raise Inconclusive("--protocol sets every scenario's offset; do not combine it with --offset", reason="BAD_ARGS")
        if a.control != PROSE_ALGORITHM or a.unit != "words":
            raise Inconclusive(f"--protocol requires --control {PROSE_ALGORITHM} and --unit words", reason="BAD_ARGS")
    protocol, offsets = _bind_protocol(a, fixtures, a.protocol)     # fails closed before any server contact
    check_baseline_identity(baseline, protocol, a.protocol, fixtures)   # a historical/other-protocol calibration is refused
    offset = ((protocol.get("default_offset") or min(offsets.values())) if protocol else (
        a.offset if a.offset is not None else fixtures.default_cursor_offset))
    if offset <= 0 or any(n <= 0 for n in (offsets or {}).values()):
        raise Inconclusive("bounce needs cursor_offset > 0 (offset 0 is the clean baseline)", reason="BAD_ARGS")
    client = LlamaClient(a.server, timeout=a.timeout, n_probs=a.n_probs)
    h = Harness(client, fixtures)
    h.preflight()
    manifest, manifest_path = None, None
    if a.control == PROSE_ALGORITHM:
        default_manifest = (os.path.join(os.path.dirname(fixtures.path) or ".", protocol["controls_manifest"])
                            if protocol else os.path.join(os.path.dirname(fixtures.path) or ".", "controls",
                                                          "neutral-prose-1.json"))
        manifest_path = a.controls_manifest or default_manifest
        manifest = load_manifest(manifest_path)
        if protocol and file_sha256(manifest_path) != protocol.get("controls_manifest_sha256"):
            raise Inconclusive("controls manifest does not match the hash recorded in the protocol",
                               reason="PROTOCOL_INVALID")
    if protocol:
        h.protocol_info = protocol_summary(protocol, a.protocol)
    records = h.bounce(baseline, a.trials, offset, a.unit, control=a.control, manifest=manifest,
                       manifest_sha256=file_sha256(manifest_path) if manifest_path else None, offsets=offsets)
    write_traces(a.out, records)
    meta = build_sidecar(trace_path=a.out, records=records, harness=h, baseline_doc=baseline,
                         baseline_file_sha256=file_sha256(a.baseline), cursor_offset=offset, unit=a.unit)
    write_sidecar(sidecar_path(a.out), meta, overwrite=True)  # rewritten together with the trace it binds
    shown = f"per-scenario windows {offsets}" if offsets else f"cursor_offset={offset}"
    print(f"bounced {a.trials} trial(s) x (treatment + matched control), {shown} {a.unit}")
    print(f"wrote {sidecar_path(a.out)} (provenance sidecar; bound to the trace by sha256)")
    print(f"disposal witness: {sum(w['vectors_absorbed_and_zeroed'] for w in h.witnesses)} per-request vectors "
          f"absorbed+zeroed, {sum(w['prompt_buffers_zeroed'] for w in h.witnesses)} prompt buffers zeroed "
          f"(collector-level only; not kernel RAM scrubbing)")
    print(f"wrote {a.out} ({len(records)} records). Next: bleediff diff --trace {a.out}")
    return EXIT_PASS


# -- diff ----------------------------------------------------------------------------------
def _arm(r: dict) -> str:
    return "control" if r["control"] else "treat"


def _load_provenance(trace_path: str, records: list[dict], a) -> tuple[dict | None, dict]:
    """Resolve (sidecar_doc, baseline_infos) for a trace. Sidecar: ``--meta`` or ``<trace>.meta.json`` if it
    exists (verified against the trace by sha256 + per-record fold_hash; a mismatch is an error, never
    ignored). ``--baseline`` supplies/cross-checks baselines for traces that predate sidecars."""
    meta_path = a.meta or (sidecar_path(trace_path) if os.path.exists(sidecar_path(trace_path)) else None)
    doc, infos = None, {}
    if meta_path:
        doc = load_sidecar(meta_path)
        join_records(trace_path, records, doc)
        infos = baseline_infos_from_sidecar(doc)
    if a.baseline:
        bdoc = load_baseline(a.baseline)
        infos = merge_baseline_infos(infos, baseline_infos_from_baseline(bdoc))
        if doc is None:
            doc = legacy_sidecar(trace_path=trace_path, records=records, baseline_doc=bdoc,
                                 baseline_file_sha256=file_sha256(a.baseline))
    return doc, infos


def _yn(v) -> str:
    return "n/a" if v is None else ("ok" if v else "FAIL")


def cmd_diff(a) -> int:
    records = read_traces(a.trace)
    doc, infos = _load_provenance(a.trace, records, a)
    print(f"{'trial':>5} {'arm':<8}{'off':>4} {'argmax':^7}{'p_A':>8}{'p_B':>8}"
          f"{'contrast_A':>12}{'contrast_B':>12}{'delta_contrast':>16}  verdict(stored)")
    for r in sorted(records, key=lambda r: (r["trial"], r["control"])):
        print(f"{r['trial']:>5} {_arm(r):<8}{r['cursor_offset']:>4} {r['raw_argmax']:^7}"
              f"{_fmt(r['p_A'])}{_fmt(r['p_B'])}{_fmt(r['contrast_A'],12)}{_fmt(r['contrast_B'],12)}"
              f"{_fmt(r['delta_contrast'],16)}  {r['verdict']}")
    g = gate(records, infos)
    s = g.stats
    if g.assessments:
        print("\nvalidity preconditions (a trial that fails any is INVALID: it is never negative evidence):")
        print(f"{'trial':>5} {'scenario':<22}{'baseline':>9}{'A-B margin*':>12}{'treat A?':>9}{'ctrl<=0.1':>10}"
              f"{'delta':>8}  classification")
        for t in g.assessments:
            margin = "n/a" if t.baseline_margin_ab is None else f"{t.baseline_margin_ab:+.4f}"
            delta = "n/a" if t.treatment_delta is None else f"{t.treatment_delta:+.3f}"
            print(f"{t.trial:>5} {(t.scenario_id or 'UNKNOWN'):<22}{_yn(t.baseline_ok):>9}{margin:>12}"
                  f"{_yn(t.treatment_ok):>9}{_yn(t.control_ok):>10}{delta:>8}  {t.classification}"
                  f"{' [' + ', '.join(t.reasons) + ']' if t.reasons else ''}")
        print("  * diagnostic only: baseline validity uses argmax(B_ref)==A; no margin threshold exists.")
    print("\npre-registered gate (thresholds unchanged):")
    if "n_trials" in s:
        print(f"  trials: {s['n_trials']}  valid: {s['n_valid']}  invalid: {s['n_invalid']}  "
              f"valid+exposed (delta_contrast > {EXPOSURE_DELTA_NATS}): {s['n_valid_exposed']}  "
              f"(PASS needs >= {s['needed']}; FAIL needs >= {s['min_valid_not_exposed_for_fail']} valid "
              f"non-exposed with no failed control)")
        print(f"  controls out of tolerance (|delta| > {CONTROL_BAND_NATS}): {s['control_out_of_tolerance'] or 'none'}")
    for reason in g.reasons:
        print(f"  - {reason}")
    print(f"\n{g.verdict}")
    if a.json_out:
        _write_json_new(a.json_out, {
            "schema": "bleediff.diff-result/1", "bleediff_version": __version__,
            "trace_sha256": file_sha256(a.trace), "verdict": g.verdict, "exit_code": g.exit_code,
            "reasons": g.reasons, "stats": g.stats,
            "provenance_origin": doc.get("origin") if doc else "none (no sidecar, no baseline)",
            "trials": [t.to_json() for t in g.assessments],
        })
        print(f"wrote {a.json_out}")
    return g.exit_code


def _write_json_new(path: str, obj: dict) -> None:
    """Write a NEW json file; refuses to overwrite so existing evidence can never be clobbered."""
    if os.path.exists(path):
        raise TraceError(f"refusing to overwrite existing {path!r}", reason="OUTPUT_EXISTS")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


# -- inspect -------------------------------------------------------------------------------
def cmd_inspect(a) -> int:
    records = read_traces(a.trace)
    rows = [r for r in records if r["trial"] == a.trial]
    if not rows:
        raise Inconclusive(f"no trial {a.trial} in {a.trace}", reason="TRACE_INVALID")
    side = {}
    meta_path = a.meta or (sidecar_path(a.trace) if os.path.exists(sidecar_path(a.trace)) else None)
    if meta_path:
        try:
            side = join_records(a.trace, records, load_sidecar(meta_path))
        except Inconclusive as e:
            print(f"WARNING: sidecar present but not usable for this trace ({e.reason}): {e}")
    for r in sorted(rows, key=lambda r: r["control"]):
        print(f"--- trial {r['trial']}  [{'matched neutral CONTROL' if r['control'] else 'TREATMENT (adjacent Tenant-B bleed)'}]")
        m = side.get((r["trial"], r["control"]))
        if m:
            cov = m.get("coverage")
            cov_txt = (f"mean {cov['mean']:.4f} / min {cov['min']:.4f} over {cov['n_requests']} requests"
                       if isinstance(cov, dict) and isinstance(cov.get("mean"), float) else "UNKNOWN")
            print(f"  scenario        {m.get('scenario_id')}")
            print(f"  prompt tokens   {m.get('request_prompt_tokens')}")
            print(f"  coverage        {cov_txt}  (diagnostic; no threshold defined)")
            print(f"  control filler  {m.get('control_filler')}")
        print(f"  model_hash      {r['model_hash']}")
        print(f"  reference_hash  {r['reference_hash']}")
        print(f"  fold_hash       {r['fold_hash']}")
        print(f"  cursor_offset   {r['cursor_offset']}")
        print(f"  raw  p_A={r['p_A']:.6f}  p_B={r['p_B']:.6f}  p_C={r['p_C']:.6f}  p_D={r['p_D']:.6f}   argmax={r['raw_argmax']}")
        print(f"  contrast_A={r['contrast_A']:+.6f}  contrast_B={r['contrast_B']:+.6f}  "
              f"delta_contrast={r['delta_contrast']:+.6f} nats")
        print(f"  verdict         {r['verdict']}")
        problems = check_record(r)
        print("  source-vector disposal witness:")
        print(f"    - record keys == frozen schema ({len(TRACE_SCHEMA)} fields), none holds a per-request vector or prompt")
        print(f"    - p_* is the 3-way fold M only; fold_hash recomputes from p_*: "
              f"{'OK' if not any('fold_hash' in p for p in problems) else 'MISMATCH'}")
        print(f"    - internal consistency: {'OK' if not problems else '; '.join(problems)}")
    print("  scope: models collector-level retention disposal (vectors zeroed in place, prompt buffers zeroed);")
    print("         it is NOT kernel/process RAM scrubbing.")
    return EXIT_PASS


# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bleediff",
        description="Contrastive-suppression evaluation of adjacent-tenant context bleed. "
                    "What survives subtraction was already in the composite.",
    )
    p.add_argument("--version", action="version", version=f"bleediff {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="compute the clean 3-request reference distributions")
    c.add_argument("--server", default=DEFAULT_SERVER)
    c.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    c.add_argument("--out", default="baseline.json")
    c.add_argument("--n-probs", type=int, default=DEFAULT_N_PROBS)
    c.add_argument("--timeout", type=float, default=120.0)
    c.add_argument("--protocol", default=None,
                   help="protocol the calibration belongs to (required for fixtures that declare requires_protocol); "
                        "its id and file hash are recorded in the baseline and checked by bounce")
    c.set_defaults(fn=cmd_calibrate)

    b = sub.add_parser("bleed", help="offline: preview the exact arena boundary displacement vs the matched control")
    b.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    b.add_argument("--offset", type=int, default=1)
    b.add_argument("--unit", choices=("words", "bytes"), default="words")
    b.add_argument("--scenario", default=None)
    b.add_argument("--controls-manifest", default=None)
    b.add_argument("--protocol", default=None, help="show each scenario's window and control from a protocol file")
    b.set_defaults(fn=cmd_bleed)

    o = sub.add_parser("bounce", help="run treatment + matched-control 3-request folds")
    o.add_argument("--server", default=DEFAULT_SERVER)
    o.add_argument("--baseline", default="baseline.json")
    o.add_argument("--trials", type=int, default=5)
    o.add_argument("--out", default="traces.jsonl")
    o.add_argument("--fixtures", default=None, help="default: the path recorded in the baseline")
    o.add_argument("--offset", type=int, default=None, help="default: fixtures' default_cursor_offset")
    o.add_argument("--unit", choices=("words", "bytes"), default="words")
    o.add_argument("--control", choices=(PROSE_ALGORITHM, FILLER_ALGORITHM), default=PROSE_ALGORITHM,
                   help=f"control construction (default {PROSE_ALGORITHM}: preserved fluent prose; "
                        f"{FILLER_ALGORITHM} is the historical filler, kept only to reproduce old evidence)")
    o.add_argument("--controls-manifest", default=None,
                   help="preserved controls (default: controls/neutral-prose-1.json next to the fixtures)")
    o.add_argument("--protocol", default=None,
                   help="versioned protocol file with per-scenario windows (e.g. fixtures/protocols/window-repair-1.json); "
                        "its controls manifest is hash-checked. Cannot be combined with --offset")
    o.add_argument("--n-probs", type=int, default=DEFAULT_N_PROBS)
    o.add_argument("--timeout", type=float, default=120.0)
    o.set_defaults(fn=cmd_bounce)

    d = sub.add_parser("diff", help="print the contrastive residual table and the PASS/FAIL gate")
    d.add_argument("--trace", default="traces.jsonl")
    d.add_argument("--meta", default=None, help="provenance sidecar (default: <trace>.meta.json if present)")
    d.add_argument("--baseline", default=None,
                   help="baseline.json; supplies baseline validity for traces that predate sidecars")
    d.add_argument("--json-out", default=None, help="also write the classification to a NEW json file "
                   "(never overwrites)")
    d.set_defaults(fn=cmd_diff)

    i = sub.add_parser("inspect", help="detailed view of a single trial")
    i.add_argument("--trace", default="traces.jsonl")
    i.add_argument("--trial", type=int, required=True)
    i.add_argument("--meta", default=None, help="provenance sidecar (default: <trace>.meta.json if present)")
    i.set_defaults(fn=cmd_inspect)

    t = sub.add_parser("track", help="interactive decision-margin tracker: edit a fragment, press Enter, see how the "
                                     "A/B/C/D probabilities and the A/B margin moved (single prompt; not the research fold)")
    t.add_argument("--server", default=DEFAULT_SERVER)
    t.add_argument("--out", required=True, help="NEW directory for the session history (never overwritten)")
    t.add_argument("--fixtures", default=TRACK_FIXTURES)
    t.add_argument("--scenario", default=TRACK_SCENARIO, help="frozen scenario whose request #3 is measured")
    t.add_argument("--fragment", default=None, help="UTF-8 text file you edit (default: <out>/fragment.txt, created empty)")
    t.add_argument("--retain-prompts", action="store_true",
                   help="also keep each measured prompt and fragment under <out>/prompts/ (off by default: only hashes are recorded)")
    t.add_argument("--arms", default=None, metavar="CANDIDATES.json",
                   help="batch mode: a UTF-8 JSON file of labelled candidate fragments; each is scored against one clean baseline "
                        "and ranked by observed movement toward the incorrect answer (see bleediff/batch.py for the format)")
    t.add_argument("--incorrect", choices=["A", "B", "C", "D"], default="B",
                   help="batch mode: the letter of the designated incorrect (competing) answer the ranking is measured toward (default B)")
    t.add_argument("--correct", choices=["A", "B", "C", "D"], default=None,
                   help="batch mode: the letter holding the correct answer. Optional while --incorrect is A or B (then the other of A/B, as in v0.3.0); "
                        "required when --incorrect is C or D, because the roles are then scored semantically (ln(p_correct / p_competing))")
    t.add_argument("--n-probs", type=int, default=DEFAULT_N_PROBS)
    t.add_argument("--timeout", type=float, default=120.0)
    t.set_defaults(fn=cmd_track)

    g = sub.add_parser("gemini", help="ANSWER-ONLY measurements on the Gemini API (hosted models that expose no A-D probabilities), or --replay saved "
                                      "evidence offline. Never produces probabilities, margins or confidence.")
    g.add_argument("--replay", default=None, metavar="SESSION_DIR", help="offline: re-read and re-analyse a saved session (no key, no network); the v0.8 pilot layout also needs --prompts")
    g.add_argument("--prompts", default=None, help="frozen prompt dataset (JSON, schema bleediff.hosted-prompts/1 or the v0.7 pilot's frozen-requests file)")
    g.add_argument("--model", default=None, help="Gemini model id, e.g. gemini-3.8-flash (live runs)")
    g.add_argument("--out", default=None, help="NEW session directory for a live run (never overwritten)")
    g.add_argument("--passes", type=int, default=1, help="repeat every prompt this many times (default 1; repeats are replicate checks, not independent samples)")
    g.add_argument("--budget", type=float, default=None, help="REQUIRED for live runs: maximum estimated spend in USD; the run stops before a request could exceed it")
    g.add_argument("--price-in", type=float, default=None, help="REQUIRED for live runs: input price in USD per 1M tokens (list price; there are no built-in prices)")
    g.add_argument("--price-out", type=float, default=None, help="REQUIRED for live runs: output price in USD per 1M tokens (thinking tokens are billed as output)")
    g.add_argument("--max-output-tokens", type=int, default=512)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--thinking-level", choices=list(THINKING_LEVELS), default=None, help="only for models that support it (a rejected value is reported, never retried)")
    g.add_argument("--thinking-budget", type=int, default=None, help="thinking token budget for models that support it (not with --thinking-level)")
    g.add_argument("--max-retries", type=int, default=2, help="retries per request for rate limits and transient errors (default 2, at most 5)")
    g.add_argument("--timeout", type=float, default=120.0)
    g.add_argument("--json-out", default=None, help="also write the analysis to a NEW json file")
    g.set_defaults(fn=cmd_gemini)

    a = sub.add_parser("atlas", help="influence atlas: aggregate, compare and rank EXISTING batch sessions across tasks and models "
                                     "(reads saved results; never queries a model)")
    a.add_argument("--manifest", required=True, help="atlas input manifest (JSON, schema bleediff.atlas-input/1)")
    a.add_argument("--out", required=True, help="NEW directory for atlas.json, candidates.csv, category-summary.csv, model-comparison.csv, report.md")
    a.add_argument("--model", action="append", default=[], help="show only this model id (repeatable)")
    a.add_argument("--task", action="append", default=[], help="show only this task id (repeatable)")
    a.add_argument("--category", action="append", default=[], help="show only candidates of this category (repeatable)")
    a.add_argument("--outcome", action="append", default=[], help="show only this correctness outcome: stayed_correct, induced_incorrect, "
                   "recovered_to_correct, baseline_incorrect_remained_incorrect, unscored (repeatable)")
    a.add_argument("--effect-marker", type=float, default=0.25, help="descriptive size in nats used for the 'at least' counts (default 0.25)")
    a.add_argument("--keep-going", action="store_true", help="exclude sources that fail validation and report them instead of stopping")
    a.set_defaults(fn=cmd_atlas)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except Inconclusive as e:
        return _inconclusive(e)
    except BrokenPipeError:  # pragma: no cover
        return EXIT_PASS


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
