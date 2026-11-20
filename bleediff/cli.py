"""Frozen CLI lexicon: calibrate | bleed | bounce | diff | inspect.

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
from .arena import load_fixtures, plan_requests
from .client import DEFAULT_N_PROBS, LlamaClient
from .errors import Inconclusive
from .fold import TRACE_SCHEMA, check_record, gate, read_traces
from .runner import Harness, load_baseline

DEFAULT_SERVER = "http://127.0.0.1:8080"
DEFAULT_FIXTURES = os.path.join("fixtures", "contexts.json")


# --------------------------------------------------------------------------------------
def _fmt(x: float, w: int = 8, p: int = 4) -> str:
    return f"{x:{w}.{p}f}"


def _inconclusive(e: Inconclusive) -> int:
    print(f"INCONCLUSIVE [{e.reason}]: {e}")
    return EXIT_INCONCLUSIVE


# -- calibrate -----------------------------------------------------------------------------
def cmd_calibrate(a) -> int:
    fixtures = load_fixtures(a.fixtures)
    client = LlamaClient(a.server, timeout=a.timeout, n_probs=a.n_probs)
    h = Harness(client, fixtures)
    info = h.preflight()
    print(f"server      {info.base_url}")
    print(f"model       {info.model_alias or info.model_path}  (hash basis: {info.model_hash_basis})")
    print(f"model_hash  {info.model_hash}")
    print(f"tokens      {h.report.to_json()['id_to_letter']}  dropped={list(h.report.dropped)}")
    base = h.calibrate()
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(base, fh, indent=2)
        fh.write("\n")
    print(f"\nclean 3-request baseline B_ref (3 per-request vectors per scenario folded and discarded)")
    print(f"{'scenario':<22}{'B_ref[A]':>9}{'B_ref[B]':>9}{'B_ref[C]':>9}{'B_ref[D]':>9}  argmax  reference_hash")
    for sid, s in base["scenarios"].items():
        r = s["B_ref"]
        print(f"{sid:<22}{_fmt(r[0],9)}{_fmt(r[1],9)}{_fmt(r[2],9)}{_fmt(r[3],9)}  {s['ref_argmax']:^6}  "
              f"{s['reference_hash'][:16]}")
    for w in base["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    print(f"\nwrote {a.out}")
    return EXIT_PASS


# -- bleed (offline) -----------------------------------------------------------------------
def cmd_bleed(a) -> int:
    fixtures = load_fixtures(a.fixtures)
    ok = True
    print(f"cursor_offset={a.offset} unit={a.unit}   (offline preview: no server contacted, no model state touched)\n")
    for sc in fixtures.scenarios:
        if a.scenario and sc.id != a.scenario:
            continue
        t = plan_requests(sc, "treatment", a.offset, a.unit)
        c = plan_requests(sc, "control", a.offset, a.unit)
        tg, cg = t.geometry, c.geometry
        slots = ", ".join(f"{s.name}[{s.start}:{s.end}]" for s in t.arena.slots)
        match = (tg.bleed_bytes == cg.bleed_bytes and tg.bleed_words == cg.bleed_words
                 and tg.boundary == cg.boundary and len(t.arena.buf) == len(c.arena.buf))
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
    offset = a.offset if a.offset is not None else fixtures.default_cursor_offset
    if offset <= 0:
        raise Inconclusive("bounce needs cursor_offset > 0 (offset 0 is the clean baseline)", reason="BAD_ARGS")
    client = LlamaClient(a.server, timeout=a.timeout, n_probs=a.n_probs)
    h = Harness(client, fixtures)
    h.preflight()
    records = h.bounce(baseline, a.trials, offset, a.unit)
    from .fold import write_traces

    write_traces(a.out, records)
    print(f"bounced {a.trials} trial(s) x (treatment + matched control), cursor_offset={offset} {a.unit}")
    print(f"disposal witness: {sum(w['vectors_absorbed_and_zeroed'] for w in h.witnesses)} per-request vectors "
          f"absorbed+zeroed, {sum(w['prompt_buffers_zeroed'] for w in h.witnesses)} prompt buffers zeroed "
          f"(collector-level only; not kernel RAM scrubbing)")
    print(f"wrote {a.out} ({len(records)} records). Next: bleediff diff --trace {a.out}")
    return EXIT_PASS


# -- diff ----------------------------------------------------------------------------------
def _arm(r: dict) -> str:
    return "control" if r["control"] else "treat"


def cmd_diff(a) -> int:
    records = read_traces(a.trace)
    print(f"{'trial':>5} {'arm':<8}{'off':>4} {'argmax':^7}{'p_A':>8}{'p_B':>8}"
          f"{'contrast_A':>12}{'contrast_B':>12}{'delta_contrast':>16}  verdict")
    for r in sorted(records, key=lambda r: (r["trial"], r["control"])):
        print(f"{r['trial']:>5} {_arm(r):<8}{r['cursor_offset']:>4} {r['raw_argmax']:^7}"
              f"{_fmt(r['p_A'])}{_fmt(r['p_B'])}{_fmt(r['contrast_A'],12)}{_fmt(r['contrast_B'],12)}"
              f"{_fmt(r['delta_contrast'],16)}  {r['verdict']}")
    g = gate(records)
    s = g.stats
    print()
    print("pre-registered gate:")
    print(f"  treatment: raw argmax==A in {s.get('n_argmax_a','?')}/{s['n_treatment']}; "
          f"delta_contrast > {EXPOSURE_DELTA_NATS} in {s.get('n_exposed','?')}/{s['n_treatment']} "
          f"(need >= {s.get('needed','?')})")
    print(f"  control:   |delta_contrast| <= {CONTROL_BAND_NATS} in {s.get('n_control_clean','?')}/{s['n_control']}")
    for reason in g.reasons:
        print(f"  - {reason}")
    print(f"\n{g.verdict}")
    return g.exit_code


# -- inspect -------------------------------------------------------------------------------
def cmd_inspect(a) -> int:
    records = read_traces(a.trace)
    rows = [r for r in records if r["trial"] == a.trial]
    if not rows:
        raise Inconclusive(f"no trial {a.trial} in {a.trace}", reason="TRACE_INVALID")
    for r in sorted(rows, key=lambda r: r["control"]):
        print(f"--- trial {r['trial']}  [{'matched neutral CONTROL' if r['control'] else 'TREATMENT (adjacent Tenant-B bleed)'}]")
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
    c.set_defaults(fn=cmd_calibrate)

    b = sub.add_parser("bleed", help="offline: preview the exact arena boundary displacement vs the matched control")
    b.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    b.add_argument("--offset", type=int, default=1)
    b.add_argument("--unit", choices=("words", "bytes"), default="words")
    b.add_argument("--scenario", default=None)
    b.set_defaults(fn=cmd_bleed)

    o = sub.add_parser("bounce", help="run treatment + matched-control 3-request folds")
    o.add_argument("--server", default=DEFAULT_SERVER)
    o.add_argument("--baseline", default="baseline.json")
    o.add_argument("--trials", type=int, default=5)
    o.add_argument("--out", default="traces.jsonl")
    o.add_argument("--fixtures", default=None, help="default: the path recorded in the baseline")
    o.add_argument("--offset", type=int, default=None, help="default: fixtures' default_cursor_offset")
    o.add_argument("--unit", choices=("words", "bytes"), default="words")
    o.add_argument("--n-probs", type=int, default=DEFAULT_N_PROBS)
    o.add_argument("--timeout", type=float, default=120.0)
    o.set_defaults(fn=cmd_bounce)

    d = sub.add_parser("diff", help="print the contrastive residual table and the PASS/FAIL gate")
    d.add_argument("--trace", default="traces.jsonl")
    d.set_defaults(fn=cmd_diff)

    i = sub.add_parser("inspect", help="detailed view of a single trial")
    i.add_argument("--trace", default="traces.jsonl")
    i.add_argument("--trial", type=int, required=True)
    i.set_defaults(fn=cmd_inspect)
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
