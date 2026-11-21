"""``bleediff track --arms FILE``: batch mode of the decision tracker (v0.3, exploratory).

Scores a list of labelled candidate fragments against ONE clean baseline and ranks the observed effects. Every candidate is the
same single-prompt measurement the interactive tracker makes (request #3 of the frozen scenario with the fragment appended
after its authorized context), through the unchanged ``LlamaClient.choice_probs``; the margin ``ln(p_A / p_B)`` and the change
from the baseline are the unchanged ``bleediff.track`` functions. This module only reads the candidate file, drives the
measurements, ranks them and writes the files.

WHAT THE RANKING IS
    Candidates are ordered by the observed movement of the A/B margin toward the designated incorrect answer (default B):
    ``movement = baseline margin - candidate margin`` nats for B (positive = moved toward B), the reverse for A. This is an
    OBSERVED CANDIDATE EFFECT of one fragment on one model and one prompt. It is not evidence that a particular word, topic
    or identifier caused it: candidates that differ in several ways at once cannot be attributed, and unmatched lengths are
    shown (fragment and prompt token counts) precisely so that length differences stay visible.

CANDIDATE FILE (UTF-8 JSON), any of
    [{"label": "x", "text": "..."}, ...]
    {"candidates": [{"label": "x", "text": "..."}, ...]}
    {"arms": {"x": "..." | {"text": "..."}}}   or   {"fragments": {...}}        (e.g. the frozen research files)
An optional per-candidate "sha256" is checked against the text. Labels must be unique. There is no automatic generation.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter

from . import CHOICES, __version__
from .arena import load_fixtures
from .client import LlamaClient
from .errors import Inconclusive, ServerUnreachable
from .meta import code_revision, file_sha256
from .roles import ROLE_NOTE, role_changes, role_scores, roles_are_ab
from .track import MAX_FRAGMENT_CHARS, MEASUREMENT_NOTE, TOLERANCE, MeasurementError, TrackSession, _model_name, _sha, _signed, _utc, compare

BATCH_SCHEMA = "bleediff.track-batch/1"
MAX_CANDIDATES = 1000
MAX_FILE_BYTES = 16_000_000
MAX_LABEL_CHARS = 200
UNREACHABLE_LIMIT = 3          # consecutive unreachable-server failures after which the run stops
OBSERVED_NOTE = ("OBSERVED CANDIDATE EFFECTS: each row is one fragment's measured effect on one model and one prompt. A large "
                 "shift does not show that a particular word, topic or identifier caused it, and candidates whose lengths or "
                 "wording differ in several ways cannot be attributed to any one of them.")
CSV_COLUMNS = ["rank", "label", "status", "movement_toward_incorrect_nats", "margin_AB", "change_from_baseline_nats",
               "p_A", "p_B", "p_C", "p_D", "preferred_choice", "preferred_differs_from_baseline", "pairwise_AB_crossed",
               "fragment_chars", "fragment_bytes", "fragment_words", "fragment_tokens", "prompt_tokens",
               "prompt_tokens_minus_modal", "fragment_sha256", "error_reason",
               "margin_correct_vs_competing", "margin_correct_vs_strongest", "newly_induced_incorrect", "correct_letter", "competing_letter"]


class CandidateFileError(Inconclusive):
    reason = "CANDIDATE_FILE_INVALID"


# --------------------------------------------------------------------------------------
# candidate file
# --------------------------------------------------------------------------------------
class Candidate:
    __slots__ = ("index", "label", "text", "problem", "problem_reason")

    def __init__(self, index: int, label: str, text: str | None, problem: str | None = None, problem_reason: str | None = None):
        self.index, self.label, self.text, self.problem, self.problem_reason = index, label, text, problem, problem_reason


def _no_duplicate_keys(pairs):
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise CandidateFileError(f"duplicate key {k!r} in the candidate file")
        seen[k] = v
    return seen


def _normalise(text: str) -> str:
    """Same normalisation as the interactive fragment file: LF newlines, surrounding whitespace stripped."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def load_candidates(path: str) -> list[Candidate]:
    """Parse and validate the candidate file. File-level problems raise ``CandidateFileError``; a bad individual candidate
    (empty, too long, NUL byte, hash mismatch) is kept as an invalid candidate so it is reported and never scored."""
    try:
        size = os.path.getsize(path)
        if size > MAX_FILE_BYTES:
            raise CandidateFileError(f"the candidate file is {size} bytes; the limit is {MAX_FILE_BYTES}")
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as e:
        raise CandidateFileError(f"cannot read the candidate file: {e.strerror or e}") from e
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise CandidateFileError(f"the candidate file is not valid UTF-8: {e}") from e
    try:
        doc = json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    except (ValueError, RecursionError) as e:
        raise CandidateFileError(f"the candidate file is not valid JSON: {e}") from e
    if isinstance(doc, list):
        items = doc
    elif isinstance(doc, dict) and isinstance(doc.get("candidates"), list):
        items = doc["candidates"]
    elif isinstance(doc, dict) and any(isinstance(doc.get(k), dict) for k in ("arms", "fragments")):
        key = "arms" if isinstance(doc.get("arms"), dict) else "fragments"
        items = [{"label": k, **(v if isinstance(v, dict) else {"text": v})} for k, v in doc[key].items()]
    else:
        raise CandidateFileError("expected a JSON list of {label, text}, or an object with 'candidates' (list) or 'arms'/'fragments' (object)")
    if not items:
        raise Inconclusive("the candidate list is empty: nothing to measure", reason="EMPTY_CANDIDATE_LIST")
    if len(items) > MAX_CANDIDATES:
        raise CandidateFileError(f"{len(items)} candidates; the limit is {MAX_CANDIDATES}")
    out, labels = [], set()
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise CandidateFileError(f"candidate {i} is not an object")
        label, body = item.get("label"), item.get("text")
        if not isinstance(label, str) or not label.strip():
            raise CandidateFileError(f"candidate {i} has no label (a non-empty string is required)")
        if len(label) > MAX_LABEL_CHARS or any(ord(ch) < 32 or ord(ch) == 127 for ch in label):
            raise CandidateFileError(f"candidate {i}: the label is too long or contains control characters")
        if label in labels:
            raise CandidateFileError(f"duplicate label {label!r}")
        labels.add(label)
        if not isinstance(body, str):
            raise CandidateFileError(f"candidate {label!r}: 'text' must be a string")
        norm = _normalise(body)
        problem = reason = None
        if "\x00" in norm:
            problem, reason = "the text contains NUL bytes", "FRAGMENT_UNREADABLE"
        elif not norm:
            problem, reason = "the text is empty (it would be the same prompt as the baseline)", "FRAGMENT_EMPTY"
        elif len(norm) > MAX_FRAGMENT_CHARS:
            problem, reason = f"the text is {len(norm)} characters; the limit is {MAX_FRAGMENT_CHARS}", "FRAGMENT_TOO_LONG"
        else:
            declared = item.get("sha256")
            if declared is not None and declared not in (_sha(norm), _sha(body)):
                problem, reason = "the text does not match its declared sha256", "FRAGMENT_HASH_MISMATCH"
        out.append(Candidate(i, label, None if problem else norm, problem, reason))
    return out


# --------------------------------------------------------------------------------------
# ranking (pure)
# --------------------------------------------------------------------------------------
def movement_toward(change_from_baseline: float, incorrect: str) -> float:
    """Movement of the A/B margin toward the designated incorrect answer, in nats, from the unchanged margin change."""
    return -change_from_baseline if incorrect == "B" else change_from_baseline


def rank_rows(rows: list[dict]) -> list[dict]:
    """Rank scored rows by ``movement`` (largest first). Ties (equal within TOLERANCE) share a rank (1, 2, 2, 4); within a tie
    the file order is kept. Returns the ranked rows (each gets ``rank`` and ``tied``)."""
    by_move = sorted(rows, key=lambda r: (-r["movement"], r["index"]))
    groups: list[list[dict]] = []
    for r in by_move:                                       # consecutive rows within TOLERANCE of the group's first row tie
        if groups and abs(r["movement"] - groups[-1][0]["movement"]) <= TOLERANCE:
            groups[-1].append(r)
        else:
            groups.append([r])
    out, pos = [], 1
    for g in groups:
        for r in sorted(g, key=lambda x: x["index"]):       # file order inside a tie
            r["rank"] = pos
            out.append(r)
        pos += len(g)
    counts = Counter(r["rank"] for r in out)
    for r in out:
        r["tied"] = counts[r["rank"]] > 1
    return out


def _csv_safe(value) -> str:
    """Guard against spreadsheet formula injection from user-supplied labels."""
    s = "" if value is None else str(value)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


def _modal(values: list[int]) -> int | None:
    if not values:
        return None
    c = Counter(values)
    top = max(c.values())
    return min(v for v, n in c.items() if n == top)


# --------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------
def _short(label: str, n: int = 30) -> str:
    return label if len(label) <= n else label[: n - 3] + "..."


def _write_csv(path: str, ranked: list[dict], failed: list[dict], correct: str = "A", competing: str = "B") -> None:
    """Columns up to ``error_reason`` are the v0.3.0 columns (margin_AB is the fixed A/B margin, movement_toward_incorrect_nats the semantic
    competing-answer movement); the last five columns are the semantic-role quantities and the letters they refer to."""
    with open(path, "x", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(CSV_COLUMNS)
        for r in ranked:
            p = r["probs"]
            w.writerow([r["rank"], _csv_safe(r["label"]), "ok", f"{r['movement']:.9f}", f"{r['margin']:.9f}", f"{r['change']:.9f}",
                        *[f"{x:.9f}" for x in p], r["preferred"], "yes" if r["flipped"] else "no", "yes" if r["crossed"] else "no",
                        r["chars"], r["bytes"], r["words"], r["fragment_tokens"], r["prompt_tokens"], r["tok_vs_modal"], r["fragment_sha256"], "",
                        f"{r['role_margin']:.9f}", f"{r['role_margin_strongest']:.9f}", "yes" if r["induced"] else "no", correct, competing])
        for r in failed:
            w.writerow(["", _csv_safe(r["label"]), r["status"], "", "", "", "", "", "", "", "", "", "", r["chars"], r["bytes"], r["words"],
                        "", "", "", r["fragment_sha256"], r["error_reason"], "", "", "", correct, competing])


def cmd_batch(args) -> int:
    out = sys.stdout
    try:
        if getattr(args, "fragment", None):
            raise Inconclusive("--arms (batch mode) and --fragment (interactive mode) cannot be combined", reason="BAD_ARGS")
        incorrect = (getattr(args, "incorrect", None) or "B").upper()
        correct = (getattr(args, "correct", None) or "").upper() or None
        if incorrect not in CHOICES:
            raise Inconclusive(f"--incorrect must be one of {', '.join(CHOICES)}", reason="BAD_ARGS")
        if correct is None:                                      # v0.3.0 behaviour: the other letter of the A/B pair
            if incorrect not in ("A", "B"):
                raise Inconclusive("--correct is required when --incorrect is C or D: the roles are scored semantically "
                                   "and the A/B margin says nothing about answers at C or D", reason="BAD_ARGS")
            correct = "B" if incorrect == "A" else "A"
        if correct not in CHOICES or correct == incorrect:
            raise Inconclusive("--correct and --incorrect must be two different letters from A, B, C, D", reason="BAD_ARGS")
        roles_default = roles_are_ab(correct, incorrect)
        cands = load_candidates(args.arms)                       # validated before the server is contacted or anything is created
        fixtures = load_fixtures(args.fixtures)
        scenarios = {s.id: s for s in fixtures.scenarios}
        if args.scenario not in scenarios:
            raise Inconclusive(f"unknown scenario {args.scenario!r}; available: {sorted(scenarios)}", reason="BAD_ARGS")
        sc = scenarios[args.scenario]
        out_dir = args.out
        if os.path.exists(out_dir):
            raise Inconclusive(f"{out_dir!r} already exists: choose a new output directory (sessions are never overwritten)",
                               reason="OUTPUT_EXISTS")
        client = LlamaClient(args.server, timeout=args.timeout, n_probs=args.n_probs)
        report = client.verify_tokenization()
        info = client.server_info()
        os.makedirs(out_dir)
        name = _model_name(info)
        session = TrackSession(client=client, scenario=sc, out_dir=out_dir, fragment_path="", model_name=name,
                               scenario_info={"id": sc.id, "title": sc.title, "fixtures_sha256": fixtures.sha256,
                                              "fixtures_file": os.path.basename(args.fixtures)},
                               model_info={"name": name}, retain_prompts=args.retain_prompts, out=out)

        def counter(prompt: str, fragment: str | None) -> dict:
            return {"fragment_tokens": len(client.tokenize("\n" + fragment, cache=False)) if fragment else 0,
                    "prompt_tokens": len(client.tokenize(prompt, cache=False))}

        session.token_counter = counter
        if not roles_default:
            session.role_labels = (correct, incorrect)                # the baseline print then names the A/B margin and the role margin separately
        session.open_history()
        file_sha = file_sha256(args.arms)
        print("BLEEDIFF TRACK (batch)", file=out)
        print(f"Model: {name}", file=out)
        print(f"Scenario: {sc.id}   Candidates: {len(cands)}   Designated incorrect answer: {incorrect}", file=out)
        print("Measurement: single prompt (request #3), margin A/B = ln(p_A / p_B); not the three-request fold.", file=out)
        if not roles_default:
            print(f"Roles: correct = {correct}, competing = {incorrect}. Ranked movement and the shown margin are for these roles "
                  f"(ln(p_correct / p_competing)); margin_AB in the files is the A/B margin and is NOT the role margin.", file=out)
        reason = "error"
        try:
            first = session.start(info, report)                  # writes the session header (provenance) and measures the clean baseline once
            if isinstance(first, Exception):
                print(f"ERROR [{getattr(first, 'reason', type(first).__name__)}]: {first}", file=out)
                reason = "baseline_failed"
                return 2
            try:
                base_roles = role_scores(first["probs"], correct, incorrect)
            except MeasurementError as e:
                print(f"ERROR [{e.reason}]: the baseline has no defined role margin: {e}", file=out)
                reason = "baseline_failed"
                return 2
            session._write({"type": "semantic_roles", "step": first["step"], "label": "baseline", "timestamp_utc": _utc(), "scores": base_roles})
            session._write({"type": "event", "event": "batch_start", "timestamp_utc": _utc(), "batch_schema": BATCH_SCHEMA,
                            "candidate_file": {"name": os.path.basename(args.arms), "sha256": file_sha, "candidates": len(cands)},
                            "designated_incorrect_answer": incorrect, "designated_correct_answer": correct,
                            "semantic_roles": {"correct_letter": correct, "competing_letter": incorrect, "roles_are_ab": roles_default, "note": ROLE_NOTE},
                            "note": "candidates are independent single-prompt measurements: change_from_previous equals change_from_baseline",
                            "observed_effects_note": OBSERVED_NOTE})
            base = first
            base_tokens = (base.get("tokens") or {}).get("prompt_tokens")
            print(f"Baseline prompt tokens: {base_tokens}", file=out)
            print("", file=out)
            scored, failed = [], []
            unreachable_run = 0
            for n, cand in enumerate(cands, 1):
                frag_info = {"sha256": _sha(cand.text) if cand.text else None, "chars": len(cand.text) if cand.text else 0,
                             "bytes": len(cand.text.encode("utf-8")) if cand.text else 0, "words": len(cand.text.split()) if cand.text else 0}
                row_fail = {"index": cand.index, "label": cand.label, "chars": frag_info["chars"], "bytes": frag_info["bytes"],
                            "words": frag_info["words"], "fragment_sha256": frag_info["sha256"] or ""}
                tag = {"candidate": {"label": cand.label, "index": cand.index}}
                if unreachable_run >= UNREACHABLE_LIMIT:
                    session._write({"type": "measurement", "step": None, "label": "candidate", "timestamp_utc": _utc(), "status": "not_run",
                                    "error": {"reason": "RUN_STOPPED", "message": "not run: the server was unreachable for "
                                              f"{UNREACHABLE_LIMIT} consecutive candidates"}, **tag})
                    failed.append({**row_fail, "status": "not_run", "error_reason": "RUN_STOPPED"})
                    continue
                if cand.problem:
                    session.steps += 1
                    session._write({"type": "measurement", "step": session.steps, "label": "candidate", "timestamp_utc": _utc(), "status": "error",
                                    "error": {"reason": cand.problem_reason, "message": cand.problem}, **tag})
                    failed.append({**row_fail, "status": "invalid", "error_reason": cand.problem_reason})
                    print(f"  [{n}/{len(cands)}] {_short(cand.label)}: INVALID ({cand.problem})", file=out)
                    continue
                session.previous = base                          # candidates are independent of one another
                res = session._measure_prompt("candidate", cand.text, extra=tag)
                if isinstance(res, Exception):
                    unreachable_run = unreachable_run + 1 if isinstance(res, ServerUnreachable) else 0
                    rsn = getattr(res, "reason", type(res).__name__)
                    failed.append({**row_fail, "status": "error", "error_reason": rsn})
                    print(f"  [{n}/{len(cands)}] {_short(cand.label)}: FAILED [{rsn}]", file=out)
                    continue
                unreachable_run = 0
                try:
                    cand_roles = role_scores(res["probs"], correct, incorrect)
                    changes = role_changes(base_roles, cand_roles)
                except MeasurementError as e:
                    session._write({"type": "semantic_roles", "step": res["step"], "label": "candidate", "timestamp_utc": _utc(), "status": "error",
                                    "error": {"reason": e.reason, "message": str(e)[:300]}, **tag})
                    failed.append({**row_fail, "status": "error", "error_reason": "ROLE_MARGIN_UNDEFINED"})
                    print(f"  [{n}/{len(cands)}] {_short(cand.label)}: NOT SCORED [ROLE_MARGIN_UNDEFINED] {e}", file=out)
                    continue
                session._write({"type": "semantic_roles", "step": res["step"], "label": "candidate", "timestamp_utc": _utc(), "scores": cand_roles, "changes": changes, **tag})
                cmp = compare(base, base, res)
                toks = res.get("tokens") or {}
                row = {"index": cand.index, "label": cand.label, "probs": res["probs"], "margin": res["margin"], "preferred": res["preferred"],
                       "change": cmp["change_from_baseline"], "flipped": cmp["choice_flipped_from_baseline"],
                       "crossed": cmp["pairwise_crossed_from_baseline"], "movement": changes["movement_toward_competing"],
                       "role_margin": cand_roles["margin_correct_vs_competing"], "role_margin_strongest": cand_roles["margin_correct_vs_strongest"],
                       "induced": changes["newly_induced_incorrect"],
                       "fragment_tokens": toks.get("fragment_tokens"), "prompt_tokens": toks.get("prompt_tokens"), "step": res["step"],
                       "duplicate_of_step": (res.get("identical_prompt_as") or {}).get("step"), **{k: frag_info[k] for k in ("chars", "bytes", "words")},
                       "fragment_sha256": frag_info["sha256"]}
                scored.append(row)
                shown = res["margin"] if roles_default else cand_roles["margin_correct_vs_competing"]
                print(f"  [{n}/{len(cands)}] {_short(cand.label)}: margin {_signed(shown)} nats, preferred {res['preferred']}", file=out)
            reason = "batch_complete"
        finally:
            session.close(reason)
        # model identity check at the end (the baseline was measured once, at the start)
        try:
            end_hash = client.server_info().model_hash
            unchanged = end_hash == info.model_hash
        except Inconclusive:
            unchanged = None
        modal = _modal([r["prompt_tokens"] for r in scored if r["prompt_tokens"] is not None])
        for r in scored:
            r["tok_vs_modal"] = (r["prompt_tokens"] - modal) if (modal is not None and r["prompt_tokens"] is not None) else ""
        ranked = rank_rows(scored)
        csv_path = os.path.join(out_dir, "leaderboard.csv")
        _write_csv(csv_path, ranked, failed, correct, incorrect)
        hist_path = os.path.join(out_dir, "history.jsonl")
        manifest = {
            "schema": BATCH_SCHEMA, "bleediff_version": __version__, "code_revision": code_revision(), "mode": "batch",
            "measurement_type": "single_prompt_pairwise_margin", "measurement_note": MEASUREMENT_NOTE, "observed_effects_note": OBSERVED_NOTE,
            "ranking_definition": (f"movement toward {incorrect} = " + ("baseline margin - candidate margin" if incorrect == "B" else "candidate margin - baseline margin")
                                   + " (nats, from the unchanged margin_AB); ties share a rank") if roles_default else
                                  (f"movement toward {incorrect} = -(change of ln(p_correct) - ln(p_competing)) with correct = {correct} and competing = {incorrect} "
                                   "(nats, semantic roles; not the A/B margin); ties share a rank"),
            "scenario": {"id": sc.id, "fixtures_sha256": fixtures.sha256}, "designated_incorrect_answer": incorrect, "designated_correct_answer": correct,
            "semantic_roles": {"correct_letter": correct, "competing_letter": incorrect, "roles_are_ab": roles_default, "note": ROLE_NOTE},
            "candidate_file": {"name": os.path.basename(args.arms), "sha256": file_sha, "candidates": len(cands)},
            "model": {"name": name, "model_hash": info.model_hash, "model_hash_unchanged_at_end": unchanged},
            "baseline": {"probabilities": dict(zip(CHOICES, base["probs"])), "margin_AB": base["margin"], "preferred_choice": base["preferred"],
                         "prompt_tokens": base_tokens, "margin_correct_vs_competing": base_roles["margin_correct_vs_competing"],
                         "margin_correct_vs_strongest": base_roles["margin_correct_vs_strongest"], "preferred_is_correct": base_roles["preferred_is_correct"]},
            "modal_prompt_tokens": modal,
            "counts": {"candidates": len(cands), "scored": len(scored), "failed_or_invalid_or_not_run": len(failed)},
            "ranking": [{"rank": r["rank"], "tied": r["tied"], "label": r["label"], "movement_toward_incorrect_nats": r["movement"],
                         "margin_AB": r["margin"], "change_from_baseline": r["change"], "preferred_choice": r["preferred"],
                         "preferred_differs_from_baseline": r["flipped"], "fragment_sha256": r["fragment_sha256"],
                         "fragment_tokens": r["fragment_tokens"], "prompt_tokens": r["prompt_tokens"],
                         "margin_correct_vs_competing": r["role_margin"], "margin_correct_vs_strongest": r["role_margin_strongest"],
                         "newly_induced_incorrect": r["induced"]} for r in ranked],
            "unscored": [{"label": r["label"], "status": r["status"], "error_reason": r["error_reason"]} for r in failed],
            "files": {"history.jsonl": file_sha256(hist_path), "leaderboard.csv": file_sha256(csv_path)},
        }
        with open(os.path.join(out_dir, "batch-manifest.json"), "x", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")
        _print_summary(out, ranked, failed, base, base_tokens, modal, incorrect, unchanged, out_dir, roles=(None if roles_default else (correct, base_roles)), base_correct=base_roles["preferred_is_correct"])
        if not scored:
            print("ERROR [NO_CANDIDATE_SCORED]: no candidate could be scored; see the failures above.", file=out)
            return 2
        return 0
    except Inconclusive as e:
        print(f"ERROR [{e.reason}]: {e}", file=out)
        return 2


def _print_summary(out, ranked, failed, base, base_tokens, modal, incorrect, unchanged, out_dir, roles=None, base_correct=True) -> None:
    """``roles`` is None for the A/B role pair (v0.3.0 output, unchanged) or (correct_letter, baseline_role_scores) for other roles."""
    print("", file=out)
    print(f"RANKED BY MOVEMENT TOWARD {incorrect} (observed candidate effects)", file=out)
    if roles is None:
        print(f"Baseline: margin A/B {_signed(base['margin'])} nats, preferred {base['preferred']}, prompt tokens {base_tokens}", file=out)
    else:
        print(f"Baseline: margin correct({roles[0]})/competing({incorrect}) {_signed(roles[1]['margin_correct_vs_competing'])} nats, preferred {base['preferred']}, "
              f"prompt tokens {base_tokens}", file=out)
        print("  'margin' below is ln(p_correct / p_competing), the role margin; 'move->' is minus its change from the baseline (movement toward the "
              "competing answer). The A/B margin is not shown (margin_AB in the files): the roles are not A and B. 'flip' is any change of the "
              "preferred answer, a different quantity from a newly induced incorrect answer.", file=out)
    head = f"{'rank':>4}  {'label':<30} {'move->' + incorrect:>9} {'margin':>8} {'pref':>5} {'flip':>5} {'ftok':>5} {'ptok':>5} {'dTok':>5}"
    print(head, file=out)
    for r in ranked:
        rk = f"{r['rank']}{'=' if r['tied'] else ''}"
        flip = (f"{base['preferred']}>{r['preferred']}" if r["flipped"] else "no")
        dt = r["tok_vs_modal"]
        shown = r["margin"] if roles is None else r["role_margin"]
        print(f"{rk:>4}  {_short(r['label']):<30} {_signed(r['movement']):>9} {_signed(shown):>8} {r['preferred']:>5} {flip:>5} "
              f"{r['fragment_tokens'] if r['fragment_tokens'] is not None else '-':>5} {r['prompt_tokens'] if r['prompt_tokens'] is not None else '-':>5} "
              f"{(('%+d' % dt) if dt != '' else '-'):>5}", file=out)
    if any(r["tied"] for r in ranked):
        print("  '=' marks tied candidates (equal movement; they share a rank).", file=out)
    if not base_correct:
        print(f"  Baseline already prefers a wrong answer ({base['preferred']}): no change below is reported as a newly induced incorrect answer.", file=out)
    else:
        induced = [r["label"] for r in ranked if r["induced"]]
        print(f"  Newly induced incorrect answers (correct baseline, the candidate's preferred answer is wrong): {len(induced)}"
              + (": " + ", ".join(_short(x, 24) for x in induced) if induced else ""), file=out)
    sizes = sorted({r["prompt_tokens"] for r in ranked if r["prompt_tokens"] is not None})
    if len(sizes) > 1:
        print(f"  LENGTH-UNMATCHED: candidate prompt token counts range {sizes[0]}-{sizes[-1]} (modal {modal}); dTok is each candidate's "
              "difference from the modal count. Compare candidates of different lengths with care.", file=out)
    dups = [r for r in ranked if r.get("duplicate_of_step")]
    if dups:
        print("  NOTE: identical text was scored more than once under different labels: " + ", ".join(_short(r["label"], 20) for r in dups), file=out)
    for r in failed:
        print(f"  NOT SCORED: {_short(r['label'])} [{r['status']}: {r['error_reason']}]", file=out)
    if unchanged is False:
        print("  WARNING: the server's model identity changed during the run; the baseline may no longer apply.", file=out)
    elif unchanged is None:
        print("  NOTE: the model identity could not be re-checked at the end of the run.", file=out)
    print("", file=out)
    print("These are observed effects of the listed fragments on this model and prompt. A large shift does not show that a "
          "particular word caused it.", file=out)
    print(f"Saved: {os.path.join(out_dir, 'history.jsonl')}, leaderboard.csv, batch-manifest.json", file=out)
