"""``bleediff atlas``: an influence atlas over EXISTING batch-session results (v0.3, exploratory).

The atlas reads saved sessions written by ``bleediff track --arms`` (``history.jsonl`` + ``batch-manifest.json``) and compares
them across tasks and models. It never contacts a model and never recomputes a probability: every number comes from the saved
probability vectors, and the margin, the preferred answer and the movement are the unchanged ``bleediff.track`` /
``bleediff.fold`` / ``bleediff.batch`` functions (each saved margin is re-derived from the saved probabilities and any
disagreement is reported, not silently replaced).

HOW TO READ THE RESULTS
* ``shift`` (movement toward the designated incorrect answer) is measured from EACH MODEL'S OWN clean baseline. Shifts are never
  compared as raw margins between different models: baselines differ and the logit scales are model-specific. Across models the
  atlas shows signs and within-cell ranks side by side, and states when only sign/rank is comparable.
* An A/B crossing (the margin changing sign) is NOT an overall answer flip: the flip is a change of the highest-probability
  answer among A-D. Both are recorded separately.
* A model whose clean baseline already prefers a wrong answer is never reported as newly misled; its outcomes are labelled
  ``baseline_incorrect_*``.
* Ranks order candidates WITHIN one task/model cell by observed movement. They describe what was measured, not a mechanism, and
  no global ranking across unrelated tasks is produced.

LIMITATION THAT TRAVELS WITH THE ``topic`` CATEGORY
The ``topic`` category is on-topic text that ALSO MENTIONS OPTION B'S SUBJECT, with no supporting evidence. The existing
experiments cannot separate topic relevance from option-subject priming.

INPUT MANIFEST (JSON, schema ``bleediff.atlas-input/1``; paths relative to the manifest's directory)
{"schema": "bleediff.atlas-input/1",
 "sessions": [{"id": "optional-unique-id", "session_dir": "results/m1/task-a",
               "task": {"id": "task-a", "name": "Readable name"}, "scenario_id": "optional, defaults to the task id",
               "model": {"id": "m1", "name": "readable", "model_hash": "optional expected header hash"},
               "presentation": "raw", "correct_answer": "A", "incorrect_answer": "B",
               "candidates": {"B_CLAIM": "claim", "TOPIC": "topic", "OFF": "offtopic", "NEUT": "neutral", "ID": "id"},
               "evidence": [{"role": "candidate_file|fixtures|other", "path": "arms/task-a.arms.json", "sha256": "optional"}],
               "replicate_of": "optional id of a primary session this one repeats (checked for bit-identity, not counted)"}]}
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from collections import Counter

from . import CHOICES, __version__
from .batch import _csv_safe, rank_rows
from .errors import Inconclusive
from .fold import argmax_choice
from .meta import code_revision, file_sha256
from .roles import ROLE_NOTE, role_changes, role_scores, roles_are_ab
from .track import MeasurementError, margin_ab, pairwise_crossed

ATLAS_SCHEMA = "bleediff.atlas/1"
INPUT_SCHEMA = "bleediff.atlas-input/1"
BATCH_SCHEMA = "bleediff.track-batch/1"
HISTORY_SCHEMA = "bleediff.track-history/1"
MEASUREMENT_TYPE = "single_prompt_pairwise_margin"
SATURATION_P = 0.995       # baseline highest probability at or above this = saturated (descriptive flag)
LOW_COVERAGE = 0.90        # answer-letter coverage below this is flagged (descriptive flag)
DEFAULT_MARKER = 0.25      # nats; a descriptive marker for "at least this large", never a significance threshold
MARGIN_TOL = 1e-9

CATEGORY_LABELS = {
    "claim": "Explicit competing claim favoring the incorrect answer",
    "topic": "On-topic text that also mentions option B's subject (no supporting evidence)",
    "offtopic": "Off-topic text in a similar reporting structure",
    "neutral": "Ordinary unrelated prose",
    "id": "Matching identifier (no verdict, no evidence)",
}
TOPIC_LIMITATION = ("The 'topic' category is on-topic text that also mentions option B's subject. The existing experiments cannot "
                    "separate topic relevance from option-subject priming, so a topic effect here is not evidence of a topic-relevance mechanism.")
COMPARISONS = {                      # name -> (category a, category b); value = shift_a - shift_b
    "topic_minus_offtopic": ("topic", "offtopic"),
    "topic_minus_neutral": ("topic", "neutral"),
    "claim_minus_topic": ("claim", "topic"),
    "id_minus_topic": ("id", "topic"),
}
COMPARISON_NOTES = {
    "topic_minus_offtopic": "topic = on-topic text that also mentions option B's subject; relevance and option-subject priming are not separated",
    "topic_minus_neutral": "topic = on-topic text that also mentions option B's subject; relevance and option-subject priming are not separated",
    "claim_minus_topic": "explicit claim versus on-topic text that also mentions option B's subject",
    "id_minus_topic": "matching identifier versus the same on-topic text without the identifier (matched only to the extent the fragments were)",
}
OUTCOMES = ("stayed_correct", "induced_incorrect", "recovered_to_correct", "baseline_incorrect_remained_incorrect", "unscored")
CANDIDATE_COLUMNS = ["task_id", "task_name", "model_id", "presentation", "session_id", "candidate_id", "category", "status", "error_reason",
                     "correct_answer", "incorrect_answer", "p_A", "p_B", "p_C", "p_D", "coverage", "baseline_preferred", "candidate_preferred",
                     "baseline_correct", "candidate_correct", "outcome", "induced_to", "overall_flip", "ab_crossed", "baseline_margin_AB",
                     "margin_AB", "shift_toward_incorrect", "rank_in_cell", "fragment_tokens", "prompt_tokens", "prompt_tokens_minus_modal",
                     "prompt_sha256", "fragment_sha256", "flags", "correct_letter", "competing_letter", "p_correct", "p_competing", "strongest_incorrect_letter", "p_strongest_incorrect",
                     "margin_correct_vs_competing", "margin_correct_vs_strongest", "change_margin_correct_vs_competing", "change_margin_correct_vs_strongest",
                     "role_order_crossed", "newly_induced_incorrect"]


class AtlasInputError(Inconclusive):
    reason = "ATLAS_INPUT_INVALID"


# --------------------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------------------
def _display(path: str, base: str) -> str:
    """A path for reports: relative to the manifest directory when inside it, otherwise only the file name."""
    try:
        rel = os.path.relpath(os.path.abspath(path), base)
    except ValueError:
        return "<external>/" + os.path.basename(path)
    return rel.replace("\\", "/") if not rel.startswith("..") else "<external>/" + os.path.basename(path)


def _json_file(path: str):
    try:
        with open(path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8-sig"))
    except (OSError, ValueError) as e:
        raise AtlasInputError(f"cannot read {os.path.basename(path)!r}: {e}") from e


def load_manifest(path: str) -> tuple[dict, str]:
    doc = _json_file(path)
    if not isinstance(doc, dict) or doc.get("schema") != INPUT_SCHEMA:
        raise AtlasInputError(f"the manifest must be an object with schema {INPUT_SCHEMA!r}")
    sessions = doc.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise AtlasInputError("the manifest needs a non-empty 'sessions' list")
    seen_ids, seen_cells = set(), set()
    for i, s in enumerate(sessions, 1):
        where = f"session #{i}"
        if not isinstance(s, dict):
            raise AtlasInputError(f"{where} is not an object")
        for key in ("session_dir", "task", "model", "presentation", "correct_answer", "incorrect_answer", "candidates"):
            if key not in s:
                raise AtlasInputError(f"{where} is missing {key!r}")
        if not isinstance(s["task"], dict) or not s["task"].get("id") or not isinstance(s["model"], dict) or not s["model"].get("id"):
            raise AtlasInputError(f"{where}: 'task' and 'model' need an 'id'")
        if s["correct_answer"] not in CHOICES or s["incorrect_answer"] not in CHOICES or s["correct_answer"] == s["incorrect_answer"]:
            raise AtlasInputError(f"{where}: correct_answer and incorrect_answer must be two different letters from {list(CHOICES)}")
        if not isinstance(s["presentation"], str) or not s["presentation"]:
            raise AtlasInputError(f"{where}: 'presentation' must be a non-empty string")
        if not isinstance(s["candidates"], dict) or not s["candidates"] or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in s["candidates"].items()):
            raise AtlasInputError(f"{where}: 'candidates' must map candidate ids to category strings")
        s.setdefault("id", f"{s['task']['id']}/{s['model']['id']}/{s['presentation']}" + ("/replicate" if s.get("replicate_of") else ""))
        if s["id"] in seen_ids:
            raise AtlasInputError(f"duplicate session id {s['id']!r}")
        seen_ids.add(s["id"])
        if not s.get("replicate_of"):
            cell = (s["task"]["id"], s["model"]["id"], s["presentation"])
            if cell in seen_cells:
                raise AtlasInputError(f"two primary sessions for the same task/model/presentation {cell}; mark repeats with 'replicate_of'")
            seen_cells.add(cell)
    ids = {s["id"] for s in sessions}
    for s in sessions:
        r = s.get("replicate_of")
        if r and (r not in ids or r == s["id"] or any(x["id"] == r and x.get("replicate_of") for x in sessions)):
            raise AtlasInputError(f"session {s['id']!r}: 'replicate_of' must name another primary session")
    return doc, os.path.dirname(os.path.abspath(path))


def _read_session(s: dict, base: str) -> dict:
    """Read and validate one batch session. Returns {'ok', 'errors', 'warnings', 'checks', 'history', 'manifest'}."""
    out = {"id": s["id"], "session_dir": _display(os.path.join(base, s["session_dir"]), base), "errors": [], "warnings": [], "checks": {}, "ok": False}
    sd = os.path.normpath(os.path.join(base, s["session_dir"]))
    hp, mp = os.path.join(sd, "history.jsonl"), os.path.join(sd, "batch-manifest.json")
    if not (os.path.isfile(hp) and os.path.isfile(mp)):
        out["errors"].append("history.jsonl or batch-manifest.json is missing")
        return out
    try:
        with open(hp, "rb") as fh:
            hist = [json.loads(ln) for ln in fh.read().decode("utf-8").splitlines() if ln.strip()]
        man = _json_file(mp)
    except (OSError, ValueError, AtlasInputError) as e:
        out["errors"].append(f"unreadable session files: {e}")
        return out
    if not hist or hist[0].get("type") != "session_start":
        out["errors"].append("history.jsonl does not start with a session_start record")
        return out
    out["checks"]["history_schema"] = hist[0].get("schema")
    if hist[0].get("schema") != HISTORY_SCHEMA:
        out["errors"].append(f"unexpected history schema {hist[0].get('schema')!r}")
    if man.get("schema") != BATCH_SCHEMA or man.get("mode") != "batch":
        out["errors"].append("batch-manifest.json is not a batch-mode manifest of the expected schema")
    if hist[0].get("measurement_type") != MEASUREMENT_TYPE or man.get("measurement_type") != MEASUREMENT_TYPE:
        out["errors"].append(f"measurement type is not {MEASUREMENT_TYPE!r}")
    # recorded hashes
    for name, path in (("history.jsonl", hp), ("leaderboard.csv", os.path.join(sd, "leaderboard.csv"))):
        recorded = (man.get("files") or {}).get(name)
        if recorded is None:
            out["warnings"].append(f"no recorded hash for {name}")
        elif not os.path.isfile(path):
            out["errors"].append(f"{name} is missing but its hash is recorded")
        else:
            ok = file_sha256(path) == recorded
            out["checks"][f"{name}_hash_matches_record"] = ok
            if not ok:
                out["errors"].append(f"{name} does not match the hash recorded in batch-manifest.json")
    # identity cross-checks
    if hist[0].get("scenario", {}).get("id") != s.get("scenario_id", s["task"]["id"]):
        out["errors"].append(f"scenario id {hist[0].get('scenario', {}).get('id')!r} does not match the task/scenario id {s.get('scenario_id', s['task']['id'])!r}")
    if man.get("designated_correct_answer") not in (None, s["correct_answer"]):
        out["errors"].append(f"the session designated {man.get('designated_correct_answer')!r} as correct, the atlas manifest says {s['correct_answer']!r}")
    if man.get("designated_incorrect_answer") != s["incorrect_answer"]:
        out["errors"].append(f"the session designated {man.get('designated_incorrect_answer')!r} as incorrect, the atlas manifest says {s['incorrect_answer']!r}")
    exp_hash = s["model"].get("model_hash")
    sess_hash = (hist[0].get("model") or {}).get("model_hash")
    if exp_hash and exp_hash != sess_hash:
        out["errors"].append("the session's model hash differs from the model hash in the atlas manifest")
    if not any(r.get("type") == "session_end" for r in hist):
        out["warnings"].append("the session has no session_end record (incomplete)")
        out["checks"]["session_end"] = False
    # evidence files
    ev_report = []
    for ev in s.get("evidence", []) or []:
        p = os.path.normpath(os.path.join(base, ev.get("path", "")))
        item = {"role": ev.get("role", "other"), "path": _display(p, base), "exists": os.path.isfile(p), "sha256": None, "hash_ok": None}
        if item["exists"]:
            item["sha256"] = file_sha256(p)
            if ev.get("sha256"):
                item["hash_ok"] = item["sha256"] == ev["sha256"]
                if not item["hash_ok"]:
                    out["errors"].append(f"evidence file {item['path']} does not match its recorded sha256")
            if item["role"] == "candidate_file" and (man.get("candidate_file") or {}).get("sha256") != item["sha256"]:
                out["errors"].append(f"evidence file {item['path']} is not the candidate file the session was run with")
            if item["role"] == "fixtures" and (man.get("scenario") or {}).get("fixtures_sha256") != item["sha256"]:
                out["errors"].append(f"evidence file {item['path']} is not the fixtures file the session was run with")
        else:
            out["errors"].append(f"evidence file {item['path']} is missing")
        ev_report.append(item)
    out["evidence"] = ev_report
    out["history"], out["manifest"] = hist, man
    out["ok"] = not out["errors"]
    return out


# --------------------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------------------
def classify(base_pref: str, cand_pref: str, correct: str, incorrect: str) -> tuple[str, str | None]:
    """Outcome relative to the baseline's correctness. A wrong baseline is never reported as newly misled."""
    bc, cc = base_pref == correct, cand_pref == correct
    if bc and cc:
        return "stayed_correct", None
    if bc and not cc:
        return "induced_incorrect", "designated_incorrect" if cand_pref == incorrect else "other_incorrect"
    if not bc and cc:
        return "recovered_to_correct", None
    return "baseline_incorrect_remained_incorrect", None


def _probs(rec: dict) -> list[float]:
    p = rec.get("probabilities") or {}
    return [p.get(c) for c in CHOICES]


def _score(rec: dict, problems: list[str]) -> dict | None:
    """Validate one saved ok-measurement with the unchanged functions; returns the verified numbers or None."""
    probs = _probs(rec)
    try:
        m = margin_ab(probs)                 # validates finiteness, non-negativity, sum and positive A and B
    except MeasurementError as e:
        problems.append(f"invalid saved probabilities: {e}")
        return None
    pref = argmax_choice(probs)
    if rec.get("margin_AB") is None or abs(rec["margin_AB"] - m) > MARGIN_TOL:
        problems.append("the saved margin_AB differs from the margin derived from the saved probabilities")
    if rec.get("preferred_choice") != pref:
        problems.append("the saved preferred_choice differs from the arg-max of the saved probabilities")
    cov = (rec.get("coverage") or {}).get("value")
    return {"probs": probs, "margin": m, "preferred": pref, "coverage": cov if isinstance(cov, (int, float)) and math.isfinite(cov) else None}


def build_cell(s: dict, src: dict) -> dict:
    hist, man = src["history"], src["manifest"]
    correct, incorrect = s["correct_answer"], s["incorrect_answer"]
    recs = [r for r in hist if r.get("type") == "measurement"]
    base_rec = next((r for r in recs if r.get("label") == "baseline"), None)
    cell = {"session_id": s["id"], "task_id": s["task"]["id"], "task_name": s["task"].get("name") or s["task"]["id"], "model_id": s["model"]["id"],
            "model_name": s["model"].get("name") or hist[0].get("model", {}).get("name"), "model_hash": hist[0].get("model", {}).get("model_hash"),
            "presentation": s["presentation"], "correct_answer": correct, "incorrect_answer": incorrect, "source": src["session_dir"],
            "scenario": hist[0].get("scenario"), "prompt_template_sha256": hist[0].get("prompt_template_sha256"), "code_revision": hist[0].get("code_revision"),
            "flags": [], "candidates": [], "baseline": None, "problems": []}
    bprob: list[str] = []
    bs = _score(base_rec, bprob) if base_rec and base_rec.get("status") == "ok" else None
    if bs is None:
        cell["problems"] += bprob or ["the session has no valid baseline measurement"]
        cell["flags"].append("INVALID_BASELINE")
        return cell
    cell["problems"] += bprob
    try:
        base_roles = role_scores(bs["probs"], correct, incorrect)
    except MeasurementError as e:
        cell["problems"].append(f"the baseline has no defined role margin: {e}")
        cell["flags"].append("INVALID_BASELINE")
        return cell
    cell["roles_are_ab"] = roles_are_ab(correct, incorrect)
    if not cell["roles_are_ab"]:
        cell["flags"].append("AB_MARGIN_NOT_ROLE_MARGIN")
    b_ok = bs["preferred"] == correct
    cell["baseline"] = {"probabilities": dict(zip(CHOICES, bs["probs"])), "preferred": bs["preferred"], "correct": b_ok, "margin_AB": bs["margin"], "coverage": bs["coverage"],
                        "roles": base_roles, "margin_correct_vs_competing": base_roles["margin_correct_vs_competing"], "margin_correct_vs_strongest": base_roles["margin_correct_vs_strongest"],
                        "prompt_tokens": (base_rec.get("tokens") or {}).get("prompt_tokens"), "prompt_sha256": (base_rec.get("prompt") or {}).get("sha256")}
    if not b_ok:
        cell["flags"].append("BASELINE_INCORRECT")
    if max(bs["probs"]) >= SATURATION_P:
        cell["flags"].append("SATURATED_BASELINE")
    if bs["coverage"] is not None and bs["coverage"] < LOW_COVERAGE:
        cell["flags"].append("LOW_COVERAGE")
    cats = s["candidates"]
    seen = set()
    for r in recs:
        if r.get("label") != "candidate":
            continue
        cid = (r.get("candidate") or {}).get("label")
        seen.add(cid)
        problems: list[str] = []
        sc = _score(r, problems) if r.get("status") == "ok" else None
        c = {"candidate_id": cid, "index": (r.get("candidate") or {}).get("index"), "category": cats.get(cid, "uncategorized"), "category_label": CATEGORY_LABELS.get(cats.get(cid, ""), None),
             "status": "ok" if sc is not None else ("invalid" if r.get("status") == "ok" else r.get("status")),
             "error_reason": ((r.get("error") or {}).get("reason") if r.get("status") != "ok" else None), "problems": problems,
             "fragment_sha256": (r.get("fragment") or {}).get("sha256"), "fragment_bytes": (r.get("fragment") or {}).get("bytes"),
             "fragment_tokens": (r.get("tokens") or {}).get("fragment_tokens"), "prompt_tokens": (r.get("tokens") or {}).get("prompt_tokens"),
             "prompt_sha256": (r.get("prompt") or {}).get("sha256"), "flags": []}
        if cid not in cats:
            c["flags"].append("UNCATEGORIZED")
        if sc is None or c["status"] != "ok":
            c["outcome"] = "unscored"
            c["flags"].append("INVALID_SCORE")
            cell["candidates"].append(c)
            continue
        try:
            cand_roles = role_scores(sc["probs"], correct, incorrect)
            changes = role_changes(base_roles, cand_roles)
        except MeasurementError as e:
            problems.append(f"no defined role margin: {e}")
            c["status"], c["outcome"] = "invalid", "unscored"
            c["flags"].append("INVALID_SCORE")
            cell["candidates"].append(c)
            continue
        outcome, to = classify(bs["preferred"], sc["preferred"], correct, incorrect)
        c.update({"probabilities": dict(zip(CHOICES, sc["probs"])), "coverage": sc["coverage"], "candidate_preferred": sc["preferred"], "candidate_correct": sc["preferred"] == correct,
                  "semantic": {**cand_roles, **changes},
                  "margin_AB": sc["margin"], "shift_toward_incorrect": changes["movement_toward_competing"],
                  "overall_flip": sc["preferred"] != bs["preferred"], "ab_crossed": pairwise_crossed(bs["margin"], sc["margin"]), "outcome": outcome, "induced_to": to})
        if sc["coverage"] is not None and sc["coverage"] < LOW_COVERAGE:
            c["flags"].append("LOW_COVERAGE")
        if "AB_MARGIN_NOT_ROLE_MARGIN" in cell["flags"]:
            c["flags"].append("AB_MARGIN_NOT_ROLE_MARGIN")
        if "BASELINE_INCORRECT" in cell["flags"]:
            c["flags"].append("BASELINE_INCORRECT")
        if "SATURATED_BASELINE" in cell["flags"]:
            c["flags"].append("SATURATED_BASELINE")
        if any("differs" in p for p in problems):
            c["flags"].append("SAVED_VALUE_MISMATCH")
        cell["candidates"].append(c)
    for missing in sorted(set(cats) - seen):
        cell["problems"].append(f"candidate id {missing!r} named in the atlas manifest is not in the session")
    # lengths
    pts = [c["prompt_tokens"] for c in cell["candidates"] if c.get("prompt_tokens") is not None]
    if pts:
        cnt = Counter(pts)
        top = max(cnt.values())
        modal = min(v for v, n in cnt.items() if n == top)
        cell["modal_prompt_tokens"] = modal
        for c in cell["candidates"]:
            c["prompt_tokens_minus_modal"] = (c["prompt_tokens"] - modal) if c.get("prompt_tokens") is not None else None
            if c.get("prompt_tokens_minus_modal"):
                c["flags"].append("TOKEN_MISMATCH")
        if len(cnt) > 1:
            cell["flags"].append("TOKEN_MISMATCH")
    if any("LOW_COVERAGE" in c["flags"] for c in cell["candidates"]) and "LOW_COVERAGE" not in cell["flags"]:
        cell["flags"].append("LOW_COVERAGE")
    ranked = rank_rows([{"index": c["index"] if c["index"] is not None else i, "movement": c["shift_toward_incorrect"], "_c": c} for i, c in enumerate(cell["candidates"]) if c.get("outcome") != "unscored"])
    for r in ranked:
        r["_c"]["rank_in_cell"], r["_c"]["rank_tied"] = r["rank"], r["tied"]
    cell["comparisons"] = paired(cell)
    return cell


def paired(cell: dict) -> dict:
    """Paired shift differences inside one task/model cell. Computed only when both categories have at least one validly scored
    candidate; several candidates of one category are averaged (n is recorded)."""
    out = {}
    for name, (a, b) in COMPARISONS.items():
        xa = [c for c in cell["candidates"] if c["category"] == a and c.get("outcome") != "unscored"]
        xb = [c for c in cell["candidates"] if c["category"] == b and c.get("outcome") != "unscored"]
        if not xa or not xb:
            out[name] = {"value": None, "reason": "missing or unscored category: " + ", ".join(k for k, v in ((a, xa), (b, xb)) if not v), "note": COMPARISON_NOTES[name]}
            continue
        va, vb = statistics.fmean(c["shift_toward_incorrect"] for c in xa), statistics.fmean(c["shift_toward_incorrect"] for c in xb)
        ta = [c["prompt_tokens"] for c in xa if c.get("prompt_tokens") is not None]
        tb = [c["prompt_tokens"] for c in xb if c.get("prompt_tokens") is not None]
        dtok = (statistics.fmean(ta) - statistics.fmean(tb)) if ta and tb else None
        fl = []
        if dtok:
            fl.append("LENGTH_UNMATCHED")
        out[name] = {"value": va - vb, "shift_a": va, "shift_b": vb, "n_a": len(xa), "n_b": len(xb), "prompt_tokens_a_minus_b": dtok, "flags": fl, "reason": None, "note": COMPARISON_NOTES[name]}
    return out


# --------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------
def _stats(values: list[float], marker: float) -> dict:
    if not values:
        return {"n": 0, "n_positive": 0, "n_negative": 0, "n_at_least_marker": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": len(values), "n_positive": sum(v > 0 for v in values), "n_negative": sum(v < 0 for v in values), "n_at_least_marker": sum(v >= marker for v in values),
            "mean": statistics.fmean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def aggregate(cells: list[dict], marker: float) -> dict:
    """Per (model, presentation) paired-effect statistics, and cell-count totals pooled over models (counts only)."""
    per = {}
    for comp in COMPARISONS:
        groups: dict[tuple, list] = {}
        for c in cells:
            v = (c.get("comparisons") or {}).get(comp, {}).get("value")
            if v is not None:
                groups.setdefault((c["model_id"], c["presentation"]), []).append(v)
        per[comp] = {f"{m}|{p}": _stats(v, marker) for (m, p), v in groups.items()}
    pooled = {}
    for comp in COMPARISONS:
        by_pres: dict[str, list] = {}
        for c in cells:
            v = (c.get("comparisons") or {}).get(comp, {}).get("value")
            if v is not None:
                by_pres.setdefault(c["presentation"], []).append(v)
        pooled[comp] = {p: {k: v for k, v in _stats(x, marker).items() if k.startswith("n")} for p, x in by_pres.items()}
    outcomes = {}
    for c in cells:
        for k in c["candidates"]:
            key = f"{c['model_id']}|{k['category']}"
            outcomes.setdefault(key, Counter())[k["outcome"]] += 1
    return {"per_model": per, "pooled_counts": pooled, "outcomes_by_model_and_category": {k: dict(v) for k, v in outcomes.items()}}


def model_comparison(cells: list[dict]) -> list[dict]:
    """Matched effects across models: same task AND same presentation AND the same candidate texts. Values are shifts measured from
    each model's own baseline; only signs and within-cell ranks are comparable across models, not magnitudes."""
    groups: dict[tuple, list] = {}
    for c in cells:
        groups.setdefault((c["task_id"], c["presentation"]), []).append(c)
    rows = []
    for (task, pres), cs in groups.items():
        base_hashes = {c["model_id"]: (c["baseline"] or {}).get("prompt_sha256") for c in cs}
        compatible = len(set(base_hashes.values())) == 1 and None not in base_hashes.values()
        frag = {c["model_id"]: {k["category"]: k["fragment_sha256"] for k in c["candidates"]} for c in cs}
        same_text = all(frag[m] == frag[cs[0]["model_id"]] for m in frag)
        for metric, getter in [(f"shift:{cat}", ("shift", cat)) for cat in CATEGORY_LABELS] + [(name, ("cmp", name)) for name in COMPARISONS]:
            vals, ranks = {}, {}
            for c in cs:
                if getter[0] == "shift":
                    ks = [k for k in c["candidates"] if k["category"] == getter[1] and k.get("outcome") != "unscored"]
                    v = statistics.fmean(k["shift_toward_incorrect"] for k in ks) if ks else None
                    r = statistics.fmean(k["rank_in_cell"] for k in ks) if ks else None
                else:
                    v = (c.get("comparisons") or {}).get(getter[1], {}).get("value")
                    r = None
                vals[c["model_id"]], ranks[c["model_id"]] = v, r
            have = {m: v for m, v in vals.items() if v is not None}
            signs = {(v > 0) - (v < 0) for v in have.values()}
            ok = compatible and same_text
            rows.append({"task_id": task, "presentation": pres, "metric": metric, "values": vals, "ranks_in_cell": ranks, "n_models": len(have),
                         "same_sign_all_models": (len(signs) == 1 and 0 not in signs) if ok and len(have) >= 2 else None,
                         "compatible": ok, "incompatibility": None if ok else ("the models were not given identical prompts" if not compatible else "candidate texts differ between models"),
                         "note": "values are shifts from each model's own baseline; compare signs and within-cell ranks across models, not magnitudes"})
    return rows


# --------------------------------------------------------------------------------------
# the atlas
# --------------------------------------------------------------------------------------
def build_atlas(manifest_path: str, *, marker: float = DEFAULT_MARKER, keep_going: bool = False) -> dict:
    doc, base = load_manifest(manifest_path)
    sources, cells, primaries = [], [], {}
    for s in doc["sessions"]:
        src = _read_session(s, base)
        rep = {k: v for k, v in src.items() if k not in ("history", "manifest")}
        rep["role"] = "replicate" if s.get("replicate_of") else "primary"
        sources.append(rep)
        if not src["ok"]:
            if not keep_going:
                raise AtlasInputError(f"source {s['id']!r} failed validation: " + "; ".join(src["errors"]) + " (use --keep-going to exclude invalid sources and continue)")
            continue
        if s.get("replicate_of"):
            primaries.setdefault("_replicas", []).append((s, src))
            continue
        cell = build_cell(s, src)
        cell["_src"] = src
        cells.append(cell)
        primaries[s["id"]] = cell
    replicate_checks = []
    for s, src in primaries.get("_replicas", []):
        prim = primaries.get(s["replicate_of"])
        if prim is None:
            replicate_checks.append({"session_id": s["id"], "replicate_of": s["replicate_of"], "bit_identical": None, "reason": "the primary session was excluded"})
            continue
        other = build_cell(s, src)
        a = {c["candidate_id"]: c for c in prim["candidates"]}
        b = {c["candidate_id"]: c for c in other["candidates"]}
        ident = (prim["baseline"] or {}).get("probabilities") == (other["baseline"] or {}).get("probabilities") and a.keys() == b.keys() and all(
            a[k].get("probabilities") == b[k].get("probabilities") and a[k].get("margin_AB") == b[k].get("margin_AB") and a[k].get("prompt_sha256") == b[k].get("prompt_sha256") for k in a)
        replicate_checks.append({"session_id": s["id"], "replicate_of": s["replicate_of"], "bit_identical": bool(ident), "candidates_compared": len(a)})
    for c in cells:
        c.pop("_src", None)
    presentations: dict[str, set] = {}
    for c in cells:
        presentations.setdefault(c["task_id"], set()).add(c["presentation"])
    warnings = [f"task {t!r} appears under several prompt presentations ({sorted(p)}); they are never combined" for t, p in sorted(presentations.items()) if len(p) > 1]
    tpl = {}
    for c in cells:
        tpl.setdefault(c["presentation"], set()).add(c["prompt_template_sha256"])
    warnings += [f"presentation {p!r} has sessions with different prompt templates; check that they are comparable" for p, h in tpl.items() if len(h) > 1]
    for src in sources:
        warnings += [f"{src['id']}: {w}" for w in src.get("warnings", [])]
    return {
        "schema": ATLAS_SCHEMA, "bleediff_version": __version__, "code_revision": code_revision(), "measurement_type": MEASUREMENT_TYPE,
        "input_manifest": {"name": os.path.basename(manifest_path), "sha256": file_sha256(manifest_path)}, "effect_marker_nats": marker,
        "thresholds": {"saturated_baseline_p": SATURATION_P, "low_coverage_below": LOW_COVERAGE},
        "limitations": [TOPIC_LIMITATION,
                        "Shifts are measured from each model's own clean baseline; raw margins are not compared across models and magnitudes are model-specific.",
                        "Observed effects are not mechanisms: a rank orders candidates within one task/model cell only.",
                        "The effect marker is a descriptive size, not a significance threshold; small numbers of tasks and models are not evidence of a universal effect."],
        "category_labels": CATEGORY_LABELS, "sources": sources, "replicate_checks": replicate_checks, "warnings": warnings, "cells": cells,
        "aggregate": aggregate(cells, marker), "model_comparison": model_comparison(cells)}


# --------------------------------------------------------------------------------------
# filtering and writing
# --------------------------------------------------------------------------------------
def _keep_cell(c: dict, f: dict) -> bool:
    return (not f.get("model") or c["model_id"] in f["model"]) and (not f.get("task") or c["task_id"] in f["task"])


def _keep_cand(k: dict, f: dict) -> bool:
    return (not f.get("category") or k["category"] in f["category"]) and (not f.get("outcome") or k.get("outcome") in f["outcome"])


def _semantic_columns(c: dict, k: dict) -> dict:
    sm = k.get("semantic")
    if not sm:
        return {col: "" for col in CANDIDATE_COLUMNS[-12:]}
    return {"correct_letter": sm["correct_letter"], "competing_letter": sm["competing_letter"], "p_correct": sm["p_correct"], "p_competing": sm["p_competing"],
            "strongest_incorrect_letter": sm["strongest_incorrect_letter"], "p_strongest_incorrect": sm["p_strongest_incorrect"],
            "margin_correct_vs_competing": sm["margin_correct_vs_competing"], "margin_correct_vs_strongest": sm["margin_correct_vs_strongest"],
            "change_margin_correct_vs_competing": sm["change_margin_correct_vs_competing"], "change_margin_correct_vs_strongest": sm["change_margin_correct_vs_strongest"],
            "role_order_crossed": sm["role_order_crossed"], "newly_induced_incorrect": sm["newly_induced_incorrect"]}


def candidate_rows(atlas: dict, f: dict) -> list[dict]:
    rows = []
    for c in atlas["cells"]:
        if not _keep_cell(c, f):
            continue
        for k in sorted(c["candidates"], key=lambda x: (x.get("rank_in_cell") or 10**6, x.get("index") or 0)):
            if not _keep_cand(k, f):
                continue
            p = k.get("probabilities") or {}
            rows.append({"task_id": c["task_id"], "task_name": c["task_name"], "model_id": c["model_id"], "presentation": c["presentation"], "session_id": c["session_id"],
                         "candidate_id": k["candidate_id"], "category": k["category"], "status": k["status"], "error_reason": k.get("error_reason") or "",
                         "correct_answer": c["correct_answer"], "incorrect_answer": c["incorrect_answer"], "p_A": p.get("A", ""), "p_B": p.get("B", ""), "p_C": p.get("C", ""), "p_D": p.get("D", ""),
                         "coverage": k.get("coverage") if k.get("coverage") is not None else "", "baseline_preferred": c["baseline"]["preferred"] if c["baseline"] else "",
                         "candidate_preferred": k.get("candidate_preferred", ""), "baseline_correct": c["baseline"]["correct"] if c["baseline"] else "",
                         "candidate_correct": k.get("candidate_correct", ""), "outcome": k.get("outcome", ""), "induced_to": k.get("induced_to") or "",
                         "overall_flip": k.get("overall_flip", ""), "ab_crossed": k.get("ab_crossed", ""),
                         "baseline_margin_AB": c["baseline"]["margin_AB"] if c["baseline"] else "", "margin_AB": k.get("margin_AB", ""),
                         "shift_toward_incorrect": k.get("shift_toward_incorrect", ""), "rank_in_cell": k.get("rank_in_cell", ""), "fragment_tokens": k.get("fragment_tokens", ""),
                         "prompt_tokens": k.get("prompt_tokens", ""), "prompt_tokens_minus_modal": k.get("prompt_tokens_minus_modal", "") if k.get("prompt_tokens_minus_modal") is not None else "",
                         "prompt_sha256": k.get("prompt_sha256") or "", "fragment_sha256": k.get("fragment_sha256") or "",
                         **_semantic_columns(c, k),
                         "flags": ";".join(sorted(set(k["flags"])))})
    return rows


def category_summary_rows(atlas: dict, f: dict, marker: float) -> list[dict]:
    rows = []
    cells = [c for c in atlas["cells"] if _keep_cell(c, f)]
    for comp in COMPARISONS:
        for c in cells:
            v = (c.get("comparisons") or {}).get(comp, {})
            rows.append({"scope": "cell", "model_id": c["model_id"], "presentation": c["presentation"], "comparison": comp, "task_id": c["task_id"], "value": "" if v.get("value") is None else v["value"],
                         "n": "", "n_positive": "", "n_negative": "", "n_at_least_marker": "", "mean": "", "median": "", "min": "", "max": "",
                         "flags": ";".join((v.get("flags") or []) + [x for x in c["flags"] if x in ("BASELINE_INCORRECT", "SATURATED_BASELINE", "LOW_COVERAGE")]),
                         "excluded_reason": v.get("reason") or "", "note": COMPARISON_NOTES[comp]})
        groups: dict[tuple, list] = {}
        for c in cells:
            v = (c.get("comparisons") or {}).get(comp, {}).get("value")
            if v is not None:
                groups.setdefault((c["model_id"], c["presentation"]), []).append(v)
        for (m, p), vals in sorted(groups.items()):
            s = _stats(vals, marker)
            rows.append({"scope": "model", "model_id": m, "presentation": p, "comparison": comp, "task_id": "", "value": "", **{k: ("" if s[k] is None else s[k]) for k in ("n", "n_positive", "n_negative", "n_at_least_marker", "mean", "median", "min", "max")},
                         "flags": "", "excluded_reason": "", "note": COMPARISON_NOTES[comp]})
        pools: dict[str, list] = {}
        for c in cells:
            v = (c.get("comparisons") or {}).get(comp, {}).get("value")
            if v is not None:
                pools.setdefault(c["presentation"], []).append(v)
        for p, vals in sorted(pools.items()):
            s = _stats(vals, marker)
            rows.append({"scope": "all_models_counts_only", "model_id": "ALL", "presentation": p, "comparison": comp, "task_id": "", "value": "", "n": s["n"], "n_positive": s["n_positive"],
                         "n_negative": s["n_negative"], "n_at_least_marker": s["n_at_least_marker"], "mean": "", "median": "", "min": "", "max": "", "flags": "",
                         "excluded_reason": "", "note": COMPARISON_NOTES[comp] + "; cell counts pooled over models, magnitudes are not pooled (model-specific scales)"})
    return rows


def model_comparison_rows(atlas: dict, f: dict) -> tuple[list[str], list[dict]]:
    models = [m for m in dict.fromkeys(c["model_id"] for c in atlas["cells"]) if not f.get("model") or m in f["model"]]
    cols = ["task_id", "presentation", "metric"] + [f"value__{m}" for m in models] + [f"rank_in_cell__{m}" for m in models] + ["n_models", "same_sign_all_models", "compatible", "incompatibility", "note"]
    rows = []
    for r in atlas["model_comparison"]:
        if f.get("task") and r["task_id"] not in f["task"]:
            continue
        row = {"task_id": r["task_id"], "presentation": r["presentation"], "metric": r["metric"], "n_models": r["n_models"],
               "same_sign_all_models": "" if r["same_sign_all_models"] is None else r["same_sign_all_models"], "compatible": r["compatible"], "incompatibility": r["incompatibility"] or "", "note": r["note"]}
        for m in models:
            row[f"value__{m}"] = "" if r["values"].get(m) is None else r["values"][m]
            row[f"rank_in_cell__{m}"] = "" if r["ranks_in_cell"].get(m) is None else r["ranks_in_cell"][m]
        rows.append(row)
    return cols, rows


def _write_csv(path: str, cols: list[str], rows: list[dict]) -> None:
    with open(path, "x", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([_csv_safe(r.get(c, "")) if isinstance(r.get(c, ""), str) else r.get(c, "") for c in cols])


def _fmt(x, d=2):
    return "-" if x is None or x == "" else f"{x:+.{d}f}"


def _cell_shifts(c: dict) -> dict:
    out = {}
    for cat in CATEGORY_LABELS:
        ks = [k for k in c["candidates"] if k["category"] == cat and k.get("outcome") != "unscored"]
        out[cat] = statistics.fmean(k["shift_toward_incorrect"] for k in ks) if ks else None
    return out


def _shown_margin(c: dict):
    """The baseline margin to display: the A/B margin when the roles are A and B (v0.3.0 output), otherwise the role margin."""
    b = c.get("baseline") or {}
    return b.get("margin_AB") if c.get("roles_are_ab", True) else b.get("margin_correct_vs_competing")


def summary_lines(atlas: dict, f: dict, marker: float) -> list[str]:
    cells = [c for c in atlas["cells"] if _keep_cell(c, f)]
    lines = ["BLEEDIFF ATLAS (observed candidate effects; reads saved results, never queries a model)",
             f"Sources: {sum(1 for s in atlas['sources'] if s['ok'])}/{len(atlas['sources'])} validated, "
             f"{sum(1 for r in atlas['replicate_checks'] if r.get('bit_identical'))}/{len(atlas['replicate_checks'])} repeat sessions bit-identical to their primary. "
             f"Cells: {len(cells)}, candidates: {sum(len(c['candidates']) for c in cells)}.",
             "LIMITATION: " + TOPIC_LIMITATION, ""]
    if any(f.values()):
        lines.append("Filters: " + ", ".join(f"{k}={'/'.join(v)}" for k, v in f.items() if v) + " (the pooled counts below use only the selected cells)")
    lines.append("Shift toward the incorrect answer (nats, from each model's own baseline; compare signs and ranks across models, not magnitudes)")
    lines.append(f"{'task':<20}{'model':<11}{'base':>5}{'m':>7}{'claim':>7}{'topic*':>7}{'offtp':>7}{'neutr':>7}{'ident':>7}  flags")
    for c in cells:
        sh = _cell_shifts(c)
        fl = ",".join(sorted(set(c["flags"]) - {"LOW_COVERAGE"} | ({"LOWCOV"} if "LOW_COVERAGE" in c["flags"] else set())))
        flips = sorted({k["candidate_id"] for k in c["candidates"] if k.get("outcome") == "induced_incorrect"})
        lines.append(f"{c['task_id'][:19]:<20}{c['model_id'][:10]:<11}{(c['baseline'] or {}).get('preferred', '?'):>5}{_fmt(_shown_margin(c)):>7}"
                     f"{_fmt(sh['claim']):>7}{_fmt(sh['topic']):>7}{_fmt(sh['offtopic']):>7}{_fmt(sh['neutral']):>7}{_fmt(sh['id']):>7}  {fl}" + (f"  INDUCED-WRONG: {','.join(flips)}" if flips else ""))
    lines.append("  *topic = on-topic text that also mentions option B's subject.")
    nonab = [c for c in cells if c.get("roles_are_ab") is False]
    if nonab:
        lines.append(f"  NOTE: in {len(nonab)} cell(s) the correct and competing answers are not at A and B. For those cells 'm' and every shift are "
                     "role margins ln(p_correct / p_competing); the A/B margin (margin_AB in atlas.json) is a different quantity there, and the "
                     "'topic' category is on-topic text that mentions the competing answer's subject.")
    lines.append("")
    lines.append(f"Paired differences, cell counts pooled over models (effect marker {marker:g} nats)")
    for comp in COMPARISONS:
        by: dict[str, list] = {}
        for c in cells:
            v = (c.get("comparisons") or {}).get(comp, {}).get("value")
            if v is not None:
                by.setdefault(c["presentation"], []).append(v)
        for p, vals in sorted(by.items()):
            s = _stats(vals, marker)
            lines.append(f"  {comp:<22} [{p}] positive {s['n_positive']}/{s['n']}, at least {marker:g}: {s['n_at_least_marker']}/{s['n']}")
    lines.append("")
    induced = [(c, k) for c in cells for k in c["candidates"] if k.get("outcome") == "induced_incorrect" and _keep_cand(k, f)]
    lines.append(f"Induced incorrect preferred answers (baseline correct, candidate not): {len(induced)}")
    for c, k in induced:
        lines.append(f"  {c['task_id']} / {c['model_id']}: {k['candidate_id']} ({k['category']}) {c['baseline']['preferred']} -> {k['candidate_preferred']}")
    wrong = [c for c in cells if "BASELINE_INCORRECT" in c["flags"]]
    for c in wrong:
        lines.append(f"  NOT counted as induced: {c['task_id']} / {c['model_id']} baseline already prefers {c['baseline']['preferred']} (correct is {c['correct_answer']})")
    cross = [(c, k) for c in cells for k in c["candidates"] if k.get("ab_crossed") and not k.get("overall_flip") and _keep_cand(k, f)]
    if cross:
        lines.append(f"A/B crossings that are NOT overall answer flips: {len(cross)} (the arg-max answer did not change)")
    for w in atlas["warnings"]:
        lines.append("WARNING: " + w)
    return lines


def write_report(path: str, atlas: dict, f: dict, marker: float) -> None:
    cells = [c for c in atlas["cells"] if _keep_cell(c, f)]
    L = ["# Influence Atlas report", "",
         "Observed candidate effects aggregated from saved batch sessions. No model was queried to produce this report.", "",
         "> **Limitation.** " + TOPIC_LIMITATION, "",
         "## How to read this", "",
         "- *Shift* is movement toward the designated incorrect answer, in nats, from that model's own clean baseline. Magnitudes are model-specific; "
         "across models compare signs and within-cell ranks only.",
         "- An A/B crossing is not an overall answer flip. A model whose baseline was already wrong is not reported as newly misled.",
         "- Ranks order candidates inside one task/model cell. They do not establish a mechanism, and there is no global ranking across tasks.",
         f"- The {marker:g}-nat marker is a descriptive size, not a significance threshold.", ""]
    if any(f.values()):
        L += ["Filters applied: " + ", ".join(f"{k}={'/'.join(v)}" for k, v in f.items() if v), ""]
    L += ["## Sources", "", f"{sum(1 for s in atlas['sources'] if s['ok'])} of {len(atlas['sources'])} sessions validated (recorded hashes, schemas, scenario and model identity)."]
    for r in atlas["replicate_checks"]:
        L.append(f"- Repeat `{r['session_id']}` of `{r['replicate_of']}`: bit-identical = {r.get('bit_identical')}")
    for s in atlas["sources"]:
        if not s["ok"]:
            L.append(f"- EXCLUDED `{s['id']}`: " + "; ".join(s["errors"]))
    L += ["", "## Semantic answer roles", "", ROLE_NOTE,
          "Baseline margins in the Task/model cells table are the A/B margin when the roles are A and B, and the role margin otherwise. Shifts and paired "
          "differences are always changes of the role margin ln(p_correct / p_competing). For these roles 'topic' means on-topic text that mentions the "
          "competing answer's subject.", "",
          "| Task | Model | Correct letter | Competing letter | Roles are A/B | Baseline p(correct) | Baseline p(competing) | Baseline role margin | Baseline preferred |", "|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        r = (c.get("baseline") or {}).get("roles")
        if r:
            L.append(f"| {c['task_id']} | {c['model_id']} | {r['correct_letter']} | {r['competing_letter']} | {r['roles_are_ab']} | {r['p_correct']:.3f} | {r['p_competing']:.3f} | {_fmt(r['margin_correct_vs_competing'])} | {c['baseline']['preferred']} |")
    L += ["", "## Task/model cells", "", "| Task | Model | Presentation | Baseline (margin) | Claim | On-topic* | Off-topic | Neutral | Identifier | Flags |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        sh = _cell_shifts(c)
        b = c["baseline"] or {}
        L.append(f"| {c['task_id']} | {c['model_id']} | {c['presentation']} | {b.get('preferred', '?')} ({_fmt(_shown_margin(c))}) | {_fmt(sh['claim'])} | {_fmt(sh['topic'])} | {_fmt(sh['offtopic'])} | {_fmt(sh['neutral'])} | {_fmt(sh['id'])} | {', '.join(c['flags']) or '-'} |")
    L += ["", "\\*On-topic text that also mentions option B's subject.", "", "## Paired differences (cell counts pooled over models)", "", "| Comparison | Presentation | Positive | At least marker | Cells |", "|---|---|---|---|---|"]
    for comp in COMPARISONS:
        by: dict[str, list] = {}
        for c in cells:
            v = (c.get("comparisons") or {}).get(comp, {}).get("value")
            if v is not None:
                by.setdefault(c["presentation"], []).append(v)
        for p, vals in sorted(by.items()):
            s = _stats(vals, marker)
            L.append(f"| {comp} | {p} | {s['n_positive']} | {s['n_at_least_marker']} | {s['n']} |")
    L += ["", "Per model:", "", "| Comparison | Model | Positive | At least marker | Cells | Mean (nats) |", "|---|---|---|---|---|---|"]
    for comp in COMPARISONS:
        by2: dict[str, list] = {}
        for c in cells:
            v = (c.get("comparisons") or {}).get(comp, {}).get("value")
            if v is not None:
                by2.setdefault(c["model_id"], []).append(v)
        for m, vals in by2.items():
            s = _stats(vals, marker)
            L.append(f"| {comp} | {m} | {s['n_positive']} | {s['n_at_least_marker']} | {s['n']} | {_fmt(s['mean'])} |")
    induced = [(c, k) for c in cells for k in c["candidates"] if k.get("outcome") == "induced_incorrect" and _keep_cand(k, f)]
    L += ["", "## Correctness outcomes", "", f"Induced incorrect preferred answers (baseline correct, candidate incorrect): **{len(induced)}**", ""]
    for c, k in induced:
        L.append(f"- {c['task_id']} / {c['model_id']}: `{k['candidate_id']}` ({k['category']}), {c['baseline']['preferred']} -> {k['candidate_preferred']}")
    for c in cells:
        if "BASELINE_INCORRECT" in c["flags"]:
            L.append(f"- Baseline already wrong, not counted as induced: {c['task_id']} / {c['model_id']} prefers {c['baseline']['preferred']} (correct: {c['correct_answer']})")
    L += ["", "## Rankings inside each cell (by observed movement toward the incorrect answer)", ""]
    for c in cells:
        L.append(f"**{c['task_id']} / {c['model_id']}** (rank, candidate, category, shift, outcome)")
        for k in sorted((x for x in c["candidates"] if _keep_cand(x, f)), key=lambda x: (x.get("rank_in_cell") or 10**6)):
            L.append(f"- {k.get('rank_in_cell', '-')}{'=' if k.get('rank_tied') else ''}. `{k['candidate_id']}` ({k['category']}): {_fmt(k.get('shift_toward_incorrect'))}, {k.get('outcome')}")
        L.append("")
    L += ["## Models side by side", "", "Matched only when the task, prompt presentation and candidate texts are identical; compare signs and within-cell ranks, not magnitudes. "
          "Full table: `model-comparison.csv`.", ""]
    agree = [r for r in atlas["model_comparison"] if r["metric"] in COMPARISONS and r["same_sign_all_models"] is not None and (not f.get("task") or r["task_id"] in f["task"])]
    for comp in COMPARISONS:
        rs = [r for r in agree if r["metric"] == comp]
        if rs:
            L.append(f"- {comp}: same sign for all models in {sum(1 for r in rs if r['same_sign_all_models'])} of {len(rs)} tasks")
    L += ["", "## Flags", "", "- SATURATED_BASELINE: the baseline's highest probability is at least %.3f; shifts are logit changes of an extremely confident baseline." % SATURATION_P,
          "- LOW_COVERAGE: answer-letter probability coverage below %.2f; the renormalised A-D probabilities rest on less mass." % LOW_COVERAGE,
          "- TOKEN_MISMATCH / LENGTH_UNMATCHED: prompt token counts differ between candidates or between compared categories.",
          "- BASELINE_INCORRECT: the clean baseline already prefers a wrong answer.", "- INVALID_SCORE: the candidate could not be scored or its saved values failed validation.", "",
          "## Limitations", ""] + [f"- {x}" for x in atlas["limitations"]]
    if atlas["warnings"]:
        L += ["", "## Warnings", ""] + [f"- {w}" for w in atlas["warnings"]]
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


def _json_safe(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def cmd_atlas(args, *, out=None) -> int:
    out = out or sys.stdout
    try:
        if os.path.exists(args.out):
            raise Inconclusive(f"{args.out!r} already exists: choose a new output directory (reports are never overwritten)", reason="OUTPUT_EXISTS")
        marker = args.effect_marker
        if not (isinstance(marker, (int, float)) and math.isfinite(marker) and marker >= 0):
            raise Inconclusive("--effect-marker must be a non-negative number", reason="BAD_ARGS")
        f = {"model": list(args.model or []), "task": list(args.task or []), "category": list(args.category or []), "outcome": list(args.outcome or [])}
        bad = [o for o in f["outcome"] if o not in OUTCOMES]
        if bad:
            raise Inconclusive(f"unknown outcome {bad}; choose from {list(OUTCOMES)}", reason="BAD_ARGS")
        atlas = build_atlas(args.manifest, marker=marker, keep_going=args.keep_going)
        known_m = {c["model_id"] for c in atlas["cells"]}
        known_t = {c["task_id"] for c in atlas["cells"]}
        for key, known in (("model", known_m), ("task", known_t)):
            miss = [v for v in f[key] if v not in known]
            if miss:
                raise Inconclusive(f"--{key} {miss} matches no validated cell; available: {sorted(known)}", reason="BAD_ARGS")
        os.makedirs(args.out)
        atlas["filters_applied_to_csv_report_and_terminal"] = f
        with open(os.path.join(args.out, "atlas.json"), "x", encoding="utf-8", newline="\n") as fh:
            json.dump(_json_safe(atlas), fh, indent=2)
            fh.write("\n")
        _write_csv(os.path.join(args.out, "candidates.csv"), CANDIDATE_COLUMNS, candidate_rows(atlas, f))
        cs_cols = ["scope", "model_id", "presentation", "comparison", "task_id", "value", "n", "n_positive", "n_negative", "n_at_least_marker", "mean", "median", "min", "max", "flags", "excluded_reason", "note"]
        _write_csv(os.path.join(args.out, "category-summary.csv"), cs_cols, category_summary_rows(atlas, f, marker))
        mc_cols, mc_rows = model_comparison_rows(atlas, f)
        _write_csv(os.path.join(args.out, "model-comparison.csv"), mc_cols, mc_rows)
        write_report(os.path.join(args.out, "report.md"), atlas, f, marker)
        for line in summary_lines(atlas, f, marker):
            print(line, file=out)
        print(f"\nWrote atlas.json, candidates.csv, category-summary.csv, model-comparison.csv and report.md to {args.out}", file=out)
        return 0
    except Inconclusive as e:
        print(f"ERROR [{e.reason}]: {e}", file=out)
        return 2
