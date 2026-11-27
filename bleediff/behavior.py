"""Offline behavioral reporting (v0.7.0): ingest saved evidence, summarise it by condition, and form explicitly PAIRED contrasts. No key, no network, no model.

INGESTION. ``load_evidence`` reads a saved session directory (``session.json`` + ``records/`` + optional ``prompts.json``), the legacy pilot layout (a directory of
``<cell>__<condition>__pass<k>.json`` plus a prompt dataset) or an observations file written by ``bleediff behavior ingest``. It only READS: the evidence is never modified, every
observation records the SHA-256 of the file it came from, and files that cannot be turned into an observation are listed as ingest problems, never silently dropped.

SEPARATION. The report has four independent blocks: ANSWER BEHAVIOR, PROVIDER-REPORTED THINKING, RESOURCE EXPENDITURE and FORMAT AND REFUSAL. Nothing combines them and there is no
composite behavioral score. Every proportion states its denominator and how many records were excluded as MISSING / UNSUPPORTED / INCONSISTENT / INVALID.

UNITS. Repeated requests of the same prompt (passes, sessions) are repeated measurements, not independent units. Summaries therefore report the number of requests, of distinct prompts
and of ``--unit-field`` units, and give the pooled-request figure next to the unit-averaged figure (the mean over units of each unit's own value). No interval, p-value or other
inference is computed here: these are descriptive summaries.

PAIRING. A contrast between two conditions pairs records that share (session, cell, pass): the same prompt template and mapping in the same session, differing only in the condition.
Conditions that share no such key are reported as unpaired, never contrasted. A pair counts for a measurement only if BOTH records are OBSERVED for it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, OrderedDict

from . import __version__
from . import answers as A
from . import telemetry as T
from .errors import Inconclusive

REPORT_SCHEMA = "bleediff.behavior-report/1"
OBSERVATIONS_FILE_SCHEMA = "bleediff.behavior-observations-file/1"
NOTE_DESCRIPTIVE = ("DESCRIPTIVE ONLY: no interval, test or composite score is computed. Repeated requests of one prompt are repeated measurements, not independent units; "
                    "cost is an estimate from reported usage and list prices, not a provider bill.")


class BehaviorError(Inconclusive):
    reason = "BEHAVIOR_INPUT_INVALID"


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------------------
# ingestion
# --------------------------------------------------------------------------------------
def _problem(file: str, code: str, detail: str = "") -> dict:
    return {"file": file, "code": code, "detail": detail}


def _read_json_bytes(raw: bytes):
    return json.loads(raw.decode("utf-8"))


def load_evidence(path: str, *, prompts_path: str | None = None, prices: tuple | None = None, cost_from_usage: bool = False) -> dict:
    """Ingest one evidence source. Returns {dir, kind, session, meta, observations, problems, files, digest}. Raises BehaviorError only when the source itself is unusable
    (not found, no record files); individual bad files become ``problems`` entries."""
    path = os.path.normpath(path)
    if os.path.isfile(path):
        return read_observations(path)
    if not os.path.isdir(path):
        raise BehaviorError(f"{path!r} is neither a session directory nor an observations file")
    base = os.path.basename(path)
    sj = os.path.join(path, "session.json")
    problems, meta, prompts = [], None, None
    if os.path.isfile(sj):
        kind = "session"
        try:
            with open(sj, "rb") as fh:
                meta = _read_json_bytes(fh.read())
        except (OSError, ValueError) as e:
            problems.append(_problem("session.json", "invalid_session_json", str(e)[:80]))
        if meta is not None and (not isinstance(meta, dict) or meta.get("schema") != A.SESSION_SCHEMA):
            problems.append(_problem("session.json", "unexpected_session_schema", str((meta or {}).get("schema") if isinstance(meta, dict) else type(meta).__name__)[:60]))
            meta = meta if isinstance(meta, dict) else None
        rdir = os.path.join(path, "records")
        files = sorted(f for f in os.listdir(rdir) if f.endswith(".json")) if os.path.isdir(rdir) else []
        rel_dir = base + "/records"
    else:
        kind = "legacy"
        rdir = path
        files = sorted(f for f in os.listdir(path) if f.endswith(".json") and "__pass" in f)
        rel_dir = base
    if not files:
        raise BehaviorError(f"{base!r} contains no record files")
    if prompts_path is None and kind == "session":
        cand = os.path.join(path, ((meta or {}).get("prompts") or {}).get("file", "prompts.json")) if isinstance(meta, dict) else os.path.join(path, "prompts.json")
        prompts_path = cand if os.path.isfile(cand) else None
    if prompts_path is not None:
        try:
            prompts = A.load_prompts(prompts_path)
        except A.AnswersError as e:
            problems.append(_problem(os.path.basename(prompts_path), "prompt_dataset_unreadable", str(e)[:80]))
        except OSError as e:
            problems.append(_problem(os.path.basename(prompts_path), "prompt_dataset_unreadable", str(e)[:80]))
    if kind == "legacy" and prompts is None:
        raise BehaviorError("the legacy pilot layout carries no scoring key: pass --prompts (the prompt dataset) so answers can be scored")
    session = (meta or {}).get("run_id") if isinstance(meta, dict) and isinstance((meta or {}).get("run_id"), str) else base
    digest = hashlib.sha256()
    observations, seen = [], set()
    for f in files:
        fp = os.path.join(rdir, f)
        with open(fp, "rb") as fh:
            raw = fh.read()
        h = _sha256(raw)
        digest.update(f.encode() + b"\0" + h.encode() + b"\n")
        try:
            rec = _read_json_bytes(raw)
        except ValueError:
            problems.append(_problem(f, "invalid_json"))
            continue
        if not isinstance(rec, dict):
            problems.append(_problem(f, "record_not_an_object"))
            continue
        if kind == "legacy":
            rec = _legacy_record(rec, prompts, f, problems)
            if rec is None:
                continue
        elif prompts is not None:
            _fill_from_prompts(rec, prompts)
        try:
            obs = T.observe_record(rec, layout=kind, evidence={"file": f"{rel_dir}/{f}", "sha256": h}, prices=prices, session=session, cost_from_usage=cost_from_usage)
        except T.TelemetryError as e:
            problems.append(_problem(f, e.reason.lower(), str(e)[:80]))
            continue
        key = (obs["identity"]["cell"], obs["identity"]["condition"], obs["identity"]["pass"])
        if key in seen:
            problems.append(_problem(f, "duplicate_record", f"{key}"))
            continue
        seen.add(key)
        observations.append(obs)
    if isinstance(meta, dict):
        with open(sj, "rb") as fh:
            digest.update(b"session.json\0" + _sha256(fh.read()).encode())
    meta_summary = {k: meta.get(k) for k in ("run_id", "provider", "model", "status", "planned_requests", "started_utc", "finished_utc", "bleediff_version")} if isinstance(meta, dict) else None
    return {"dir": base, "kind": kind, "session": session, "meta": meta_summary, "observations": observations, "problems": problems, "files": len(files), "digest": digest.hexdigest()}


def _fill_from_prompts(rec: dict, prompts: dict) -> None:
    item = prompts["by_key"].get((rec.get("cell"), rec.get("condition")))
    if not item:
        return
    for k_rec, k_item in (("correct_letter", "correct_letter"), ("competing_letter", "competing_letter"), ("mapping", "mapping"), ("question", "question")):
        if rec.get(k_rec) is None and item.get(k_item) is not None:
            rec[k_rec] = item[k_item]


def _legacy_record(rec: dict, prompts: dict, fname: str, problems: list) -> dict | None:
    parts = A._split_name(rec.get("name"))
    if not parts:
        problems.append(_problem(fname, "legacy_name_unreadable", str(rec.get("name"))[:60]))
        return None
    cell, cond, pas = parts
    out = {"name": rec["name"], "cell": cell, "condition": cond, "pass": pas, "prompt_sha256": rec.get("request_prompt_sha256"), "provider": rec.get("provider"), "model": rec.get("model"), "http_status": rec.get("http_status"),
           "response": rec.get("response"), "estimated_cost_usd": rec.get("estimated_cost_usd"), "cost_basis": rec.get("cost_basis"),
           "model_version": rec.get("model_version")}
    if not out["model"] and isinstance(rec.get("request_body"), dict):
        out["model"] = rec["request_body"].get("model")
    if out["provider"] is None:
        out["provider"] = A.detect_provider(out["response"])
    _fill_from_prompts(out, prompts)
    return out


# --------------------------------------------------------------------------------------
# observation files (the stable interchange format)
# --------------------------------------------------------------------------------------
def write_observations(path: str, sources: list[dict]) -> int:
    """Write observations as JSON Lines to a NEW file (never overwrites). The first line is a header naming each source and its digest. Returns the number of observations."""
    if os.path.exists(path):
        raise BehaviorError(f"{path!r} already exists (observation files are never overwritten)")
    n = 0
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        header = {"record_type": "header", "schema": OBSERVATIONS_FILE_SCHEMA, "bleediff_version": __version__,
                  "sources": [{"dir": s["dir"], "kind": s["kind"], "session": s["session"], "files": s["files"], "digest": s["digest"], "problems": len(s["problems"])} for s in sources]}
        fh.write(json.dumps(header, sort_keys=True) + "\n")
        for s in sources:
            for o in s["observations"]:
                fh.write(json.dumps(o, sort_keys=True) + "\n")
                n += 1
    return n


def read_observations(path: str) -> dict:
    problems, observations, header = [], [], None
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as e:
        raise BehaviorError(f"cannot read {os.path.basename(path)!r}: {e}") from None
    for i, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError:
            problems.append(_problem(f"line {i}", "invalid_json"))
            continue
        if isinstance(d, dict) and d.get("record_type") == "header":
            header = d
            continue
        bad = T.validate_observation(d)
        if bad:
            problems.append(_problem(f"line {i}", "invalid_observation", "; ".join(bad)[:120]))
            continue
        observations.append(d)
    if not observations:
        raise BehaviorError("the observations file holds no valid observation")
    return {"dir": os.path.basename(path), "kind": "observations", "session": None, "meta": header, "observations": observations, "problems": problems,
            "files": len(observations), "digest": _sha256(raw)}


# --------------------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------------------
def _stats(values: list) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "sum": None, "mean": None, "median": None, "min": None, "max": None}
    return {"n": n, "sum": math.fsum(values), "mean": math.fsum(values) / n, "median": statistics.median(values), "min": min(values), "max": max(values)}


def _states(obs: list[dict], name: str) -> dict:
    c = Counter(o["measures"][name]["state"] for o in obs)
    return {s: c.get(s, 0) for s in T.STATES}


def _vals(obs: list[dict], name: str) -> list:
    return [o["measures"][name]["value"] for o in obs if o["measures"][name]["state"] == T.OBSERVED]


def _unit_of(o: dict, unit_field: str):
    v = o["identity"].get(unit_field)
    return v if v is not None else "(none)"


def _indicator(o: dict, name: str):
    """A 0/1/None reading of one observation for unit-level averaging: booleans, and the 'correct' role."""
    if name == "correct":
        m = o["measures"]["answer_role"]
        return None if m["state"] != T.OBSERVED else (1 if m["value"] == "correct" else 0)
    m = o["measures"][name]
    if m["state"] != T.OBSERVED:
        return None
    v = m["value"]
    return int(v) if isinstance(v, bool) else v


UNIT_MEASURES = ("correct", "thinking_activation", "thinking_tokens", "output_tokens", "input_tokens", "latency_s", "cost_usd", "format_compliant", "refusal")


def _unit_means(obs: list[dict], unit_field: str) -> dict:
    out = {}
    units = sorted({_unit_of(o, unit_field) for o in obs}, key=str)
    for name in UNIT_MEASURES:
        per = {}
        for o in obs:
            v = _indicator(o, name)
            if v is not None:
                per.setdefault(_unit_of(o, unit_field), []).append(v)
        means = [math.fsum(v) / len(v) for v in per.values()]
        out[name] = {"units_with_observed": len(means), "units_without_observed": len(units) - len(means), "mean_of_unit_means": (math.fsum(means) / len(means)) if means else None}
    return out


def summarize_group(obs: list[dict], unit_field: str) -> dict:
    """One group of observations (a condition, within one scope). Four separate blocks; no combined score."""
    n = len(obs)
    answer_role_obs = [o["measures"]["answer_role"]["value"] for o in obs if o["measures"]["answer_role"]["state"] == T.OBSERVED]
    rc = Counter(answer_role_obs)
    scored = len(answer_role_obs)
    thinking_obs = _vals(obs, "thinking_activation")
    think_tokens = _vals(obs, "thinking_tokens")
    active_tokens = [v for v in think_tokens if v > 0]
    caps = Counter(o["context"]["output_cap_hit"] for o in obs)
    attempts = [o["context"]["attempts"] for o in obs if o["context"]["attempts"] is not None]
    cost_obs = [o["measures"]["cost_usd"] for o in obs if o["measures"]["cost_usd"]["state"] == T.OBSERVED]
    fmt, ref = _vals(obs, "format_compliant"), _vals(obs, "refusal")
    return {
        "units": {"requests": n, "distinct_prompts": len({o["identity"].get("prompt_sha256") or (o["identity"]["cell"], o["identity"]["condition"]) for o in obs}), "distinct_units": len({_unit_of(o, unit_field) for o in obs}),
                  "unit_field": unit_field},
        "states": {name: _states(obs, name) for name in T.MEASURES},
        "answer_behavior": {"status_counts": dict(Counter(o["answer_status"] for o in obs)),
                            "scored": scored, "unscored": n - scored,
                            "correct": rc.get("correct", 0), "competing": rc.get("competing", 0), "other_incorrect": rc.get("other_incorrect", 0), "no_answer": rc.get(T.NO_ANSWER, 0),
                            "strict_correct_rate": (rc.get("correct", 0) / scored) if scored else None,
                            "note": "rate = correct / scored records; no-answer records count as not correct; unscored = role MISSING/UNSUPPORTED/INCONSISTENT/INVALID"},
        "thinking": {"activation": {"observed": len(thinking_obs), "activated": sum(1 for v in thinking_obs if v), "rate": (sum(1 for v in thinking_obs if v) / len(thinking_obs)) if thinking_obs else None,
                                    "excluded": n - len(thinking_obs), "note": "rate = activated / OBSERVED; excluded records are not counted as zero"},
                     "thinking_tokens": _stats(think_tokens),
                     "thinking_tokens_among_activated": dict(_stats(active_tokens), note="a self-selected subset: descriptive only"),
                     "corroboration": dict(Counter(o["context"]["thinking_corroboration"] for o in obs))},
        "resources": {"input_tokens": _stats(_vals(obs, "input_tokens")), "output_tokens": _stats(_vals(obs, "output_tokens")),
                      "non_thinking_output_tokens": _stats(_vals(obs, "non_thinking_output_tokens")),
                      "cache_read_tokens": _stats(_vals(obs, "cache_read_tokens")), "cache_write_tokens": _stats(_vals(obs, "cache_write_tokens")),
                      "latency_s": dict(_stats(_vals(obs, "latency_s")), note="client-measured wall time of the whole request as recorded by the harness; includes retries when attempts > 1"),
                      "cost_usd": dict(_stats([m["value"] for m in cost_obs]), estimate=True, records_with_failed_attempt_charge=sum(1 for m in cost_obs if m.get("includes_worst_case_charge_for_failed_attempts")),
                                       note="an ESTIMATE from reported usage and list prices (or the harness's recorded estimate), not a provider bill"),
                      "output_cap": {"cap_reached": caps.get(True, 0), "not_reached": caps.get(False, 0), "unknown": caps.get(None, 0),
                                     "note": "token counts of capped records are right-censored at the cap"},
                      "records_with_retries": sum(1 for a in attempts if a > 1), "records_without_attempt_count": n - len(attempts)},
        "format_and_refusal": {"format_compliant": {"observed": len(fmt), "compliant": sum(1 for v in fmt if v), "excluded": n - len(fmt)},
                               "refusal": {"observed": len(ref), "refused": sum(1 for v in ref if v), "excluded": n - len(ref)}},
        "unit_averaged": _unit_means(obs, unit_field),
        "integrity": {"records_with_problems": sum(1 for o in obs if o["problems"]), "problem_codes": dict(Counter(p for o in obs for p in o["problems"]))},
    }


# --------------------------------------------------------------------------------------
# paired contrasts
# --------------------------------------------------------------------------------------
PAIRED_MEASURES = ("thinking_activation", "thinking_tokens", "output_tokens", "input_tokens", "latency_s", "cost_usd", "format_compliant", "refusal", "correct")


def _pair_key(o: dict):
    i = o["identity"]
    return (i["session"], i["cell"], i["pass"])


def contrast(obs: list[dict], cond_a: str, cond_b: str, unit_field: str) -> dict:
    """Paired contrast A - B over records sharing (session, cell, pass). Descriptive: pooled paired mean, unit-averaged paired mean, sign counts and (for booleans) the 2x2 discordance
    table. A pair is used for a measurement only if both records are OBSERVED for it; the rest are counted by the pair of states."""
    a = {_pair_key(o): o for o in obs if o["identity"]["condition"] == cond_a}
    b = {_pair_key(o): o for o in obs if o["identity"]["condition"] == cond_b}
    keys = sorted(set(a) & set(b), key=str)
    res = {"conditions": [cond_a, cond_b], "pairing": "records sharing (session, cell, pass)", "records": {cond_a: len(a), cond_b: len(b)},
           "pairs": len(keys), f"{cond_a}_only": len(set(a) - set(b)), f"{cond_b}_only": len(set(b) - set(a)), "unit_field": unit_field, "measures": {}, "note": NOTE_DESCRIPTIVE}
    if not keys:
        res["verdict"] = "NO PAIRS: these conditions share no (session, cell, pass); no contrast is formed"
        return res
    for name in PAIRED_MEASURES:
        diffs, per_unit, excluded, signs, disc = [], {}, Counter(), Counter(), Counter()
        for k in keys:
            ia, ib = _indicator(a[k], name), _indicator(b[k], name)
            if ia is None or ib is None:
                sa = "OBSERVED" if ia is not None else _state_of(a[k], name)
                sb = "OBSERVED" if ib is not None else _state_of(b[k], name)
                excluded[f"{cond_a}:{sa} / {cond_b}:{sb}"] += 1
                continue
            d = ia - ib
            diffs.append(d)
            per_unit.setdefault(_unit_of(a[k], unit_field), []).append(d)
            signs["a_higher" if d > 0 else ("b_higher" if d < 0 else "equal")] += 1
            if name in ("thinking_activation", "correct", "format_compliant", "refusal"):
                disc[f"{cond_a}={int(ia)},{cond_b}={int(ib)}"] += 1
        unit_means = [math.fsum(v) / len(v) for v in per_unit.values()]
        entry = {"pairs_used": len(diffs), "pairs_excluded": dict(excluded),
                 "pooled_mean_difference": (math.fsum(diffs) / len(diffs)) if diffs else None,
                 "units_used": len(unit_means), "unit_averaged_mean_difference": (math.fsum(unit_means) / len(unit_means)) if unit_means else None,
                 "sign_counts": {s: signs.get(s, 0) for s in ("a_higher", "b_higher", "equal")}}
        if disc or name in ("thinking_activation", "correct", "format_compliant", "refusal"):
            entry["discordance"] = {f"{cond_a}={x},{cond_b}={y}": disc.get(f"{cond_a}={x},{cond_b}={y}", 0) for x in (1, 0) for y in (1, 0)}
        res["measures"][name] = entry
    changed = Counter()
    for k in keys:
        oa, ob = a[k]["measures"]["answer_outcome"], b[k]["measures"]["answer_outcome"]
        if oa["state"] != T.OBSERVED or ob["state"] != T.OBSERVED:
            changed["outcome_not_comparable"] += 1
        elif oa["value"] == ob["value"]:
            changed["same_outcome"] += 1
        else:
            changed["different_outcome"] += 1
    res["answer_outcome_pairs"] = dict(changed)
    return res


def _state_of(o: dict, name: str) -> str:
    if name == "correct":
        return o["measures"]["answer_role"]["state"]
    return o["measures"][name]["state"]


# --------------------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------------------
def build_report(sources: list[dict], *, unit_field: str = "question", select_unit: str | None = None, contrasts: list[tuple[str, str]] | None = None,
                 baseline: str | None = None) -> dict:
    obs_all = [o for s in sources for o in s["observations"]]
    if not obs_all:
        raise BehaviorError("no observations to report")
    rx = re.compile(select_unit) if select_unit else None
    if rx is not None:
        obs_sel = [o for o in obs_all if rx.search(str(o["identity"].get(unit_field)))]
        if not obs_sel:
            raise BehaviorError(f"--select-unit {select_unit!r} matches no observation")
    else:
        obs_sel = obs_all
    sessions = sorted({str(o["identity"]["session"]) for o in obs_sel})
    conditions = list(OrderedDict.fromkeys(o["identity"]["condition"] for o in obs_sel))
    scopes = OrderedDict()
    scope_names = [("pooled", None)] + ([(f"session:{s}", s) for s in sessions] if len(sessions) > 1 else [])
    for label, sess in scope_names:
        pool = [o for o in obs_sel if sess is None or str(o["identity"]["session"]) == sess]
        present = {o["identity"]["condition"] for o in pool}
        scopes[label] = {"conditions": OrderedDict((c, summarize_group([o for o in pool if o["identity"]["condition"] == c], unit_field)) for c in conditions if c in present)}
    pairs = list(contrasts or [])
    if baseline:
        pairs += [(c, baseline) for c in conditions if c != baseline and (c, baseline) not in pairs]
    for c1, c2 in pairs:
        for c in (c1, c2):
            if c not in conditions:
                raise BehaviorError(f"condition {c!r} is not in the evidence (conditions: {conditions})")
    for label, sess in scope_names:
        pool = [o for o in obs_sel if sess is None or str(o["identity"]["session"]) == sess]
        scopes[label]["contrasts"] = [contrast(pool, c1, c2, unit_field) for c1, c2 in pairs]
    problems = [dict(p, source=s["dir"]) for s in sources for p in s["problems"]]
    return {"schema": REPORT_SCHEMA, "bleediff_version": __version__, "offline": True, "note": NOTE_DESCRIPTIVE,
            "inputs": [{"dir": s["dir"], "kind": s["kind"], "session": s["session"], "files": s["files"], "observations": len(s["observations"]), "digest": s["digest"],
                        "problems": len(s["problems"]), "meta": s["meta"]} for s in sources],
            "adapters": sorted({(o["adapter"]["id"], o["adapter"]["version"], o["adapter"]["validation"]) for o in obs_sel}),
            "selection": {"unit_field": unit_field, "select_unit": select_unit, "observations_selected": len(obs_sel), "observations_total": len(obs_all)},
            "observations_digest": T.observation_digest(obs_sel), "conditions": conditions, "sessions": sessions, "scopes": scopes, "ingest_problems": problems}


# --------------------------------------------------------------------------------------
# text rendering
# --------------------------------------------------------------------------------------
def _f(x, p: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3e}" if (x != 0 and abs(x) < 10 ** -p) else f"{x:.{p}f}"
    return str(x)


def _statline(s: dict, p: int = 3) -> str:
    return f"n={s['n']} mean={_f(s['mean'], p)} median={_f(s['median'], p)} min={_f(s['min'], p)} max={_f(s['max'], p)}"


def report_lines(rep: dict, *, scope: str | None = None) -> list[str]:
    """Text report. ``scope`` None prints every scope (pooled, then one per session when there are several)."""
    L = ["BEHAVIORAL TELEMETRY REPORT (OFFLINE: no key, no network, no model contacted)", "  " + rep["note"]]
    for i in rep["inputs"]:
        L.append(f"  input {i['dir']} [{i['kind']}]: {i['files']} file(s) -> {i['observations']} observation(s), {i['problems']} ingest problem(s), digest {i['digest'][:12]}")
    sel = rep["selection"]
    L.append(f"  observations {sel['observations_selected']} of {sel['observations_total']} (unit field {sel['unit_field']!r}" + (f", selected /{sel['select_unit']}/" if sel["select_unit"] else "") + ")")
    for a in rep["adapters"]:
        L.append(f"  adapter {a[0]} v{a[1]}: {a[2]}")
    for label in ([scope] if scope else list(rep["scopes"])):
        L.extend(_scope_lines(rep["scopes"][label], label))
    if rep["ingest_problems"]:
        L.append("")
        L.append(f"INGEST PROBLEMS ({len(rep['ingest_problems'])}): " + "; ".join(f"{p['source']}/{p['file']}: {p['code']}" for p in rep["ingest_problems"][:10]))
    return L


def _scope_lines(sc: dict, scope: str) -> list[str]:
    L: list[str] = []
    for cond, g in sc["conditions"].items():
        u = g["units"]
        L.append("")
        L.append(f"== {scope} | condition {cond}: {u['requests']} request(s), {u['distinct_prompts']} distinct prompt(s), {u['distinct_units']} {u['unit_field']} unit(s)")
        ab = g["answer_behavior"]
        L.append(f"  ANSWER BEHAVIOR  statuses {ab['status_counts']}; scored {ab['scored']} (unscored {ab['unscored']}): correct {ab['correct']}, competing {ab['competing']}, "
                 f"other {ab['other_incorrect']}, no answer {ab['no_answer']}; strict correct rate {_f(ab['strict_correct_rate'])}")
        th = g["thinking"]
        L.append(f"  THINKING (provider-reported)  activation {th['activation']['activated']}/{th['activation']['observed']} observed = {_f(th['activation']['rate'])} (excluded {th['activation']['excluded']}); "
                 f"tokens {_statline(th['thinking_tokens'], 1)}; among activated {_statline(th['thinking_tokens_among_activated'], 1)}; corroboration {th['corroboration']}")
        rs = g["resources"]
        L.append(f"  RESOURCES  input {_statline(rs['input_tokens'], 1)} | output {_statline(rs['output_tokens'], 1)} | latency_s {_statline(rs['latency_s'], 3)}")
        L.append(f"             cost_usd (estimate) n={rs['cost_usd']['n']} total={_f(rs['cost_usd']['sum'], 6)} mean={_f(rs['cost_usd']['mean'], 8)}; cap reached {rs['output_cap']['cap_reached']}; retries {rs['records_with_retries']}")
        fr = g["format_and_refusal"]
        L.append(f"  FORMAT/REFUSAL  compliant {fr['format_compliant']['compliant']}/{fr['format_compliant']['observed']} (excluded {fr['format_compliant']['excluded']}); "
                 f"refusals {fr['refusal']['refused']}/{fr['refusal']['observed']} (excluded {fr['refusal']['excluded']})")
        ua = g["unit_averaged"]
        L.append(f"  UNIT-AVERAGED  thinking_activation {_f(ua['thinking_activation']['mean_of_unit_means'])} over {ua['thinking_activation']['units_with_observed']} unit(s); "
                 f"correct {_f(ua['correct']['mean_of_unit_means'])}; thinking_tokens {_f(ua['thinking_tokens']['mean_of_unit_means'], 2)}; cost_usd {_f(ua['cost_usd']['mean_of_unit_means'], 8)}")
        bad = {name: {k: v for k, v in st.items() if k != T.OBSERVED and v} for name, st in g["states"].items()}
        bad = {k: v for k, v in bad.items() if v}
        L.append("  NOT-OBSERVED STATES  " + (json.dumps(bad, sort_keys=True) if bad else "none (every measurement of every record is OBSERVED)"))
        if g["integrity"]["problem_codes"]:
            L.append(f"  INTEGRITY PROBLEMS  {g['integrity']['problem_codes']}")
    for c in sc.get("contrasts", []):
        a, b = c["conditions"]
        L.append("")
        L.append(f"== {scope} | PAIRED CONTRAST {a} - {b}: {c['pairs']} pair(s) ({a}-only {c[a + '_only']}, {b}-only {c[b + '_only']}); pairing {c['pairing']}")
        if "verdict" in c:
            L.append("  " + c["verdict"])
            continue
        for name, m in c["measures"].items():
            ex = f" excluded {sum(m['pairs_excluded'].values())}" if m["pairs_excluded"] else ""
            line = f"  {name:20s} pairs {m['pairs_used']:5d}{ex}: pooled diff {_f(m['pooled_mean_difference'], 4)}, unit-averaged diff {_f(m['unit_averaged_mean_difference'], 4)} over {m['units_used']} unit(s); signs {m['sign_counts']}"
            if "discordance" in m:
                line += f"; discordance {m['discordance']}"
            L.append(line)
        L.append(f"  answer outcomes {c['answer_outcome_pairs']}")
    return L


# --------------------------------------------------------------------------------------
# CLI: bleediff behavior ingest | report
# --------------------------------------------------------------------------------------
def _prices(args) -> tuple | None:
    pi, po = getattr(args, "price_in", None), getattr(args, "price_out", None)
    if (pi is None) != (po is None):
        raise BehaviorError("give both --price-in and --price-out (USD per 1M tokens), or neither", reason="BAD_ARGS")
    if pi is None:
        return None
    for v in (pi, po):
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
            raise BehaviorError("prices must be finite and non-negative", reason="BAD_ARGS")
    return (pi, po)


def _load_all(args) -> list[dict]:
    prices = _prices(args)
    if args.cost_from_usage and prices is None:
        raise BehaviorError("--cost-from-usage needs --price-in and --price-out", reason="BAD_ARGS")
    return [load_evidence(pth, prompts_path=args.prompts, prices=prices, cost_from_usage=args.cost_from_usage) for pth in args.source]


def _write_json_new(path: str, obj) -> None:
    if os.path.exists(path):
        raise BehaviorError(f"{path!r} already exists (output files are never overwritten)", reason="OUTPUT_EXISTS")
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")


def cmd_ingest(args, *, out=None) -> int:
    out = out or sys.stdout
    sources = _load_all(args)
    n = write_observations(args.out, sources)
    print("OFFLINE INGEST (no key, no network, no model contacted; the evidence was only read)", file=out)
    for s in sources:
        print(f"  {s['dir']} [{s['kind']}]: {s['files']} file(s) -> {len(s['observations'])} observation(s), {len(s['problems'])} problem(s)", file=out)
        for p in s["problems"][:10]:
            print(f"    PROBLEM {p['file']}: {p['code']} {p['detail']}", file=out)
    print(f"wrote {n} observation(s) to {os.path.basename(args.out)} (schema {T.OBSERVATION_SCHEMA})", file=out)
    return 2 if (args.strict and any(s["problems"] for s in sources)) else 0


def cmd_report(args, *, out=None) -> int:
    out = out or sys.stdout
    if args.json_out and os.path.exists(args.json_out):
        raise BehaviorError(f"{os.path.basename(args.json_out)!r} already exists (output files are never overwritten)", reason="OUTPUT_EXISTS")
    sources = _load_all(args)
    contrasts = [tuple(c) for c in (args.contrast or [])]
    rep = build_report(sources, unit_field=args.unit_field, select_unit=args.select_unit, contrasts=contrasts, baseline=args.baseline)
    for line in report_lines(rep):
        print(line, file=out)
    if args.json_out:
        _write_json_new(args.json_out, rep)
        print(f"\nwrote {os.path.basename(args.json_out)} (schema {REPORT_SCHEMA})", file=out)
    return 2 if (args.strict and rep["ingest_problems"]) else 0


def add_parser(sub) -> None:
    b = sub.add_parser("behavior", help="OFFLINE behavioral telemetry: ingest saved hosted sessions and report answer behavior, provider-reported thinking, resource expenditure and "
                                        "format/refusal separately, with explicit measurement states (no key, no network, no model)")
    bs = b.add_subparsers(dest="behavior_cmd", required=True)
    for name, fn, help_text in (("ingest", cmd_ingest, "read saved evidence and write observations (JSON Lines) to a NEW file"),
                                ("report", cmd_report, "summarise saved evidence by condition and form explicitly paired contrasts")):
        p = bs.add_parser(name, help=help_text)
        p.add_argument("source", nargs="+", help="a saved session directory (session.json + records/), a legacy pilot directory (needs --prompts) or an observations file")
        p.add_argument("--prompts", default=None, help="prompt dataset giving the scoring key / mapping / question (needed for the legacy layout; optional otherwise)")
        p.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens, used to estimate cost for records that carry no cost of their own (or all records with --cost-from-usage)")
        p.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens (same use)")
        p.add_argument("--cost-from-usage", action="store_true", help="price EVERY record from its reported usage and the prices above, ignoring the harness's recorded cost "
                                                                       "(which can include a worst-case charge for failed attempts)")
        p.add_argument("--strict", action="store_true", help="exit 2 if any file could not be ingested")
        if name == "ingest":
            p.add_argument("--out", required=True, help="NEW observations file (never overwritten)")
        else:
            p.add_argument("--unit-field", default="question", help="identity field defining the unit for unit-averaged figures (default: question)")
            p.add_argument("--select-unit", default=None, metavar="REGEX", help="keep only observations whose unit value matches this regular expression")
            p.add_argument("--baseline", default=None, help="contrast every other condition with this one (paired)")
            p.add_argument("--contrast", nargs=2, action="append", metavar=("A", "B"), help="a paired contrast A - B (repeatable)")
            p.add_argument("--json-out", default=None, help="also write the full report to a NEW JSON file")
        p.set_defaults(fn=fn)
