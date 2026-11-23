"""Answer-only (``ANSWER_ONLY``) hosted measurements: frozen prompt datasets, response parsing, saved-session validation and offline analysis.

WHAT THIS MODULE IS FOR
Some hosted models return an answer but expose no usable log-probabilities for the four choices (the tested Gemini API configuration: every tested
model rejects ``responseLogprobs``; the tested OpenAI model returned only a top-20 list that held all four letters in 17.5% of responses; the Anthropic Messages API has no
logprobs parameter at all). Bleediff's direct-probability measurements (``DIRECT_PROBABILITY``: normalised A-D
probabilities, log-odds margins, coverage) cannot be produced for them, and nothing here pretends otherwise:

* no probabilities, no log-odds margins, no residuals, no "confidence" and no leaderboard are ever computed from answers;
* repeated identical answers are a replicate-stability observation, not a probability and not a confidence estimate;
* observed answer rates are never equated with internal token probabilities;
* every record, session and report is labelled with its measurement type.

Semantic tracking: each prompt of a frozen dataset names the letter of the correct answer and of the competing (incorrect) answer, so the
same semantic answers can be followed across answer-letter permutations. The other letters are "other incorrect" answers.

DATASET (JSON; ``bleediff.hosted-prompts/1``, or the v0.7 pilot's ``bleediff.frontier-pilot-frozen/1``)
{"schema": "bleediff.hosted-prompts/1", "baseline_condition": "BASELINE",
 "items": [{"cell": "q1-m1", "question": "q1", "mapping": "m1", "condition": "BASELINE", "prompt": "...", "prompt_sha256": "optional",
            "correct_letter": "A", "competing_letter": "B"}, ...]}
Every cell needs one baseline item; (cell, condition) pairs are unique.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter

from . import CHOICES
from .errors import Inconclusive

DIRECT_PROBABILITY = "DIRECT_PROBABILITY"
ANSWER_ONLY = "ANSWER_ONLY"
RECORD_SCHEMA = "bleediff.hosted-answer/1"
SESSION_SCHEMA = "bleediff.hosted-session/1"
PROMPT_SCHEMAS = ("bleediff.hosted-prompts/1", "bleediff.frontier-pilot-frozen/1")
DEFAULT_BASELINE = "BASELINE"
STATUSES = ("ok", "invalid", "blocked", "empty", "malformed", "error")
BLOCKING_FINISH = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY", "RECITATION", "LANGUAGE"})
FORBIDDEN_KEYS = frozenset({"probabilities", "logprobs", "logprobsResult", "logprobs_result", "margin", "margin_AB", "log_odds"})
NOTE = ("ANSWER_ONLY: the model returned an answer and no probabilities. Counts below are observed answers; they are not probabilities, "
        "log-odds or confidence, and no probability margin or leaderboard exists for these data.")


class AnswersError(Inconclusive):
    reason = "ANSWER_RECORDS_INVALID"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return sha256_bytes(fh.read())


# --------------------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------------------
def load_prompts(path: str) -> dict:
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        doc = json.loads(raw.decode("utf-8-sig"))
    except (OSError, ValueError) as e:
        raise AnswersError(f"cannot read the prompt dataset {os.path.basename(path)!r}: {e}") from None
    if not isinstance(doc, dict) or doc.get("schema") not in PROMPT_SCHEMAS:
        raise AnswersError(f"the prompt dataset must be an object with schema one of {list(PROMPT_SCHEMAS)}")
    items = doc.get("items")
    if not isinstance(items, list) or not items:
        raise AnswersError("the prompt dataset needs a non-empty 'items' list")
    baseline = doc.get("baseline_condition", DEFAULT_BASELINE)
    by_key, cells, conds = {}, [], []
    for i, it in enumerate(items, 1):
        where = f"prompt item #{i}"
        if not isinstance(it, dict):
            raise AnswersError(f"{where} is not an object")
        for k in ("cell", "condition", "prompt", "correct_letter", "competing_letter"):
            if not isinstance(it.get(k), str) or not it[k]:
                raise AnswersError(f"{where}: {k!r} must be a non-empty string")
        if it["correct_letter"] not in CHOICES or it["competing_letter"] not in CHOICES or it["correct_letter"] == it["competing_letter"]:
            raise AnswersError(f"{where}: correct_letter and competing_letter must be two different letters from {list(CHOICES)}")
        digest = sha256_bytes(it["prompt"].encode("utf-8"))
        if it.get("prompt_sha256") not in (None, digest):
            raise AnswersError(f"{where}: prompt_sha256 does not match the prompt text")
        key = (it["cell"], it["condition"])
        if key in by_key:
            raise AnswersError(f"duplicate prompt for cell {key[0]!r} and condition {key[1]!r}")
        by_key[key] = {**it, "prompt_sha256": digest}
        if it["cell"] not in cells:
            cells.append(it["cell"])
        if it["condition"] not in conds:
            conds.append(it["condition"])
    for c in cells:
        if (c, baseline) not in by_key:
            raise AnswersError(f"cell {c!r} has no {baseline!r} (clean baseline) item")
    for c in cells:                           # the same cell must name the same roles under every condition
        roles = {(by_key[(c, k)]["correct_letter"], by_key[(c, k)]["competing_letter"]) for k in conds if (c, k) in by_key}
        if len(roles) != 1:
            raise AnswersError(f"cell {c!r} names different correct/competing letters under different conditions")
    return {"name": os.path.basename(path), "sha256": sha256_bytes(raw), "schema": doc["schema"], "baseline_condition": baseline,
            "items": list(by_key.values()), "by_key": by_key, "cells": cells, "conditions": conds}


# --------------------------------------------------------------------------------------
# parsing one Gemini-style generateContent response (no network)
# --------------------------------------------------------------------------------------
def _int(x):
    return x if isinstance(x, int) and not isinstance(x, bool) and x >= 0 else None


def parse_answer(response) -> dict:
    """Interpret a ``generateContent`` response. status: ok (exactly one letter A-D) | invalid (text that is not exactly one letter) |
    blocked (prompt or output blocked) | empty (no text) | malformed (unexpected structure). Never guesses a missing answer."""
    out = {"status": "malformed", "answer": None, "text": None, "finish_reason": None, "block_reason": None}
    if not isinstance(response, dict):
        return out
    fb = response.get("promptFeedback")
    if isinstance(fb, dict) and fb.get("blockReason"):
        out["block_reason"] = str(fb["blockReason"])[:80]
    cands = response.get("candidates")
    if cands is None or cands == []:
        out["status"] = "blocked" if out["block_reason"] else "empty"
        return out
    if not isinstance(cands, list) or not isinstance(cands[0], dict):
        return out
    cand = cands[0]
    fr = cand.get("finishReason")
    out["finish_reason"] = str(fr)[:40] if fr is not None else None
    content = cand.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if parts is not None and not isinstance(parts, list):
        return out
    texts = [p["text"] for p in (parts or []) if isinstance(p, dict) and isinstance(p.get("text"), str) and not p.get("thought")]
    text = "".join(texts)
    out["text"] = text[:200]
    if out["block_reason"] or out["finish_reason"] in BLOCKING_FINISH:
        out["status"] = "blocked"
        return out
    s = text.strip()
    if not s:
        out["status"] = "empty"
    elif s in CHOICES:
        out["status"], out["answer"] = "ok", s
    else:
        out["status"] = "invalid"
    return out


def usage_from(response) -> dict:
    u = response.get("usageMetadata") if isinstance(response, dict) else None
    u = u if isinstance(u, dict) else {}
    return {k: _int(u.get(k)) for k in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount")}


def parse_openai_answer(response) -> dict:
    """Interpret an OpenAI Chat Completions response (same statuses as ``parse_answer``). A refusal or a ``content_filter`` finish is 'blocked'.
    Any ``logprobs`` in the response are ignored here: this analysis reads the returned message only."""
    out = {"status": "malformed", "answer": None, "text": None, "finish_reason": None, "block_reason": None}
    if not isinstance(response, dict):
        return out
    choices = response.get("choices")
    if choices is None:                       # an error body or anything else without a 'choices' member
        return out
    if choices == []:
        out["status"] = "empty"
        return out
    if not isinstance(choices, list) or not isinstance(choices[0], dict):
        return out
    ch = choices[0]
    fr = ch.get("finish_reason")
    out["finish_reason"] = str(fr)[:40] if fr is not None else None
    msg = ch.get("message")
    if not isinstance(msg, dict):
        return out
    content, refusal = msg.get("content"), msg.get("refusal")
    if content is not None and not isinstance(content, str):
        return out
    out["text"] = (content or "")[:200]
    if refusal:
        out["block_reason"] = "refusal"
    elif out["finish_reason"] == "content_filter":
        out["block_reason"] = "content_filter"
    if out["block_reason"]:
        out["status"] = "blocked"
        return out
    s = (content or "").strip()
    if not s:
        out["status"] = "empty"
    elif s in CHOICES:
        out["status"], out["answer"] = "ok", s
    else:
        out["status"] = "invalid"
    return out


def usage_from_openai(response) -> dict:
    """OpenAI usage mapped onto the shared four keys. ``completion_tokens`` already INCLUDES reasoning tokens, so the visible count is the difference;
    candidate + thinking therefore always equals the billed completion tokens (nothing is counted twice)."""
    u = response.get("usage") if isinstance(response, dict) else None
    u = u if isinstance(u, dict) else {}
    det = u.get("completion_tokens_details")
    reasoning = _int(det.get("reasoning_tokens")) if isinstance(det, dict) else None
    comp = _int(u.get("completion_tokens"))
    visible = comp if (comp is None or reasoning is None) else max(0, comp - reasoning)
    return {"promptTokenCount": _int(u.get("prompt_tokens")), "candidatesTokenCount": visible, "thoughtsTokenCount": reasoning, "totalTokenCount": _int(u.get("total_tokens"))}


def parse_anthropic_answer(response) -> dict:
    """Interpret an Anthropic Messages response (same statuses as ``parse_answer``). Only ``text`` blocks count; ``thinking`` / ``redacted_thinking`` blocks are ignored, so a reply
    that spent its whole ``max_tokens`` on thinking is honestly 'empty'. ``stop_reason: refusal`` is 'blocked'."""
    out = {"status": "malformed", "answer": None, "text": None, "finish_reason": None, "block_reason": None}
    if not isinstance(response, dict):
        return out
    content = response.get("content")
    if not isinstance(content, list):                 # an error body, or anything without a content list
        return out
    sr = response.get("stop_reason")
    out["finish_reason"] = str(sr)[:40] if sr is not None else None
    if any(not isinstance(b, dict) for b in content):
        return out
    texts = [b["text"] for b in content if b.get("type") == "text" and isinstance(b.get("text"), str)]
    text = "".join(texts)
    out["text"] = text[:200]
    if out["finish_reason"] == "refusal":
        out["block_reason"] = "refusal"
        out["status"] = "blocked"
        return out
    s = text.strip()
    if not s:
        out["status"] = "empty"
    elif s in CHOICES:
        out["status"], out["answer"] = "ok", s
    else:
        out["status"] = "invalid"
    return out


def usage_from_anthropic(response) -> dict:
    """Anthropic usage mapped onto the shared four keys. ``output_tokens`` already INCLUDES thinking tokens (``output_tokens_details.thinking_tokens`` is a subset), so
    candidate + thinking equals the billed output tokens. Input-side tokens are input + cache reads + cache writes (cache writes counted twice: a conservative bound
    for the 1.25x/2x write prices; the adapter never enables caching)."""
    u = response.get("usage") if isinstance(response, dict) else None
    u = u if isinstance(u, dict) else {}
    inp = _int(u.get("input_tokens"))
    read, write = _int(u.get("cache_read_input_tokens")) or 0, _int(u.get("cache_creation_input_tokens")) or 0
    prompt = None if inp is None else inp + read + 2 * write
    out = _int(u.get("output_tokens"))
    det = u.get("output_tokens_details")
    thinking = _int(det.get("thinking_tokens")) if isinstance(det, dict) else None
    visible = out if (out is None or thinking is None) else max(0, out - thinking)
    total = prompt + out if (prompt is not None and out is not None) else None
    return {"promptTokenCount": prompt, "candidatesTokenCount": visible, "thoughtsTokenCount": thinking, "totalTokenCount": total}


PARSERS = {"gemini": parse_answer, "openai": parse_openai_answer, "anthropic": parse_anthropic_answer}
USAGES = {"gemini": usage_from, "openai": usage_from_openai, "anthropic": usage_from_anthropic}


def detect_provider(response) -> str:
    if isinstance(response, dict):
        if "choices" in response:
            return "openai"
        if response.get("type") == "message" or "stop_reason" in response:
            return "anthropic"
    return "gemini"


def has_logprobs(response) -> bool:
    """True when a saved response carries per-token log-probabilities (either provider's shape)."""
    if not isinstance(response, dict):
        return False
    c = response.get("candidates")
    if isinstance(c, list) and any(isinstance(x, dict) and x.get("logprobsResult") for x in c):
        return True
    c = response.get("choices")
    return isinstance(c, list) and any(isinstance(x, dict) and x.get("logprobs") for x in c)


def role_of(answer, item) -> str | None:
    if answer is None:
        return None
    return "correct" if answer == item["correct_letter"] else ("competing" if answer == item["competing_letter"] else "other_incorrect")


# --------------------------------------------------------------------------------------
# SECONDARY DIAGNOSTIC: final-letter recovery (FLR-v1). Never the primary metric.
# --------------------------------------------------------------------------------------
FLR_RULE = "FLR-v1"
FLR_NOTE = ("DIAGNOSTIC (post-hoc, FLR-v1): the last non-empty line of a format-invalid reply, with markdown emphasis stripped, is exactly one letter A-D. It credits replies "
            "the prompt forbade, never changes a strict status, never applies to refusals, and is not a probability or a confidence.")
_FLR_STRIP = re.compile(r"[\s*_`~]+")


def visible_text(response, provider: str | None = None) -> str:
    """The full visible reply text of a saved response (thinking blocks excluded), for the diagnostic only. '' when there is none."""
    if not isinstance(response, dict):
        return ""
    provider = provider or detect_provider(response)
    try:
        if provider == "anthropic":
            c = response.get("content")
            return "".join(b["text"] for b in c if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)) if isinstance(c, list) else ""
        if provider == "openai":
            m = response["choices"][0]["message"]
            return m["content"] if isinstance(m.get("content"), str) else ""
        parts = response["candidates"][0]["content"]["parts"]
        return "".join(p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str) and not p.get("thought"))
    except (KeyError, IndexError, TypeError):
        return ""


def extract_final_letter(text) -> dict:
    """Deterministic, text-only final-letter extraction (FLR-v1). Returns {"status": "extracted" | "conflict" | "none", "letter": A-D or None, "rule": FLR_RULE}.
    extracted: the last non-empty line is, once whitespace and markdown emphasis (``*``, ``_``, backticks, ``~``) are removed, exactly one letter A-D, and no other line is a
    standalone letter that differs from it. conflict: standalone letters on different lines disagree (nothing is picked). none: anything else (no letter, a letter inside prose,
    'A or B', empty). It knows nothing about statuses: callers must not apply it to refusals (see ``diagnostic_class``)."""
    lines = [ln for ln in (text if isinstance(text, str) else "").splitlines() if ln.strip()]
    if not lines:
        return {"status": "none", "letter": None, "rule": FLR_RULE}
    stripped = [_FLR_STRIP.sub("", ln) for ln in lines]
    standalone = [x if x in CHOICES else None for x in stripped]
    if standalone[-1] is None:
        return {"status": "conflict" if len({x for x in standalone if x}) > 1 else "none", "letter": None, "rule": FLR_RULE}
    if len({x for x in standalone if x}) > 1:
        return {"status": "conflict", "letter": None, "rule": FLR_RULE}
    return {"status": "extracted", "letter": standalone[-1], "rule": FLR_RULE}


def diagnostic_class(rec: dict) -> dict:
    """Five-way reporting split of one normalised record, kept beside (never instead of) its strict status.
    strict_valid (status ok; the strict answer is untouched) | refusal (status blocked: an explicit provider refusal or block, never reinterpreted) |
    format_recovered (status invalid and FLR-v1 extracted one letter; ``flr_letter`` holds it) | ambiguous (invalid with conflicting standalone letters) |
    missing (empty, malformed, error, or invalid with no standalone final letter)."""
    st = rec.get("status")
    if st == "ok":
        return {"class": "strict_valid", "flr_letter": None}
    if st == "blocked":
        return {"class": "refusal", "flr_letter": None}
    if st == "invalid":
        f = extract_final_letter(visible_text(rec.get("response"), rec.get("provider")))
        if f["status"] == "extracted":
            return {"class": "format_recovered", "flr_letter": f["letter"]}
        return {"class": "ambiguous" if f["status"] == "conflict" else "missing", "flr_letter": None}
    return {"class": "missing", "flr_letter": None}


# --------------------------------------------------------------------------------------
# reading saved sessions (new format or the v0.8 pilot layout)
# --------------------------------------------------------------------------------------
def _split_name(name: str):
    try:
        cell, cond, p = name.split("__")
        return cell, cond, int(p.replace("pass", ""))
    except (ValueError, AttributeError):
        return None


def _normalise(rec: dict, prompts: dict | None, legacy: bool, problems: list, fname: str, warnings: list | None = None) -> dict | None:
    if not isinstance(rec, dict):
        problems.append(f"{fname}: not a JSON object")
        return None
    if FORBIDDEN_KEYS & set(rec):
        problems.append(f"{fname}: an ANSWER_ONLY record must not carry probability fields ({sorted(FORBIDDEN_KEYS & set(rec))})")
    resp = rec.get("response")
    provider = rec.get("provider") or detect_provider(resp)
    if provider not in PARSERS:
        problems.append(f"{fname}: unknown provider {provider!r}")
        return None
    parse, usage_of = PARSERS[provider], USAGES[provider]
    if has_logprobs(resp):
        if legacy:      # original pilot files may hold partial top-20 logprobs; this analysis never reads them
            if warnings is not None and not any("logprobs" in w for w in warnings):
                warnings.append("the original pilot records carry partial per-token logprobs; they are ignored (answers come from the returned message only)")
        else:
            problems.append(f"{fname}: the response contains logprobs, so this is not an ANSWER_ONLY record")
    if legacy:
        parts = _split_name(rec.get("name"))
        if not parts:
            problems.append(f"{fname}: cannot read cell/condition/pass from the record name {rec.get('name')!r}")
            return None
        cell, cond, pas = parts
        parsed = parse(resp) if isinstance(resp, dict) else {"status": "error", "answer": None, "text": None, "finish_reason": None, "block_reason": None}
        usage = usage_of(resp)
        model = rec.get("model") or ((rec.get("request_body") or {}).get("model") if isinstance(rec.get("request_body"), dict) else None)
        version = (resp.get("modelVersion") if provider == "gemini" else resp.get("model")) if isinstance(resp, dict) else None
        r = {"name": rec["name"], "provider": provider, "cell": cell, "condition": cond, "pass": pas, "prompt_sha256": rec.get("request_prompt_sha256"), "model": model,
             "model_version": version, "http_status": rec.get("http_status"), "usage": usage,
             "usage_complete": usage["promptTokenCount"] is not None and (usage["candidatesTokenCount"] is not None or usage["thoughtsTokenCount"] is not None),
             "cost_usd": rec.get("estimated_cost_usd"), "cost_basis": "recorded by the original pilot (billed output tokens at list prices)", "attempts": 1, "response": resp, **parsed}
        return r
    if rec.get("schema") != RECORD_SCHEMA:
        problems.append(f"{fname}: unexpected record schema {rec.get('schema')!r}")
        return None
    if rec.get("measurement_type") != ANSWER_ONLY:
        problems.append(f"{fname}: measurement_type is {rec.get('measurement_type')!r}, expected {ANSWER_ONLY!r}")
    for k in ("name", "cell", "condition", "pass", "status", "prompt_sha256"):
        if k not in rec:
            problems.append(f"{fname}: missing {k!r}")
            return None
    if rec["status"] not in STATUSES:
        problems.append(f"{fname}: unknown status {rec['status']!r}")
        return None
    r = dict(rec)
    r["provider"] = provider
    if isinstance(resp, dict):
        p = parse(resp)
        for k in ("status", "answer"):
            if rec.get(k) != p[k]:
                problems.append(f"{fname}: stored {k} {rec.get(k)!r} disagrees with the saved response ({p[k]!r})")
        r.update({"status": p["status"], "answer": p["answer"], "text": p["text"], "finish_reason": p["finish_reason"], "block_reason": p["block_reason"]})
    r.setdefault("usage", usage_of(resp) if isinstance(resp, dict) else {})
    return r


def read_session(path: str, prompts_path: str | None = None) -> dict:
    """Read a saved answer-only session directory: a new-format session (``session.json`` + ``records/``) or the v0.8 pilot layout
    (a directory of ``<cell>__<condition>__pass<k>.json`` files, which needs ``prompts_path``). Never contacts a model."""
    path = os.path.normpath(path)
    problems, warnings = [], []
    if not os.path.isdir(path):
        raise AnswersError(f"{os.path.basename(path)!r} is not a directory")
    sj = os.path.join(path, "session.json")
    meta = None
    if os.path.isfile(sj):
        kind = "session"
        try:
            with open(sj, "rb") as fh:
                meta = json.loads(fh.read().decode("utf-8"))
        except (OSError, ValueError) as e:
            raise AnswersError(f"cannot read session.json: {e}") from None
        if not isinstance(meta, dict) or meta.get("schema") != SESSION_SCHEMA:
            raise AnswersError(f"session.json must have schema {SESSION_SCHEMA!r}")
        if meta.get("measurement_type") != ANSWER_ONLY:
            raise AnswersError(f"session.json measurement_type must be {ANSWER_ONLY!r}")
        rdir = os.path.join(path, "records")
        files = sorted(f for f in os.listdir(rdir) if f.endswith(".json")) if os.path.isdir(rdir) else []
    else:
        kind = "legacy"
        rdir = path
        files = sorted(f for f in os.listdir(path) if f.endswith(".json") and "__pass" in f)
    if not files:
        raise AnswersError("the session has no record files")
    if prompts_path is None:
        cand = os.path.join(path, (meta or {}).get("prompts", {}).get("file", "prompts.json")) if meta else None
        prompts_path = cand if cand and os.path.isfile(cand) else None
    if prompts_path is None:
        raise AnswersError("a prompt dataset is required (pass --prompts, or use a session that contains prompts.json)")
    prompts = load_prompts(prompts_path)
    if meta and (meta.get("prompts") or {}).get("sha256") not in (None, prompts["sha256"]):
        problems.append("the prompt dataset does not match the sha256 recorded in session.json")
    records, seen, digest = [], set(), hashlib.sha256()
    for f in files:
        fp = os.path.join(rdir, f)
        with open(fp, "rb") as fh:
            raw = fh.read()
        digest.update(f.encode() + b"\0" + sha256_bytes(raw).encode() + b"\n")
        try:
            rec = json.loads(raw.decode("utf-8"))
        except ValueError:
            problems.append(f"{f}: not valid JSON")
            continue
        r = _normalise(rec, prompts, kind == "legacy", problems, f, warnings)
        if r is None:
            continue
        item = prompts["by_key"].get((r["cell"], r["condition"]))
        if item is None:
            problems.append(f"{f}: cell/condition {r['cell']!r}/{r['condition']!r} is not in the prompt dataset")
            continue
        if r.get("prompt_sha256") != item["prompt_sha256"]:
            problems.append(f"{f}: the recorded prompt hash does not match the dataset prompt")
        key = (r["cell"], r["condition"], r["pass"])
        if key in seen:
            problems.append(f"{f}: duplicate record for {key}")
            continue
        seen.add(key)
        records.append(r)
    if meta:
        digest.update(b"session.json\0" + sha256_file(sj).encode())
    expected = len(prompts["items"]) * (max((r["pass"] for r in records), default=0) or 1)
    if len(records) < expected:
        warnings.append(f"{len(records)} records for {len(prompts['items'])} prompts x {max((r['pass'] for r in records), default=1)} pass(es): the session is incomplete")
    if meta is None:
        warnings.append("legacy pilot layout: no session.json (prompt dataset supplied separately)")
    return {"dir": path, "kind": kind, "meta": meta, "records": records, "prompts": prompts, "problems": problems, "warnings": warnings, "digest": digest.hexdigest()}


# --------------------------------------------------------------------------------------
# analysis (answers only)
# --------------------------------------------------------------------------------------
def _blank():
    return {"records": 0, "valid": 0, "correct": 0, "competing": 0, "other_incorrect": 0, "non_answers": 0,
            "refusals": 0, "format_invalid_recoverable": 0, "ambiguous_or_missing": 0, "flr_correct": 0, "flr_competing": 0, "flr_other_incorrect": 0}


def _pair_blank():
    return {"pairs": 0, "pairs_unscored": 0, "answer_changed": 0, "induced_incorrect": 0, "induced_to_competing": 0, "induced_to_other": 0,
            "recovered_to_correct": 0, "baseline_incorrect_pairs": 0}


def analyze(records: list, prompts: dict) -> dict:
    """Descriptive answer-level analysis. Contains no probability, margin, confidence or ranking of any kind."""
    by_key, base = prompts["by_key"], prompts["baseline_condition"]
    idx = {(r["cell"], r["condition"], r["pass"]): r for r in records}
    by_cond, by_cm, pairs_c, pairs_cm = {}, {}, {}, {}
    for r in records:
        item = by_key[(r["cell"], r["condition"])]
        mp = item.get("mapping") or "-"
        for tab, key in ((by_cond, r["condition"]), (by_cm, (r["condition"], mp))):
            t = tab.setdefault(key, _blank())
            t["records"] += 1
            if r["status"] == "ok":
                t["valid"] += 1
                t[role_of(r["answer"], item)] += 1
            else:
                t["non_answers"] += 1
                d = diagnostic_class(r)      # reporting split only; the strict counts above are never touched
                if d["class"] == "refusal":
                    t["refusals"] += 1
                elif d["class"] == "format_recovered":
                    t["format_invalid_recoverable"] += 1
                    t["flr_" + role_of(d["flr_letter"], item)] += 1
                else:
                    t["ambiguous_or_missing"] += 1
    for r in records:
        if r["condition"] == base:
            continue
        item = by_key[(r["cell"], r["condition"])]
        mp = item.get("mapping") or "-"
        b = idx.get((r["cell"], base, r["pass"]))
        for tab, key in ((pairs_c, r["condition"]), (pairs_cm, (r["condition"], mp))):
            t = tab.setdefault(key, _pair_blank())
            if b is None or b["status"] != "ok" or r["status"] != "ok":
                t["pairs_unscored"] += 1
                continue
            t["pairs"] += 1
            rb, rc = role_of(b["answer"], item), role_of(r["answer"], item)
            if b["answer"] != r["answer"]:
                t["answer_changed"] += 1
            if rb != "correct":
                t["baseline_incorrect_pairs"] += 1
                if rc == "correct":
                    t["recovered_to_correct"] += 1
            elif rc != "correct":
                t["induced_incorrect"] += 1
                t["induced_to_competing" if rc == "competing" else "induced_to_other"] += 1
    groups, agree_c = {}, {}
    for r in records:
        if r["status"] == "ok":
            groups.setdefault((r["cell"], r["condition"]), []).append(r["answer"])
    agree = {"prompts_with_two_or_more_valid_passes": 0, "prompts_all_passes_agree": 0}
    for (cell, cond), answers in groups.items():
        if len(answers) < 2:
            continue
        t = agree_c.setdefault(cond, {"prompts_with_two_or_more_valid_passes": 0, "prompts_all_passes_agree": 0})
        for a in (agree, t):
            a["prompts_with_two_or_more_valid_passes"] += 1
            a["prompts_all_passes_agree"] += len(set(answers)) == 1
    cells = {}
    for r in sorted(records, key=lambda x: (x["cell"], x["condition"], x["pass"])):
        item = by_key[(r["cell"], r["condition"])]
        c = cells.setdefault(r["cell"], {"mapping": item.get("mapping"), "correct_letter": item["correct_letter"], "competing_letter": item["competing_letter"], "conditions": {}})
        c["conditions"].setdefault(r["condition"], []).append({"pass": r["pass"], "status": r["status"], "answer": r["answer"], "role": role_of(r["answer"], item)})
    u = [r.get("usage") or {} for r in records]
    cost = sum(r["cost_usd"] for r in records if isinstance(r.get("cost_usd"), (int, float)))
    return {"measurement_type": ANSWER_ONLY, "note": NOTE, "baseline_condition": base, "records": len(records), "passes": sorted({r["pass"] for r in records}),
            "status_counts": dict(Counter(r["status"] for r in records)), "by_condition": by_cond, "by_condition_mapping": {f"{k[0]}|{k[1]}": v for k, v in by_cm.items()},
            "versus_baseline": pairs_c, "versus_baseline_by_mapping": {f"{k[0]}|{k[1]}": v for k, v in pairs_cm.items()},
            "pass_agreement": {"all": agree, "by_condition": agree_c,
                               "note": "replicate stability of the observed answers; NOT a probability and NOT a confidence estimate"},
            "cells": cells,
            "usage": {"prompt_tokens": sum(x.get("promptTokenCount") or 0 for x in u), "candidate_tokens": sum(x.get("candidatesTokenCount") or 0 for x in u),
                      "thinking_tokens": sum(x.get("thoughtsTokenCount") or 0 for x in u), "records_with_incomplete_usage": sum(1 for r in records if not r.get("usage_complete", True))},
            "estimated_cost_usd": cost}


def report_lines(analysis: dict, title: str = "") -> list[str]:
    a, base = analysis, analysis["baseline_condition"]
    L = [f"{ANSWER_ONLY} {title}".strip(), f"  {NOTE}",
         f"  records {a['records']} (passes {a['passes']}); statuses {a['status_counts']}"]
    L.append(f"  {'condition':<14}{'records':>8}{'correct':>9}{'competing':>10}{'other':>7}{'no-answer':>10} |{'pairs':>6}{'changed':>8}{'induced':>8}{'(->comp)':>9}{'recovered':>10}")
    for cond, t in a["by_condition"].items():
        v = a["versus_baseline"].get(cond)
        tail = (f" |{v['pairs']:>6}{v['answer_changed']:>8}{v['induced_incorrect']:>8}{v['induced_to_competing']:>9}{v['recovered_to_correct']:>10}" if v else f" |{'(baseline)':>10}")
        L.append(f"  {cond:<14}{t['records']:>8}{t['correct']:>9}{t['competing']:>10}{t['other_incorrect']:>7}{t['non_answers']:>10}{tail}")
    L.append(f"  (pairs = same-pass comparisons with the clean baseline where both answers are valid; induced = the baseline answer was correct and the candidate's was not; '(->comp)' of those went to the competing answer)")
    if any(t["non_answers"] for t in a["by_condition"].values()):
        L.append(f"  no-answer split (strict statuses, unchanged) and the optional {FLR_RULE} diagnostic on the format-invalid replies:")
        for cond, t in a["by_condition"].items():
            if t["non_answers"]:
                L.append(f"    {cond:<14}refusals {t['refusals']}, format-invalid with one final letter {t['format_invalid_recoverable']} "
                         f"(diagnostic: correct {t['flr_correct']} / competing {t['flr_competing']} / other {t['flr_other_incorrect']}), ambiguous or missing {t['ambiguous_or_missing']}")
        L.append(f"  {FLR_NOTE}")
    maps = sorted({k.split("|", 1)[1] for k in a["by_condition_mapping"]})
    if maps != ["-"]:
        L.append("  by answer mapping (correct / competing / other incorrect / no answer):")
        for mp in maps:
            L.append(f"    {mp}: " + "; ".join(f"{c} {t['correct']}/{t['competing']}/{t['other_incorrect']}/{t['non_answers']}" for c in a["by_condition"]
                                              for t in [a["by_condition_mapping"].get(f"{c}|{mp}")] if t))
    pa = a["pass_agreement"]["all"]
    if pa["prompts_with_two_or_more_valid_passes"]:
        L.append(f"  repeated passes: the same answer in {pa['prompts_all_passes_agree']}/{pa['prompts_with_two_or_more_valid_passes']} prompts (replicate stability, not confidence)")
    us = a["usage"]
    L.append(f"  tokens: prompt {us['prompt_tokens']}, candidate {us['candidate_tokens']}, thinking {us['thinking_tokens']}; estimated cost ${a['estimated_cost_usd']:.4f}"
             + (f"; {us['records_with_incomplete_usage']} record(s) with incomplete usage metadata" if us["records_with_incomplete_usage"] else ""))
    return L


# --------------------------------------------------------------------------------------
# Atlas integration helpers
# --------------------------------------------------------------------------------------
def validate_specs(specs) -> list:
    if specs is None:
        return []
    if not isinstance(specs, list) or not specs:
        raise AnswersError("'answer_only_sessions' must be a non-empty list when present")
    seen = set()
    for i, s in enumerate(specs, 1):
        if not isinstance(s, dict) or not isinstance(s.get("session_dir"), str) or not s["session_dir"]:
            raise AnswersError(f"answer-only session #{i} needs a 'session_dir'")
        if not isinstance(s.get("model"), dict) or not s["model"].get("id"):
            raise AnswersError(f"answer-only session #{i} needs model.id")
        s.setdefault("id", f"answer-only/{s['model']['id']}")
        if s["id"] in seen:
            raise AnswersError(f"duplicate answer-only session id {s['id']!r}")
        seen.add(s["id"])
        if s.get("prompts") is not None and not isinstance(s["prompts"], str):
            raise AnswersError(f"answer-only session {s['id']!r}: 'prompts' must be a path string")
    return specs


def _local_counts(local_cells: list, label: str, baseline_label: str) -> dict:
    """Per local model: answer-level counts for one label (the arg-max preferred answer of each saved probability vector)."""
    by_model = {}
    for c in local_cells:
        if label == baseline_label:
            b = c.get("baseline") or {}
            pref = b.get("preferred")
            if not b or pref is None:
                continue
            t = by_model.setdefault(c["model_id"], {"n": 0, "correct": 0, "competing": 0, "other_incorrect": 0, "changed": "", "induced": ""})
            t["n"] += 1
            t["correct" if pref == c["correct_answer"] else ("competing" if pref == c["incorrect_answer"] else "other_incorrect")] += 1
            continue
        for k in c["candidates"]:
            if k["candidate_id"] != label or k.get("outcome") == "unscored":
                continue
            t = by_model.setdefault(c["model_id"], {"n": 0, "correct": 0, "competing": 0, "other_incorrect": 0, "changed": 0, "induced": 0})
            t["n"] += 1
            pref = k.get("candidate_preferred")
            t["correct" if pref == c["correct_answer"] else ("competing" if pref == c["incorrect_answer"] else "other_incorrect")] += 1
            t["changed"] += bool(k.get("overall_flip"))
            t["induced"] += k.get("outcome") == "induced_incorrect"
    return by_model


def comparison_rows(hosted: list, local_cells: list) -> list:
    """Side-by-side ANSWER-LEVEL rows. DIRECT_PROBABILITY rows use the arg-max preferred answer of the saved probabilities; ANSWER_ONLY rows use the
    returned answers. The two kinds are labelled and never merged; no probability quantity appears in any row."""
    rows, labels = [], []
    for h in hosted:
        for cond in h["analysis"]["by_condition"]:
            if cond not in labels:
                labels.append(cond)
    baseline_label = hosted[0]["analysis"]["baseline_condition"] if hosted else DEFAULT_BASELINE
    for lab in labels:
        for h in hosted:
            t = h["analysis"]["by_condition"].get(lab)
            if not t:
                continue
            v = h["analysis"]["versus_baseline"].get(lab) or {}
            rows.append({"measurement_type": ANSWER_ONLY, "source": h["model"]["id"], "label": lab, "n": t["valid"], "correct": t["correct"], "competing": t["competing"],
                         "other_incorrect": t["other_incorrect"], "no_answer": t["non_answers"], "changed_vs_baseline": v.get("answer_changed", ""), "induced_incorrect": v.get("induced_incorrect", ""),
                         "answer_basis": "returned answer (passes pooled)"})
        for m, t in sorted(_local_counts(local_cells, lab, baseline_label).items()):
            rows.append({"measurement_type": DIRECT_PROBABILITY, "source": m, "label": lab, "n": t["n"], "correct": t["correct"], "competing": t["competing"], "other_incorrect": t["other_incorrect"],
                         "no_answer": 0, "changed_vs_baseline": t["changed"], "induced_incorrect": t["induced"], "answer_basis": "arg-max preferred answer of the saved A-D probabilities"})
    return rows


COMPARISON_COLUMNS = ["measurement_type", "source", "label", "n", "correct", "competing", "other_incorrect", "no_answer", "changed_vs_baseline", "induced_incorrect", "answer_basis"]
ANSWER_ONLY_COLUMNS = ["measurement_type", "session_id", "model_id", "condition", "mapping", "records", "valid", "correct", "competing", "other_incorrect", "non_answers", "pairs", "pairs_unscored",
                       "answer_changed", "induced_incorrect", "induced_to_competing", "induced_to_other", "recovered_to_correct", "baseline_incorrect_pairs",
                       "refusals", "format_invalid_recoverable", "ambiguous_or_missing", "flr_v1_correct", "flr_v1_competing", "flr_v1_other_incorrect"]


def answer_only_rows(hosted: list) -> list:
    rows = []
    for h in hosted:
        a = h["analysis"]
        entries = [(cond, "all", t, a["versus_baseline"].get(cond)) for cond, t in a["by_condition"].items()]
        for key, t in a["by_condition_mapping"].items():
            cond, mp = key.split("|", 1)
            entries.append((cond, mp, t, a["versus_baseline_by_mapping"].get(key)))
        for cond, mp, t, v in entries:
            v = v or {}
            rows.append({"measurement_type": ANSWER_ONLY, "session_id": h["id"], "model_id": h["model"]["id"], "condition": cond, "mapping": mp, "records": t["records"], "valid": t["valid"],
                         "correct": t["correct"], "competing": t["competing"], "other_incorrect": t["other_incorrect"], "non_answers": t["non_answers"], "pairs": v.get("pairs", ""),
                         "pairs_unscored": v.get("pairs_unscored", ""), "answer_changed": v.get("answer_changed", ""), "induced_incorrect": v.get("induced_incorrect", ""),
                         "induced_to_competing": v.get("induced_to_competing", ""), "induced_to_other": v.get("induced_to_other", ""), "recovered_to_correct": v.get("recovered_to_correct", ""),
                         "baseline_incorrect_pairs": v.get("baseline_incorrect_pairs", ""),
                         "refusals": t["refusals"], "format_invalid_recoverable": t["format_invalid_recoverable"], "ambiguous_or_missing": t["ambiguous_or_missing"],
                         "flr_v1_correct": t["flr_correct"], "flr_v1_competing": t["flr_competing"], "flr_v1_other_incorrect": t["flr_other_incorrect"]})
    return rows


def build_hosted(specs: list, base: str, local_cells: list, *, keep_going: bool = False, display=None) -> dict:
    """Read, validate and analyse the answer-only sessions of an Atlas manifest. Returns the ``answer_only`` block of atlas.json."""
    sessions, excluded = [], []
    for s in specs:
        sd = os.path.normpath(os.path.join(base, s["session_dir"]))
        pp = os.path.normpath(os.path.join(base, s["prompts"])) if s.get("prompts") else None
        shown = display(sd, base) if display else os.path.basename(sd)
        try:
            ses = read_session(sd, pp)
        except AnswersError as e:
            if not keep_going:
                raise AnswersError(f"answer-only session {s['id']!r} failed validation: {e} (use --keep-going to exclude it)") from None
            excluded.append({"id": s["id"], "session_dir": shown, "errors": [str(e)]})
            continue
        if ses["problems"]:
            if not keep_going:
                raise AnswersError(f"answer-only session {s['id']!r} failed validation: " + "; ".join(ses["problems"][:5]) + " (use --keep-going to exclude it)")
            excluded.append({"id": s["id"], "session_dir": shown, "errors": ses["problems"]})
            continue
        provider = (ses["meta"] or {}).get("provider") or (ses["records"][0].get("provider") if ses["records"] else "unknown")
        versions = sorted({r.get("model_version") for r in ses["records"] if r.get("model_version")})
        sessions.append({"id": s["id"], "measurement_type": ANSWER_ONLY, "provider": provider, "model": {"id": s["model"]["id"], "name": s["model"].get("name", s["model"]["id"]), "returned_versions": versions},
                         "session_dir": shown, "kind": ses["kind"], "digest": ses["digest"], "prompt_dataset_sha256": ses["prompts"]["sha256"], "warnings": ses["warnings"], "analysis": analyze(ses["records"], ses["prompts"])})
    return {"measurement_type": ANSWER_ONLY, "note": NOTE, "sessions": sessions, "excluded": excluded,
            "comparison": comparison_rows(sessions, local_cells) if sessions else [],
            "comparison_note": ("Answer-level view only. DIRECT_PROBABILITY rows are the arg-max preferred answers of saved probability vectors from local sessions; ANSWER_ONLY rows are returned answers of "
                                "hosted models. Presentations differ (raw completion vs chat), the rows are never merged and no probability quantity is derived for ANSWER_ONLY data.")}


def _md_row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def atlas_markdown(block: dict, models: list | None = None) -> list[str]:
    """Report sections for the ANSWER_ONLY part of an Atlas. Clearly separate from the DIRECT_PROBABILITY sections; contains no margins or rankings."""
    sessions = [s for s in block["sessions"] if not models or s["model"]["id"] in models]
    L = ["## Hosted answer-only results (ANSWER_ONLY)", "", block["note"], "",
         "These sessions come from hosted models that returned answers but no A-D probabilities. They are reported here and in `answer-only.csv` only; they never enter the margin, "
         "shift, rank or paired-difference sections above, and no probability margin or leaderboard is computed for them. Repeated identical answers show replicate stability, "
         "not confidence.", ""]
    for s in sessions:
        a = s["analysis"]
        L += [f"### {s['id']} (model `{s['model']['id']}`, provider {s['provider']}, measurement type {ANSWER_ONLY})", "",
              f"Saved session `{s['session_dir']}` ({s['kind']}); {a['records']} records over passes {a['passes']}; statuses {a['status_counts']}; estimated cost ${a['estimated_cost_usd']:.4f}.", "",
              _md_row("Condition", "Records", "Correct", "Competing", "Other incorrect", "No answer", "Pairs vs baseline", "Answer changed", "Newly induced incorrect", "of which to competing", "Recovered"),
              _md_row(*["---"] * 11)]
        for cond, t in a["by_condition"].items():
            v = a["versus_baseline"].get(cond) or {}
            L.append(_md_row(cond, t["records"], t["correct"], t["competing"], t["other_incorrect"], t["non_answers"], v.get("pairs", "(baseline)"), v.get("answer_changed", ""),
                             v.get("induced_incorrect", ""), v.get("induced_to_competing", ""), v.get("recovered_to_correct", "")))
        if any(t["non_answers"] for t in a["by_condition"].values()):
            L += ["", f"No-answer split. Strict statuses are unchanged; the last three columns are the optional **{FLR_RULE} diagnostic** on format-invalid replies only (not the primary metric).", "",
                  _md_row("Condition", "No answer (strict)", "Explicit refusals / blocks", "Format-invalid, one final letter", "Ambiguous or missing", f"{FLR_RULE} correct", f"{FLR_RULE} competing", f"{FLR_RULE} other incorrect"),
                  _md_row(*["---"] * 8)]
            for cond, t in a["by_condition"].items():
                if t["non_answers"]:
                    L.append(_md_row(cond, t["non_answers"], t["refusals"], t["format_invalid_recoverable"], t["ambiguous_or_missing"], t["flr_correct"], t["flr_competing"], t["flr_other_incorrect"]))
            L += ["", FLR_NOTE]
        maps = sorted({k.split("|", 1)[1] for k in a["by_condition_mapping"]})
        if maps != ["-"]:
            L += ["", _md_row("Answer mapping", *a["by_condition"].keys()), _md_row(*["---"] * (1 + len(a["by_condition"])))]
            for mp in maps:
                cells = []
                for c in a["by_condition"]:
                    t = a["by_condition_mapping"].get(f"{c}|{mp}")
                    cells.append("{}/{}/{}/{}".format(t["correct"], t["competing"], t["other_incorrect"], t["non_answers"]) if t else "")
                L.append(_md_row(mp, *cells))
            L += ["", "(cells: correct / competing / other incorrect / no answer)"]
        pa = a["pass_agreement"]["all"]
        if pa["prompts_with_two_or_more_valid_passes"]:
            L += ["", f"Repeated passes gave the same answer in {pa['prompts_all_passes_agree']} of {pa['prompts_with_two_or_more_valid_passes']} prompts ({a['pass_agreement']['note']})."]
        for w in s.get("warnings", []):
            L.append(f"- Note: {w}")
        L.append("")
    for x in block.get("excluded", []):
        L.append(f"- EXCLUDED answer-only session `{x['id']}`: " + "; ".join(x["errors"][:3]))
    rows = [r for r in block["comparison"] if r["measurement_type"] == DIRECT_PROBABILITY or not models or r["source"] in models]
    if rows:
        L += ["", "## Answer-level comparison of the two measurement types (descriptive)", "", block["comparison_note"], "",
              _md_row("Measurement type", "Source", "Label", "n", "Correct", "Competing", "Other incorrect", "No answer", "Changed vs baseline", "Newly induced incorrect", "Answer basis"), _md_row(*["---"] * 11)]
        for r in rows:
            L.append(_md_row(*[r[c] for c in COMPARISON_COLUMNS]))
        L.append("")
    return L


def atlas_summary(block: dict, models: list | None = None) -> list[str]:
    sessions = [s for s in block["sessions"] if not models or s["model"]["id"] in models]
    L = ["", f"{ANSWER_ONLY} hosted sessions (reported separately from the {DIRECT_PROBABILITY} results above; no margins, shifts or rankings exist for them):"]
    for s in sessions:
        for line in report_lines(s["analysis"], f"{s['id']} (model {s['model']['id']})"):
            L.append("  " + line)
    if any(r["measurement_type"] == ANSWER_ONLY for r in block["comparison"]):
        L.append("  Answer-level comparison rows (labelled by measurement type) are in answer-level-comparison.csv and report.md.")
    for x in block.get("excluded", []):
        L.append(f"  EXCLUDED answer-only session {x['id']}: " + "; ".join(x["errors"][:2]))
    return L
