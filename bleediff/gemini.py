"""``bleediff gemini``: ANSWER-ONLY measurements on the Google Gemini API (``generateContent``), plus offline replay of saved evidence.

* Live mode sends the prompts of a frozen dataset (see ``bleediff.answers``) to one model, for one or more passes, parses each reply as one
  answer letter A-D, and writes a NEW session directory: ``session.json``, ``prompts.json`` (a byte copy of the dataset) and
  ``records/<cell>__<condition>__pass<k>.json`` (request configuration, full response, usage, estimated cost). Every record is labelled ``ANSWER_ONLY``.
* Replay mode (``--replay DIR``) re-reads saved evidence, re-parses every saved response and prints the descriptive analysis. It needs no key and no network.

MEASUREMENT INTEGRITY. The tested Gemini API configuration rejects ``responseLogprobs`` for every tested model ("Logprobs is not enabled for
this model"), so this adapter NEVER requests logprobs, never probes for them, and never produces probabilities, log-odds, confidence or a
probability-margin ranking. See ``bleediff/answers.py``.

SECRETS. The key is read from ``GEMINI_API_KEY``, sent only in the ``x-goog-api-key`` header, and is never printed, saved, put in a URL, or included in an
error message (messages are redacted, and underlying transport exceptions are not chained).

COST AND BUDGET. Live runs require an explicit ``--budget`` (USD) and explicit list prices (``--price-in`` / ``--price-out``, USD per 1M tokens): there are
no built-in prices, because they change. Before every request the worst case of the NEXT request (an upper bound on its prompt tokens plus a reserve for
output and thinking tokens) must fit under the remaining budget, otherwise the run stops. After a response the cost comes from ``usageMetadata`` (prompt
tokens at the input price; candidate + thinking tokens, and any tokens only visible in ``totalTokenCount``, at the output price). Missing or inconsistent
usage metadata is charged at the worst-case bound and flagged, never treated as free. Retries are bounded (``--max-retries``, default 2).
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from . import __version__
from .answers import (ANSWER_ONLY, RECORD_SCHEMA, SESSION_SCHEMA, AnswersError, analyze, load_prompts, parse_answer, read_session, report_lines, sha256_bytes, usage_from)
from .errors import Inconclusive
from .meta import code_revision

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
ENV_KEY = "GEMINI_API_KEY"
THINKING_LEVELS = ("minimal", "low", "medium", "high")
TRANSIENT_HTTP = frozenset({429, 500, 502, 503, 504})
MAX_CONSECUTIVE_FAILURES = 3
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
_KEYLIKE = re.compile(r"AIza[0-9A-Za-z_\-]{10,}|(?i:x-goog-api-key)[\"':= ]+[^\s\"',}]+|(?i:[?&]key=)[^\s&\"']+")


def redact(text, key: str | None = None) -> str:
    """Remove the API key (and anything shaped like one) from a message."""
    s = str(text)
    if key:
        s = s.replace(key, "[REDACTED]")
    return _KEYLIKE.sub("[REDACTED]", s)


class GeminiError(Inconclusive):
    """A Gemini adapter failure. The message is always redacted."""

    reason = "GEMINI_ERROR"

    def __init__(self, message: str, *, reason: str | None = None, http_status: int | None = None, transient: bool = False, key: str | None = None):
        super().__init__(redact(message, key), reason=reason)
        self.http_status, self.transient = http_status, transient


def api_key(env=None) -> str:
    key = (os.environ if env is None else env).get(ENV_KEY, "")
    if not isinstance(key, str) or not key.strip():
        raise GeminiError(f"{ENV_KEY} is not set: export your Gemini API key in the environment (it is never read from a file or printed)", reason="MISSING_API_KEY")
    return key.strip()


def normalise_model(model: str) -> str:
    m = model[len("models/"):] if isinstance(model, str) and model.startswith("models/") else model
    if not isinstance(m, str) or not _MODEL_RE.match(m):
        raise Inconclusive(f"invalid model name {model!r}: use a plain Gemini model id such as 'gemini-3.8-flash'", reason="BAD_ARGS")
    return m


def build_request(prompt: str, *, temperature: float = 1.0, top_p: float = 1.0, seed: int = 0, max_output_tokens: int = 512,
                  thinking_level: str | None = None, thinking_budget: int | None = None) -> dict:
    """One ``generateContent`` body. Log-probabilities are deliberately NOT requested (the tested configuration rejects them)."""
    gen = {"temperature": temperature, "topP": top_p, "maxOutputTokens": max_output_tokens, "candidateCount": 1, "seed": seed}
    if thinking_level is not None and thinking_budget is not None:
        raise Inconclusive("use either --thinking-level or --thinking-budget, not both", reason="BAD_ARGS")
    if thinking_level is not None:
        gen["thinkingConfig"] = {"thinkingLevel": thinking_level}
    elif thinking_budget is not None:
        gen["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    return {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gen}


# --------------------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------------------
class TransportError(Exception):
    """Network-level failure (no HTTP response). Carries only an exception class name, never a message that could contain a URL or a header."""


def _http_post(url: str, body: bytes, key: str, timeout: float):
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers.items()), r.read(8_000_000)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers.items()) if e.headers else {}, e.read(200_000)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise TransportError(type(e).__name__) from None


def _server_message(raw: bytes, key: str) -> str:
    try:
        msg = json.loads(raw.decode("utf-8", "replace")).get("error", {}).get("message", "")
    except (ValueError, AttributeError):
        msg = raw[:200].decode("utf-8", "replace")
    return redact(" ".join(str(msg).split())[:300], key)


def generate(post, model: str, body: dict, key: str, *, timeout: float, max_retries: int, sleep) -> dict:
    """POST one request with bounded retries. Returns {'json', 'attempts', 'http_status'}; raises GeminiError (redacted) otherwise."""
    url = f"{API_BASE}/models/{model}:generateContent"
    data = json.dumps(body).encode("utf-8")
    attempts = 0
    while True:
        attempts += 1
        try:
            status, headers, raw = post(url, data, key, timeout)
        except TransportError as e:
            if attempts <= max_retries:
                sleep(min(30, 2 ** attempts))
                continue
            raise GeminiError(f"network error ({e}) after {attempts} attempt(s)", reason="TRANSPORT_ERROR", transient=True, key=key) from None
        if status == 200:
            try:
                return {"json": json.loads(raw.decode("utf-8")), "attempts": attempts, "http_status": 200}
            except ValueError:
                raise GeminiError("the API returned HTTP 200 with a body that is not JSON", reason="BAD_RESPONSE", http_status=200, key=key) from None
        msg = _server_message(raw, key)
        if status in TRANSIENT_HTTP:
            if attempts <= max_retries:
                ra = (headers or {}).get("Retry-After") or (headers or {}).get("retry-after")
                wait = min(60, int(ra)) if isinstance(ra, str) and ra.isdigit() else min(30, 2 ** attempts)
                sleep(wait)
                continue
            raise GeminiError(f"HTTP {status} after {attempts} attempt(s): {msg}", reason="RATE_LIMITED" if status == 429 else "SERVER_ERROR", http_status=status, transient=True, key=key)
        reason = {401: "AUTH_FAILED", 403: "AUTH_FAILED", 404: "MODEL_NOT_AVAILABLE", 400: "BAD_REQUEST"}.get(status, "HTTP_ERROR")
        raise GeminiError(f"HTTP {status}: {msg}", reason=reason, http_status=status, key=key)


# --------------------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------------------
def prompt_tokens_upper(prompt: str) -> int:
    """A deliberately high upper bound: one token per UTF-8 byte, plus a margin for chat framing."""
    return len(prompt.encode("utf-8")) + 64


def billed_cost(usage: dict, price_in: float, price_out: float, prompt_upper: int, out_reserve: int):
    """(cost_usd, basis, usage_complete). Prices are USD per token. Missing or inconsistent usage metadata is charged at the worst-case bound."""
    pt, ct, th, tt = (usage.get(k) for k in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount"))
    worst = prompt_upper * price_in + out_reserve * price_out
    if pt is None or (ct is None and th is None and tt is None) or (tt is not None and tt < pt):
        return worst, "upper_bound_missing_or_inconsistent_usage", False
    out = (ct or 0) + (th or 0)
    basis = "usage_metadata"
    if tt is not None and tt - pt > out:
        out, basis = tt - pt, "usage_metadata_total_minus_prompt"          # tokens only visible in the total are billed as output, conservatively
    return pt * price_in + out * price_out, basis, True


# --------------------------------------------------------------------------------------
# a live session
# --------------------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_new(path: str, obj) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")


def run_session(*, prompts_path: str, model: str, out_dir: str, passes: int, budget: float, price_in_per_m: float, price_out_per_m: float, key: str,
                post=None, sleep=time.sleep, max_output_tokens: int = 512, temperature: float = 1.0, top_p: float = 1.0, seed: int = 0, thinking_level: str | None = None,
                thinking_budget: int | None = None, max_retries: int = 2, timeout: float = 120.0, log=lambda s: None) -> dict:
    post = post or _http_post
    model = normalise_model(model)
    if not (isinstance(passes, int) and 1 <= passes <= 20):
        raise Inconclusive("--passes must be an integer from 1 to 20", reason="BAD_ARGS")
    for name, v in (("--budget", budget), ("--price-in", price_in_per_m), ("--price-out", price_out_per_m)):
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
            raise Inconclusive(f"{name} must be a non-negative number", reason="BAD_ARGS")
    if budget <= 0:
        raise Inconclusive("--budget must be greater than zero (live runs need an explicit spending cap)", reason="BAD_ARGS")
    if not (isinstance(max_retries, int) and 0 <= max_retries <= 5):
        raise Inconclusive("--max-retries must be an integer from 0 to 5", reason="BAD_ARGS")
    if thinking_level is not None and thinking_level not in THINKING_LEVELS:
        raise Inconclusive(f"--thinking-level must be one of {list(THINKING_LEVELS)}", reason="BAD_ARGS")
    if not (isinstance(max_output_tokens, int) and 1 <= max_output_tokens <= 65536):
        raise Inconclusive("--max-output-tokens must be an integer from 1 to 65536", reason="BAD_ARGS")
    if os.path.exists(out_dir):
        raise Inconclusive(f"{out_dir!r} already exists: choose a new session directory (evidence is never overwritten)", reason="OUTPUT_EXISTS")
    prompts = load_prompts(prompts_path)
    pin, pout = price_in_per_m / 1e6, price_out_per_m / 1e6
    gen_probe = build_request("x", temperature=temperature, top_p=top_p, seed=seed, max_output_tokens=max_output_tokens, thinking_level=thinking_level, thinking_budget=thinking_budget)["generationConfig"]
    os.makedirs(os.path.join(out_dir, "records"))
    with open(prompts_path, "rb") as fh, open(os.path.join(out_dir, "prompts.json"), "xb") as dst:
        dst.write(fh.read())
    meta = {"schema": SESSION_SCHEMA, "measurement_type": ANSWER_ONLY, "provider": "gemini", "model": model, "endpoint": f"{API_BASE}/models/{model}:generateContent",
            "generation_config": gen_probe, "logprobs_requested": False,
            "logprobs_note": "never requested: every tested Gemini model rejects responseLogprobs, so this session is ANSWER_ONLY",
            "prompts": {"file": "prompts.json", "source_name": prompts["name"], "sha256": prompts["sha256"], "items": len(prompts["items"])}, "passes": passes,
            "budget": {"cap_usd": budget, "price_in_per_million_tokens": price_in_per_m, "price_out_per_million_tokens": price_out_per_m,
                       "rules": "worst case of the next request must fit under the remaining cap; missing usage is charged at the worst-case bound; failed transport/5xx requests are charged at the worst-case bound; 4xx/429 are not charged"},
            "max_retries": max_retries, "bleediff_version": __version__, "code_revision": code_revision(), "started_utc": _now()}
    spent, max_seen_out, consecutive, n_req = 0.0, 0, 0, 0
    status, stop_reason, loop_done = "complete", None, False
    try:
        for pas in range(1, passes + 1):
            for item in prompts["items"]:
                name = f"{item['cell']}__{item['condition']}__pass{pas}"
                upper_prompt = prompt_tokens_upper(item["prompt"])
                reserve = max(max_output_tokens + (thinking_budget or 0), 2 * max_seen_out)
                worst = upper_prompt * pin + reserve * pout
                if spent + worst > budget:
                    status, stop_reason = "partial", f"BUDGET_EXHAUSTED: ${spent:.6f} spent; the worst case of the next request (${worst:.6f}) would exceed the ${budget:.2f} cap"
                    raise StopIteration
                body = build_request(item["prompt"], temperature=temperature, top_p=top_p, seed=seed, max_output_tokens=max_output_tokens, thinking_level=thinking_level, thinking_budget=thinking_budget)
                base = {"schema": RECORD_SCHEMA, "measurement_type": ANSWER_ONLY, "provider": "gemini", "name": name, "cell": item["cell"], "question": item.get("question"),
                        "mapping": item.get("mapping"), "condition": item["condition"], "pass": pas, "prompt_sha256": item["prompt_sha256"], "fragment_sha256": item.get("fragment_sha256"),
                        "correct_letter": item["correct_letter"], "competing_letter": item["competing_letter"], "model": model,
                        "request": {"endpoint": meta["endpoint"], "generationConfig": body["generationConfig"], "contents_sha256": sha256_bytes(json.dumps(body["contents"], sort_keys=True).encode("utf-8")),
                                    "credentials_recorded": False}}
                t0 = time.time()
                n_req += 1
                try:
                    got = generate(post, model, body, key, timeout=timeout, max_retries=max_retries, sleep=sleep)
                except GeminiError as e:
                    charged = worst if e.reason in ("TRANSPORT_ERROR", "SERVER_ERROR", "BAD_RESPONSE") else 0.0
                    spent += charged
                    _write_new(os.path.join(out_dir, "records", name + ".json"), {**base, "status": "error", "answer": None, "http_status": e.http_status, "attempts": None, "elapsed_s": round(time.time() - t0, 3),
                                                                              "error": {"reason": e.reason, "message": str(e)[:400], "transient": e.transient}, "usage": {}, "usage_complete": False,
                                                                              "cost_usd": charged, "cost_basis": "worst_case_bound (the request may have been processed)" if charged else "not charged (request rejected)"})
                    consecutive += 1
                    if not e.transient or consecutive >= MAX_CONSECUTIVE_FAILURES:
                        status, stop_reason = "aborted", f"{e.reason}: {e}"
                        raise StopIteration
                    continue
                consecutive = 0
                resp = got["json"]
                parsed = parse_answer(resp)
                usage = usage_from(resp)
                cost, basis, complete = billed_cost(usage, pin, pout, upper_prompt, reserve)
                spent += cost
                max_seen_out = max(max_seen_out, (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0), (usage.get("totalTokenCount") or 0) - (usage.get("promptTokenCount") or 0))
                _write_new(os.path.join(out_dir, "records", name + ".json"), {**base, **parsed, "model_version": resp.get("modelVersion") if isinstance(resp, dict) else None, "http_status": got["http_status"],
                                                                              "attempts": got["attempts"], "elapsed_s": round(time.time() - t0, 3), "usage": usage, "usage_complete": complete, "cost_usd": cost,
                                                                              "cost_basis": basis, "response": resp, "error": None})
                log(f"{name}: {parsed['status']} {parsed['answer'] or ''} (${spent:.4f} of ${budget:.2f})")
        loop_done = True
    except StopIteration:
        loop_done = True
    finally:
        if not loop_done and status == "complete":
            status, stop_reason = "aborted", "interrupted or unexpected error"
        meta.update({"finished_utc": _now(), "status": status, "stop_reason": stop_reason, "requests_attempted": n_req, "estimated_spend_usd": spent})
        _write_new(os.path.join(out_dir, "session.json"), meta)
    return {"status": status, "stop_reason": stop_reason, "estimated_spend_usd": spent, "requests_attempted": n_req, "session_dir": out_dir}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def cmd_gemini(args, *, out=None, post=None, sleep=time.sleep, env=None) -> int:
    import sys
    out = out or sys.stdout
    key = None
    try:
        if args.replay:
            live_only = [n for n in ("model", "out", "budget", "price_in", "price_out") if getattr(args, n, None) is not None]
            if live_only:
                raise Inconclusive("--replay reads saved evidence only; do not combine it with " + ", ".join("--" + n.replace("_", "-") for n in live_only), reason="BAD_ARGS")
            ses = read_session(args.replay, args.prompts)
            title = f"replay of {os.path.basename(os.path.normpath(args.replay))} ({ses['kind']}; model {(ses['meta'] or {}).get('model') or ses['records'][0].get('model')})"
            print("OFFLINE REPLAY of saved evidence (no key used, no network, no model contacted)", file=out)
            for w in ses["warnings"]:
                print(f"NOTE: {w}", file=out)
            if ses["problems"]:
                for p in ses["problems"][:20]:
                    print(f"PROBLEM: {p}", file=out)
                raise AnswersError(f"{len(ses['problems'])} validation problem(s) in the saved evidence")
            analysis = analyze(ses["records"], ses["prompts"])
        else:
            missing = [n for n in ("prompts", "model", "out", "budget", "price_in", "price_out") if getattr(args, n, None) is None]
            if missing:
                raise Inconclusive("a live run needs " + ", ".join("--" + n.replace("_", "-") for n in missing) + " (the spending cap and list prices are explicit by design); or use --replay DIR", reason="BAD_ARGS")
            model = normalise_model(args.model)
            key = api_key(env)
            print(f"LIVE Gemini ANSWER-ONLY run: model {model}, {args.passes} pass(es), budget ${args.budget:.2f} (this sends requests with your key and may cost money)", file=out)
            res = run_session(prompts_path=args.prompts, model=model, out_dir=args.out, passes=args.passes, budget=args.budget, price_in_per_m=args.price_in, price_out_per_m=args.price_out, key=key,
                              post=post, sleep=sleep, max_output_tokens=args.max_output_tokens, temperature=args.temperature, top_p=args.top_p, seed=args.seed, thinking_level=args.thinking_level,
                              thinking_budget=args.thinking_budget, max_retries=args.max_retries, timeout=args.timeout)
            if res["requests_attempted"] == 0:
                print(f"no request was sent; estimated spend $0.0000 of ${args.budget:.2f}; session saved to {args.out}", file=out)
                print(f"STOPPED ({res['status']}): {res['stop_reason']}", file=out)
                return 2
            ses = read_session(args.out)
            analysis = analyze(ses["records"], ses["prompts"])
            title = f"session {os.path.basename(os.path.normpath(args.out))} (model {model})"
            if ses["problems"]:
                for p in ses["problems"][:20]:
                    print(f"PROBLEM: {p}", file=out)
        for line in report_lines(analysis, title):
            print(line, file=out)
        if args.json_out:
            if os.path.exists(args.json_out):
                raise Inconclusive(f"{args.json_out!r} already exists: choose a new file (reports are never overwritten)", reason="OUTPUT_EXISTS")
            _write_new(args.json_out, {"measurement_type": ANSWER_ONLY, "digest": ses["digest"], "analysis": analysis})
            print(f"wrote {args.json_out}", file=out)
        if not args.replay:
            print(f"estimated spend ${res['estimated_spend_usd']:.4f} of ${args.budget:.2f}; session saved to {args.out}", file=out)
            if res["status"] != "complete":
                print(f"STOPPED ({res['status']}): {res['stop_reason']}", file=out)
                return 2
        return 0
    except Inconclusive as e:
        print(f"ERROR [{e.reason}]: {redact(str(e), key)}", file=out)
        return 2
    except Exception as e:                                           # never let an unexpected exception echo a secret
        print(f"ERROR [UNEXPECTED]: {type(e).__name__}", file=out)
        return 2
