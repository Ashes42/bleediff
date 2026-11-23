"""Shared engine of the hosted ANSWER_ONLY adapters (``bleediff gemini``, ``bleediff openai``, ``bleediff anthropic``).

A provider module supplies a small ``Provider`` record (endpoint, auth header, request builder, response parser, usage mapping, output-token reserve); this
module supplies everything else once: bounded retries, secret redaction, cost estimation and the enforced spending cap, the session writer (``session.json``,
``prompts.json``, ``records/*.json``), offline replay and the command handler. The analysis and the Atlas helpers live in ``bleediff.answers`` and are shared too.

MEASUREMENT INTEGRITY. These adapters produce ANSWER_ONLY data. They never request, derive or approximate probabilities, log-odds, confidence or a ranking from
answers. A model that exposes usable A-D token probabilities could support DIRECT_PROBABILITY measurements, but that is a different adapter, not this one.

SECRETS. Keys come from the environment only, travel only in a request header, and are removed (exact value and key-shaped strings) from every message and record;
transport exceptions are never chained, so their text cannot leak a key.

SPENDING LIMITS ARE ESTIMATES. The cap is enforced against YOUR supplied list prices and the usage numbers the provider reports; it is not a guaranteed provider
billing total (cached-token discounts, tiers, taxes, retries on the provider side and price changes can all differ). Before every request the worst case of the NEXT
request must fit under the remaining cap:
    worst = (UTF-8 bytes of the prompt + 64) x input price  +  output reserve x output price
where the output reserve is the provider's bound on billed output tokens (OpenAI: ``max_completion_tokens``, which includes reasoning tokens; Anthropic: ``max_tokens``, a hard cap on thinking plus text; Gemini: ``maxOutputTokens``
plus a thinking allowance, the explicit thinking budget or, when none is set, another ``maxOutputTokens``), raised to twice the largest output count observed so far.
After a response the cost is computed from the reported usage (candidate + thinking tokens, and any tokens visible only in the total, at the output price). Missing or
inconsistent usage is charged at ``worst`` and flagged; a response whose cost exceeds the bound it was admitted under stops the run (COST_BOUND_EXCEEDED). Retries are bounded and budget-admitted: a retry is sent only if one more full worst-case attempt fits under the remaining cap, and every failed
attempt that may have been billed (transport error, HTTP 5xx) is charged at ``worst``; 429 and other 4xx responses are not. None of this is a guarantee about the provider's bill.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from . import __version__
from .answers import ANSWER_ONLY, RECORD_SCHEMA, SESSION_SCHEMA, AnswersError, analyze, load_prompts, read_session, report_lines, sha256_bytes
from .errors import Inconclusive
from .meta import code_revision

TRANSIENT_HTTP = frozenset({429, 500, 502, 503, 504, 529})          # 529 = Anthropic "overloaded"
NON_RETRYABLE_QUOTA_CODES = frozenset({"insufficient_quota", "billing_not_active"})
MAX_CONSECUTIVE_FAILURES = 3
PASSES_RANGE = (1, 20)                                   # accepted by run_session; `bleediff experiment validate` checks the same ranges
MAX_OUTPUT_TOKENS_RANGE = (1, 65536)
MAX_RETRIES_RANGE = (0, 5)
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_OUTPUT_TOKENS = {"anthropic": 1024, "openai": 512, "gemini": 512}      # the CLI defaults of --max-output-tokens
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
_KEYLIKE = re.compile(r"AIza[0-9A-Za-z_\-]{10,}|sk-[A-Za-z0-9_\-]{16,}|(?i:x-goog-api-key[\"':= ]+[^\s\"',}]+)|(?i:[?&]key=[^\s&\"']+)|(?i:authorization[\"':= ]+(?:bearer )?[^\s\"',}]+)|(?i:bearer [A-Za-z0-9._\-]{12,})")


def redact(text, key: str | None = None) -> str:
    """Remove the API key (and anything shaped like one) from a message."""
    s = str(text)
    if key:
        s = s.replace(key, "[REDACTED]")
    return _KEYLIKE.sub("[REDACTED]", s)


class HostedError(Inconclusive):
    """A hosted-adapter failure. The message is always redacted."""

    reason = "HOSTED_ERROR"

    def __init__(self, message: str, *, reason: str | None = None, http_status: int | None = None, transient: bool = False, key: str | None = None):
        super().__init__(redact(message, key), reason=reason)
        self.http_status, self.transient = http_status, transient
        self.billable_attempts = 0           # attempts of this request that may have been processed (and billed) by the provider; set by generate()


class TransportError(Exception):
    """Network-level failure (no HTTP response). Carries only an exception class name, never a message that could contain a URL or a header."""


@dataclass(frozen=True)
class Provider:
    name: str                                            # "gemini" | "openai"
    label: str
    env_key: str
    endpoint: Callable[[str], str]                       # model -> URL
    auth_headers: Callable[[str], dict]                  # key -> headers (the only place the key is used)
    build_request: Callable[[str, str, dict, dict], dict]    # (prompt, model, cfg, options) -> request body; raises Inconclusive(BAD_ARGS) for unsupported parameters
    request_record: Callable[[dict, str], dict]          # (body, endpoint) -> provenance of the request WITHOUT the prompt text
    parse: Callable[[object], dict]
    usage: Callable[[object], dict]                      # -> promptTokenCount / candidatesTokenCount / thoughtsTokenCount / totalTokenCount
    version_of: Callable[[object], str | None]
    output_reserve: Callable[[int, dict], int]           # (max output tokens, options) -> bound on billed output tokens of one request
    logprobs_note: str


def api_key(provider: Provider, env=None) -> str:
    key = (os.environ if env is None else env).get(provider.env_key, "")
    if not isinstance(key, str) or not key.strip():
        raise HostedError(f"{provider.env_key} is not set: export your {provider.label} API key in the environment (it is never read from a file or printed)", reason="MISSING_API_KEY")
    return key.strip()


def normalise_model(model) -> str:
    m = model[len("models/"):] if isinstance(model, str) and model.startswith("models/") else model
    if not isinstance(m, str) or not _MODEL_RE.match(m):
        raise Inconclusive(f"invalid model name {model!r}: use a plain model id such as 'gemini-3.8-flash', 'gpt-4.1-mini' or 'claude-haiku-5-5'", reason="BAD_ARGS")
    return m


# --------------------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------------------
def http_post(provider: Provider, url: str, body: bytes, key: str, timeout: float):
    headers = {"Content-Type": "application/json", **provider.auth_headers(key)}
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers.items()), r.read(8_000_000)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers.items()) if e.headers else {}, e.read(200_000)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise TransportError(type(e).__name__) from None


def _server_error(raw: bytes, key: str):
    """(redacted message, error code or None) from an error body of either provider."""
    try:
        err = json.loads(raw.decode("utf-8", "replace")).get("error", {})
        msg = err.get("message", "") if isinstance(err, dict) else str(err)
        code = err.get("code") if isinstance(err, dict) else None
        code = code if isinstance(code, str) else (err.get("type") if isinstance(err, dict) and isinstance(err.get("type"), str) else None)
    except (ValueError, AttributeError):
        msg, code = raw[:200].decode("utf-8", "replace"), None
    return redact(" ".join(str(msg).split())[:300], key), code


SIGNATURE_PLACEHOLDER = "[REDACTED:opaque-thinking-signature]"
REASONING_PLACEHOLDER = "[REDACTED:private-reasoning]"


def scrub_private_reasoning(obj):
    """A copy of a provider response with opaque thinking signatures and any returned reasoning text replaced by placeholders, so saved records never hold them.

    Anthropic ``thinking`` / ``redacted_thinking`` blocks, Gemini ``thoughtSignature`` and ``thought: true`` parts. The answer analysis never reads these fields (it uses only visible text
    blocks and the token counts), so replay is unaffected. Idempotent."""
    if isinstance(obj, list):
        return [scrub_private_reasoning(x) for x in obj]
    if not isinstance(obj, dict):
        return obj
    out, kind = {}, obj.get("type")
    for k, v in obj.items():
        nonempty = isinstance(v, str) and v and not v.startswith("[REDACTED:")
        if k in ("signature", "thoughtSignature", "thought_signature") and nonempty:
            out[k] = SIGNATURE_PLACEHOLDER
        elif (kind == "thinking" and k == "thinking" and nonempty) or (obj.get("thought") is True and k == "text" and nonempty):
            out[k] = REASONING_PLACEHOLDER
        elif kind == "redacted_thinking" and k == "data" and nonempty:
            out[k] = SIGNATURE_PLACEHOLDER
        else:
            out[k] = scrub_private_reasoning(v)
    return out


def generate(provider: Provider, post, model: str, body: dict, key: str, *, timeout: float, max_retries: int, sleep, admit=None) -> dict:
    """POST one request with bounded retries. Returns {'json', 'attempts', 'http_status', 'billable_failures'}; raises HostedError (redacted) otherwise.

    RETRY ACCOUNTING. An attempt that ends in a transport error or an HTTP 5xx (including 529) may have been processed and billed by the provider, so it is *billable*; 429 and other
    4xx responses are not. ``admit(billable_so_far)`` is called before every RETRY and must return True if one more full worst-case attempt still fits under the remaining estimated
    budget; if it returns False the retry is not sent and the request fails with BUDGET_EXHAUSTED. Every raised HostedError and the returned dict carry the count of billable
    attempts (``billable_attempts`` / ``billable_failures``) so the caller can charge each one conservatively."""
    url = provider.endpoint(model)
    data = json.dumps(body).encode("utf-8")
    attempts = billable = 0

    def fail(message, **kw):
        e = HostedError(message, key=key, **kw)
        e.billable_attempts = billable
        return e

    def may_retry() -> bool:
        return attempts <= max_retries and (admit is None or admit(billable))

    while True:
        attempts += 1
        try:
            status, headers, raw = post(url, data, key, timeout)
        except TransportError as e:
            billable += 1
            if may_retry():
                sleep(min(30, 2 ** attempts))
                continue
            if attempts <= max_retries:
                raise fail(f"BUDGET_EXHAUSTED: a retry after attempt {attempts} (network error {e}) would exceed the estimated budget", reason="BUDGET_EXHAUSTED") from None
            raise fail(f"network error ({e}) after {attempts} attempt(s)", reason="TRANSPORT_ERROR", transient=True) from None
        if status == 200:
            try:
                return {"json": json.loads(raw.decode("utf-8")), "attempts": attempts, "http_status": 200, "billable_failures": billable}
            except ValueError:
                billable += 1
                raise fail("the API returned HTTP 200 with a body that is not JSON", reason="BAD_RESPONSE", http_status=200) from None
        msg, code = _server_error(raw, key)
        if status >= 500:
            billable += 1
        if status == 429 and code in NON_RETRYABLE_QUOTA_CODES:         # an exhausted quota will not recover by retrying
            raise fail(f"HTTP 429 ({code}): {msg}", reason="QUOTA_EXHAUSTED", http_status=429)
        if status in TRANSIENT_HTTP:
            if may_retry():
                ra = (headers or {}).get("Retry-After") or (headers or {}).get("retry-after")
                sleep(min(60, int(ra)) if isinstance(ra, str) and ra.isdigit() else min(30, 2 ** attempts))
                continue
            if attempts <= max_retries:
                raise fail(f"BUDGET_EXHAUSTED: a retry after attempt {attempts} (HTTP {status}) would exceed the estimated budget", reason="BUDGET_EXHAUSTED", http_status=status)
            raise fail(f"HTTP {status} after {attempts} attempt(s): {msg}", reason="RATE_LIMITED" if status == 429 else "SERVER_ERROR", http_status=status, transient=True)
        reason = {401: "AUTH_FAILED", 403: "AUTH_FAILED", 404: "MODEL_NOT_AVAILABLE", 400: "BAD_REQUEST"}.get(status, "HTTP_ERROR")
        raise fail(f"HTTP {status}: {msg}", reason=reason, http_status=status)


# --------------------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------------------
def prompt_tokens_upper(prompt: str) -> int:
    """A deliberately high upper bound: one token per UTF-8 byte, plus a margin for chat framing."""
    return len(prompt.encode("utf-8")) + 64


def request_reserve(provider: Provider, max_output_tokens: int, options: dict, max_seen_out: int = 0) -> int:
    """Output tokens the guard reserves for the next request: the provider's bound, raised to twice the largest output seen so far in the session."""
    return max(provider.output_reserve(max_output_tokens, options), 2 * max_seen_out)


def admission_bound(prompt_upper: int, price_in: float, price_out: float, reserve: int) -> float:
    """Worst-case cost (USD; prices per token) of one request: the figure that must fit under the remaining budget before it is sent."""
    return prompt_upper * price_in + reserve * price_out


def billed_cost(usage: dict, price_in: float, price_out: float, prompt_upper: int, out_reserve: int):
    """(cost_usd, basis, usage_complete). Prices are USD per token. Missing or inconsistent usage metadata is charged at the worst-case bound."""
    pt, ct, th, tt = (usage.get(k) for k in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount"))
    worst = admission_bound(prompt_upper, price_in, price_out, out_reserve)
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


def write_new(path: str, obj) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")


def run_session(provider: Provider, *, prompts_path: str, model: str, out_dir: str, passes: int, budget: float, price_in_per_m: float, price_out_per_m: float, key: str,
                post=None, sleep=time.sleep, max_output_tokens: int = 512, temperature=None, top_p=None, seed: int = 0, options: dict | None = None,
                max_retries: int = DEFAULT_MAX_RETRIES, timeout: float = 120.0, log=lambda s: None) -> dict:
    post = post or (lambda url, body, k, t: http_post(provider, url, body, k, t))
    options = dict(options or {})
    model = normalise_model(model)
    if not (isinstance(passes, int) and PASSES_RANGE[0] <= passes <= PASSES_RANGE[1]):
        raise Inconclusive("--passes must be an integer from 1 to 20", reason="BAD_ARGS")
    for name, v in (("--budget", budget), ("--price-in", price_in_per_m), ("--price-out", price_out_per_m)):
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
            raise Inconclusive(f"{name} must be a non-negative number", reason="BAD_ARGS")
    if budget <= 0:
        raise Inconclusive("--budget must be greater than zero (live runs need an explicit spending cap)", reason="BAD_ARGS")
    if not (isinstance(max_retries, int) and MAX_RETRIES_RANGE[0] <= max_retries <= MAX_RETRIES_RANGE[1]):
        raise Inconclusive("--max-retries must be an integer from 0 to 5", reason="BAD_ARGS")
    if not (isinstance(max_output_tokens, int) and MAX_OUTPUT_TOKENS_RANGE[0] <= max_output_tokens <= MAX_OUTPUT_TOKENS_RANGE[1]):
        raise Inconclusive("--max-output-tokens must be an integer from 1 to 65536", reason="BAD_ARGS")
    if os.path.exists(out_dir):
        raise Inconclusive(f"{out_dir!r} already exists: choose a new session directory (evidence is never overwritten)", reason="OUTPUT_EXISTS")
    cfg = {"temperature": temperature, "top_p": top_p, "seed": seed, "max_output_tokens": max_output_tokens}
    endpoint = provider.endpoint(model)
    probe_body = provider.build_request("x", model, cfg, options)                 # validates every parameter against the provider/model BEFORE anything is created
    prompts = load_prompts(prompts_path)
    pin, pout = price_in_per_m / 1e6, price_out_per_m / 1e6
    os.makedirs(os.path.join(out_dir, "records"))
    with open(prompts_path, "rb") as fh, open(os.path.join(out_dir, "prompts.json"), "xb") as dst:
        dst.write(fh.read())
    rr = provider.request_record(probe_body, endpoint)
    meta = {"schema": SESSION_SCHEMA, "measurement_type": ANSWER_ONLY, "provider": provider.name, "model": model, "endpoint": endpoint,
            "generation_config": rr.get("generationConfig") or rr.get("parameters"),
            "logprobs_requested": False, "logprobs_note": provider.logprobs_note,
            "prompts": {"file": "prompts.json", "source_name": prompts["name"], "sha256": prompts["sha256"], "items": len(prompts["items"])}, "passes": passes,
            "budget": {"cap_usd": budget, "price_in_per_million_tokens": price_in_per_m, "price_out_per_million_tokens": price_out_per_m, "kind": "ESTIMATE from your supplied prices and reported usage; not a provider billing total",
                       "rules": "the worst case of the next request must fit under the remaining cap; missing usage is charged at the worst-case bound; a response exceeding its admitted bound stops the run; "
                                "every failed transport/5xx attempt (including ones that are retried) is charged at the worst-case bound and a retry must fit under the remaining cap as one more worst-case attempt; 4xx/429 are not charged"},
            "max_retries": max_retries, "bleediff_version": __version__, "code_revision": code_revision(), "started_utc": _now()}
    spent, max_seen_out, consecutive, n_req = 0.0, 0, 0, 0
    status, stop_reason, loop_done = "complete", None, False
    try:
        for pas in range(1, passes + 1):
            for item in prompts["items"]:
                name = f"{item['cell']}__{item['condition']}__pass{pas}"
                upper_prompt = prompt_tokens_upper(item["prompt"])
                reserve = request_reserve(provider, max_output_tokens, options, max_seen_out)
                worst = admission_bound(upper_prompt, pin, pout, reserve)
                if spent + worst > budget:
                    status, stop_reason = "partial", f"BUDGET_EXHAUSTED: ${spent:.6f} spent (estimate); the worst case of the next request (${worst:.6f}) would exceed the ${budget:.2f} cap"
                    raise StopIteration
                body = provider.build_request(item["prompt"], model, cfg, options)
                req_rec = {**provider.request_record(body, endpoint), "credentials_recorded": False}
                base = {"schema": RECORD_SCHEMA, "measurement_type": ANSWER_ONLY, "provider": provider.name, "name": name, "cell": item["cell"], "question": item.get("question"),
                        "mapping": item.get("mapping"), "condition": item["condition"], "pass": pas, "prompt_sha256": item["prompt_sha256"], "fragment_sha256": item.get("fragment_sha256"),
                        "correct_letter": item["correct_letter"], "competing_letter": item["competing_letter"], "model": model, "request": req_rec}
                t0 = time.time()
                n_req += 1
                try:
                    got = generate(provider, post, model, body, key, timeout=timeout, max_retries=max_retries, sleep=sleep,
                                   admit=lambda billable, _s=spent, _w=worst: _s + (billable + 1) * _w <= budget)       # a retry must fit as one more full worst-case attempt
                except HostedError as e:
                    charged = worst * e.billable_attempts                # every attempt that may have been billed is charged at the worst-case bound
                    spent += charged
                    write_new(os.path.join(out_dir, "records", name + ".json"), {**base, "status": "error", "answer": None, "http_status": e.http_status, "attempts": None, "elapsed_s": round(time.time() - t0, 3),
                                                                             "error": {"reason": e.reason, "message": str(e)[:400], "transient": e.transient}, "usage": {}, "usage_complete": False,
                                                                             "cost_usd": charged, "billable_attempts": e.billable_attempts,
                                                                             "cost_basis": f"worst_case_bound x {e.billable_attempts} attempt(s) that may have been processed" if charged else "not charged (request rejected)"})
                    consecutive += 1
                    if not e.transient or consecutive >= MAX_CONSECUTIVE_FAILURES:
                        status, stop_reason = "aborted", f"{e.reason}: {e}"
                        raise StopIteration
                    continue
                consecutive = 0
                resp = got["json"]
                parsed = provider.parse(resp)
                usage = provider.usage(resp)
                cost, basis, complete = billed_cost(usage, pin, pout, upper_prompt, reserve)
                exceeded = complete and cost > worst * (1 + 1e-9) + 1e-12
                retried = got.get("billable_failures", 0)
                if retried:                                              # earlier failed attempts of this request may also have been billed: each is charged at the worst-case bound
                    cost += retried * worst
                    basis = f"{basis}+{retried}_failed_attempt(s)_at_worst_case_bound"
                spent += cost
                max_seen_out = max(max_seen_out, (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0), (usage.get("totalTokenCount") or 0) - (usage.get("promptTokenCount") or 0))
                write_new(os.path.join(out_dir, "records", name + ".json"), {**base, **parsed, "model_version": provider.version_of(resp), "http_status": got["http_status"], "attempts": got["attempts"],
                                                                         "elapsed_s": round(time.time() - t0, 3), "usage": usage, "usage_complete": complete, "cost_usd": cost, "cost_basis": basis,
                                                                         "cost_exceeded_preflight_bound": bool(exceeded), "response": scrub_private_reasoning(resp), "error": None})
                log(f"{name}: {parsed['status']} {parsed['answer'] or ''} (${spent:.4f} of ${budget:.2f})")
                if exceeded:
                    status, stop_reason = "aborted", f"COST_BOUND_EXCEEDED: the reported cost (${cost:.6f}) exceeded the bound (${worst:.6f}) the request was admitted under"
                    raise StopIteration
        loop_done = True
    except StopIteration:
        loop_done = True
    finally:
        if not loop_done and status == "complete":
            status, stop_reason = "aborted", "interrupted or unexpected error"
        meta.update({"finished_utc": _now(), "status": status, "stop_reason": stop_reason, "requests_attempted": n_req, "estimated_spend_usd": spent})
        write_new(os.path.join(out_dir, "session.json"), meta)
    return {"status": status, "stop_reason": stop_reason, "estimated_spend_usd": spent, "requests_attempted": n_req, "session_dir": out_dir}


# --------------------------------------------------------------------------------------
# CLI handler (one for every provider)
# --------------------------------------------------------------------------------------
def cmd_hosted(args, provider: Provider, option_names: tuple, *, out=None, post=None, sleep=time.sleep, env=None) -> int:
    out = out or sys.stdout
    key = None
    try:
        if args.replay:
            live_only = [n for n in ("model", "out", "budget", "price_in", "price_out") if getattr(args, n, None) is not None]
            if live_only:
                raise Inconclusive("--replay reads saved evidence only; do not combine it with " + ", ".join("--" + n.replace("_", "-") for n in live_only), reason="BAD_ARGS")
            ses = read_session(args.replay, args.prompts)
            first = ses["records"][0] if ses["records"] else {}
            title = f"replay of {os.path.basename(os.path.normpath(args.replay))} ({ses['kind']}; {first.get('provider', '?')} model {(ses['meta'] or {}).get('model') or first.get('model')})"
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
            key = api_key(provider, env)
            print(f"LIVE {provider.label} ANSWER-ONLY run: model {model}, {args.passes} pass(es), estimated-spend cap ${args.budget:.2f} (this sends requests with your key and may cost money)", file=out)
            res = run_session(provider, prompts_path=args.prompts, model=model, out_dir=args.out, passes=args.passes, budget=args.budget, price_in_per_m=args.price_in, price_out_per_m=args.price_out, key=key,
                              post=post, sleep=sleep, max_output_tokens=args.max_output_tokens, temperature=args.temperature, top_p=args.top_p, seed=getattr(args, "seed", 0),
                              options={n: getattr(args, n, None) for n in option_names if getattr(args, n, None) is not None}, max_retries=args.max_retries, timeout=args.timeout)
            if res["requests_attempted"] == 0:
                print(f"no request was sent; estimated spend $0.0000 of ${args.budget:.2f}; session saved to {args.out}", file=out)
                print(f"STOPPED ({res['status']}): {res['stop_reason']}", file=out)
                return 2
            ses = read_session(args.out)
            analysis = analyze(ses["records"], ses["prompts"])
            title = f"session {os.path.basename(os.path.normpath(args.out))} ({provider.name} model {model})"
            for p in ses["problems"][:20]:
                print(f"PROBLEM: {p}", file=out)
        for line in report_lines(analysis, title):
            print(line, file=out)
        if args.json_out:
            if os.path.exists(args.json_out):
                raise Inconclusive(f"{args.json_out!r} already exists: choose a new file (reports are never overwritten)", reason="OUTPUT_EXISTS")
            write_new(args.json_out, {"measurement_type": ANSWER_ONLY, "digest": ses["digest"], "analysis": analysis})
            print(f"wrote {args.json_out}", file=out)
        if not args.replay:
            print(f"estimated spend ${res['estimated_spend_usd']:.4f} of ${args.budget:.2f} (an ESTIMATE from your prices and the reported usage, not a provider billing total); session saved to {args.out}", file=out)
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
