"""``bleediff gemini``: ANSWER-ONLY measurements on the Google Gemini API (``generateContent``), plus offline replay of saved evidence.

This module is only the Gemini provider record (endpoint, ``x-goog-api-key`` header, request body, thinking parameters, output-token reserve). The engine (retries, redaction,
cost estimation, the spending cap, the session writer, replay and the command handler) is ``bleediff.hosted``; the analysis is ``bleediff.answers``.

The tested Gemini API configuration rejects ``responseLogprobs`` for every tested model ("Logprobs is not enabled for this model"), so this adapter NEVER requests logprobs,
never probes for them, and never produces probabilities, log-odds, confidence or a probability-margin ranking. Keys come from ``GEMINI_API_KEY`` only.
Spending limits are estimates from your supplied prices and the reported usage (see ``bleediff/hosted.py``).
"""

from __future__ import annotations

import json
import time

from . import hosted as H
from .answers import parse_answer, sha256_bytes, usage_from
from .errors import Inconclusive
from .hosted import MAX_CONSECUTIVE_FAILURES, TRANSIENT_HTTP, HostedError, Provider, TransportError, billed_cost, prompt_tokens_upper, redact  # noqa: F401  (re-exported)

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
ENV_KEY = "GEMINI_API_KEY"
THINKING_LEVELS = ("minimal", "low", "medium", "high")
GeminiError = HostedError
normalise_model = H.normalise_model


def build_request(prompt: str, *, temperature: float = 1.0, top_p: float = 1.0, seed: int = 0, max_output_tokens: int = 512,
                  thinking_level: str | None = None, thinking_budget: int | None = None) -> dict:
    """One ``generateContent`` body. Log-probabilities are deliberately NOT requested (the tested configuration rejects them)."""
    gen = {"temperature": temperature, "topP": top_p, "maxOutputTokens": max_output_tokens, "candidateCount": 1, "seed": seed}
    if thinking_level is not None and thinking_budget is not None:
        raise Inconclusive("use either --thinking-level or --thinking-budget, not both", reason="BAD_ARGS")
    if thinking_level is not None and thinking_level not in THINKING_LEVELS:
        raise Inconclusive(f"--thinking-level must be one of {list(THINKING_LEVELS)}", reason="BAD_ARGS")
    if thinking_budget is not None and not (isinstance(thinking_budget, int) and thinking_budget >= 0):
        raise Inconclusive("--thinking-budget must be a non-negative integer", reason="BAD_ARGS")
    if thinking_level is not None:
        gen["thinkingConfig"] = {"thinkingLevel": thinking_level}
    elif thinking_budget is not None:
        gen["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    return {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gen}


def _build(prompt: str, model: str, cfg: dict, options: dict) -> dict:
    return build_request(prompt, temperature=1.0 if cfg["temperature"] is None else cfg["temperature"], top_p=1.0 if cfg["top_p"] is None else cfg["top_p"], seed=cfg["seed"],
                         max_output_tokens=cfg["max_output_tokens"], thinking_level=options.get("thinking_level"), thinking_budget=options.get("thinking_budget"))


def _reserve(max_output_tokens: int, options: dict) -> int:
    """Bound on billed output tokens of one request: the visible-output cap plus a thinking allowance (the explicit budget, or - when none is set and thinking may
    run - another full cap, because it is not certain that thinking tokens count against maxOutputTokens)."""
    tb = options.get("thinking_budget")
    return max_output_tokens + (max_output_tokens if tb is None else tb)


GEMINI = Provider(
    name="gemini", label="Gemini", env_key=ENV_KEY, endpoint=lambda m: f"{API_BASE}/models/{m}:generateContent", auth_headers=lambda k: {"x-goog-api-key": k},
    build_request=_build,
    request_record=lambda body, ep: {"endpoint": ep, "generationConfig": body["generationConfig"], "contents_sha256": sha256_bytes(json.dumps(body["contents"], sort_keys=True).encode("utf-8"))},
    parse=parse_answer, usage=usage_from, version_of=lambda r: r.get("modelVersion") if isinstance(r, dict) else None, output_reserve=_reserve,
    logprobs_note="never requested: every tested Gemini model rejects responseLogprobs, so this session is ANSWER_ONLY")


def api_key(env=None) -> str:
    return H.api_key(GEMINI, env)


def _http_post(url: str, body: bytes, key: str, timeout: float):
    return H.http_post(GEMINI, url, body, key, timeout)


def generate(post, model: str, body: dict, key: str, *, timeout: float, max_retries: int, sleep) -> dict:
    return H.generate(GEMINI, post, model, body, key, timeout=timeout, max_retries=max_retries, sleep=sleep)


def run_session(*, thinking_level: str | None = None, thinking_budget: int | None = None, temperature: float = 1.0, top_p: float = 1.0, **kw) -> dict:
    options = {k: v for k, v in (("thinking_level", thinking_level), ("thinking_budget", thinking_budget)) if v is not None}
    return H.run_session(GEMINI, temperature=temperature, top_p=top_p, options=options, **kw)


def cmd_gemini(args, *, out=None, post=None, sleep=time.sleep, env=None) -> int:
    return H.cmd_hosted(args, GEMINI, ("thinking_level", "thinking_budget"), out=out, post=post, sleep=sleep, env=env)
