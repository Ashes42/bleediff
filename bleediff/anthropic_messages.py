"""``bleediff anthropic``: ANSWER-ONLY measurements on the Anthropic Messages API (Claude), plus offline replay of saved evidence.

Only the Anthropic provider record lives here (endpoint, ``x-api-key`` + ``anthropic-version`` headers, request body, thinking/effort parameter rules, output-token reserve). The engine
(retries, redaction, cost estimation, the spending cap, sessions, replay, the command handler) is ``bleediff.hosted``; the analysis is ``bleediff.answers``.

ANSWER_ONLY. The Messages API has no logprobs parameter, so no A-D probabilities exist: this adapter derives no probabilities, log-odds, confidence or ranking. Keys come from
``ANTHROPIC_API_KEY`` only.

THINKING. Claude 4.7+ and 5.x use ADAPTIVE thinking (the model decides whether to think; ``output_config.effort`` steers it; ``budget_tokens`` is rejected). ``max_tokens`` is a hard cap on
thinking PLUS text, and ``output_tokens`` in the usage already includes thinking tokens. A live probe of ``claude-haiku-5-5`` with ``max_tokens`` 8 returned an empty thinking block and no text
(``stop_reason: max_tokens``): such a reply is recorded as an empty non-answer, never guessed. Use a ``--max-output-tokens`` large enough for thinking (default 1024) and, if needed,
``--effort low``. That cap is also the per-request spending reserve. ``--temperature`` and ``--top-p`` are optional, never both, and manual thinking needs temperature 1.
Spending limits are estimates from your supplied prices and the reported usage (see ``bleediff/hosted.py``); prompt-cache discounts are ignored (cached input is charged at the full price).
"""

from __future__ import annotations

import json
import time

from . import hosted as H
from .answers import parse_anthropic_answer, sha256_bytes, usage_from_anthropic
from .errors import Inconclusive
from .hosted import HostedError, Provider, redact  # noqa: F401  (re-exported)

API_BASE = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"
ENV_KEY = "ANTHROPIC_API_KEY"
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")
CLAUDE_THINKING = ("adaptive", "disabled")
AnthropicError = HostedError
normalise_model = H.normalise_model


def build_request(prompt: str, model: str, *, temperature: float | None = None, top_p: float | None = None, max_output_tokens: int = 1024,
                  effort: str | None = None, thinking: str | None = None, thinking_budget: int | None = None) -> dict:
    """One Messages body. Only parameters the API documents are sent; optional ones only when you ask. There is no logprobs parameter to send."""
    body = {"model": model, "max_tokens": max_output_tokens, "messages": [{"role": "user", "content": prompt}]}
    if temperature is not None and top_p is not None:
        raise Inconclusive("use either --temperature or --top-p, not both (some Claude models reject both together)", reason="BAD_ARGS")
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if thinking is not None and thinking_budget is not None:
        raise Inconclusive("use either --thinking or --thinking-budget, not both", reason="BAD_ARGS")
    if thinking is not None:
        if thinking not in CLAUDE_THINKING:
            raise Inconclusive(f"--thinking must be one of {list(CLAUDE_THINKING)}", reason="BAD_ARGS")
        body["thinking"] = {"type": thinking}
    elif thinking_budget is not None:
        if not (isinstance(thinking_budget, int) and 1024 <= thinking_budget < max_output_tokens):
            raise Inconclusive("--thinking-budget must be an integer >= 1024 and below --max-output-tokens", reason="BAD_ARGS")
        if temperature is not None and temperature != 1:
            raise Inconclusive("manual thinking requires temperature 1 (omit --temperature)", reason="BAD_ARGS")
        body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    if effort is not None:
        if effort not in CLAUDE_EFFORTS:
            raise Inconclusive(f"--effort must be one of {list(CLAUDE_EFFORTS)}", reason="BAD_ARGS")
        body["output_config"] = {"effort": effort}
    return body


def _build(prompt: str, model: str, cfg: dict, options: dict) -> dict:
    return build_request(prompt, model, temperature=cfg["temperature"], top_p=cfg["top_p"], max_output_tokens=cfg["max_output_tokens"], effort=options.get("effort"),
                         thinking=options.get("thinking"), thinking_budget=options.get("thinking_budget"))


ANTHROPIC = Provider(
    name="anthropic", label="Anthropic", env_key=ENV_KEY, endpoint=lambda m: f"{API_BASE}/messages", auth_headers=lambda k: {"x-api-key": k, "anthropic-version": API_VERSION},
    build_request=_build,
    request_record=lambda body, ep: {"endpoint": ep, "parameters": {k: v for k, v in body.items() if k != "messages"}, "messages_sha256": sha256_bytes(json.dumps(body["messages"], sort_keys=True).encode("utf-8"))},
    parse=parse_anthropic_answer, usage=usage_from_anthropic, version_of=lambda r: r.get("model") if isinstance(r, dict) else None,
    output_reserve=lambda max_out, options: max_out,                     # max_tokens is a hard cap on thinking + text
    logprobs_note="not available: the Anthropic Messages API has no logprobs parameter, so this session is ANSWER_ONLY")


def api_key(env=None) -> str:
    return H.api_key(ANTHROPIC, env)


def run_session(*, effort: str | None = None, thinking: str | None = None, thinking_budget: int | None = None, **kw) -> dict:
    options = {k: v for k, v in (("effort", effort), ("thinking", thinking), ("thinking_budget", thinking_budget)) if v is not None}
    return H.run_session(ANTHROPIC, options=options, **kw)


def cmd_anthropic(args, *, out=None, post=None, sleep=time.sleep, env=None) -> int:
    return H.cmd_hosted(args, ANTHROPIC, ("effort", "thinking", "thinking_budget"), out=out, post=post, sleep=sleep, env=env)
