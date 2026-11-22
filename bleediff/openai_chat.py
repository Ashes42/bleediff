"""``bleediff openai``: ANSWER-ONLY measurements on the OpenAI Chat Completions API, plus offline replay of saved evidence.

Only the OpenAI provider record lives here (endpoint, ``Authorization: Bearer`` header, request body, model-family parameter rules, output-token reserve). The engine
(retries, redaction, cost estimation, the spending cap, sessions, replay, the command handler) is ``bleediff.hosted``; the analysis is ``bleediff.answers``.

ANSWER_ONLY. This adapter requests no logprobs and derives no probabilities, log-odds, confidence or ranking. (The original v0.7 pilot also asked for top-20 logprobs and found
all four answer letters in only 28 of 160 responses; that partial exposure is not used here. A model that exposes usable A-D token probabilities could back a DIRECT_PROBABILITY
adapter, which is a different thing.) Keys come from ``OPENAI_API_KEY`` only.

SUPPORTED PARAMETERS BY MODEL FAMILY (a deliberately small table; unknown families get provider defaults only):
* ``gpt-4*`` (gpt-4.1, gpt-4o, ...): ``temperature`` (default 1), ``top_p`` (default 1), ``seed``, ``max_completion_tokens``.
* reasoning families (``o1``, ``o3``, ``o4``, ``gpt-5``, ``gpt-6``): no ``temperature``/``top_p`` (an explicit value is refused before any request), optional ``--reasoning-effort``;
  ``max_completion_tokens`` bounds visible AND reasoning tokens, so it is also the output reserve of the spending cap.
* anything else: only ``seed`` and ``max_completion_tokens``; ``--temperature``, ``--top-p`` and ``--reasoning-effort`` are refused.
Spending limits are estimates from your supplied prices and the reported usage (see ``bleediff/hosted.py``); cached-token discounts are ignored (conservative).
"""

from __future__ import annotations

import json
import time

from . import hosted as H
from .answers import parse_openai_answer, sha256_bytes, usage_from_openai
from .errors import Inconclusive
from .hosted import HostedError, Provider, redact  # noqa: F401  (re-exported)

API_BASE = "https://api.openai.com/v1"
ENV_KEY = "OPENAI_API_KEY"
REASONING_EFFORTS = ("minimal", "low", "medium", "high")
REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5", "gpt-6")
SAMPLING_PREFIXES = ("gpt-4",)
OpenAIError = HostedError
normalise_model = H.normalise_model


def model_family(model: str) -> str:
    """'sampling' (temperature/top_p allowed), 'reasoning' (reasoning_effort allowed) or 'other' (provider defaults only)."""
    m = model.lower()
    if m.startswith(REASONING_PREFIXES):
        return "reasoning"
    return "sampling" if m.startswith(SAMPLING_PREFIXES) else "other"


def build_request(prompt: str, model: str, *, temperature: float | None = None, top_p: float | None = None, seed: int = 0, max_output_tokens: int = 512,
                  reasoning_effort: str | None = None) -> dict:
    """One Chat Completions body with only the parameters the model family supports. Logprobs are deliberately NOT requested."""
    fam = model_family(model)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_completion_tokens": max_output_tokens, "n": 1, "seed": seed}
    if fam == "sampling":
        body["temperature"] = 1.0 if temperature is None else temperature
        body["top_p"] = 1.0 if top_p is None else top_p
    elif temperature is not None or top_p is not None:
        raise Inconclusive(f"model {model!r} is not in the sampling-parameter family: --temperature and --top-p are not sent for it (omit them)", reason="BAD_ARGS")
    if reasoning_effort is not None:
        if fam != "reasoning":
            raise Inconclusive(f"--reasoning-effort is only supported for reasoning models (o1, o3, o4, gpt-5, gpt-6); {model!r} is not one", reason="BAD_ARGS")
        if reasoning_effort not in REASONING_EFFORTS:
            raise Inconclusive(f"--reasoning-effort must be one of {list(REASONING_EFFORTS)}", reason="BAD_ARGS")
        body["reasoning_effort"] = reasoning_effort
    return body


def _build(prompt: str, model: str, cfg: dict, options: dict) -> dict:
    return build_request(prompt, model, temperature=cfg["temperature"], top_p=cfg["top_p"], seed=cfg["seed"], max_output_tokens=cfg["max_output_tokens"], reasoning_effort=options.get("reasoning_effort"))


OPENAI = Provider(
    name="openai", label="OpenAI", env_key=ENV_KEY, endpoint=lambda m: f"{API_BASE}/chat/completions", auth_headers=lambda k: {"Authorization": "Bearer " + k},
    build_request=_build,
    request_record=lambda body, ep: {"endpoint": ep, "parameters": {k: v for k, v in body.items() if k != "messages"}, "messages_sha256": sha256_bytes(json.dumps(body["messages"], sort_keys=True).encode("utf-8"))},
    parse=parse_openai_answer, usage=usage_from_openai, version_of=lambda r: r.get("model") if isinstance(r, dict) else None,
    output_reserve=lambda max_out, options: max_out,
    logprobs_note="never requested: this adapter is ANSWER_ONLY (top-20 logprobs cover all four answer letters only when the model is uncertain, so they are not a usable A-D distribution)")


def api_key(env=None) -> str:
    return H.api_key(OPENAI, env)


def run_session(*, reasoning_effort: str | None = None, **kw) -> dict:
    return H.run_session(OPENAI, options={"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}, **kw)


def cmd_openai(args, *, out=None, post=None, sleep=time.sleep, env=None) -> int:
    return H.cmd_hosted(args, OPENAI, ("reasoning_effort",), out=out, post=post, sleep=sleep, env=env)
