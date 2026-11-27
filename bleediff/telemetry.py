"""Behavioral telemetry (v0.7.0): ONE observation per saved request, with every measurement carrying an explicit measurement STATE.

This module reads a saved hosted-answer record (``bleediff.hosted-answer/1``, the record every ``bleediff anthropic|openai|gemini`` run writes, and the legacy pilot layout) and
produces a ``bleediff.behavior-observation/1``. It never contacts a provider, never needs a key and never modifies the evidence: the raw provider response stays in the record,
and each measurement names the exact field it was read from (``source``).

THE FIVE STATES (a measurement is always exactly one of them; only OBSERVED has a value):

    OBSERVED      the provider (or the harness) reported a valid value. ZERO IS A LEGITIMATE OBSERVED VALUE: ``thinking_tokens = 0`` means the provider said "0 thinking tokens".
    MISSING       the adapter defines this measurement for the provider, but the record does not contain it (field absent or null, no response, error record). Never converted to zero.
    UNSUPPORTED   this adapter does not establish this measurement for this provider or layout (for example reasoning tokens of a provider whose semantics v0.7.0 has not
                  validated, or latency in a layout that never recorded it). The raw provider field, if any, is preserved passively under ``provider_fields``.
    INCONSISTENT  values are present but contradict each other or a documented invariant (thinking tokens above the output tokens, a stored figure that disagrees with the saved
                  response). The conflicting value is kept in ``reported``; it is never used in an aggregate.
    INVALID       the field is present but is not a valid measurement of its kind (a bool, a string, a negative or non-finite number, a structure of the wrong type).

SEPARATION. Answer behaviour (``answer_outcome``, ``answer_role``), format and refusal, provider-reported thinking (``thinking_tokens``, ``thinking_activation``) and resource
expenditure (token counts, latency, cost) are separate measurements. Nothing here combines them, and there is no composite behavioral score.

ADAPTERS exist only where the semantics can be established. ``anthropic`` is validated against archived evidence (Research 002: 2,560 Messages-API records). ``openai`` and ``gemini``
reuse the v0.6 answer parsers and the v0.6 usage mapping for answers, format, refusal and input tokens; their reasoning/thinking telemetry is UNSUPPORTED here: different providers do
not expose equivalent reasoning measurements, and v0.7.0 does not assume they do.

ANSWER_ONLY: this layer derives no probabilities, margins, confidence or ranking.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Callable

from . import answers as A
from .errors import Inconclusive

OBSERVATION_SCHEMA = "bleediff.behavior-observation/1"
OBSERVED, MISSING, UNSUPPORTED, INCONSISTENT, INVALID = "OBSERVED", "MISSING", "UNSUPPORTED", "INCONSISTENT", "INVALID"
STATES = (OBSERVED, MISSING, UNSUPPORTED, INCONSISTENT, INVALID)
NO_ANSWER = "no_answer"
ROLES = ("correct", "competing", "other_incorrect", NO_ANSWER)

#: Measurement name -> kind of value an OBSERVED measurement must hold.
MEASURE_KINDS = {
    "input_tokens": "count", "cache_read_tokens": "count", "cache_write_tokens": "count",
    "output_tokens": "count", "thinking_tokens": "count", "non_thinking_output_tokens": "count",
    "thinking_activation": "bool",
    "answer_outcome": "outcome", "answer_role": "role", "format_compliant": "bool", "refusal": "bool",
    "latency_s": "seconds", "cost_usd": "usd",
}
MEASURES = tuple(MEASURE_KINDS)
ANSWER_MEASURES = ("answer_outcome", "answer_role")
THINKING_MEASURES = ("thinking_tokens", "thinking_activation")
RESOURCE_MEASURES = ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "non_thinking_output_tokens", "latency_s", "cost_usd")
FORMAT_MEASURES = ("format_compliant", "refusal")


_ABSENT = object()


class TelemetryError(Inconclusive):
    reason = "TELEMETRY_INVALID"


# --------------------------------------------------------------------------------------
# measurement constructors
# --------------------------------------------------------------------------------------
def safe(raw):
    """A JSON-safe, bounded copy of a reported value (non-finite floats and large structures become short strings)."""
    if isinstance(raw, float) and not math.isfinite(raw):
        return repr(raw)
    if isinstance(raw, (dict, list, tuple)):
        s = json.dumps(raw, sort_keys=True, default=repr)
        return s if len(s) <= 120 else s[:117] + "..."
    if isinstance(raw, str):
        return raw if len(raw) <= 120 else raw[:117] + "..."
    return raw


def measure(state: str, value=None, *, source: str | None = None, reason: str | None = None, reported=None, derived_from: list | None = None, **extra) -> dict:
    if state not in STATES:
        raise ValueError(f"unknown measurement state {state!r}")
    m = {"state": state, "value": value if state == OBSERVED else None, "source": source}
    if reason:
        m["reason"] = reason
    if reported is not None:
        m["reported"] = safe(reported)
    if derived_from:
        m["derived_from"] = list(derived_from)
    m.update({k: v for k, v in extra.items() if v is not None})
    return m


def observed(value, source: str, **extra) -> dict:
    return measure(OBSERVED, value, source=source, **extra)


def missing(source: str | None, reason: str) -> dict:
    return measure(MISSING, source=source, reason=reason)


def unsupported(reason: str, source: str | None = None) -> dict:
    return measure(UNSUPPORTED, source=source, reason=reason)


def read_count(container, key: str, path: str) -> dict:
    """A non-negative integer field. Absent or null is MISSING; a bool, float, string or negative number is INVALID; zero is OBSERVED."""
    if not isinstance(container, dict) or key not in container:
        return missing(path, "field_absent")
    v = container[key]
    if v is None:
        return missing(path, "field_null")
    if isinstance(v, bool) or not isinstance(v, int):
        return measure(INVALID, source=path, reason="not_an_integer", reported=v)
    if v < 0:
        return measure(INVALID, source=path, reason="negative", reported=v)
    return observed(v, path)


def read_number(container, key: str, path: str, kind_reason: str = "not_a_number") -> dict:
    """A finite non-negative real (seconds, USD)."""
    if not isinstance(container, dict) or key not in container:
        return missing(path, "field_absent")
    v = container[key]
    if v is None:
        return missing(path, "field_null")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return measure(INVALID, source=path, reason=kind_reason, reported=v)
    if isinstance(v, float) and not math.isfinite(v):
        return measure(INVALID, source=path, reason="not_finite", reported=v)
    if v < 0:
        return measure(INVALID, source=path, reason="negative", reported=v)
    return observed(float(v) if isinstance(v, float) else v, path)


def inherit(inputs: list[dict], reason_prefix: str, *, value=None, source: str, derived_from: list[str]) -> dict:
    """A derived measurement. If every input is OBSERVED the caller's value is used; otherwise it takes the worst input state (UNSUPPORTED, INVALID, INCONSISTENT, MISSING in that order),
    so an absent input can never turn into a number."""
    for st in (UNSUPPORTED, INVALID, INCONSISTENT, MISSING):
        if any(i["state"] == st for i in inputs):
            return measure(st, source=source, reason=f"{reason_prefix}_{st.lower()}", derived_from=derived_from)
    return observed(value, source, derived_from=derived_from)


# --------------------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Adapter:
    id: str
    version: str
    provider: str
    validation: str                                            # what the semantics rest on, in one sentence
    thinking: bool                                             # does this adapter establish provider-reported thinking telemetry?
    cap_finish: str                                            # the finish_reason that means "output token cap reached"
    usage: Callable[[object, object, dict], dict]              # (response, request parameters, ctx) -> measures
    corroboration: Callable[[object], bool | None]             # does the saved reply contain a thinking block? None = cannot tell


def _usage_container(resp, prefix: str, usage_key: str):
    """(usage dict or None, reason-if-not-usable, state). Distinguishes an absent response, a malformed one and an absent usage object."""
    if resp is None:
        return None, "response_absent", MISSING
    if not isinstance(resp, dict):
        return None, "response_not_an_object", INVALID
    u = resp.get(usage_key)
    if u is None:
        return None, "usage_absent", MISSING
    if not isinstance(u, dict):
        return None, "usage_not_an_object", INVALID
    return u, None, None


def _blank_usage(state: str, reason: str, resp, src: str) -> dict:
    names = ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "thinking_tokens")
    return {n: measure(state, source=src, reason=reason, reported=resp if state == INVALID else None) for n in names}


def _anthropic_usage(resp, params, ctx) -> dict:
    u, why, st = _usage_container(resp, "response", "usage")
    if u is None:
        return _blank_usage(st, why, resp, "response.usage")
    base = "response.usage."
    out = {"input_tokens": read_count(u, "input_tokens", base + "input_tokens"),
           "cache_read_tokens": read_count(u, "cache_read_input_tokens", base + "cache_read_input_tokens"),
           "cache_write_tokens": read_count(u, "cache_creation_input_tokens", base + "cache_creation_input_tokens"),
           "output_tokens": read_count(u, "output_tokens", base + "output_tokens")}
    det = u.get("output_tokens_details", _ABSENT)
    tpath = base + "output_tokens_details.thinking_tokens"
    if det is _ABSENT:
        out["thinking_tokens"] = missing(tpath, "output_tokens_details_absent")
    elif det is None:
        out["thinking_tokens"] = missing(tpath, "output_tokens_details_null")
    elif not isinstance(det, dict):
        out["thinking_tokens"] = measure(INVALID, source=base + "output_tokens_details", reason="output_tokens_details_not_an_object", reported=det)
    else:
        out["thinking_tokens"] = read_count(det, "thinking_tokens", tpath)
    # invariant 1: thinking tokens are a SUBSET of output tokens (documented in the adapter and visible in all archived Research 002 records)
    th, op = out["thinking_tokens"], out["output_tokens"]
    if th["state"] == OBSERVED and op["state"] == OBSERVED and th["value"] > op["value"]:
        for k in ("thinking_tokens", "output_tokens"):
            out[k] = measure(INCONSISTENT, source=out[k]["source"], reason="thinking_exceeds_output_tokens", reported=out[k]["value"])
    # invariant 2: max_tokens is a hard cap on thinking plus text
    cap = (params or {}).get("max_tokens") if isinstance(params, dict) else None
    op = out["output_tokens"]
    if op["state"] == OBSERVED and isinstance(cap, int) and not isinstance(cap, bool) and op["value"] > cap:
        out["output_tokens"] = measure(INCONSISTENT, source=op["source"], reason="output_exceeds_max_tokens", reported=op["value"])
    # cross-check against the usage the harness stored in the record (when it stored one)
    stored = ctx.get("stored_usage")
    if isinstance(stored, dict) and stored:
        recomputed = A.usage_from_anthropic(resp)
        links = {"promptTokenCount": ("input_tokens", "cache_read_tokens", "cache_write_tokens"), "thoughtsTokenCount": ("thinking_tokens",),
                 "candidatesTokenCount": ("output_tokens", "thinking_tokens"), "totalTokenCount": ("output_tokens", "input_tokens")}
        for k, targets in links.items():
            if k in stored and stored.get(k) != recomputed.get(k):
                ctx["problems"].append(f"stored_usage_disagrees_with_response:{k}")
                for t in targets:
                    if out[t]["state"] in (OBSERVED, MISSING):
                        out[t] = measure(INCONSISTENT, source=out[t]["source"], reason="stored_usage_disagrees_with_response", reported=out[t]["value"])
    return out


def _anthropic_thinking_block(resp) -> bool | None:
    content = resp.get("content") if isinstance(resp, dict) else None
    if not isinstance(content, list):
        return None
    return any(isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking") for b in content)


def _openai_usage(resp, params, ctx) -> dict:
    u, why, st = _usage_container(resp, "response", "usage")
    if u is None:
        out = _blank_usage(st, why, resp, "response.usage")
        out["cache_read_tokens"] = unsupported("not_defined_for_this_adapter")
        out["cache_write_tokens"] = unsupported("not_defined_for_this_adapter")
        out["thinking_tokens"] = unsupported("reasoning_telemetry_semantics_not_established")
        return out
    det = u.get("completion_tokens_details")
    if isinstance(det, dict) and det.get("reasoning_tokens") is not None:
        ctx["provider_fields"]["usage.completion_tokens_details.reasoning_tokens"] = safe(det["reasoning_tokens"])
    # completion_tokens is the billed output total and INCLUDES reasoning tokens (the v0.6 usage mapping relies on the same documented semantics)
    return {"input_tokens": read_count(u, "prompt_tokens", "response.usage.prompt_tokens"),
            "cache_read_tokens": unsupported("not_defined_for_this_adapter"), "cache_write_tokens": unsupported("not_defined_for_this_adapter"),
            "output_tokens": read_count(u, "completion_tokens", "response.usage.completion_tokens"),
            "thinking_tokens": unsupported("reasoning_telemetry_semantics_not_established")}


def _gemini_usage(resp, params, ctx) -> dict:
    u, why, st = _usage_container(resp, "response", "usageMetadata")
    unsup_think = unsupported("reasoning_telemetry_semantics_not_established")
    unsup_out = unsupported("candidate_thought_split_semantics_not_established")
    if u is None:
        out = _blank_usage(st, why, resp, "response.usageMetadata")
        out.update({"cache_read_tokens": unsupported("not_defined_for_this_adapter"), "cache_write_tokens": unsupported("not_defined_for_this_adapter"),
                    "output_tokens": unsup_out, "thinking_tokens": unsup_think})
        return out
    for k in ("candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount"):
        if u.get(k) is not None:
            ctx["provider_fields"][f"usageMetadata.{k}"] = safe(u[k])
    return {"input_tokens": read_count(u, "promptTokenCount", "response.usageMetadata.promptTokenCount"),
            "cache_read_tokens": unsupported("not_defined_for_this_adapter"), "cache_write_tokens": unsupported("not_defined_for_this_adapter"),
            "output_tokens": unsup_out, "thinking_tokens": unsup_think}


ADAPTERS = {
    "anthropic": Adapter("anthropic-messages", "1", "anthropic",
                         "validated against archived Research 002 evidence (2,560 Messages-API records, claude-haiku-5-5): usage.output_tokens_details.thinking_tokens is a subset of usage.output_tokens",
                         True, "max_tokens", _anthropic_usage, _anthropic_thinking_block),
    "openai": Adapter("openai-chat", "1", "openai",
                      "answers, format, refusal and input tokens reuse the v0.6 parser; completion_tokens is read as the billed output total; reasoning telemetry is not established in v0.7.0",
                      False, "length", _openai_usage, lambda r: None),
    "gemini": Adapter("gemini-generate", "1", "gemini",
                      "answers, format, refusal and input tokens reuse the v0.6 parser; output/thought token split and reasoning telemetry are not established in v0.7.0",
                      False, "MAX_TOKENS", _gemini_usage, lambda r: None),
}


# --------------------------------------------------------------------------------------
# answer, format, refusal
# --------------------------------------------------------------------------------------
def _answer_measures(rec: dict, resp, provider: str, ctx: dict) -> tuple[dict, dict]:
    stored_status = rec.get("status")
    if isinstance(resp, dict):
        parsed = A.PARSERS[provider](resp)
        status, letter = parsed["status"], parsed["answer"]
        ctx["finish_reason"], ctx["block_reason"] = parsed["finish_reason"], parsed["block_reason"]
    else:
        parsed = None
        status, letter = ("error" if stored_status == "error" else "malformed"), None
        ctx["finish_reason"], ctx["block_reason"] = rec.get("finish_reason"), rec.get("block_reason")
    src = "response (parsed by the v0.6 " + provider + " parser)"
    out = {}
    disagree = parsed is not None and ((stored_status is not None and stored_status != status) or (rec.get("answer") != letter and "answer" in rec))
    if parsed is None:
        reason = "response_absent" if resp is None else "response_not_an_object"
        st = MISSING if resp is None else INVALID
        for k in ("answer_outcome", "answer_role", "format_compliant", "refusal"):
            out[k] = measure(st, source="response", reason=reason, reported=resp if st == INVALID else None)
    elif status == "malformed":
        for k in ("answer_outcome", "answer_role", "format_compliant", "refusal"):
            out[k] = measure(INVALID, source="response", reason="response_structure_not_recognized")
    elif disagree:
        ctx["problems"].append("stored_answer_disagrees_with_response")
        rep = {"stored_status": stored_status, "stored_answer": rec.get("answer"), "recomputed_status": status, "recomputed_answer": letter}
        for k in ("answer_outcome", "answer_role", "format_compliant", "refusal"):
            out[k] = measure(INCONSISTENT, source=src, reason="stored_answer_disagrees_with_response", reported=rep)
    else:
        out["answer_outcome"] = observed(letter if status == "ok" else NO_ANSWER, src)
        correct, competing = rec.get("correct_letter"), rec.get("competing_letter")
        if status != "ok":
            out["answer_role"] = observed(NO_ANSWER, src, derived_from=["answer_outcome"])
        elif correct not in A.CHOICES:
            out["answer_role"] = missing("record.correct_letter", "scoring_key_absent")
        else:
            role = "correct" if letter == correct else ("competing" if competing in A.CHOICES and letter == competing else "other_incorrect")
            out["answer_role"] = observed(role, "record.correct_letter/competing_letter", derived_from=["answer_outcome"])
        out["format_compliant"] = observed(status == "ok", src)
        out["refusal"] = observed(status == "blocked", src)
    return out, {"status": status, "stored_status": stored_status}


# --------------------------------------------------------------------------------------
# latency and cost (provider-independent: they come from the harness record)
# --------------------------------------------------------------------------------------
def _latency(rec: dict, layout: str) -> dict:
    if layout == "legacy":
        return unsupported("the legacy pilot layout never recorded request latency", "record.elapsed_s")
    return read_number(rec, "elapsed_s", "record.elapsed_s")


def _derived_cost(input_m: dict, output_m: dict, cache: list[dict], prices: tuple) -> dict:
    """Cost from supplied list prices and reported usage. Cache tokens are not priced here, so a record that reports cache activity cannot be priced this way."""
    src = "derived:supplied_prices x reported usage"
    if any(c["state"] == OBSERVED and c["value"] > 0 for c in cache):
        return measure(MISSING, source=src, reason="cache_tokens_present_not_priced", derived_from=["input_tokens", "output_tokens"])
    if input_m["state"] != OBSERVED or output_m["state"] != OBSERVED:
        return inherit([input_m, output_m], "price_input", source=src, derived_from=["input_tokens", "output_tokens"])
    pin, pout = prices
    return observed(input_m["value"] * pin / 1e6 + output_m["value"] * pout / 1e6, src, derived_from=["input_tokens", "output_tokens"],
                    note="estimate from supplied list prices x reported usage; any failed-attempt charge of the harness is not included")


def _cost(rec: dict, layout: str, input_m: dict, output_m: dict, cache: list[dict], prices: tuple | None, from_usage: bool) -> tuple[dict, str | None]:
    """Estimated expenditure in USD. Always an ESTIMATE from list prices and reported usage (or the harness's recorded estimate), never a provider bill. With ``from_usage`` (and prices)
    every record is priced from its reported usage and the harness's recorded cost is ignored; otherwise a recorded cost wins and prices only fill records that carry none."""
    key = "estimated_cost_usd" if layout == "legacy" else "cost_usd"
    basis = rec.get("cost_basis") if isinstance(rec.get("cost_basis"), str) else None
    if from_usage and prices is not None:
        return _derived_cost(input_m, output_m, cache, prices), "derived_from_supplied_prices"
    if key in rec and rec[key] is not None:
        m = read_number(rec, key, "record." + key)
        if m["state"] == OBSERVED and basis and basis.startswith("upper_bound"):
            # the harness charged a worst-case bound because usage telemetry was missing: that is not a measurement of what was spent
            return measure(MISSING, source="record." + key, reason="usage_missing_harness_charged_upper_bound", reported=m["value"]), basis
        if m["state"] == OBSERVED and basis and "failed_attempt" in basis:
            m["includes_worst_case_charge_for_failed_attempts"] = True
        return m, basis
    if prices is not None:
        return _derived_cost(input_m, output_m, cache, prices), "derived_from_supplied_prices"
    return missing("record." + key, "field_absent"), basis


# --------------------------------------------------------------------------------------
# one observation
# --------------------------------------------------------------------------------------
def observe_record(rec: dict, *, layout: str = "session", evidence: dict | None = None, prices: tuple | None = None, session: str | None = None, cost_from_usage: bool = False) -> dict:
    """Turn one saved record into a behavior observation. ``layout`` is 'session' (``bleediff.hosted-answer/1``) or 'legacy' (the v0.8 pilot files). ``prices`` = (input, output)
    USD per 1M tokens, used for records that carry no cost of their own, or for every record when ``cost_from_usage`` is set. Raises TelemetryError if the record is not a JSON object or has no cell/condition/pass identity."""
    if not isinstance(rec, dict):
        raise TelemetryError("record is not a JSON object", reason="RECORD_NOT_AN_OBJECT")
    resp = rec.get("response")
    provider = rec.get("provider") or A.detect_provider(resp)
    for k in ("cell", "condition", "pass"):
        if k not in rec:
            raise TelemetryError(f"record has no {k!r}", reason="RECORD_IDENTITY_MISSING")
    adapter = ADAPTERS.get(provider)
    ctx = {"problems": [], "provider_fields": {}, "stored_usage": rec.get("usage")}
    params = ((rec.get("request") or {}).get("parameters") if isinstance(rec.get("request"), dict) else None)
    if adapter is None:
        ad = {"id": "none", "version": "1", "provider": str(provider), "validation": "no adapter for this provider: provider-dependent measurements are UNSUPPORTED"}
        meas = {k: unsupported("no_adapter_for_provider") for k in ("answer_outcome", "answer_role", "format_compliant", "refusal", "input_tokens", "cache_read_tokens",
                                                                      "cache_write_tokens", "output_tokens", "thinking_tokens")}
        ans_ctx = {"status": "unsupported", "stored_status": rec.get("status")}
        ctx["finish_reason"] = ctx["block_reason"] = None
        cap_hit = None
    else:
        ad = {"id": adapter.id, "version": adapter.version, "provider": adapter.provider, "validation": adapter.validation}
        meas, ans_ctx = _answer_measures(rec, resp, provider, ctx)
        meas.update(adapter.usage(resp, params, ctx))
        cap_hit = None if not isinstance(resp, dict) or ctx["finish_reason"] is None else ctx["finish_reason"] == adapter.cap_finish
    th = meas["thinking_tokens"]
    if adapter is not None and not adapter.thinking:
        meas["thinking_activation"] = unsupported("reasoning_telemetry_semantics_not_established")
        meas["non_thinking_output_tokens"] = unsupported("needs thinking_tokens")
    else:
        meas["thinking_activation"] = inherit([th], "thinking_tokens", value=(th["value"] or 0) > 0 if th["state"] == OBSERVED else None, source="derived: thinking_tokens > 0",
                                              derived_from=["thinking_tokens"])
        meas["non_thinking_output_tokens"] = inherit([th, meas["output_tokens"]], "input", value=(meas["output_tokens"]["value"] - th["value"]) if th["state"] == OBSERVED and meas["output_tokens"]["state"] == OBSERVED else None,
                                                     source="derived: output_tokens - thinking_tokens", derived_from=["output_tokens", "thinking_tokens"])
    meas["latency_s"] = _latency(rec, layout)
    meas["cost_usd"], cost_basis = _cost(rec, layout, meas["input_tokens"], meas["output_tokens"], [meas["cache_read_tokens"], meas["cache_write_tokens"]], prices, cost_from_usage)
    block = adapter.corroboration(resp) if adapter is not None else None
    act = meas["thinking_activation"]
    if act["state"] == OBSERVED and block is not None:
        corro = "agrees" if block == act["value"] else ("counter_positive_no_thinking_block" if act["value"] else "counter_zero_with_thinking_block")
    else:
        corro = "not_assessable"
    attempts = rec.get("attempts")
    obs = {"schema": OBSERVATION_SCHEMA,
           "identity": {"session": session, "name": rec.get("name"), "cell": rec.get("cell"), "question": rec.get("question"), "mapping": rec.get("mapping"),
                        "condition": rec.get("condition"), "pass": rec.get("pass"), "prompt_sha256": rec.get("prompt_sha256"), "provider": str(provider), "model": rec.get("model"), "model_version": rec.get("model_version")},
           "evidence": dict(evidence or {}, layout=layout, record_schema=rec.get("schema"), http_status=rec.get("http_status")),
           "adapter": ad,
           "answer_status": ans_ctx["status"], "stored_status": ans_ctx["stored_status"],
           "measures": {k: meas[k] for k in MEASURES},
           "context": {"finish_reason": ctx["finish_reason"], "block_reason": ctx["block_reason"], "output_cap_hit": cap_hit,
                       "attempts": attempts if isinstance(attempts, int) and not isinstance(attempts, bool) else None, "cost_basis": cost_basis,
                       "thinking_block_present": block, "thinking_corroboration": corro},
           "provider_fields": ctx["provider_fields"],
           "problems": ctx["problems"]}
    return obs


# --------------------------------------------------------------------------------------
# validation of an observation (used when observations are read back from a file)
# --------------------------------------------------------------------------------------
def _kind_ok(kind: str, v) -> bool:
    if kind == "count":
        return isinstance(v, int) and not isinstance(v, bool) and v >= 0
    if kind == "bool":
        return isinstance(v, bool)
    if kind in ("seconds", "usd"):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
    if kind == "outcome":
        return v in A.CHOICES or v == NO_ANSWER
    if kind == "role":
        return v in ROLES
    return False


def validate_observation(obs) -> list[str]:
    """Structural and state-semantics problems of an observation (empty list = valid). OBSERVED needs a value of the right kind; every other state must carry no value."""
    p = []
    if not isinstance(obs, dict) or obs.get("schema") != OBSERVATION_SCHEMA:
        return [f"schema is not {OBSERVATION_SCHEMA!r}"]
    for k in ("identity", "evidence", "adapter", "measures", "context"):
        if not isinstance(obs.get(k), dict):
            p.append(f"{k} is missing or not an object")
    if p:
        return p
    ident = obs["identity"]
    for k in ("cell", "condition", "pass"):
        if ident.get(k) is None:
            p.append(f"identity.{k} is missing")
    ms = obs["measures"]
    for name, kind in MEASURE_KINDS.items():
        m = ms.get(name)
        if not isinstance(m, dict) or m.get("state") not in STATES:
            p.append(f"measures.{name} has no valid state")
            continue
        if m["state"] == OBSERVED:
            if not _kind_ok(kind, m.get("value")):
                p.append(f"measures.{name} is OBSERVED with an invalid {kind} value {m.get('value')!r}")
        elif m.get("value") is not None:
            p.append(f"measures.{name} is {m['state']} but carries a value (only OBSERVED may)")
    return p


def observation_digest(obs_list: list[dict]) -> str:
    h = hashlib.sha256()
    for o in obs_list:
        h.update(json.dumps(o, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
    return h.hexdigest()
