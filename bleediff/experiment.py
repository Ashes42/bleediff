"""Author a small ANSWER_ONLY experiment from a short, human-readable spec (``bleediff experiment``). Offline; never contacts a model or an API.

The hosted commands (``bleediff anthropic|openai|gemini``) already run any prompt dataset of schema ``bleediff.hosted-prompts/1``. What was missing
was a way to *write* one: the bundled datasets are 80 hand-frozen prompts with no generator. This module is that missing step and nothing more:

    experiment.json  --(validate --write-prompts)-->  prompts.json  --(bleediff anthropic ...)-->  session  --(--replay)-->  strict + diagnostic report

It adds no measurement of its own. The prompts it writes are loaded back through ``answers.load_prompts`` (the very loader the live run uses), so a
dataset that passes ``validate`` is accepted by the runners. A spec lists questions, a clean BASELINE, named context fragments (conditions) and
explicit answer-letter mappings (letter -> semantic role). Prompts follow the layout of the frozen pilot prompts: fragment text goes on the line after the evidence.
"""

from __future__ import annotations

import json
import math
import os
import re

from . import CHOICES
from .answers import DEFAULT_BASELINE, AnswersError, load_prompts, sha256_bytes
from .errors import Inconclusive
from .anthropic_messages import ANTHROPIC
from .gemini import GEMINI
from .hosted import (DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MAX_RETRIES, MAX_CONSECUTIVE_FAILURES, MAX_OUTPUT_TOKENS_RANGE, MAX_RETRIES_RANGE, PASSES_RANGE, admission_bound,
                     prompt_tokens_upper, request_reserve)
from .openai_chat import OPENAI

SPEC_SCHEMA = "bleediff.experiment/1"
PROMPTS_SCHEMA = "bleediff.hosted-prompts/1"
ROLES = ("correct", "competing", "distractor_1", "distractor_2")
PREAMBLE = "You are a decision-support assistant. Use only the evidence provided."
INSTRUCTION = "Reply with exactly one letter: A, B, C or D.\nAnswer:"
DEMONSTRATION_BELOW = 30          # an illustrative convention for the wording of the label, NOT a power calculation
PROVIDERS = {"anthropic": ANTHROPIC, "openai": OPENAI, "gemini": GEMINI}
_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,40}$")
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")

DEFAULT_MAPPINGS = {                       # letter -> role: the correct answer sits at A, B, C and D across the four mappings
    "m1": {"A": "correct", "B": "competing", "C": "distractor_1", "D": "distractor_2"},
    "m2": {"A": "competing", "B": "correct", "C": "distractor_1", "D": "distractor_2"},
    "m3": {"A": "distractor_1", "B": "distractor_2", "C": "correct", "D": "competing"},
    "m4": {"A": "distractor_1", "B": "distractor_2", "C": "competing", "D": "correct"},
}

STARTER = {
    "schema": SPEC_SCHEMA,
    "id": "my-first-experiment",
    "description": "EDIT ME. Does a neighbouring tenant's note that names the competing answer move the model away from the correct answer, compared with a neutral note of similar character length? (Edit the question and both notes.)",
    "questions": [{
        "id": "q1",
        "question": "Which planet in the list is closest to the Sun?",
        "options": {"correct": "Mercury", "competing": "Venus", "distractor_1": "Earth", "distractor_2": "Mars"},
        "evidence": "Tenant 7 almanac: Mean distances from the Sun are 0.39 AU for Mercury, 0.72 AU for Venus, 1.00 AU for Earth and 1.52 AU for Mars, so Mercury is the closest planet to the Sun.",
    }],
    "conditions": {
        "TREATMENT_COMPETING_CLAIM": "Tenant 9 forum post: Many people say Venus is the planet closest to the Sun.",
        "CONTROL_NEUTRAL": "Tenant 9 forum post: Many people say the library opens early on Thursdays in term.",
    },
    "mappings": DEFAULT_MAPPINGS,
}


class ExperimentError(Inconclusive):
    reason = "EXPERIMENT_SPEC_INVALID"


def _nonempty(x) -> bool:
    return isinstance(x, str) and x.strip() != "" and x == x.strip()


def load_spec(path: str) -> dict:
    try:
        with open(path, "rb") as fh:
            doc = json.loads(fh.read().decode("utf-8-sig"))
    except (OSError, ValueError) as e:
        raise ExperimentError(f"cannot read the experiment spec {os.path.basename(path)!r}: {e}") from None
    problems = check_spec(doc)
    if problems:
        raise ExperimentError(f"{os.path.basename(path)} is not a valid experiment spec ({len(problems)} problem(s)):\n  - " + "\n  - ".join(problems))
    return doc


def check_spec(doc) -> list[str]:
    """Every problem found (an empty list means valid). Never raises."""
    if not isinstance(doc, dict):
        return ["the spec must be a JSON object"]
    p = []
    if doc.get("schema") != SPEC_SCHEMA:
        p.append(f"schema must be {SPEC_SCHEMA!r}")
    if not (isinstance(doc.get("id"), str) and _ID.match(doc["id"])):
        p.append("id must be lowercase letters, digits and hyphens (for example 'my-first-experiment')")
    qs = doc.get("questions")
    seen = set()
    if not isinstance(qs, list) or not qs:
        p.append("questions must be a non-empty list")
        qs = []
    for i, q in enumerate(qs, 1):
        w = f"question #{i}"
        if not isinstance(q, dict):
            p.append(f"{w} is not an object")
            continue
        if not (isinstance(q.get("id"), str) and _ID.match(q["id"])):
            p.append(f"{w}: id must be lowercase letters, digits and hyphens")
        elif q["id"] in seen:
            p.append(f"{w}: duplicate id {q['id']!r}")
        else:
            seen.add(q["id"])
        for k in ("question", "evidence"):
            if not _nonempty(q.get(k)):
                p.append(f"{w}: {k!r} must be non-empty text without leading or trailing whitespace")
        o = q.get("options")
        if not isinstance(o, dict) or set(o) != set(ROLES):
            p.append(f"{w}: options needs exactly the four keys {list(ROLES)}")
        else:
            if not all(_nonempty(o[r]) for r in ROLES):
                p.append(f"{w}: every option text must be non-empty")
            elif len({o[r].casefold() for r in ROLES}) != 4:
                p.append(f"{w}: the four option texts must be different from one another")
    conds = doc.get("conditions")
    if not isinstance(conds, dict) or not conds:
        p.append("conditions must be a non-empty object {NAME: fragment text} (the clean BASELINE is implicit and must not be listed)")
        conds = {}
    for name, frag in conds.items():
        if not _NAME.match(name):
            p.append(f"condition name {name!r} must be UPPER_SNAKE_CASE")
        elif name == DEFAULT_BASELINE:
            p.append(f"{DEFAULT_BASELINE} is the implicit clean baseline; do not list it as a condition")
        if not _nonempty(frag):
            p.append(f"condition {name!r}: the fragment must be non-empty text without leading or trailing whitespace")
    maps = doc.get("mappings")
    if not isinstance(maps, dict) or not maps:
        p.append("mappings must be a non-empty object {NAME: {letter: role}}")
        maps = {}
    for name, m in maps.items():
        if not _ID.match(str(name)):
            p.append(f"mapping name {name!r} must be lowercase letters, digits and hyphens")
        if not (isinstance(m, dict) and set(m) == set(CHOICES) and sorted(m.values()) == sorted(ROLES)):
            p.append(f"mapping {name!r} must assign each of the letters {list(CHOICES)} exactly one of the roles {list(ROLES)}")
    if isinstance(maps, dict) and len({json.dumps(m, sort_keys=True) for m in maps.values()}) != len(maps):
        p.append("two mappings are identical")
    return p


def _letters(mapping: dict) -> dict:
    return {role: letter for letter, role in mapping.items()}


def render_prompt(spec: dict, q: dict, mapping: dict, fragment: str | None) -> str:
    opts = "\n".join(f"{letter}) {q['options'][mapping[letter]]}" for letter in CHOICES)
    evidence = q["evidence"] + (f"\n{fragment}" if fragment else "")
    return f"{spec.get('preamble') or PREAMBLE}\n\nQuestion: {q['question']}\nOptions:\n{opts}\n\nEvidence:\n{evidence}\n\n{INSTRUCTION}"


def build_prompts(spec: dict) -> dict:
    """The ``bleediff.hosted-prompts/1`` dataset for a (valid) spec: every question x mapping x (BASELINE + each condition)."""
    items = []
    for q in spec["questions"]:
        for mname, m in spec["mappings"].items():
            lt = _letters(m)
            for cond, frag in [(DEFAULT_BASELINE, None), *spec["conditions"].items()]:
                prompt = render_prompt(spec, q, m, frag)
                items.append({"cell": f"{q['id']}-{mname}", "question": q["id"], "mapping": mname, "condition": cond,
                              "fragment_sha256": None if frag is None else sha256_bytes(frag.encode("utf-8")),
                              "prompt": prompt, "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                              "correct_letter": lt["correct"], "competing_letter": lt["competing"]})
    return {"schema": PROMPTS_SCHEMA, "baseline_condition": DEFAULT_BASELINE,
            "note": f"Generated by `bleediff experiment` from spec {spec['id']!r}. {len(spec['questions'])} question(s) x {len(spec['mappings'])} mapping(s) x "
                    f"{1 + len(spec['conditions'])} condition(s). Mappings of one question are not independent questions.",
            "mappings": {n: {"correct": _letters(m)["correct"], "competing": _letters(m)["competing"], "letters": m} for n, m in spec["mappings"].items()},
            "items": items}


def classification(spec: dict) -> str:
    n = len(spec["questions"])
    if n < DEMONSTRATION_BELOW:
        return (f"DEMONSTRATION. {n} question(s) is far too few to support a statistical claim: the mappings re-letter the SAME question, so they are not independent samples, "
                f"and repeated passes (--passes) are replicate checks of the same prompt. Counts show what the model did on these prompts; they do not generalise. "
                f"A valid spec shows only that the prompts are well-formed, not that any context caused any change.")
    return (f"LARGER SET ({n} questions). The {DEMONSTRATION_BELOW}-question line is an illustrative convention for this label, not a power analysis: bleediff computes no p-values or confidence intervals. "
            f"Rewording one question {n} times does not create {n} independent questions; each substantive question needs its own evidence and its own controls, "
            f"and the design should be written down before any replies are read. A valid spec still shows only that the prompts are well-formed, not that any context caused any change.")


def warnings(spec: dict) -> list[str]:
    w = []
    frags = spec["conditions"]
    if not any("CONTROL" in c for c in frags):
        w.append("no condition name contains CONTROL (a name heuristic only). Without a neutral control of similar character length, a change could come from extra text of any kind. A name such as CONTROL_NEUTRAL does not make a fragment neutral.")
    if len(frags) >= 2:
        lens = {k: len(v) for k, v in frags.items()}
        lo, hi = min(lens.values()), max(lens.values())
        if hi > 1.5 * lo:
            w.append(f"fragment character counts differ by more than 1.5x ({lens}); length can confound the comparison. Aim for similar character length (token counts may still differ).")
    for q in spec["questions"]:
        for cond, frag in frags.items():
            if "CONTROL" not in cond:
                continue
            for r in ("correct", "competing"):
                if q["options"][r].casefold() in frag.casefold():
                    w.append(f"{q['id']}/{cond}: a control fragment contains the {r} option's text, so it is probably not neutral (text match only).")
    if len({m["A"] for m in spec["mappings"].values()}) == 1 and len(spec["mappings"]) > 1:
        w.append("every mapping puts the same role at letter A; vary the letters so letter preference cannot masquerade as an effect.")
    roles_at = {(role, letter) for m in spec["mappings"].values() for letter, role in m.items() if role in ("correct", "competing")}
    if len({letter for role, letter in roles_at if role == "correct"}) < 2:
        w.append("the correct answer sits at one letter only; add a mapping that moves it.")
    return list(dict.fromkeys(w))


def _cents(x: float) -> float:
    return max(0.01, math.ceil(round(x * 100, 6)) / 100)             # the runner needs a budget above zero


def _admission_walk(uppers, provider, cap, options, pin, pout, retries: int, grow: bool):
    """Replay the hosted runner's admission rule over the requests, assuming every successful reply is billed at the provider's full output bound.
    Returns (peak budget needed to admit every request, total charged). The reserve and the per-request bound come from bleediff.hosted itself.
    grow: after a reply that used the whole bound the runner doubles the reserve of every later request. Failed attempts (5xx / network) are charged at the
    current bound and a retry is admitted only if one more full bound fits."""
    base = request_reserve(provider, cap, options)
    grown = request_reserve(provider, cap, options, base) if grow else base
    spent = need = 0.0
    for k, u in enumerate(uppers):
        w = admission_bound(u, pin, pout, base if k == 0 else grown)
        need = max(need, spent + (retries + 1) * w)
        spent += retries * w + admission_bound(u, pin, pout, base)
    return need, spent


def estimate(prompts: dict, provider_name: str, passes: int, max_output_tokens: int, price_in_per_m: float, price_out_per_m: float, max_retries: int = DEFAULT_MAX_RETRIES) -> dict:
    """Budget figures from the hosted engine's own reserve and admission functions. ESTIMATES under stated assumptions, never a provider bill."""
    provider = PROVIDERS[provider_name]
    pin, pout = price_in_per_m / 1e6, price_out_per_m / 1e6
    uppers = [prompt_tokens_upper(i["prompt"]) for i in prompts["items"]] * passes
    args = (uppers, provider, max_output_tokens, {}, pin, pout)
    fixed, ceiling = _admission_walk(*args, 0, False)
    grown = _admission_walk(*args, 0, True)[0]
    retry = _admission_walk(*args, max_retries, True)[0]
    return {"provider": provider_name, "requests": len(uppers), "input_tokens_upper": sum(uppers), "reserve_tokens": request_reserve(provider, max_output_tokens, {}),
            "max_output_tokens": max_output_tokens, "max_retries": max_retries,
            "success_ceiling_usd": ceiling, "admit_fixed_usd": fixed, "admit_grown_usd": grown, "admit_retries_usd": retry,
            "budget_fixed": _cents(fixed), "budget_grown": _cents(grown), "budget_retries": _cents(retry)}


def write_new_json(path: str, obj) -> None:
    """Create a NEW file (never overwrites). Filesystem failures become structured errors."""
    try:
        with open(path, "x", encoding="utf-8", newline="\n") as fh:
            json.dump(obj, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
    except FileExistsError:
        raise Inconclusive(f"{path!r} already exists: choose a new file (nothing is overwritten)", reason="OUTPUT_EXISTS") from None
    except OSError as e:
        raise Inconclusive(f"cannot write {path!r}: {type(e).__name__} ({e.strerror or 'filesystem error'}); the directory must exist and be writable", reason="OUTPUT_UNWRITABLE") from None


def cmd_init(a) -> int:
    d = a.dir
    if os.path.exists(d):
        raise Inconclusive(f"{d!r} already exists: choose a new directory (nothing is overwritten)", reason="OUTPUT_EXISTS")
    try:
        os.makedirs(d)
    except OSError as e:
        raise Inconclusive(f"cannot create {d!r}: {type(e).__name__} ({e.strerror or 'filesystem error'})", reason="OUTPUT_UNWRITABLE") from None
    path = os.path.join(d, "experiment.json")
    write_new_json(path, STARTER)
    print(f"wrote {path}  (a valid starter: 1 question x 4 mappings x BASELINE + 2 conditions = 12 prompts; no model, no key, no network)")
    print("\nNext:\n  1. edit it (question, options, evidence, the two fragments; give the neutral control a similar character length to the treatment; token counts may differ)")
    print(f"  2. python -m bleediff experiment validate {path} --write-prompts {os.path.join(d, 'prompts.json')} --provider anthropic --price-in 1 --price-out 5   (example prices: use current list prices)")
    print("  3. read docs/FIRST_EXPERIMENT.md for the live and replay commands")
    return 0


def _check_args(a) -> bool:
    """Validate every numeric argument BEFORE anything is written; True when a budget estimate was requested."""
    for name, v, (lo, hi) in (("--passes", a.passes, PASSES_RANGE), ("--max-output-tokens", a.max_output_tokens, MAX_OUTPUT_TOKENS_RANGE), ("--max-retries", a.max_retries, MAX_RETRIES_RANGE)):
        if v is not None and not lo <= v <= hi:
            raise Inconclusive(f"{name} must be an integer from {lo} to {hi} (the live runner's range)", reason="BAD_ARGS")
    asked = (a.provider, a.price_in, a.price_out)
    if all(x is None for x in asked):
        return False
    if any(x is None for x in asked):
        raise Inconclusive("a budget estimate needs --provider, --price-in and --price-out together", reason="BAD_ARGS")
    for name, v in (("--price-in", a.price_in), ("--price-out", a.price_out)):
        if not (math.isfinite(v) and v >= 0):
            raise Inconclusive(f"{name} must be a non-negative number", reason="BAD_ARGS")
    return True


def _print_estimate(e: dict, price_in: float, price_out: float, n_prompts: int, passes: int) -> None:
    extra = "" if e["reserve_tokens"] == e["max_output_tokens"] else " (the cap plus the provider's thinking allowance)"
    print(f"\nBudget estimate for {e['provider']} under the live runner's own rules: {e['requests']} requests ({n_prompts} prompts x {passes} pass(es)); prices ${price_in}/M in, ${price_out}/M out; "
          f"output cap {e['max_output_tokens']}; reserve {e['reserve_tokens']} output tokens per request{extra}; input bounded at 1 token per prompt byte + 64 (far above real usage).")
    print(f"  1. One successful attempt per request, each billed at its full bound: at most ${e['success_ceiling_usd']:.4f}. A ceiling under these assumptions; answer-only replies usually cost much less.")
    print(f"  2. Budget needed for the guard to ADMIT every request (reserve unchanged, no retries): ${e['budget_fixed']:.2f}.")
    print(f"  3. Same, if some reply uses more than half the reserve (the guard then doubles the reserve of later requests): ${e['budget_grown']:.2f}.")
    print(f"  4. Same, if every request needed {e['max_retries']} retries (--max-retries; failed 5xx/network attempts are charged at the bound and a retry must fit as one more full bound): ${e['budget_retries']:.2f}. "
          f"An upper planning figure, not an expectation: the runner also stops after {MAX_CONSECUTIVE_FAILURES} consecutive failed requests.")
    print("A --budget below these can stop the run early (status 'partial', BUDGET_EXHAUSTED) with fewer records. They are estimates from the prices you supplied, not a provider billing limit or a guarantee that a run completes. "
          "Check current prices with your provider first." + (" These Gemini figures assume no --thinking-budget." if e["provider"] == "gemini" else ""))


def cmd_validate(a) -> int:
    want_estimate = _check_args(a)
    spec = load_spec(a.spec)
    prompts = build_prompts(spec)
    nq, nm, nc = len(spec["questions"]), len(spec["mappings"]), 1 + len(spec["conditions"])
    print(f"spec {spec['id']!r}: VALID ({nq} question(s) x {nm} mapping(s) x {nc} condition(s) [{', '.join([DEFAULT_BASELINE, *spec['conditions']])}] = {len(prompts['items'])} prompts)")
    print("\nAnswer-letter mappings (letter = role); the correct and competing letters are what strict scoring compares against:")
    for n, m in spec["mappings"].items():
        print(f"  {n}: " + "  ".join(f"{letter}={role}" for letter, role in m.items()))
    for line in warnings(spec):
        print(f"WARNING: {line}")
    print("These checks are heuristics: condition names are matched by text, and lengths are compared in characters (similar character length; token counts may differ, so this is not tokenizer parity). "
          "No warning does not prove that a control is neutral or that the design is sound.")
    if a.write_prompts:
        if os.path.exists(a.write_prompts):
            raise Inconclusive(f"{a.write_prompts!r} already exists: choose a new file (nothing is overwritten)", reason="OUTPUT_EXISTS")
        write_new_json(a.write_prompts, prompts)
        try:
            loaded = load_prompts(a.write_prompts)           # the loader the live runners use
        except AnswersError:
            os.remove(a.write_prompts)
            raise
        print(f"\nwrote {a.write_prompts} and re-validated it with the runner's own loader ({len(loaded['items'])} items, sha256 {loaded['sha256'][:12]}...)")
    if want_estimate:
        cap = a.max_output_tokens if a.max_output_tokens is not None else DEFAULT_MAX_OUTPUT_TOKENS[a.provider]
        e = estimate(prompts, a.provider, a.passes, cap, a.price_in, a.price_out, a.max_retries)
        _print_estimate(e, a.price_in, a.price_out, len(prompts["items"]), a.passes)
    else:
        print("\n(add --provider, --price-in and --price-out for a budget estimate)")
    print("\n" + classification(spec))
    print("Strict scoring: a reply counts only if it is exactly one letter A-D; refusals and prose are non-answers, never 'incorrect'. "
          "FLR-v2 recovery of a final letter is a separate, labelled diagnostic printed beside the strict counts.")
    return 0


def add_parser(sub) -> None:
    e = sub.add_parser("experiment", help="author and validate a small ANSWER_ONLY experiment offline (no model, no key, no network); runs use `bleediff anthropic|openai|gemini`")
    es = e.add_subparsers(dest="experiment_cmd", required=True)
    i = es.add_parser("init", help="write a valid starter experiment.json into a NEW directory")
    i.add_argument("dir")
    i.set_defaults(fn=cmd_init)
    v = es.add_parser("validate", help="validate an experiment spec; optionally write the prompt dataset and estimate the budget")
    v.add_argument("spec")
    v.add_argument("--write-prompts", default=None, metavar="PROMPTS.json", help="write the bleediff.hosted-prompts/1 dataset to this NEW file (its directory must exist)")
    v.add_argument("--provider", choices=sorted(PROVIDERS), default=None, help="provider whose reserve rules the budget estimate follows (needs both prices)")
    v.add_argument("--passes", type=int, default=2, help=f"repeats per prompt, for the estimate ({PASSES_RANGE[0]}-{PASSES_RANGE[1]}, default 2)")
    v.add_argument("--max-output-tokens", type=int, default=None, help=f"per-request output cap ({MAX_OUTPUT_TOKENS_RANGE[0]}-{MAX_OUTPUT_TOKENS_RANGE[1]}); default: the runner default for the provider {DEFAULT_MAX_OUTPUT_TOKENS}")
    v.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help=f"retries per request assumed by figure 4 ({MAX_RETRIES_RANGE[0]}-{MAX_RETRIES_RANGE[1]}, default {DEFAULT_MAX_RETRIES}, as in the runner)")
    v.add_argument("--price-in", type=float, default=None, help="input USD per 1M tokens (no built-in prices; use your provider's current list price)")
    v.add_argument("--price-out", type=float, default=None, help="output USD per 1M tokens")
    v.set_defaults(fn=cmd_validate)
