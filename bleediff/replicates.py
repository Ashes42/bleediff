"""neutral-prose-replicates/1: deterministically diversified neutral controls for ONE scenario (v0.1.5).

WHY THIS EXISTS
---------------
The frozen ``neutral-prose/1`` generator (v0.1.2; source pinned by tests/data/frozen_neutral_prose_f2299f0.json and by
the protocol's frozen-source hashes) is deterministic in its *inputs*: its seed is a hash of
(algorithm, scenario, offset, treatment fragment, tokenizer fingerprint, lexicon) and it returns the first passing
candidate. It therefore yields exactly ONE control per input tuple and has no seed, replicate or domain parameter.
The robustness study needs several controls for one scenario. Passing a synthetic fingerprint would put a false
identity into a provenance field and editing the generator would break its pin, so this separately versioned module
IMPORTS the frozen internals (RNG, lexicon, slot builder, solver, grammar recogniser, lexical/skeleton checks,
token counter, prompt-parity check, verifier) UNCHANGED and supplies only a different candidate-stream seed, a forced
semantic domain and an extra, fixed rubric.

WHAT IS FIXED IN ADVANCE (everything here is defined before any candidate is generated)
-------------------------------------------------------------------------------------
* 12 replicate indices k = 0..11. Domain of replicate k = ``sorted(domain ids)[k mod 8]``: bakery, festival,
  greenhouse, library, orchestra, seaside, trail, workshop for k = 0..7; bakery, festival, greenhouse, library again
  for k = 8..11. Not chosen by outcome.
* Seed of replicate k = first 16 hex digits of sha256 over a canonical JSON of (replicate algorithm, scenario, k,
  domain, offset, unit, treatment-fragment hash, tokenizer fingerprint, lexicon hash, rubric hash).
* Candidate stream: the frozen draw procedure with the domain forced instead of drawn. A candidate must pass EVERY
  frozen structural requirement (exact bytes, words, fragment tokens, whole request-3 prompt tokens, insertion
  boundary, punctuation skeleton, lexical bans, grammar). Nothing is relaxed.
* A structurally valid candidate is then screened by RUBRIC/1 (below). A rejected candidate is recorded in full
  (text + reasons) and the NEXT candidate of the same stream is tried. A replicate that exhausts its draw budget is
  FAILED and preserved as such: there is no substitution of another domain, seed or replicate.
* Replicates are processed in order k = 0..11; the distinctness rules compare against the previously ACCEPTED
  replicates and the historical control C0 only.
* No completion probability is ever consulted. This module cannot call /completion.

WHAT THIS DOES NOT CLAIM
------------------------
Different seeds and domains do NOT make the controls statistically independent draws from some population of
neutral text. They are deterministically diversified samples from one constrained generator.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys

from . import UNKNOWN, __version__
from . import control as C
from .arena import Fixtures, Scenario, load_fixtures
from .control import ControlMatchError, Rng

REPLICATE_ALGORITHM = "neutral-prose-replicates/1"
MANIFEST_SCHEMA_V3 = "bleediff.controls/3"
N_REPLICATES = 12
MIN_QUALIFYING = 10                     # fewer accepted controls => the later study must report INSUFFICIENT
DEFAULT_MAX_DRAWS = C.DEFAULT_MAX_DRAWS  # per replicate, including every rejected draw
JACCARD_MAX = 0.5                       # content-word Jaccard above this vs an accepted control / C0 is a rejection
STEM_MIN_LEN = 4
DOMAINS = tuple(sorted(c["id"] for c in C.CLUSTERS))


def domain_for(k: int) -> str:
    if not 0 <= k < N_REPLICATES:
        raise ValueError("replicate index out of range")
    return DOMAINS[k % len(DOMAINS)]


# --------------------------------------------------------------------------------------
# RUBRIC/1 -- fixed before any candidate exists
# --------------------------------------------------------------------------------------
#: Fixed stopword policy: these ordinary function words are ALLOWED to overlap with the task text.
STOPWORDS = tuple(sorted(set("""
a an the and or but nor so yet of to in on at by for from with without into onto upon over under above below beside
near along across through during before after while until as than that this these those it its their his her our
your they them he she we you i is are was were be been being am has have had do does did will would shall should can
could may might not no each some many several any all both either neither other such only also very more most same
own too just then there here when where which who whom whose what how why if""".split())))

#: Task-independent semantic-field words prohibited in every control: warning / failure / security / audit /
#: infrastructure vocabulary and rating words. Chosen by meaning before any generation; includes the idiom words
#: ("flag(s)") that made one earlier control a potential semantic confound.
PROHIBITED_FIELD_WORDS = tuple(sorted(set("""
warning warnings warn warned alert alerts alarm alarms flag flags flagged danger dangerous hazard hazards risk risks
risky threat threats severity failure failures failed fail fails error errors fault faults breach breaches violation
violations vulnerable vulnerability exploit exploits attack attacks secure security insecure unsafe safe safety
critical urgent severe serious emergency incident incidents outage outages crash crashes broken defect defects bug
bugs patch patches audit audits compliance compliant baseline rating ratings score scores grade grades level levels
priority priorities escalate escalated production deploy deployment config configuration setting settings enable
enabled disable disabled debug logging log logs logged module modules server servers network networks password
passwords token tokens cipher ciphers encryption encrypted certificate certificates admin administrator subnet
endpoint endpoints listener listeners session sessions timeout timeouts payment payments cardholder card cards data
record records scan scans scanner test tested testing tests report reports""".split())))

RUBRIC = {
    "rubric_id": "rubric/1",
    "task_vocabulary_policy": (
        "TASK content words = every alphabetic word (lower case) in the question, the four options, the three "
        "authorized contexts and the whole adjacent slot (treatment fragment AND the unread trailing sentence) of the "
        "scenario, minus STOPWORDS. A candidate may not contain any TASK content word (R1) and may not contain a "
        "word of >= STEM_MIN_LEN letters that is a strict prefix or extension of a TASK content word of >= "
        "STEM_MIN_LEN letters (R2). STOPWORDS may overlap freely: ordinary overlapping words are NOT prohibited."),
    "stopwords": list(STOPWORDS),
    "prohibited_field_words": list(PROHIBITED_FIELD_WORDS),
    "stem_min_len": STEM_MIN_LEN,
    "numeral_policy": "no numeral token (clock time, quantity) may contain a digit run that occurs in the scenario's "
                      "prompt material (R4)",
    "distinctness_policy": f"text must differ from every accepted control and from C0 (R5); content-word Jaccard "
                           f"with each of them must be <= {JACCARD_MAX} (R6)",
    "applies_to": "words after the fixed 'Tenant <n>' prefix",
    "rejection_codes": {
        "R1_TASK_WORD": "word occurs in the task text and is not a stopword",
        "R2_TASK_STEM": "word is a strict prefix/extension of a task content word (>= 4 letters)",
        "R3_FIELD_WORD": "word is in the prohibited semantic-field list",
        "R4_NUMERAL": "numeral shares a digit run with the scenario's prompt material",
        "R5_DUPLICATE": "text identical to an accepted control or C0",
        "R6_OVERLAP": "content-word Jaccard with an accepted control or C0 above the maximum",
    },
    "manual_review": "After generation and BEFORE any inference a human reviews the accepted set for contradictory "
                     "adjectives, implausible quantities, idioms and unintended implications. A manual rejection is "
                     "recorded with a reason and the replicate advances to the next candidate of the same stream; the "
                     "review may not look at any model output.",
}
RUBRIC_SHA256 = hashlib.sha256(json.dumps({**RUBRIC, "jaccard_max": JACCARD_MAX, "n_replicates": N_REPLICATES,
                                           "min_qualifying": MIN_QUALIFYING}, sort_keys=True).encode()).hexdigest()

_ALPHA = re.compile(r"[a-z]+")
_DIGITS = re.compile(r"\d+")


def task_vocabulary(sc: Scenario) -> tuple[frozenset, frozenset]:
    """(task content words, digit runs): everything the scenario shows the model, minus stopwords."""
    blob = " ".join([sc.question, *sc.options.values(), *sc.authorized, sc.adjacent]).lower()
    return frozenset(_ALPHA.findall(blob)) - frozenset(STOPWORDS), frozenset(_DIGITS.findall(blob))


def content_words(text: str, prefix_len: int = 2) -> frozenset:
    words = text.split()[prefix_len:]
    return frozenset(w for w in (C._norm(x) for x in words) if w.isalpha() and w not in STOPWORDS)


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def screen(text: str, sc: Scenario, *, prefix_len: int = 2, accepted_texts: tuple = ()) -> list[tuple[str, str]]:
    """RUBRIC/1. Returns [(code, detail)]; empty = passes. Pure function of text, scenario and accepted texts."""
    task_words, task_digits = task_vocabulary(sc)
    problems: list[tuple[str, str]] = []
    for raw in text.split()[prefix_len:]:
        core = C._norm(raw)
        for run in _DIGITS.findall(core):
            if run in task_digits:
                problems.append(("R4_NUMERAL", f"digit run {run!r} occurs in the scenario text"))
        for w in _ALPHA.findall(core):
            if w in task_words:
                problems.append(("R1_TASK_WORD", w))
            elif len(w) >= STEM_MIN_LEN and any(len(t) >= STEM_MIN_LEN and t != w and (t.startswith(w) or w.startswith(t))
                                                for t in task_words):
                problems.append(("R2_TASK_STEM", w))
            if w in PROHIBITED_FIELD_WORDS:
                problems.append(("R3_FIELD_WORD", w))
    mine = content_words(text, prefix_len)
    for other in accepted_texts:
        if other == text:
            problems.append(("R5_DUPLICATE", "identical to an accepted control or C0"))
        else:
            j = jaccard(mine, content_words(other, prefix_len))
            if j > JACCARD_MAX:
                problems.append(("R6_OVERLAP", f"content-word Jaccard {j:.3f} > {JACCARD_MAX}"))
    seen, uniq = set(), []
    for p in problems:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


# --------------------------------------------------------------------------------------
# seeds and identities
# --------------------------------------------------------------------------------------
def replicate_seed_material(*, scenario_id: str, k: int, offset: int, unit: str, treatment_sha256: str,
                            fingerprint: str) -> dict:
    return {"algorithm": REPLICATE_ALGORITHM, "scenario_id": scenario_id, "replicate": k, "domain": domain_for(k),
            "offset": offset, "unit": unit, "treatment_fragment_sha256": treatment_sha256,
            "tokenizer_fingerprint": fingerprint, "lexicon_sha256": C.LEXICON_SHA256, "rubric_sha256": RUBRIC_SHA256}


def replicate_seed_hex(**kw) -> str:
    return hashlib.sha256(json.dumps(replicate_seed_material(**kw), sort_keys=True).encode()).hexdigest()


def replicate_control_id(**kw) -> str:
    return replicate_seed_hex(**kw)[:32]


# --------------------------------------------------------------------------------------
# candidate stream (a faithful copy of the frozen draw loop with seed and domain made explicit)
# --------------------------------------------------------------------------------------
def search_stream(sc: Scenario, offset: int, unit: str, tokenize, *, seed: int, forced_domain: str | None,
                  screen_fn=None, max_draws: int = DEFAULT_MAX_DRAWS, fingerprint: str | None = None) -> dict:
    """Run one candidate stream. ``forced_domain=None`` and ``screen_fn=None`` reproduce the frozen draw order exactly
    (used only to anchor this copy against ``control.generate_control``). Returns a result dict; exhaustion is a
    PRESERVED failure, not an exception. Tokenizer problems that make exact matching impossible still raise."""
    if unit != "words":
        raise ControlMatchError("neutral-prose/1 supports unit='words' only", reason="CONTROL_UNSUPPORTED_UNIT")
    treat = C.treatment_fragment(sc, offset, unit)
    sk = C.parse_skeleton(treat)
    vocab, runs = C.scenario_vocabulary(sc)
    counter = C.TokenCounter(tokenize) if tokenize is not None else None
    fp = fingerprint or (C.tokenizer_fingerprint(tokenize) if tokenize is not None else UNKNOWN)
    want_tokens = counter.exact("\n" + treat) if counter else None
    if counter is not None and counter.additive(treat.split()) != want_tokens:
        raise ControlMatchError("this tokenizer is not additive over words for the treatment fragment; exact token "
                                "matching cannot be guaranteed", reason="CONTROL_TOKENIZER_NOT_ADDITIVE")
    pre_bytes = sum(len(w) for w in sk.prefix)
    target_bytes = sk.bytes - (sk.words - 1) - pre_bytes
    target_tokens = (want_tokens - counter.additive(list(sk.prefix))) if counter else 0
    rng = Rng(seed)
    rejected = {"structure": 0, "infeasible": 0, "lexical": 0, "grammar": 0, "tokens": 0, "prompt": 0}
    rubric_log: list[dict] = []
    sig_cache: dict = {}
    for draw in range(max_draws):
        cl = C._CLUSTER_BY_ID[forced_domain] if forced_domain else rng.choice(C.CLUSTERS)
        specs = C.build_slots(rng, cl, sk)
        if specs is None:
            rejected["structure"] += 1
            continue
        groups = []
        for spec in specs:
            key = (cl["id"], spec[4], spec[1], spec[2], spec[3])
            if key not in sig_cache:
                sig_cache[key] = C._signature_groups(spec, counter, vocab, runs)
            groups.append(sig_cache[key])
        solved = C.solve_slots(rng, groups, target_bytes, target_tokens)
        if solved is None:
            rejected["infeasible"] += 1
            continue
        words = list(sk.prefix) + [w for part in solved for w in part]
        text = " ".join(words)
        if len(text.encode()) != sk.bytes or len(words) != sk.words:
            raise AssertionError("solver returned a candidate that violates the byte/word budget")
        if C.lexical_problems(words, sk, vocab, runs) or C.skeleton_problems(words, sk):
            rejected["lexical"] += 1
            continue
        if C.recognize_fragment(words, sk, cl["id"]):
            rejected["grammar"] += 1
            continue
        checks = ["bytes", "words", "skeleton", "lexical", "grammar"]
        parity = "PENDING"
        counts = {"bytes": (sk.bytes, len(text.encode())), "words": (sk.words, len(words))}
        if counter is not None:
            got = counter.exact("\n" + text)
            if got != want_tokens:
                rejected["tokens"] += 1
                continue
            pp = C.prompt_parity(sc, offset, unit, text, tokenize)
            if not (pp["requests_1_2_identical"] and pp["geometry_identical"] and pp["surrounding_prompt_identical"]
                    and pp["n_requests"] == (3, 3) and pp["prompt_tokens"][0] == pp["prompt_tokens"][1]):
                rejected["prompt"] += 1
                continue
            parity = "VERIFIED"
            checks += ["fragment_tokens", "prompt_tokens", "geometry", "surrounding_prompt"]
            counts["fragment_tokens"] = (want_tokens, got)
            counts["prompt_tokens"] = pp["prompt_tokens"]
        if screen_fn is not None:
            problems = screen_fn(text)
            if problems:                                  # structurally valid but rubric-rejected: preserved in full
                rubric_log.append({"draw": draw, "text": text, "reasons": [list(p) for p in problems]})
                continue
            checks.append("rubric/1")
        return {"status": "ACCEPTED", "draw": draw, "cluster": cl["id"], "text": text, "counts": counts,
                "token_parity": parity, "checks": checks, "skeleton": sk.to_json(), "fingerprint": fp,
                "treatment_sha256": C.fragment_sha256(treat), "structural_rejections": rejected,
                "rubric_rejections": rubric_log, "draws_used": draw + 1}
    return {"status": "FAILED", "reason": "REPLICATE_FAILED_NO_FEASIBLE_CANDIDATE", "draws_used": max_draws,
            "structural_rejections": rejected, "rubric_rejections": rubric_log, "fingerprint": fp,
            "treatment_sha256": C.fragment_sha256(treat),
            "note": f"no candidate satisfied every requirement within {max_draws} draws; nothing was relaxed and no "
                    f"other domain was substituted"}


def generate_replicate(sc: Scenario, offset: int, k: int, tokenize=None, *, accepted_texts: tuple = (),
                       max_draws: int = DEFAULT_MAX_DRAWS, fingerprint: str | None = None, unit: str = "words") -> dict:
    treat = C.treatment_fragment(sc, offset, unit)
    fp = fingerprint or (C.tokenizer_fingerprint(tokenize) if tokenize is not None else UNKNOWN)
    t_sha = C.fragment_sha256(treat)
    ident = dict(scenario_id=sc.id, k=k, offset=offset, unit=unit, treatment_sha256=t_sha, fingerprint=fp)
    seed_hex = replicate_seed_hex(**ident)
    res = search_stream(sc, offset, unit, tokenize, seed=int(seed_hex[:16], 16), forced_domain=domain_for(k),
                        screen_fn=lambda text: screen(text, sc, accepted_texts=accepted_texts),
                        max_draws=max_draws, fingerprint=fp)
    out = {"replicate": k, "domain": domain_for(k), "seed_hex16": seed_hex[:16], "control_id": seed_hex[:32],
           "status": res["status"], "draws_used": res["draws_used"],
           "structural_rejections": res["structural_rejections"], "rubric_rejections": res["rubric_rejections"]}
    if res["status"] == "ACCEPTED":
        text = res["text"]
        out["control"] = {
            "scenario_id": sc.id, "algorithm": REPLICATE_ALGORITHM, "control_id": seed_hex[:32],
            "enumeration_index": res["draw"], "cluster": res["cluster"], "offset": offset, "unit": unit,
            "fragment_text": text, "fragment_sha256": C.fragment_sha256(text), "treatment_fragment_sha256": t_sha,
            "skeleton": res["skeleton"], "counts": {k_: list(v) for k_, v in res["counts"].items()},
            "tokenizer_fingerprint": fp, "lexicon_sha256": C.LEXICON_SHA256, "token_parity": res["token_parity"],
            "checks": res["checks"]}
    else:
        out["reason"] = res["reason"]
        out["note"] = res["note"]
    return out


def build_replicate_manifest(fixtures: Fixtures, scenario_id: str, offset: int, tokenize=None, *, c0_entry: dict,
                             c0_manifest_sha256: str, protocol_id: str = UNKNOWN, unit: str = "words",
                             max_draws: int = DEFAULT_MAX_DRAWS) -> dict:
    sc = {s.id: s for s in fixtures.scenarios}[scenario_id]
    fp = C.tokenizer_fingerprint(tokenize) if tokenize is not None else UNKNOWN
    treat = C.treatment_fragment(sc, offset, unit)
    accepted = [c0_entry["fragment_text"]]
    reps = []
    for k in range(N_REPLICATES):
        r = generate_replicate(sc, offset, k, tokenize, accepted_texts=tuple(accepted), max_draws=max_draws,
                               fingerprint=fp, unit=unit)
        if r["status"] == "ACCEPTED":
            accepted.append(r["control"]["fragment_text"])
        reps.append(r)
    n_ok = sum(r["status"] == "ACCEPTED" for r in reps)
    return {
        "schema": MANIFEST_SCHEMA_V3, "algorithm": REPLICATE_ALGORITHM, "base_algorithm": C.ALGORITHM,
        "bleediff_version": __version__,
        "generator": {"software_version": __version__, "control_algorithm": C.ALGORITHM,
                      "replicate_algorithm": REPLICATE_ALGORITHM, "protocol_id": protocol_id},
        "lexicon_sha256": C.LEXICON_SHA256, "rubric_sha256": RUBRIC_SHA256, "tokenizer_fingerprint": fp,
        "fixtures_sha256": fixtures.sha256, "scenario_id": scenario_id, "offset": offset, "unit": unit,
        "treatment_fragment_sha256": C.fragment_sha256(treat),
        "c0": {"role": "historical control, OUTSIDE the primary 12-control analysis", "control_id": c0_entry["control_id"],
               "fragment_sha256": c0_entry["fragment_sha256"], "source_manifest_sha256": c0_manifest_sha256},
        "allocation": {"rule": "domain(k) = sorted(domain ids)[k mod 8]", "domains": list(DOMAINS), "n_replicates": N_REPLICATES},
        "seed_rule": "sha256(canonical JSON of replicate_seed_material)[:16] as the SplitMix64 seed",
        "selection_rule": "first candidate of the replicate's own stream that passes every frozen structural check and "
                          "RUBRIC/1; rejected candidates are recorded; no model output is consulted",
        "replicates": reps,
        "summary": {"n_requested": N_REPLICATES, "n_accepted": n_ok, "n_failed": N_REPLICATES - n_ok,
                    "min_qualifying": MIN_QUALIFYING, "qualifies_for_analysis": n_ok >= MIN_QUALIFYING,
                    "if_not_qualifying": "the later study must report INSUFFICIENT"},
        "independence_statement": "These are deterministically diversified samples from a constrained text generator. "
                                  "Distinct seeds and domains do not make them statistically independent draws.",
    }


def verify_replicate_manifest(manifest: dict, fixtures: Fixtures, tokenize=None, *, c0_entry: dict | None = None) -> list[str]:
    """Static (``tokenize=None``) or live verification. Empty list = consistent."""
    probs: list[str] = []
    if manifest.get("schema") != MANIFEST_SCHEMA_V3 or manifest.get("algorithm") != REPLICATE_ALGORITHM:
        return ["schema/algorithm mismatch"]
    if manifest.get("fixtures_sha256") != fixtures.sha256:
        return ["controls were built for different fixtures"]
    if manifest.get("lexicon_sha256") != C.LEXICON_SHA256:
        probs.append("lexicon changed since the controls were selected")
    if manifest.get("rubric_sha256") != RUBRIC_SHA256:
        probs.append("rubric changed since the controls were selected")
    sc = {s.id: s for s in fixtures.scenarios}.get(manifest.get("scenario_id"))
    if sc is None:
        return probs + ["unknown scenario"]
    offset, unit = manifest["offset"], manifest.get("unit", "words")
    treat = C.treatment_fragment(sc, offset, unit)
    t_sha = C.fragment_sha256(treat)
    if manifest.get("treatment_fragment_sha256") != t_sha:
        return probs + ["treatment fragment differs from the one the controls were built for"]
    reps = manifest.get("replicates", [])
    if [r.get("replicate") for r in reps] != list(range(N_REPLICATES)):
        return probs + ["replicates must be exactly indices 0..11 in order"]
    fp = manifest.get("tokenizer_fingerprint")
    accepted = [c0_entry["fragment_text"]] if c0_entry else []
    n_ok = 0
    for r in reps:
        k = r["replicate"]
        tag = f"replicate {k}"
        if r.get("domain") != domain_for(k):
            probs.append(f"{tag}: domain {r.get('domain')!r} is not the preregistered {domain_for(k)!r}")
        seed = replicate_seed_hex(scenario_id=sc.id, k=k, offset=offset, unit=unit, treatment_sha256=t_sha, fingerprint=fp)
        if r.get("seed_hex16") != seed[:16] or r.get("control_id") != seed[:32]:
            probs.append(f"{tag}: seed/control_id do not follow the preregistered derivation")
        for rej in r.get("rubric_rejections", []):
            again = screen(rej["text"], sc, accepted_texts=tuple(accepted))
            if not again:
                probs.append(f"{tag}: a recorded rubric rejection does not reproduce (draw {rej['draw']})")
        if r.get("status") == "ACCEPTED":
            n_ok += 1
            e = r["control"]
            if e["cluster"] != domain_for(k):
                probs.append(f"{tag}: cluster {e['cluster']!r} is not the forced domain")
            if screen(e["fragment_text"], sc, accepted_texts=tuple(accepted)):
                probs.append(f"{tag}: accepted control fails RUBRIC/1")
            sub = {"lexicon_sha256": manifest["lexicon_sha256"], "tokenizer_fingerprint": fp}
            entry = {**e, "enumeration_index": e["enumeration_index"]}
            if tokenize is not None:
                probs += [f"{tag}: {p}" for p in C.verify_control(sc, entry, sub, offset, unit, tokenize)]
            else:
                if C.fragment_sha256(e["fragment_text"]) != e["fragment_sha256"]:
                    probs.append(f"{tag}: stored text does not match its hash")
                sk = C.parse_skeleton(treat)
                words = e["fragment_text"].split()
                vocab, runs = C.scenario_vocabulary(sc)
                for p in (C.lexical_problems(words, sk, vocab, runs) + C.skeleton_problems(words, sk)
                          + C.recognize_fragment(words, sk, e["cluster"])):
                    probs.append(f"{tag}: {p}")
                if len(e["fragment_text"].encode()) != sk.bytes or len(words) != sk.words:
                    probs.append(f"{tag}: byte/word parity broken")
            accepted.append(e["fragment_text"])
        elif r.get("status") == "FAILED":
            if not r.get("reason"):
                probs.append(f"{tag}: a failed replicate must record its reason")
        else:
            probs.append(f"{tag}: unknown status")
    s = manifest.get("summary", {})
    if s.get("n_accepted") != n_ok or s.get("n_failed") != N_REPLICATES - n_ok or \
            s.get("qualifies_for_analysis") != (n_ok >= MIN_QUALIFYING):
        probs.append("summary is inconsistent with the replicate records")
    return probs


def anchor_reproduces_frozen(sc: Scenario, offset: int, tokenize, c0_entry: dict, *,
                             max_draws: int = DEFAULT_MAX_DRAWS, unit: str = "words") -> dict:
    """Phase 1B gate: this module's draw loop, run in FROZEN-compatible mode (frozen seed, drawn domain, no rubric),
    must reproduce the historical control C0 exactly under the real tokenizer. If it does not, the copy of the draw
    loop has drifted from the frozen generator and no replicate may be trusted."""
    fp = C.tokenizer_fingerprint(tokenize)
    t_sha = C.fragment_sha256(C.treatment_fragment(sc, offset, unit))
    cid = C.control_id_for(scenario_id=sc.id, offset=offset, unit=unit, treatment_sha256=t_sha, fingerprint=fp)
    res = search_stream(sc, offset, unit, tokenize, seed=int(cid[:16], 16), forced_domain=None, screen_fn=None,
                        max_draws=max_draws, fingerprint=fp)
    ok = (res["status"] == "ACCEPTED" and cid == c0_entry["control_id"] and res["text"] == c0_entry["fragment_text"]
          and res["draw"] == c0_entry["enumeration_index"] and res["cluster"] == c0_entry["cluster"])
    return {"reproduces_c0": ok, "control_id": cid, "c0_control_id": c0_entry["control_id"],
            "draw": res.get("draw"), "c0_draw": c0_entry["enumeration_index"]}


def write_manifest(path: str, manifest: dict) -> None:
    if os.path.exists(path):
        raise ControlMatchError(f"refusing to overwrite existing {path!r}", reason="CONTROL_MANIFEST_EXISTS")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Tokenize-only CLI. NOT used in the Phase 1A checkpoint (no final controls are generated there)."""
    import argparse

    from .errors import Inconclusive

    p = argparse.ArgumentParser(prog="python -m bleediff.replicates", description="neutral-prose-replicates/1 (tokenize only)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "verify", "anchor"):
        s = sub.add_parser(name)
        s.add_argument("--fixtures", required=True)
        s.add_argument("--scenario", required=True)
        s.add_argument("--offset", type=int, required=True)
        s.add_argument("--c0-manifest", required=True, help="the frozen controls manifest holding C0")
        s.add_argument("--server", required=True, help="used for /tokenize ONLY (route it through a guard that refuses /completion)")
        s.add_argument("--timeout", type=float, default=60.0)
        s.add_argument("--protocol-id", default=UNKNOWN)
        if name == "generate":
            s.add_argument("--out", required=True)
            s.add_argument("--max-draws", type=int, default=DEFAULT_MAX_DRAWS)
        elif name == "verify":
            s.add_argument("--manifest", required=True)
    a = p.parse_args(argv)
    try:
        fx = load_fixtures(a.fixtures)
        c0m = C.load_manifest(a.c0_manifest)
        c0 = c0m["controls"][a.scenario]
        with open(a.c0_manifest, "rb") as fh:
            c0_sha = hashlib.sha256(fh.read()).hexdigest()
        tok = C._client_tokenizer(a.server, a.timeout)
        if a.cmd == "anchor":
            sc = {x.id: x for x in fx.scenarios}[a.scenario]
            res = anchor_reproduces_frozen(sc, a.offset, tok, c0)
            print(json.dumps(res))
            return 0 if res["reproduces_c0"] else 2
        if a.cmd == "generate":
            m = build_replicate_manifest(fx, a.scenario, a.offset, tok, c0_entry=c0, c0_manifest_sha256=c0_sha,
                                         protocol_id=a.protocol_id, max_draws=a.max_draws)
            write_manifest(a.out, m)
            for r in m["replicates"]:
                print(f"replicate {r['replicate']:>2} {r['domain']:<10} {r['status']:<8} draws={r['draws_used']} "
                      f"rubric_rejections={len(r['rubric_rejections'])}")
            print(f"accepted {m['summary']['n_accepted']}/{N_REPLICATES}; qualifies: {m['summary']['qualifies_for_analysis']}")
            return 0
        with open(a.manifest, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        probs = verify_replicate_manifest(m, fx, tok, c0_entry=c0)
        for q in probs:
            print("PROBLEM:", q)
        print("OK" if not probs else f"{len(probs)} problem(s)")
        return 0 if not probs else 2
    except Inconclusive as e:
        print(f"INCONCLUSIVE [{e.reason}]: {e}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
