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
PROTOCOL_DRAW_BUDGET = 5000             # EXACTLY this many draws per replicate, every rejected draw included. Fixed by
                                        # the amended protocol (AMENDMENT-001); the CLI exposes no override.
DEFAULT_MAX_DRAWS = PROTOCOL_DRAW_BUDGET  # kept as an alias; equals the frozen generator's default (tested)
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

RUBRIC_ID = "rubric/1.1"
RUBRIC = {
    "rubric_id": RUBRIC_ID,
    "revision_note": (
        "R1-R6 are mechanically identical to rubric/1. AMENDMENT-001 removed the manual-review clause (which allowed a "
        "human to reject a candidate and advance the stream) from this hashed rubric: human review is annotation-only "
        "(REVIEW_POLICY) and never affects selection."),
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
                           f"with each of them must be <= {JACCARD_MAX} (R6). Only C0's fixed TEXT takes part; C0's "
                           f"observed probability result never enters selection.",
    "applies_to": "words after the fixed 'Tenant <n>' prefix",
    "rejection_codes": {
        "R1_TASK_WORD": "word occurs in the task text and is not a stopword",
        "R2_TASK_STEM": "word is a strict prefix/extension of a task content word (>= 4 letters)",
        "R3_FIELD_WORD": "word is in the prohibited semantic-field list",
        "R4_NUMERAL": "numeral shares a digit run with the scenario's prompt material",
        "R5_DUPLICATE": "text identical to an accepted control or C0",
        "R6_OVERLAP": "content-word Jaccard with an accepted control or C0 above the maximum",
    },
}
RUBRIC_SHA256 = hashlib.sha256(json.dumps({**RUBRIC, "jaccard_max": JACCARD_MAX, "n_replicates": N_REPLICATES,
                                           "min_qualifying": MIN_QUALIFYING}, sort_keys=True).encode()).hexdigest()

#: Human review is ANNOTATION-ONLY. It is deliberately NOT part of the hashed rubric and NOT part of any seed.
REVIEW_SCHEMA = "bleediff.replicate-review/1"
REVIEW_POLICY = {
    "policy_id": "review/1",
    "mode": "ANNOTATION_ONLY",
    "when": "after generation, before any inference; may not look at any model output",
    "may_record_per_accepted_candidate": ["confound_present (true/false)", "concern_phrases (exact words or phrases of "
                                          "concern, each must occur in the control text)",
                                          "severity (none/minor/moderate/serious)", "reason", "review_date",
                                          "reviewer_id (non-identifying)"],
    "may_not": ["replace a candidate", "regenerate a candidate", "reorder candidates",
                "discard or exclude a candidate from the primary predefined set", "advance a stream",
                "edit any text"],
    "no_go": "If serious answer-related contamination is identified the reviewer may issue NO_GO_FOR_INFERENCE. That "
             "blocks inference and requires a NEW protocol version; it never replaces a candidate.",
    "primary_analysis": "always the complete set of accepted candidates, in replicate order",
    "sensitivity_analysis_S1": "a separately labelled re-analysis that excludes accepted candidates annotated "
                               "confound_present = true; fewer than 10 remaining => INSUFFICIENT; never replaces the "
                               "primary analysis",
}
REVIEW_POLICY_SHA256 = hashlib.sha256(json.dumps(REVIEW_POLICY, sort_keys=True).encode()).hexdigest()

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
                  screen_fn=None, max_draws: int = PROTOCOL_DRAW_BUDGET, fingerprint: str | None = None) -> dict:
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
            checks.append(RUBRIC_ID)
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
                       max_draws: int = PROTOCOL_DRAW_BUDGET, fingerprint: str | None = None, unit: str = "words") -> dict:
    treat = C.treatment_fragment(sc, offset, unit)
    fp = fingerprint or (C.tokenizer_fingerprint(tokenize) if tokenize is not None else UNKNOWN)
    t_sha = C.fragment_sha256(treat)
    ident = dict(scenario_id=sc.id, k=k, offset=offset, unit=unit, treatment_sha256=t_sha, fingerprint=fp)
    seed_hex = replicate_seed_hex(**ident)
    res = search_stream(sc, offset, unit, tokenize, seed=int(seed_hex[:16], 16), forced_domain=domain_for(k),
                        screen_fn=lambda text: screen(text, sc, accepted_texts=accepted_texts),
                        max_draws=max_draws, fingerprint=fp)
    out = {"replicate": k, "domain": domain_for(k), "seed_hex16": seed_hex[:16], "control_id": seed_hex[:32],
           "status": res["status"], "effective_draw_budget": max_draws, "draws_used": res["draws_used"],
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




# --------------------------------------------------------------------------------------
# facts about the header pool (a documented inaccuracy of the original preregistration, now verifiable)
# --------------------------------------------------------------------------------------
def header_pool_facts(sc: Scenario, offset: int, unit: str = "words") -> dict:
    """How many header phrases each domain really offers for THIS scenario's skeleton, before and after RUBRIC."""
    sk = C.parse_skeleton(C.treatment_fragment(sc, offset, unit))
    per = {}
    for cl in C.CLUSTERS:
        heads = [h for h in cl["headers"] if len(h) == sk.header_len]
        ok = [h for h in heads if not screen("Tenant 4 " + " ".join(h) + ": zzz", sc)]
        per[cl["id"]] = {"all_headers": len(cl["headers"]), "eligible_for_skeleton": len(heads), "surviving_rubric": len(ok)}
    return {"header_words_required_by_skeleton": sk.header_len, "per_domain": per}


# --------------------------------------------------------------------------------------
# canonical forms and identities
# --------------------------------------------------------------------------------------
def canonical(obj) -> str:
    """Canonical JSON text used for every byte-for-byte comparison."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def selection_sha256(manifest: dict) -> str:
    return hashlib.sha256(canonical(manifest["selection"]).encode("ascii")).hexdigest()


ANCHOR_SCHEMA = "bleediff.replicate-anchor/1"


def anchor_evidence_bytes(evidence: dict) -> bytes:
    return (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode("utf-8")


def anchor_evidence_sha256(evidence: dict) -> str:
    return hashlib.sha256(anchor_evidence_bytes(evidence)).hexdigest()


def anchor_reproduces_frozen(sc: Scenario, offset: int, tokenize, c0_entry: dict, *,
                             max_draws: int = PROTOCOL_DRAW_BUDGET, unit: str = "words") -> dict:
    """This module's draw loop, run in FROZEN-compatible mode (frozen seed, drawn domain, no rubric), must reproduce the
    historical control C0 exactly under the real tokenizer. If it does not, the copy of the draw loop has drifted from
    the frozen generator (or the tokenizer differs) and no replicate may be trusted."""
    fp = C.tokenizer_fingerprint(tokenize)
    t_sha = C.fragment_sha256(C.treatment_fragment(sc, offset, unit))
    cid = C.control_id_for(scenario_id=sc.id, offset=offset, unit=unit, treatment_sha256=t_sha, fingerprint=fp)
    res = search_stream(sc, offset, unit, tokenize, seed=int(cid[:16], 16), forced_domain=None, screen_fn=None,
                        max_draws=max_draws, fingerprint=fp)
    ok = (res["status"] == "ACCEPTED" and cid == c0_entry["control_id"] and res["text"] == c0_entry["fragment_text"]
          and res["draw"] == c0_entry["enumeration_index"] and res["cluster"] == c0_entry["cluster"])
    return {"reproduces_c0": ok, "control_id": cid, "c0_control_id": c0_entry["control_id"],
            "draw": res.get("draw"), "c0_draw": c0_entry["enumeration_index"], "_result": res}


def run_anchor(sc: Scenario, offset: int, tokenize, c0_entry: dict, *, unit: str = "words") -> dict:
    """Deterministic, timestamp-free anchor EVIDENCE. A failed anchor is recorded, never hidden."""
    fp = C.tokenizer_fingerprint(tokenize)
    a = anchor_reproduces_frozen(sc, offset, tokenize, c0_entry, unit=unit)
    res = a["_result"]
    return {
        "schema": ANCHOR_SCHEMA, "scenario_id": sc.id, "offset": offset, "unit": unit,
        "tokenizer_fingerprint": fp, "draw_budget": PROTOCOL_DRAW_BUDGET,
        "frozen_generator": {"algorithm": C.ALGORITHM, "lexicon_sha256": C.LEXICON_SHA256},
        "c0": {"control_id": c0_entry["control_id"], "fragment_sha256": c0_entry["fragment_sha256"],
               "enumeration_index": c0_entry["enumeration_index"], "cluster": c0_entry["cluster"],
               "recorded_tokenizer_fingerprint": c0_entry.get("tokenizer_fingerprint")},
        "recomputed": {"control_id": a["control_id"], "status": res["status"], "draw": res.get("draw"),
                       "cluster": res.get("cluster"),
                       "fragment_sha256": C.fragment_sha256(res["text"]) if res["status"] == "ACCEPTED" else None},
        "reproduces_c0": a["reproduces_c0"],
    }


def _anchor_problems(anchor: dict, fp: str, c0_entry: dict, scenario_id: str, offset: int, unit: str) -> list[str]:
    p: list[str] = []
    if not isinstance(anchor, dict) or anchor.get("schema") != ANCHOR_SCHEMA:
        return ["anchor evidence is missing or has the wrong schema"]
    if anchor.get("reproduces_c0") is not True:
        p.append("the anchor did not reproduce C0")
    if anchor.get("tokenizer_fingerprint") != fp:
        p.append("anchor was recorded under a different tokenizer")
    if (anchor.get("scenario_id"), anchor.get("offset"), anchor.get("unit")) != (scenario_id, offset, unit):
        p.append("anchor is for a different scenario/offset/unit")
    if anchor.get("draw_budget") != PROTOCOL_DRAW_BUDGET:
        p.append("anchor draw budget differs from the protocol budget")
    c0 = anchor.get("c0", {})
    rc = anchor.get("recomputed", {})
    if c0.get("control_id") != c0_entry["control_id"] or c0.get("fragment_sha256") != c0_entry["fragment_sha256"] \
            or rc.get("control_id") != c0_entry["control_id"] or rc.get("fragment_sha256") != c0_entry["fragment_sha256"]:
        p.append("anchor does not identify the protocol's C0")
    if anchor.get("frozen_generator") != {"algorithm": C.ALGORITHM, "lexicon_sha256": C.LEXICON_SHA256}:
        p.append("anchor frozen-generator identity differs")
    return p


def _check_controlling(controlling: dict, fp: str, scenario_id: str, offset: int, unit: str, c0_entry: dict) -> list[str]:
    p: list[str] = []
    need = ("protocol_id", "protocol_revision", "original_preregistration_sha256", "original_protocol_sha256",
            "amendment_sha256", "effective_protocol_sha256", "rubric_sha256", "frozen_generator", "draw_budget",
            "tokenizer_fingerprint_expected", "c0", "scenario_id", "offset", "unit", "fixtures_sha256")
    missing = [k for k in need if not isinstance(controlling, dict) or k not in controlling]
    if missing:
        return [f"controlling documents are incomplete: missing {missing}"]
    if controlling["rubric_sha256"] != RUBRIC_SHA256:
        p.append("controlling rubric hash differs from the implemented rubric")
    if controlling["draw_budget"] != PROTOCOL_DRAW_BUDGET:
        p.append("controlling draw budget differs from the fixed protocol budget")
    if controlling["tokenizer_fingerprint_expected"] != fp:
        p.append("tokenizer fingerprint differs from the one the protocol expects")
    if (controlling["scenario_id"], controlling["offset"], controlling["unit"]) != (scenario_id, offset, unit):
        p.append("scenario/offset/unit differ from the protocol's")
    if controlling["c0"].get("control_id") != c0_entry["control_id"] or controlling["c0"].get("fragment_sha256") != c0_entry["fragment_sha256"]:
        p.append("C0 differs from the protocol's C0")
    if controlling["frozen_generator"].get("lexicon_sha256") != C.LEXICON_SHA256 \
            or controlling["frozen_generator"].get("algorithm") != C.ALGORITHM:
        p.append("frozen generator identity differs")
    return p


# --------------------------------------------------------------------------------------
# selection (deterministic) and manifest (selection + provenance)
# --------------------------------------------------------------------------------------
def build_replicate_selection(fixtures: Fixtures, scenario_id: str, offset: int, tokenize, *, c0_entry: dict,
                              c0_manifest_sha256: str, max_draws: int = PROTOCOL_DRAW_BUDGET,
                              unit: str = "words") -> dict:
    """The deterministic part of a manifest. Same inputs => byte-identical output. Needs a tokenizer."""
    if tokenize is None:
        raise ControlMatchError("replicate selection needs a tokenizer", reason="CONTROL_TOKENIZER_REQUIRED")
    sc = {s.id: s for s in fixtures.scenarios}[scenario_id]
    fp = C.tokenizer_fingerprint(tokenize)
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
        "scenario_id": scenario_id, "offset": offset, "unit": unit, "fixtures_sha256": fixtures.sha256,
        "treatment_fragment_sha256": C.fragment_sha256(treat), "lexicon_sha256": C.LEXICON_SHA256,
        "rubric_sha256": RUBRIC_SHA256, "tokenizer_fingerprint": fp, "effective_draw_budget": max_draws,
        "c0": {"role": "historical control, OUTSIDE the primary 12-control analysis; only its fixed TEXT takes part in "
                       "distinctness checks, its observed result never enters selection",
               "control_id": c0_entry["control_id"], "fragment_sha256": c0_entry["fragment_sha256"],
               "source_manifest_sha256": c0_manifest_sha256},
        "allocation": {"rule": "domain(k) = sorted(domain ids)[k mod 8]", "domains": list(DOMAINS), "n_replicates": N_REPLICATES},
        "seed_rule": "sha256(canonical JSON of replicate_seed_material)[:16] as the SplitMix64 seed",
        "selection_rule": "first candidate of the replicate's own stream that passes every frozen structural check and "
                          f"{RUBRIC_ID}; rejected candidates are recorded; no model output is consulted; a failed "
                          "replicate stays failed",
        "replicates": reps,
        "summary": {"n_requested": N_REPLICATES, "n_accepted": n_ok, "n_failed": N_REPLICATES - n_ok,
                    "min_qualifying": MIN_QUALIFYING, "qualifies_for_analysis": n_ok >= MIN_QUALIFYING,
                    "if_not_qualifying": "the later study must report INSUFFICIENT"},
    }


INDEPENDENCE_STATEMENT = ("These are deterministically diversified samples from a constrained text generator. Distinct "
                          "seeds and domains do not make them statistically independent or exchangeable observations.")


def build_replicate_manifest(fixtures: Fixtures, scenario_id: str, offset: int, tokenize, *, c0_entry: dict,
                             c0_manifest_sha256: str, controlling: dict, anchor: dict, code_revision: dict,
                             max_draws: int = PROTOCOL_DRAW_BUDGET, unit: str = "words") -> dict:
    """Selection + provenance. Refuses to run unless the controlling documents are complete and consistent and the
    anchor evidence shows that this module's draw loop reproduces C0 under the live tokenizer."""
    if tokenize is None:
        raise ControlMatchError("replicate generation needs a tokenizer", reason="CONTROL_TOKENIZER_REQUIRED")
    fp = C.tokenizer_fingerprint(tokenize)
    probs = _check_controlling(controlling, fp, scenario_id, offset, unit, c0_entry)
    probs += _anchor_problems(anchor, fp, c0_entry, scenario_id, offset, unit)
    if max_draws != controlling.get("draw_budget"):
        probs.append(f"draw budget {max_draws} differs from the protocol's fixed budget {controlling.get('draw_budget')}")
    if probs:
        raise ControlMatchError("generation refused: " + "; ".join(probs), reason="CONTROL_PROVENANCE_INVALID")
    sel = build_replicate_selection(fixtures, scenario_id, offset, tokenize, c0_entry=c0_entry,
                                    c0_manifest_sha256=c0_manifest_sha256, max_draws=max_draws, unit=unit)
    a_sha = anchor_evidence_sha256(anchor)
    return {
        "schema": MANIFEST_SCHEMA_V3, "algorithm": REPLICATE_ALGORITHM, "base_algorithm": C.ALGORITHM,
        "bleediff_version": __version__,
        "selection": sel,
        "provenance": {
            "generator": {"software_version": __version__, "control_algorithm": C.ALGORITHM,
                          "replicate_algorithm": REPLICATE_ALGORITHM, "protocol_id": controlling["protocol_id"],
                          "protocol_revision": controlling["protocol_revision"]},
            "controlling_documents": controlling,
            "anchor": {"evidence": anchor, "evidence_sha256": a_sha},
            "generation": {
                "invocation": {
                    "command": "python -m bleediff.replicates generate",
                    "parameters": {"scenario_id": scenario_id, "offset": offset, "unit": unit,
                                   "draw_budget_per_replicate": max_draws, "fixtures_sha256": fixtures.sha256,
                                   "c0_manifest_sha256": c0_manifest_sha256, "anchor_evidence_sha256": a_sha,
                                   "protocol_id": controlling["protocol_id"],
                                   "protocol_revision": controlling["protocol_revision"],
                                   "effective_protocol_sha256": controlling["effective_protocol_sha256"]},
                    "note": "canonical parameters. The CLI has no free parameter besides the output path, the "
                            "anchor-evidence path and the tokenizer server; the server address, local paths and "
                            "timestamps are deliberately not recorded"},
                "code_revision": code_revision},
            "independence_statement": INDEPENDENCE_STATEMENT,
        },
    }


def verify_replicate_manifest(manifest: dict, fixtures: Fixtures, tokenize, *, c0_entry: dict, c0_manifest_sha256: str,
                              controlling: dict) -> list[str]:
    """Structure, provenance AND byte-identical deterministic regeneration. Empty list = verified.

    A different but structurally valid control is NEVER accepted as equivalent: the selection is regenerated from the
    recorded identities and compared as canonical JSON."""
    probs: list[str] = []
    if tokenize is None:
        return ["regeneration requires a tokenizer: a manifest cannot be verified without it"]
    if manifest.get("schema") != MANIFEST_SCHEMA_V3 or manifest.get("algorithm") != REPLICATE_ALGORITHM \
            or not isinstance(manifest.get("selection"), dict) or not isinstance(manifest.get("provenance"), dict):
        return ["schema/algorithm/layout mismatch"]
    sel, prov = manifest["selection"], manifest["provenance"]
    if sel.get("fixtures_sha256") != fixtures.sha256:
        return ["controls were built for different fixtures"]
    sc = {s.id: s for s in fixtures.scenarios}.get(sel.get("scenario_id"))
    if sc is None:
        return ["unknown scenario"]
    offset, unit = sel.get("offset"), sel.get("unit", "words")
    fp = C.tokenizer_fingerprint(tokenize)
    if sel.get("tokenizer_fingerprint") != fp:
        probs.append("the live tokenizer differs from the one the manifest records")
    # ---- provenance -------------------------------------------------------------------------------------------
    probs += _check_controlling(controlling, fp, sc.id, offset, unit, c0_entry)
    if prov.get("controlling_documents") != controlling:
        probs.append("recorded controlling documents differ from the current controlling documents")
    gen = prov.get("generator", {})
    if gen.get("protocol_id") != controlling.get("protocol_id") or gen.get("protocol_revision") != controlling.get("protocol_revision") \
            or gen.get("replicate_algorithm") != REPLICATE_ALGORITHM or gen.get("control_algorithm") != C.ALGORITHM:
        probs.append("generator identity differs from the amended protocol")
    anc = prov.get("anchor", {})
    probs += _anchor_problems(anc.get("evidence"), fp, c0_entry, sc.id, offset, unit)
    if isinstance(anc.get("evidence"), dict) and anc.get("evidence_sha256") != anchor_evidence_sha256(anc["evidence"]):
        probs.append("anchor evidence hash does not match its content")
    if isinstance(anc.get("evidence"), dict):
        live = run_anchor(sc, offset, tokenize, c0_entry, unit=unit)
        if live != anc["evidence"]:
            probs.append("anchor evidence differs from a live re-run of the anchor")
    g = prov.get("generation", {})
    inv = (g.get("invocation") or {}).get("parameters", {})
    if inv.get("draw_budget_per_replicate") != controlling.get("draw_budget"):
        probs.append("recorded invocation draw budget disagrees with the protocol")
    if inv.get("anchor_evidence_sha256") != anc.get("evidence_sha256") or inv.get("effective_protocol_sha256") != controlling.get("effective_protocol_sha256"):
        probs.append("recorded invocation does not match the recorded anchor/protocol identities")
    rev = g.get("code_revision")
    if not isinstance(rev, dict) or set(rev) != {"git_commit", "dirty"} or rev.get("git_commit") in (None, UNKNOWN) \
            or rev.get("dirty") is not False:
        probs.append("code revision must be recorded as a clean git commit")
    if prov.get("independence_statement") != INDEPENDENCE_STATEMENT:
        probs.append("independence statement missing or altered")
    # ---- settings ---------------------------------------------------------------------------------------------
    if sel.get("effective_draw_budget") != controlling.get("draw_budget") or controlling.get("draw_budget") != PROTOCOL_DRAW_BUDGET:
        probs.append(f"recorded draw budget {sel.get('effective_draw_budget')!r} disagrees with the protocol's fixed budget")
    if sel.get("lexicon_sha256") != C.LEXICON_SHA256 or sel.get("rubric_sha256") != RUBRIC_SHA256:
        probs.append("lexicon or rubric changed since the controls were selected")
    treat = C.treatment_fragment(sc, offset, unit)
    if sel.get("treatment_fragment_sha256") != C.fragment_sha256(treat):
        probs.append("treatment fragment differs from the one the controls were built for")
    if (sel.get("c0") or {}).get("control_id") != c0_entry["control_id"] or (sel.get("c0") or {}).get("source_manifest_sha256") != c0_manifest_sha256:
        probs.append("recorded C0 identity differs")
    # ---- per-replicate structure (kept for precise diagnostics) -------------------------------------------------
    reps = sel.get("replicates", [])
    if [r.get("replicate") for r in reps] != list(range(N_REPLICATES)):
        return probs + ["replicates must be exactly indices 0..11 in order"]
    t_sha = C.fragment_sha256(treat)
    accepted = [c0_entry["fragment_text"]]
    n_ok = 0
    for r in reps:
        k = r["replicate"]
        tag = f"replicate {k}"
        if r.get("domain") != domain_for(k):
            probs.append(f"{tag}: domain {r.get('domain')!r} is not the preregistered {domain_for(k)!r}")
        seed = replicate_seed_hex(scenario_id=sc.id, k=k, offset=offset, unit=unit, treatment_sha256=t_sha, fingerprint=fp)
        if r.get("seed_hex16") != seed[:16] or r.get("control_id") != seed[:32]:
            probs.append(f"{tag}: seed/control_id do not follow the preregistered derivation")
        if r.get("effective_draw_budget") != controlling.get("draw_budget"):
            probs.append(f"{tag}: recorded draw budget differs from the protocol's")
        for rej in r.get("rubric_rejections", []):
            if not screen(rej["text"], sc, accepted_texts=tuple(accepted)):
                probs.append(f"{tag}: a recorded rubric rejection does not reproduce (draw {rej['draw']})")
        if r.get("status") == "ACCEPTED":
            n_ok += 1
            e = r.get("control", {})
            if e.get("cluster") != domain_for(k):
                probs.append(f"{tag}: cluster {e.get('cluster')!r} is not the forced domain")
            if screen(e.get("fragment_text", ""), sc, accepted_texts=tuple(accepted)):
                probs.append(f"{tag}: accepted control fails {RUBRIC_ID}")
            sub = {"lexicon_sha256": sel["lexicon_sha256"], "tokenizer_fingerprint": fp}
            probs += [f"{tag}: {p}" for p in C.verify_control(sc, e, sub, offset, unit, tokenize)]
            accepted.append(e.get("fragment_text", ""))
        elif r.get("status") == "FAILED":
            if not r.get("reason"):
                probs.append(f"{tag}: a failed replicate must record its reason")
            if r.get("draws_used") != controlling.get("draw_budget"):
                probs.append(f"{tag}: a failed replicate must have consumed the whole budget")
        else:
            probs.append(f"{tag}: unknown status")
    s = sel.get("summary", {})
    if s.get("n_accepted") != n_ok or s.get("n_failed") != N_REPLICATES - n_ok or \
            s.get("qualifies_for_analysis") != (n_ok >= MIN_QUALIFYING):
        probs.append("summary is inconsistent with the replicate records")
    # ---- deterministic regeneration: byte-identical or fail ------------------------------------------------------
    regen = build_replicate_selection(fixtures, sc.id, offset, tokenize, c0_entry=c0_entry,
                                      c0_manifest_sha256=c0_manifest_sha256, max_draws=controlling["draw_budget"], unit=unit)
    if canonical(regen) != canonical(sel):
        probs.append("selection is NOT byte-identical to its deterministic regeneration")
        for a, b in zip(regen["replicates"], reps):
            if canonical(a) != canonical(b):
                fields = sorted(f for f in set(a) | set(b) if canonical(a.get(f)) != canonical(b.get(f)))
                probs.append(f"replicate {a['replicate']} differs from regeneration in: {fields}")
        for f in sorted(set(regen) | set(sel)):
            if f != "replicates" and canonical(regen.get(f)) != canonical(sel.get(f)):
                probs.append(f"selection field {f!r} differs from regeneration")
    return probs


# --------------------------------------------------------------------------------------
# human review: ANNOTATION ONLY
# --------------------------------------------------------------------------------------
_REVIEW_KEYS = {"schema", "selection_sha256", "reviewer_id", "review_date", "annotations", "verdict", "verdict_reason"}
_ANNOT_KEYS = {"replicate", "confound_present", "concern_phrases", "severity", "reason", "review_date", "reviewer_id"}
REVIEW_SEVERITIES = ("none", "minor", "moderate", "serious")
REVIEW_VERDICTS = ("NO_ACTION", "NO_GO_FOR_INFERENCE")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_REVIEWER = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def validate_review(review: dict, manifest: dict) -> list[str]:
    """Any key that is not an annotation field (e.g. 'replacement', 'discard', 'order') is rejected outright."""
    import datetime

    probs: list[str] = []
    if not isinstance(review, dict) or review.get("schema") != REVIEW_SCHEMA:
        return ["review schema mismatch"]
    for k in sorted(set(review) - _REVIEW_KEYS):
        probs.append(f"forbidden review field {k!r}: a review can only annotate")
    if review.get("selection_sha256") != selection_sha256(manifest):
        probs.append("review is not bound to this manifest's selection")
    if not isinstance(review.get("reviewer_id"), str) or not _REVIEWER.match(review["reviewer_id"]):
        probs.append("reviewer_id must be a short non-identifying token (letters, digits, '_' or '-')")

    def date_ok(d) -> bool:
        try:
            return isinstance(d, str) and bool(_DATE.match(d)) and bool(datetime.date.fromisoformat(d))
        except ValueError:
            return False

    if not date_ok(review.get("review_date")):
        probs.append("review_date must be an ISO date")
    if review.get("verdict") not in REVIEW_VERDICTS:
        probs.append(f"verdict must be one of {REVIEW_VERDICTS}")
    if review.get("verdict") == "NO_GO_FOR_INFERENCE" and not str(review.get("verdict_reason") or "").strip():
        probs.append("a NO_GO_FOR_INFERENCE verdict needs a reason")
    accepted = {r["replicate"]: r["control"]["fragment_text"] for r in manifest["selection"]["replicates"]
                if r["status"] == "ACCEPTED"}
    seen = set()
    anns = review.get("annotations")
    if not isinstance(anns, list):
        return probs + ["annotations must be a list"]
    for i, a in enumerate(anns):
        tag = f"annotation {i}"
        if not isinstance(a, dict):
            probs.append(f"{tag}: not an object")
            continue
        for k in sorted(set(a) - _ANNOT_KEYS):
            probs.append(f"{tag}: forbidden field {k!r}: a review can only annotate")
        k = a.get("replicate")
        if k not in accepted:
            probs.append(f"{tag}: replicate {k!r} is not an accepted candidate")
            continue
        if k in seen:
            probs.append(f"{tag}: replicate {k} annotated twice")
        seen.add(k)
        if not isinstance(a.get("confound_present"), bool):
            probs.append(f"{tag}: confound_present must be true or false")
        if a.get("severity") not in REVIEW_SEVERITIES:
            probs.append(f"{tag}: severity must be one of {REVIEW_SEVERITIES}")
        if a.get("confound_present") is False and a.get("severity") not in (None, "none"):
            probs.append(f"{tag}: no confound means severity 'none'")
        phrases = a.get("concern_phrases", [])
        if not isinstance(phrases, list) or not all(isinstance(x, str) and x.strip() for x in phrases):
            probs.append(f"{tag}: concern_phrases must be a list of non-empty strings")
        else:
            if a.get("confound_present") is True and not phrases:
                probs.append(f"{tag}: a confound needs the exact words or phrases of concern")
            for ph in phrases:
                if ph.lower() not in accepted[k].lower():
                    probs.append(f"{tag}: concern phrase {ph!r} does not occur in the control text")
        if not isinstance(a.get("reason"), str) or not a["reason"].strip():
            probs.append(f"{tag}: a reason is required")
        if not date_ok(a.get("review_date")):
            probs.append(f"{tag}: review_date must be an ISO date")
        if not isinstance(a.get("reviewer_id"), str) or not _REVIEWER.match(a["reviewer_id"]):
            probs.append(f"{tag}: reviewer_id must be a short non-identifying token")
    return probs


def review_effects(review: dict, manifest: dict) -> dict:
    """What a VALID review may change: nothing about selection. It yields the unchanged primary set, a separately
    labelled sensitivity set (S1) and an inference-blocked flag."""
    probs = validate_review(review, manifest)
    if probs:
        raise ControlMatchError("review rejected: " + "; ".join(probs), reason="REVIEW_INVALID")
    primary = [r["replicate"] for r in manifest["selection"]["replicates"] if r["status"] == "ACCEPTED"]
    flagged = sorted(a["replicate"] for a in review["annotations"] if a["confound_present"])
    return {"primary_set": primary, "primary_set_preserved": True,
            "sensitivity_S1_excluding_annotated": [k for k in primary if k not in flagged],
            "sensitivity_label": "SENSITIVITY (not the primary analysis)", "annotated_confound": flagged,
            "inference_blocked": review["verdict"] == "NO_GO_FOR_INFERENCE",
            "inference_blocked_note": "a NO_GO verdict blocks inference and requires a new protocol version; it replaces nothing"}


def write_manifest(path: str, manifest: dict) -> None:
    if os.path.exists(path):
        raise ControlMatchError(f"refusing to overwrite existing {path!r}", reason="CONTROL_MANIFEST_EXISTS")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")


def _write_new(path: str, data: bytes) -> None:
    if os.path.exists(path):
        raise ControlMatchError(f"refusing to overwrite existing {path!r}", reason="CONTROL_MANIFEST_EXISTS")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def main(argv: list[str] | None = None) -> int:
    """Tokenize-only CLI. Scenario, offset, fixtures, C0, budget and protocol identity come ONLY from the controlling
    documents; the CLI has no parameter that could change them. Not used in the amendment checkpoint."""
    import argparse

    from . import meta as meta_mod
    from . import robustness as RB
    from .errors import Inconclusive

    p = argparse.ArgumentParser(prog="python -m bleediff.replicates", description="neutral-prose-replicates/1 (tokenize only)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("anchor", "generate", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--repo-root", default=".")
        s.add_argument("--server", required=True, help="used for /tokenize ONLY (route it through a guard that refuses /completion)")
        s.add_argument("--timeout", type=float, default=60.0)
        if name == "anchor":
            s.add_argument("--out", required=True, help="anchor evidence file (written even if the anchor FAILS)")
        elif name == "generate":
            s.add_argument("--anchor-evidence", required=True)
            s.add_argument("--out", required=True)
        else:
            s.add_argument("--manifest", required=True)
    a = p.parse_args(argv)
    try:
        ctl = RB.load_controlling_documents(a.repo_root)
        fx = load_fixtures(os.path.join(a.repo_root, *RB.FIXTURES_REL.split("/")))
        c0_path = os.path.join(a.repo_root, *RB.SEMANTIC_CONTROLS_REL.split("/"))
        c0 = C.load_manifest(c0_path)["controls"][ctl["scenario_id"]]
        with open(c0_path, "rb") as fh:
            c0_sha = hashlib.sha256(fh.read()).hexdigest()
        sc = {x.id: x for x in fx.scenarios}[ctl["scenario_id"]]
        tok = C._client_tokenizer(a.server, a.timeout)
        if a.cmd == "anchor":
            ev = run_anchor(sc, ctl["offset"], tok, c0, unit=ctl["unit"])
            _write_new(a.out, anchor_evidence_bytes(ev))
            print(json.dumps({"reproduces_c0": ev["reproduces_c0"], "evidence_sha256": anchor_evidence_sha256(ev)}))
            return 0 if ev["reproduces_c0"] else 2
        if a.cmd == "generate":
            rev = meta_mod.code_revision()
            if rev.get("dirty") is not False:
                raise ControlMatchError("generation requires a clean, committed working tree so the code revision is exact",
                                        reason="CONTROL_PROVENANCE_INVALID")
            with open(a.anchor_evidence, "r", encoding="utf-8") as fh:
                anchor = json.load(fh)
            m = build_replicate_manifest(fx, sc.id, ctl["offset"], tok, c0_entry=c0, c0_manifest_sha256=c0_sha,
                                         controlling=ctl, anchor=anchor, code_revision=rev, unit=ctl["unit"])
            write_manifest(a.out, m)
            for r in m["selection"]["replicates"]:
                print(f"replicate {r['replicate']:>2} {r['domain']:<10} {r['status']:<8} draws={r['draws_used']}/"
                      f"{r['effective_draw_budget']} rubric_rejections={len(r['rubric_rejections'])}")
            s_ = m["selection"]["summary"]
            print(f"accepted {s_['n_accepted']}/{N_REPLICATES}; qualifies: {s_['qualifies_for_analysis']}")
            return 0
        with open(a.manifest, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        probs = verify_replicate_manifest(m, fx, tok, c0_entry=c0, c0_manifest_sha256=c0_sha, controlling=ctl)
        for q in probs:
            print("PROBLEM:", q)
        print("VERIFIED (byte-identical regeneration)" if not probs else f"{len(probs)} problem(s): STOP")
        return 0 if not probs else 2
    except (Inconclusive, RB.RobustnessError) as e:
        print(f"INCONCLUSIVE [{getattr(e, 'reason', 'ERROR')}]: {e}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
