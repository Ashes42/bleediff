"""neutral-prose/1: deterministic, fluent, structurally matched neutral control fragments (v0.1.2).

WHY THIS EXISTS
---------------
The v0.1.0/v0.1.1 control ("neutral-filler/2") swapped every word of the adjacent tenant's text for same-length
pool words, shifted digits and gibberish. The result matched bytes, words and tokens but was word salad with
answer-like tokens ("Yes", "Use", "Get"), a stray "1" where "Tenant 4" had been, and no syntax. It displaced the
output by up to 0.42 nats deterministically, so it could not tell "ordinary sensitivity to appended text" apart
from "a displaced adjacent-context contribution".

WHAT THIS DOES
--------------
For one scenario and one displacement (offset, unit) it writes ordinary, grammatical English prose that matches
the treatment fragment's *structure*: the same fixed ``Tenant 4`` prefix, header length, sentence/clause
skeleton (where the colon, comma and periods fall), truncation point, byte length, word count and token count
under the real tokenizer, and the same whole-prompt token count for request #3. It is off-topic neutral prose
(bakery, trail, orchestra, greenhouse, ...) with no vocabulary from the scenario, no answer-label words, no
ordinals/number words, no imperatives.

NO OVERFITTING (the important part)
-----------------------------------
Generation never sees, calls or depends on any model output. Candidates come from a seeded stream (SplitMix64
seeded by ``control_id``, a hash of the algorithm, scenario, offset, treatment-fragment hash, tokenizer
fingerprint and lexicon hash). The selection rule is fixed in advance: **the first candidate in that stream that
passes every structural acceptance check**. Acceptance uses only bytes, words, punctuation skeleton, lexical
bans, a grammar recognizer, and tokenizer counts. Nothing here may be re-run to "improve" a result; the selected
controls are preserved in a manifest and a new algorithm/lexicon version changes every control_id.

If exact matching is impossible within ``max_draws`` the generator raises ControlMatchError (INCONCLUSIVE).
It never relaxes a constraint silently.

LIMITS: structural checks cannot establish semantic neutrality. A human review of the actual text is a separate,
required step (see the v0.1.2 report). Topic relevance still differs between arms by construction: the
treatment is on-topic evidence, the control is off-topic prose.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import sys
from dataclasses import dataclass, field

from . import UNKNOWN, __version__
from .arena import Fixtures, Scenario, load_fixtures, plan_requests
from .errors import FixtureError, Inconclusive

ALGORITHM = "neutral-prose/1"
MANIFEST_SCHEMA = "bleediff.controls/1"           # one offset for every scenario (v0.1.2)
MANIFEST_SCHEMA_V2 = "bleediff.controls/2"        # per-scenario offsets (protocol amendments, v0.1.3)
ACCEPTED_MANIFEST_SCHEMAS = (MANIFEST_SCHEMA, MANIFEST_SCHEMA_V2)
DEFAULT_MAX_DRAWS = 5_000
DEFAULT_MANIFEST = os.path.join("fixtures", "controls", "neutral-prose-1.json")


class ControlMatchError(Inconclusive):
    """Exact structural matching was not achieved (or a stored control no longer validates)."""

    reason = "CONTROL_MATCH_FAILED"


# --------------------------------------------------------------------------------------
# deterministic RNG (own implementation: stable across Python versions)
# --------------------------------------------------------------------------------------
_MASK = (1 << 64) - 1


class Rng:
    """SplitMix64."""

    def __init__(self, seed: int):
        self.s = seed & _MASK

    def next(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & _MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        return self.next() % n

    def choice(self, seq):
        return seq[self.below(len(seq))]

    def chance(self, num: int, den: int) -> bool:
        return self.below(den) < num


# --------------------------------------------------------------------------------------
# lexicon: off-topic, benign, ordinary prose. Every cluster is a self-consistent micro-scene.
# Any edit changes LEXICON_SHA256 and therefore every control_id (versioning by construction).
# --------------------------------------------------------------------------------------
def _c(id, headers, subjects, subj_adj, verbs, objects, obj_adj, advs, pps, times):
    return {"id": id, "headers": headers, "subjects": subjects, "subj_adj": subj_adj, "verbs": verbs,
            "objects": objects, "obj_adj": obj_adj, "advs": advs, "pps": pps, "times": times}


CLUSTERS = (
    _c("bakery",
       (("bakery", "ledger"), ("oven", "rota"), ("pastry", "diary"), ("flour", "notes"),
        ("morning", "bread", "ledger"), ("village", "bakery", "diary"), ("pastry", "counter", "notes"),
        ("rye", "bread", "roster"), ("early", "morning", "baking", "notes")),
       (("baker", "bakers"), ("apprentice", "apprentices"), ("cook", "cooks"), ("customer", "customers"),
        ("neighbor", "neighbors")),
       ("young", "tired", "careful", "cheerful", "patient", "quiet", "friendly"),
       ("carried", "arranged", "counted", "stacked", "inspected", "admired"),
       (("loaf", "loaves"), ("tray", "trays"), ("basket", "baskets"), ("pastry", "pastries"), ("roll", "rolls"),
        ("bun", "buns"), ("cake", "cakes")),
       ("warm", "fresh", "golden", "crusty", "small", "wooden", "sweet"),
       ("slowly", "quietly", "carefully", "gently", "patiently"),
       (("beside", "the", "oven"), ("near", "the", "door"), ("inside", "the", "shop"), ("under", "the", "awning"),
        ("near", "the", "front", "window"), ("beside", "the", "brick", "oven"), ("along", "the", "narrow", "counter"),
        ("inside", "the", "small", "shop"), ("under", "the", "striped", "awning"), ("behind", "the", "wooden", "door"),
        ("across", "the", "tiled", "floor"), ("along", "the", "narrow", "stone", "counter"),
        ("beside", "the", "old", "brick", "oven"), ("inside", "the", "small", "corner", "shop")),
       (("before", "sunrise"), ("after", "school"), ("on", "quiet", "mornings"), ("after", "the", "lunch", "rush"),
        ("during", "the", "early", "morning"), ("before", "the", "shop", "opened"), ("at", "7:45"),
        ("around", "6:30"), ("by", "8:15"), ("at", "5:40"))),
    _c("trail",
       (("trail", "diary"), ("hiking", "notes"), ("scout", "roster"), ("summit", "journal"),
        ("trail", "guide", "notes"), ("hiking", "club", "minutes"), ("summit", "path", "diary"),
        ("ridge", "walk", "ledger"), ("weekend", "hiking", "club", "notes")),
       (("hiker", "hikers"), ("guide", "guides"), ("walker", "walkers"), ("scout", "scouts"),
        ("traveler", "travelers")),
       ("weary", "cheerful", "careful", "quiet", "young", "patient", "experienced"),
       ("photographed", "admired", "studied", "approached", "described", "remembered"),
       (("path", "paths"), ("bridge", "bridges"), ("ridge", "ridges"), ("stream", "streams"),
        ("meadow", "meadows"), ("boulder", "boulders"), ("signpost", "signposts")),
       ("narrow", "steep", "mossy", "winding", "wooden", "distant", "rocky"),
       ("steadily", "cautiously", "happily", "slowly", "quietly"),
       (("near", "the", "summit"), ("beside", "the", "stream"), ("along", "the", "ridge"),
        ("below", "the", "rocky", "summit"), ("near", "the", "stone", "hut"), ("beside", "the", "pine", "forest"),
        ("across", "the", "green", "valley"), ("under", "the", "tall", "pines"),
        ("along", "the", "winding", "river", "bank"), ("beside", "the", "old", "stone", "hut")),
       (("at", "dawn"), ("after", "the", "morning", "mist"), ("before", "the", "afternoon", "rain"),
        ("during", "the", "long", "climb"), ("around", "9:40"), ("by", "11:20"), ("at", "5:50"))),
    _c("orchestra",
       (("rehearsal", "notes"), ("choir", "roster"), ("concert", "diary"),
        ("village", "orchestra", "minutes"), ("choir", "practice", "notes"), ("music", "hall", "diary"),
        ("spring", "concert", "rehearsal", "notes")),
       (("violinist", "violinists"), ("conductor", "conductors"), ("singer", "singers"), ("pianist", "pianists"),
        ("musician", "musicians")),
       ("young", "patient", "gifted", "nervous", "cheerful", "tired", "focused"),
       ("rehearsed", "repeated", "practiced", "performed", "studied", "remembered", "admired"),
       (("melody", "melodies"), ("passage", "passages"), ("concerto", "concertos"), ("chorus", "choruses"),
        ("waltz", "waltzes"), ("lullaby", "lullabies"), ("symphony", "symphonies")),
       ("gentle", "lively", "familiar", "slow", "tender", "cheerful", "dramatic"),
       ("softly", "carefully", "patiently", "beautifully", "quietly"),
       (("on", "the", "stage"), ("inside", "the", "hall"), ("near", "the", "piano"),
        ("on", "the", "small", "stage"), ("beside", "the", "grand", "piano"), ("inside", "the", "old", "hall"),
        ("near", "the", "tall", "windows"), ("under", "the", "warm", "lights"),
        ("inside", "the", "old", "village", "hall"), ("beside", "the", "grand", "black", "piano")),
       (("before", "the", "concert"), ("after", "the", "interval"), ("during", "the", "long", "rehearsal"),
        ("on", "Sunday", "evenings"), ("at", "6:30"), ("around", "7:10"))),
    _c("greenhouse",
       (("garden", "notes"), ("seedling", "diary"), ("greenhouse", "ledger"),
        ("spring", "planting", "notes"), ("herb", "garden", "diary"), ("school", "garden", "minutes"),
        ("early", "spring", "planting", "diary")),
       (("gardener", "gardeners"), ("student", "students"), ("volunteer", "volunteers"), ("farmer", "farmers"),
        ("botanist", "botanists")),
       ("young", "patient", "careful", "cheerful", "quiet", "tired", "friendly"),
       ("watered", "tended", "planted", "sorted", "inspected", "labeled", "admired"),
       (("seedling", "seedlings"), ("tomato", "tomatoes"), ("flowerpot", "flowerpots"), ("herb", "herbs"),
        ("orchid", "orchids"), ("sunflower", "sunflowers"), ("cabbage", "cabbages")),
       ("tiny", "green", "healthy", "fragrant", "bright", "delicate", "leafy"),
       ("gently", "carefully", "patiently", "cheerfully", "slowly"),
       (("under", "the", "roof"), ("beside", "the", "bench"), ("near", "the", "wall"),
        ("beside", "the", "glass", "wall"), ("under", "the", "curved", "roof"), ("along", "the", "wooden", "bench"),
        ("inside", "the", "warm", "greenhouse"), ("near", "the", "sunny", "window"),
        ("along", "the", "long", "wooden", "bench"), ("inside", "the", "warm", "glass", "greenhouse")),
       (("during", "the", "spring"), ("after", "the", "morning", "frost"), ("before", "the", "summer", "heat"),
        ("at", "noon"), ("around", "10:30"), ("by", "4:45"))),
    _c("seaside",
       (("lighthouse", "diary"), ("beach", "notes"), ("pier", "ledger"),
        ("seaside", "village", "notes"), ("fishing", "boat", "diary"), ("coastal", "walk", "minutes"),
        ("old", "lighthouse", "keeper", "diary")),
       (("fisherman", "fishermen"), ("keeper", "keepers"), ("child", "children"), ("sailor", "sailors"),
        ("painter", "painters")),
       ("weathered", "cheerful", "careful", "quiet", "young", "patient", "sunburnt"),
       ("carried", "gathered", "sorted", "inspected", "admired", "counted"),
       (("net", "nets"), ("lantern", "lanterns"), ("shell", "shells"), ("rope", "ropes"), ("bucket", "buckets"),
        ("boat", "boats"), ("sail", "sails")),
       ("old", "salty", "bright", "heavy", "wet", "small", "faded"),
       ("slowly", "cheerfully", "patiently", "carefully", "quietly"),
       (("below", "the", "cliff"), ("beside", "the", "pier"), ("along", "the", "beach"),
        ("along", "the", "sandy", "beach"), ("beside", "the", "wooden", "pier"), ("below", "the", "white", "cliff"),
        ("near", "the", "small", "lighthouse"), ("under", "the", "pale", "sky"),
        ("along", "the", "wide", "sandy", "beach"), ("beside", "the", "old", "wooden", "pier")),
       (("at", "sunset"), ("before", "the", "evening", "tide"), ("after", "the", "afternoon", "storm"),
        ("during", "the", "quiet", "morning"), ("around", "6:20"), ("by", "9:10"))),
    _c("library",
       (("library", "notes"), ("reading", "diary"), ("book", "ledger"),
        ("reading", "room", "notes"), ("village", "library", "minutes"), ("story", "hour", "diary"),
        ("quiet", "reading", "room", "notes")),
       (("librarian", "librarians"), ("reader", "readers"), ("visitor", "visitors"), ("student", "students"),
        ("author", "authors")),
       ("quiet", "patient", "careful", "cheerful", "young", "tired", "friendly"),
       ("sorted", "stacked", "borrowed", "catalogued", "admired", "remembered"),
       (("novel", "novels"), ("atlas", "atlases"), ("postcard", "postcards"), ("magazine", "magazines"),
        ("pamphlet", "pamphlets"), ("notebook", "notebooks"), ("bookmark", "bookmarks")),
       ("dusty", "heavy", "colorful", "thin", "leather", "faded", "popular"),
       ("quietly", "carefully", "slowly", "happily", "patiently"),
       (("beside", "the", "lamp"), ("near", "the", "desk"), ("along", "the", "shelves"),
        ("beside", "the", "reading", "table"), ("along", "the", "tall", "shelves"), ("near", "the", "front", "desk"),
        ("under", "the", "green", "lamp"), ("inside", "the", "quiet", "library"),
        ("beside", "the", "long", "reading", "table"), ("along", "the", "tall", "oak", "shelves")),
       (("on", "rainy", "afternoons"), ("before", "the", "library", "closed"), ("after", "the", "morning", "visit"),
        ("during", "the", "long", "winter"), ("at", "4:30"), ("around", "2:15"))),
    _c("festival",
       (("festival", "notes"), ("parade", "roster"), ("fair", "diary"),
        ("summer", "fair", "notes"), ("lantern", "festival", "diary"), ("village", "parade", "minutes"),
        ("midsummer", "street", "fair", "notes")),
       (("child", "children"), ("neighbor", "neighbors"), ("painter", "painters"), ("dancer", "dancers"),
        ("musician", "musicians")),
       ("cheerful", "young", "proud", "tired", "excited", "friendly", "patient"),
       ("hung", "tied", "carried", "decorated", "gathered", "counted", "admired"),
       (("lantern", "lanterns"), ("banner", "banners"), ("ribbon", "ribbons"), ("kite", "kites"),
        ("garland", "garlands"), ("flag", "flags"), ("balloon", "balloons")),
       ("bright", "paper", "red", "golden", "colorful", "tiny", "silky"),
       ("cheerfully", "happily", "carefully", "quickly", "proudly"),
       (("above", "the", "square"), ("beside", "the", "fountain"), ("along", "the", "street"),
        ("across", "the", "village", "square"), ("along", "the", "cobbled", "street"),
        ("above", "the", "wooden", "stage"), ("beside", "the", "old", "fountain"), ("near", "the", "town", "clock"),
        ("across", "the", "wide", "village", "square"), ("along", "the", "narrow", "cobbled", "street")),
       (("before", "the", "parade"), ("during", "the", "summer", "fair"), ("on", "festival", "evenings"),
        ("at", "5:45"), ("around", "8:25"), ("by", "3:50"))),
    _c("workshop",
       (("workshop", "notes"), ("craft", "diary"), ("studio", "ledger"),
        ("village", "craft", "notes"), ("pottery", "studio", "diary"), ("wood", "shop", "minutes"),
        ("old", "village", "craft", "diary")),
       (("carpenter", "carpenters"), ("apprentice", "apprentices"), ("potter", "potters"), ("weaver", "weavers"),
        ("sculptor", "sculptors")),
       ("patient", "careful", "young", "quiet", "skilled", "tired", "cheerful"),
       ("shaped", "sanded", "carved", "measured", "polished", "inspected", "admired"),
       (("stool", "stools"), ("spoon", "spoons"), ("bowl", "bowls"), ("basket", "baskets"), ("shelf", "shelves"),
        ("frame", "frames"), ("vase", "vases")),
       ("smooth", "wooden", "rough", "small", "sturdy", "pale", "curved"),
       ("slowly", "carefully", "patiently", "steadily", "quietly"),
       (("beside", "the", "bench"), ("along", "the", "wall"), ("near", "the", "window"),
        ("beside", "the", "low", "bench"), ("under", "the", "hanging", "lamp"), ("along", "the", "back", "wall"),
        ("inside", "the", "small", "workshop"), ("near", "the", "open", "window"),
        ("beside", "the", "long", "low", "bench"), ("inside", "the", "small", "stone", "workshop")),
       (("before", "the", "evening", "bell"), ("after", "the", "midday", "meal"), ("during", "the", "cold", "season"),
        ("at", "10:20"), ("around", "3:35"), ("by", "7:55"))),
)

SG_DETS = ("the", "each")
PL_DETS = ("the", "several", "many", "some")
CONJUNCTIONS = ("while", "as", "until")
#: Quantities (>= 2-digit runs; no run of length 1 in 1-4) and clock times (hours 5-12, two-digit minutes).
QUANTITIES = ("18", "26", "35", "42", "57", "63", "75", "84", "96", "120", "215", "340", "465", "580", "725",
              "860", "5,200", "6,450", "7,800", "8,100", "9,350")

#: Function words that may legitimately recur in the scenario text.
FUNCTION_WORDS = frozenset(
    "the of and to in on at for with by from as while until before after near beside along across over under "
    "through during each some many several their its into onto than that this these those below above behind "
    "inside around by".split())

#: Words that must never appear in a control: answer-like, label-like, imperative-capable, evaluative,
#: ranking/ordinal, number words that map to option positions, mnemonic words for A-D, structural words.
GLOBAL_BANNED = frozenset(
    "yes no true false correct incorrect right wrong answer answers choose chose select selected option options "
    "choice choices pick best better worst worse should must need needs recommend recommended prefer preferred "
    "ignore please note important critical major minor severe urgent serious primary secondary first second third "
    "fourth last one two three four single both pair alpha beta gamma delta apple banana cat dog a b c d use set "
    "get run go do see end key map tenant severity sev".split()) | frozenset(
    # magnitude / rate / rating / evaluation words: could weakly prime scale-like options (severity, capacity,
    # latency, "best"). Added in the v0.1.2 pre-inference semantic review; chosen by meaning, not by any output.
    "small tiny large big huge heavy light thin low high tall short long wide narrow slow slowly quick quickly "
    "fast steady steadily popular healthy skilled gifted experienced sturdy delicate excited beautifully grand "
    "final full empty".split())

#: Reviewer-defined B anchors for the v0.1.2 fragment-integrity audit (NOT used by generation).
FRAGMENT_ANCHORS = {
    "incident-triage": ("CDN", "cache invalidation"),
    "architecture-selection": ("DynamoDB",),
    "vendor-routing": ("Harbor Reefer",),
    "config-audit": ("Debug logging",),
    "triage-severity": ("major",),
}

PROBE_TEXTS = ("A", " A", "B", " B", "C", " C", "D", " D", "\nTenant 4 edge operations record: The", " 1,300 7:45 240",
               " the baker carefully polished the golden trays", " Several tired musicians rehearsed", ".", ",", ":")


def _lexicon_blob() -> str:
    return json.dumps(
        {"algorithm": ALGORITHM, "clusters": CLUSTERS, "sg_dets": SG_DETS, "pl_dets": PL_DETS,
         "conj": CONJUNCTIONS, "quantities": QUANTITIES, "function_words": sorted(FUNCTION_WORDS),
         "banned": sorted(GLOBAL_BANNED)}, sort_keys=True)


LEXICON_SHA256 = hashlib.sha256(_lexicon_blob().encode()).hexdigest()
_CLUSTER_BY_ID = {c["id"]: c for c in CLUSTERS}

# --------------------------------------------------------------------------------------
# skeleton of the treatment fragment
# --------------------------------------------------------------------------------------
_TERMINALS = ".,:"


@dataclass(frozen=True)
class Skeleton:
    prefix: tuple            # fixed leading words shared by both arms, e.g. ("Tenant", "4")
    header_len: int          # words after the prefix up to and including the colon word
    clauses: tuple           # ((n_words, terminator), ...) complete clauses after the header
    partial: int             # words in the truncated final clause (0 = none)
    partial_kind: str        # "main" (new sentence) | "tail" (continues after a comma)
    words: int
    bytes: int

    def to_json(self) -> dict:
        return {"prefix": list(self.prefix), "header_len": self.header_len,
                "clauses": [list(c) for c in self.clauses], "partial": self.partial,
                "partial_kind": self.partial_kind, "words": self.words, "bytes": self.bytes}


def _terminal(word: str) -> str | None:
    return word[-1] if word and word[-1] in _TERMINALS else None


def parse_skeleton(fragment: str) -> Skeleton:
    words = fragment.split()
    if len(words) < 8:
        raise ControlMatchError("treatment fragment too short to mirror", reason="CONTROL_UNSUPPORTED_SKELETON")
    prefix = tuple(words[:2]) if words[0] == "Tenant" and words[1].isdigit() else ()
    rest = words[len(prefix):]
    header_len = 0
    for i, w in enumerate(rest):
        if _terminal(w) == ":":
            header_len = i + 1
            break
    body = rest[header_len:]
    clauses, run, last_term = [], 0, None
    for w in body:
        run += 1
        t = _terminal(w)
        if t in (".", ","):
            clauses.append((run, t))
            run, last_term = 0, t
    partial_kind = "tail" if last_term == "," else "main"
    if any(_terminal(w) == ":" for w in body):
        raise ControlMatchError("a second colon is not supported", reason="CONTROL_UNSUPPORTED_SKELETON")
    return Skeleton(prefix, header_len, tuple(clauses), run, partial_kind, len(words), len(fragment.encode()))


# --------------------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------------------
_OPT_CACHE: dict = {}


def _np_options(cl: dict, role: str, length: int) -> list:
    """Every noun phrase of exactly `length` words for this cluster/role: det|NUM  adj*  noun (agreeing)."""
    key = (cl["id"], role, length)
    got = _OPT_CACHE.get(key)
    if got is not None:
        return got
    adjs = cl["subj_adj"] if role == "subj" else cl["obj_adj"]
    nouns = cl["subjects"] if role == "subj" else cl["objects"]
    out = []
    for a in itertools.permutations(adjs, length - 2):
        for sg, pl in nouns:
            out += [(d,) + a + (sg,) for d in SG_DETS]
            out += [(d,) + a + (pl,) for d in PL_DETS]
            if role == "obj":
                out += [(q,) + a + (pl,) for q in QUANTITIES]
    _OPT_CACHE[key] = out
    return out


def _slot_options(cl: dict, kind: str, length: int) -> list:
    if kind in ("subj", "obj"):
        return _np_options(cl, kind, length)
    if kind == "adv":
        return [(a,) for a in cl["advs"]]
    if kind == "verb":
        return [(v,) for v in cl["verbs"]]
    if kind == "conj":
        return [(c,) for c in CONJUNCTIONS]
    if kind == "pp":
        return [p for p in cl["pps"] if len(p) == length]
    if kind == "time":
        return [t for t in cl["times"] if len(t) == length]
    raise ValueError(kind)


def _clause_shape(rng: Rng, cl: dict, kind: str, n: int):
    """A random sequence of (slot_kind, length) for a clause of exactly n words, or None."""
    pp_l = (0,) + tuple(sorted({len(p) for p in cl["pps"]}))
    tm_l = (0,) + tuple(sorted({len(t) for t in cl["times"]}))
    lead = 1 if kind == "tail" else 0
    combos = [c for c in itertools.product((2, 3, 4), (0, 1), (1,), (2, 3, 4), pp_l, tm_l) if sum(c) == n - lead]
    if not combos:
        return None
    s, adv, v, o, pp, tm = combos[rng.below(len(combos))]
    seq = [("conj", 1)] if lead else []
    seq += [("subj", s)] + ([("adv", 1)] if adv else []) + [("verb", 1), ("obj", o)]
    seq += ([("pp", pp)] if pp else []) + ([("time", tm)] if tm else [])
    return seq


def build_slots(rng: Rng, cl: dict, sk: Skeleton):
    """Random STRUCTURE for one draw: an ordered list of slot specs (options, words_kept, suffix, capitalise, tag).
    The partial clause is built as a longer clause of which only the first `partial` words are kept."""
    specs = []
    if sk.header_len:
        heads = [h for h in cl["headers"] if len(h) == sk.header_len]
        if not heads:
            return None
        specs.append((heads, sk.header_len, ":", False, ("header", sk.header_len)))

    def add_clause(kind, n, keep, term):
        seq = _clause_shape(rng, cl, kind, n)
        if seq is None:
            return False
        pos = 0
        for slot_kind, length in seq:
            inc = max(0, min(length, keep - pos))
            if inc > 0:
                opts = _slot_options(cl, slot_kind, length)
                if not opts:
                    return False
                suffix = term if (term and keep == n and pos + length == n) else ""
                specs.append((opts, inc, suffix, kind == "main" and pos == 0, (slot_kind, length)))
            pos += length
        return True

    prev = None
    for n, term in sk.clauses:
        if not add_clause("tail" if prev == "," else "main", n, n, term):
            return None
        prev = term
    if sk.partial:
        full = sk.partial + 2 + rng.below(4)               # a longer clause, cut mid-sentence
        if not add_clause("tail" if prev == "," else "main", full, sk.partial, None):
            return None
    return specs


def _word_ok(word: str, vocab: frozenset, runs: frozenset) -> bool:
    core = _norm(word)
    if any(ch.isdigit() for ch in core):
        return all(not ((len(r) == 1 and r in "1234") or (len(r) >= 2 and r in runs)) for r in _DIGITS.findall(core))
    return bool(core) and len(core) > 1 and core not in GLOBAL_BANNED and not (core in vocab and core not in FUNCTION_WORDS)


def _signature_groups(spec, counter, vocab, runs) -> dict:
    """{(bytes, tokens): [word tuples]} for one slot, lexically pre-filtered, in deterministic order."""
    opts, inc, suffix, cap, _tag = spec
    groups: dict = {}
    for o in opts:
        w = list(o[:inc])
        if cap:
            w[0] = w[0][:1].upper() + w[0][1:]
        if suffix:
            w[-1] = w[-1] + suffix
        if not all(_word_ok(x, vocab, runs) for x in w):
            continue
        sig = (sum(len(x) for x in w), counter.units(w) if counter else 0)
        groups.setdefault(sig, set()).add(tuple(w))
    return {sig: sorted(v) for sig, v in sorted(groups.items())}


def solve_slots(rng: Rng, groups: list, target_bytes: int, target_tokens: int):
    """Exact knapsack: pick one option per slot so that total bytes and tokens hit the targets. Deterministic;
    random choice among feasible options is seeded. Returns a list of word tuples, or None if infeasible."""
    n = len(groups)
    if any(not g for g in groups):
        return None
    lo_b = [min(s[0] for s in g) for g in groups] + [0]
    hi_b = [max(s[0] for s in g) for g in groups] + [0]
    lo_t = [min(s[1] for s in g) for g in groups] + [0]
    hi_t = [max(s[1] for s in g) for g in groups] + [0]
    for i in range(n - 1, -1, -1):                       # suffix sums
        lo_b[i] += lo_b[i + 1]; hi_b[i] += hi_b[i + 1]; lo_t[i] += lo_t[i + 1]; hi_t[i] += hi_t[i + 1]
    reach = [{(0, 0)}]
    for i, g in enumerate(groups):
        nxt = set()
        for b, t in reach[i]:
            for sb, st in g:
                nb, nt = b + sb, t + st
                rb, rt = target_bytes - nb, target_tokens - nt
                if lo_b[i + 1] <= rb <= hi_b[i + 1] and lo_t[i + 1] <= rt <= hi_t[i + 1]:
                    nxt.add((nb, nt))
        if not nxt:
            return None
        reach.append(nxt)
    if (target_bytes, target_tokens) not in reach[n]:
        return None
    chosen, cur = [], (target_bytes, target_tokens)
    for i in range(n - 1, -1, -1):
        feas = [(sig, len(opts)) for sig, opts in groups[i].items()
                if (cur[0] - sig[0], cur[1] - sig[1]) in reach[i]]
        total = sum(c for _, c in feas)
        r = rng.below(total)
        for sig, c in feas:                              # weight by option count (approx. uniform over options)
            if r < c:
                break
            r -= c
        chosen.append(rng.choice(groups[i][sig]))
        cur = (cur[0] - sig[0], cur[1] - sig[1])
    chosen.reverse()
    return chosen


# --------------------------------------------------------------------------------------
# independent grammar recognizer (does not reuse the generator's code path)
# --------------------------------------------------------------------------------------
def _norm(w: str) -> str:
    return re.sub(r"^[.,:]+|[.,:]+$", "", w).lower()


def _seq(t, i, seq, trunc):
    """Match tuple `seq` at t[i:]; returns end index, or None. Running out of tokens mid-way is only
    acceptable when truncation is allowed."""
    for k, s in enumerate(seq):
        if i + k >= len(t):
            return len(t) if trunc else None
        if t[i + k] != s:
            return None
    return i + len(seq)


def _np_ends(t, i, cl, role, trunc):
    ends = set()
    if i >= len(t):
        return {len(t)} if trunc else set()
    nouns = cl["subjects"] if role == "subj" else cl["objects"]
    adjs = set(cl["subj_adj"] if role == "subj" else cl["obj_adj"])
    heads = [(d, "sg") for d in SG_DETS] + [(d, "pl") for d in PL_DETS]
    if role == "obj":
        heads += [(q.lower(), "num") for q in QUANTITIES]
    for head, kind in heads:
        if t[i] != head:
            continue
        j = i + 1
        for nadj in (0, 1, 2):
            k = j
            ok = True
            for _ in range(nadj):
                if k >= len(t):
                    if trunc:
                        ends.add(len(t))
                    ok = False
                    break
                if t[k] not in adjs:
                    ok = False
                    break
                k += 1
            if not ok:
                continue
            if k >= len(t):
                if trunc:
                    ends.add(len(t))
                continue
            for sg, pl in nouns:
                if (t[k] == sg and kind in ("sg",) or t[k] == pl and kind in ("pl", "num")
                        or (head == "the" and t[k] in (sg, pl))):
                    ends.add(k + 1)
    return ends


def _optional(t, ends, options, trunc):
    out = set(ends)
    for e in ends:
        for opt in options:
            r = _seq(t, e, tuple(x.lower() for x in opt), trunc)
            if r is not None:
                out.add(r)
        if e >= len(t) and trunc:
            out.add(len(t))
    return out


def recognize_clause(tokens: list[str], cl: dict, kind: str, complete: bool) -> bool:
    """True iff `tokens` (normalised) is a valid derivation (complete) or a valid prefix of one (truncated) of
    SUBJ [ADV] VERB OBJ [PP] [TIME], optionally introduced by a subordinating conjunction (kind='tail')."""
    trunc = not complete
    t = list(tokens)
    start = 0
    if kind == "tail":
        if not t or t[0] not in CONJUNCTIONS:
            return False
        start = 1
        if len(t) == 1:
            return trunc
    ends = _np_ends(t, start, cl, "subj", trunc)
    if not ends:
        return False
    ends = _optional(t, ends, [(a,) for a in cl["advs"]], trunc)
    nxt = set()
    for e in ends:
        if e >= len(t):
            if trunc:
                nxt.add(len(t))
        elif t[e] in cl["verbs"]:
            nxt.add(e + 1)
    ends = set()
    for e in nxt:
        if e >= len(t):
            if trunc:
                ends.add(len(t))
        else:
            ends |= _np_ends(t, e, cl, "obj", trunc)
    if not ends:
        return False
    ends = _optional(t, ends, cl["pps"], trunc)
    ends = _optional(t, ends, cl["times"], trunc)
    return len(t) in ends


def recognize_fragment(words: list[str], sk: Skeleton, cluster_id: str) -> list[str]:
    """Problems found when re-parsing a candidate against the grammar and the skeleton (empty = coherent)."""
    cl = _CLUSTER_BY_ID.get(cluster_id)
    if cl is None:
        return [f"unknown cluster {cluster_id!r}"]
    problems = []
    if len(words) != sk.words:
        return [f"word count {len(words)} != {sk.words}"]
    if tuple(words[:len(sk.prefix)]) != sk.prefix:
        problems.append("fixed prefix differs")
    pos = len(sk.prefix)
    if sk.header_len:
        h = [_norm(w) for w in words[pos:pos + sk.header_len]]
        if not any(tuple(h) == tuple(x) for x in cl["headers"]) or _terminal(words[pos + sk.header_len - 1]) != ":":
            problems.append("header is not a header of the chosen cluster")
        pos += sk.header_len
    prev = None
    for idx, (n, term) in enumerate(sk.clauses):
        seg = words[pos:pos + n]
        kind = "tail" if prev == "," else "main"
        toks = [_norm(w) for w in seg]
        if _terminal(seg[-1]) != term or any(_terminal(w) for w in seg[:-1]):
            problems.append(f"clause {idx + 1}: punctuation skeleton differs")
        if kind == "main" and not seg[0][:1].isupper():
            problems.append(f"clause {idx + 1}: sentence does not start with a capital")
        if not recognize_clause(toks, cl, kind, True):
            problems.append(f"clause {idx + 1}: not a valid {kind} clause of cluster {cluster_id}")
        pos += n
        prev = term
    if sk.partial:
        seg = words[pos:pos + sk.partial]
        kind = "tail" if prev == "," else "main"
        if any(_terminal(w) for w in seg):
            problems.append("truncated clause carries terminal punctuation")
        if kind == "main" and not seg[0][:1].isupper():
            problems.append("truncated sentence does not start with a capital")
        if not recognize_clause([_norm(w) for w in seg], cl, kind, False):
            problems.append(f"truncated clause is not a valid prefix of a {kind} clause of cluster {cluster_id}")
    return problems


# --------------------------------------------------------------------------------------
# structural acceptance checks (no model output is ever consulted)
# --------------------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-z]+")
_DIGITS = re.compile(r"\d+")


def scenario_vocabulary(sc: Scenario) -> tuple[frozenset, frozenset]:
    """(alphabetic words, digit-runs) occurring anywhere in the scenario's prompt material."""
    blob = " ".join([sc.question, *sc.options.values(), *sc.authorized, sc.adjacent])
    return frozenset(_WORD_RE.findall(blob.lower())), frozenset(_DIGITS.findall(blob))


def lexical_problems(words: list[str], sk: Skeleton, vocab: frozenset, digit_runs: frozenset) -> list[str]:
    problems = []
    for w in words[len(sk.prefix):]:
        core = _norm(w)
        if any(ch.isdigit() for ch in core):
            for run in _DIGITS.findall(core):
                if len(run) == 1 and run in "1234":
                    problems.append(f"single digit {run!r} (maps to an option position)")
                if len(run) >= 2 and run in digit_runs:
                    problems.append(f"numeral run {run!r} occurs in the scenario text")
            continue
        if not core:
            problems.append("empty word")
        elif len(core) == 1:
            problems.append(f"lone letter {core!r}")
        elif core in GLOBAL_BANNED:
            problems.append(f"banned word {core!r}")
        elif core in vocab and core not in FUNCTION_WORDS:
            problems.append(f"word {core!r} occurs in the scenario text")
        for m in _WORD_RE.findall(core):
            if len(core) > 1 and m != core:
                problems.append(f"unexpected characters in {w!r}")
    cores = [_norm(w) for w in words[len(sk.prefix):]]
    content = [c for c in cores if c and c not in FUNCTION_WORDS]
    for a, b in zip(cores, cores[1:]):
        if a and a == b:
            problems.append(f"repeated adjacent word {a!r}")
    for c in set(content):
        if content.count(c) > 1:
            problems.append(f"content word {c!r} repeated")
    stems: dict = {}
    for c in content:
        if c.isalpha() and len(c) >= 5:
            stems.setdefault(c[:5], set()).add(c)
    for stem, ws in sorted(stems.items()):
        if len(ws) > 1:
            problems.append(f"words share a stem: {sorted(ws)}")
    return problems


def skeleton_problems(words: list[str], sk: Skeleton) -> list[str]:
    cand = parse_skeleton(" ".join(words)) if len(words) >= 8 else None
    problems = []
    if cand is None or cand.to_json() != {**sk.to_json(), "bytes": cand.bytes}:
        problems.append("word/punctuation skeleton differs from the treatment fragment")
    return problems


# --------------------------------------------------------------------------------------
# tokenizer helpers
# --------------------------------------------------------------------------------------
def tokenizer_fingerprint(tokenize) -> str:
    """SHA-256 over the token ids of a fixed probe set: identifies the tokenizer, not the model file."""
    rows = [list(tokenize(t)) for t in PROBE_TEXTS]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


class TokenCounter:
    """Exact and additive token counts through a ``tokenize(text) -> list[int]`` callable (cached)."""

    def __init__(self, tokenize):
        self.tokenize = tokenize
        self._unit: dict = {}

    def exact(self, text: str) -> int:
        return len(self.tokenize(text))

    def units(self, words) -> int:
        """Additive tokens of words that follow a space (the first word of a fragment follows a newline)."""
        total = 0
        for w in words:
            key = (" ", w)
            n = self._unit.get(key)
            if n is None:
                n = self._unit[key] = len(self.tokenize(" " + w))
            total += n
        return total

    def additive(self, words: list[str]) -> int:
        total = 0
        for i, w in enumerate(words):
            left = "\n" if i == 0 else " "
            key = (left, w)
            n = self._unit.get(key)
            if n is None:
                n = self._unit[key] = len(self.tokenize(left + w))
            total += n
        return total


# --------------------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------------------
def fragment_sha256(text: str) -> str:
    return hashlib.sha256(b"bleediff-fragment\0" + text.encode("ascii")).hexdigest()


def control_id_for(*, scenario_id: str, offset: int, unit: str, treatment_sha256: str, fingerprint: str) -> str:
    blob = json.dumps({"algorithm": ALGORITHM, "scenario_id": scenario_id, "offset": offset, "unit": unit,
                       "treatment_fragment_sha256": treatment_sha256, "tokenizer_fingerprint": fingerprint,
                       "lexicon_sha256": LEXICON_SHA256}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


@dataclass
class ControlRecord:
    scenario_id: str
    algorithm: str
    control_id: str
    enumeration_index: int
    cluster: str
    offset: int
    unit: str
    fragment_text: str
    fragment_sha256: str
    treatment_fragment_sha256: str
    skeleton: dict
    counts: dict                       # bytes, words, fragment_tokens, prompt_tokens (treatment & control)
    tokenizer_fingerprint: str
    lexicon_sha256: str
    token_parity: str                  # "VERIFIED" | "PENDING"
    checks: list = field(default_factory=list)

    def to_json(self) -> dict:
        return dict(self.__dict__)


def treatment_fragment(sc: Scenario, offset: int, unit: str = "words") -> str:
    plan = plan_requests(sc, "treatment", offset, unit)
    try:
        return plan.bleed_fragment.decode("ascii")
    finally:
        plan.dispose()


def prompt_parity(sc: Scenario, offset: int, unit: str, control_slot: str, tokenize) -> dict:
    """Whole-request token parity plus geometry/surround identity, using the REAL plan_requests prompts."""
    t = plan_requests(sc, "treatment", offset, unit)
    c = plan_requests(sc, "control", offset, unit, filler=control_slot)
    try:
        t_prompts = [bytes(p) for p in t.prompts]
        c_prompts = [bytes(p) for p in c.prompts]
        tg, cg = t.geometry, c.geometry
        res = {
            "requests_1_2_identical": t_prompts[:2] == c_prompts[:2],
            "n_requests": (len(t_prompts), len(c_prompts)),
            "geometry_identical": (tg.boundary, tg.read_end, tg.bleed_bytes, tg.bleed_words, tg.clamped)
            == (cg.boundary, cg.read_end, cg.bleed_bytes, cg.bleed_words, cg.clamped) and not cg.clamped,
        }
        # surrounding prompt identical = everything outside the displaced fragment is byte-identical
        head_t, tail_t = _split_prompt(t_prompts[2], t.bleed_fragment)
        head_c, tail_c = _split_prompt(c_prompts[2], c.bleed_fragment)
        res["surrounding_prompt_identical"] = head_t == head_c and tail_t == tail_c and head_t is not None
        res["prompt_tokens"] = (len(tokenize(t_prompts[2].decode("ascii"))), len(tokenize(c_prompts[2].decode("ascii"))))
        return res
    finally:
        t.dispose()
        c.dispose()


def _split_prompt(prompt: bytes, fragment: bytes):
    i = prompt.find(fragment)
    return (None, None) if i < 0 else (prompt[:i], prompt[i + len(fragment):])


# --------------------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------------------
def generate_control(sc: Scenario, offset: int, unit: str = "words", tokenize=None, *,
                     max_draws: int = DEFAULT_MAX_DRAWS, fingerprint: str | None = None) -> ControlRecord:
    """Deterministically build the control for one scenario. With ``tokenize=None`` only the static checks run
    and the record is marked token_parity='PENDING' (it is NOT a validated control)."""
    if unit != "words":
        raise ControlMatchError("neutral-prose/1 supports unit='words' only", reason="CONTROL_UNSUPPORTED_UNIT")
    treat = treatment_fragment(sc, offset, unit)
    sk = parse_skeleton(treat)
    vocab, runs = scenario_vocabulary(sc)
    counter = TokenCounter(tokenize) if tokenize is not None else None
    fp = fingerprint or (tokenizer_fingerprint(tokenize) if tokenize is not None else UNKNOWN)
    t_sha = fragment_sha256(treat)
    cid = control_id_for(scenario_id=sc.id, offset=offset, unit=unit, treatment_sha256=t_sha, fingerprint=fp)
    rng = Rng(int(cid[:16], 16))
    want_tokens = counter.exact("\n" + treat) if counter else None
    if counter is not None and counter.additive(treat.split()) != want_tokens:
        raise ControlMatchError(
            "this tokenizer is not additive over words for the treatment fragment; exact token matching cannot "
            "be guaranteed", reason="CONTROL_TOKENIZER_NOT_ADDITIVE")
    # budget the solver must hit exactly: everything except the fixed shared prefix
    pre_bytes = sum(len(w) for w in sk.prefix)
    target_bytes = sk.bytes - (sk.words - 1) - pre_bytes
    target_tokens = (want_tokens - counter.additive(list(sk.prefix))) if counter else 0
    rejected = {"structure": 0, "infeasible": 0, "lexical": 0, "grammar": 0, "tokens": 0, "prompt": 0}
    sig_cache: dict = {}
    for draw in range(max_draws):
        cl = rng.choice(CLUSTERS)
        specs = build_slots(rng, cl, sk)
        if specs is None:
            rejected["structure"] += 1
            continue
        groups = []
        for spec in specs:
            key = (cl["id"], spec[4], spec[1], spec[2], spec[3])
            if key not in sig_cache:
                sig_cache[key] = _signature_groups(spec, counter, vocab, runs)
            groups.append(sig_cache[key])
        solved = solve_slots(rng, groups, target_bytes, target_tokens)
        if solved is None:
            rejected["infeasible"] += 1
            continue
        words = list(sk.prefix) + [w for part in solved for w in part]
        cluster = cl["id"]
        text = " ".join(words)
        if len(text.encode()) != sk.bytes or len(words) != sk.words:   # solver invariant; never relaxed
            raise AssertionError("solver returned a candidate that violates the byte/word budget")
        if lexical_problems(words, sk, vocab, runs) or skeleton_problems(words, sk):
            rejected["lexical"] += 1
            continue
        if recognize_fragment(words, sk, cluster):
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
            pp = prompt_parity(sc, offset, unit, text, tokenize)
            if not (pp["requests_1_2_identical"] and pp["geometry_identical"] and pp["surrounding_prompt_identical"]
                    and pp["n_requests"] == (3, 3) and pp["prompt_tokens"][0] == pp["prompt_tokens"][1]):
                rejected["prompt"] += 1
                continue
            parity = "VERIFIED"
            checks += ["fragment_tokens", "prompt_tokens", "geometry", "surrounding_prompt"]
            counts["fragment_tokens"] = (want_tokens, got)
            counts["prompt_tokens"] = pp["prompt_tokens"]
        return ControlRecord(
            scenario_id=sc.id, algorithm=ALGORITHM, control_id=cid, enumeration_index=draw, cluster=cluster,
            offset=offset, unit=unit, fragment_text=text, fragment_sha256=fragment_sha256(text),
            treatment_fragment_sha256=t_sha, skeleton=sk.to_json(), counts=counts, tokenizer_fingerprint=fp,
            lexicon_sha256=LEXICON_SHA256, token_parity=parity, checks=checks)
    raise ControlMatchError(
        f"no candidate matched scenario {sc.id!r} exactly within {max_draws} draws "
        f"(rejections: {rejected}); constraints were NOT relaxed", reason="CONTROL_MATCH_FAILED")


def build_controls(fixtures: Fixtures, offset: int, unit: str = "words", tokenize=None, *,
                   max_draws: int = DEFAULT_MAX_DRAWS) -> dict:
    fp = tokenizer_fingerprint(tokenize) if tokenize is not None else UNKNOWN
    recs = {sc.id: generate_control(sc, offset, unit, tokenize, max_draws=max_draws, fingerprint=fp)
            for sc in fixtures.scenarios}
    return {
        "schema": MANIFEST_SCHEMA, "algorithm": ALGORITHM, "bleediff_version": __version__,
        "lexicon_sha256": LEXICON_SHA256, "tokenizer_fingerprint": fp, "fixtures_sha256": fixtures.sha256,
        "offset": offset, "unit": unit,
        "selection_rule": "first candidate in the SplitMix64 stream seeded by control_id that passes every "
                          "structural check; no model output is consulted",
        "controls": {sid: r.to_json() for sid, r in recs.items()},
    }


def build_protocol_controls(fixtures: Fixtures, offsets: dict, unit: str = "words", tokenize=None, *,
                            previous: dict | None = None, previous_sha256: str | None = None,
                            protocol_id: str | None = None, max_draws: int = DEFAULT_MAX_DRAWS) -> dict:
    """Controls for a protocol with PER-SCENARIO offsets (schema bleediff.controls/2).

    An entry from ``previous`` is carried over VERBATIM when its offset, treatment fragment, tokenizer
    fingerprint and lexicon are unchanged (the frozen algorithm would regenerate it identically; that is
    checked separately). Only scenarios whose window changed are generated, with the unchanged algorithm.

    Three different identities are recorded and must not be confused (see ``generator``):
    the SOFTWARE that wrote the file, the control ALGORITHM that produced each control, and the experimental
    PROTOCOL the controls belong to. ``bleediff_version`` is kept for compatibility and means the first of these."""
    fp = tokenizer_fingerprint(tokenize) if tokenize is not None else UNKNOWN
    controls, reused, generated = {}, [], []
    for sc in fixtures.scenarios:
        off = offsets[sc.id]
        old = (previous or {}).get("controls", {}).get(sc.id)
        t_sha = fragment_sha256(treatment_fragment(sc, off, unit))
        if (old and old.get("offset") == off and old.get("treatment_fragment_sha256") == t_sha
                and old.get("tokenizer_fingerprint") == fp and old.get("lexicon_sha256") == LEXICON_SHA256):
            controls[sc.id] = json.loads(json.dumps(old))
            reused.append(sc.id)
        else:
            controls[sc.id] = generate_control(sc, off, unit, tokenize, max_draws=max_draws, fingerprint=fp).to_json()
            generated.append(sc.id)
    return {
        "schema": MANIFEST_SCHEMA_V2, "algorithm": ALGORITHM, "bleediff_version": __version__,
        "generator": {
            "software_version": __version__,          # bleediff release that WROTE this file (not the protocol)
            "control_algorithm": ALGORITHM,           # algorithm that produced each control (frozen since v0.1.2)
            "protocol_id": protocol_id or UNKNOWN,    # experimental protocol these controls belong to
        },
        "lexicon_sha256": LEXICON_SHA256, "tokenizer_fingerprint": fp, "fixtures_sha256": fixtures.sha256,
        "unit": unit, "offsets": dict(offsets),
        "selection_rule": "first candidate in the SplitMix64 stream seeded by control_id that passes every "
                          "structural check; no model output is consulted",
        "carried_over": {"from_manifest_sha256": previous_sha256 or UNKNOWN, "scenarios": reused},
        "generated": generated,
        "controls": controls,
    }


def write_manifest(path: str, manifest: dict) -> None:
    if os.path.exists(path):
        raise FixtureError(f"refusing to overwrite existing {path!r}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")


def load_manifest(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise ControlMatchError(f"cannot read control manifest {path!r}: {e}",
                                reason="CONTROL_MANIFEST_INVALID") from e
    if doc.get("schema") not in ACCEPTED_MANIFEST_SCHEMAS or doc.get("algorithm") != ALGORITHM:
        raise ControlMatchError("control manifest schema/algorithm mismatch", reason="CONTROL_MANIFEST_INVALID")
    return doc


def verify_control(sc: Scenario, entry: dict, manifest: dict, offset: int, unit: str, tokenize) -> list[str]:
    """Re-verify a PRESERVED control against the live tokenizer and the current fixtures. Empty list = valid."""
    problems = []
    if manifest.get("lexicon_sha256") != LEXICON_SHA256:
        problems.append("lexicon changed since the controls were selected")
    sel_offset = entry.get("offset", manifest.get("offset"))     # per-entry since v0.1.2 records; v2 manifests need it
    sel_unit = entry.get("unit", manifest.get("unit"))
    if (sel_offset, sel_unit) != (offset, unit):
        problems.append(f"controls were selected for offset={sel_offset} {sel_unit}, not {offset} {unit}")
        return problems
    treat = treatment_fragment(sc, offset, unit)
    if fragment_sha256(treat) != entry["treatment_fragment_sha256"]:
        problems.append("treatment fragment differs from the one the control was built for (fixtures changed)")
        return problems
    text = entry["fragment_text"]
    if fragment_sha256(text) != entry["fragment_sha256"]:
        problems.append("stored control text does not match its recorded hash")
    sk = parse_skeleton(treat)
    vocab, runs = scenario_vocabulary(sc)
    words = text.split()
    problems += [f"lexical: {p}" for p in lexical_problems(words, sk, vocab, runs)]
    problems += [f"skeleton: {p}" for p in skeleton_problems(words, sk)]
    problems += [f"grammar: {p}" for p in recognize_fragment(words, sk, entry["cluster"])]
    if len(text.encode()) != sk.bytes or len(words) != sk.words:
        problems.append("byte/word parity broken")
    fp = tokenizer_fingerprint(tokenize)
    if fp != manifest.get("tokenizer_fingerprint"):
        problems.append("tokenizer differs from the one the controls were validated against")
        return problems
    if len(tokenize("\n" + text)) != len(tokenize("\n" + treat)):
        problems.append("fragment token parity broken")
    pp = prompt_parity(sc, offset, unit, text, tokenize)
    if not (pp["requests_1_2_identical"] and pp["geometry_identical"] and pp["surrounding_prompt_identical"]):
        problems.append("insertion geometry / surrounding prompt differs")
    if pp["prompt_tokens"][0] != pp["prompt_tokens"][1]:
        problems.append(f"whole-prompt token parity broken {pp['prompt_tokens']}")
    return problems


# --------------------------------------------------------------------------------------
# treatment-fragment integrity (does the displaced fragment contain its B evidence?)
# --------------------------------------------------------------------------------------
def fragment_integrity(fixtures: Fixtures, offset: int, unit: str = "words") -> list[dict]:
    out = []
    for sc in fixtures.scenarios:
        frag = treatment_fragment(sc, offset, unit)
        anchors = FRAGMENT_ANCHORS.get(sc.id, ())
        present = [a for a in anchors if a.lower() in frag.lower()]
        full = sc.adjacent
        pos = {a: full.lower().find(a.lower()) for a in anchors}
        words_before = {a: (len(full[:p].split()) + 1 if p >= 0 else None) for a, p in pos.items()}
        out.append({"scenario_id": sc.id, "option_B": sc.options["B"], "anchors": list(anchors),
                    "anchors_in_fragment": present, "all_anchors_present": len(present) == len(anchors),
                    "anchor_first_word_index_in_slot": words_before, "fragment": frag})
    return out


# --------------------------------------------------------------------------------------
# module CLI: tokenize-only generation / verification. (Deliberately NOT a `bleediff` subcommand: the
# bleediff CLI lexicon is frozen.) Never calls /completion.
# --------------------------------------------------------------------------------------
def _client_tokenizer(url: str, timeout: float):
    from .client import LlamaClient

    client = LlamaClient(url, timeout=timeout)
    return (lambda text: client.tokenize(text, cache=False))


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m bleediff.control", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "verify", "static"):
        s = sub.add_parser(name)
        s.add_argument("--fixtures", default=os.path.join("fixtures", "contexts.json"))
        s.add_argument("--offset", type=int, default=24)
        s.add_argument("--unit", default="words")
        if name != "static":
            s.add_argument("--server", default="http://127.0.0.1:8080", help="used for /tokenize ONLY")
            s.add_argument("--timeout", type=float, default=60.0)
        if name == "generate":
            s.add_argument("--out", default=DEFAULT_MANIFEST)
            s.add_argument("--max-draws", type=int, default=DEFAULT_MAX_DRAWS)
            s.add_argument("--override", action="append", default=[], metavar="SCENARIO=N",
                           help="per-scenario offset (protocol amendment); makes a bleediff.controls/2 manifest")
            s.add_argument("--reuse", default=None,
                           help="previous manifest whose unchanged controls are carried over verbatim")
            s.add_argument("--protocol-id", default=None, help="recorded in the manifest's generator block")
        if name == "verify":
            s.add_argument("--manifest", default=DEFAULT_MANIFEST)
    a = p.parse_args(argv)
    try:
        fx = load_fixtures(a.fixtures)
        if a.cmd == "static":
            for sc in fx.scenarios:
                r = generate_control(sc, a.offset, a.unit, None)
                print(f"{sc.id}: {r.token_parity} (static checks passed; draw {r.enumeration_index}, cluster {r.cluster})")
            print("token parity is PENDING: static output is not a validated control")
            return 0
        tok = _client_tokenizer(a.server, a.timeout)
        if a.cmd == "generate":
            if a.override or a.reuse:
                offs = {sc.id: a.offset for sc in fx.scenarios}
                for item in a.override:
                    sid, _, n = item.partition("=")
                    if sid not in offs or not n.isdigit():
                        raise ControlMatchError(f"bad --override {item!r}", reason="BAD_ARGS")
                    offs[sid] = int(n)
                prev = load_manifest(a.reuse) if a.reuse else None
                psha = None
                if a.reuse:
                    with open(a.reuse, "rb") as fh:
                        psha = hashlib.sha256(fh.read()).hexdigest()
                manifest = build_protocol_controls(fx, offs, a.unit, tok, previous=prev, previous_sha256=psha,
                                                   protocol_id=a.protocol_id, max_draws=a.max_draws)
            else:
                manifest = build_controls(fx, a.offset, a.unit, tok, max_draws=a.max_draws)
            write_manifest(a.out, manifest)
            for sid, c in manifest["controls"].items():
                print(f"{sid}: draw {c['enumeration_index']}, cluster {c['cluster']}, tokens {c['counts']['fragment_tokens']}, "
                      f"prompt tokens {c['counts']['prompt_tokens']}")
            print(f"wrote {a.out}")
            return 0
        manifest = load_manifest(a.manifest)
        bad = 0
        for sc in fx.scenarios:
            entry = manifest["controls"].get(sc.id)
            off = (manifest.get("offsets") or {}).get(sc.id, a.offset)
            probs = ["no control recorded"] if entry is None else verify_control(sc, entry, manifest, off, a.unit, tok)
            print(f"{sc.id}: {'OK' if not probs else 'INVALID: ' + '; '.join(probs)}")
            bad += bool(probs)
        return 0 if not bad else 2
    except Inconclusive as e:
        print(f"INCONCLUSIVE [{e.reason}]: {e}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
