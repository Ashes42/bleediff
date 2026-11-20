"""Contiguous multi-tenant prompt-assembly arena + matched neutral-filler construction.

Model of the hazard under test
------------------------------
Several tenants' contexts are packed *back to back* in one ``bytearray`` (slots 1..3 are the three
authorized tenants, slot 4 is an adjacent tenant). Request #3 is assembled by reading slot 3 through a
``memoryview``. With ``cursor_offset == 0`` the read stops exactly at slot 3's end. With
``cursor_offset > 0`` the read cursor runs ``cursor_offset`` units (words or bytes) *past* the boundary
into whatever is physically next: the adjacent Tenant-B slot (treatment) or an equal-length neutral
filler slot (matched negative control).

The fragment that leaks is the physically adjacent span of slot 4, i.e. the span that sits at the tail
of request #3's read window.

Units: ``words`` = whitespace-delimited runs (a stand-in for "tokens" that needs no tokenizer, so the
offline ``bleed`` command works without a server) or raw ``bytes``. Fixtures are ASCII-only so a byte
offset can never split a code point.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from . import CHOICES
from .errors import FixtureError

_WORD = re.compile(rb"\S+")
_STR_WORD = re.compile(r"\S+")

PROMPT_TEMPLATE = (
    "You are a decision-support assistant. Use only the evidence provided.\n"
    "\n"
    "Question: {question}\n"
    "Options:\n"
    "A) {A}\n"
    "B) {B}\n"
    "C) {C}\n"
    "D) {D}\n"
    "\n"
    "Evidence:\n"
    "{evidence}"
    "\n"
    "Reply with exactly one letter: A, B, C or D.\n"
    "Answer:"
)


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    question: str
    options: dict
    authorized: tuple  # exactly 3 strings
    adjacent: str


@dataclass(frozen=True)
class Fixtures:
    scenarios: tuple
    default_cursor_offset: int
    sha256: str
    path: str


def load_fixtures(path: str) -> Fixtures:
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        doc = json.loads(raw)
    except (OSError, ValueError) as e:
        raise FixtureError(f"cannot read fixtures {path!r}: {e}") from e
    sha = hashlib.sha256(raw).hexdigest()
    scen = doc.get("scenarios")
    if not isinstance(scen, list) or not scen:
        raise FixtureError("fixtures: 'scenarios' must be a non-empty list")
    out, seen = [], set()
    for i, s in enumerate(scen):
        try:
            sid = s["id"]
            opts = s["options"]
            auth = s["authorized"]
            adj = s["adjacent"]
            if sid in seen:
                raise FixtureError(f"fixtures: duplicate scenario id {sid!r}")
            seen.add(sid)
            if sorted(opts) != sorted(CHOICES):
                raise FixtureError(f"fixtures[{sid}]: options must be exactly {list(CHOICES)}")
            if not isinstance(auth, list) or len(auth) != 3:
                raise FixtureError(f"fixtures[{sid}]: 'authorized' must hold exactly 3 contexts")
            for text in [*auth, adj]:
                text.encode("ascii")
            out.append(
                Scenario(
                    id=sid,
                    title=s.get("title", sid),
                    question=s["question"],
                    options=dict(opts),
                    authorized=tuple(auth),
                    adjacent=adj,
                )
            )
        except (KeyError, TypeError) as e:
            raise FixtureError(f"fixtures: scenario #{i} malformed: {e!r}") from e
        except UnicodeEncodeError as e:
            raise FixtureError(f"fixtures: scenario #{i} must be ASCII-only: {e}") from e
    offset = doc.get("default_cursor_offset", 24)
    if not isinstance(offset, int) or offset < 0:
        raise FixtureError("fixtures: default_cursor_offset must be a non-negative integer")
    return Fixtures(scenarios=tuple(out), default_cursor_offset=offset, sha256=sha, path=path)


# --------------------------------------------------------------------------------------
# Arena
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SlotRange:
    name: str
    start: int
    end: int


@dataclass(frozen=True)
class Geometry:
    """Exact displacement of the read cursor past slot 3's boundary."""

    unit: str
    requested_offset: int
    boundary: int          # byte index of slot 3's end (== start of the adjacent slot)
    read_end: int          # byte index where request 3's read actually stops
    bleed_bytes: int
    bleed_words: int
    clamped: bool          # the adjacent slot was shorter than the requested offset

    def to_json(self) -> dict:
        return dict(self.__dict__)


class Arena:
    """Slots packed contiguously in one bytearray; reads go through a memoryview."""

    VICTIM = 2  # zero-based index of slot #3

    def __init__(self, named_slots: list[tuple[str, str]]):
        if len(named_slots) != 4:
            raise FixtureError("arena needs exactly 4 slots: 3 authorized tenants + 1 adjacent")
        encoded = [(n, (t if t.endswith("\n") else t + "\n").encode("ascii")) for n, t in named_slots]
        self.buf = bytearray(sum(len(b) for _, b in encoded))
        self.view = memoryview(self.buf)
        self.slots: list[SlotRange] = []
        pos = 0
        for name, data in encoded:
            self.view[pos : pos + len(data)] = data
            self.slots.append(SlotRange(name, pos, pos + len(data)))
            pos += len(data)

    # -- reads ---------------------------------------------------------------------------
    def read(self, start: int, end: int) -> bytes:
        return bytes(self.view[start:end])

    def geometry(self, cursor_offset: int, unit: str = "words") -> Geometry:
        if cursor_offset < 0:
            raise ValueError("cursor_offset must be >= 0")
        if unit not in ("words", "bytes"):
            raise ValueError("unit must be 'words' or 'bytes'")
        victim = self.slots[self.VICTIM]
        adjacent = self.slots[self.VICTIM + 1]
        boundary = victim.end
        if cursor_offset == 0:
            return Geometry(unit, 0, boundary, boundary, 0, 0, False)
        limit = adjacent.end
        if unit == "bytes":
            read_end = min(boundary + cursor_offset, limit)
            clamped = boundary + cursor_offset > limit
        else:
            region = bytes(self.view[boundary:limit])
            ends = [m.end() for m in _WORD.finditer(region)]
            clamped = len(ends) < cursor_offset
            read_end = boundary + (ends[min(cursor_offset, len(ends)) - 1] if ends else 0)
        words = len(_WORD.findall(bytes(self.view[boundary:read_end])))
        return Geometry(unit, cursor_offset, boundary, read_end, read_end - boundary, words, clamped)

    def dispose(self) -> None:
        """Zero and release the arena (collector-level disposal; not kernel RAM scrubbing)."""
        self.view.release()
        self.buf[:] = bytes(len(self.buf))
        self.buf.clear()
        self.slots = []

    def request_view(self, index: int, cursor_offset: int = 0, unit: str = "words") -> bytes:
        """What request ``index`` (0-based) sees. Only the victim slot overreads."""
        slot = self.slots[index]
        end = slot.end
        if index == self.VICTIM:
            end = self.geometry(cursor_offset, unit).read_end
        return self.read(slot.start, end)


# --------------------------------------------------------------------------------------
# Neutral filler (matched negative control)
# --------------------------------------------------------------------------------------
_NEUTRAL = {
    1: ["x", "y", "z"],
    2: ["of", "to", "in", "on", "at", "by", "it", "is", "up", "so", "we", "he", "me", "us", "go", "do"],
    3: ["the", "and", "for", "are", "not", "but", "all", "any", "can", "day", "end", "few", "get", "how",
        "key", "low", "map", "new", "old", "one", "out", "set", "two", "use", "way", "who", "yes"],
    4: ["that", "with", "from", "this", "they", "have", "been", "were", "when", "what", "some", "time",
        "more", "very", "over", "also", "only", "like", "name", "list", "part", "view", "line", "then",
        "than", "each", "many", "most", "same", "case"],
    5: ["where", "which", "their", "there", "about", "other", "these", "those", "first", "after", "under",
        "table", "sheet", "plain", "stone", "river", "paper", "cloud", "light", "green", "white", "small",
        "large", "early", "field"],
    6: ["people", "before", "around", "number", "should", "little", "window", "garden", "simple", "bridge",
        "market", "season", "common", "during", "across", "normal"],
    7: ["between", "through", "another", "because", "general", "example", "picture", "morning", "evening",
        "country", "quarter", "project", "history", "company", "pattern"],
    8: ["together", "standard", "children", "building", "possible", "national", "research", "language",
        "everyone", "overhead", "interior", "magazine"],
    9: ["important", "different", "following", "sometimes", "wonderful", "available", "community"],
    10: ["understand", "everything", "throughout", "population", "particular", "difference"],
    11: ["development", "environment", "interesting", "opportunity", "traditional", "performance",
         "association", "information"],
    12: ["organization", "intelligence", "professional", "particularly", "relationship", "unfortunately"],
    13: ["international", "significantly", "representative", "understanding"],
}
_MAX_DIRECT = max(_NEUTRAL)


def _neutral_candidates(length: int) -> list[str]:
    """All neutral cores of exactly ``length`` characters (deterministic order)."""
    if length in _NEUTRAL:
        return list(_NEUTRAL[length])
    # Compose hyphenated cores: word + "-" + (shorter core), e.g. 13 -> "population-the".
    first = min(_MAX_DIRECT, length - 2)
    if first < 1:
        return ["x"] * 1 if length == 1 else []
    out = []
    for w in _NEUTRAL[first]:
        for rest in _neutral_candidates(length - first - 1)[:3]:
            out.append(f"{w}-{rest}")
    return out


_SPLIT = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)

#: Identifies the filler-construction algorithm in provenance records (bump if make_neutral_filler changes).
FILLER_ALGORITHM = "neutral-filler/2"


def _shape_preserving(core: str, n: int, words: bool = False) -> str:
    """Same length and character classes as ``core``; every digit/letter is replaced by a different one.

    With ``words=True`` an alphabetic run is replaced by a neutral word of the same length when one exists
    (so ``24-hour`` -> ``17-over``-like), instead of letter-shifted nonsense.
    """
    out = []
    pos = 0
    for idx, run in enumerate(re.findall(r"\d+|[A-Za-z]+|[^A-Za-z\d]+", core)):
        if run[0].isdigit():
            for ch in run:
                out.append(str((int(ch) + 1 + (pos * 3 + n) % 9) % 10))
                pos += 1
        elif run[0].isalpha():
            pool = _NEUTRAL.get(len(run))
            if words and pool:
                out.append(pool[(n + idx * 5) % len(pool)])
            else:
                for ch in run:
                    base = ord("A") if ch.isupper() else ord("a")
                    out.append(chr(base + (ord(ch) - base + 1 + (pos * 3 + n) % 9 + 2) % 26))
                    pos += 1
                continue
            pos += len(run)
        else:
            out.append(run)
            pos += len(run)
    return "".join(out)


def _hybrid(length: int, n: int) -> str:
    """A neutral word plus a short nonsense tail, exactly ``length`` letters: reaches token counts between
    a plain word (1 token) and pure nonsense (many tokens)."""
    sizes = [a for a in _NEUTRAL if 2 <= a <= length - 1]
    if not sizes:
        return _gibberish(length, n)
    a = sizes[n % len(sizes)]
    pool = _NEUTRAL[a]
    return pool[(n * 3) % len(pool)] + _gibberish(length - a, n + a)


def _mirror(core: str, n: int) -> str:
    """Keep internal punctuation (hyphens etc.) where it is; swap each alphabetic run for a neutral word
    of the same length, so compound words like ``on-demand`` keep their token structure."""
    out = []
    for idx, run in enumerate(re.findall(r"[A-Za-z]+|[^A-Za-z]+", core)):
        if run[0].isalpha():
            pool = _NEUTRAL.get(len(run))
            out.append(pool[(n + idx * 5) % len(pool)] if pool else _gibberish(len(run), n + idx))
        else:
            out.append(run)
    return "".join(out)


_CONS, _VOWS = "bdfgklmnprstvz", "aeiou"


def _gibberish(length: int, n: int) -> str:
    """Pronounceable nonsense of exactly ``length`` lowercase letters (never a standalone A-D letter)."""
    if length == 1:
        return "xyz"[n % 3]
    return "".join(
        (_CONS[(n * 5 + i * 3) % len(_CONS)] if i % 2 == 0 else _VOWS[(n + i) % len(_VOWS)])
        for i in range(length)
    )


def make_neutral_filler(text: str, count_tokens=None, salt: int = 0) -> str:
    """Return neutral text with *exactly* the same whitespace layout and per-word byte lengths as ``text``.

    Punctuation at word edges is preserved; each word's core is swapped for neutral vocabulary of the
    identical character length (so byte length and word count match at *every* prefix, which makes the
    control match the treatment for any ``cursor_offset``). When ``count_tokens`` (str -> int) is given,
    candidates are additionally chosen to match the original word's token count, so the control also
    matches in model tokens.
    """
    forbidden = {m.group(2).lower() for w in _STR_WORD.findall(text) if (m := _SPLIT.match(w))}
    counter = [0]

    def cased(cand: str, core: str) -> str:
        return cand[0].upper() + cand[1:] if core[0].isupper() and len(cand) > 1 else cand

    def swap(match: re.Match) -> str:
        word = match.group(0)
        lead, core, trail = _SPLIT.match(word).groups()
        if not core:
            return word
        counter[0] += 1
        n = counter[0] * 7 + len(core) + salt * 13
        if any(ch.isdigit() for ch in core):
            # Numbers / codes: keep the *shape* (digit->different digit, letter->different letter,
            # punctuation kept) so digit-splitting tokenizers see the same number of tokens.
            ordered = [_shape_preserving(core, n + v, words=True) for v in range(6)] \
                + [_shape_preserving(core, n + v) for v in range(6)]
        else:
            cands = [c for c in _neutral_candidates(len(core)) if c.lower() not in forbidden] or \
                _neutral_candidates(len(core))
            k = n % len(cands)
            ordered = cands[k:] + cands[:k]
            if re.search(r"[^A-Za-z]", core):  # compound word: mirror its internal punctuation first
                ordered = [_mirror(core, n + v) for v in range(6)] + ordered
            if count_tokens is not None:
                # Pronounceable nonsense of the same length widens the reachable token counts.
                ordered = ordered + [_hybrid(len(core), n + v) for v in range(12)] \
                    + [_gibberish(len(core), n + v) for v in range(10)]
        pick = ordered[0]
        if count_tokens is not None:
            # Tokenise each word with the same left context it has in the arena (usually a space).
            left = text[match.start() - 1] if match.start() > 0 else "\n"
            target = count_tokens(left + word)
            best = None
            for cand in ordered:
                diff = abs(count_tokens(left + lead + cased(cand, core) + trail) - target)
                if best is None or diff < best:
                    best, pick = diff, cand
                if diff == 0:
                    break
        pick = cased(pick, core)
        assert len(pick) == len(core)
        return lead + pick + trail

    out = _STR_WORD.sub(swap, text)
    assert len(out) == len(text) and out.split() and len(out.split()) == len(text.split())
    return out


# --------------------------------------------------------------------------------------
# Prompt assembly
# --------------------------------------------------------------------------------------
def render_prompt(scenario: Scenario, evidence: bytes) -> bytearray:
    """Render into a *mutable* buffer so the harness can zero it after use."""
    head, tail = PROMPT_TEMPLATE.split("{evidence}")
    head = head.format(question=scenario.question, **scenario.options)
    return bytearray(head.encode("ascii") + evidence + tail.encode("ascii"))


def scrub(buf: bytearray) -> None:
    """Zero a harness-owned buffer in place (collector-level disposal, NOT kernel RAM scrubbing)."""
    buf[:] = bytes(len(buf))
    buf.clear()


@dataclass
class RequestPlan:
    """The three authorized requests for one scenario in one mode. Holds prompt buffers (mutable)."""

    mode: str                      # "clean" | "treatment" | "control"
    cursor_offset: int
    geometry: Geometry
    prompts: list                  # 3 x bytearray
    bleed_fragment: bytes          # the bytes that crossed the boundary (b"" for clean)
    arena: Arena

    def dispose(self) -> None:
        """Zero every prompt buffer and the arena, and drop the fragment (collector-level only)."""
        for buf in self.prompts:
            scrub(buf)
        self.prompts.clear()
        self.arena.dispose()
        self.bleed_fragment = b""


def plan_requests(
    scenario: Scenario,
    mode: str,
    cursor_offset: int = 0,
    unit: str = "words",
    filler: str | None = None,
) -> RequestPlan:
    if mode not in ("clean", "treatment", "control"):
        raise ValueError(mode)
    if mode == "control":
        adjacent = filler if filler is not None else make_neutral_filler(scenario.adjacent)
        name = "neutral-filler"
    else:
        adjacent = scenario.adjacent
        name = "tenant-B-adjacent"
    arena = Arena(
        [("tenant-1", scenario.authorized[0]), ("tenant-2", scenario.authorized[1]),
         ("tenant-3", scenario.authorized[2]), (name, adjacent)]
    )
    offset = 0 if mode == "clean" else cursor_offset
    geom = arena.geometry(offset, unit)
    prompts = [
        render_prompt(scenario, arena.request_view(i, offset if i == Arena.VICTIM else 0, unit))
        for i in range(3)
    ]
    fragment = arena.read(geom.boundary, geom.read_end)
    return RequestPlan(mode, offset, geom, prompts, fragment, arena)
