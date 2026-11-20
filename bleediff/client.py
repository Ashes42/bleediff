"""llama-server client: tokenizer verification + unwarped 4-choice probability extraction.

THE PROBABILITY TRAP
--------------------
With greedy sampling (temperature=0) and ``post_sampling_probs=true`` llama-server reports the
distribution *after* the sampler chain, which collapses to ``{A: 1.0}`` (zero-probability tokens are
filtered out of the response). Sub-threshold mass on ``B`` would be destroyed before we ever see it.

This client therefore asks for the *pre-sampling* distribution (``post_sampling_probs=false``,
reported as ``logprob`` / ``top_logprobs``: a softmax over the full vocabulary of the raw logits,
see ``get_token_probabilities`` in llama.cpp ``tools/server/server-common.cpp``), AND additionally
pins the sampler chain to the identity transform (temperature=1.0, top_k=0, top_p=1.0, min_p=0.0,
repeat_penalty=1.0) so that even a server that reports post-sampling numbers reports unwarped ones.
The choice mass is then renormalised over {A, B, C, D} only (after folding leading-space variants
such as " A" into the same choice), so the vector is deterministic and noise-free.

Anything that looks collapsed (a choice at exactly zero / -inf / null, or a single-entry
distribution) raises ProbabilityCollapse -> INCONCLUSIVE instead of silently producing a wrong number.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from . import CHOICES
from .errors import (
    Inconclusive,
    MissingProbabilities,
    ProbabilityCollapse,
    ServerUnreachable,
    TokenizationError,
)

DEFAULT_N_PROBS = 32
#: If a choice token is absent from the top-N logprobs, retry with a wider window before failing.
N_PROBS_ESCALATION = (1, 4, 16)  # multipliers of n_probs: 32 -> 128 -> 512

#: llama.cpp serialises a zero probability as the lowest finite float32 rather than null.
_FLOAT_LOWEST = -3.0e38


# --------------------------------------------------------------------------------------
# Tokenization verification
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TokenizationReport:
    #: letter -> tuple of accepted single-token surface forms, e.g. ("A", " A")
    variants: dict[str, tuple[str, ...]]
    #: token id -> letter
    id_to_letter: dict[int, str]
    #: surface string -> letter
    surface_to_letter: dict[str, str]
    #: surface forms that tokenised to more than one token and were therefore excluded
    dropped: tuple[str, ...] = ()

    def grammar(self) -> str:
        """GBNF restricting generation to the accepted single-token choice surfaces."""
        alts = []
        for letter in CHOICES:
            for surface in self.variants[letter]:
                alts.append(json.dumps(surface))
        return "root ::= " + " | ".join(alts) + "\n"

    def to_json(self) -> dict:
        return {
            "variants": {k: list(v) for k, v in self.variants.items()},
            "id_to_letter": {str(k): v for k, v in sorted(self.id_to_letter.items())},
            "dropped": list(self.dropped),
        }


def verify_tokenization_with(tokenize) -> TokenizationReport:
    """Pure verification logic over a ``tokenize(text) -> list[int]`` callable.

    Fail closed (TokenizationError) when:
      * a bare letter does not tokenise to exactly one token,
      * tokenization is non-deterministic across two calls,
      * two different letters share a token id (ambiguous mapping).
    A leading-space variant (" A") that is *not* a single token is simply excluded from the grammar
    and from the probability mass; one that is a single token is folded into its letter.
    """
    variants: dict[str, list[str]] = {}
    id_to_letter: dict[int, str] = {}
    surface_to_letter: dict[str, str] = {}
    dropped: list[str] = []

    for letter in CHOICES:
        accepted: list[str] = []
        for surface in (letter, " " + letter):
            first = tokenize(surface)
            second = tokenize(surface)
            if first != second:
                raise TokenizationError(
                    f"tokenization of {surface!r} is non-deterministic: {first} vs {second}"
                )
            if len(first) != 1:
                if surface == letter:
                    raise TokenizationError(
                        f"choice {letter!r} does not map to exactly one token: {surface!r} -> {first}"
                    )
                dropped.append(surface)
                continue
            tok = first[0]
            owner = id_to_letter.get(tok)
            if owner is not None and owner != letter:
                raise TokenizationError(
                    f"ambiguous tokenization: token id {tok} maps to both {owner!r} and {letter!r}"
                )
            id_to_letter[tok] = letter
            surface_to_letter[surface] = letter
            accepted.append(surface)
        variants[letter] = accepted

    return TokenizationReport(
        variants={k: tuple(v) for k, v in variants.items()},
        id_to_letter=id_to_letter,
        surface_to_letter=surface_to_letter,
        dropped=tuple(dropped),
    )


# --------------------------------------------------------------------------------------
# Probability parsing (pure; unit-testable without a server)
# --------------------------------------------------------------------------------------
def _logsumexp(values: list[float]) -> float:
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def _entry_logprob(item: dict) -> float | None:
    """Return a log-probability from either ``logprob`` (pre-sampling) or ``prob`` (post-sampling)."""
    if "logprob" in item:
        lp = item["logprob"]
        if lp is None:
            return float("-inf")
        return float(lp)
    for key in ("prob", "p"):
        if key in item:
            p = item[key]
            if p is None:
                return float("-inf")
            p = float(p)
            return math.log(p) if p > 0.0 else float("-inf")
    return None


def parse_choice_probs(response: dict, report: TokenizationReport) -> list[float]:
    """Extract the normalised [pA, pB, pC, pD] vector from a ``/completion`` response.

    Raises MissingProbabilities if fields are absent or a choice token is outside the returned
    top-N, and ProbabilityCollapse if the response carries the signature of a collapsed / warped
    distribution.
    """
    comp = response.get("completion_probabilities")
    if not isinstance(comp, list) or not comp:
        raise MissingProbabilities("response has no completion_probabilities (is n_probs honoured?)")
    entry = comp[0]
    if not isinstance(entry, dict):
        raise MissingProbabilities("completion_probabilities[0] is not an object")

    top = None
    for key in ("top_logprobs", "top_probs"):
        if isinstance(entry.get(key), list):
            top = entry[key]
            break
    if top is None and isinstance(entry.get("probs"), list):  # legacy llama.cpp format
        top = [
            {"token": p.get("tok_str"), "prob": p.get("prob")} for p in entry["probs"] if isinstance(p, dict)
        ]
    if top is None:
        raise MissingProbabilities("no top_logprobs / top_probs / probs list in completion_probabilities[0]")

    # Candidate pool = the top list plus the sampled token's own entry (deduplicated).
    pool = list(top)
    if _entry_logprob(entry) is not None and ("id" in entry or "token" in entry):
        pool.append({k: entry[k] for k in ("id", "token", "logprob", "prob") if k in entry})

    # Collapse signature: a post-sampling / greedy response lists a single token.
    distinct = {(it.get("id"), it.get("token")) for it in pool if isinstance(it, dict)}
    if len(distinct) <= 1:
        raise ProbabilityCollapse(
            "response lists a single token: this is the post-sampling / temperature=0 collapse "
            "signature ({A: 1.0, others: 0}); refusing to use it"
        )

    per_letter: dict[str, list[float]] = {c: [] for c in CHOICES}
    seen: set = set()
    for item in pool:
        if not isinstance(item, dict):
            continue
        letter = None
        tok_id = item.get("id")
        if isinstance(tok_id, int) and tok_id in report.id_to_letter:
            letter = report.id_to_letter[tok_id]
        elif tok_id is None and item.get("token") in report.surface_to_letter:
            letter = report.surface_to_letter[item["token"]]
        if letter is None:
            continue
        key = (letter, tok_id if tok_id is not None else item.get("token"))
        if key in seen:
            continue
        seen.add(key)
        lp = _entry_logprob(item)
        if lp is None:
            raise MissingProbabilities(f"choice entry {item!r} carries no logprob/prob")
        if lp == float("-inf") or lp <= _FLOAT_LOWEST or math.isnan(lp):
            raise ProbabilityCollapse(
                f"choice {letter!r} has zero probability (logprob={item.get('logprob', item.get('prob'))!r}); "
                "this is the greedy/post-sampling collapse signature"
            )
        per_letter[letter].append(lp)

    missing = [c for c in CHOICES if not per_letter[c]]
    if missing:
        raise MissingProbabilities(
            f"choice token(s) {missing} absent from the returned top-N probabilities"
        )

    masses = [_logsumexp(per_letter[c]) for c in CHOICES]
    z = _logsumexp(masses)
    probs = [math.exp(m - z) for m in masses]
    return probs


# --------------------------------------------------------------------------------------
# Server metadata + client
# --------------------------------------------------------------------------------------
@dataclass
class ServerInfo:
    base_url: str
    model_path: str | None
    model_alias: str | None
    build_info: str | None
    model_hash: str
    model_hash_basis: str  # "file-header" | "identifier"
    props: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "base_url": self.base_url,
            "model_path": self.model_path,
            "model_alias": self.model_alias,
            "build_info": self.build_info,
            "model_hash": self.model_hash,
            "model_hash_basis": self.model_hash_basis,
        }


def compute_model_hash(model_path: str | None, model_alias: str | None) -> tuple[str, str]:
    """SHA-256 of the GGUF header (first 1 MiB + size) when the file is locally readable, else of the
    model identifier metadata. Returns (hex, basis)."""
    if model_path:
        try:
            if os.path.isfile(model_path):
                h = hashlib.sha256()
                with open(model_path, "rb") as fh:
                    h.update(b"gguf-header\0")
                    h.update(fh.read(1024 * 1024))
                h.update(b"\0size=" + str(os.path.getsize(model_path)).encode())
                return h.hexdigest(), "file-header"
        except OSError:
            pass
    ident = json.dumps({"model_path": model_path, "model_alias": model_alias}, sort_keys=True)
    return hashlib.sha256(b"identifier\0" + ident.encode()).hexdigest(), "identifier"


class LlamaClient:
    def __init__(self, base_url: str, *, timeout: float = 120.0, n_probs: int = DEFAULT_N_PROBS):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.n_probs = n_probs
        self._report: TokenizationReport | None = None
        self.last_n_probs_used: int = n_probs
        self._tok_cache: dict[str, list[int]] = {}

    # -- transport ---------------------------------------------------------------------
    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = self.base_url + path
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
        except urllib.error.HTTPError as e:
            raise Inconclusive(
                f"{method} {path} -> HTTP {e.code}: {e.read()[:300]!r}", reason="SERVER_ERROR"
            ) from e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            raise ServerUnreachable(f"cannot reach llama-server at {self.base_url}: {e}") from e
        try:
            return json.loads(body)
        except ValueError as e:
            raise Inconclusive(f"{method} {path} returned non-JSON", reason="SERVER_ERROR") from e

    # -- tokenization ------------------------------------------------------------------
    def tokenize(self, text: str, *, cache: bool = True) -> list[int]:
        if cache and text in self._tok_cache:
            return list(self._tok_cache[text])
        out = self._request(
            "POST", "/tokenize", {"content": text, "add_special": False, "parse_special": False}
        )
        toks = out.get("tokens")
        if not isinstance(toks, list) or not all(isinstance(t, int) for t in toks):
            raise TokenizationError(f"/tokenize returned an unusable payload: {out!r}"[:300])
        if cache:
            self._tok_cache[text] = list(toks)
        return toks

    def verify_tokenization(self) -> TokenizationReport:
        # Never cache during verification: determinism is exactly what is being tested.
        report = verify_tokenization_with(lambda s: self.tokenize(s, cache=False))
        self._report = report
        return report

    @property
    def report(self) -> TokenizationReport:
        if self._report is None:
            raise TokenizationError("verify_tokenization() must run before any probability request")
        return self._report

    # -- metadata ----------------------------------------------------------------------
    def server_info(self) -> ServerInfo:
        props = self._request("GET", "/props")
        gen = props.get("default_generation_settings") or {}
        path = props.get("model_path") or gen.get("model") or None
        alias = props.get("model_alias") or None
        if not path and not alias:
            try:
                models = self._request("GET", "/v1/models")
                alias = (models.get("data") or [{}])[0].get("id")
            except Inconclusive:
                alias = None
        if not path and not alias:
            raise Inconclusive(
                "server exposes no model identifier (/props model_path / model_alias)",
                reason="MODEL_IDENTITY_UNKNOWN",
            )
        mh, basis = compute_model_hash(path, alias)
        return ServerInfo(
            base_url=self.base_url,
            model_path=path,
            model_alias=alias,
            build_info=props.get("build_info"),
            model_hash=mh,
            model_hash_basis=basis,
            props=props,
        )

    # -- completion --------------------------------------------------------------------
    def completion_payload(self, prompt: str, n_probs: int | None = None) -> dict:
        """The exact, frozen request body. Note: NO temperature=0, NO post_sampling_probs=true."""
        return {
            "prompt": prompt,
            "n_predict": 1,
            "grammar": self.report.grammar(),
            "cache_prompt": False,
            "n_probs": n_probs or self.n_probs,
            "post_sampling_probs": False,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "typical_p": 1.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "mirostat": 0,
            "dry_multiplier": 0.0,
            "seed": 0,
            "stream": False,
        }

    def choice_probs(self, prompt: str) -> list[float]:
        """Return the normalised [pA, pB, pC, pD] vector for ``prompt`` (a fresh mutable list)."""
        last: Inconclusive | None = None
        for mult in N_PROBS_ESCALATION:
            n = self.n_probs * mult
            resp = self._request("POST", "/completion", self.completion_payload(prompt, n))
            try:
                vec = parse_choice_probs(resp, self.report)
            except MissingProbabilities as e:
                last = e
                continue  # widen the top-N window and retry
            self.last_n_probs_used = n
            return vec
        assert last is not None
        raise last
