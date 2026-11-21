"""``bleediff track``: an interactive decision-margin tracker (v0.3 feature, exploratory).

You edit a text fragment in any editor, press Enter, and see how the model's A/B/C/D answer probabilities and the A/B
decision margin changed. Each measurement is ONE prompt: request #3 of a frozen scenario (its authorized context with the
fragment appended after it, at the same fixed insertion point the displacement experiments used).

MEASUREMENT TYPE (do not confuse it with the research fold)
    ``single_prompt_pairwise_margin``: margin_AB = ln(p_A / p_B) of one prompt. It is NOT the three-request fold
    M = (p1 + p2 + p3) / 3 and NOT delta_contrast of ``calibrate`` / ``bounce`` / ``diff``; those equations are untouched.

The probabilities come from the SAME validated extraction every other Bleediff command uses
(``LlamaClient.choice_probs``: pre-sampling log-probabilities renormalised over the four choice tokens, tokenization
verified first, missing or collapsed distributions refused). Nothing is imputed: a missing or non-finite probability is an
error, never a zero. The tracker adds no grammar, mask or constraint of its own.

The overall preferred answer is the highest-probability choice among A-D. A pairwise A/B crossing is reported as a
crossing; it is an overall answer flip only when the arg-max choice itself changes.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field

from . import CHOICES, UNKNOWN, __version__
from .arena import PROMPT_TEMPLATE, Scenario, load_fixtures, render_prompt
from .client import LlamaClient
from .errors import Inconclusive
from .fold import argmax_choice
from .meta import code_revision

HISTORY_SCHEMA = "bleediff.track-history/1"
MEASUREMENT_TYPE = "single_prompt_pairwise_margin"
MEASUREMENT_NOTE = (
    "One prompt per measurement: request #3 of the scenario, the clean authorized context (baseline) or that context with "
    "the fragment appended after it. margin_AB = ln(p_A/p_B). This is NOT the three-request fold used by calibrate/bounce/diff "
    "and its values are not delta_contrast.")
DEFAULT_FIXTURES = os.path.join("fixtures", "semantic-v2", "contexts.json")
DEFAULT_SCENARIO = "config-audit-v2"
TOLERANCE = 1e-9          # margins closer than this are reported as unchanged
MAX_FRAGMENT_CHARS = 20000


class MeasurementError(Inconclusive):
    reason = "INVALID_MEASUREMENT"


class FragmentError(Inconclusive):
    reason = "FRAGMENT_UNREADABLE"


# --------------------------------------------------------------------------------------
# pure measurement logic
# --------------------------------------------------------------------------------------
def validate_probs(probs) -> list[str]:
    """Problems with a four-choice probability vector. Empty list = usable for the A/B margin."""
    if not isinstance(probs, (list, tuple)) or len(probs) != len(CHOICES):
        return ["expected four probabilities (A, B, C, D)"]
    problems = []
    for c, p in zip(CHOICES, probs):
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p):
            problems.append(f"p_{c} is missing or not finite")
        elif p < 0:
            problems.append(f"p_{c} is negative")
    if not problems and abs(sum(probs) - 1.0) > 1e-6:
        problems.append("the four probabilities do not sum to 1")
    if not problems and (probs[0] <= 0 or probs[1] <= 0):
        problems.append("p_A or p_B is not positive, so the A/B margin is undefined")
    return problems


def margin_ab(probs) -> float:
    """Signed pairwise margin ln(p_A / p_B). Positive favours A over B, negative favours B over A."""
    problems = validate_probs(probs)
    if problems:
        raise MeasurementError("; ".join(problems))
    return math.log(probs[0]) - math.log(probs[1])


def pairwise_order(margin: float) -> str:
    return "A_over_B" if margin > 0 else ("B_over_A" if margin < 0 else "A_equals_B")


def pairwise_crossed(prev_margin: float, cur_margin: float) -> bool:
    """True when A and B swapped order (strictly positive <-> strictly negative). Not the same as an answer flip."""
    return prev_margin * cur_margin < 0


def compare(prev: dict | None, base: dict, cur: dict) -> dict:
    """Changes of ``cur`` relative to the baseline and to the previous successful measurement (dicts with margin, preferred)."""
    out = {"change_from_baseline": cur["margin"] - base["margin"],
           "choice_flipped_from_baseline": cur["preferred"] != base["preferred"],
           "pairwise_crossed_from_baseline": pairwise_crossed(base["margin"], cur["margin"])}
    if prev is None:
        out.update(change_from_previous=None, choice_flipped_from_previous=None, pairwise_crossed_from_previous=None)
    else:
        out.update(change_from_previous=cur["margin"] - prev["margin"],
                   choice_flipped_from_previous=cur["preferred"] != prev["preferred"],
                   pairwise_crossed_from_previous=pairwise_crossed(prev["margin"], cur["margin"]))
    return out


def describe_margin_change(prev: float, cur: float, *, preferred: str | None = None, prev_preferred: str | None = None) -> str:
    """Plain description of how the A/B margin moved. No confidence labels."""
    if abs(cur - prev) <= TOLERANCE:
        return "A/B margin unchanged"
    if pairwise_crossed(prev, cur):
        text = "A/B order crossed: " + ("B now above A" if cur < 0 else "A now above B")
        if preferred is not None and prev_preferred is not None:
            if preferred == prev_preferred:
                text += f" (overall preferred answer unchanged: {preferred})"
            else:
                text += f" (overall preferred answer changed: {prev_preferred} -> {preferred})"
        return text
    if cur == 0:
        return "A and B are now tied"
    if prev == 0:
        return "A/B margin opened toward " + ("A" if cur > 0 else "B")
    return "A/B margin widened" if abs(cur) > abs(prev) else "A/B margin narrowed"


# --------------------------------------------------------------------------------------
# prompt construction (the SAME renderer and insertion point as the displacement experiments)
# --------------------------------------------------------------------------------------
def _slot3_bytes(sc: Scenario) -> bytes:
    text = sc.authorized[2]
    return (text if text.endswith("\n") else text + "\n").encode("ascii")


def build_prompt(sc: Scenario, fragment: str | None) -> str:
    """Request #3's prompt. ``fragment=None`` is the clean baseline (identical to ``plan_requests(clean)``); a fragment is
    appended directly after the authorized context's terminating newline, exactly where the displaced words landed."""
    evidence = _slot3_bytes(sc)
    if fragment:
        evidence += fragment.encode("utf-8")
    return bytes(render_prompt(sc, evidence)).decode("utf-8")


def read_fragment(path: str) -> str:
    """UTF-8 text of the fragment file, newlines normalised to LF and surrounding whitespace stripped (so hashes do not
    depend on the editor's line endings or a trailing newline)."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as e:
        raise FragmentError(f"cannot read the fragment file {path!r}: {e}") from e
    if b"\x00" in raw:
        raise FragmentError(f"{path!r} looks like a binary file (NUL bytes)")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise FragmentError(f"{path!r} is not valid UTF-8: {e}") from e
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) > MAX_FRAGMENT_CHARS:
        raise FragmentError(f"the fragment is {len(text)} characters; the limit is {MAX_FRAGMENT_CHARS}")
    return text


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# --------------------------------------------------------------------------------------
# terminal rendering
# --------------------------------------------------------------------------------------
def _bar_glyphs(stream) -> tuple[str, str]:
    enc = getattr(stream, "encoding", None) or "ascii"
    try:
        "█░".encode(enc)
        return "█", "░"
    except (UnicodeEncodeError, LookupError):
        return "#", "."


def format_probs(probs: list[float], stream=None, width: int = 24) -> list[str]:
    full, empty = _bar_glyphs(stream if stream is not None else sys.stdout)
    lines = []
    for c, p in zip(CHOICES, probs):
        n = max(0, min(width, round(p * width)))
        lines.append(f"  {c}  {full * n}{empty * (width - n)} {p * 100:5.1f}%")
    return lines


def _signed(x: float) -> str:
    return f"{x:+.3f}"


# --------------------------------------------------------------------------------------
# the session
# --------------------------------------------------------------------------------------
@dataclass
class TrackSession:
    client: LlamaClient
    scenario: Scenario
    scenario_info: dict
    out_dir: str
    fragment_path: str
    model_name: str
    model_info: dict
    retain_prompts: bool = False
    out: object = None                       # text stream for the terminal view
    steps: int = 0
    edits: int = 0
    baseline: dict | None = None
    previous: dict | None = None
    ok_measurements: list = field(default_factory=list)
    token_counter: object = None             # batch mode: callable(prompt, fragment) -> dict of tokenizer counts
    role_labels: object = None               # batch mode, roles not at A/B: (correct_letter, competing_letter) so the baseline print names its margins
    _history: object = None

    # -- persistence -----------------------------------------------------------------
    def _write(self, record: dict) -> None:
        self._history.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._history.flush()

    def _print(self, *lines: str) -> None:
        for ln in lines:
            print(ln, file=self.out)

    def open_history(self) -> None:
        self._history = open(os.path.join(self.out_dir, "history.jsonl"), "x", encoding="utf-8", newline="\n")

    def close(self, reason: str) -> None:
        if self._history is not None and not self._history.closed:
            self._write({"type": "session_end", "timestamp_utc": _utc(), "steps": self.steps, "successful_measurements": len(self.ok_measurements),
                         "reason": reason})
            self._history.close()

    def write_start(self, info, report) -> None:
        payload = self.client.completion_payload("")
        grammar = payload.pop("grammar", "")
        payload.pop("prompt", None)
        rev = code_revision()
        self._write({
            "type": "session_start", "schema": HISTORY_SCHEMA, "timestamp_utc": _utc(), "bleediff_version": __version__,
            "code_revision": rev, "measurement_type": MEASUREMENT_TYPE, "measurement_note": MEASUREMENT_NOTE,
            "margin_definition": "margin_AB = ln(p_A / p_B); positive favours A over B; reference labels fixed for the whole session",
            "choices": list(CHOICES), "scenario": self.scenario_info, "insertion_point": "appended after the authorized context of request #3",
            "prompt_template_sha256": _sha(PROMPT_TEMPLATE),
            "model": {**self.model_info, "model_hash": info.model_hash, "model_hash_basis": info.model_hash_basis, "build_info": info.build_info or UNKNOWN},
            "server_flags": info.public_flags(),
            "inference": {"n_probs": self.client.n_probs, "request_body_without_prompt_and_grammar": payload,
                          "grammar_sha256": _sha(grammar), "note": "the unchanged Bleediff request: pre-sampling logprobs, no extra constraint"},
            "tokenization": report.to_json(),
            "prompt_retention": "explicit local files (--retain-prompts)" if self.retain_prompts else "off: only hashes are recorded",
        })

    # -- measuring -------------------------------------------------------------------
    def _measure_prompt(self, label: str, fragment: str | None, *, forced: bool = False, extra: dict | None = None) -> dict | None:
        """Score one prompt and record it. Returns the measurement dict, or an Inconclusive error when it failed (recorded).

        ``extra`` (batch mode) adds fields such as the candidate label to the record; ``token_counter`` (batch mode) adds
        tokenizer counts. Neither touches the probabilities or the margin."""
        self.steps += 1
        step = self.steps
        prompt = build_prompt(self.scenario, fragment)
        base_rec = {"type": "measurement", "step": step, "label": label, "timestamp_utc": _utc(),
                    "fragment": {"present": bool(fragment), "sha256": _sha(fragment) if fragment else None,
                                 "chars": len(fragment) if fragment else 0, "bytes": len(fragment.encode("utf-8")) if fragment else 0,
                                 "words": len(fragment.split()) if fragment else 0},
                    "prompt": {"sha256": _sha(prompt), "chars": len(prompt)}, "forced_repeat": forced, **(extra or {})}
        if self.retain_prompts:
            os.makedirs(os.path.join(self.out_dir, "prompts"), exist_ok=True)
            with open(os.path.join(self.out_dir, "prompts", f"step-{step:02d}.txt"), "x", encoding="utf-8", newline="\n") as fh:
                fh.write(prompt)
            if fragment:
                with open(os.path.join(self.out_dir, "prompts", f"step-{step:02d}.fragment.txt"), "x", encoding="utf-8", newline="\n") as fh:
                    fh.write(fragment)
        try:
            if self.token_counter is not None:
                base_rec["tokens"] = self.token_counter(prompt, fragment)
            probs = self.client.choice_probs(prompt)
            coverage = {"value": self.client.last_coverage, "basis": self.client.last_coverage_basis}
            n_probs_used = self.client.last_n_probs_used
            margin = margin_ab(probs)                       # validates: finite, non-negative, sums to 1, A and B positive
        except Inconclusive as e:
            self._write({**base_rec, "status": "error", "error": {"reason": getattr(e, "reason", type(e).__name__), "message": str(e)[:500]}})
            return e                                         # type: ignore[return-value]
        preferred = argmax_choice(probs)
        cur = {"step": step, "label": label, "probs": list(probs), "margin": margin, "preferred": preferred,
               "fragment_sha": base_rec["fragment"]["sha256"], "prompt_sha": base_rec["prompt"]["sha256"], "tokens": base_rec.get("tokens")}
        rec = {**base_rec, "status": "ok", "error": None, "probabilities": dict(zip(CHOICES, probs)), "coverage": coverage,
               "n_probs_used": n_probs_used, "margin_AB": margin, "pairwise_order": pairwise_order(margin), "preferred_choice": preferred}
        if self.baseline is not None:
            rec.update(compare(self.previous, self.baseline, cur))
            rec["edit"] = self.edits
        else:
            rec.update(change_from_baseline=None, change_from_previous=None, choice_flipped_from_baseline=None,
                       choice_flipped_from_previous=None, pairwise_crossed_from_baseline=None, pairwise_crossed_from_previous=None)
        same = next((m for m in reversed(self.ok_measurements) if m["prompt_sha"] == cur["prompt_sha"]), None)
        rec["identical_prompt_as_step"] = same["step"] if same else None
        rec["probabilities_identical_to_that_step"] = (same["probs"] == cur["probs"]) if same else None
        self._write(rec)
        cur["identical_prompt_as"] = same
        self.ok_measurements.append(cur)
        return cur

    def start(self, info, report) -> dict | Exception:
        """Write the session header and measure the clean authorized baseline."""
        self.write_start(info, report)
        result = self._measure_prompt("baseline", None)
        if isinstance(result, Exception):
            return result
        self.baseline = result
        self.previous = result
        margin_lines = [f"Margin A/B: {_signed(result['margin'])} nats"]
        if self.role_labels is not None:                 # batch mode with answers at other letters: say which margin is which
            from .roles import role_scores
            correct, competing = self.role_labels
            margin_lines = [f"Margin A/B (letters A and B only; NOT the role margin): {_signed(result['margin'])} nats"]
            try:
                rm = role_scores(result["probs"], correct, competing)["margin_correct_vs_competing"]
                margin_lines.append(f"Role margin ln(p_{correct} / p_{competing}) (correct {correct} vs competing {competing}): {_signed(rm)} nats")
            except MeasurementError:
                margin_lines.append(f"Role margin (correct {correct} vs competing {competing}): undefined for this baseline")
        self._print("", "BASELINE", *format_probs(result["probs"], self.out), *margin_lines, f"Preferred answer: {result['preferred']}")
        return result

    def rescore(self, *, force: bool = False) -> dict | Exception | None:
        """Re-read the fragment file and measure it. Returns the measurement, an error, or None when nothing was scored."""
        try:
            fragment = read_fragment(self.fragment_path)
        except FragmentError as e:
            self.steps += 1
            self._write({"type": "measurement", "step": self.steps, "label": "edit", "timestamp_utc": _utc(), "status": "error",
                         "error": {"reason": e.reason, "message": str(e)[:500]}})
            return e
        sha = _sha(fragment) if fragment else None
        last = self.ok_measurements[-1]
        if not fragment and not force:
            self._print("The fragment file is empty (the same prompt as the baseline). Edit and save it, then press Enter.")
            self._write({"type": "event", "event": "fragment_empty_not_rescored", "timestamp_utc": _utc(), "step_after": self.steps})
            return None
        if sha == last["fragment_sha"] and not force:
            self._print(f"The fragment is unchanged since step {last['step']}; not rescored. Edit it, or type 'again' to measure it again.")
            self._write({"type": "event", "event": "fragment_unchanged_not_rescored", "timestamp_utc": _utc(), "unchanged_since_step": last["step"]})
            return None
        self.edits += 1
        label = "repeat" if (sha == last["fragment_sha"]) else "edit"
        result = self._measure_prompt(label, fragment or None, forced=force)
        if isinstance(result, Exception):
            self.edits -= 1
            return result
        cmp = compare(self.previous, self.baseline, result)
        lines = ["", f"EDIT {self.edits:02d}" + ("  (repeat of the same prompt)" if label == "repeat" else ""), *format_probs(result["probs"], self.out),
                 f"Margin A/B: {_signed(result['margin'])} nats",
                 f"Change from baseline: {_signed(cmp['change_from_baseline'])} nats",
                 f"Change from previous: {_signed(cmp['change_from_previous'])} nats"]
        prev_pref = self.previous["preferred"]
        if cmp["choice_flipped_from_previous"]:
            pref = f"Preferred answer: {result['preferred']} (changed from {prev_pref})"
        elif cmp["choice_flipped_from_baseline"]:
            pref = f"Preferred answer: {result['preferred']} (unchanged since previous; baseline was {self.baseline['preferred']})"
        else:
            pref = f"Preferred answer: {result['preferred']} (unchanged)"
        lines += [pref, "Status: " + describe_margin_change(self.previous["margin"], result["margin"],
                                                           preferred=result["preferred"], prev_preferred=prev_pref)]
        same = result.get("identical_prompt_as")
        if same is not None:
            if same["probs"] == result["probs"]:
                lines.append(f"Note: this exact prompt was already measured at step {same['step']}; the probabilities are identical.")
            else:
                lines.append(f"WARNING: this exact prompt was measured at step {same['step']} with DIFFERENT probabilities "
                             "(the server is not deterministic for identical prompts).")
        self._print(*lines)
        self.previous = result
        return result


def _model_name(info) -> str:
    name = info.model_alias or (os.path.basename(info.model_path) if info.model_path else "unknown-model")
    name = os.path.basename(str(name).replace("\\", "/"))
    return name[:-5] if name.lower().endswith(".gguf") else name


def run_interactive(session: TrackSession, input_fn=None) -> str:
    """The Enter-to-rescore loop. Returns why it ended."""
    input_fn = input_fn or input
    session._print("", f"Edit this file in any editor and save it: {session.fragment_path}")
    reason = "quit"
    while True:
        try:
            cmd = input_fn("Press Enter to rescore the edited fragment, or type quit. ")
        except EOFError:
            reason = "end_of_input"
            break
        except KeyboardInterrupt:
            reason = "interrupted"
            break
        c = cmd.strip().lower()
        if c in ("quit", "q", "exit"):
            break
        if c not in ("", "again", "r", "repeat"):
            session._print(f"Unknown command {cmd.strip()!r}. Press Enter to rescore, 'again' to remeasure the same text, or 'quit'.")
            continue
        result = session.rescore(force=c in ("again", "r", "repeat"))
        if isinstance(result, Exception):
            session._print(f"ERROR [{getattr(result, 'reason', type(result).__name__)}]: {result}",
                           "Nothing was recorded as a measurement for this attempt. Fix the cause and press Enter to try again.")
    return reason


def cmd_track(args, *, input_fn=None) -> int:
    """Entry point used by ``bleediff track``. Errors are printed clearly and return exit code 2."""
    if getattr(args, "arms", None):          # batch mode: score a list of candidate fragments (see bleediff/batch.py)
        from .batch import cmd_batch
        return cmd_batch(args)
    out = sys.stdout
    input_fn = input_fn or input             # looked up at call time so the builtin can be replaced in tests
    try:
        fixtures = load_fixtures(args.fixtures)
        scenarios = {s.id: s for s in fixtures.scenarios}
        if args.scenario not in scenarios:
            raise Inconclusive(f"unknown scenario {args.scenario!r}; available: {sorted(scenarios)}", reason="BAD_ARGS")
        sc = scenarios[args.scenario]
        out_dir = args.out
        if os.path.exists(out_dir):
            raise Inconclusive(f"{out_dir!r} already exists: choose a new output directory (sessions are never overwritten)",
                               reason="OUTPUT_EXISTS")
        client = LlamaClient(args.server, timeout=args.timeout, n_probs=args.n_probs)
        report = client.verify_tokenization()        # nothing is created on disk until the server and tokenizer are usable
        info = client.server_info()
        os.makedirs(out_dir)
        fragment_path = args.fragment or os.path.join(out_dir, "fragment.txt")
        if not os.path.exists(fragment_path):
            if not os.path.isdir(os.path.dirname(os.path.abspath(fragment_path))):
                raise Inconclusive(f"the directory of {fragment_path!r} does not exist", reason="BAD_ARGS")
            with open(fragment_path, "x", encoding="utf-8", newline="\n"):
                pass                                  # an empty file for the user to edit; an existing user file is never touched
        name = _model_name(info)
        session = TrackSession(client=client, scenario=sc, out_dir=out_dir, fragment_path=fragment_path, model_name=name,
                               scenario_info={"id": sc.id, "title": sc.title, "fixtures_sha256": fixtures.sha256,
                                              "fixtures_file": os.path.basename(args.fixtures)},
                               model_info={"name": name}, retain_prompts=args.retain_prompts, out=out)
        session.open_history()
        print("BLEEDIFF TRACK", file=out)
        print(f"Model: {name}", file=out)
        print(f"Scenario: {sc.id}", file=out)
        print("Measurement: single prompt (request #3), margin A/B = ln(p_A / p_B); not the three-request fold.", file=out)
        print(f"Session history: {os.path.join(out_dir, 'history.jsonl')}", file=out)
        try:
            first = session.start(info, report)
            if isinstance(first, Exception):
                print(f"ERROR [{getattr(first, 'reason', type(first).__name__)}]: {first}", file=out)
                session.close("baseline_failed")
                return 2
            reason = run_interactive(session, input_fn)
        finally:
            session.close(locals().get("reason", "error"))
        print("", file=out)
        print(f"Session saved: {os.path.join(out_dir, 'history.jsonl')} ({session.steps} step(s)).", file=out)
        return 0
    except Inconclusive as e:
        print(f"ERROR [{e.reason}]: {e}", file=out)
        return 2
