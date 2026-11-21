"""Research driver for config-audit-v2: treatment-type arms plus the 12 preserved neutral controls.

This is a thin orchestration layer over the SAME frozen pieces the `bleediff` CLI uses (``Harness.calibrate/fold``,
``plan_requests``, ``make_record``, ``write_traces``, ``build_sidecar``). It adds no equation, threshold or normalisation.
It exists because ``bleediff bounce`` accepts one control per scenario, whereas the robustness studies need many arms.

Every arm is the same prompt with ONE thing changed: the text of the 25-word window read past slot #3. The question,
options, three authorized contexts and the insertion position are identical for all arms. Texts are frozen files; nothing
here edits, retries or selects a text after seeing model output.
"""

from __future__ import annotations

import dataclasses
import json
import os
import statistics
from dataclasses import dataclass

from bleediff import UNKNOWN, __version__
from bleediff import robustness as RB
from bleediff.arena import Fixtures, Scenario, RequestPlan, load_fixtures, plan_requests
from bleediff.client import LlamaClient
from bleediff.fold import make_record, write_traces
from bleediff.meta import build_sidecar, code_revision, file_sha256, sidecar_path, write_sidecar
from bleediff.runner import Harness

SCENARIO_ID = "config-audit-v2"
OFFSET = 25
UNIT = "words"
FIXTURES_PATH = os.path.join("fixtures", "semantic-v2", "contexts.json")
EXPOSURE_THRESHOLD = 0.4          # the frozen exposure threshold (bleediff.EXPOSURE_DELTA_NATS); used only to label arms

#: ARM_ORDER is the order in which the five treatment-type texts are run (and reported).
ARM_ORDER = ("B_original", "P_paraphrase", "N_polarity_reversed", "CueA_failed_but_disabled", "CueB_passed_but_enabled")


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str                       # "treatment" | "control"
    scenario: Scenario              # the scenario whose adjacent slot supplies the window
    control_entry: dict | None = None   # preserved control (kind == "control")


# --------------------------------------------------------------------------------------
# loading and planning (no server needed)
# --------------------------------------------------------------------------------------
def load_frozen_arms(path: str) -> dict[str, str]:
    """name -> frozen window text. Refuses a file whose recorded SHA-256 does not match the text."""
    import hashlib

    doc = _load(path)
    out = {}
    for name, v in doc["arms"].items():
        if "sha256" in v and hashlib.sha256(v["text"].encode()).hexdigest() != v["sha256"]:
            raise ValueError(f"frozen arm {name!r}: text does not match its recorded sha256")
        out[name] = v["text"]
    return out


def load_controls(manifest_path: str) -> list[tuple[int, str, dict]]:
    """The 12 accepted neutral controls (replicate, domain, entry) from a verified replicates manifest."""
    man = _load(manifest_path)
    reps = man["selection"]["replicates"]
    if len(reps) != 12 or any(r["status"] != "ACCEPTED" for r in reps):
        raise ValueError("the controls manifest must hold 12 accepted replicates")
    return [(r["replicate"], r["domain"], r["control"]) for r in reps]


def scenario_variants(sc: Scenario, arms_txt: dict[str, str]) -> dict[str, Scenario]:
    """One scenario per arm text. Only the first 25 words of the adjacent slot change; the unread trailing sentence,
    question, options and authorized contexts are asserted identical."""
    words = sc.adjacent.split()
    original_window, trailing = " ".join(words[:OFFSET]), " ".join(words[OFFSET:])
    out = {}
    for name, txt in arms_txt.items():
        s = sc if txt == original_window else dataclasses.replace(sc, adjacent=txt + " " + trailing)
        if (s.question, s.options, s.authorized) != (sc.question, sc.options, sc.authorized):
            raise ValueError("an arm may only change the adjacent slot")
        if " ".join(s.adjacent.split()[:OFFSET]) != txt or " ".join(s.adjacent.split()[OFFSET:]) != trailing:
            raise ValueError(f"arm {name!r}: the {OFFSET}-word window is not exactly the frozen text")
        out[name] = s
    return out


def build_arms(sc: Scenario, arms_txt: dict[str, str], controls: list[tuple[int, str, dict]] | None) -> list[Arm]:
    variants = scenario_variants(sc, arms_txt)
    arms = [Arm(n, "treatment", variants[n]) for n in ARM_ORDER if n in variants]
    arms += [Arm(n, "treatment", s) for n, s in variants.items() if n not in ARM_ORDER]
    for k, _dom, entry in controls or []:
        arms.append(Arm(f"control-{k:02d}", "control", sc, entry))
    return arms


def plan_for(arm: Arm) -> RequestPlan:
    """The three request prompts for one arm, built by the frozen planner."""
    if arm.kind == "control":
        return plan_requests(arm.scenario, "control", OFFSET, UNIT, filler=arm.control_entry["fragment_text"])
    return plan_requests(arm.scenario, "treatment", OFFSET, UNIT)


# --------------------------------------------------------------------------------------
# analysis (pure; works from saved records)
# --------------------------------------------------------------------------------------
def label_for(delta: float, control_max: float) -> str:
    """HIGH: above the frozen exposure threshold AND above every neutral control; LOW: at or below the control maximum;
    INTERMEDIATE otherwise. Descriptive labels only; no new threshold."""
    if delta > EXPOSURE_THRESHOLD and delta > control_max:
        return "HIGH"
    return "LOW" if delta <= control_max else "INTERMEDIATE"


def cue_table(deltas: dict[str, float]) -> dict | None:
    """The 2x2 of verdict word (failed/passed) x logging state (enabled/disabled), when all four arms exist.
    ``interaction_contrast`` = 0 would mean strictly additive cues; it is not zero in any measured model."""
    need = ("B_original", "CueA_failed_but_disabled", "CueB_passed_but_enabled", "N_polarity_reversed")
    if not all(k in deltas for k in need):
        return None
    fe, fd, pe, pd_ = (deltas[k] for k in need)
    return {"failed+enabled (original B)": fe, "failed+disabled": fd, "passed+enabled": pe, "passed+disabled (polarity reversed)": pd_,
            "interaction_contrast": fe - fd - pe + pd_,
            "mean_effect_of_verdict_failed_vs_passed": ((fe - pe) + (fd - pd_)) / 2,
            "mean_effect_of_logging_enabled_vs_disabled": ((fe - fd) + (pe - pd_)) / 2}


def summarize(names: list[str], records1: list[dict], records2: list[dict], *, reference_range: dict | None = None,
              reference_original_delta: float | None = None, baseline_argmax: str = "A") -> dict:
    """Per-arm table, control statistics, cue table and (when 12 controls are present) the frozen robustness analysis.

    ``reference_range`` ({'min','max','median'}) and ``reference_original_delta`` are used only when this run has no
    controls / no B_original arm of its own (e.g. a cue-probe-only run)."""
    by1, by2 = dict(zip(names, records1)), dict(zip(names, records2))
    ctrl = [by1[n]["delta_contrast"] for n in names if n.startswith("control-")]
    if ctrl:
        crange = {"source": "this run's controls", "min": min(ctrl), "max": max(ctrl), "median": statistics.median(ctrl),
                  "mean": statistics.fmean(ctrl), "sd": statistics.stdev(ctrl) if len(ctrl) > 1 else None,
                  "n": len(ctrl), "n_within_0.1": sum(abs(x) <= 0.1 for x in ctrl)}
    elif reference_range:
        crange = {"source": "reference range supplied by the caller (not measured in this run)", **reference_range}
    else:
        crange = None
    if "B_original" in by1:                                   # FIX (v0.2.0): a run without B_original no longer crashes
        d_orig = by1["B_original"]["delta_contrast"]
    else:
        d_orig = reference_original_delta
    arms = {}
    for n in names:
        if n.startswith("control-"):
            continue
        r = by1[n]
        d = r["delta_contrast"]
        arms[n] = {"p_A": r["p_A"], "p_B": r["p_B"], "p_C": r["p_C"], "p_D": r["p_D"], "raw_argmax": r["raw_argmax"], "delta_contrast": d,
                   "difference_from_original_B": (d - d_orig) if d_orig is not None else None,
                   "position_vs_neutral_controls": (None if crange is None else
                                                    "above the neutral-control range" if d > crange["max"] else
                                                    "below the neutral-control range" if d < crange["min"] else "inside the neutral-control range"),
                   "label": None if crange is None else label_for(d, crange["max"]), "reproduced_identically": r == by2[n]}
    out = {"control_range": crange, "arms": arms, "repeatability_bit_identical": records1 == records2,
           "cue_table": cue_table({n: a["delta_contrast"] for n, a in arms.items()})}
    return out


def robustness_observations(controls: list[tuple[int, str, dict]], names: list[str], records: list[dict], baseline_argmax: str) -> dict:
    by = dict(zip(names, records))
    t = by["B_original"]
    return {"schema": RB.OBSERVATIONS_SCHEMA, "protocol_id": "research-driver", "n_planned": len(controls),
            "treatment": {"delta_contrast": t["delta_contrast"], "raw_argmax": t["raw_argmax"], "baseline_argmax": baseline_argmax},
            "controls": [{"id": e["control_id"][:12], "replicate": k, "domain": d, "status": "OK",
                          "delta_contrast": by[f"control-{k:02d}"]["delta_contrast"], "raw_argmax": by[f"control-{k:02d}"]["raw_argmax"]}
                         for k, d, e in controls]}


# --------------------------------------------------------------------------------------
# running (needs a server)
# --------------------------------------------------------------------------------------
def run(*, server: str, out_dir: str, frozen_path: str, controls_path: str | None, baseline: str = "fresh",
        reference_analysis: str | None = None, timeout: float = 1800.0, fixtures_path: str = FIXTURES_PATH) -> dict:
    """Calibrate (fresh, or compatibility-checked reuse), run every arm twice, write traces + sidecars + analysis.json.

    ``baseline`` is ``"fresh"`` or ``"reuse:<baseline.json>"`` (accepted only if a fresh clean fold reproduces it exactly)."""
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to write into existing {out_dir!r}")
    os.makedirs(out_dir)
    fx_full = load_fixtures(fixtures_path)
    sc = {s.id: s for s in fx_full.scenarios}[SCENARIO_ID]
    fx1 = dataclasses.replace(fx_full, scenarios=(sc,))             # same fixtures hash, only the scenario under study
    controls = load_controls(controls_path) if controls_path else None
    arms = build_arms(sc, load_frozen_arms(frozen_path), controls)
    client = LlamaClient(server, timeout=timeout, n_probs=32)
    log: list[str] = []

    def say(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        log.append(line)

    h0 = Harness(client, fx1)
    h0.preflight()
    tok_report = h0.report.to_json()
    say("tokenization verified:", tok_report["id_to_letter"], "dropped", list(tok_report["dropped"]))
    fresh = h0.calibrate(fixtures_path_for_record=fixtures_path)
    if baseline.startswith("reuse:"):
        est_path = baseline[6:]
        est = _load(est_path)
        compat = {"model_hash_equal": est["model_hash"] == fresh["model_hash"], "fixtures_sha_equal": est["fixtures_sha256"] == fresh["fixtures_sha256"],
                  "B_ref_identical": est["scenarios"][SCENARIO_ID]["B_ref"] == fresh["scenarios"][SCENARIO_ID]["B_ref"]}
        say("compatibility with the supplied baseline:", compat)
        if not all(compat.values()):
            raise RuntimeError("the supplied baseline is not compatible with this server: recalibrate (nothing reused silently)")
        base_doc, base_path = est, est_path
    else:
        base_path = os.path.join(out_dir, "baseline.json")
        with open(base_path, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(fresh, indent=2) + "\n")
        h1 = Harness(client, fx1)
        h1.preflight()
        again = h1.calibrate(fixtures_path_for_record=fixtures_path)
        with open(os.path.join(out_dir, "baseline-repeat.json"), "x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(again, indent=2) + "\n")
        say("baseline repeat identical:", again["scenarios"][SCENARIO_ID]["B_ref"] == fresh["scenarios"][SCENARIO_ID]["B_ref"])
        base_doc = fresh
    ref = base_doc["scenarios"][SCENARIO_ID]
    b_ref = [float(x) for x in ref["B_ref"]]
    say("baseline: B_ref", [round(x, 4) for x in b_ref], "argmax", ref["ref_argmax"])

    passes, tokinfo = [], {}
    for p in (1, 2):
        h = Harness(client, fx1)
        info = h.preflight()
        if base_doc["model_hash"] != info.model_hash:
            raise RuntimeError("model identity drift")
        h.cursor_offsets = {SCENARIO_ID: OFFSET}
        h.control_algorithm = "neutral-prose-replicates/1" if controls else "NOT_APPLICABLE"
        h.control_manifest_sha256 = file_sha256(controls_path) if controls_path else "NOT_APPLICABLE"
        h.protocol_info = {"protocol_id": "NONE", "note": "research driver; texts frozen before inference", "frozen_sha256": file_sha256(frozen_path)}
        records = []
        for trial, arm in enumerate(arms, 1):
            plan = plan_for(arm)
            m = h.fold(plan, measure_tokens=True)
            rec = make_record(trial=trial, model_hash=info.model_hash, cursor_offset=OFFSET, m=m, b_ref=b_ref, ref_hash=ref["reference_hash"],
                              control=arm.kind == "control")
            records.append(rec)
            diag = h._last_diag or {}
            ptok = (diag.get("request_prompt_tokens") or [None] * 3)
            h.meta_records.append({"trial": trial, "control": arm.kind == "control", "fold_hash": rec["fold_hash"], "scenario_id": SCENARIO_ID,
                                   "cursor_offset": OFFSET, "request_prompt_tokens": diag.get("request_prompt_tokens"), "coverage": diag.get("coverage"),
                                   "control_filler": ({"sha256": arm.control_entry["fragment_sha256"], "algorithm": "neutral-prose-replicates/1",
                                                       "control_id": arm.control_entry["control_id"], "enumeration_index": arm.control_entry["enumeration_index"],
                                                       "salt": "NOT_APPLICABLE", "repro_id": arm.control_entry["control_id"],
                                                       "token_parity": arm.control_entry["token_parity"]} if arm.kind == "control" else "NOT_APPLICABLE")})
            if p == 1:
                tokinfo[arm.name] = {"request3_prompt_tokens": ptok[2], "coverage_mean": (diag.get("coverage") or {}).get("mean")}
            say(f"pass {p} {arm.name:<26} argmax {rec['raw_argmax']} p_A {rec['p_A']:.4f} p_B {rec['p_B']:.4f} delta {rec['delta_contrast']:+.4f}")
        trace = os.path.join(out_dir, f"traces-pass{p}.jsonl")
        write_traces(trace, records)
        write_sidecar(sidecar_path(trace), build_sidecar(trace_path=trace, records=records, harness=h, baseline_doc=base_doc,
                                                         baseline_file_sha256=file_sha256(base_path), cursor_offset=OFFSET, unit=UNIT))
        passes.append(records)

    names = [a.name for a in arms]
    ref_range = ref_delta = None
    if reference_analysis:                                     # only used for a run without its own controls / B_original arm
        r0 = _load(reference_analysis)["passes"]["pass1"]["descriptive"]
        ref_range = {k: r0["signed_delta"][k] for k in ("min", "max", "median")}
        ref_delta = r0["treatment"]["delta_contrast"]
    out = {"driver": "research/arms.py", "bleediff_version": __version__, "code_revision": code_revision(),
           "baseline": {"B_ref": b_ref, "argmax": ref["ref_argmax"], "ab_margin": ref["ab_margin"], "coverage": ref["coverage"],
                        "reference_hash": ref["reference_hash"], "mode": baseline.split(":")[0]},
           "tokenization": tok_report, "token_info": tokinfo,
           **summarize(names, passes[0], passes[1], reference_range=ref_range, reference_original_delta=ref_delta,
                       baseline_argmax=ref["ref_argmax"])}
    if controls:
        man_fp = _load(controls_path)["selection"]["tokenizer_fingerprint"]
        from bleediff import control as C

        live_fp = C.tokenizer_fingerprint(lambda t: client.tokenize(t, cache=False))
        out["controls_tokenizer_fingerprint_matches_server"] = man_fp == live_fp     # False => token parity of controls is NOT guaranteed
        obs = robustness_observations(controls, names, passes[0], ref["ref_argmax"])
        out["robustness_analysis_original_B_vs_controls"] = RB.analyze(obs)
    with open(os.path.join(out_dir, "analysis.json"), "x", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    with open(os.path.join(out_dir, "run.log"), "x", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")
    say("repeatability: bit-identical =", out["repeatability_bit_identical"])
    return out
