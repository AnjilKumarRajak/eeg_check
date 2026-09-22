"""Pre-registration: freeze hypotheses, hashes and the matched-N prediction BEFORE
test-set contact; guard test-set access through a logging wrapper.

The prereg document's SHA-256 prints in every table. The deviations log is
append-only. Test-split reads outside the wrapper are a protocol violation by
construction (the experiment drivers only receive the wrapped loader).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

HYPOTHESES = [
    {"id": "H1", "primary": True,
     "statement": "dI_hat > 0 on EEG-present tokens under BOTH nulls "
                  "(text_cluster_derangement, temporal_roll) on the clean text split",
     "direction": "greater", "alpha": 0.05},
    {"id": "H2", "statement": "selection accuracy at pre-registered N* within a factor "
                              "of 1.5 (in bits) of the value predicted from val-split dI_hat",
     "direction": "two-sided-band", "alpha": 0.05},
    {"id": "H3", "statement": "Nykopp ITR under the primary (most conservative) "
                              "denominator falls in the pre-registered 1-60 bits/min band",
     "direction": "band", "alpha": 0.05},
    {"id": "H4", "statement": "partial confidence AUC (after length/frequency/prior-logp "
                              "partialling) > 0.5 and dominates the cheap-confidence baseline",
     "direction": "greater", "alpha": 0.05},
    {"id": "H5", "statement": "every noise arm (zeroed, gaussian_matched, derangement, "
                              "amplitude_only, position_only) collapses to chance "
                              "(TOST +/-2pp) on the selection metric",
     "direction": "equivalence", "alpha": 0.05},
    {"id": "H6", "statement": "gaze-residualized EEG attribute decoding > chance for at "
                              "least one of {topic, sentiment, relation} (Holm-corrected)",
     "direction": "greater", "alpha": 0.05},
    {"id": "H7", "statement": "comparability corrected BLEU-1 of the selected output "
                              "exceeds the reproduced corrected baseline",
     "direction": "greater", "alpha": 0.05},
]


def write_prereg(out_dir: str, frozen: dict) -> str:
    """Write prereg.json + prereg.md; return the SHA-256 (short) of the json.

    `frozen` must include: config_sha, split_fingerprints, prior_ids, pool_hashes,
    matched_N_prediction, code_rev, seeds, n_perm, n_boot. Missing keys are an error —
    an incomplete prereg is worse than none.
    """
    required = ["config_sha", "split_fingerprints", "prior_ids", "matched_N_prediction",
                "code_rev", "seeds", "n_perm", "n_boot"]
    missing = [k for k in required if k not in frozen]
    if missing:
        raise ValueError(f"prereg incomplete; missing {missing}")
    os.makedirs(out_dir, exist_ok=True)
    prev = os.path.join(out_dir, "prereg.json")
    if os.path.exists(prev):
        # A re-freeze (e.g. a re-run after a crash) is a deviation: keep the old file and
        # log it, never overwrite silently.
        old = open(prev).read()
        old_sha = hashlib.sha256(old.encode()).hexdigest()[:16]
        with open(os.path.join(out_dir, f"prereg.superseded_{old_sha}.json"), "w") as fh:
            fh.write(old)
        log_deviation(out_dir, f"prereg re-frozen; previous prereg sha {old_sha} kept as "
                               f"prereg.superseded_{old_sha}.json")
    doc = {"hypotheses": HYPOTHESES, "frozen": frozen,
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "deviations": []}
    js = json.dumps(doc, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "prereg.json"), "w") as fh:
        fh.write(js)
    sha = hashlib.sha256(js.encode()).hexdigest()[:16]
    with open(os.path.join(out_dir, "prereg.md"), "w") as fh:
        fh.write(f"# Pre-registration (sha {sha})\n\n")
        for h in HYPOTHESES:
            tag = " **[PRIMARY]**" if h.get("primary") else ""
            fh.write(f"- **{h['id']}**{tag}: {h['statement']} "
                     f"(direction: {h['direction']}, alpha={h['alpha']})\n")
        fh.write("\n## Frozen artifacts\n```json\n"
                 + json.dumps(frozen, indent=2, sort_keys=True) + "\n```\n")
    return sha


def log_deviation(out_dir: str, text: str) -> None:
    path = os.path.join(out_dir, "deviations.log")
    with open(path, "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {text}\n")


class GuardedTestAccess:
    """The only sanctioned door to the test split. Every open is logged."""

    def __init__(self, test_h5_path: str, prereg_dir: str):
        self.path = test_h5_path
        self.prereg_dir = prereg_dir
        self.ledger = os.path.join(prereg_dir, "test_access.log")
        if not os.path.exists(os.path.join(prereg_dir, "prereg.json")):
            raise RuntimeError("test access refused: no prereg.json — freeze the "
                               "pre-registration before touching the test split")

    def load(self, caller: str, config_sha: str, evidence: str = "eeg"):
        from .data import read_split
        os.makedirs(self.prereg_dir, exist_ok=True)
        with open(self.ledger, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  caller={caller}  "
                     f"config={config_sha}  evidence={evidence}\n")
        return read_split(self.path, evidence=evidence)


def check_gate_artifact(runs_dir: str, gate_name: str) -> dict:
    """Drivers call this to refuse running before their predecessor's gate is green."""
    path = os.path.join(runs_dir, f"gate_{gate_name}.json")
    if not os.path.exists(path):
        raise RuntimeError(f"predecessor gate '{gate_name}' missing at {path}; "
                           "run its experiment first")
    with open(path) as fh:
        g = json.load(fh)
    if not g.get("passed", False):
        raise RuntimeError(
            f"predecessor gate '{gate_name}' FAILED. To proceed anyway (exploratory), "
            f"run: python experiments/override_gate.py <runs_dir> {gate_name} --reason '...' "
            f"-- the override is logged and downstream records are labeled "
            f"'{gate_name}_OVERRIDDEN'.")
    if g.get("override") or (g.get("detail") or {}).get("override"):
        print(f"  !! gate '{gate_name}' is OVERRIDDEN ({g.get('override_reason', 'no reason')}) "
              f"— downstream results are exploratory, not validated", flush=True)
    return g


def _make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    elif hasattr(obj, "item"):
        return _make_json_serializable(obj.item())
    elif isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_gate_artifact(runs_dir: str, gate_name: str, passed: bool, detail: dict) -> None:
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"gate_{gate_name}.json")
    data = {"passed": bool(passed), "detail": _make_json_serializable(detail),
            "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, allow_nan=False)
        fh.flush()
        os.fsync(fh.fileno())


def collect_green_gates(runs_dir: str) -> list:
    """Derive the gates_passed list from what is ACTUALLY on disk — never hardcode.

    Labels: '<name>' for a clean pass, '<name>_OVERRIDDEN' when the artifact carries
    an override marker, '<name>_partial' when the gate passed with a recorded partial
    validity range (E1's validated_range mechanism). Ledger consumers therefore see
    exactly what kind of green light each stage had.
    """
    out = []
    if not os.path.isdir(runs_dir):
        return out
    for fn in sorted(os.listdir(runs_dir)):
        if not (fn.startswith("gate_") and fn.endswith(".json")):
            continue
        name = fn[len("gate_"):-len(".json")]
        try:
            with open(os.path.join(runs_dir, fn)) as fh:
                g = json.load(fh)
        except Exception:
            continue
        if not g.get("passed", False):
            continue
        detail = g.get("detail", {}) or {}
        if g.get("override") or detail.get("override"):
            out.append(f"{name}_OVERRIDDEN")
        elif detail.get("partial_pass"):
            out.append(f"{name}_partial")
        else:
            out.append(name)
    return out


def apply_gate_override(runs_dir: str, gate_name: str, reason: str) -> str:
    """The SANCTIONED way to unblock a failed gate — never edit the JSON by hand.

    Rewrites gate_<name>.json with passed=true and an explicit override block that
    preserves the original artifact verbatim under detail.original, and appends the
    reason to the prereg deviations log. check_gate_artifact prints a loud warning on
    overridden gates, and collect_green_gates labels downstream records
    '<name>_OVERRIDDEN' so no reader can mistake the run for a clean pass.
    """
    path = os.path.join(runs_dir, f"gate_{gate_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no gate to override at {path}")
    with open(path) as fh:
        original = json.load(fh)
    if original.get("passed", False) and not original.get("override"):
        return "gate already passing; no override needed"
    data = {
        "passed": True,
        "override": True,
        "override_reason": reason,
        "override_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "detail": {"override": True, "original": original,
                   **{k: v for k, v in (original.get("detail") or {}).items()}},
        "at": original.get("at", ""),
    }
    with open(path, "w") as fh:
        json.dump(_make_json_serializable(data), fh, indent=2, allow_nan=False)
        fh.flush(); os.fsync(fh.fileno())
    prereg_dir = os.path.join(runs_dir, "prereg")
    os.makedirs(prereg_dir, exist_ok=True)
    log_deviation(prereg_dir,
                  f"GATE OVERRIDE gate_{gate_name}: {reason} (original passed="
                  f"{original.get('passed')}, preserved in detail.original)")
    return f"gate_{gate_name} overridden; deviation logged; downstream records will carry '{gate_name}_OVERRIDDEN'"
