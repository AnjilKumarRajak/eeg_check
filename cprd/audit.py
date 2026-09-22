"""Audit ledger: every reported number is a record, or it does not exist.

The table renderer REFUSES to print any value lacking a ledger entry with a complete
gates_passed list. This extends the existing pattern (DecodeResult carrying
config_hash/checkpoint; zero_gamma_check written to results files) into a hard rule.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


def code_git_rev(repo_dir: str | None = None) -> str:
    try:
        d = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=d,
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def sha256_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class Record:
    metric: str
    value: float
    ci_lo: Optional[float] = None
    ci_hi: Optional[float] = None
    p: Optional[float] = None
    n_clusters: Optional[int] = None
    n_sentences: Optional[int] = None
    split_fingerprint: str = ""
    config_sha: str = ""
    checkpoint: str = ""
    code_rev: str = ""
    prior_id: str = ""
    pool_hash: str = ""
    seed: Optional[int] = None
    n_perm: Optional[int] = None
    n_boot: Optional[int] = None
    prereg_sha: str = ""
    gates_passed: list = field(default_factory=list)
    arm: str = ""
    experiment: str = ""
    objective: str = ""          # 'raw_dI_ascent' (pre-registered) | 'hybrid_contrast' (exploratory)
    evidence: str = "eeg"        # 'eeg' | 'gaze' -- which channel produced this number
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def key(self) -> str:
        return sha256_of([self.metric, self.experiment, self.arm, self.config_sha, self.seed])


class Ledger:
    """Append-only JSONL ledger."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def append(self, rec: Record) -> str:
        d = asdict(rec)
        d["_key"] = rec.key()
        with open(self.path, "a") as fh:
            fh.write(json.dumps(d) + "\n")
        return d["_key"]

    def load(self) -> list:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def render_table(self, experiment: str, required_gates: tuple = ()) -> str:
        """Markdown table of one experiment's records. REFUSES incomplete records."""
        rows = [r for r in self.load() if r.get("experiment") == experiment]
        ok, refused = [], []
        for r in rows:
            missing = [g for g in required_gates if g not in r.get("gates_passed", [])]
            if missing or not r.get("split_fingerprint") or not r.get("config_sha"):
                refused.append((r.get("metric"), r.get("arm"),
                                missing or ["missing provenance"]))
            else:
                ok.append(r)
        lines = [f"## {experiment}", "",
                 "| metric | arm | value | 95% CI | p | n_clusters | gates |",
                 "|---|---|---|---|---|---|---|"]
        for r in ok:
            ci = (f"[{r['ci_lo']:.4g}, {r['ci_hi']:.4g}]"
                  if r.get("ci_lo") is not None else "—")
            p = f"{r['p']:.4g}" if r.get("p") is not None else "—"
            lines.append(f"| {r['metric']} | {r['arm']} | {r['value']:.4g} | {ci} | {p} "
                         f"| {r.get('n_clusters', '—')} | {len(r.get('gates_passed', []))} |")
        if refused:
            lines += ["", "**REFUSED (incomplete provenance/gates — not rendered as results):**"]
            for m, a, why in refused:
                lines.append(f"- {m} [{a}]: {', '.join(map(str, why))}")
        return "\n".join(lines)
