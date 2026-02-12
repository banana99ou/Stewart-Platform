from __future__ import annotations

"""
paper/Fig_Gen.py

One-click paper figure generator from real log files.

What it does (high level):
- Loads tick.csv + stewart_ack.csv from selected runs under ./logs/
- Applies a consistent analysis window (default: skip first 10s)
- Estimates FC mounting bias from PX4 attitude vs commanded platform pose (px4 - cmd) in hold_0
- Generates:
  - Fig2: Pitch response overlay (hero run)
  - Fig3: Pitch error time series (raw vs bias-corrected) (hero run)
  - Fig4: Mounting bias time series (pre-fix run vs hero run)
  - Fig5: Latency: (a) hero run time series, (b) aggregate distributions for runs #5–#9
- Writes all outputs to paper/fig/

Run with:
  . .venv/bin/activate
  python paper/Fig_Gen.py

Notes:
- This script intentionally does NOT depend on precomputed postprocess_metrics.json.
- It uses only CSV logs so figures are always consistent with the chosen windowing.
"""

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


# ---------------- Config ----------------

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
OUT_DIR = ROOT / "paper" / "fig"

# Baseline dataset (post-mount fix + speed PID enabled)
BASELINE_RUN_DIRS = [
    LOGS_DIR / "run-20260208-202549 #5 (fix mounting)",
    LOGS_DIR / "run-20260208-210113 #6",
    LOGS_DIR / "run-20260208-210614 #7",
    LOGS_DIR / "run-20260208-210808 #8",
    LOGS_DIR / "run-20260208-211109 #9",
]

# Representative run for time-series plots
HERO_RUN_DIR = LOGS_DIR / "run-20260208-210808 #8"

# Pre-fix run (loose mount / different config) used only for “before vs after” illustration
PREFIX_RUN_DIR = LOGS_DIR / "run-20260208-201951 #4"

# Analysis window policy
SKIP_FIRST_S = 10.0


# ---------------- Utilities ----------------

def _wrap_deg(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


def _to_float(s: Any) -> float:
    if s is None:
        return float("nan")
    if isinstance(s, (int, float)):
        return float(s)
    st = str(s).strip()
    if st == "":
        return float("nan")
    try:
        return float(st)
    except Exception:
        return float("nan")


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            return []
        return [dict(row) for row in r]


def _ensure_outdir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunData:
    run_dir: Path
    tick_rows: list[dict[str, str]]
    ack_rows: list[dict[str, str]]

    # Derived series (tick timeline)
    t_s: np.ndarray
    analysis_mask: np.ndarray

    # Common signals
    scenario_phase: np.ndarray
    pid_enabled: np.ndarray

    # X-Plane
    xp_pitch: np.ndarray

    # PX4
    px4_pitch: np.ndarray

    # Commanded platform pose
    cmd_pitch: np.ndarray

    # Stewart / ACK joined fields (best-effort)
    serial_rtt_ms_by_ack: np.ndarray  # in ACK time axis
    ack_t_s: np.ndarray

    # “Sample ages” (at tick)
    xp_age_ms: np.ndarray
    px4_age_ms: np.ndarray

    # Mounting bias (estimated from hold_0)
    mount_bias_pitch_deg: float


def load_run(run_dir: Path, *, skip_first_s: float) -> RunData:
    tick_path = run_dir / "tick.csv"
    ack_path = run_dir / "stewart_ack.csv"

    tick_rows = _read_csv_dicts(tick_path)
    if not tick_rows:
        raise RuntimeError(f"tick.csv missing/empty: {tick_path}")
    ack_rows = _read_csv_dicts(ack_path)

    t_tick_ns = np.array([_to_float(r.get("t_tick_ns")) for r in tick_rows], dtype=float)
    t0 = float(t_tick_ns[np.isfinite(t_tick_ns)][0])
    t_s = (t_tick_ns - t0) / 1e9
    analysis_mask = np.isfinite(t_s) & (t_s >= float(skip_first_s))

    scenario_phase = np.array([str(r.get("scenario_phase") or "").strip() for r in tick_rows], dtype=object)
    pid_enabled = np.array([_to_float(r.get("pid_enabled")) for r in tick_rows], dtype=float)

    xp_pitch = np.array([_to_float(r.get("xp_pitch_deg")) for r in tick_rows], dtype=float)
    px4_pitch = np.array([_to_float(r.get("px4_pitch_deg")) for r in tick_rows], dtype=float)
    cmd_pitch = np.array([_to_float(r.get("cmd_pitch_deg")) for r in tick_rows], dtype=float)

    xp_t_rx_ns = np.array([_to_float(r.get("xp_t_rx_ns")) for r in tick_rows], dtype=float)
    px4_t_rx_ns = np.array([_to_float(r.get("px4_t_rx_ns")) for r in tick_rows], dtype=float)

    xp_age_ms = (t_tick_ns - xp_t_rx_ns) / 1e6
    px4_age_ms = (t_tick_ns - px4_t_rx_ns) / 1e6
    xp_age_ms[~np.isfinite(xp_t_rx_ns)] = np.nan
    px4_age_ms[~np.isfinite(px4_t_rx_ns)] = np.nan

    # Estimate mounting bias from steady window: hold_0 & pid_enabled & analysis_mask
    mask_bias = analysis_mask & (scenario_phase == "hold_0") & np.isfinite(pid_enabled) & (pid_enabled > 0.5)
    d = _wrap_deg(px4_pitch - cmd_pitch)
    if np.any(mask_bias & np.isfinite(d)):
        mount_bias_pitch_deg = float(np.nanmean(d[mask_bias]))
    else:
        mount_bias_pitch_deg = float("nan")

    # ACK time axis and RTT
    if ack_rows:
        ack_t_rx_ns = np.array([_to_float(r.get("t_rx_ns")) for r in ack_rows], dtype=float)
        ack_t_s = (ack_t_rx_ns - t0) / 1e9
        serial_rtt_ms = np.array([_to_float(r.get("serial_rtt_ms")) for r in ack_rows], dtype=float)
    else:
        ack_t_s = np.zeros(0, dtype=float)
        serial_rtt_ms = np.zeros(0, dtype=float)

    return RunData(
        run_dir=run_dir,
        tick_rows=tick_rows,
        ack_rows=ack_rows,
        t_s=t_s,
        analysis_mask=analysis_mask,
        scenario_phase=scenario_phase,
        pid_enabled=pid_enabled,
        xp_pitch=xp_pitch,
        px4_pitch=px4_pitch,
        cmd_pitch=cmd_pitch,
        serial_rtt_ms_by_ack=serial_rtt_ms,
        ack_t_s=ack_t_s,
        xp_age_ms=xp_age_ms,
        px4_age_ms=px4_age_ms,
        mount_bias_pitch_deg=mount_bias_pitch_deg,
    )


# ---------------- Plotting ----------------

# Font size multiplier for Fig1, Fig2, Fig5b (2 = double)
_FONT_SCALE = 1.7


def _mpl():
    # Avoid GUI backend issues in headless runs
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    return plt


def fig2_pitch_response(run: RunData) -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))

    t = run.t_s
    m = run.analysis_mask

    # Setpoint is optional; when present it is already in tick.csv (sp_pitch_deg).
    sp = np.array([_to_float(r.get("sp_pitch_deg")) for r in run.tick_rows], dtype=float)
    if np.any(np.isfinite(sp)):
        ax.plot(t[m], sp[m], "k--", lw=1.5, label="setpoint pitch (deg)")

    ax.plot(t[m], run.xp_pitch[m], color="tab:blue", lw=1.2, label="X-Plane pitch (deg)")
    ax.plot(t[m], run.px4_pitch[m], color="tab:orange", lw=1.2, label="PX4 pitch (deg)")

    # XP shifted by mounting bias (estimated from PX4 vs cmd in hold_0)
    pb = run.mount_bias_pitch_deg
    if np.isfinite(pb):
        ax.plot(
            t[m],
            _wrap_deg(run.xp_pitch + pb)[m],
            color="tab:blue",
            lw=1.2,
            ls=":",
            label=f"Fixed X-Plane pitch ({pb:+.2f} deg)",
        )

    ax.set_title("Fig. 2. Trim transition: pitch response", fontsize=int(12 * _FONT_SCALE))
    ax.set_xlabel("time (s) from run start", fontsize=int(10 * _FONT_SCALE))
    ax.set_ylabel("pitch (deg)", fontsize=int(10 * _FONT_SCALE))
    ax.tick_params(axis="both", labelsize=int(10 * _FONT_SCALE))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=int(9 * _FONT_SCALE))
    fig.tight_layout()

    out = OUT_DIR / "Fig2_PitchResponse.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig3_pitch_error(run: RunData) -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.2))

    t = run.t_s
    m = run.analysis_mask

    err_raw = _wrap_deg(run.px4_pitch - run.xp_pitch)
    ax.plot(t[m], err_raw[m], color="tab:red", lw=1.2, label="raw error: PX4 - X-Plane (deg)")

    pb = run.mount_bias_pitch_deg
    if np.isfinite(pb):
        err_bc = _wrap_deg((run.px4_pitch - pb) - run.xp_pitch)
        ax.plot(t[m], err_bc[m], color="tab:green", lw=1.2, label="bias-corrected: (PX4 - b) - X-Plane (deg)")

    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_title("Fig. 3. Pitch tracking error")
    ax.set_xlabel("time (s) from run start")
    ax.set_ylabel("error (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out = OUT_DIR / "Fig3_PitchError.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig4_mounting_bias_timeseries(pref: RunData, hero: RunData) -> Path:
    plt = _mpl()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=False, sharey=True)

    for ax, run, title in [
        (axes[0], pref, "pre-fix run (loose mounting)"),
        (axes[1], hero, "baseline hero run (firm mounting)"),
    ]:
        t = run.t_s
        m = run.analysis_mask
        bias_ts = _wrap_deg(run.px4_pitch - run.cmd_pitch)
        ax.plot(t[m], bias_ts[m], lw=1.1, color="tab:purple", label="wrap(PX4 pitch - cmd pitch)")
        pb = run.mount_bias_pitch_deg
        if np.isfinite(pb):
            ax.axhline(pb, color="k", lw=1.0, alpha=0.35, label=f"mean in hold_0: {pb:+.2f} deg")
        ax.set_ylabel("deg")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Fig. 4. Mounting bias evidence — {title}", fontsize=11)
        ax.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("time (s) from run start (analysis window)")
    fig.tight_layout()

    out = OUT_DIR / "Fig4_MountingBias_TimeSeries.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig5_latency(run_hero: RunData, runs_baseline: list[RunData]) -> tuple[Path, Path]:
    plt = _mpl()

    # (a) hero run time series
    fig_a, ax_a = plt.subplots(1, 1, figsize=(11, 4.5))
    t = run_hero.t_s
    m = run_hero.analysis_mask
    ax_a.plot(t[m], run_hero.xp_age_ms[m], lw=1.1, label="X-Plane sample age at tick (ms)")
    ax_a.plot(t[m], run_hero.px4_age_ms[m], lw=1.1, label="PX4 sample age at tick (ms)")
    if run_hero.ack_t_s.size > 0:
        # only plot ACK series within analysis window time range
        ack_mask = np.isfinite(run_hero.ack_t_s) & (run_hero.ack_t_s >= float(SKIP_FIRST_S))
        ax_a.plot(run_hero.ack_t_s[ack_mask], run_hero.serial_rtt_ms_by_ack[ack_mask], lw=1.1, label="Stewart serial RTT (ms)")
    ax_a.set_title("Fig. 5(a). Latency time series")
    ax_a.set_xlabel("time (s) from run start")
    ax_a.set_ylabel("ms")
    ax_a.grid(True, alpha=0.25)
    ax_a.legend(loc="upper right", fontsize=9)
    fig_a.tight_layout()
    out_a = OUT_DIR / "Fig5a_Latency_TimeSeries.png"
    fig_a.savefig(out_a, dpi=200)
    plt.close(fig_a)

    # (b) aggregate distributions across baseline runs (#5–#9)
    # Concatenate arrays after applying each run's analysis mask.
    xp_age_all = np.concatenate([r.xp_age_ms[r.analysis_mask] for r in runs_baseline])
    px_age_all = np.concatenate([r.px4_age_ms[r.analysis_mask] for r in runs_baseline])

    serial_rtt_all = []
    for r in runs_baseline:
        if r.ack_t_s.size == 0:
            continue
        ack_mask = np.isfinite(r.ack_t_s) & (r.ack_t_s >= float(SKIP_FIRST_S))
        serial_rtt_all.append(r.serial_rtt_ms_by_ack[ack_mask])
    serial_rtt_all = np.concatenate(serial_rtt_all) if serial_rtt_all else np.zeros(0, dtype=float)

    fig_b, ax_b = plt.subplots(1, 1, figsize=(11, 4.8))
    data = [xp_age_all, px_age_all, serial_rtt_all]
    labels = ["X-Plane sample age (ms)", "PX4 sample age (ms)", "Serial RTT (ms)"]
    # Matplotlib >=3.9 renamed "labels" -> "tick_labels"
    ax_b.boxplot(data, tick_labels=labels, showfliers=False)
    ax_b.set_title("Fig. 3. Latency distributions", fontsize=int(12 * _FONT_SCALE))
    ax_b.set_ylabel("ms", fontsize=int(10 * _FONT_SCALE))
    ax_b.tick_params(axis="both", labelsize=int(10 * _FONT_SCALE))
    ax_b.grid(True, axis="y", alpha=0.25)
    fig_b.tight_layout()
    out_b = OUT_DIR / "Fig5b_Latency_Distribution_Baseline.png"
    fig_b.savefig(out_b, dpi=200)
    plt.close(fig_b)

    return out_a, out_b


# ---------------- Main ----------------

def _check_runs_exist(paths: Iterable[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        msg = "\n".join(str(p) for p in missing)
        raise SystemExit(f"Missing run directories:\n{msg}")


def main() -> int:
    _ensure_outdir()
    _check_runs_exist([HERO_RUN_DIR, PREFIX_RUN_DIR, *BASELINE_RUN_DIRS])

    # Load runs
    hero = load_run(HERO_RUN_DIR, skip_first_s=SKIP_FIRST_S)
    pref = load_run(PREFIX_RUN_DIR, skip_first_s=SKIP_FIRST_S)
    baseline = [load_run(p, skip_first_s=SKIP_FIRST_S) for p in BASELINE_RUN_DIRS]

    # Generate figures
    out = {
        "Fig2": str(fig2_pitch_response(hero)),
        "Fig3": str(fig3_pitch_error(hero)),
        "Fig4": str(fig4_mounting_bias_timeseries(pref, hero)),
    }
    a, b = fig5_latency(hero, baseline)
    out["Fig5a"] = str(a)
    out["Fig5b"] = str(b)

    # Write a lightweight manifest for reproducibility
    manifest = {
        "hero_run": str(HERO_RUN_DIR),
        "pref_fix_run": str(PREFIX_RUN_DIR),
        "baseline_runs": [str(p) for p in BASELINE_RUN_DIRS],
        "skip_first_s": float(SKIP_FIRST_S),
        "mounting_bias_pitch_deg_hero": float(hero.mount_bias_pitch_deg) if np.isfinite(hero.mount_bias_pitch_deg) else None,
        "outputs": out,
    }
    (OUT_DIR / "FigGen_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[Fig_Gen] wrote figures to: {OUT_DIR}")
    for k, v in out.items():
        print(f"[Fig_Gen] {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

