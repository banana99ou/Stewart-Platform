"""
presentation/Fig_Gen.py

Generate presentation-ready figures sized to match the display dimensions
in generate_presentation.py, with unified font styling and no figure titles
(the slide itself provides the title).

Figure sizes (width) are taken from the _img() calls in generate_presentation.py:
  Fig2  -> 11.5"    (slide 7, single figure)
  Fig3  ->  8.0"    (slide 8, left)
  Fig4  ->  8.0"    (slide 8, right)
  Fig5a ->  8.5"    (slide 9, left)
  Fig5b ->  8.5"    (slide 9, right)

Run with:
  . .venv/bin/activate
  python presentation/Fig_Gen.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Import data-loading infrastructure from the paper figure generator so we
# don't duplicate CSV parsing, RunData, bias estimation, etc.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper.Fig_Gen import (  # noqa: E402
    BASELINE_RUN_DIRS,
    HERO_RUN_DIR,
    PREFIX_RUN_DIR,
    SKIP_FIRST_S,
    RunData,
    _check_runs_exist,
    _to_float,
    _wrap_deg,
    load_run,
)

OUT_DIR = Path(__file__).resolve().parent / "fig"
DPI = 200

# Display widths from generate_presentation.py (inches)
FIG2_W = 11.5
FIG3_W = 8.0
FIG4_W = 8.0
FIG5A_W = 8.5
FIG5B_W = 8.5

FONT_NAME = "Nanum Square"


def _mpl():
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_NAME, "NanumSquare", "DejaVu Sans"],
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": DPI,
    })
    return plt


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig2_pitch_response(run: RunData) -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(1, 1, figsize=(FIG2_W, 4.5))

    t = run.t_s
    m = run.analysis_mask

    sp = np.array(
        [_to_float(r.get("sp_pitch_deg")) for r in run.tick_rows], dtype=float
    )
    if np.any(np.isfinite(sp)):
        ax.plot(t[m], sp[m], "k--", lw=1.5, label="setpoint pitch (deg)")

    ax.plot(t[m], run.xp_pitch[m], color="tab:blue", lw=1.2,
            label="X-Plane pitch (deg)")
    ax.plot(t[m], run.px4_pitch[m], color="tab:orange", lw=1.2,
            label="PX4 pitch (deg)")

    pb = run.mount_bias_pitch_deg
    if np.isfinite(pb):
        ax.plot(
            t[m],
            _wrap_deg(run.xp_pitch + pb)[m],
            color="tab:blue", lw=1.2, ls=":",
            label=f"Fixed X-Plane pitch ({pb:+.2f}\u00b0)",
        )

    ax.set_xlabel("time (s) from run start")
    ax.set_ylabel("pitch (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    out = OUT_DIR / "Fig2_PitchResponse.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig3_pitch_error(run: RunData) -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(1, 1, figsize=(FIG3_W, 4.0))

    t = run.t_s
    m = run.analysis_mask

    err_raw = _wrap_deg(run.px4_pitch - run.xp_pitch)
    ax.plot(t[m], err_raw[m], color="tab:red", lw=1.2,
            label="raw error: PX4 \u2212 X-Plane (deg)")

    pb = run.mount_bias_pitch_deg
    if np.isfinite(pb):
        err_bc = _wrap_deg((run.px4_pitch - pb) - run.xp_pitch)
        ax.plot(t[m], err_bc[m], color="tab:green", lw=1.2,
                label="bias-corrected (deg)")

    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("time (s) from run start")
    ax.set_ylabel("error (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    out = OUT_DIR / "Fig3_PitchError.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig4_mounting_bias_timeseries(pref: RunData, hero: RunData) -> Path:
    plt = _mpl()
    fig, axes = plt.subplots(2, 1, figsize=(FIG4_W, 4.0),
                             sharex=False, sharey=True)

    for ax, run, subtitle in [
        (axes[0], pref, "pre-fix run (loose mounting)"),
        (axes[1], hero, "baseline hero run (firm mounting)"),
    ]:
        t = run.t_s
        m = run.analysis_mask
        bias_ts = _wrap_deg(run.px4_pitch - run.cmd_pitch)
        ax.plot(t[m], bias_ts[m], lw=1.1, color="tab:purple",
                label="wrap(PX4 pitch \u2212 cmd pitch)")
        pb = run.mount_bias_pitch_deg
        if np.isfinite(pb):
            ax.axhline(pb, color="k", lw=1.0, alpha=0.35,
                       label=f"mean in hold_0: {pb:+.2f}\u00b0")
        ax.set_ylabel("deg")
        ax.grid(True, alpha=0.3)
        ax.set_title(subtitle)
        ax.legend(loc="best")

    axes[-1].set_xlabel("time (s) from run start (analysis window)")
    fig.tight_layout()

    out = OUT_DIR / "Fig4_MountingBias_TimeSeries.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig5_latency(
    run_hero: RunData, runs_baseline: list[RunData]
) -> tuple[Path, Path]:
    plt = _mpl()

    # (a) hero run time series
    fig_a, ax_a = plt.subplots(1, 1, figsize=(FIG5A_W, 3.8))
    t = run_hero.t_s
    m = run_hero.analysis_mask
    ax_a.plot(t[m], run_hero.xp_age_ms[m], lw=1.1,
              label="X-Plane sample age (ms)")
    ax_a.plot(t[m], run_hero.px4_age_ms[m], lw=1.1,
              label="PX4 sample age (ms)")
    if run_hero.ack_t_s.size > 0:
        ack_mask = (np.isfinite(run_hero.ack_t_s)
                    & (run_hero.ack_t_s >= float(SKIP_FIRST_S)))
        ax_a.plot(run_hero.ack_t_s[ack_mask],
                  run_hero.serial_rtt_ms_by_ack[ack_mask],
                  lw=1.1, label="Stewart serial RTT (ms)")
    ax_a.set_xlabel("time (s) from run start")
    ax_a.set_ylabel("ms")
    ax_a.grid(True, alpha=0.25)
    ax_a.legend(loc="upper right")
    fig_a.tight_layout()

    out_a = OUT_DIR / "Fig5a_Latency_TimeSeries.png"
    fig_a.savefig(out_a, dpi=DPI)
    plt.close(fig_a)

    # (b) aggregate distributions across baseline runs
    xp_age_all = np.concatenate(
        [r.xp_age_ms[r.analysis_mask] for r in runs_baseline])
    px_age_all = np.concatenate(
        [r.px4_age_ms[r.analysis_mask] for r in runs_baseline])

    serial_rtt_parts: list[np.ndarray] = []
    for r in runs_baseline:
        if r.ack_t_s.size == 0:
            continue
        ack_mask = (np.isfinite(r.ack_t_s)
                    & (r.ack_t_s >= float(SKIP_FIRST_S)))
        serial_rtt_parts.append(r.serial_rtt_ms_by_ack[ack_mask])
    serial_rtt_all = (np.concatenate(serial_rtt_parts)
                      if serial_rtt_parts
                      else np.zeros(0, dtype=float))

    fig_b, ax_b = plt.subplots(1, 1, figsize=(FIG5B_W, 3.8))
    data = [xp_age_all, px_age_all, serial_rtt_all]
    labels = ["X-Plane\nsample age", "PX4\nsample age", "Serial\nRTT"]
    ax_b.boxplot(data, tick_labels=labels, showfliers=False)
    ax_b.set_ylabel("ms")
    ax_b.grid(True, axis="y", alpha=0.25)
    fig_b.tight_layout()

    out_b = OUT_DIR / "Fig5b_Latency_Distribution_Baseline.png"
    fig_b.savefig(out_b, dpi=DPI)
    plt.close(fig_b)

    return out_a, out_b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _check_runs_exist([HERO_RUN_DIR, PREFIX_RUN_DIR, *BASELINE_RUN_DIRS])

    hero = load_run(HERO_RUN_DIR, skip_first_s=SKIP_FIRST_S)
    pref = load_run(PREFIX_RUN_DIR, skip_first_s=SKIP_FIRST_S)
    baseline = [load_run(p, skip_first_s=SKIP_FIRST_S) for p in BASELINE_RUN_DIRS]

    out = {
        "Fig2": str(fig2_pitch_response(hero)),
        "Fig3": str(fig3_pitch_error(hero)),
        "Fig4": str(fig4_mounting_bias_timeseries(pref, hero)),
    }
    a, b = fig5_latency(hero, baseline)
    out["Fig5a"] = str(a)
    out["Fig5b"] = str(b)

    print(f"[presentation/Fig_Gen] wrote figures to: {OUT_DIR}")
    for k, v in out.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
