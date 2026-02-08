from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _wrap_deg(x: np.ndarray) -> np.ndarray:
    """
    Wrap degrees into [-180, 180).
    Works with NaNs.
    """
    return (x + 180.0) % 360.0 - 180.0


def _nan_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(np.square(x))))


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


def _to_int(s: Any) -> Optional[int]:
    if s is None:
        return None
    if isinstance(s, int):
        return int(s)
    st = str(s).strip()
    if st == "":
        return None
    try:
        return int(st)
    except Exception:
        return None


def _stats_ms_from_ns_diff(t1_ns: np.ndarray, t0_ns: np.ndarray) -> np.ndarray:
    """
    Compute (t1_ns - t0_ns) in milliseconds, preserving NaNs if either side is not finite.
    Inputs may be float arrays containing NaNs.
    """
    t1_ns = np.asarray(t1_ns, dtype=float)
    t0_ns = np.asarray(t0_ns, dtype=float)
    out = (t1_ns - t0_ns) / 1e6
    out[~(np.isfinite(t1_ns) & np.isfinite(t0_ns))] = np.nan
    return out


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    # Gracefully handle empty files.
    if path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            return []
        return [dict(row) for row in r]


def _pick_latest_run_dir(logs_dir: Path) -> Path:
    if not logs_dir.exists():
        raise SystemExit(f"logs dir not found: {logs_dir}")
    run_dirs = [p for p in logs_dir.iterdir() if p.is_dir() and p.name.startswith("run-")]
    if not run_dirs:
        raise SystemExit(f"no run directories found in: {logs_dir}")
    # Name is run-YYYYMMDD-HHMMSS[-suffix] so lexicographic sort works for the timestamp prefix.
    run_dirs.sort(key=lambda p: p.name)
    return run_dirs[-1]


@dataclass(frozen=True)
class SeriesStats:
    n: int
    mean: float
    std: float
    rms: float
    max_abs: float
    p95_abs: float


def _stats(x: np.ndarray) -> SeriesStats:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    n = int(np.count_nonzero(finite))
    if n == 0:
        nan = float("nan")
        return SeriesStats(n=0, mean=nan, std=nan, rms=nan, max_abs=nan, p95_abs=nan)

    xf = x[finite]
    return SeriesStats(
        n=n,
        mean=float(np.mean(xf)),
        std=float(np.std(xf)),
        rms=_nan_rms(x),
        max_abs=float(np.max(np.abs(xf))),
        p95_abs=float(np.percentile(np.abs(xf), 95)),
    )


def _summary_stats(x: np.ndarray) -> dict[str, float]:
    """
    Summary stats for (typically non-negative) scalars like alpha.
    Returns NaNs if no finite samples.
    """
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        nan = float("nan")
        return {
            "n": 0.0,
            "mean": nan,
            "std": nan,
            "min": nan,
            "p05": nan,
            "p50": nan,
            "p95": nan,
            "max": nan,
        }
    xf = x[finite]
    return {
        "n": float(xf.size),
        "mean": float(np.mean(xf)),
        "std": float(np.std(xf)),
        "min": float(np.min(xf)),
        "p05": float(np.percentile(xf, 5)),
        "p50": float(np.percentile(xf, 50)),
        "p95": float(np.percentile(xf, 95)),
        "max": float(np.max(xf)),
    }


def _save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _maybe_import_matplotlib(*, show: bool):
    try:
        import matplotlib

        # Default behavior: save-only, works on Windows/headless.
        # If --show is requested, let Matplotlib pick an interactive backend.
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401

        return plt
    except Exception as e:
        raise SystemExit(
            "matplotlib is required for plots.\n"
            "Install deps:\n"
            "  pip install -r requirements.txt\n\n"
            f"Original import error: {e}"
        )


def _enable_legend_toggle(fig) -> None:
    """
    Make legend entries clickable to toggle the corresponding plotted lines.
    Only useful for interactive backends (i.e., when running with --show).
    """
    # Build mappings per-figure so labels don't leak across figures.
    artist_to_label: dict[object, str] = {}
    label_to_lines: dict[str, list[object]] = {}
    label_to_leg_artists: dict[str, list[object]] = {}

    for ax in getattr(fig, "axes", []):
        # Map label -> lines on this axis
        for line in ax.get_lines():
            label = str(line.get_label())
            if not label or label.startswith("_"):
                continue
            label_to_lines.setdefault(label, []).append(line)

        leg = ax.get_legend()
        if leg is None:
            continue

        # Make legend handles and texts pickable.
        leg_lines = leg.get_lines()
        leg_texts = leg.get_texts()
        for legline, text in zip(leg_lines, leg_texts):
            label = str(text.get_text())
            if not label:
                continue
            for a in (legline, text):
                try:
                    a.set_picker(True)
                except Exception:
                    pass
                artist_to_label[a] = label
                label_to_leg_artists.setdefault(label, []).append(a)
            try:
                legline.set_pickradius(6)
            except Exception:
                pass

            # Dim legend entries for lines that are initially hidden.
            visible = True
            if label in label_to_lines and label_to_lines[label]:
                visible = bool(label_to_lines[label][0].get_visible())
            alpha = 1.0 if visible else 0.2
            try:
                legline.set_alpha(alpha)
                text.set_alpha(alpha)
            except Exception:
                pass

    if not artist_to_label:
        return

    def _on_pick(event) -> None:
        label = artist_to_label.get(event.artist)
        if not label:
            return
        lines = label_to_lines.get(label, [])
        if not lines:
            return
        new_vis = not bool(lines[0].get_visible())
        for ln in lines:
            try:
                ln.set_visible(new_vis)
            except Exception:
                pass
        alpha = 1.0 if new_vis else 0.2
        for a in label_to_leg_artists.get(label, []):
            try:
                a.set_alpha(alpha)
            except Exception:
                pass
        try:
            fig.canvas.draw_idle()
        except Exception:
            try:
                fig.canvas.draw()
            except Exception:
                pass

    # Note: mpl_connect returns an id; we don't need it since the figure lifetime is short.
    try:
        fig.canvas.mpl_connect("pick_event", _on_pick)
    except Exception:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description="Post-process a HILS run: tracking error + latency plots.")
    p.add_argument("run_dir", nargs="?", default="", help="Path to logs/run-*/ directory. If omitted, uses latest run in --logs-dir.")
    p.add_argument("--logs-dir", default="logs", help="Logs root dir that contains run-* folders (default: logs).")
    p.add_argument("--out-prefix", default="postprocess", help="Output file prefix (default: postprocess).")
    p.add_argument(
        "--skip-first-s",
        type=float,
        default=0.0,
        help="Ignore the first N seconds of the run when computing metrics (default: 0).",
    )
    p.add_argument(
        "--skip-until-pid-enabled",
        action="store_true",
        help="If tick.csv has pid_enabled, start metrics at the first sample where pid_enabled==1.",
    )
    p.add_argument(
        "--extra-skip-s",
        type=float,
        default=0.0,
        help="Additional seconds to skip after the analysis start condition (default: 0).",
    )
    p.add_argument(
        "--trim-plots",
        action="store_true",
        help="If set, plots start at the analysis window start (otherwise plots show full run with a vertical marker).",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Also open interactive plot windows (in addition to saving PNGs). Requires a GUI-capable Matplotlib backend.",
    )
    args = p.parse_args()

    logs_dir = Path(args.logs_dir)
    run_dir = Path(args.run_dir) if str(args.run_dir).strip() else _pick_latest_run_dir(logs_dir)
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    tick_path = run_dir / "tick.csv"
    ack_path = run_dir / "stewart_ack.csv"
    meta_path = run_dir / "run_meta.json"

    tick_rows = _load_csv_rows(tick_path)
    if not tick_rows:
        raise SystemExit(f"tick.csv missing or empty: {tick_path}")

    # Tick time axis
    t_tick_ns = np.array([_to_float(r.get("t_tick_ns")) for r in tick_rows], dtype=float)
    t0 = float(t_tick_ns[np.isfinite(t_tick_ns)][0])
    t_s = (t_tick_ns - t0) / 1e9

    # Analysis window (ignore initial transient / chaos)
    skip_first_s = float(args.skip_first_s)
    extra_skip_s = float(args.extra_skip_s)
    analysis_start_s = skip_first_s

    # Optional: start when pid_enabled flips on (if available)
    pid_enabled = np.array([_to_float(r.get("pid_enabled")) for r in tick_rows], dtype=float)
    has_pid_enabled = bool(np.any(np.isfinite(pid_enabled)))
    if bool(args.skip_until_pid_enabled) and has_pid_enabled:
        idx = np.where(np.isfinite(pid_enabled) & (pid_enabled > 0.5))[0]
        if idx.size > 0:
            analysis_start_s = max(analysis_start_s, float(t_s[int(idx[0])]))

    analysis_start_s = float(analysis_start_s + max(0.0, extra_skip_s))
    analysis_mask = np.isfinite(t_s) & (t_s >= analysis_start_s)

    # Plot window: either full run or trimmed to analysis window
    plot_mask = analysis_mask if bool(args.trim_plots) else (np.isfinite(t_s))

    # X-Plane signals (optional)
    xp_t_rx_ns = np.array([_to_float(r.get("xp_t_rx_ns")) for r in tick_rows], dtype=float)
    xp_roll = np.array([_to_float(r.get("xp_roll_deg")) for r in tick_rows], dtype=float)
    xp_pitch = np.array([_to_float(r.get("xp_pitch_deg")) for r in tick_rows], dtype=float)
    xp_heading = np.array([_to_float(r.get("xp_heading_deg")) for r in tick_rows], dtype=float)
    # Optional yaw alignment / helper columns (emitted by hils_host.py)
    xp_heading0_wrapped = np.array([_to_float(r.get("xp_heading0_wrapped_deg")) for r in tick_rows], dtype=float)
    xp_heading_wrapped = np.array([_to_float(r.get("xp_heading_wrapped_deg")) for r in tick_rows], dtype=float)

    # PX4 signals (optional)
    px_t_rx_ns = np.array([_to_float(r.get("px4_t_rx_ns")) for r in tick_rows], dtype=float)
    px_roll = np.array([_to_float(r.get("px4_roll_deg")) for r in tick_rows], dtype=float)
    px_pitch = np.array([_to_float(r.get("px4_pitch_deg")) for r in tick_rows], dtype=float)
    px_yaw = np.array([_to_float(r.get("px4_yaw_deg")) for r in tick_rows], dtype=float)
    px4_yaw_wrapped = np.array([_to_float(r.get("px4_yaw_wrapped_deg")) for r in tick_rows], dtype=float)
    px4_yaw_aligned_to_sim = np.array([_to_float(r.get("px4_yaw_aligned_to_sim_deg")) for r in tick_rows], dtype=float)

    # Commanded pose (always present in tick.csv)
    cmd_roll = np.array([_to_float(r.get("cmd_roll_deg")) for r in tick_rows], dtype=float)
    cmd_pitch = np.array([_to_float(r.get("cmd_pitch_deg")) for r in tick_rows], dtype=float)
    cmd_yaw = np.array([_to_float(r.get("cmd_yaw_deg")) for r in tick_rows], dtype=float)

    # Scenario setpoint (optional; present for trim-transition scenario runs)
    sp_pitch = np.array([_to_float(r.get("sp_pitch_deg")) for r in tick_rows], dtype=float)
    has_sp_pitch = bool(np.any(np.isfinite(sp_pitch)))

    # Latency definitions (sample age at tick, ms)
    xp_age_ms = (t_tick_ns - xp_t_rx_ns) / 1e6
    px_age_ms = (t_tick_ns - px_t_rx_ns) / 1e6
    xp_age_ms[~np.isfinite(xp_t_rx_ns)] = np.nan
    px_age_ms[~np.isfinite(px_t_rx_ns)] = np.nan

    # Tracking errors (deg)
    # - Preferred: PX4 vs X-Plane (physical vs sim) if PX4 is present
    # - Fallback: X-Plane vs cmd (software chain consistency)
    err = {}

    def _diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = a - b
        return _wrap_deg(d)

    has_px4 = bool(np.any(np.isfinite(px_roll)) or np.any(np.isfinite(px_pitch)) or np.any(np.isfinite(px_yaw)))
    has_xp = bool(np.any(np.isfinite(xp_roll)) or np.any(np.isfinite(xp_pitch)) or np.any(np.isfinite(xp_heading)))

    if has_px4 and has_xp:
        err["roll_px4_minus_xp_deg"] = _diff(px_roll, xp_roll)
        err["pitch_px4_minus_xp_deg"] = _diff(px_pitch, xp_pitch)
        # Yaw/heading: prefer bias-aligned yaw if available (eliminates constant offset).
        # - xp_heading_wrapped_deg: wrap180(sim heading) in [-180,180)
        # - px4_yaw_aligned_to_sim_deg: wrap180(px4 yaw after bias alignment)
        if np.any(np.isfinite(px4_yaw_aligned_to_sim)) and np.any(np.isfinite(xp_heading_wrapped)):
            err["yaw_px4_minus_xp_heading_deg"] = _diff(px4_yaw_aligned_to_sim, xp_heading_wrapped)
        else:
            # Fallback: raw px4 yaw vs raw xp heading (can show a large constant bias).
            err["yaw_px4_minus_xp_heading_deg"] = _diff(px_yaw, xp_heading)

    if has_xp:
        err["roll_xp_minus_cmd_deg"] = _diff(xp_roll, cmd_roll)
        err["pitch_xp_minus_cmd_deg"] = _diff(xp_pitch, cmd_pitch)
        # Commanded yaw is typically relative to initial heading; reconstruct a comparable sim-relative yaw if possible.
        if np.any(np.isfinite(xp_heading_wrapped)) and np.any(np.isfinite(xp_heading0_wrapped)):
            sim_yaw_rel = _wrap_deg(xp_heading_wrapped - xp_heading0_wrapped)
            err["heading_xp_minus_cmd_yaw_deg"] = _diff(sim_yaw_rel, cmd_yaw)
        elif np.any(np.isfinite(xp_heading_wrapped)):
            err["heading_xp_minus_cmd_yaw_deg"] = _diff(xp_heading_wrapped, cmd_yaw)
        else:
            err["heading_xp_minus_cmd_yaw_deg"] = _diff(xp_heading, cmd_yaw)

    if has_px4:
        err["roll_px4_minus_cmd_deg"] = _diff(px_roll, cmd_roll)
        err["pitch_px4_minus_cmd_deg"] = _diff(px_pitch, cmd_pitch)
        # Prefer bias-aligned yaw and compare in a sim-relative frame if possible.
        if np.any(np.isfinite(px4_yaw_aligned_to_sim)) and np.any(np.isfinite(xp_heading0_wrapped)):
            px4_yaw_rel = _wrap_deg(px4_yaw_aligned_to_sim - xp_heading0_wrapped)
            err["yaw_px4_minus_cmd_deg"] = _diff(px4_yaw_rel, cmd_yaw)
        elif np.any(np.isfinite(px4_yaw_wrapped)):
            err["yaw_px4_minus_cmd_deg"] = _diff(px4_yaw_wrapped, cmd_yaw)
        else:
            err["yaw_px4_minus_cmd_deg"] = _diff(px_yaw, cmd_yaw)

    # Stewart ACK (optional)
    ack_rows = _load_csv_rows(ack_path)
    serial_rtt_ms = None
    mcu_dt_ms = None
    ack_t_s = None
    ack_in_analysis = None
    # End-to-end latencies (optional; requires ACKs + tick join)
    e2e_xp_rx_to_ack_ms = None
    e2e_px4_rx_to_ack_ms = None
    # Saturation stats (optional; requires ACKs with sat and/or alpha)
    sat_summary: dict[str, Any] | None = None
    tick_alpha: Optional[np.ndarray] = None
    tick_sat: Optional[np.ndarray] = None
    if ack_rows:
        ack_t_rx_ns = np.array([_to_float(r.get("t_rx_ns")) for r in ack_rows], dtype=float)
        ack_t_s = (ack_t_rx_ns - t0) / 1e9
        serial_rtt_ms = np.array([_to_float(r.get("serial_rtt_ms")) for r in ack_rows], dtype=float)
        mcu_dt_ms = np.array([_to_float(r.get("dt_us")) / 1000.0 for r in ack_rows], dtype=float)

        # Saturation fields.
        # Note: Many logs use "MIN" ACKs where alpha/mag fields are NaN, but sat may still be present.
        ack_sat = np.array([_to_float(r.get("sat")) for r in ack_rows], dtype=float)
        ack_alpha = np.array([_to_float(r.get("alpha")) for r in ack_rows], dtype=float)

        # Join ACK -> tick using cmd_t_send_ns (PoseCmd.t_send_ns), which equals tick's t_tick_ns.
        cmd_t_send_ns = np.array([_to_float(r.get("cmd_t_send_ns")) for r in ack_rows], dtype=float)

        alpha_by_tick: dict[int, float] = {}
        sat_by_tick: dict[int, float] = {}
        for i, ts in enumerate(cmd_t_send_ns):
            if not np.isfinite(ts):
                continue
            k = int(ts)
            alpha_by_tick[k] = float(ack_alpha[i]) if np.isfinite(ack_alpha[i]) else float("nan")
            sat_by_tick[k] = float(ack_sat[i]) if np.isfinite(ack_sat[i]) else float("nan")

        tick_t_tick_ns_i = np.array([_to_int(r.get("t_tick_ns")) or 0 for r in tick_rows], dtype=np.int64)
        tick_alpha = np.array([alpha_by_tick.get(int(tt), float("nan")) for tt in tick_t_tick_ns_i], dtype=float)
        tick_sat = np.array([sat_by_tick.get(int(tt), float("nan")) for tt in tick_t_tick_ns_i], dtype=float)

        # Saturation summary over the analysis window (prefer alpha if available, else fall back to sat flag).
        # We filter ACK rows by whether their cmd_t_send_ns maps to a tick inside analysis_mask.
        analysis_tick_keys = set(int(x) for x in tick_t_tick_ns_i[analysis_mask])
        ack_in_analysis = np.array([int(ts) in analysis_tick_keys if np.isfinite(ts) else False for ts in cmd_t_send_ns], dtype=bool)

        ack_alpha_win = ack_alpha[ack_in_analysis]
        ack_sat_win = ack_sat[ack_in_analysis]
        ack_t_rx_ns_win = ack_t_rx_ns[ack_in_analysis]

        alpha_finite = np.isfinite(ack_alpha_win)
        sat_finite = np.isfinite(ack_sat_win)
        any_alpha = bool(np.any(alpha_finite))
        any_sat = bool(np.any(sat_finite))

        alpha_tol = 0.999  # treat alpha >= tol as "alpha==1" for float logs

        # Count-weighted saturation percentage.
        pct_saturated_count = float("nan")
        method = "none"
        if any_alpha:
            method = "alpha"
            pct_saturated_count = 100.0 * float(np.mean((ack_alpha_win[alpha_finite] < alpha_tol)))
        elif any_sat:
            method = "sat_flag"
            pct_saturated_count = 100.0 * float(np.mean((ack_sat_win[sat_finite] > 0.5)))

        # Time-weighted saturation percentage (only meaningful if ACK receive timestamps are present).
        pct_saturated_time = float("nan")
        if np.any(np.isfinite(ack_t_rx_ns_win)):
            if any_alpha:
                finite = np.isfinite(ack_t_rx_ns_win) & np.isfinite(ack_alpha_win)
                tt = ack_t_rx_ns_win[finite]
                aa = ack_alpha_win[finite]
                if tt.size >= 2:
                    dt = np.diff(tt) / 1e9
                    dt = np.clip(dt, 0.0, np.nanpercentile(dt, 99))  # cap extreme gaps
                    w = np.r_[dt, float(np.nanmedian(dt)) if np.any(np.isfinite(dt)) else 0.0]
                    denom = float(np.sum(w)) if float(np.sum(w)) > 0 else float("nan")
                    pct_saturated_time = 100.0 * float(np.sum(w[aa < alpha_tol]) / denom) if np.isfinite(denom) else float("nan")
            elif any_sat:
                finite = np.isfinite(ack_t_rx_ns_win) & np.isfinite(ack_sat_win)
                tt = ack_t_rx_ns_win[finite]
                ss = ack_sat_win[finite]
                if tt.size >= 2:
                    dt = np.diff(tt) / 1e9
                    dt = np.clip(dt, 0.0, np.nanpercentile(dt, 99))
                    w = np.r_[dt, float(np.nanmedian(dt)) if np.any(np.isfinite(dt)) else 0.0]
                    denom = float(np.sum(w)) if float(np.sum(w)) > 0 else float("nan")
                    pct_saturated_time = 100.0 * float(np.sum(w[ss > 0.5]) / denom) if np.isfinite(denom) else float("nan")

        # Alpha distribution (only if alpha samples exist).
        alpha_stats = _summary_stats(ack_alpha_win) if any_alpha else None
        alpha_hist = None
        if any_alpha:
            bins = np.linspace(0.0, 1.0, 21)  # 0.0..1.0 inclusive edges (20 bins)
            hist, edges = np.histogram(ack_alpha_win[np.isfinite(ack_alpha_win)], bins=bins)
            alpha_hist = {"bin_edges": [float(x) for x in edges], "counts": [int(x) for x in hist]}

        sat_summary = {
            "method": method,
            "alpha_tol": float(alpha_tol),
            "pct_saturated_count": float(pct_saturated_count),
            "pct_saturated_time": float(pct_saturated_time),
            "alpha_stats": alpha_stats,
            "alpha_hist": alpha_hist,
            "n_ack_total": int(len(ack_rows)),
            "n_ack_used": int(np.count_nonzero(ack_in_analysis)),
        }

        # End-to-end definition (host monotonic clock):
        #   X-Plane/PX4 receive timestamp (latest sample used at tick)
        #     -> ACK receive timestamp for the pose command sent at that tick.
        #
        # Join strategy:
        # - tick.csv row has t_tick_ns and xp_t_rx_ns / px4_t_rx_ns (the latest sample held at that tick)
        # - stewart_ack.csv row has cmd_t_send_ns (PoseCmd.t_send_ns), which equals the tick's t_tick_ns
        #
        # Note: ACK pairing in host code is FIFO; this assumes 1 ACK per pose line and no reordering.
        # (cmd_t_send_ns already parsed above for saturation join)
        # Build map: tick t_tick_ns -> latest rx times at that tick.
        tick_by_t: dict[int, tuple[float, float]] = {}
        for r in tick_rows:
            tt = _to_int(r.get("t_tick_ns"))
            if tt is None:
                continue
            tick_by_t[tt] = (_to_float(r.get("xp_t_rx_ns")), _to_float(r.get("px4_t_rx_ns")))

        xp_rx_at_send = np.full_like(cmd_t_send_ns, np.nan, dtype=float)
        px4_rx_at_send = np.full_like(cmd_t_send_ns, np.nan, dtype=float)
        for i, ts in enumerate(cmd_t_send_ns):
            if not np.isfinite(ts):
                continue
            k = int(ts)
            v = tick_by_t.get(k)
            if v is None:
                continue
            xp_rx_at_send[i] = v[0]
            px4_rx_at_send[i] = v[1]

        e2e_xp_rx_to_ack_ms = _stats_ms_from_ns_diff(ack_t_rx_ns, xp_rx_at_send)
        e2e_px4_rx_to_ack_ms = _stats_ms_from_ns_diff(ack_t_rx_ns, px4_rx_at_send)

    # Write metrics JSON
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    # Apply analysis window to ACK-derived arrays for metrics (best-effort).
    serial_rtt_ms_win = serial_rtt_ms
    mcu_dt_ms_win = mcu_dt_ms
    e2e_xp_rx_to_ack_ms_win = e2e_xp_rx_to_ack_ms
    e2e_px4_rx_to_ack_ms_win = e2e_px4_rx_to_ack_ms
    if ack_in_analysis is not None:
        if serial_rtt_ms is not None:
            serial_rtt_ms_win = np.asarray(serial_rtt_ms)[ack_in_analysis]
        if mcu_dt_ms is not None:
            mcu_dt_ms_win = np.asarray(mcu_dt_ms)[ack_in_analysis]
        if e2e_xp_rx_to_ack_ms is not None:
            e2e_xp_rx_to_ack_ms_win = np.asarray(e2e_xp_rx_to_ack_ms)[ack_in_analysis]
        if e2e_px4_rx_to_ack_ms is not None:
            e2e_px4_rx_to_ack_ms_win = np.asarray(e2e_px4_rx_to_ack_ms)[ack_in_analysis]

    metrics: dict[str, Any] = {
        "run_dir": str(run_dir),
        "meta": meta,
        "analysis_window": {
            "analysis_start_s": float(analysis_start_s),
            "skip_first_s": float(skip_first_s),
            "skip_until_pid_enabled": bool(args.skip_until_pid_enabled),
            "extra_skip_s": float(extra_skip_s),
            "trim_plots": bool(args.trim_plots),
            "has_pid_enabled": bool(has_pid_enabled),
            "n_tick_total": int(len(tick_rows)),
            "n_tick_used": int(np.count_nonzero(analysis_mask)),
        },
        "saturation": sat_summary,
        "latency_ms": {
            "xplane_sample_age": _stats(xp_age_ms[analysis_mask]).__dict__,
            "px4_sample_age": _stats(px_age_ms[analysis_mask]).__dict__,
            "serial_rtt": _stats(serial_rtt_ms_win).__dict__ if serial_rtt_ms_win is not None else None,
            "mcu_dt": _stats(mcu_dt_ms_win).__dict__ if mcu_dt_ms_win is not None else None,
            # End-to-end (sample receive -> ACK receive), using host monotonic timestamps.
            "xplane_rx_to_ack": _stats(e2e_xp_rx_to_ack_ms_win).__dict__ if e2e_xp_rx_to_ack_ms_win is not None else None,
            "px4_rx_to_ack": _stats(e2e_px4_rx_to_ack_ms_win).__dict__ if e2e_px4_rx_to_ack_ms_win is not None else None,
        },
        "tracking_error_deg": {k: _stats(v[analysis_mask]).__dict__ for k, v in err.items()},
    }

    # Tracking error conditioned on alpha (preferred) or sat flag (fallback).
    if tick_alpha is not None or tick_sat is not None:
        cond: dict[str, Any] = {}
        alpha_tol = float(sat_summary.get("alpha_tol", 0.999)) if sat_summary else 0.999

        use_alpha = bool(tick_alpha is not None and np.any(np.isfinite(tick_alpha)))
        use_sat = bool((not use_alpha) and tick_sat is not None and np.any(np.isfinite(tick_sat)))

        if use_alpha:
            a = np.asarray(tick_alpha, dtype=float)
            mask_in = analysis_mask & np.isfinite(a) & (a >= alpha_tol)
            mask_out = analysis_mask & np.isfinite(a) & (a < alpha_tol)
            cond["method"] = "alpha"
            cond["alpha_tol"] = float(alpha_tol)
        elif use_sat:
            s = np.asarray(tick_sat, dtype=float)
            mask_in = analysis_mask & np.isfinite(s) & (s <= 0.5)
            mask_out = analysis_mask & np.isfinite(s) & (s > 0.5)
            cond["method"] = "sat_flag"
        else:
            mask_in = None
            mask_out = None
            cond["method"] = "none"

        if mask_in is not None and mask_out is not None:
            cond["tracking_error_deg"] = {
                k: {"in_range": _stats(v[mask_in]).__dict__, "saturated": _stats(v[mask_out]).__dict__} for k, v in err.items()
            }
            metrics["tracking_error_deg_conditioned"] = cond

    out_prefix = str(args.out_prefix).strip() or "postprocess"
    metrics_path = run_dir / f"{out_prefix}_metrics.json"
    _save_json(metrics_path, metrics)

    # Plots
    plt = _maybe_import_matplotlib(show=bool(args.show))
    t_plot = t_s[plot_mask]

    def _mark_analysis_start(ax) -> None:
        if not bool(args.trim_plots) and float(analysis_start_s) > 0.0:
            ax.axvline(float(analysis_start_s), color="k", linewidth=1.0, alpha=0.25)

    # Pitch response plot (setpoint vs measured). High-signal for trim-transition scenario.
    if has_sp_pitch or has_xp or has_px4:
        fig0, ax0 = plt.subplots(1, 1, figsize=(11, 4.5))
        if has_sp_pitch:
            ax0.plot(t_plot, sp_pitch[plot_mask], linewidth=1.5, color="k", linestyle="--", label="setpoint pitch (deg)")
        if has_xp:
            ax0.plot(t_plot, xp_pitch[plot_mask], linewidth=1.2, color="tab:blue", label="X-Plane pitch (deg)")
        if has_px4:
            ax0.plot(t_plot, px_pitch[plot_mask], linewidth=1.2, color="tab:orange", label="PX4 pitch (deg)")
        ax0.set_title("Pitch response (trim transition)")
        ax0.set_ylabel("pitch (deg)")
        ax0.set_xlabel("time (s) from run start")
        ax0.grid(True, alpha=0.3)
        _mark_analysis_start(ax0)
        ax0.legend(loc="best")
        if bool(args.show):
            _enable_legend_toggle(fig0)
        fig0.tight_layout()
        pitch_resp_path = run_dir / f"{out_prefix}_pitch_response.png"
        fig0.savefig(pitch_resp_path, dpi=160)
        if not bool(args.show):
            plt.close(fig0)

    # Tracking error plot
    fig1, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    axes = list(axes)

    def _plot_err(ax, y: np.ndarray, label: str):
        if y is None:
            return
        ax.plot(t_plot, y[plot_mask], linewidth=1.1, label=label)

    # Choose what to emphasize in the legend/title
    title_bits = []
    if has_px4 and has_xp:
        title_bits.append("PX4 - X-Plane")
    if has_px4:
        title_bits.append("PX4 - cmd")
    if has_xp:
        title_bits.append("X-Plane - cmd")
    title = "Tracking error (deg): " + (", ".join(title_bits) if title_bits else "no data")

    if "roll_px4_minus_xp_deg" in err:
        _plot_err(axes[0], err["roll_px4_minus_xp_deg"], "roll: px4 - xp")
    if "roll_px4_minus_cmd_deg" in err:
        _plot_err(axes[0], err["roll_px4_minus_cmd_deg"], "roll: px4 - cmd")
    if "roll_xp_minus_cmd_deg" in err:
        _plot_err(axes[0], err["roll_xp_minus_cmd_deg"], "roll: xp - cmd")

    if "pitch_px4_minus_xp_deg" in err:
        _plot_err(axes[1], err["pitch_px4_minus_xp_deg"], "pitch: px4 - xp")
    if "pitch_px4_minus_cmd_deg" in err:
        _plot_err(axes[1], err["pitch_px4_minus_cmd_deg"], "pitch: px4 - cmd")
    if "pitch_xp_minus_cmd_deg" in err:
        _plot_err(axes[1], err["pitch_xp_minus_cmd_deg"], "pitch: xp - cmd")

    if "yaw_px4_minus_xp_heading_deg" in err:
        _plot_err(axes[2], err["yaw_px4_minus_xp_heading_deg"], "yaw: px4 - xp (aligned/relative if available)")
    if "yaw_px4_minus_cmd_deg" in err:
        _plot_err(axes[2], err["yaw_px4_minus_cmd_deg"], "yaw: px4 - cmd (relative if available)")
    if "heading_xp_minus_cmd_yaw_deg" in err:
        _plot_err(axes[2], err["heading_xp_minus_cmd_yaw_deg"], "yaw: xp - cmd (relative if available)")

    for ax, ylabel in zip(axes, ["roll error (deg)", "pitch error (deg)", "yaw/heading error (deg)"]):
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        _mark_analysis_start(ax)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=9)

    # Unify y-axis scale across roll/pitch/yaw in this plot (per-plot scaling).
    # Use plotted window (plot_mask) so trimming doesn't get dominated by skipped transients.
    y_max = 0.0
    for ax in axes:
        for ln in ax.get_lines():
            try:
                y = np.asarray(ln.get_ydata(), dtype=float)
            except Exception:
                continue
            if y.size == 0:
                continue
            finite = np.isfinite(y)
            if not np.any(finite):
                continue
            y_max = max(y_max, float(np.max(np.abs(y[finite]))))
    if y_max > 0.0 and np.isfinite(y_max):
        lim = float(y_max * 1.05)
        for ax in axes:
            ax.set_ylim(-lim, +lim)
    axes[-1].set_xlabel("time (s) from run start")
    fig1.suptitle(title)
    fig1.tight_layout(rect=[0, 0.02, 1, 0.96])
    if bool(args.show):
        _enable_legend_toggle(fig1)
    tracking_path = run_dir / f"{out_prefix}_tracking_error.png"
    fig1.savefig(tracking_path, dpi=160)
    if not args.show:
        plt.close(fig1)

    # Latency plot
    fig2, ax = plt.subplots(1, 1, figsize=(11, 4.5))
    ax.plot(t_plot, xp_age_ms[plot_mask], label="X-Plane sample age at tick (ms)", linewidth=1.1)
    ax.plot(t_plot, px_age_ms[plot_mask], label="PX4 sample age at tick (ms)", linewidth=1.1)
    _mark_analysis_start(ax)
    if serial_rtt_ms is not None:
        ack_mask = np.isfinite(ack_t_s)
        if bool(args.trim_plots):
            ack_mask = ack_mask & (ack_t_s >= float(analysis_start_s))
        ax.plot(ack_t_s[ack_mask], np.asarray(serial_rtt_ms)[ack_mask], label="Stewart serial RTT (ms)", linewidth=1.1)
    if mcu_dt_ms is not None:
        ack_mask = np.isfinite(ack_t_s)
        if bool(args.trim_plots):
            ack_mask = ack_mask & (ack_t_s >= float(analysis_start_s))
        ax.plot(ack_t_s[ack_mask], np.asarray(mcu_dt_ms)[ack_mask], label="Stewart MCU dt_us (ms)", linewidth=1.1)
    if e2e_xp_rx_to_ack_ms is not None:
        ack_mask = np.isfinite(ack_t_s)
        if bool(args.trim_plots):
            ack_mask = ack_mask & (ack_t_s >= float(analysis_start_s))
        ax.plot(
            ack_t_s[ack_mask],
            np.asarray(e2e_xp_rx_to_ack_ms)[ack_mask],
            label="E2E: X-Plane rx -> ACK rx (ms)",
            linewidth=1.1,
            alpha=0.85,
        )
    if e2e_px4_rx_to_ack_ms is not None:
        ack_mask = np.isfinite(ack_t_s)
        if bool(args.trim_plots):
            ack_mask = ack_mask & (ack_t_s >= float(analysis_start_s))
        ax.plot(
            ack_t_s[ack_mask],
            np.asarray(e2e_px4_rx_to_ack_ms)[ack_mask],
            label="E2E: PX4 rx -> ACK rx (ms)",
            linewidth=1.1,
            alpha=0.85,
        )
    ax.set_title("Latency / timing")
    ax.set_xlabel("time (s) from run start")
    ax.set_ylabel("ms")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig2.tight_layout()
    if bool(args.show):
        _enable_legend_toggle(fig2)
    latency_path = run_dir / f"{out_prefix}_latency.png"
    fig2.savefig(latency_path, dpi=160)
    if not args.show:
        plt.close(fig2)

    # Saturation plots (alpha(t) + histogram) if available.
    wrote_sat_plots = False
    if sat_summary is not None:
        # Prefer alpha if present; else fall back to sat.
        any_alpha = bool(sat_summary.get("alpha_stats") is not None)
        any_sat = bool(ack_rows is not None and len(ack_rows) > 0)

        if any_alpha and ack_t_s is not None:
            # alpha(t) vs ACK time
            fig3, ax3 = plt.subplots(1, 1, figsize=(11, 3.8))
            # re-parse for plot arrays (keeps code locality; overhead is tiny)
            ack_alpha_plot = np.array([_to_float(r.get("alpha")) for r in ack_rows], dtype=float)
            ack_mask = np.isfinite(ack_t_s)
            if bool(args.trim_plots):
                ack_mask = ack_mask & (ack_t_s >= float(analysis_start_s))
            ax3.plot(ack_t_s[ack_mask], ack_alpha_plot[ack_mask], linewidth=1.0, label="alpha (ACK)")
            ax3.axhline(1.0, color="k", linewidth=0.8, alpha=0.4)
            ax3.set_title("Saturation scale alpha(t) (1.0 = no scaling)")
            ax3.set_xlabel("time (s) from run start")
            ax3.set_ylabel("alpha")
            ax3.set_ylim(0.0, 1.05)
            ax3.grid(True, alpha=0.25)
            _mark_analysis_start(ax3)
            ax3.legend(loc="lower left", fontsize=9)
            fig3.tight_layout()
            if bool(args.show):
                _enable_legend_toggle(fig3)
            p3 = run_dir / f"{out_prefix}_alpha_time.png"
            fig3.savefig(p3, dpi=160)
            if not args.show:
                plt.close(fig3)
            wrote_sat_plots = True

            # Histogram
            fig4, ax4 = plt.subplots(1, 1, figsize=(7.5, 4.0))
            finite = np.isfinite(ack_alpha_plot)
            ax4.hist(ack_alpha_plot[finite], bins=np.linspace(0.0, 1.0, 21), edgecolor="k", alpha=0.85)
            ax4.set_title("Alpha distribution (saturation scale)")
            ax4.set_xlabel("alpha")
            ax4.set_ylabel("count")
            ax4.grid(True, alpha=0.25)
            fig4.tight_layout()
            if bool(args.show):
                _enable_legend_toggle(fig4)
            p4 = run_dir / f"{out_prefix}_alpha_hist.png"
            fig4.savefig(p4, dpi=160)
            if not args.show:
                plt.close(fig4)
            wrote_sat_plots = True

            # Optional: overlay command magnitude with alpha on the tick timeline (only if we successfully joined tick_alpha).
            if tick_alpha is not None and np.any(np.isfinite(tick_alpha)):
                a_tick = np.asarray(tick_alpha, dtype=float)
                cmd_mag = np.nanmax(np.vstack([np.abs(cmd_roll), np.abs(cmd_pitch), np.abs(cmd_yaw)]), axis=0)

                fig5, ax5 = plt.subplots(1, 1, figsize=(11, 4.0))
                ax5.plot(
                    t_plot,
                    cmd_mag[plot_mask],
                    linewidth=1.0,
                    color="tab:blue",
                    label="cmd magnitude (deg): max(|roll|,|pitch|,|yaw|)",
                )
                ax5.set_xlabel("time (s) from run start")
                ax5.set_ylabel("cmd magnitude (deg)", color="tab:blue")
                ax5.tick_params(axis="y", labelcolor="tab:blue")
                ax5.grid(True, alpha=0.25)
                _mark_analysis_start(ax5)

                ax5b = ax5.twinx()
                ax5b.plot(t_plot, a_tick[plot_mask], linewidth=1.0, color="tab:orange", label="alpha (joined to tick)")
                ax5b.set_ylabel("alpha", color="tab:orange")
                ax5b.tick_params(axis="y", labelcolor="tab:orange")
                ax5b.set_ylim(0.0, 1.05)

                # combined legend
                h1, l1 = ax5.get_legend_handles_labels()
                h2, l2 = ax5b.get_legend_handles_labels()
                ax5.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
                fig5.suptitle("Command magnitude vs saturation scale alpha")
                fig5.tight_layout(rect=[0, 0.02, 1, 0.96])
                if bool(args.show):
                    _enable_legend_toggle(fig5)
                p5 = run_dir / f"{out_prefix}_alpha_vs_cmd.png"
                fig5.savefig(p5, dpi=160)
                if not args.show:
                    plt.close(fig5)
                wrote_sat_plots = True

        elif any_sat and ack_t_s is not None:
            # sat(t) vs ACK time (for MIN logs where alpha is NaN)
            fig3, ax3 = plt.subplots(1, 1, figsize=(11, 3.2))
            ack_sat_plot = np.array([_to_float(r.get("sat")) for r in ack_rows], dtype=float)
            ack_mask = np.isfinite(ack_t_s)
            if bool(args.trim_plots):
                ack_mask = ack_mask & (ack_t_s >= float(analysis_start_s))
            ax3.step(ack_t_s[ack_mask], ack_sat_plot[ack_mask], where="post", linewidth=1.0, label="sat flag (ACK)")
            ax3.set_title("Saturation flag sat(t)")
            ax3.set_xlabel("time (s) from run start")
            ax3.set_ylabel("sat (0/1)")
            ax3.set_ylim(-0.1, 1.1)
            ax3.grid(True, alpha=0.25)
            _mark_analysis_start(ax3)
            ax3.legend(loc="upper right", fontsize=9)
            fig3.tight_layout()
            if bool(args.show):
                _enable_legend_toggle(fig3)
            p3 = run_dir / f"{out_prefix}_sat_time.png"
            fig3.savefig(p3, dpi=160)
            if not args.show:
                plt.close(fig3)
            wrote_sat_plots = True

    print(f"[postprocess] run_dir: {run_dir}")
    print(f"[postprocess] wrote: {metrics_path}")
    print(f"[postprocess] wrote: {tracking_path}")
    print(f"[postprocess] wrote: {latency_path}")
    if wrote_sat_plots:
        print(f"[postprocess] wrote: {out_prefix}_alpha_time.png / {out_prefix}_alpha_hist.png (if alpha present) OR {out_prefix}_sat_time.png")
    if args.show:
        print("[postprocess] showing plots (close windows to exit)...")
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


