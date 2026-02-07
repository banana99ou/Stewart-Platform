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


def main() -> int:
    p = argparse.ArgumentParser(description="Post-process a HILS run: tracking error + latency plots.")
    p.add_argument("run_dir", nargs="?", default="", help="Path to logs/run-*/ directory. If omitted, uses latest run in --logs-dir.")
    p.add_argument("--logs-dir", default="logs", help="Logs root dir that contains run-* folders (default: logs).")
    p.add_argument("--out-prefix", default="postprocess", help="Output file prefix (default: postprocess).")
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

    # X-Plane signals (optional)
    xp_t_rx_ns = np.array([_to_float(r.get("xp_t_rx_ns")) for r in tick_rows], dtype=float)
    xp_roll = np.array([_to_float(r.get("xp_roll_deg")) for r in tick_rows], dtype=float)
    xp_pitch = np.array([_to_float(r.get("xp_pitch_deg")) for r in tick_rows], dtype=float)
    xp_heading = np.array([_to_float(r.get("xp_heading_deg")) for r in tick_rows], dtype=float)

    # PX4 signals (optional)
    px_t_rx_ns = np.array([_to_float(r.get("px4_t_rx_ns")) for r in tick_rows], dtype=float)
    px_roll = np.array([_to_float(r.get("px4_roll_deg")) for r in tick_rows], dtype=float)
    px_pitch = np.array([_to_float(r.get("px4_pitch_deg")) for r in tick_rows], dtype=float)
    px_yaw = np.array([_to_float(r.get("px4_yaw_deg")) for r in tick_rows], dtype=float)

    # Commanded pose (always present in tick.csv)
    cmd_roll = np.array([_to_float(r.get("cmd_roll_deg")) for r in tick_rows], dtype=float)
    cmd_pitch = np.array([_to_float(r.get("cmd_pitch_deg")) for r in tick_rows], dtype=float)
    cmd_yaw = np.array([_to_float(r.get("cmd_yaw_deg")) for r in tick_rows], dtype=float)

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
        # NOTE: xp provides heading; px4 provides yaw. They should be comparable for small pitch/roll.
        err["yaw_px4_minus_xp_heading_deg"] = _diff(px_yaw, xp_heading)

    if has_xp:
        err["roll_xp_minus_cmd_deg"] = _diff(xp_roll, cmd_roll)
        err["pitch_xp_minus_cmd_deg"] = _diff(xp_pitch, cmd_pitch)
        err["heading_xp_minus_cmd_yaw_deg"] = _diff(xp_heading, cmd_yaw)

    if has_px4:
        err["roll_px4_minus_cmd_deg"] = _diff(px_roll, cmd_roll)
        err["pitch_px4_minus_cmd_deg"] = _diff(px_pitch, cmd_pitch)
        err["yaw_px4_minus_cmd_deg"] = _diff(px_yaw, cmd_yaw)

    # Stewart ACK (optional)
    ack_rows = _load_csv_rows(ack_path)
    serial_rtt_ms = None
    mcu_dt_ms = None
    ack_t_s = None
    # End-to-end latencies (optional; requires ACKs + tick join)
    e2e_xp_rx_to_ack_ms = None
    e2e_px4_rx_to_ack_ms = None
    if ack_rows:
        ack_t_rx_ns = np.array([_to_float(r.get("t_rx_ns")) for r in ack_rows], dtype=float)
        ack_t_s = (ack_t_rx_ns - t0) / 1e9
        serial_rtt_ms = np.array([_to_float(r.get("serial_rtt_ms")) for r in ack_rows], dtype=float)
        mcu_dt_ms = np.array([_to_float(r.get("dt_us")) / 1000.0 for r in ack_rows], dtype=float)

        # End-to-end definition (host monotonic clock):
        #   X-Plane/PX4 receive timestamp (latest sample used at tick)
        #     -> ACK receive timestamp for the pose command sent at that tick.
        #
        # Join strategy:
        # - tick.csv row has t_tick_ns and xp_t_rx_ns / px4_t_rx_ns (the latest sample held at that tick)
        # - stewart_ack.csv row has cmd_t_send_ns (PoseCmd.t_send_ns), which equals the tick's t_tick_ns
        #
        # Note: ACK pairing in host code is FIFO; this assumes 1 ACK per pose line and no reordering.
        cmd_t_send_ns = np.array([_to_float(r.get("cmd_t_send_ns")) for r in ack_rows], dtype=float)
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

    metrics: dict[str, Any] = {
        "run_dir": str(run_dir),
        "meta": meta,
        "latency_ms": {
            "xplane_sample_age": _stats(xp_age_ms).__dict__,
            "px4_sample_age": _stats(px_age_ms).__dict__,
            "serial_rtt": _stats(serial_rtt_ms).__dict__ if serial_rtt_ms is not None else None,
            "mcu_dt": _stats(mcu_dt_ms).__dict__ if mcu_dt_ms is not None else None,
            # End-to-end (sample receive -> ACK receive), using host monotonic timestamps.
            "xplane_rx_to_ack": _stats(e2e_xp_rx_to_ack_ms).__dict__ if e2e_xp_rx_to_ack_ms is not None else None,
            "px4_rx_to_ack": _stats(e2e_px4_rx_to_ack_ms).__dict__ if e2e_px4_rx_to_ack_ms is not None else None,
        },
        "tracking_error_deg": {k: _stats(v).__dict__ for k, v in err.items()},
    }

    out_prefix = str(args.out_prefix).strip() or "postprocess"
    metrics_path = run_dir / f"{out_prefix}_metrics.json"
    _save_json(metrics_path, metrics)

    # Plots
    plt = _maybe_import_matplotlib(show=bool(args.show))

    # Tracking error plot
    fig1, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    axes = list(axes)

    def _plot_err(ax, y: np.ndarray, label: str):
        if y is None:
            return
        ax.plot(t_s, y, linewidth=1.1, label=label)

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
        _plot_err(axes[2], err["yaw_px4_minus_xp_heading_deg"], "yaw: px4 - xp(heading)")
    if "yaw_px4_minus_cmd_deg" in err:
        _plot_err(axes[2], err["yaw_px4_minus_cmd_deg"], "yaw: px4 - cmd")
    if "heading_xp_minus_cmd_yaw_deg" in err:
        _plot_err(axes[2], err["heading_xp_minus_cmd_yaw_deg"], "heading: xp - cmd(yaw)")

    for ax, ylabel in zip(axes, ["roll error (deg)", "pitch error (deg)", "yaw/heading error (deg)"]):
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("time (s) from run start")
    fig1.suptitle(title)
    fig1.tight_layout(rect=[0, 0.02, 1, 0.96])
    tracking_path = run_dir / f"{out_prefix}_tracking_error.png"
    fig1.savefig(tracking_path, dpi=160)
    if not args.show:
        plt.close(fig1)

    # Latency plot
    fig2, ax = plt.subplots(1, 1, figsize=(11, 4.5))
    ax.plot(t_s, xp_age_ms, label="X-Plane sample age at tick (ms)", linewidth=1.1)
    ax.plot(t_s, px_age_ms, label="PX4 sample age at tick (ms)", linewidth=1.1)
    if serial_rtt_ms is not None:
        ax.plot(ack_t_s, serial_rtt_ms, label="Stewart serial RTT (ms)", linewidth=1.1)
    if mcu_dt_ms is not None:
        ax.plot(ack_t_s, mcu_dt_ms, label="Stewart MCU dt_us (ms)", linewidth=1.1)
    if e2e_xp_rx_to_ack_ms is not None:
        ax.plot(ack_t_s, e2e_xp_rx_to_ack_ms, label="E2E: X-Plane rx -> ACK rx (ms)", linewidth=1.1, alpha=0.85)
    if e2e_px4_rx_to_ack_ms is not None:
        ax.plot(ack_t_s, e2e_px4_rx_to_ack_ms, label="E2E: PX4 rx -> ACK rx (ms)", linewidth=1.1, alpha=0.85)
    ax.set_title("Latency / timing")
    ax.set_xlabel("time (s) from run start")
    ax.set_ylabel("ms")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig2.tight_layout()
    latency_path = run_dir / f"{out_prefix}_latency.png"
    fig2.savefig(latency_path, dpi=160)
    if not args.show:
        plt.close(fig2)

    print(f"[postprocess] run_dir: {run_dir}")
    print(f"[postprocess] wrote: {metrics_path}")
    print(f"[postprocess] wrote: {tracking_path}")
    print(f"[postprocess] wrote: {latency_path}")
    if args.show:
        print("[postprocess] showing plots (close windows to exit)...")
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


