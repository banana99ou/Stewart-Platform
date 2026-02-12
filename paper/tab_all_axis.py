"""
paper/tab_all_axis.py

Standalone script to compute all-axis (roll, pitch, yaw) error metrics for runs #5–#9.
Outputs a markdown table matching the format of Table 1 but for all axes.

Uses same logic as Fig_Gen.py (hold_0, skip_first_s=10).
Does NOT modify Fig_Gen.py or any existing code.

Run: python paper/tab_all_axis.py
"""

import csv
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
SKIP_FIRST_S = 10.0

BASELINE_RUN_DIRS = [
    LOGS_DIR / "run-20260208-202549 #5 (fix mounting)",
    LOGS_DIR / "run-20260208-210113 #6",
    LOGS_DIR / "run-20260208-210614 #7",
    LOGS_DIR / "run-20260208-210808 #8",
    LOGS_DIR / "run-20260208-211109 #9",
]


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


def compute_run_metrics(run_dir: Path) -> dict:
    tick_path = run_dir / "tick.csv"
    tick_rows = _read_csv_dicts(tick_path)
    if not tick_rows:
        raise RuntimeError(f"tick.csv missing/empty: {tick_path}")

    t_tick_ns = np.array([_to_float(r.get("t_tick_ns")) for r in tick_rows], dtype=float)
    t0 = float(t_tick_ns[np.isfinite(t_tick_ns)][0])
    t_s = (t_tick_ns - t0) / 1e9
    analysis_mask = np.isfinite(t_s) & (t_s >= float(SKIP_FIRST_S))

    scenario_phase = np.array([str(r.get("scenario_phase") or "").strip() for r in tick_rows], dtype=object)
    pid_enabled = np.array([_to_float(r.get("pid_enabled")) for r in tick_rows], dtype=float)

    xp_roll = np.array([_to_float(r.get("xp_roll_deg")) for r in tick_rows], dtype=float)
    xp_pitch = np.array([_to_float(r.get("xp_pitch_deg")) for r in tick_rows], dtype=float)
    xp_heading_wrapped = np.array([_to_float(r.get("xp_heading_wrapped_deg")) for r in tick_rows], dtype=float)
    xp_heading0_wrapped = np.array([_to_float(r.get("xp_heading0_wrapped_deg")) for r in tick_rows], dtype=float)

    px4_roll = np.array([_to_float(r.get("px4_roll_deg")) for r in tick_rows], dtype=float)
    px4_pitch = np.array([_to_float(r.get("px4_pitch_deg")) for r in tick_rows], dtype=float)
    px4_yaw = np.array([_to_float(r.get("px4_yaw_deg")) for r in tick_rows], dtype=float)
    px4_yaw_aligned = np.array([_to_float(r.get("px4_yaw_aligned_to_sim_deg")) for r in tick_rows], dtype=float)

    cmd_roll = np.array([_to_float(r.get("cmd_roll_deg")) for r in tick_rows], dtype=float)
    cmd_pitch = np.array([_to_float(r.get("cmd_pitch_deg")) for r in tick_rows], dtype=float)
    cmd_yaw = np.array([_to_float(r.get("cmd_yaw_deg")) for r in tick_rows], dtype=float)

    mask_bias = analysis_mask & (scenario_phase == "hold_0") & np.isfinite(pid_enabled) & (pid_enabled > 0.5)

    def _bias_mean(meas: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
        d = _wrap_deg(meas - ref)
        m = mask_bias & np.isfinite(d)
        if not np.any(m):
            return float("nan"), float("nan")
        df = d[m]
        return float(np.mean(df)), float(np.std(df))

    b_roll_mean, b_roll_std = _bias_mean(px4_roll, cmd_roll)
    b_pitch_mean, b_pitch_std = _bias_mean(px4_pitch, cmd_pitch)

    # Yaw: use px4_yaw_aligned_to_sim vs xp_heading_wrapped when available; else px4 vs cmd
    has_aligned = np.any(np.isfinite(px4_yaw_aligned)) and np.any(np.isfinite(xp_heading_wrapped))
    if has_aligned:
        h0 = float(np.nanmean(xp_heading0_wrapped))
        px4_yaw_rel = _wrap_deg(px4_yaw_aligned - h0)
        d_yaw = _wrap_deg(px4_yaw_rel - cmd_yaw)
    else:
        d_yaw = _wrap_deg(px4_yaw - cmd_yaw)
    m_yaw = mask_bias & np.isfinite(d_yaw)
    b_yaw_mean = float(np.mean(d_yaw[m_yaw])) if np.any(m_yaw) else float("nan")
    b_yaw_std = float(np.std(d_yaw[m_yaw])) if np.any(m_yaw) else float("nan")

    m = analysis_mask

    def _err_stats(err: np.ndarray) -> tuple[float, float]:
        e = err[m]
        e = e[np.isfinite(e)]
        if e.size == 0:
            return float("nan"), float("nan")
        return float(np.sqrt(np.mean(e**2))), float(np.mean(e))

    # Raw: PX4 - X-Plane
    err_raw_roll = _wrap_deg(px4_roll - xp_roll)
    err_raw_pitch = _wrap_deg(px4_pitch - xp_pitch)
    if has_aligned:
        err_raw_yaw = _wrap_deg(px4_yaw_aligned - xp_heading_wrapped)
    else:
        err_raw_yaw = _wrap_deg(px4_yaw - xp_heading_wrapped)

    raw_rms_roll, raw_mean_roll = _err_stats(err_raw_roll)
    raw_rms_pitch, raw_mean_pitch = _err_stats(err_raw_pitch)
    raw_rms_yaw, raw_mean_yaw = _err_stats(err_raw_yaw)

    # Bias-corrected: (PX4 - b) - X-Plane
    err_bc_roll = _wrap_deg((px4_roll - b_roll_mean) - xp_roll) if np.isfinite(b_roll_mean) else err_raw_roll
    err_bc_pitch = _wrap_deg((px4_pitch - b_pitch_mean) - xp_pitch) if np.isfinite(b_pitch_mean) else err_raw_pitch
    if has_aligned:
        h0 = float(np.nanmean(xp_heading0_wrapped))
        px4_yaw_rel = _wrap_deg(px4_yaw_aligned - h0)
        sim_yaw_rel = _wrap_deg(xp_heading_wrapped - xp_heading0_wrapped)
        err_bc_yaw = _wrap_deg((px4_yaw_rel - b_yaw_mean) - sim_yaw_rel) if np.isfinite(b_yaw_mean) else err_raw_yaw
    else:
        err_bc_yaw = _wrap_deg((px4_yaw - b_yaw_mean) - xp_heading_wrapped) if np.isfinite(b_yaw_mean) else err_raw_yaw

    bc_rms_roll, bc_mean_roll = _err_stats(err_bc_roll)
    bc_rms_pitch, bc_mean_pitch = _err_stats(err_bc_pitch)
    bc_rms_yaw, bc_mean_yaw = _err_stats(err_bc_yaw)

    # Single number across all axes: 3D RMS = sqrt(mean(e_roll^2 + e_pitch^2 + e_yaw^2))
    def _r3(e_r: np.ndarray, e_p: np.ndarray, e_y: np.ndarray) -> tuple[float, float]:
        ok = m & np.isfinite(e_r) & np.isfinite(e_p) & np.isfinite(e_y)
        if not np.any(ok):
            return float("nan"), float("nan")
        sq = e_r[ok] ** 2 + e_p[ok] ** 2 + e_y[ok] ** 2
        mag = np.sqrt(np.maximum(sq, 0.0))
        return float(np.sqrt(np.mean(sq))), float(np.mean(mag))

    raw_rms_3d, raw_mean_3d = _r3(err_raw_roll, err_raw_pitch, err_raw_yaw)
    bc_rms_3d, bc_mean_3d = _r3(err_bc_roll, err_bc_pitch, err_bc_yaw)

    return {
        "b_roll": (b_roll_mean, b_roll_std),
        "b_pitch": (b_pitch_mean, b_pitch_std),
        "b_yaw": (b_yaw_mean, b_yaw_std),
        "raw": {"roll": (raw_rms_roll, raw_mean_roll), "pitch": (raw_rms_pitch, raw_mean_pitch), "yaw": (raw_rms_yaw, raw_mean_yaw)},
        "bc": {"roll": (bc_rms_roll, bc_mean_roll), "pitch": (bc_rms_pitch, bc_mean_pitch), "yaw": (bc_rms_yaw, bc_mean_yaw)},
        "raw_3d": (raw_rms_3d, raw_mean_3d),
        "bc_3d": (bc_rms_3d, bc_mean_3d),
    }


def main() -> int:
    runs = []
    for run_dir in BASELINE_RUN_DIRS:
        if not run_dir.exists():
            print(f"Missing: {run_dir}")
            continue
        run_num = run_dir.name.split("#")[-1].strip().split()[0]

        try:
            m = compute_run_metrics(run_dir)
            runs.append((run_num, m))
        except Exception as e:
            print(f"Error {run_dir}: {e}")
            raise

    # Single table: one error number across all axes (3D RMS)
    print("\n### Table: All-axis error (runs #5–#9)")
    print("| Run | raw RMS (deg) | raw mean (deg) | bias-corrected RMS (deg) | bias-corrected mean (deg) |")
    print("|---:|---:|---:|---:|---:|")
    for run_num, m in runs:
        raw_rms, raw_mean = m["raw_3d"]
        bc_rms, bc_mean = m["bc_3d"]
        print(f"| \\#{run_num} | {raw_rms:.4f} | {raw_mean:.4f} | {bc_rms:.4f} | {bc_mean:.4f} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
