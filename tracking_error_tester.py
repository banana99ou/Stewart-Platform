"""
tracking_error_tester.py

Purpose
-------
Run an attitude tracking sweep to diagnose Euler coupling / axis convention issues:
- Connect to PX4 via MAVLink (ATTITUDE message).
- Connect to the Stewart platform ESP32 via serial (pose lines + ACK parsing).
- Zero PX4 yaw to the first received yaw sample (so startup heading becomes yaw=0).
- Command a set of (roll, pitch) steps at multiple yaw angles and measure tracking error.

This is especially useful for the suspected issue:
  "platform pitch command is not observed as PX4 pitch when yaw is high"

Outputs
-------
Creates a new run directory under `logs/`:
- run_meta.json
- summary.csv          (one row per test case)
- px4_att_samples.csv  (all PX4 samples collected during each hold window)
- stewart_ack.csv      (ACK lines paired with sent commands; best-effort)

Dependencies
------------
  pip install pyserial pymavlink numpy

Typical usage (Windows)
-----------------------
  python tracking_error_tester.py --stewart-com COM13 --px4-com COM9
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Reuse proven comms + helpers from the existing host script.
from hils_host import (  # type: ignore
    PoseCmd,
    Px4Mavlink,
    StewartSerial,
    angle_error_deg,
    monotonic_ns,
    normalize_serial_port,
    utc_iso,
    wrap180,
)


@dataclass(frozen=True)
class Case:
    case_id: int
    roll_cmd_deg: float
    pitch_cmd_deg: float
    yaw_cmd_deg: float
    settle_s: float
    hold_s: float


def _mkdir_run(log_dir: str, run_name: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    suffix = f"-{run_name.strip()}" if run_name.strip() else ""
    d = Path(log_dir) / f"run-{stamp}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _mean_std(x: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(x)), float(np.std(x))


def build_cases(*, yaw_list: list[float], step_deg: float, settle_s: float, hold_s: float) -> list[Case]:
    """
    Test plan focused on roll/pitch at multiple yaw.
    Includes control cases (roll-only) to detect axis swapping.
    """
    cases: list[Case] = []
    cid = 0
    for yaw in yaw_list:
        for pitch in (0.0, +step_deg, -step_deg):
            if pitch == 0.0:
                continue
            cid += 1
            cases.append(Case(cid, roll_cmd_deg=0.0, pitch_cmd_deg=pitch, yaw_cmd_deg=yaw, settle_s=settle_s, hold_s=hold_s))
        for roll in (0.0, +step_deg, -step_deg):
            if roll == 0.0:
                continue
            cid += 1
            cases.append(Case(cid, roll_cmd_deg=roll, pitch_cmd_deg=0.0, yaw_cmd_deg=yaw, settle_s=settle_s, hold_s=hold_s))
        # Combined case (optional, helps show coupling)
        cid += 1
        cases.append(Case(cid, roll_cmd_deg=+step_deg, pitch_cmd_deg=+step_deg, yaw_cmd_deg=yaw, settle_s=settle_s, hold_s=hold_s))
        cid += 1
        cases.append(Case(cid, roll_cmd_deg=-step_deg, pitch_cmd_deg=+step_deg, yaw_cmd_deg=yaw, settle_s=settle_s, hold_s=hold_s))
    return cases


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PX4 <-> Stewart tracking error tester (yaw zeroed to first PX4 yaw).")
    p.add_argument(
        "run_dir",
        nargs="?",
        default="",
        help="If provided, only generate plots from an existing run directory (must contain summary.csv) and exit.",
    )
    p.add_argument("--stewart-com", default=r"\\.\COM14", help="ESP32 serial port, e.g. COM13")
    p.add_argument("--stewart-baud", type=int, default=115200)
    p.add_argument("--stewart-open-delay-s", type=float, default=1.5)
    p.add_argument("--px4-com", default=r"\\.\COM13", help="PX4 MAVLink serial port, e.g. COM9")
    p.add_argument("--px4-baud", type=int, default=115200)
    p.add_argument("--px4-att-hz", type=float, default=100.0)
    p.add_argument("--px4-dialect", default="common")
    p.add_argument("--pose-z-mm", type=float, default=0.0, help="Platform z (mm) used for all cases")

    p.add_argument("--yaw-list", default="0,15,30,40", help="Comma-separated yaw test values (deg)")
    p.add_argument("--step-deg", type=float, default=10.0, help="Roll/pitch step (deg)")
    p.add_argument("--settle-s", type=float, default=1.0, help="Settle time after commanding a case")
    p.add_argument("--hold-s", type=float, default=2.0, help="Hold time window for error statistics")

    p.add_argument("--log-dir", default="logs")
    p.add_argument("--run-name", default="tracking-test")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    p.add_argument("--home-wait-s", type=float, default=1.0, help="Initial wait at home pose for yaw0 capture")
    p.add_argument("--yaw-zero", action="store_true", default=True, help="Zero yaw to first PX4 yaw sample (default on).")
    p.add_argument(
        "--zero-rp",
        action="store_true",
        default=True,
        help="Also zero roll/pitch to the first PX4 roll/pitch samples (default on).",
    )
    return p.parse_args()


def _read_summary_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"summary.csv not found: {path}")
    rows: list[dict] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(dict(row))
    return rows


def _read_ack_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(dict(row))
    return rows


def _to_float(row: dict, key: str) -> float:
    v = row.get(key, "")
    try:
        return float(v)
    except Exception:
        return float("nan")


def plot_run(run_dir: Path) -> Path:
    """
    Generate a couple of high-signal plots to reveal coupling vs yaw.
    Writes PNGs into run_dir and returns run_dir.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.csv"
    rows = _read_summary_csv(summary_path)
    if not rows:
        raise RuntimeError(f"No rows in {summary_path}")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting. Install it: pip install matplotlib") from e

    # Classify cases
    pitch_only = [r for r in rows if abs(_to_float(r, "cmd_roll_deg")) < 1e-6 and abs(_to_float(r, "cmd_pitch_deg")) > 1e-6]
    roll_only = [r for r in rows if abs(_to_float(r, "cmd_pitch_deg")) < 1e-6 and abs(_to_float(r, "cmd_roll_deg")) > 1e-6]

    def series(case_rows: list[dict], xk: str, yk: str) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([_to_float(r, xk) for r in case_rows], dtype=float)
        y = np.array([_to_float(r, yk) for r in case_rows], dtype=float)
        # sort by x for nicer lines
        idx = np.argsort(x)
        return x[idx], y[idx]

    def split_by_sign(case_rows: list[dict], sign_key: str) -> tuple[list[dict], list[dict]]:
        pos = [r for r in case_rows if _to_float(r, sign_key) > 0]
        neg = [r for r in case_rows if _to_float(r, sign_key) < 0]
        return pos, neg

    # Figure 1: error vs yaw, split by +step / -step
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    pp, pn = split_by_sign(pitch_only, "cmd_pitch_deg")
    rp, rn = split_by_sign(roll_only, "cmd_roll_deg")

    # Pitch-only: pitch error vs yaw
    x, y = series(pp, "cmd_yaw_deg", "err_pitch_deg")
    ax[0, 0].plot(x, y, "o-", label="+pitch")
    x, y = series(pn, "cmd_yaw_deg", "err_pitch_deg")
    ax[0, 0].plot(x, y, "o-", label="-pitch")
    ax[0, 0].axhline(0, color="k", linewidth=1, alpha=0.3)
    ax[0, 0].set_title("Pitch-only: pitch error vs yaw")
    ax[0, 0].set_ylabel("err_pitch (deg)")
    ax[0, 0].legend()
    ax[0, 0].grid(True, alpha=0.3)

    # Pitch-only: roll error vs yaw (cross-axis coupling)
    x, y = series(pp, "cmd_yaw_deg", "err_roll_deg")
    ax[1, 0].plot(x, y, "o-", label="+pitch")
    x, y = series(pn, "cmd_yaw_deg", "err_roll_deg")
    ax[1, 0].plot(x, y, "o-", label="-pitch")
    ax[1, 0].axhline(0, color="k", linewidth=1, alpha=0.3)
    ax[1, 0].set_title("Pitch-only: roll error vs yaw (coupling)")
    ax[1, 0].set_xlabel("cmd_yaw (deg)")
    ax[1, 0].set_ylabel("err_roll (deg)")
    ax[1, 0].legend()
    ax[1, 0].grid(True, alpha=0.3)

    # Roll-only: roll error vs yaw
    x, y = series(rp, "cmd_yaw_deg", "err_roll_deg")
    ax[0, 1].plot(x, y, "o-", label="+roll")
    x, y = series(rn, "cmd_yaw_deg", "err_roll_deg")
    ax[0, 1].plot(x, y, "o-", label="-roll")
    ax[0, 1].axhline(0, color="k", linewidth=1, alpha=0.3)
    ax[0, 1].set_title("Roll-only: roll error vs yaw")
    ax[0, 1].legend()
    ax[0, 1].grid(True, alpha=0.3)

    # Roll-only: pitch error vs yaw (cross-axis coupling)
    x, y = series(rp, "cmd_yaw_deg", "err_pitch_deg")
    ax[1, 1].plot(x, y, "o-", label="+roll")
    x, y = series(rn, "cmd_yaw_deg", "err_pitch_deg")
    ax[1, 1].plot(x, y, "o-", label="-roll")
    ax[1, 1].axhline(0, color="k", linewidth=1, alpha=0.3)
    ax[1, 1].set_title("Roll-only: pitch error vs yaw (coupling)")
    ax[1, 1].set_xlabel("cmd_yaw (deg)")
    ax[1, 1].legend()
    ax[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    out1 = run_dir / "tracking_error_vs_yaw.png"
    fig.savefig(out1, dpi=160)
    plt.close(fig)

    # Figure 2: yaw error vs yaw (sanity check; estimator yaw can be weird)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
    x_all, y_all = series(rows, "cmd_yaw_deg", "err_yaw_deg")
    ax2.plot(x_all, y_all, "o-")
    ax2.axhline(0, color="k", linewidth=1, alpha=0.3)
    ax2.set_title("Yaw tracking: err_yaw vs cmd_yaw (relative yaw)")
    ax2.set_xlabel("cmd_yaw (deg)")
    ax2.set_ylabel("err_yaw (deg)")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = run_dir / "yaw_error_vs_yaw.png"
    fig2.savefig(out2, dpi=160)
    plt.close(fig2)

    # Figure 3: saturation scaling alpha vs yaw (pulled from stewart_ack.csv)
    ack_rows = _read_ack_csv(run_dir / "stewart_ack.csv")
    ack_full = [r for r in ack_rows if str(r.get("status", "")).strip() and str(r.get("status", "")).strip().upper() != "MIN"]

    # If ACK has cmd_* fields (it should), use them.
    def ack_pitch_only(rows_: list[dict]) -> list[dict]:
        out = []
        for r in rows_:
            if abs(_to_float(r, "cmd_roll_deg")) < 1e-6 and abs(_to_float(r, "cmd_pitch_deg")) > 1e-6:
                out.append(r)
        return out

    def ack_roll_only(rows_: list[dict]) -> list[dict]:
        out = []
        for r in rows_:
            if abs(_to_float(r, "cmd_pitch_deg")) < 1e-6 and abs(_to_float(r, "cmd_roll_deg")) > 1e-6:
                out.append(r)
        return out

    if ack_full:
        fig3, ax3 = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
        p = ack_pitch_only(ack_full)
        r = ack_roll_only(ack_full)

        def split_by_sign_ack(case_rows: list[dict], sign_key: str) -> tuple[list[dict], list[dict]]:
            pos = [x for x in case_rows if _to_float(x, sign_key) > 0]
            neg = [x for x in case_rows if _to_float(x, sign_key) < 0]
            return pos, neg

        pp, pn = split_by_sign_ack(p, "cmd_pitch_deg")
        rp, rn = split_by_sign_ack(r, "cmd_roll_deg")

        def series_ack(case_rows: list[dict], xk: str, yk: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            x = np.array([_to_float(rr, xk) for rr in case_rows], dtype=float)
            y = np.array([_to_float(rr, yk) for rr in case_rows], dtype=float)
            sat = np.array([_to_float(rr, "sat") for rr in case_rows], dtype=float)
            idx = np.argsort(x)
            return x[idx], y[idx], sat[idx]

        # pitch-only alpha
        x, y, sat = series_ack(pp, "cmd_yaw_deg", "alpha")
        ax3[0].plot(x, y, "o-", label="+pitch")
        ax3[0].scatter(x[sat > 0.5], y[sat > 0.5], s=80, facecolors="none", edgecolors="r", label="SAT")
        x, y, sat = series_ack(pn, "cmd_yaw_deg", "alpha")
        ax3[0].plot(x, y, "o-", label="-pitch")
        ax3[0].scatter(x[sat > 0.5], y[sat > 0.5], s=80, facecolors="none", edgecolors="r")
        ax3[0].set_title("Pitch-only: saturation scale (alpha) vs yaw")
        ax3[0].set_xlabel("cmd_yaw (deg)")
        ax3[0].set_ylabel("alpha (1.0 = no scaling)")
        ax3[0].set_ylim(0.0, 1.05)
        ax3[0].grid(True, alpha=0.3)
        ax3[0].legend()

        # roll-only alpha
        x, y, sat = series_ack(rp, "cmd_yaw_deg", "alpha")
        ax3[1].plot(x, y, "o-", label="+roll")
        ax3[1].scatter(x[sat > 0.5], y[sat > 0.5], s=80, facecolors="none", edgecolors="r", label="SAT")
        x, y, sat = series_ack(rn, "cmd_yaw_deg", "alpha")
        ax3[1].plot(x, y, "o-", label="-roll")
        ax3[1].scatter(x[sat > 0.5], y[sat > 0.5], s=80, facecolors="none", edgecolors="r")
        ax3[1].set_title("Roll-only: saturation scale (alpha) vs yaw")
        ax3[1].set_xlabel("cmd_yaw (deg)")
        ax3[1].set_ylim(0.0, 1.05)
        ax3[1].grid(True, alpha=0.3)
        ax3[1].legend()

        fig3.tight_layout()
        out3 = run_dir / "saturation_alpha_vs_yaw.png"
        fig3.savefig(out3, dpi=160)
        plt.close(fig3)

    return run_dir


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    # Plot-only mode
    if str(args.run_dir).strip():
        run_dir = Path(str(args.run_dir)).expanduser()
        plot_run(run_dir)
        logging.info("Plots written to: %s", str(run_dir))
        return 0

    run_dir = _mkdir_run(str(args.log_dir), str(args.run_name))
    meta = {
        "created_utc": utc_iso(),
        "stewart": {"com": str(args.stewart_com), "baud": int(args.stewart_baud)},
        "px4": {"com": str(args.px4_com), "baud": int(args.px4_baud), "att_hz": float(args.px4_att_hz), "dialect": str(args.px4_dialect)},
        "pose_z_mm": float(args.pose_z_mm),
        "yaw_list": str(args.yaw_list),
        "step_deg": float(args.step_deg),
        "settle_s": float(args.settle_s),
        "hold_s": float(args.hold_s),
        "yaw_zero": bool(args.yaw_zero),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    # Prepare CSV writers
    sum_f = open(run_dir / "summary.csv", "w", newline="", encoding="utf-8")
    samp_f = open(run_dir / "px4_att_samples.csv", "w", newline="", encoding="utf-8")
    ack_f = open(run_dir / "stewart_ack.csv", "w", newline="", encoding="utf-8")
    # Pre-declare fieldnames so later rows can add fields without crashing.
    sum_fieldnames = [
        "case_id",
        "cmd_roll_deg",
        "cmd_pitch_deg",
        "cmd_yaw_deg",
        "px4_roll_rel_mean_deg",
        "px4_roll_rel_std_deg",
        "px4_pitch_rel_mean_deg",
        "px4_pitch_rel_std_deg",
        "px4_yaw_rel_mean_deg",
        "px4_yaw_rel_std_deg",
        "err_roll_deg",
        "err_pitch_deg",
        "err_yaw_deg",
        "n_samples",
    ]
    samp_fieldnames = [
        # base PX4 attitude fields
        "t_rx_ns",
        "time_boot_ms",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "rollspeed_deg_s",
        "pitchspeed_deg_s",
        "yawspeed_deg_s",
        # tester annotations
        "case_id",
        "phase",
        "roll0_deg",
        "pitch0_deg",
        "yaw0_deg",
        "roll_rel_deg",
        "pitch_rel_deg",
        "yaw_rel_deg",
    ]

    sum_w = csv.DictWriter(sum_f, fieldnames=sum_fieldnames, extrasaction="ignore")
    sum_w.writeheader()
    samp_w = csv.DictWriter(samp_f, fieldnames=samp_fieldnames, extrasaction="ignore")
    samp_w.writeheader()

    # ACK CSV is variable depending on whether firmware emits "full" or "min" ACK.
    ack_w: Optional[csv.DictWriter] = None

    def write_ack_row(row: dict) -> None:
        nonlocal ack_w
        if ack_w is None:
            ack_w = csv.DictWriter(ack_f, fieldnames=list(row.keys()), extrasaction="ignore")
            ack_w.writeheader()
        ack_w.writerow(row)

    # Connections
    logging.info("Connecting Stewart serial: %s @ %d", normalize_serial_port(str(args.stewart_com)), int(args.stewart_baud))
    st = StewartSerial(str(args.stewart_com), int(args.stewart_baud), float(args.stewart_open_delay_s))
    logging.info("Connecting PX4 MAVLink: %s @ %d", normalize_serial_port(str(args.px4_com)), int(args.px4_baud))
    px4 = Px4Mavlink(str(args.px4_com), int(args.px4_baud), float(args.px4_att_hz), str(args.px4_dialect))

    # Parse yaw list
    yaw_list = [float(x.strip()) for x in str(args.yaw_list).split(",") if x.strip()]
    cases = build_cases(yaw_list=yaw_list, step_deg=float(args.step_deg), settle_s=float(args.settle_s), hold_s=float(args.hold_s))
    logging.info("Prepared %d cases", len(cases))

    # Helper: poll + log ACKs (best-effort)
    def drain_acks() -> None:
        for ack, pose in st.poll_acks():
            row = asdict(ack)
            if pose is not None:
                row.update({f"cmd_{k}": v for k, v in asdict(pose).items()})
                row["serial_rtt_ms"] = (ack.t_rx_ns - pose.t_send_ns) / 1e6
            write_ack_row(row)

    # Home pose & yaw0 capture
    t0 = monotonic_ns()
    home = PoseCmd(t_send_ns=t0, x_mm=0.0, y_mm=0.0, z_mm=float(args.pose_z_mm), roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0)
    st.send_pose(home)
    logging.info("Sent home pose. Waiting %.2fs to capture yaw0...", float(args.home_wait_s))
    t_end = monotonic_ns() + int(float(args.home_wait_s) * 1e9)
    yaw0: Optional[float] = None
    roll0: Optional[float] = None
    pitch0: Optional[float] = None
    # Drain PX4 until we see at least one sample
    while monotonic_ns() < t_end or yaw0 is None or (bool(args.zero_rp) and (roll0 is None or pitch0 is None)):
        drain_acks()
        m = px4.poll()
        if m is None:
            time.sleep(0.002)
            continue
        if yaw0 is None:
            yaw0 = wrap180(float(m.yaw_deg))
            logging.info("Yaw0 captured: %.2f deg (wrapped)", yaw0)
        if bool(args.zero_rp):
            if roll0 is None:
                roll0 = float(m.roll_deg)
                logging.info("Roll0 captured: %.2f deg", roll0)
            if pitch0 is None:
                pitch0 = float(m.pitch_deg)
                logging.info("Pitch0 captured: %.2f deg", pitch0)
        # Log baseline samples too
        row = asdict(m)
        row.update(
            {
                "case_id": "",
                "phase": "home",
                "roll0_deg": roll0 if roll0 is not None else "",
                "pitch0_deg": pitch0 if pitch0 is not None else "",
                "yaw0_deg": yaw0,
                "roll_rel_deg": (float(m.roll_deg) - float(roll0)) if (roll0 is not None and bool(args.zero_rp)) else float(m.roll_deg),
                "pitch_rel_deg": (float(m.pitch_deg) - float(pitch0)) if (pitch0 is not None and bool(args.zero_rp)) else float(m.pitch_deg),
                "yaw_rel_deg": wrap180(float(m.yaw_deg) - float(yaw0)),
            }
        )
        samp_w.writerow(row)

    assert yaw0 is not None
    if bool(args.zero_rp):
        assert roll0 is not None and pitch0 is not None
    else:
        roll0 = 0.0
        pitch0 = 0.0

    def yaw_rel(yaw_deg: float) -> float:
        return wrap180(float(yaw_deg) - float(yaw0)) if bool(args.yaw_zero) else wrap180(float(yaw_deg))

    def roll_rel(roll_deg: float) -> float:
        return float(roll_deg - float(roll0)) if bool(args.zero_rp) else float(roll_deg)

    def pitch_rel(pitch_deg: float) -> float:
        return float(pitch_deg - float(pitch0)) if bool(args.zero_rp) else float(pitch_deg)

    # Run cases
    try:
        for c in cases:
            logging.info("Case %d: cmd r=%.1f p=%.1f y=%.1f (settle %.1fs hold %.1fs)", c.case_id, c.roll_cmd_deg, c.pitch_cmd_deg, c.yaw_cmd_deg, c.settle_s, c.hold_s)
            pose = PoseCmd(
                t_send_ns=monotonic_ns(),
                x_mm=0.0,
                y_mm=0.0,
                z_mm=float(args.pose_z_mm),
                roll_deg=float(c.roll_cmd_deg),
                pitch_deg=float(c.pitch_cmd_deg),
                yaw_deg=float(c.yaw_cmd_deg),
            )
            st.send_pose(pose)

            # settle
            t_settle_end = monotonic_ns() + int(c.settle_s * 1e9)
            while monotonic_ns() < t_settle_end:
                drain_acks()
                m = px4.poll()
                if m is not None:
                    row = asdict(m)
                    row.update(
                        {
                            "case_id": c.case_id,
                            "phase": "settle",
                            "roll0_deg": float(roll0),
                            "pitch0_deg": float(pitch0),
                            "yaw0_deg": float(yaw0),
                            "roll_rel_deg": roll_rel(m.roll_deg),
                            "pitch_rel_deg": pitch_rel(m.pitch_deg),
                            "yaw_rel_deg": yaw_rel(m.yaw_deg),
                        }
                    )
                    samp_w.writerow(row)
                else:
                    time.sleep(0.001)

            # hold window
            t_hold_end = monotonic_ns() + int(c.hold_s * 1e9)
            samples = []
            while monotonic_ns() < t_hold_end:
                drain_acks()
                m = px4.poll()
                if m is None:
                    time.sleep(0.001)
                    continue
                samples.append(m)
                row = asdict(m)
                row.update(
                    {
                        "case_id": c.case_id,
                        "phase": "hold",
                        "roll0_deg": float(roll0),
                        "pitch0_deg": float(pitch0),
                        "yaw0_deg": float(yaw0),
                        "roll_rel_deg": roll_rel(m.roll_deg),
                        "pitch_rel_deg": pitch_rel(m.pitch_deg),
                        "yaw_rel_deg": yaw_rel(m.yaw_deg),
                    }
                )
                samp_w.writerow(row)

            if not samples:
                logging.warning("No PX4 ATTITUDE samples received for case %d", c.case_id)
                continue

            roll = np.array([roll_rel(s.roll_deg) for s in samples], dtype=float)
            pitch = np.array([pitch_rel(s.pitch_deg) for s in samples], dtype=float)
            yawr = np.array([yaw_rel(s.yaw_deg) for s in samples], dtype=float)

            roll_mu, roll_sd = _mean_std(roll)
            pitch_mu, pitch_sd = _mean_std(pitch)
            yaw_mu, yaw_sd = _mean_std(yawr)

            err_roll = angle_error_deg(float(c.roll_cmd_deg), roll_mu)
            err_pitch = angle_error_deg(float(c.pitch_cmd_deg), pitch_mu)
            err_yaw = angle_error_deg(float(c.yaw_cmd_deg), yaw_mu)

            sum_row = {
                "case_id": c.case_id,
                "cmd_roll_deg": c.roll_cmd_deg,
                "cmd_pitch_deg": c.pitch_cmd_deg,
                "cmd_yaw_deg": c.yaw_cmd_deg,
                "px4_roll_rel_mean_deg": roll_mu,
                "px4_roll_rel_std_deg": roll_sd,
                "px4_pitch_rel_mean_deg": pitch_mu,
                "px4_pitch_rel_std_deg": pitch_sd,
                "px4_yaw_rel_mean_deg": yaw_mu,
                "px4_yaw_rel_std_deg": yaw_sd,
                "err_roll_deg": err_roll,
                "err_pitch_deg": err_pitch,
                "err_yaw_deg": err_yaw,
                "n_samples": int(len(samples)),
            }
            sum_w.writerow(sum_row)

    except KeyboardInterrupt:
        logging.info("Ctrl+C: stopping test.")
    finally:
        # Return to home pose
        try:
            st.send_pose(PoseCmd(t_send_ns=monotonic_ns(), x_mm=0.0, y_mm=0.0, z_mm=float(args.pose_z_mm), roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0))
        except Exception:
            pass
        try:
            drain_acks()
        except Exception:
            pass
        try:
            st.close()
        except Exception:
            pass

        for f in (sum_f, samp_f, ack_f):
            try:
                f.close()
            except Exception:
                pass

    # Always attempt to plot from the run artifacts we just wrote.
    try:
        plot_run(run_dir)
        logging.info("Plots written to: %s", str(run_dir))
    except Exception as e:
        logging.warning("Plot generation failed: %s", e)

    logging.info("Done. Results in: %s", str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


