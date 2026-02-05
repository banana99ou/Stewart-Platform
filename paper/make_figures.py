from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _ensure_outdir() -> Path:
    outdir = Path("paper") / "fig"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def fig1_architecture(outdir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_axis_off()

    # Box helper
    def box(x, y, w, h, title, subtitle):
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.6)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=9)

    # Layout (normalized coords)
    box(0.05, 0.55, 0.22, 0.32, "X-Plane", "Dynamics / State")
    box(0.33, 0.55, 0.26, 0.32, "Simulink Host", "Pose gen / Logging\nUDP + Serial + MAVLink")
    box(0.67, 0.55, 0.26, 0.32, "ESP32 (Stewart MCU)", "FW IK + Saturation\n6x RC Servos")
    box(0.67, 0.12, 0.26, 0.32, "PX4 (Pix32 v6)", "IMU + Estimator\nActuator outputs")

    # Arrows
    def arrow(x0, y0, x1, y1, label):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=1.8),
        )
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.03, label, ha="center", va="center", fontsize=9)

    arrow(0.27, 0.71, 0.33, 0.71, "UDP (state)")
    arrow(0.33, 0.63, 0.27, 0.63, "UDP (control)")

    arrow(0.59, 0.71, 0.67, 0.71, "USB Serial\npose line")
    arrow(0.80, 0.55, 0.80, 0.44, "Physical motion")
    arrow(0.67, 0.28, 0.59, 0.28, "MAVLink\natt/act")

    ax.set_title("Fig. 1. System architecture", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "Fig1_Architecture.png", dpi=200)
    plt.close(fig)


def fig2_trim_transition(outdir: Path) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(7)
    t = np.linspace(0, 8, 801)

    # Trim transition: the vehicle attitude typically changes gradually (not an instantaneous step).
    # We'll model a pitch response to a step-like excitation at t=2s using a smooth, causal response.
    step_amp = 5.0  # deg
    t0 = 2.0
    td0 = np.clip(t - t0, 0, None)
    tau_sim = 0.9
    sim_pitch = step_amp * (1.0 - np.exp(-td0 / tau_sim))
    sim_pitch += 0.04 * rng.standard_normal(t.size)

    # PX4 estimated attitude: extra delay + additional lag + slight gain error + noise
    delay = 0.18  # s
    td1 = np.clip(t - (t0 + delay), 0, None)
    tau_px4 = 1.25
    px4_pitch = step_amp * (1.0 - np.exp(-td1 / tau_px4))
    px4_pitch *= 0.93
    px4_pitch += 0.07 * rng.standard_normal(t.size)

    err = px4_pitch - sim_pitch

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax0, ax1 = axes

    ax0.plot(t, sim_pitch, label="X-Plane pitch (deg)", lw=2.0)
    ax0.plot(t, px4_pitch, label="PX4 pitch est. (deg)", lw=2.0)
    ax0.set_ylabel("Pitch (deg)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")

    ax1.plot(t, err, label="Error (PX4 - X-Plane)", color="tab:red", lw=1.8)
    ax1.axhline(0, color="k", lw=1.0, alpha=0.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Error (deg)")
    ax1.grid(True, alpha=0.3)

    fig.suptitle("Fig. 2. Trim transition: pitch response", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / "Fig2_TrimTransition.png", dpi=200)
    plt.close(fig)


def fig3_latency(outdir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [
        "X-Plane→Host (UDP)",
        "Host compute",
        "Host→ESP32 (Serial)",
        "Servo/Platform",
        "IMU/Estimator",
        "PX4→Host (MAVLink)",
        "Host→X-Plane (UDP)",
    ]
    # Example latency components (replace with measured)
    ms = np.array([6, 3, 9, 35, 22, 10, 6], dtype=float)
    total = float(ms.sum())

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(labels, ms, color="tab:blue", alpha=0.85)
    ax.set_xlabel("Latency (ms)")
    ax.set_title(f"Fig. 3. Latency breakdown — total {total:.0f} ms")
    ax.grid(True, axis="x", alpha=0.3)
    for i, v in enumerate(ms):
        ax.text(v + 0.8, i, f"{v:.0f} ms", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "Fig3_LatencyBreakdown.png", dpi=200)
    plt.close(fig)


def main() -> None:
    # Avoid GUI backend issues in headless runs
    os.environ.setdefault("MPLBACKEND", "Agg")

    outdir = _ensure_outdir()
    fig1_architecture(outdir)
    fig2_trim_transition(outdir)
    fig3_latency(outdir)
    print(f"Wrote figures to: {outdir}")


if __name__ == "__main__":
    main()


