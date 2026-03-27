from __future__ import annotations

"""
Aggregate latency statistics for baseline runs #5-#9.

Definitions used here:
- X-Plane sample age: host tick time minus the latest X-Plane sample receive time
  available to the controller at that tick.
- PX4 sample age: host tick time minus the latest PX4 sample receive time available
  to the controller at that tick.
- Serial RTT: ACK-based round-trip time recorded by the host/firmware path.
- End-to-end latency: not "X-Plane rx -> ACK rx". In this script it is the
  observed response delay that best aligns the host-received X-Plane pitch
  trajectory with the bias-corrected PX4 pitch trajectory over the
  trim-transition scenario.

The end-to-end estimator is intentionally shape-based because the PX4 signal is
noisy and asynchronous, so exact sample-by-sample matching is not reliable.
For each trim-transition repeat, the script searches a positive lag window and
selects the lag that maximizes normalized correlation between the resampled
X-Plane and PX4 pitch traces. A single representative delay per run is then
taken as the median of that run's repeat-level delays, and those run-level
delays are aggregated across baseline runs #5-#9.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paper.Fig_Gen import BASELINE_RUN_DIRS, SKIP_FIRST_S, load_run

MATCH_PHASES = ("hold_0", "warmup_step", "hold_step", "warmup_return", "hold_return")
LAG_SEARCH_MS = np.arange(0.0, 150.0 + 1.0, 1.0, dtype=float)
GRID_DT_S = 0.005
MIN_GRID_SAMPLES = 50


@dataclass(frozen=True)
class PitchStream:
    t_ns: np.ndarray
    pitch_deg: np.ndarray


@dataclass(frozen=True)
class TransitionWindow:
    repeat_idx: int
    start_ns: float
    end_ns: float
    phases: tuple[str, ...]


@dataclass(frozen=True)
class SegmentEstimate:
    repeat_idx: int
    delay_ms: float
    corr: float
    rmse_deg: float
    n_samples: int
    start_s: float
    end_s: float


def _to_float(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except Exception:
        return float("nan")


def _safe_stats(values: np.ndarray) -> dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean_ms": float("nan"), "median_ms": float("nan"), "count": 0}
    return {
        "mean_ms": float(np.mean(vals)),
        "median_ms": float(np.median(vals)),
        "count": int(vals.size),
    }


def _read_pitch_stream(path: Path) -> PitchStream:
    t_ns_list: list[float] = []
    pitch_list: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            t_ns = _to_float(row.get("t_rx_ns"))
            pitch_deg = _to_float(row.get("pitch_deg"))
            if np.isfinite(t_ns) and np.isfinite(pitch_deg):
                t_ns_list.append(t_ns)
                pitch_list.append(pitch_deg)
    if not t_ns_list:
        return PitchStream(np.zeros(0, dtype=float), np.zeros(0, dtype=float))
    t_ns = np.asarray(t_ns_list, dtype=float)
    pitch_deg = np.asarray(pitch_list, dtype=float)
    order = np.argsort(t_ns, kind="stable")
    t_ns = t_ns[order]
    pitch_deg = pitch_deg[order]
    unique_t_ns, unique_idx = np.unique(t_ns, return_index=True)
    return PitchStream(unique_t_ns, pitch_deg[unique_idx])


def _extract_transition_windows(run) -> list[TransitionWindow]:
    grouped_times: dict[int, dict[str, list[float]]] = {}
    for row in run.tick_rows:
        phase = str(row.get("scenario_phase") or "").strip()
        if phase not in MATCH_PHASES:
            continue
        t_tick_ns = _to_float(row.get("t_tick_ns"))
        pid_enabled = _to_float(row.get("pid_enabled"))
        repeat_raw = _to_float(row.get("repeat_idx"))
        if not np.isfinite(t_tick_ns):
            continue
        if np.isfinite(pid_enabled) and pid_enabled <= 0.5:
            continue
        repeat_idx = int(repeat_raw) if np.isfinite(repeat_raw) else 0
        grouped_times.setdefault(repeat_idx, {}).setdefault(phase, []).append(t_tick_ns)

    windows: list[TransitionWindow] = []
    for repeat_idx, phase_times in sorted(grouped_times.items()):
        if "hold_step" not in phase_times:
            continue
        start_phase = "hold_0" if "hold_0" in phase_times else "warmup_step"
        end_phase = "hold_return" if "hold_return" in phase_times else "hold_step"
        start_ns = min(phase_times[start_phase])
        end_ns = max(phase_times[end_phase])
        phases = tuple(sorted(phase_times))
        if end_ns > start_ns:
            windows.append(
                TransitionWindow(
                    repeat_idx=repeat_idx,
                    start_ns=float(start_ns),
                    end_ns=float(end_ns),
                    phases=phases,
                )
            )
    return windows


def _normalize(signal: np.ndarray) -> np.ndarray | None:
    sig = np.asarray(signal, dtype=float)
    if sig.size == 0:
        return None
    std = float(np.std(sig))
    if not np.isfinite(std) or std < 1e-9:
        return None
    return (sig - float(np.mean(sig))) / std


def _estimate_segment_delay(
    xp_stream: PitchStream,
    px4_stream: PitchStream,
    bias_deg: float,
    window: TransitionWindow,
    run_t0_ns: float,
) -> SegmentEstimate | None:
    pad_ns = float(np.max(LAG_SEARCH_MS) * 1e6)

    xp_mask = (
        np.isfinite(xp_stream.t_ns)
        & np.isfinite(xp_stream.pitch_deg)
        & (xp_stream.t_ns >= window.start_ns)
        & (xp_stream.t_ns <= window.end_ns)
    )
    px4_mask = (
        np.isfinite(px4_stream.t_ns)
        & np.isfinite(px4_stream.pitch_deg)
        & (px4_stream.t_ns >= window.start_ns)
        & (px4_stream.t_ns <= window.end_ns + pad_ns)
    )
    if np.count_nonzero(xp_mask) < 2 or np.count_nonzero(px4_mask) < 2:
        return None

    xp_t_s = (xp_stream.t_ns[xp_mask] - window.start_ns) / 1e9
    xp_y = xp_stream.pitch_deg[xp_mask]
    px4_t_s = (px4_stream.t_ns[px4_mask] - window.start_ns) / 1e9
    px4_y = px4_stream.pitch_deg[px4_mask] - (bias_deg if np.isfinite(bias_deg) else 0.0)

    best: tuple[float, float, float, int] | None = None
    for lag_ms in LAG_SEARCH_MS:
        lag_s = float(lag_ms) / 1000.0
        overlap_start_s = max(0.0, float(xp_t_s[0]), float(px4_t_s[0] - lag_s))
        overlap_end_s = min(float(xp_t_s[-1]), float(px4_t_s[-1] - lag_s))
        if overlap_end_s <= overlap_start_s:
            continue

        grid = np.arange(overlap_start_s, overlap_end_s, GRID_DT_S, dtype=float)
        if grid.size < MIN_GRID_SAMPLES:
            continue

        xp_interp = np.interp(grid, xp_t_s, xp_y)
        px4_interp = np.interp(grid + lag_s, px4_t_s, px4_y)
        xp_norm = _normalize(xp_interp)
        px4_norm = _normalize(px4_interp)
        if xp_norm is None or px4_norm is None:
            continue

        corr = float(np.mean(xp_norm * px4_norm))
        rmse_deg = float(np.sqrt(np.mean((px4_interp - xp_interp) ** 2)))
        candidate = (corr, -rmse_deg, lag_ms, int(grid.size))
        if best is None or candidate > best:
            best = candidate

    if best is None:
        return None

    corr, neg_rmse_deg, delay_ms, n_samples = best
    return SegmentEstimate(
        repeat_idx=window.repeat_idx,
        delay_ms=float(delay_ms),
        corr=float(corr),
        rmse_deg=float(-neg_rmse_deg),
        n_samples=int(n_samples),
        start_s=float((window.start_ns - run_t0_ns) / 1e9),
        end_s=float((window.end_ns - run_t0_ns) / 1e9),
    )


def compute_latency_stats() -> dict[str, object]:
    runs = [load_run(path, skip_first_s=SKIP_FIRST_S) for path in BASELINE_RUN_DIRS]

    xp_age_all = np.concatenate([run.xp_age_ms[run.analysis_mask] for run in runs])
    px4_age_all = np.concatenate([run.px4_age_ms[run.analysis_mask] for run in runs])

    serial_rtt_parts: list[np.ndarray] = []
    for run in runs:
        if run.ack_t_s.size == 0:
            continue
        ack_mask = np.isfinite(run.ack_t_s) & (run.ack_t_s >= float(SKIP_FIRST_S))
        serial_rtt_parts.append(run.serial_rtt_ms_by_ack[ack_mask])
    serial_rtt_all = (
        np.concatenate(serial_rtt_parts) if serial_rtt_parts else np.zeros(0, dtype=float)
    )

    per_run: list[dict[str, object]] = []
    per_run_delays_ms: list[float] = []
    per_segment_delays_ms: list[float] = []

    for run in runs:
        xp_stream = _read_pitch_stream(run.run_dir / "xplane_att.csv")
        px4_stream = _read_pitch_stream(run.run_dir / "px4_att.csv")
        windows = _extract_transition_windows(run)
        if not windows or xp_stream.t_ns.size == 0 or px4_stream.t_ns.size == 0:
            per_run.append(
                {
                    "run_dir": str(run.run_dir),
                    "mount_bias_pitch_deg": float(run.mount_bias_pitch_deg),
                    "observed_end_to_end_ms": float("nan"),
                    "repeat_count": 0,
                    "segments": [],
                }
            )
            continue

        t0_ns = _to_float(run.tick_rows[0].get("t_tick_ns")) if run.tick_rows else 0.0
        segment_estimates: list[SegmentEstimate] = []
        for window in windows:
            estimate = _estimate_segment_delay(
                xp_stream=xp_stream,
                px4_stream=px4_stream,
                bias_deg=float(run.mount_bias_pitch_deg),
                window=window,
                run_t0_ns=float(t0_ns),
            )
            if estimate is not None:
                segment_estimates.append(estimate)

        run_delay_ms = float("nan")
        if segment_estimates:
            delay_values = np.array([seg.delay_ms for seg in segment_estimates], dtype=float)
            run_delay_ms = float(np.median(delay_values))
            per_run_delays_ms.append(run_delay_ms)
            per_segment_delays_ms.extend(delay_values.tolist())

        per_run.append(
            {
                "run_dir": str(run.run_dir),
                "mount_bias_pitch_deg": float(run.mount_bias_pitch_deg),
                "observed_end_to_end_ms": run_delay_ms,
                "repeat_count": len(segment_estimates),
                "segments": [
                    {
                        "repeat_idx": seg.repeat_idx,
                        "delay_ms": seg.delay_ms,
                        "corr": seg.corr,
                        "rmse_deg": seg.rmse_deg,
                        "n_samples": seg.n_samples,
                        "window_s": [seg.start_s, seg.end_s],
                    }
                    for seg in segment_estimates
                ],
            }
        )

    return {
        "definition": {
            "end_to_end": (
                "Observed response delay from host-received X-Plane pitch to "
                "bias-corrected PX4 pitch, estimated by shape matching over the "
                "trim-transition scenario."
            ),
            "matching_method": "positive-lag normalized cross-correlation on resampled pitch traces",
            "transition_phases": list(MATCH_PHASES),
            "lag_search_ms": [float(LAG_SEARCH_MS[0]), float(LAG_SEARCH_MS[-1])],
            "grid_dt_ms": float(GRID_DT_S * 1000.0),
            "aggregation": "median delay per run, then aggregate across baseline runs #5-#9",
        },
        "xplane_sample_age": _safe_stats(xp_age_all),
        "px4_sample_age": _safe_stats(px4_age_all),
        "serial_rtt": _safe_stats(serial_rtt_all),
        "end_to_end": _safe_stats(np.asarray(per_run_delays_ms, dtype=float)),
        "end_to_end_segments": _safe_stats(np.asarray(per_segment_delays_ms, dtype=float)),
        "per_run": per_run,
    }


def main() -> int:
    stats = compute_latency_stats()
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
