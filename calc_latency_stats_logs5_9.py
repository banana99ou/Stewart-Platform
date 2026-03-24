from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper.Fig_Gen import BASELINE_RUN_DIRS, SKIP_FIRST_S, load_run


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


def compute_latency_stats() -> dict[str, dict[str, float]]:
    runs = [load_run(p, skip_first_s=SKIP_FIRST_S) for p in BASELINE_RUN_DIRS]

    xp_age_all = np.concatenate([r.xp_age_ms[r.analysis_mask] for r in runs])
    px4_age_all = np.concatenate([r.px4_age_ms[r.analysis_mask] for r in runs])

    serial_rtt_parts: list[np.ndarray] = []
    e2e_parts: list[np.ndarray] = []
    for r in runs:
        if r.ack_t_s.size == 0:
            continue
        ack_mask = np.isfinite(r.ack_t_s) & (r.ack_t_s >= float(SKIP_FIRST_S))
        serial_rtt_parts.append(r.serial_rtt_ms_by_ack[ack_mask])
        e2e_mask = ack_mask & np.isfinite(r.e2e_xp_ms_by_ack)
        if np.any(e2e_mask):
            e2e_parts.append(r.e2e_xp_ms_by_ack[e2e_mask])

    serial_rtt_all = np.concatenate(serial_rtt_parts) if serial_rtt_parts else np.zeros(0, dtype=float)
    e2e_all = np.concatenate(e2e_parts) if e2e_parts else np.zeros(0, dtype=float)

    return {
        "xplane_sample_age": _safe_stats(xp_age_all),
        "px4_sample_age": _safe_stats(px4_age_all),
        "serial_rtt": _safe_stats(serial_rtt_all),
        "end_to_end": _safe_stats(e2e_all),
    }


def main() -> int:
    stats = compute_latency_stats()
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
