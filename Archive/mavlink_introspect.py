"""
mavlink_introspect.py

Two complementary ways to answer: "what data can I extract from Pixhawk?"

1) List what the MAVLink dialect *defines* (static catalog):
   - message IDs
   - message names
   - field names

2) Sniff what your Pixhawk is *actually streaming right now* (dynamic):
   - message types observed on the link
   - approximate rates (Hz)
   - field names + one example payload

Notes
-----
- MAVLink is huge. No autopilot will stream "everything" by default.
- The most practical way is to sniff messages, then request only what you need.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


def _load_dialect_module(dialect: str):
    """
    Load a pymavlink dialect module like:
      pymavlink.dialects.v20.ardupilotmega
      pymavlink.dialects.v20.common
    """
    last_err = None
    for ver in ("v20", "v10"):
        mod_name = f"pymavlink.dialects.{ver}.{dialect}"
        try:
            return importlib.import_module(mod_name)
        except Exception as e:  # pragma: no cover (best-effort)
            last_err = e
    raise RuntimeError(f"Failed to import MAVLink dialect '{dialect}'. Last error: {last_err}")


def _safe_msg_name(msg_cls) -> str:
    # pymavlink message classes usually expose `.name` or `.msgname`
    for attr in ("name", "msgname"):
        v = getattr(msg_cls, attr, None)
        if isinstance(v, str) and v:
            return v
    # Fallback: class name like MAVLink_attitude_message -> ATTITUDE-ish
    return getattr(msg_cls, "__name__", "UNKNOWN")


def _safe_fieldnames(msg_cls) -> list[str]:
    v = getattr(msg_cls, "fieldnames", None)
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return []


def list_defined_messages(dialect: str, limit: int = 0) -> int:
    """
    Print a catalog of all message types defined in the dialect.
    """
    mod = _load_dialect_module(dialect)
    mavlink_map = getattr(mod, "mavlink_map", None)
    if not isinstance(mavlink_map, dict):
        raise RuntimeError(f"Dialect '{dialect}' has no mavlink_map (unexpected).")

    printed = 0
    for msg_id, msg_cls in sorted(mavlink_map.items(), key=lambda kv: int(kv[0])):
        name = _safe_msg_name(msg_cls)
        fields = _safe_fieldnames(msg_cls)
        field_str = ", ".join(fields) if fields else "(fields unknown)"
        print(f"{int(msg_id):4d}  {name}  ::  {field_str}")
        printed += 1
        if limit and printed >= limit:
            break
    return printed


def _build_name_to_id(dialect: str) -> Dict[str, int]:
    mod = _load_dialect_module(dialect)
    mavlink_map = getattr(mod, "mavlink_map", {})
    out: Dict[str, int] = {}
    for msg_id, msg_cls in mavlink_map.items():
        name = _safe_msg_name(msg_cls).upper()
        out[name] = int(msg_id)
    return out


def _request_message_interval(master, msg_id: int, hz: float):
    # Import here so --list-defined works without a serial connection.
    from pymavlink import mavutil

    interval_us = int(1e6 / hz) if hz > 0 else -1  # -1 disables stream
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        msg_id,
        interval_us,
        0,
        0,
        0,
        0,
        0,
    )


COMMON_REQUESTS = {
    # Attitude (what you asked for)
    "ATTITUDE": 50.0,
    "ATTITUDE_QUATERNION": 50.0,
    # Estimator / navigation (often useful)
    "ODOMETRY": 20.0,
    "LOCAL_POSITION_NED": 20.0,
    "GLOBAL_POSITION_INT": 10.0,
    # Actuator outputs (typical "control surface cmd" equivalents)
    "SERVO_OUTPUT_RAW": 50.0,   # common on ArduPilot
    "RC_CHANNELS": 20.0,        # input channels (RC)
    "ACTUATOR_OUTPUT_STATUS": 50.0,  # common on PX4
}


@dataclass
class MsgStats:
    msg_id: int
    msg_type: str
    count: int = 0
    first_t: float = 0.0
    last_t: float = 0.0
    example: Optional[Dict[str, Any]] = None

    def update(self, t: float, msg) -> None:
        if self.count == 0:
            self.first_t = t
        self.last_t = t
        self.count += 1
        if self.example is None:
            try:
                self.example = msg.to_dict()
            except Exception:
                # Fallback: limited info
                self.example = {"_type": self.msg_type}

    def rate_hz(self) -> float:
        dt = self.last_t - self.first_t
        if self.count <= 1 or dt <= 1e-9:
            return 0.0
        return (self.count - 1) / dt


def sniff_link(
    com: str,
    baud: int,
    dialect: str,
    seconds: float,
    request_common: bool,
    request_hz: float,
    print_new: bool,
    json_out: str | None,
) -> Dict[str, MsgStats]:
    """
    Connect to a MAVLink serial link and observe what messages arrive.
    """
    # Make dialect selection explicit for this process.
    os.environ.setdefault("MAVLINK20", "1")
    os.environ["MAVLINK_DIALECT"] = dialect

    from pymavlink import mavutil

    name_to_id = _build_name_to_id(dialect)

    master = mavutil.mavlink_connection(com, baud=baud)
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print(f"Heartbeat from system={master.target_system} component={master.target_component}")

    if request_common:
        print("Requesting common messages (best-effort)...")
        for name, hz in COMMON_REQUESTS.items():
            msg_id = name_to_id.get(name.upper())
            if msg_id is None:
                continue
            _request_message_interval(master, msg_id, request_hz if request_hz > 0 else hz)

    stats: Dict[str, MsgStats] = {}
    start = time.time()
    end = start + seconds if seconds > 0 else float("inf")

    while time.time() < end:
        msg = master.recv_match(blocking=True, timeout=1)
        if msg is None:
            continue

        msg_type = msg.get_type()
        if msg_type == "BAD_DATA":
            continue

        try:
            msg_id = int(msg.get_msgId())
        except Exception:
            msg_id = -1

        t = time.time()
        if msg_type not in stats:
            stats[msg_type] = MsgStats(msg_id=msg_id, msg_type=msg_type)
            if print_new:
                print(f"NEW: {msg_type} (id={msg_id})")

        stats[msg_type].update(t, msg)

    # Print summary (sorted by observed rate, then count)
    rows = sorted(stats.values(), key=lambda s: (s.rate_hz(), s.count), reverse=True)
    print("")
    print("Observed message summary:")
    for s in rows:
        print(f"- {s.msg_type:28s} id={s.msg_id:4d}  rate~{s.rate_hz():6.1f} Hz  count={s.count}")

    if json_out:
        payload = {k: asdict(v) | {"rate_hz": v.rate_hz()} for k, v in stats.items()}
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nWrote JSON: {json_out}")

    return stats


def parse_args():
    p = argparse.ArgumentParser(description="List MAVLink message catalog and sniff what Pixhawk is streaming.")

    p.add_argument("--dialect", default="ardupilotmega", help="MAVLink dialect (e.g. ardupilotmega, common)")

    p.add_argument("--list-defined", action="store_true", help="Print all message types defined in the dialect (static catalog)")
    p.add_argument("--list-limit", type=int, default=0, help="Limit --list-defined output (0 = no limit)")

    p.add_argument("--sniff", action="store_true", help="Connect and sniff messages on the link")
    p.add_argument("--seconds", type=float, default=10.0, help="Sniff duration (0 = run forever)")
    p.add_argument("--com", default=r"\\.\COM13", help=r"Serial port, e.g. \\.\COM13")
    p.add_argument("--baud", type=int, default=115200, help="Serial baud (often ignored on USB CDC)")

    p.add_argument("--request-common", action="store_true", help="Request a useful set of common messages (best-effort)")
    p.add_argument("--request-hz", type=float, default=0.0, help="If >0, force all requested messages to this rate (Hz)")
    p.add_argument("--print-new", action="store_true", help="Print each new message type as it is first seen")
    p.add_argument("--json-out", default="", help="Write sniff results to JSON file (optional)")

    return p.parse_args()


def main():
    args = parse_args()

    if not args.list_defined and not args.sniff:
        raise SystemExit("Pick one: --list-defined and/or --sniff")

    if args.list_defined:
        n = list_defined_messages(args.dialect, limit=args.list_limit)
        print(f"\nListed {n} messages from dialect '{args.dialect}'.")

    if args.sniff:
        sniff_link(
            com=args.com,
            baud=args.baud,
            dialect=args.dialect,
            seconds=args.seconds,
            request_common=args.request_common,
            request_hz=args.request_hz,
            print_new=args.print_new,
            json_out=(args.json_out or None),
        )


if __name__ == "__main__":
    main()















