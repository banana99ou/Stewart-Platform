"""
hils_host.py — Single-process Python host for the Stewart Platform Physical HILS

Goals (conference-grade, paper-friendly):
- Own every link (X-Plane UDP, ESP32 serial, PX4 MAVLink) under one host clock
- Run a deterministic 100 Hz control "tick"
- Log everything needed for: tracking error, latency/jitter, saturation statistics

Notes / current constraints:
- ESP32 pose command format (firmware): ASCII line
    "x y z roll pitch yaw\n"
  So we DO NOT embed extra tokens in the command line unless firmware is updated.
- ESP32 emits an ACK/status line per parsed pose command (see main/main.ino):
    "ACK seq=... rx_us=... done_us=... dt_us=... sat=... a=... ..."
  We parse these lines and pair them FIFO with sent pose commands.
- X-Plane UDP packet formats vary by configuration. This script supports:
  - Receiving the same X-Plane UDP packet layout used by the Simulink model in this repo:
      - UDP RX: 127.0.0.1:49004
      - Payload: "DATA\\0" header (5 bytes) + 9x8 float32 matrix (MATLAB column-major)
    We decode pitch/roll/heading and p/q/r from the matrix using the same indices as the Simulink decoder.
  - Transmitting "DATA\\0" packets back to X-Plane (UDP TX: 127.0.0.1:49001) with:
      - group 8  : control surfaces (elevator/aileron/rudder; others = -999)
      - group 25 : throttle (others = -999)

Examples on how to run this code:
---------------------------------

# Typical run (with X-Plane, PX4, Stewart platform serial on COM13):

    python hils_host.py --xplane-recv-port 49004 --xplane-send-port 49001 --stewart-com COM13 --px4-port COM5

# Minimal "platform-only" run (no PX4, no X-Plane injection):

    python hils_host.py --no-px4 --xplane-send none --stewart-com COM13

# Disable X-Plane receive, log only PX4 and Stewart:

    python hils_host.py --no-xplane --stewart-com COM13 --px4-port COM5

# Custom X-Plane IP/ports (e.g., for remote X-Plane instance):

    python hils_host.py --xplane-ip 192.168.1.50 --xplane-recv-port 49010 --xplane-send-port 49012 --stewart-com COM13

# See all options and help:

    python hils_host.py --help

"""

# Quick start (typical ports used by the Simulink model):
# - X-Plane -> host UDP: 127.0.0.1:49004 (DATA\0 + 9x8 float32 matrix)
# - host -> X-Plane UDP: 127.0.0.1:49001 (DATA\0 groups 8 + 25)
# - host -> ESP32: serial (pose "x y z roll pitch yaw\n" @115200)
# - PX4 -> host: MAVLink serial (ATTITUDE requested at 100 Hz)
#
# If your X-Plane network config uses different ports or a different DATA layout,
# adjust `XPlaneReceiver` and `XPlaneSender` accordingly.
#
# Minimal "platform-only" run (no PX4, no X-Plane injection):
#   python hils_host.py --no-px4 --xplane-send none --stewart-com COM13

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import struct
import sys
import time
import logging
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np


# ---------------- Time helpers ----------------

def monotonic_ns() -> int:
    return time.perf_counter_ns()


def utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sleep_until_ns(t_target_ns: int) -> None:
    # Coarse sleep; good enough for 100 Hz in CPython.
    while True:
        now = monotonic_ns()
        dt_ns = t_target_ns - now
        if dt_ns <= 0:
            return
        if dt_ns > 2_000_000:  # >2ms
            time.sleep((dt_ns - 1_000_000) / 1e9)
        else:
            # Busy wait for the last ~2ms
            return


# ---------------- PID ----------------

class PID:
    """
    Simple PID on degrees (not radians).
    """

    def __init__(self, kp: float, ki: float, kd: float, i_limit: float, out_limit: float):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.i_limit = float(i_limit)
        self.out_limit = float(out_limit)
        self.i = 0.0
        self.prev_err = 0.0
        self.has_prev = False

    def update(self, setpoint_deg: float, meas_deg: float, dt_s: float) -> float:
        if not (0.0 < dt_s < 10.0):
            dt_s = 0.0
        err = float(setpoint_deg) - float(meas_deg)

        # integral
        self.i += err * dt_s
        self.i = max(-self.i_limit, min(self.i, self.i_limit))

        # derivative
        d = 0.0
        if self.has_prev and dt_s > 0:
            d = (err - self.prev_err) / dt_s
        self.prev_err = err
        self.has_prev = True

        u = self.kp * err + self.ki * self.i + self.kd * d
        u = max(-self.out_limit, min(u, self.out_limit))
        return float(u)


# ---------------- ESP32 Stewart serial ----------------

def normalize_serial_port(port: str) -> str:
    p = (port or "").strip()
    if not p:
        raise ValueError("serial port is empty")
    if sys.platform.startswith("win"):
        up = p.upper()
        if up.startswith(r"\\.\COM") or up.startswith(r"\\\\.\\COM"):
            return p
        if up.startswith("COM"):
            return r"\\.\{}".format(up)
    return p


ACK_RE = re.compile(
    r"^ACK\s+seq=(?P<seq>\d+)\s+status=(?P<status>\S+)\s+flags=(?P<flags>\S+)\s+"
    r"rx_us=(?P<rx_us>\d+)\s+done_us=(?P<done_us>\d+)\s+dt_us=(?P<dt_us>\d+)\s+"
    r"ik_mask=0x(?P<ik_mask>[0-9A-Fa-f]{2})\s+sat=(?P<sat>[01])\s+"
    r"lim=(?P<lim>[-+0-9\.]+)\s+mag=(?P<mag_in>[-+0-9\.]+)->(?P<mag_out>[-+0-9\.]+)\s+"
    r"a=(?P<a>[-+0-9\.]+)"
)


@dataclass
class StewartAck:
    t_rx_ns: int
    seq: int
    status: str
    flags: str
    rx_us: int
    done_us: int
    dt_us: int
    ik_mask: int
    sat: int
    lim_deg: float
    mag_in_deg: float
    mag_out_deg: float
    alpha: float
    raw_line: str


@dataclass
class PoseCmd:
    t_send_ns: int
    x_mm: float
    y_mm: float
    z_mm: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


class StewartSerial:
    def __init__(self, port: str, baud: int = 115200, open_delay_s: float = 1.5):
        import serial  # local import for better error messages when deps missing

        self.port = normalize_serial_port(port)
        self.baud = int(baud)
        self.ser = serial.Serial(self.port, baudrate=self.baud, timeout=0)
        # ESP32 dev boards often auto-reset on serial open. Give it a moment to boot
        # so the first pose line isn't lost.
        if open_delay_s and open_delay_s > 0:
            time.sleep(float(open_delay_s))
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception:
            pass
        self._rx_buf = bytearray()
        self.sent_queue: Deque[PoseCmd] = deque()

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass

    def send_pose(self, cmd: PoseCmd) -> None:
        line = f"{cmd.x_mm:.3f} {cmd.y_mm:.3f} {cmd.z_mm:.3f} {cmd.roll_deg:.3f} {cmd.pitch_deg:.3f} {-1*cmd.yaw_deg:.3f}\n"
        # print("sending pose: %s", line)
        self.ser.write(line.encode("ascii", errors="ignore"))
        self.sent_queue.append(cmd)

    def _parse_line(self, line: str, t_rx_ns: int) -> Optional[StewartAck]:
        m = ACK_RE.match(line.strip())
        if not m:
            return None
        gd = m.groupdict()
        return StewartAck(
            t_rx_ns=t_rx_ns,
            seq=int(gd["seq"]),
            status=str(gd["status"]),
            flags=str(gd["flags"]),
            rx_us=int(gd["rx_us"]),
            done_us=int(gd["done_us"]),
            dt_us=int(gd["dt_us"]),
            ik_mask=int(gd["ik_mask"], 16),
            sat=int(gd["sat"]),
            lim_deg=float(gd["lim"]),
            mag_in_deg=float(gd["mag_in"]),
            mag_out_deg=float(gd["mag_out"]),
            alpha=float(gd["a"]),
            raw_line=line.strip(),
        )

    def poll_acks(self) -> list[Tuple[StewartAck, Optional[PoseCmd]]]:
        """
        Returns list of (ack, paired_pose_cmd).
        Pairing strategy: FIFO — the next ACK corresponds to the oldest outstanding pose cmd.
        This works as long as:
          - firmware emits one ACK per successfully parsed pose command, and
          - we don't interleave other pose traffic on the same link.
        """
        out: list[Tuple[StewartAck, Optional[PoseCmd]]] = []

        n = self.ser.in_waiting
        if n <= 0:
            return out

        data = self.ser.read(n)
        if not data:
            return out

        self._rx_buf.extend(data)

        while True:
            try:
                idx = self._rx_buf.index(b"\n")
            except ValueError:
                break
            raw = self._rx_buf[:idx + 1]
            del self._rx_buf[:idx + 1]
            try:
                line = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                continue
            if not line:
                continue
            t_rx = monotonic_ns()
            ack = self._parse_line(line, t_rx)
            if ack is None:
                continue
            pose = self.sent_queue.popleft() if self.sent_queue else None
            out.append((ack, pose))

        return out


# ---------------- PX4 MAVLink ----------------

@dataclass
class Px4Attitude:
    t_rx_ns: int
    time_boot_ms: Optional[int]
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    rollspeed_deg_s: Optional[float]
    pitchspeed_deg_s: Optional[float]
    yawspeed_deg_s: Optional[float]


class Px4Mavlink:
    def __init__(self, com: str, baud: int = 115200, attitude_hz: float = 100.0, dialect: str = "common"):
        from pymavlink import mavutil  # type: ignore

        os.environ.setdefault("MAVLINK20", "1")
        os.environ["MAVLINK_DIALECT"] = dialect

        self.mavutil = mavutil
        self.port = normalize_serial_port(com)
        self.baud = int(baud)
        self.attitude_hz = float(attitude_hz)
        self.master = mavutil.mavlink_connection(self.port, baud=self.baud, autoreconnect=False, robust_parsing=True)
        hb = self.master.wait_heartbeat(timeout=10.0)
        if hb is None:
            raise RuntimeError(f"PX4 heartbeat not received within 10s (port={self.port!r}).")

        # Request ATTITUDE at desired rate.
        self._request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, self.attitude_hz)

        self.latest: Optional[Px4Attitude] = None

    def _request_message_interval(self, msg_id: int, hz: float) -> None:
        interval_us = int(1e6 / hz) if hz > 0 else 0
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            self.mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            msg_id,
            interval_us,
            0, 0, 0, 0, 0,
        )

    def poll(self) -> Optional[Px4Attitude]:
        msg = self.master.recv_match(type="ATTITUDE", blocking=False)
        if msg is None:
            return None
        t_rx = monotonic_ns()
        # pymavlink ATTITUDE fields: roll/pitch/yaw (rad), rollspeed/pitchspeed/yawspeed (rad/s), time_boot_ms (ms) optional
        roll = float(np.rad2deg(msg.roll))
        pitch = float(np.rad2deg(msg.pitch))
        yaw = float(np.rad2deg(msg.yaw))

        logging.debug("roll: %s, pitch: %s, yaw: %s", roll, pitch, yaw)

        time_boot_ms = getattr(msg, "time_boot_ms", None)
        att = Px4Attitude(
            t_rx_ns=t_rx,
            time_boot_ms=int(time_boot_ms) if time_boot_ms is not None else None,
            roll_deg=roll,
            pitch_deg=pitch,
            yaw_deg=yaw,
            rollspeed_deg_s=float(np.rad2deg(msg.rollspeed)) if hasattr(msg, "rollspeed") else None,
            pitchspeed_deg_s=float(np.rad2deg(msg.pitchspeed)) if hasattr(msg, "pitchspeed") else None,
            yawspeed_deg_s=float(np.rad2deg(msg.yawspeed)) if hasattr(msg, "yawspeed") else None,
        )
        self.latest = att
        return att


# ---------------- X-Plane UDP (best-effort) ----------------

@dataclass
class XPlaneState:
    t_rx_ns: int
    # Primary attitude signals (degrees)
    roll_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    heading_deg: Optional[float] = None
    # Body rates (deg/s)
    p_deg_s: Optional[float] = None
    q_deg_s: Optional[float] = None
    r_deg_s: Optional[float] = None
    # Optional extras that are commonly useful
    v_ind_kias: Optional[float] = None
    v_true_ktas: Optional[float] = None
    climb_rate: Optional[float] = None
    throttle_actual: Optional[float] = None


class XPlaneReceiver:
    """
    Receive X-Plane "DATA\\0" packets in the same raw matrix layout used by the Simulink model.

    Expected payload: header "DATA\\0" (5 bytes) + 72 float32 (9x8 matrix).
    Matrix is interpreted in MATLAB/Simulink column-major order.
    """

    # X-Plane packets we see in practice may be either:
    # - "DATA\0" (5 bytes) + N * (int32 index + 8 float32)  [classic documented format]
    # - "DATA"  (4 bytes) + N * (int32 index + 8 float32)   [seen in some toolchains]
    #
    # We therefore accept any payload that starts with "DATA" and then auto-detect
    # header length by checking which remaining length is a multiple of group size.
    DATA_PREFIX = b"DATA"
    GROUP = struct.Struct("<i8f")  # index + 8 float32 (36 bytes)
    GROUP_SIZE = GROUP.size

    def __init__(self, listen_host: str, listen_port: int):
        self.addr = (listen_host, int(listen_port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.addr)
        self.sock.setblocking(False)
        self.latest: Optional[XPlaneState] = None
        self._warn_every_n = 200  # rate-limit noisy warnings
        self._warn_count = 0

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass

    def poll(self) -> Optional[XPlaneState]:
        try:
            data, _src = self.sock.recvfrom(4096)
        except BlockingIOError:
            return None
        t_rx = monotonic_ns()
        if not data.startswith(self.DATA_PREFIX):
            # Unknown payload; ignore.
            return None

        # Auto-detect header length (4 vs 5 bytes)
        hdr_len = None
        for cand in (5, 4):
            rem = len(data) - cand
            if rem >= self.GROUP_SIZE and (rem % self.GROUP_SIZE) == 0:
                hdr_len = cand
                break
        if hdr_len is None:
            # Can't parse reliably; rate-limit warning.
            self._warn_count += 1
            if (self._warn_count % self._warn_every_n) == 1:
                logging.warning("X-Plane UDP: unrecognized DATA packet length=%d (expected rem multiple of %d).", len(data), self.GROUP_SIZE)
            return None

        payload = data[hdr_len:]

        # Parse N groups.
        groups: Dict[int, Tuple[float, ...]] = {}
        off = 0
        while off + self.GROUP_SIZE <= len(payload):
            idx, *vals = self.GROUP.unpack_from(payload, off)
            off += self.GROUP_SIZE
            groups[int(idx)] = tuple(float(v) for v in vals)

        # Decode the signals we care about. Use group indices (robust to ordering).
        # Typical X-Plane mapping:
        # - index 17: pitch, roll, heading (true) in degrees
        # - index 16: angular rates (often P,Q,R) — exact ordering can differ by aircraft/config
        pitch = roll = heading = None
        p = q = r = None

        if 17 in groups:
            v = groups[17]
            # Use the common convention: v[0]=pitch, v[1]=roll, v[2]=heading
            pitch = float(v[0])
            roll = float(v[1])
            heading = float(v[2])

        if 16 in groups:
            v = groups[16]
            # Match the Simulink decoder intent (q,p,r taken from rows 2..4):
            # we interpret v[0]=q, v[1]=p, v[2]=r.
            q = float(v[0])
            p = float(v[1])
            r = float(v[2])

        # Optional extra: IAS/TAS/climb are often in other indices; we leave them None unless later needed.
        v_ind = None
        v_true = None
        climb = None
        thr_act = None

        # print(f"pitch: {pitch}, roll: {roll}, heading: {heading}, p: {p}, q: {q}, r: {r}")

        st = XPlaneState(
            t_rx_ns=t_rx,
            roll_deg=roll,
            pitch_deg=pitch,
            heading_deg=heading,
            p_deg_s=p,
            q_deg_s=q,
            r_deg_s=r,
            v_ind_kias=v_ind,
            v_true_ktas=v_true,
            climb_rate=climb,
            throttle_actual=thr_act,
        )
        self.latest = st
        return st


class XPlaneSender:
    """
    X-Plane control injection using the same "DATA\\0" packing strategy as the Simulink model:
      - Header: "DATA\\0"
      - group 8  : elevator/aileron/rudder (others = -999)
      - group 25 : throttle (others = -999)
    """

    DATA_HDR = b"DATA\x00"
    GROUP = struct.Struct("<i8f")  # index(int32) + 8 float32

    DREF_HDR = b"DREF\x00"
    DREF = struct.Struct("<f")  # single float32

    def __init__(self, host: str, port: int):
        self.target = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last_override_s = 0.0

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass

    def send_dref(self, name: str, value: float) -> None:
        # X-Plane format: "DREF\0" + float32 + dataref string + "\0"
        bname = name.encode("ascii", errors="ignore")[:500]
        pkt = self.DREF_HDR + self.DREF.pack(float(value)) + bname + b"\x00"
        self.sock.sendto(pkt, self.target)

    def send_controls(self, elevator: float, aileron: float, rudder: float, throttle: float = 0.5) -> None:
        """
        Sends two DATA groups:
          - 8: [elevator, aileron, rudder, -999, -999, -999, -999, -999]
          - 25:[throttle, -999, ...]
        This matches the Simulink pack function snippet.
        """
        # print(f"sending controls: elevator: {elevator}, aileron: {aileron}, rudder: {rudder}, throttle: {throttle}")
        now = time.time()
        if now - self._last_override_s > 1.0:
            self._last_override_s = now
            # These are the usual knobs that make injected controls actually apply.
            self.send_dref("sim/operation/override/override_joystick", 1.0)
            self.send_dref("sim/operation/override/override_throttles", 1.0)

        def cl(x: float, lo: float, hi: float) -> float:
            return float(max(lo, min(hi, x)))

        no = float(-999.0)
        ele = cl(elevator, -1.0, 1.0)
        ail = cl(aileron, -1.0, 1.0)
        rud = cl(rudder, -1.0, 1.0)
        thr = cl(throttle, 0.0, 1.0)

        g8 = self.GROUP.pack(8, ele, ail, rud, no, no, no, no, no)
        g25 = self.GROUP.pack(25, thr, no, no, no, no, no, no, no)
        pkt = self.DATA_HDR + g8 + g25
        self.sock.sendto(pkt, self.target)
        print(f"self.target: {self.target}")


# ---------------- Logger ----------------

class RunLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._tick_f = open(self.run_dir / "tick.csv", "w", newline="", encoding="utf-8")
        self._ack_f = open(self.run_dir / "stewart_ack.csv", "w", newline="", encoding="utf-8")
        self._px4_f = open(self.run_dir / "px4_att.csv", "w", newline="", encoding="utf-8")
        self._xp_f = open(self.run_dir / "xplane_att.csv", "w", newline="", encoding="utf-8")

        self.tick_w: Optional[csv.DictWriter] = None
        self.ack_w: Optional[csv.DictWriter] = None
        self.px4_w: Optional[csv.DictWriter] = None
        self.xp_w: Optional[csv.DictWriter] = None

    def close(self) -> None:
        for f in (self._tick_f, self._ack_f, self._px4_f, self._xp_f):
            try:
                f.close()
            except Exception:
                pass

    def write_meta(self, meta: Dict) -> None:
        (self.run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _ensure_writer(current: Optional[csv.DictWriter], f, row: Dict) -> csv.DictWriter:
        if current is not None:
            current.writerow(row)
            return current
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
        return w

    def log_xplane(self, st: XPlaneState) -> None:
        row = asdict(st)
        self.xp_w = self._ensure_writer(self.xp_w, self._xp_f, row)

    def log_px4(self, att: Px4Attitude) -> None:
        row = asdict(att)
        self.px4_w = self._ensure_writer(self.px4_w, self._px4_f, row)

    def log_ack(self, ack: StewartAck, pose: Optional[PoseCmd]) -> None:
        row = asdict(ack)
        if pose is not None:
            row.update({f"cmd_{k}": v for k, v in asdict(pose).items()})
            row["serial_rtt_ms"] = (ack.t_rx_ns - pose.t_send_ns) / 1e6
        self.ack_w = self._ensure_writer(self.ack_w, self._ack_f, row)

    def log_tick(self, row: Dict) -> None:
        self.tick_w = self._ensure_writer(self.tick_w, self._tick_f, row)


# ---------------- Main ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Python host for Stewart Platform Physical HILS (100 Hz).")

    # X-Plane
    p.add_argument("--xplane-rx-host", default="127.0.0.1", help="Host bind for X-Plane UDP receive (DATA packets).")
    p.add_argument("--xplane-rx-port", type=int, default=49004, help="UDP port to receive X-Plane DATA packets.")
    p.add_argument("--xplane-tx-host", default="127.0.0.1", help="X-Plane UDP destination host.")
    p.add_argument("--xplane-tx-port", type=int, default=49001, help="X-Plane UDP destination port (controls).")
    p.add_argument("--xplane-send", choices=["none", "data"], default="data", help="Control injection method.")

    # ESP32 (Stewart)
    p.add_argument("--stewart-com", default="COM14", help="ESP32 serial port (pose command + ACK).")
    p.add_argument("--stewart-baud", type=int, default=115200, help="ESP32 serial baud.")
    p.add_argument("--stewart-open-delay-s", type=float, default=1.5, help="Delay after opening ESP32 serial (handles auto-reset).")
    p.add_argument("--pose-z-mm", type=float, default=0.0, help="Default Z translation (mm) for pose commands.")

    # PX4
    p.add_argument("--px4-com", default="COM13", help="PX4 serial port (MAVLink).")
    p.add_argument("--px4-baud", type=int, default=115200, help="PX4 baud (often ignored on USB CDC).")
    p.add_argument("--px4-att-hz", type=float, default=100.0, help="Requested PX4 ATTITUDE stream rate (Hz).")
    p.add_argument("--px4-dialect", default="common", help="MAVLink dialect (PX4: common).")
    p.add_argument("--no-px4", action="store_true", help="Disable PX4/MAVLink entirely (platform-only run).")

    # Control tick
    p.add_argument("--tick-hz", type=float, default=100.0, help="Host control tick (Hz).")

    # PID (control surfaces)
    p.add_argument("--sp-roll-deg", type=float, default=0.0, help="Roll setpoint (deg).")
    p.add_argument("--sp-pitch-deg", type=float, default=0.0, help="Pitch setpoint (deg).")
    p.add_argument("--sp-yaw-deg", type=float, default=0.0, help="Yaw setpoint (deg).")

    p.add_argument("--kp-roll", type=float, default=0.2)
    p.add_argument("--ki-roll", type=float, default=0.0)
    p.add_argument("--kd-roll", type=float, default=0.1)
    p.add_argument("--kp-pitch", type=float, default=0.7)
    p.add_argument("--ki-pitch", type=float, default=0.0)
    p.add_argument("--kd-pitch", type=float, default=0.1)
    p.add_argument("--kp-yaw", type=float, default=0.2)
    p.add_argument("--ki-yaw", type=float, default=0.0)
    p.add_argument("--kd-yaw", type=float, default=0.0)

    p.add_argument("--i-limit", type=float, default=10.0)
    p.add_argument("--u-limit", type=float, default=1.0, help="Control output limit (normalized, -1..+1).")
    p.add_argument("--throttle", type=float, default=1.0, help="Injected throttle command (clamped to 0..1).")

    # Logging
    p.add_argument("--run-name", default="", help="Optional run name suffix.")
    p.add_argument("--log-dir", default="logs", help="Base log directory.")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    tick_hz = float(args.tick_hz)
    if tick_hz <= 0:
        raise SystemExit("--tick-hz must be > 0")
    tick_dt_ns = int(round(1e9 / tick_hz))

    run_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    suffix = f"-{args.run_name.strip()}" if args.run_name.strip() else ""
    run_dir = Path(args.log_dir) / f"run-{run_stamp}{suffix}"

    log = RunLogger(run_dir)
    log.write_meta(
        {
            "created_utc": utc_iso(),
            "tick_hz": tick_hz,
            "xplane_rx": {"host": args.xplane_rx_host, "port": int(args.xplane_rx_port)},
            "xplane_tx": {"host": args.xplane_tx_host, "port": int(args.xplane_tx_port), "mode": args.xplane_send},
            "stewart": {"com": args.stewart_com, "baud": int(args.stewart_baud)},
            "px4": (
                {"enabled": False}
                if args.no_px4
                else {"enabled": True, "com": args.px4_com, "baud": int(args.px4_baud), "att_hz": float(args.px4_att_hz), "dialect": args.px4_dialect}
            ),
            "pid": {
                "sp_deg": {"roll": float(args.sp_roll_deg), "pitch": float(args.sp_pitch_deg), "yaw": float(args.sp_yaw_deg)},
                "kp": {"roll": float(args.kp_roll), "pitch": float(args.kp_pitch), "yaw": float(args.kp_yaw)},
                "ki": {"roll": float(args.ki_roll), "pitch": float(args.ki_pitch), "yaw": float(args.ki_yaw)},
                "kd": {"roll": float(args.kd_roll), "pitch": float(args.kd_pitch), "yaw": float(args.kd_yaw)},
                "i_limit": float(args.i_limit),
                "u_limit": float(args.u_limit),
            },
        }
    )

    xp_rx = XPlaneReceiver(args.xplane_rx_host, int(args.xplane_rx_port))
    xp_tx = None if args.xplane_send == "none" else XPlaneSender(args.xplane_tx_host, int(args.xplane_tx_port))
    print(f"[hils_host] stewart_serial={args.stewart_com} baud={int(args.stewart_baud)}")
    if args.no_px4:
        print("[hils_host] px4: disabled (--no-px4)")
    else:
        print(f"[hils_host] px4_serial={args.px4_com} baud={int(args.px4_baud)} att_hz={float(args.px4_att_hz)}")
    print(f"[hils_host] xplane_rx={args.xplane_rx_host}:{int(args.xplane_rx_port)} xplane_tx={args.xplane_tx_host}:{int(args.xplane_tx_port)} mode={args.xplane_send}")

    st = StewartSerial(args.stewart_com, int(args.stewart_baud), float(args.stewart_open_delay_s))
    px4 = None if args.no_px4 else Px4Mavlink(args.px4_com, int(args.px4_baud), float(args.px4_att_hz), args.px4_dialect)

    pid_roll = PID(args.kp_roll, args.ki_roll, args.kd_roll, args.i_limit, args.u_limit) if px4 is not None else None
    pid_pitch = PID(args.kp_pitch, args.ki_pitch, args.kd_pitch, args.i_limit, args.u_limit) if px4 is not None else None
    pid_yaw = PID(args.kp_yaw, args.ki_yaw, args.kd_yaw, args.i_limit, args.u_limit) if px4 is not None else None

    print(f"[hils_host] run_dir={run_dir}")
    print("[hils_host] Ctrl+C to stop.")

    last_tick_ns = monotonic_ns()
    next_tick_ns = last_tick_ns + tick_dt_ns

    last_px4_ns = None
    tick_count = 0
    ack_total = 0
    warned_no_ack = False

    try:
        while True:
            sleep_until_ns(next_tick_ns)
            t_tick = monotonic_ns()

            # Poll inputs (drain buffers)
            while True:
                m = xp_rx.poll()
                if m is None:
                    break
                log.log_xplane(m)

            while True:
                if px4 is None:
                    break
                m = px4.poll()
                if m is None:
                    break
                log.log_px4(m)

            for ack, pose in st.poll_acks():
                logging.debug("ack: %s, pose: %s", ack, pose)
                log.log_ack(ack, pose)
                ack_total += 1

            # Build pose command: by default, drive platform to match X-Plane attitude if available.
            xp = xp_rx.latest
            roll_cmd = float(xp.roll_deg) if (xp and xp.roll_deg is not None) else 0.0
            pitch_cmd = float(xp.pitch_deg) if (xp and xp.pitch_deg is not None) else 0.0
            yaw_cmd = float(xp.heading_deg) if (xp and xp.heading_deg is not None) else 0.0

            pose = PoseCmd(
                t_send_ns=t_tick,
                x_mm=0.0,
                y_mm=0.0,
                z_mm=float(args.pose_z_mm),
                roll_deg=roll_cmd,
                pitch_deg=pitch_cmd,
                yaw_deg=yaw_cmd,
            )
            st.send_pose(pose)

            # If we're sending poses but receiving no ACKs, something is wrong:
            # - wrong COM port (not the ESP32)
            # - ESP32 firmware not running / held in reset
            # - STEWART_STATUS_SERIAL disabled in firmware
            # - serial wiring issue
            if not warned_no_ack and tick_count >= int(tick_hz) and ack_total == 0:
                warned_no_ack = True
                logging.warning(
                    "No ESP32 ACK received after ~1s of sending pose commands. "
                    "Most likely wrong --stewart-com or ESP32 not emitting status over serial."
                )

            # Compute control injection from PX4 attitude (custom PID in host).
            # This is the simplest demonstrator loop: PX4 attitude -> PID -> X-Plane controls.
            px = px4.latest if px4 is not None else None
            if px is not None and last_px4_ns is not None:
                dt_s = (px.t_rx_ns - last_px4_ns) / 1e9
            else:
                dt_s = 0.0

            if px is not None and pid_roll is not None and pid_pitch is not None and pid_yaw is not None:
                last_px4_ns = px.t_rx_ns
                u_ail = pid_roll.update(float(args.sp_roll_deg), px.roll_deg, dt_s)
                u_ele = pid_pitch.update(float(args.sp_pitch_deg), px.pitch_deg, dt_s)
                u_rud = pid_yaw.update(float(args.sp_yaw_deg), px.yaw_deg, dt_s)
            else:
                u_ail = u_ele = u_rud = 0.0

            if xp_tx is not None and args.xplane_send == "data":
                # Note ordering matches Simulink pack: (elevator, aileron, rudder, throttle)
                xp_tx.send_controls(u_ele, u_ail, u_rud, throttle=float(args.throttle))

            # Tick log (latest-sample-hold snapshot)
            row = {
                "t_tick_ns": t_tick,
                "tick_idx": tick_count,
                "xp_t_rx_ns": xp.t_rx_ns if xp else "",
                "xp_roll_deg": xp.roll_deg if xp else "",
                "xp_pitch_deg": xp.pitch_deg if xp else "",
                "xp_heading_deg": xp.heading_deg if xp else "",
                "px4_t_rx_ns": px.t_rx_ns if px else "",
                "px4_time_boot_ms": px.time_boot_ms if px else "",
                "px4_roll_deg": px.roll_deg if px else "",
                "px4_pitch_deg": px.pitch_deg if px else "",
                "px4_yaw_deg": px.yaw_deg if px else "",
                "cmd_roll_deg": pose.roll_deg,
                "cmd_pitch_deg": pose.pitch_deg,
                "cmd_yaw_deg": pose.yaw_deg,
                "u_ail": u_ail,
                "u_ele": u_ele,
                "u_rud": u_rud,
                "throttle_cmd": float(args.throttle),
                "queue_depth": len(st.sent_queue),
                "ack_total": ack_total,
            }
            log.log_tick(row)

            tick_count += 1
            next_tick_ns += tick_dt_ns

    except KeyboardInterrupt:
        print("\n[hils_host] stopped.")
        return 0
    finally:
        try:
            xp_rx.close()
            if xp_tx is not None:
                xp_tx.close()
            st.close()
        finally:
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())


