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


# ---------------- Angle helpers ----------------

def wrap180(deg: float) -> float:
    """
    Wrap degrees into [-180, 180).
    """
    x = (float(deg) + 180.0) % 360.0 - 180.0
    return 180.0 if x == -180.0 else x


def unwrap_deg(prev_unwrapped: Optional[float], new_wrapped_deg: float) -> float:
    """
    Unwrap a wrapped angle (typically in [-180,180)) into a continuous signal.
    """
    nw = float(new_wrapped_deg)
    if prev_unwrapped is None:
        return nw
    # delta in [-180,180)
    delta = wrap180(nw - wrap180(prev_unwrapped))
    return float(prev_unwrapped + delta)


def angle_error_deg(setpoint_deg: float, meas_deg: float) -> float:
    """
    Smallest signed angle error (deg) in [-180, 180).
    """
    return wrap180(float(setpoint_deg) - float(meas_deg))


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

    def update_angle(self, setpoint_deg: float, meas_deg: float, dt_s: float) -> float:
        """
        PID for angular signals (wraps error into [-180,180) to avoid discontinuities).
        """
        if not (0.0 < dt_s < 10.0):
            dt_s = 0.0
        err = angle_error_deg(setpoint_deg, meas_deg)

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

    def reset(self) -> None:
        self.i = 0.0
        self.prev_err = 0.0
        self.has_prev = False


# ---------------- Scenarios ----------------

@dataclass(frozen=True)
class TrimTransitionScenario:
    """
    Time-based, deterministic trim transition scenario for conference baseline runs:
      0 deg -> +step deg -> 0 deg (optional return).

    Each plateau can be split into:
      warmup_s: allow the loop to settle
      hold_s  : "measurement" window at the setpoint
    """

    warmup_s: float
    hold_s: float
    step_pitch_deg: float
    return_to_zero: bool = True

    @property
    def total_s(self) -> float:
        # 0 -> step -> 0 (each has warmup + hold)
        if self.return_to_zero:
            return 3.0 * float(self.warmup_s) + 3.0 * float(self.hold_s)
        return 2.0 * float(self.warmup_s) + 2.0 * float(self.hold_s)


def trim_transition_setpoint(t_s: float, scn: TrimTransitionScenario) -> tuple[float, str, bool]:
    """
    Returns (sp_pitch_deg, phase_name, done).
    Purely time-based so each run has deterministic length.
    """
    t = max(0.0, float(t_s))
    w = max(0.0, float(scn.warmup_s))
    h = max(0.0, float(scn.hold_s))
    step = float(scn.step_pitch_deg)

    # Phases:
    #   warmup_0 (0) -> hold_0 (0) -> warmup_step (+step) -> hold_step (+step)
    #   -> (optional) warmup_return (0) -> hold_return (0) -> done
    if t < w:
        return 0.0, "warmup_0", False
    if t < w + h:
        return 0.0, "hold_0", False
    if t < 2 * w + h:
        return step, "warmup_step", False
    if t < 2 * w + 2 * h:
        return step, "hold_step", False

    if not bool(scn.return_to_zero):
        return 0.0, "done", True

    if t < 3 * w + 2 * h:
        return 0.0, "warmup_return", False
    if t < 3 * w + 3 * h:
        return 0.0, "hold_return", False
    return 0.0, "done", True


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


ACK_RE_FULL = re.compile(
    r"^ACK\s+seq=(?P<seq>\d+)\s+status=(?P<status>\S+)\s+flags=(?P<flags>\S+)\s+"
    r"rx_us=(?P<rx_us>\d+)\s+done_us=(?P<done_us>\d+)\s+dt_us=(?P<dt_us>\d+)\s+"
    r"ik_mask=0x(?P<ik_mask>[0-9A-Fa-f]{2})\s+sat=(?P<sat>[01])\s+"
    r"lim=(?P<lim>[-+0-9\.]+)\s+mag=(?P<mag_in>[-+0-9\.]+)->(?P<mag_out>[-+0-9\.]+)\s+"
    r"a=(?P<a>[-+0-9\.]+)"
)

# Minimal ACK emitted by firmware (recommended for 100 Hz operation):
#   ACK seq=<n> rx_us=<t> done_us=<t> sat=<0|1> ik=<XX>
ACK_RE_MIN = re.compile(
    r"^ACK\s+seq=(?P<seq>\d+)\s+rx_us=(?P<rx_us>\d+)\s+done_us=(?P<done_us>\d+)\s+sat=(?P<sat>[01])\s+ik=(?P<ik>[0-9A-Fa-f]{2})$"
)


@dataclass
class StewartAck:
    t_rx_ns: int
    seq: int
    rx_us: int
    done_us: int
    dt_us: int
    ik_mask: int
    sat: int
    status: str = ""
    flags: str = ""
    lim_deg: float = float("nan")
    mag_in_deg: float = float("nan")
    mag_out_deg: float = float("nan")
    alpha: float = float("nan")
    raw_line: str = ""


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
        logging.debug("sending pose: %s", line)
        self.ser.write(line.encode("ascii", errors="ignore"))
        self.sent_queue.append(cmd)

    def _parse_line(self, line: str, t_rx_ns: int) -> Optional[StewartAck]:
        s = line.strip()

        m = ACK_RE_FULL.match(s)
        if m:
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
                raw_line=s,
            )

        m = ACK_RE_MIN.match(s)
        if m:
            gd = m.groupdict()
            rx_us = int(gd["rx_us"])
            done_us = int(gd["done_us"])
            dt_us = max(0, done_us - rx_us)
            return StewartAck(
                t_rx_ns=t_rx_ns,
                seq=int(gd["seq"]),
                status="MIN",
                flags="",
                rx_us=rx_us,
                done_us=done_us,
                dt_us=dt_us,
                ik_mask=int(gd["ik"], 16),
                sat=int(gd["sat"]),
                raw_line=s,
            )

        return None

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
        # On Windows, X-Plane may bind its own UDP source port (e.g. "Port we send from").
        # If that equals our listen port, the bind will fail with WinError 10048.
        # SO_REUSEADDR improves behavior on some platforms, but the correct fix is to
        # set X-Plane "Port we send from" to something else, and "Port we send to"
        # to our --xplane-rx-port.
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        try:
            self.sock.bind(self.addr)
        except OSError as e:
            raise OSError(
                f"Failed to bind X-Plane RX UDP socket on {self.addr}. "
                f"If X-Plane is set to 'Port we send from' = {self.addr[1]}, change that to a different port "
                f"(e.g. 49005) and keep 'Port we send to' = {self.addr[1]}."
            ) from e
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
            data, _src = self.sock.recvfrom(8192)
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

    # Header nuance:
    # - X-Plane documentation commonly shows "DATA\\0" (null terminator) => b"DATA\\x00" (5 bytes).
    # - The Simulink model in this repo uses the literal ASCII '0' (48) as the 5th byte => b"DATA0".
    # - Some setups use just "DATA" (4 bytes).
    #
    # Make this configurable for byte-for-byte parity with existing toolchains.
    DATA_HDR_NULL = b"DATA\x00"
    DATA_HDR_ASCII0 = b"DATA0"
    DATA_HDR_4 = b"DATA"
    GROUP = struct.Struct("<i8f")  # index(int32) + 8 float32

    DREF_HDR = b"DREF\x00"
    DREF = struct.Struct("<f")  # single float32

    def __init__(self, host: str, port: int, header_mode: str = "data0"):
        self.target = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last_override_s = 0.0
        self._dump_remaining = 0

        hm = (header_mode or "").strip().lower()
        if hm in ("data0", "ascii0", "data_ascii0"):
            self.data_hdr = self.DATA_HDR_ASCII0
        elif hm in ("null", "data_null", "data\\0", "data\0"):
            self.data_hdr = self.DATA_HDR_NULL
        elif hm in ("data", "data4", "4"):
            self.data_hdr = self.DATA_HDR_4
        else:
            raise ValueError(f"Unknown X-Plane TX header mode: {header_mode!r} (use data0|null|data)")

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

    def enable_packet_dump(self, n: int = 1) -> None:
        self._dump_remaining = max(0, int(n))

    def send_controls(self, elevator: float, aileron: float, rudder: float, throttle: float = 0.5) -> None:
        """
        Sends two DATA groups:
          - 8: [elevator, aileron, rudder, -999, -999, -999, -999, -999]
          - 25:[throttle, -999, ...]
        This matches the Simulink pack function snippet.
        """
        # Keep this function quiet by default (it can run at high rate).
        # If you need verification, use --xplane-tx-dump-n to dump bytes.
        # if now - self._last_override_s > 1.0:
        #     self._last_override_s = now
        #     # These are the usual knobs that make injected controls actually apply.
        #     self.send_dref("sim/operation/override/override_joystick", 1.0)
        #     self.send_dref("sim/operation/override/override_throttles", 1.0)

        def cl(x: float, lo: float, hi: float) -> float:
            return float(max(lo, min(hi, x)))

        no = float(-999.0)
        ele = cl(elevator, -1.0, 1.0)
        ail = cl(aileron, -1.0, 1.0)
        rud = cl(rudder, -1.0, 1.0)
        thr = cl(throttle, 0.0, 1.0)

        g8 = self.GROUP.pack(8, ele, ail, rud, no, no, no, no, no)
        g25 = self.GROUP.pack(25, thr, no, no, no, no, no, no, no)
        # Simulink parity: header must be "DATA0" (ASCII '0' byte 48), followed immediately by records.
        # DO NOT insert an extra record for '48' — it's a header byte, not a DATA group index.
        pkt = self.data_hdr + g8 + g25
        self.sock.sendto(pkt, self.target)
        logging.warning("Sent controls: elevator: %s, aileron: %s, rudder: %s, throttle: %s", ele, ail, rud, thr)
        if self._dump_remaining > 0:
            print(f"[xplane_tx] target={self.target} header={self.data_hdr!r} len={len(pkt)} hex={pkt.hex()}")
            self._dump_remaining -= 1


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
    p.add_argument("--xplane-tx-port", type=int, default=49000, help="X-Plane UDP destination port (controls).")
    p.add_argument("--xplane-send", choices=["none", "data"], default="data", help="Control injection method.")
    p.add_argument(
        "--xplane-tx-hdr",
        choices=["data0", "null", "data"],
        default="data0",
        help="Outgoing X-Plane DATA header bytes. data0 matches Simulink (b'DATA0').",
    )
    p.add_argument(
        "--xplane-tx-dump-n",
        type=int,
        default=0,
        help="If >0, hex-dump the first N outgoing X-Plane control packets (byte-for-byte debug vs Simulink).",
    )

    # ESP32 (Stewart)
    p.add_argument("--stewart-com", default="COM14", help="ESP32 serial port (pose command + ACK).")
    p.add_argument("--stewart-baud", type=int, default=500000, help="ESP32 serial baud.")
    p.add_argument("--stewart-open-delay-s", type=float, default=1.5, help="Delay after opening ESP32 serial (handles auto-reset).")
    p.add_argument("--pose-z-mm", type=float, default=20.0, help="Default Z translation (mm) for pose commands.")
    p.add_argument("--pose-start-hold-s", type=float, default=0.0, help="Hold zero pose (r/p/y=0) for this many seconds after start.")
    p.add_argument("--pose-ramp-s", type=float, default=3.0, help="Ramp pose commands from zero to target over this many seconds after hold.")
    p.add_argument("--platform-yaw-relative", action="store_true", default=True, help="Command platform yaw relative to initial X-Plane heading (so initial heading becomes 0 deg).")

    # PX4
    p.add_argument("--px4-com", default="COM13", help="PX4 serial port (MAVLink).")
    p.add_argument("--px4-baud", type=int, default=115200, help="PX4 baud (often ignored on USB CDC).")
    p.add_argument("--px4-att-hz", type=float, default=100.0, help="Requested PX4 ATTITUDE stream rate (Hz).")
    p.add_argument("--px4-dialect", default="common", help="MAVLink dialect (PX4: common).")
    p.add_argument("--no-px4", action="store_true", help="Disable PX4/MAVLink entirely (platform-only run).")

    # Control tick
    p.add_argument("--tick-hz", type=float, default=50.0, help="Host control tick (Hz).")

    # Heading alignment (X-Plane heading vs PX4 yaw)
    p.add_argument(
        "--yaw-bias",
        action="store_true", default=True,
        help="Enable yaw bias/unwrap alignment between X-Plane heading (0..360) and PX4 yaw (-180..180).",
    )
    p.add_argument(
        "--yaw-platform-flip",
        action="store_true",
        help="Flip sim heading sign when comparing to PX4 yaw (use if platform yaw convention is inverted).",
    )
    p.add_argument(
        "--yaw-bias-adapt-rate",
        type=float,
        default=0.0,
        help="If >0, slowly adapt yaw bias to cancel PX4 drift when yaw is 'still'. Units: 1/s (typical 0.01..0.1).",
    )
    p.add_argument(
        "--yaw-still-thresh-deg-s",
        type=float,
        default=2.0,
        help="Consider yaw 'still' if estimated sim yaw rate is below this (deg/s).",
    )

    # PID (control surfaces)
    p.add_argument("--sp-roll-deg", type=float, default=0.0, help="Roll setpoint (deg).")
    p.add_argument("--sp-pitch-deg", type=float, default=0.0, help="Pitch setpoint (deg).")
    p.add_argument("--sp-yaw-deg", type=float, default=0.0, help="Yaw setpoint (deg).")

    p.add_argument("--kp-roll", type=float, default=0.1)
    p.add_argument("--ki-roll", type=float, default=0.02)
    p.add_argument("--kd-roll", type=float, default=0.0)
    p.add_argument("--kp-pitch", type=float, default=0.03)
    p.add_argument("--ki-pitch", type=float, default=0.1)
    p.add_argument("--kd-pitch", type=float, default=0.0)
    p.add_argument("--kp-yaw", type=float, default=0.0)
    p.add_argument("--ki-yaw", type=float, default=0.0)
    p.add_argument("--kd-yaw", type=float, default=0.0)
    p.add_argument("--pid-start-delay-s", type=float, default=3.5, help="Delay before enabling PX4->PID->X-Plane control injection (lets PX4 settle).")

    p.add_argument("--i-limit", type=float, default=30.0)
    p.add_argument("--u-limit", type=float, default=1.0, help="Control output limit (normalized, -1..+1).")
    p.add_argument("--throttle", type=float, default=1.0, help="Injected throttle command (clamped to 0..1).")

    # Logging
    p.add_argument("--run-name", default="", help="Optional run name suffix.")
    p.add_argument("--log-dir", default="logs", help="Base log directory.")

    # Scenario (time-varying pitch setpoint, conference baseline)
    p.add_argument("--scenario", choices=["none", "trim-transition"], default="none", help="Run a finite scenario and exit cleanly.")
    p.add_argument("--scenario-warmup-s", type=float, default=10.0, help="Warmup duration per plateau (s).")
    p.add_argument("--scenario-hold-s", type=float, default=10.0, help="Hold/measurement duration per plateau (s).")
    p.add_argument("--scenario-step-pitch-deg", type=float, default=5.0, help="Pitch setpoint step (deg).")
    p.add_argument("--scenario-no-return", action="store_true", help="If set, do not return to 0 deg at the end (shorter run).")
    p.add_argument("--scenario-repeats", type=int, default=1, help="Number of repeats; each repeat is its own logs/run-...-repXX directory.")
    p.add_argument("--repeat-wait-enter", action="store_true", help="Between repeats, wait for Enter (manual X-Plane reset).")

    # Ready gate (prevents silently starting a scenario without live inputs/ACKs)
    p.add_argument(
        "--ready-timeout-s",
        type=float,
        default=10.0,
        help="Wait up to this long for initial X-Plane + PX4 (if enabled) + ESP32 ACK before starting scenario timer.",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    tick_hz = float(args.tick_hz)
    if tick_hz <= 0:
        raise SystemExit("--tick-hz must be > 0")
    tick_dt_ns = int(round(1e9 / tick_hz))

    # Scenario config
    scenario: Optional[TrimTransitionScenario] = None
    if str(args.scenario).strip().lower() == "trim-transition":
        scenario = TrimTransitionScenario(
            warmup_s=float(args.scenario_warmup_s),
            hold_s=float(args.scenario_hold_s),
            step_pitch_deg=float(args.scenario_step_pitch_deg),
            return_to_zero=(not bool(args.scenario_no_return)),
        )

    repeats = max(1, int(args.scenario_repeats))
    if scenario is None:
        # For infinite runs, repeats are meaningless (user stops with Ctrl+C).
        repeats = 1

    # Session-stable run name; each repeat gets its own directory with -repXX suffix.
    session_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    suffix = f"-{args.run_name.strip()}" if args.run_name.strip() else ""
    base_run_name = f"run-{session_stamp}{suffix}"

    xp_rx = XPlaneReceiver(args.xplane_rx_host, int(args.xplane_rx_port))
    xp_tx = None if args.xplane_send == "none" else XPlaneSender(args.xplane_tx_host, int(args.xplane_tx_port), header_mode=str(args.xplane_tx_hdr))
    if xp_tx is not None and int(args.xplane_tx_dump_n) > 0:
        xp_tx.enable_packet_dump(int(args.xplane_tx_dump_n))
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

    print("[hils_host] Ctrl+C to stop (or scenario will exit cleanly).")

    # Track the last tick timestamp for shutdown pose cmd.
    t_tick = monotonic_ns()

    try:
        for rep_idx in range(repeats):
            if rep_idx > 0 and bool(args.repeat_wait_enter):
                input("\n[hils_host] Reset X-Plane now, then press Enter to start next repeat...")

            rep_suffix = f"-rep{rep_idx+1:02d}" if repeats > 1 else ""
            run_dir = Path(args.log_dir) / f"{base_run_name}{rep_suffix}"
            log = RunLogger(run_dir)

            meta: Dict = {
                "created_utc": utc_iso(),
                "session_stamp": session_stamp,
                "repeat_idx": rep_idx + 1,
                "repeat_count": repeats,
                "tick_hz": tick_hz,
                "xplane_rx": {"host": args.xplane_rx_host, "port": int(args.xplane_rx_port)},
                "xplane_tx": {"host": args.xplane_tx_host, "port": int(args.xplane_tx_port), "mode": args.xplane_send, "hdr": str(args.xplane_tx_hdr)},
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
                    "pid_start_delay_s": float(args.pid_start_delay_s),
                },
                "scenario": {"kind": "none"},
            }
            if scenario is not None:
                meta["scenario"] = {
                    "kind": "trim-transition",
                    "warmup_s": float(scenario.warmup_s),
                    "hold_s": float(scenario.hold_s),
                    "step_pitch_deg": float(scenario.step_pitch_deg),
                    "return_to_zero": bool(scenario.return_to_zero),
                    "expected_total_s": float(scenario.total_s),
                    "ready_timeout_s": float(args.ready_timeout_s),
                }
            log.write_meta(meta)

            print(f"[hils_host] run_dir={run_dir}")

            # Reset PID state each repeat (requested).
            if pid_roll is not None:
                pid_roll.reset()
            if pid_pitch is not None:
                pid_pitch.reset()
            if pid_yaw is not None:
                pid_yaw.reset()

            # Clear serial buffers / pairing queue to avoid cross-repeat contamination.
            try:
                st.ser.reset_input_buffer()
                st.ser.reset_output_buffer()
            except Exception:
                pass
            st.sent_queue.clear()

            # Per-repeat timing
            t_repeat_start_ns = monotonic_ns()
            last_tick_ns = monotonic_ns()
            next_tick_ns = last_tick_ns + tick_dt_ns

            # Scenario clock starts only once inputs are "ready"
            ready = False
            t_ready_start_ns = monotonic_ns()
            t_scn0_ns: Optional[int] = None

            # Per-repeat state
            last_px4_ns = None
            tick_count = 0
            ack_total = 0
            warned_no_ack = False

            # Integrity stats (lightweight)
            t_prev_tick_ns: Optional[int] = None
            max_tick_gap_ms = 0.0
            xp_seen = 0
            px4_seen = 0
            ack_seen = 0
            max_queue_depth = 0

            # Yaw alignment state (continuous/unwrapped degrees)
            yaw_sim_unw: Optional[float] = None
            yaw_px4_unw: Optional[float] = None
            yaw_bias_deg: Optional[float] = None  # px4 - sim (after optional platform flip)
            last_sim_yaw_for_rate: Optional[Tuple[int, float]] = None  # (t_ns, sim_unw)
            xp_heading0_wrapped: Optional[float] = None

            try:
                while True:
                    sleep_until_ns(next_tick_ns)
                    t_tick = monotonic_ns()
                    t_elapsed_s = (t_tick - t_repeat_start_ns) / 1e9

                    if t_prev_tick_ns is not None:
                        gap_ms = (t_tick - t_prev_tick_ns) / 1e6
                        if gap_ms > max_tick_gap_ms:
                            max_tick_gap_ms = float(gap_ms)
                    t_prev_tick_ns = t_tick

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

                    # Latest samples
                    xp = xp_rx.latest
                    px = px4.latest if px4 is not None else None

                    if xp is not None and (xp.roll_deg is not None or xp.pitch_deg is not None or xp.heading_deg is not None):
                        xp_seen += 1
                    if px is not None:
                        px4_seen += 1
                    if ack_total > 0:
                        ack_seen = ack_total

                    # Ready gate: require at least one X-Plane sample, one PX4 sample (if enabled), and at least one ACK.
                    if not ready and scenario is not None:
                        has_xp = xp_seen > 0
                        has_px4 = True if px4 is None else (px4_seen > 0)
                        has_ack = ack_total > 0
                        if has_xp and has_px4 and has_ack:
                            ready = True
                            t_scn0_ns = t_tick
                            logging.info("[scenario] ready: starting scenario timer (repeat %d/%d)", rep_idx + 1, repeats)
                        else:
                            if (t_tick - t_ready_start_ns) / 1e9 > float(args.ready_timeout_s):
                                raise RuntimeError(
                                    f"Ready gate timeout after {float(args.ready_timeout_s):.1f}s "
                                    f"(has_xp={has_xp}, has_px4={has_px4}, has_ack={has_ack})."
                                )

                    # Scenario time (0 until ready)
                    t_run_s = 0.0 if (t_scn0_ns is None) else (t_tick - int(t_scn0_ns)) / 1e9

                    # Build pose command: by default, drive platform to match X-Plane attitude if available.
                    roll_target = float(xp.roll_deg) if (xp and xp.roll_deg is not None) else 0.0
                    pitch_target = float(xp.pitch_deg) if (xp and xp.pitch_deg is not None) else 0.0
                    yaw_target = float(xp.heading_deg) if (xp and xp.heading_deg is not None) else 0.0

                    # If requested, make platform yaw relative to the initial sim heading (so startup heading becomes yaw=0).
                    if bool(args.platform_yaw_relative) and xp and xp.heading_deg is not None:
                        h0 = xp_heading0_wrapped
                        if h0 is None:
                            h0 = wrap180(float(xp.heading_deg))
                            xp_heading0_wrapped = h0
                            logging.info("Platform yaw-relative init: xp_heading0_wrapped=%.2f", h0)
                        yaw_target = wrap180(wrap180(float(xp.heading_deg)) - float(h0))

                    # Startup shaping: optional hold + ramp to avoid initial jerk.
                    hold_s = max(0.0, float(args.pose_start_hold_s))
                    ramp_s = max(0.0, float(args.pose_ramp_s))
                    if t_elapsed_s < hold_s:
                        a = 0.0
                    elif ramp_s <= 0.0:
                        a = 1.0
                    else:
                        a = min(1.0, max(0.0, (t_elapsed_s - hold_s) / ramp_s))

                    roll_cmd = float(roll_target) * a
                    pitch_cmd = float(pitch_target) * a
                    # yaw can be large (heading). Keep ramp in wrapped space to avoid 0..360 weirdness.
                    yaw_cmd = wrap180(float(yaw_target)) * a

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

                    max_queue_depth = max(max_queue_depth, len(st.sent_queue))

                    # If we're sending poses but receiving no ACKs, something is wrong:
                    if not warned_no_ack and tick_count >= int(tick_hz) and ack_total == 0:
                        warned_no_ack = True
                        logging.warning(
                            "No ESP32 ACK received after ~1s of sending pose commands. "
                            "Most likely wrong --stewart-com or ESP32 not emitting status over serial."
                        )

                    # Compute control injection from PX4 attitude (custom PID in host).
                    if px is not None and last_px4_ns is not None:
                        dt_s = (px.t_rx_ns - last_px4_ns) / 1e9
                    else:
                        dt_s = 0.0

                    # Yaw bias/unwrap alignment (for logging + yaw PID)
                    yaw_bias_active = bool(args.yaw_bias)
                    sim_yaw_wrapped = None
                    px4_yaw_wrapped = None
                    px4_yaw_aligned_to_sim = None
                    sim_yaw_aligned_to_px4 = None
                    yaw_rate_sim = None

                    xp_for_yaw = xp_rx.latest
                    if yaw_bias_active and xp_for_yaw is not None and xp_for_yaw.heading_deg is not None and px is not None:
                        sim_y = wrap180(float(xp_for_yaw.heading_deg))
                        if bool(args.yaw_platform_flip):
                            sim_y = wrap180(-sim_y)
                        px_y = wrap180(float(px.yaw_deg))

                        yaw_sim_unw = unwrap_deg(yaw_sim_unw, sim_y)
                        yaw_px4_unw = unwrap_deg(yaw_px4_unw, px_y)
                        sim_yaw_wrapped = sim_y
                        px4_yaw_wrapped = px_y

                        if yaw_bias_deg is None:
                            yaw_bias_deg = float(yaw_px4_unw - yaw_sim_unw)
                            logging.info(
                                "Yaw bias init: sim=%.2f px4=%.2f bias(px4-sim)=%.2f (platform_flip=%s)",
                                yaw_sim_unw,
                                yaw_px4_unw,
                                yaw_bias_deg,
                                bool(args.yaw_platform_flip),
                            )

                        # Optional drift cancellation: adapt bias when sim yaw is effectively still.
                        adapt_rate = float(args.yaw_bias_adapt_rate)
                        if adapt_rate > 0.0:
                            if last_sim_yaw_for_rate is not None:
                                t0, y0 = last_sim_yaw_for_rate
                                dt_rate = max(1e-6, (t_tick - t0) / 1e9)
                                yaw_rate_sim = float((yaw_sim_unw - y0) / dt_rate)
                            last_sim_yaw_for_rate = (t_tick, yaw_sim_unw)

                            if yaw_rate_sim is not None and abs(yaw_rate_sim) < float(args.yaw_still_thresh_deg_s):
                                err = wrap180((yaw_px4_unw - yaw_sim_unw) - float(yaw_bias_deg))
                                dt_tick = float(tick_dt_ns) / 1e9
                                yaw_bias_deg = float(yaw_bias_deg + err * adapt_rate * dt_tick)

                        px4_yaw_aligned_to_sim = wrap180(float(yaw_px4_unw - float(yaw_bias_deg)))
                        sim_yaw_aligned_to_px4 = wrap180(float(yaw_sim_unw + float(yaw_bias_deg)))

                    pid_enabled = (t_elapsed_s >= float(args.pid_start_delay_s))
                    if (not pid_enabled) and (pid_roll is not None or pid_pitch is not None or pid_yaw is not None):
                        # Prevent integral windup / noisy derivative during startup jostle.
                        if pid_roll is not None:
                            pid_roll.reset()
                        if pid_pitch is not None:
                            pid_pitch.reset()
                        if pid_yaw is not None:
                            pid_yaw.reset()

                    # Scenario-driven pitch setpoint (time-varying)
                    sp_pitch_deg = float(args.sp_pitch_deg)
                    scenario_phase = ""
                    scenario_done = False
                    if scenario is not None:
                        if not ready:
                            sp_pitch_deg = 0.0
                            scenario_phase = "waiting_ready"
                        else:
                            sp_pitch_deg, scenario_phase, scenario_done = trim_transition_setpoint(t_run_s, scenario)

                    if pid_enabled and px is not None and pid_roll is not None and pid_pitch is not None and pid_yaw is not None:
                        last_px4_ns = px.t_rx_ns
                        u_ail = pid_roll.update(float(args.sp_roll_deg), px.roll_deg, dt_s)
                        u_ele = pid_pitch.update(float(sp_pitch_deg), px.pitch_deg, dt_s)
                        yaw_meas = px4_yaw_aligned_to_sim if (yaw_bias_active and px4_yaw_aligned_to_sim is not None) else float(px.yaw_deg)
                        u_rud = pid_yaw.update_angle(float(args.sp_yaw_deg), float(yaw_meas), dt_s)
                    else:
                        u_ail = u_ele = u_rud = 0.0

                    if xp_tx is not None and args.xplane_send == "data":
                        xp_tx.send_controls(u_ele, u_ail, u_rud, throttle=float(args.throttle))

                    # Tick log (latest-sample-hold snapshot)
                    row = {
                        "t_tick_ns": t_tick,
                        "tick_idx": tick_count,
                        "t_elapsed_s": t_elapsed_s,
                        "t_run_s": t_run_s,
                        "repeat_idx": rep_idx + 1,
                        "scenario_phase": scenario_phase,
                        "sp_pitch_deg": sp_pitch_deg,
                        "pid_enabled": int(pid_enabled),
                        "xp_t_rx_ns": xp.t_rx_ns if xp else "",
                        "xp_roll_deg": xp.roll_deg if xp else "",
                        "xp_pitch_deg": xp.pitch_deg if xp else "",
                        "xp_heading_deg": xp.heading_deg if xp else "",
                        "px4_t_rx_ns": px.t_rx_ns if px else "",
                        "px4_time_boot_ms": px.time_boot_ms if px else "",
                        "px4_roll_deg": px.roll_deg if px else "",
                        "px4_pitch_deg": px.pitch_deg if px else "",
                        "px4_yaw_deg": px.yaw_deg if px else "",
                        "yaw_bias_deg": yaw_bias_deg if yaw_bias_deg is not None else "",
                        "xp_heading0_wrapped_deg": xp_heading0_wrapped if xp_heading0_wrapped is not None else "",
                        "xp_heading_wrapped_deg": sim_yaw_wrapped if sim_yaw_wrapped is not None else "",
                        "px4_yaw_wrapped_deg": px4_yaw_wrapped if px4_yaw_wrapped is not None else "",
                        "px4_yaw_aligned_to_sim_deg": px4_yaw_aligned_to_sim if px4_yaw_aligned_to_sim is not None else "",
                        "sim_yaw_aligned_to_px4_deg": sim_yaw_aligned_to_px4 if sim_yaw_aligned_to_px4 is not None else "",
                        "sim_yaw_rate_deg_s": yaw_rate_sim if yaw_rate_sim is not None else "",
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

                    if scenario is not None and ready and scenario_done:
                        # Send neutral controls once at end of repeat for safety.
                        if xp_tx is not None and args.xplane_send == "data":
                            xp_tx.send_controls(0.0, 0.0, 0.0, throttle=float(args.throttle))
                        break

            finally:
                # Write a lightweight integrity summary next to the logs.
                integrity = {
                    "repeat_idx": rep_idx + 1,
                    "tick_rows": int(tick_count),
                    "xp_seen_count": int(xp_seen),
                    "px4_seen_count": int(px4_seen),
                    "ack_total": int(ack_total),
                    "max_queue_depth": int(max_queue_depth),
                    "max_tick_gap_ms": float(max_tick_gap_ms),
                    "ready_gate_used": bool(scenario is not None),
                    "ready": bool(ready) if scenario is not None else True,
                    "scenario_total_s": float(scenario.total_s) if scenario is not None else None,
                }
                try:
                    (run_dir / "integrity.json").write_text(json.dumps(integrity, indent=2, sort_keys=True), encoding="utf-8")
                except Exception:
                    pass
                try:
                    log.close()
                except Exception:
                    pass

        return 0

    except KeyboardInterrupt:
        pose = PoseCmd(
                t_send_ns=t_tick,
                x_mm=0.0,
                y_mm=0.0,
                z_mm=0.0,
                roll_deg=0.0,
                pitch_deg=0.0,
                yaw_deg=0.0,
            )
        st.send_pose(pose)
        st.close()
        print("\n[hils_host] stopped.")
        return 0
    finally:
        try:
            xp_rx.close()
            if xp_tx is not None:
                xp_tx.close()
            
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())


