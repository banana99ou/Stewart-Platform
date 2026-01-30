import argparse
import logging
import os
import socket
import struct
import sys
import time
import numpy as np


DEFAULT_COM = r"\\.\COM13"   # Windows COM10+ needs \\.\COMxx
DEFAULT_BAUD = 115200
DEFAULT_ATTITUDE_HZ = 50.0
DEFAULT_UDP_HOST = "127.0.0.1"
DEFAULT_UDP_PORT = 32901

# Simulink UDP packet: header "PID1" + 3 float32
#   [u_roll, u_pitch, u_yaw]
PID1_HEADER = b"PID1"
PID1_STRUCT = struct.Struct("<4s3f")

# PID tuning (edit here)
ROLL_SETPOINT_RAD = 0.0
PITCH_SETPOINT_RAD = 0.0
YAW_SETPOINT_RAD = 0.0

ROLL_KP, ROLL_KI, ROLL_KD = 2.0, 0.0, 0.0
PITCH_KP, PITCH_KI, PITCH_KD = 2.0, 0.0, 0.0
YAW_KP, YAW_KI, YAW_KD = 0.0, 0.0, 10.0

INTEGRATOR_LIMIT = 10.0
OUTPUT_LIMIT = 10.0


def request_message_interval(master, mavutil, msg_id: int, hz: float) -> None:
    interval_us = int(1e6 / hz) if hz > 0 else 0
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        msg_id,
        interval_us,
        0, 0, 0, 0, 0,
    )


class PID:
    def __init__(self, kp: float, ki: float, kd: float, i_limit: float, out_limit: float):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.i_limit = float(i_limit)
        self.out_limit = float(out_limit)
        self.i = 0.0
        self.prev_err = 0.0
        self.has_prev = False

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        if dt <= 0 or not (dt < 10.0):
            dt = 0.0

        err = float(setpoint) - float(measurement)

        # Integral (clamped)
        self.i += err * dt
        if self.i > self.i_limit:
            self.i = self.i_limit
        elif self.i < -self.i_limit:
            self.i = -self.i_limit

        # Derivative on error (finite-diff)
        d = 0.0
        if self.has_prev and dt > 0:
            d = (err - self.prev_err) / dt
        self.prev_err = err
        self.has_prev = True

        u = self.kp * err + self.ki * self.i + self.kd * d
        if u > self.out_limit:
            u = self.out_limit
        elif u < -self.out_limit:
            u = -self.out_limit
        return float(u)


def parse_args():
    p = argparse.ArgumentParser(description="Pix32/Pixhawk serial ATTITUDE -> PID -> UDP (Simulink).")
    p.add_argument("--com", default="COM13", help=r"Serial port, e.g. COM13 (Windows) or /dev/ttyACM0 (Linux)")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud (often ignored on USB CDC)")
    p.add_argument("--attitude-hz", type=float, default=DEFAULT_ATTITUDE_HZ, help="Requested ATTITUDE rate (Hz)")
    p.add_argument("--udp-host", default=DEFAULT_UDP_HOST, help="UDP destination host (Simulink)")
    p.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT, help="UDP destination port")
    p.add_argument("--dialect", default="common", help="MAVLink dialect (PX4: common; ArduPilot: ardupilotmega)")
    p.add_argument("--info-hz", type=float, default=1.0, help="INFO status print rate (Hz). Use 0 to disable.")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return p.parse_args()


def normalize_serial_port(port: str) -> str:
    """
    Windows quirk:
      - COM1..COM9 often work as-is.
      - COM10+ should be opened as '\\\\.\\COM10' (Win32 device namespace).
    Accepts user-friendly inputs like 'com13' and normalizes.
    """
    p = (port or "").strip()
    if not p:
        return DEFAULT_COM

    if sys.platform.startswith("win"):
        up = p.upper()
        # If user already provided a device path like \\.\COM13, keep it.
        if up.startswith(r"\\.\COM") or up.startswith(r"\\\\.\\COM"):
            return p
        if up.startswith("COM"):
            return r"\\.\{}".format(up)
    return p


def setup_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger("pix32_pid_to_simulink_udp")


def main():
    args = parse_args()
    log = setup_logging(args.log_level)

    os.environ.setdefault("MAVLINK20", "1")
    os.environ["MAVLINK_DIALECT"] = args.dialect
    from pymavlink import mavutil

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_target = (args.udp_host, args.udp_port)
    log.info("UDP target=%s:%s packet=PID1(<4s3f) bytes=%d", args.udp_host, args.udp_port, PID1_STRUCT.size)

    pid_roll = PID(ROLL_KP, ROLL_KI, ROLL_KD, INTEGRATOR_LIMIT, OUTPUT_LIMIT)
    pid_pitch = PID(PITCH_KP, PITCH_KI, PITCH_KD, INTEGRATOR_LIMIT, OUTPUT_LIMIT)
    pid_yaw = PID(YAW_KP, YAW_KI, YAW_KD, INTEGRATOR_LIMIT, OUTPUT_LIMIT)

    port = normalize_serial_port(args.com)
    log.info("Opening MAVLink serial port=%r baud=%d dialect=%s", port, int(args.baud), args.dialect)
    master = mavutil.mavlink_connection(port, baud=args.baud, autoreconnect=False, robust_parsing=True)
    hb = master.wait_heartbeat(timeout=10.0)
    if hb is None:
        log.warning("Heartbeat not received within 10s (port=%r).", port)
    else:
        log.info("Heartbeat received (system=%s component=%s).", master.target_system, master.target_component)

    request_message_interval(master, mavutil, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, args.attitude_hz)
    log.info("Requested ATTITUDE stream at %.1f Hz.", float(args.attitude_hz))

    last_t = time.monotonic()
    last_rx = last_t
    last_info = last_t
    rx_count = 0
    rx_count_since = last_t

    while True:
        try:
            msg = master.recv_match(type="ATTITUDE", blocking=True, timeout=1.0)
            now_t = time.monotonic()

            if msg is None:
                if (now_t - last_rx) >= 2.0:
                    log.warning("No ATTITUDE received for %.1f s", now_t - last_rx)
                continue

            dt = now_t - last_t
            last_t = now_t
            last_rx = now_t
            rx_count += 1

            roll = float(np.rad2deg(msg.roll))
            pitch = float(np.rad2deg(msg.pitch))
            yaw = float(np.rad2deg(msg.yaw))

            u_roll = pid_roll.update(ROLL_SETPOINT_RAD, roll, dt)
            u_pitch = pid_pitch.update(PITCH_SETPOINT_RAD, pitch, dt)
            u_yaw = pid_yaw.update(YAW_SETPOINT_RAD, yaw, dt)

            pkt = PID1_STRUCT.pack(PID1_HEADER, u_roll, u_pitch, u_yaw)
            sock.sendto(pkt, udp_target)

            if args.info_hz > 0 and (now_t - last_info) >= 1.0 / float(args.info_hz):
                dt_stats = now_t - rx_count_since
                rx_hz = (rx_count / dt_stats) if dt_stats > 0 else 0.0
                log.info(
                    "rx=%.1f Hz ATT rpy(rad)=(%.3f,%.3f,%.3f) u=(%.3f,%.3f,%.3f)",
                    rx_hz,
                    roll,
                    pitch,
                    yaw,
                    u_roll,
                    u_pitch,
                    u_yaw,
                )
                last_info = now_t
                rx_count = 0
                rx_count_since = now_t

            log.debug(
                "ATT rpy(rad)=(%.3f,%.3f,%.3f) dt=%.4f -> u=(%.3f,%.3f,%.3f)",
                roll,
                pitch,
                yaw,
                dt,
                u_roll,
                u_pitch,
                u_yaw,
            )
        except KeyboardInterrupt:
            log.info("Stopped (Ctrl+C).")
            return


if __name__ == "__main__":
    main()


