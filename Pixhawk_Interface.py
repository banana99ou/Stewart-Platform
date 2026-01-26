import argparse
import socket
import struct
import time

from pymavlink import mavutil

# Default values (override via CLI args)
DEFAULT_PORT = r"\\.\COM13"  # For Windows COM10+, use the \\.\COM13 form
DEFAULT_BAUD = 115200  # ignored by many USB CDC links, but required by pymavlink API
DEFAULT_IMU_HZ = 50
DEFAULT_UDP_HOST = "127.0.0.1"
DEFAULT_UDP_PORT = 32901

HIGHRES_IMU_MSG_ID = mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU

# UDP packet layout (fixed size, little-endian):
#   0..3   : ASCII header "IMU1"
#   4..31  : 7 x float32 => [t_sec, ax, ay, az, gx, gy, gz]
PACKET_HEADER = b"IMU1"
PACKET_STRUCT = struct.Struct("<4s7f")
PACKET_SIZE = PACKET_STRUCT.size  # 32 bytes

def request_message_interval(master, msg_id: int, hz: float):
    interval_us = int(1e6 / hz)
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        msg_id,
        interval_us,
        0, 0, 0, 0, 0
    )

def parse_args():
    p = argparse.ArgumentParser(description="Read Pixhawk IMU over MAVLink (USB/COM) and forward to Simulink via UDP.")
    p.add_argument("--com", default=DEFAULT_PORT, help=r"Serial port, e.g. \\.\COM13")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud (often ignored on USB CDC)")
    p.add_argument("--imu-hz", type=float, default=DEFAULT_IMU_HZ, help="Requested IMU stream rate (Hz)")
    p.add_argument("--udp-host", default=DEFAULT_UDP_HOST, help="UDP destination host (Simulink machine)")
    p.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT, help="UDP destination port")
    p.add_argument("--print-hz", type=float, default=DEFAULT_IMU_HZ, help="Console print rate (Hz)")
    return p.parse_args()


def main():
    args = parse_args()

    # UDP sender (non-blocking; sendto() on UDP does not require connect()).
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_target = (args.udp_host, args.udp_port)

    master = mavutil.mavlink_connection(args.com, baud=args.baud)

    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print(f"Heartbeat from system={master.target_system} component={master.target_component}")

    # Ask autopilot to stream HIGHRES_IMU
    request_message_interval(master, HIGHRES_IMU_MSG_ID, args.imu_hz)

    print(f"Reading IMU and sending UDP to {udp_target} (packet={PACKET_SIZE} bytes)... Ctrl+C to stop")
    last_print = 0.0

    while True:
        msg = master.recv_match(type=["HIGHRES_IMU", "RAW_IMU"], blocking=True, timeout=2)
        if msg is None:
            continue

        now = time.time()
        if msg.get_type() == "HIGHRES_IMU":
            # Units: accel m/s^2, gyro rad/s
            ax, ay, az = msg.xacc, msg.yacc, msg.zacc
            gx, gy, gz = msg.xgyro, msg.ygyro, msg.zgyro
        else:  # RAW_IMU
            # Units: accel milli-g, gyro milli-rad/s (typically). Convert to SI-ish:
            ax, ay, az = [v * 9.80665 / 1000.0 for v in (msg.xacc, msg.yacc, msg.zacc)]  # m/s^2
            gx, gy, gz = [v / 1000.0 for v in (msg.xgyro, msg.ygyro, msg.zgyro)]         # rad/s

        # Send fixed-size packet for easy Simulink unpacking.
        # Use float32 timestamp (seconds since epoch) for convenience.
        pkt = PACKET_STRUCT.pack(PACKET_HEADER, float(now), float(ax), float(ay), float(az), float(gx), float(gy), float(gz))
        sock.sendto(pkt, udp_target)

        # Print at ~print_hz (avoid flooding if multiple msgs arrive)
        if args.print_hz > 0 and (now - last_print) >= 1.0 / args.print_hz:
            last_print = now
            print(f"accel [m/s^2]=({ax: .3f},{ay: .3f},{az: .3f})  gyro [rad/s]=({gx: .3f},{gy: .3f},{gz: .3f})")

if __name__ == "__main__":
    main()