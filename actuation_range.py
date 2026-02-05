import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np

import coordinate_generator as cg


@dataclass(frozen=True)
class ActuationRange:
    dx_mm: Tuple[Optional[float], Optional[float]]
    dy_mm: Tuple[Optional[float], Optional[float]]
    dz_mm: Tuple[Optional[float], Optional[float]]
    roll_deg: Tuple[Optional[float], Optional[float]]
    pitch_deg: Tuple[Optional[float], Optional[float]]
    yaw_deg: Tuple[Optional[float], Optional[float]]

# =========================
# Arduino-accurate IK port
# (from `main/main.ino`)
# =========================

ARDUINO_LINK_LENGTH_MM = 127.0

def _v3(x: float, y: float, z: float) -> np.ndarray:
    return np.array([float(x), float(y), float(z)], dtype=float)

def _v_dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])

def _v_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ], dtype=float)

def _v_norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a))

def _v_normed(a: np.ndarray) -> np.ndarray:
    n = _v_norm(a)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    return a / n

def _rotate_z(p: np.ndarray, deg: float) -> np.ndarray:
    rad = float(deg) * np.pi / 180.0
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return _v3(p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2])

def _rotate_x(p: np.ndarray, deg: float) -> np.ndarray:
    rad = float(deg) * np.pi / 180.0
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return _v3(p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c)

def _rotate_y(p: np.ndarray, deg: float) -> np.ndarray:
    rad = float(deg) * np.pi / 180.0
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return _v3(p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c)

def _mirror_x(p: np.ndarray) -> np.ndarray:
    return _v3(-p[0], p[1], p[2])

def arduino_generate_coordinates() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact port of `generateCoordinates()` constants in `main/main.ino`.
    Returns: bottomAnchor, horn0Anchor, upperAnchor as (6,3) float arrays.
    """
    # const Vec3 B1_pos = {-50.0f, 100.0f, 10.0f};
    # const Vec3 H1     = {-97.5f, 108.0f + 3.3f / 2.0f, 10.0f};
    # const Vec3 U1     = { -7.5f, 108.0f + 3.3f / 2.0f, 99.60469f};
    B1_pos = _v3(-50.0, 100.0, 10.0)
    y_off = 108.0 + 3.3 / 2.0
    H1 = _v3(-97.5, y_off, 10.0)
    U1 = _v3(-7.5, y_off, 99.60469)

    bottom = np.zeros((6, 3), dtype=float)
    horn0 = np.zeros((6, 3), dtype=float)
    upper = np.zeros((6, 3), dtype=float)

    # servos 1,3,5: rotations of servo1
    anglesZ = [0.0, 120.0, 240.0]
    for i, ang in enumerate(anglesZ):
        idx = i * 2  # 0,2,4
        bottom[idx] = _rotate_z(B1_pos, ang)
        horn0[idx] = _rotate_z(H1, ang)
        upper[idx] = _rotate_z(U1, ang)

    # servos 2,4,6: X-mirror of servo1 rotated
    B1m = _mirror_x(B1_pos)
    H1m = _mirror_x(H1)
    U1m = _mirror_x(U1)

    bottom[1] = _rotate_z(B1m, 120.0)
    horn0[1] = _rotate_z(H1m, 120.0)
    upper[1] = _rotate_z(U1m, 120.0)

    bottom[3] = _rotate_z(B1m, 240.0)
    horn0[3] = _rotate_z(H1m, 240.0)
    upper[3] = _rotate_z(U1m, 240.0)

    bottom[5] = B1m
    horn0[5] = H1m
    upper[5] = U1m

    return bottom, horn0, upper

def arduino_get_servo_axis(idx: int) -> np.ndarray:
    base_axis = _v3(0.0, 1.0, 0.0)
    if idx == 0 or idx == 5:
        angle = 0.0
    elif idx == 1 or idx == 2:
        angle = 120.0
    elif idx == 3 or idx == 4:
        angle = -120.0
    else:
        angle = 0.0
    a = _rotate_z(base_axis, angle)
    return _v_normed(a)

def arduino_build_servo_frame(axis: np.ndarray) -> np.ndarray:
    ez = np.asarray(axis, dtype=float)
    ref = _v3(0.0, 0.0, 1.0)
    if abs(_v_dot(ref, ez)) > 0.99:
        ref = _v3(1.0, 0.0, 0.0)
    ex = _v_normed(_v_cross(ref, ez))
    ey = _v_cross(ez, ex)
    # Rows are basis vectors; matches Arduino mat3Mul(R, v) with rows.
    R = np.array([
        [ex[0], ex[1], ex[2]],
        [ey[0], ey[1], ey[2]],
        [ez[0], ez[1], ez[2]],
    ], dtype=float)
    return R

def arduino_solve_servo_angle_z_axis(B: np.ndarray, H0: np.ndarray, Utarget: np.ndarray, L: float) -> Tuple[float, float, bool]:
    # Port of solveServoAngleZAxis()
    h = H0 - B
    s = Utarget - B

    a = float(s[0] * h[0] + s[1] * h[1])
    b = float(-s[0] * h[1] + s[1] * h[0])
    gamma = 0.5 * (float(_v_dot(s, s)) + float(_v_dot(h, h)) - float(L) * float(L))
    gamma_p = float(gamma - s[2] * h[2])

    r = float(np.hypot(a, b))
    if r < 1e-6:
        return 0.0, 0.0, False

    x = float(gamma_p / r)
    if x < -1.0 or x > 1.0:
        return 0.0, 0.0, False

    phi = float(np.arctan2(b, a))
    d = float(np.arccos(x))

    theta1 = (phi + d) * 180.0 / np.pi
    theta2 = (phi - d) * 180.0 / np.pi
    return float(theta1), float(theta2), True

def arduino_pick_solution(theta1: float, theta2: float, min_deg: float, max_deg: float, prefer_deg: float = 0.0) -> Tuple[Optional[float], bool]:
    candidates: List[float] = []
    for t in (theta1, theta2):
        tn = float(np.fmod(t + 180.0, 360.0))
        if tn < 0:
            tn += 360.0
        tn -= 180.0
        if tn >= min_deg and tn <= max_deg:
            candidates.append(tn)
    if not candidates:
        return None, False
    best = candidates[0]
    best_diff = abs(best - prefer_deg)
    for c in candidates[1:]:
        d = abs(c - prefer_deg)
        if d < best_diff:
            best_diff = d
            best = c
    return float(best), True

def arduino_transform_upper_anchors(src: np.ndarray, *, dz: float, roll_deg: float, pitch_deg: float, yaw_deg: float, dx: float, dy: float) -> np.ndarray:
    """
    Port of Arduino transformUpperAnchors().
    Notes:
      - roll: rotation about +Y, but applied as rotateY(..., -rollDeg)
      - pitch: rotation about +X, but applied as rotateX(..., -pitchDeg)
      - yaw: rotation about +Z, applied as rotateZ(..., yawDeg) after wrapping to [-180,180]
    """
    src = np.asarray(src, dtype=float)
    center = src.mean(axis=0)

    yaw_deg = float(yaw_deg)
    if yaw_deg > 180.0:
        yaw_deg = yaw_deg - 360.0

    out = np.zeros_like(src, dtype=float)
    for i in range(6):
        p = src[i] - center
        p = _rotate_x(p, -pitch_deg)
        p = _rotate_y(p, -roll_deg)
        p = _rotate_z(p, yaw_deg)
        p = p + center
        p = _v3(p[0] + dx, p[1] + dy, p[2] + dz)
        out[i] = p
    return out

def arduino_compute_all_servo_angles(
    bottom: np.ndarray,
    horn0: np.ndarray,
    upper_t: np.ndarray,
    L: float,
    *,
    min_deg: float,
    max_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Port of computeAllServoAngles().
    Returns:
      angles: (6,) float (undefined where failed)
      success: (6,) bool
    """
    angles = np.zeros(6, dtype=float)
    success = np.zeros(6, dtype=bool)
    for i in range(6):
        axis = arduino_get_servo_axis(i)
        R = arduino_build_servo_frame(axis)  # global -> local

        B = bottom[i]
        H0 = horn0[i]
        U = upper_t[i]

        h_local = R @ (H0 - B)
        u_local = R @ (U - B)
        origin = np.zeros(3, dtype=float)

        t1, t2, ok = arduino_solve_servo_angle_z_axis(origin, h_local, u_local, L)
        if not ok:
            success[i] = False
            angles[i] = 0.0
            continue

        th, ok2 = arduino_pick_solution(t1, t2, min_deg, max_deg, prefer_deg=0.0)
        if not ok2 or th is None:
            success[i] = False
            angles[i] = 0.0
            continue

        success[i] = True
        angles[i] = float(th)
    return angles, success

def _rotvec_to_rotmat(rotvec_rad: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation vector (rad) -> rotation matrix."""
    r = np.asarray(rotvec_rad, dtype=float).reshape(3)
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    kx, ky, kz = float(k[0]), float(k[1]), float(k[2])
    K = np.array([[0.0, -kz,  ky],
                  [kz,  0.0, -kx],
                  [-ky, kx,  0.0]], dtype=float)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)

def _transform_upper_anchors_rotmat(src: np.ndarray, R: np.ndarray, *, dx: float, dy: float, dz: float) -> np.ndarray:
    """Rotate about plate center with rotation matrix R, then translate."""
    src = np.asarray(src, dtype=float)
    center = src.mean(axis=0)
    centered = src - center
    # points are rows, so apply as (R @ p) => centered @ R.T
    out = centered @ R.T
    out = out + center
    out[:, 0] += float(dx)
    out[:, 1] += float(dy)
    out[:, 2] += float(dz)
    return out

def rotation_boundary_for_z_sweep(
    *,
    z_vals_mm: np.ndarray,
    n_dirs: int = 400,
    max_search_deg: float = 60.0,
    tol_deg: float = 0.25,
    servo_min_deg: float = -90.0,
    servo_max_deg: float = 90.0,
    rng_seed: int = 0,
) -> dict:
    """
    Step-3 engine: for each z in z_vals_mm, sample rotation directions on S^2 and
    binary-search max reachable rotation magnitude along that direction.

    Returns a dict with:
      - z_vals_mm: (Nz,)
      - dirs: (Ndirs,3) unit vectors
      - max_deg: (Nz, Ndirs) per-dir max magnitude (deg) in rotation-vector space
      - worst_deg: (Nz,) min over directions (conservative inscribed sphere radius)
    """
    bottom, horn0, upper0 = arduino_generate_coordinates()
    L = float(ARDUINO_LINK_LENGTH_MM)

    rng = np.random.default_rng(int(rng_seed))
    dirs = rng.normal(size=(int(n_dirs), 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    z_vals_mm = np.asarray(z_vals_mm, dtype=float).reshape(-1)
    max_deg = np.zeros((z_vals_mm.shape[0], dirs.shape[0]), dtype=float)

    def reachable(z_mm: float, rotvec_deg: np.ndarray) -> bool:
        R = _rotvec_to_rotmat(rotvec_deg * (np.pi / 180.0))
        upper_t = _transform_upper_anchors_rotmat(upper0, R, dx=0.0, dy=0.0, dz=float(z_mm))
        _, ok = arduino_compute_all_servo_angles(
            bottom, horn0, upper_t, L,
            min_deg=float(servo_min_deg), max_deg=float(servo_max_deg),
        )
        return bool(ok.all())

    for zi, z_mm in enumerate(z_vals_mm):
        for di, u in enumerate(dirs):
            lo, hi = 0.0, float(max_search_deg)
            # If hi is reachable, we keep it as bounded by max_search_deg.
            # Binary search between lo (reachable) and hi (unknown/unreachable).
            # We assume 0 deg is reachable; if not, boundary is 0.
            if not reachable(z_mm, u * 0.0):
                max_deg[zi, di] = 0.0
                continue

            # If even the top is reachable, boundary is at least hi.
            if reachable(z_mm, u * hi):
                max_deg[zi, di] = hi
                continue

            while (hi - lo) > float(tol_deg):
                mid = 0.5 * (lo + hi)
                if reachable(z_mm, u * mid):
                    lo = mid
                else:
                    hi = mid
            max_deg[zi, di] = lo

    worst_deg = max_deg.min(axis=1)
    return {
        "z_vals_mm": z_vals_mm,
        "dirs": dirs,
        "max_deg": max_deg,
        "worst_deg": worst_deg,
        "meta": {
            "n_dirs": int(n_dirs),
            "max_search_deg": float(max_search_deg),
            "tol_deg": float(tol_deg),
            "servo_min_deg": float(servo_min_deg),
            "servo_max_deg": float(servo_max_deg),
            "link_length_mm": float(L),
        },
    }


def load_rotation_boundary_npz(npz_path: str = "rot_boundary_z_sweep.npz") -> Dict[str, np.ndarray]:
    """
    Load `.npz` produced by `rotation_boundary_for_z_sweep()`.
    Returns a plain dict with numpy arrays.
    """
    data = dict(np.load(npz_path, allow_pickle=True))
    # Ensure core keys exist
    for k in ("z_vals_mm", "dirs", "max_deg", "worst_deg"):
        if k not in data:
            raise KeyError(f"{npz_path}: missing key '{k}'")
    return data


def inscribed_rotation_sphere_deg_at_z(result: Dict[str, np.ndarray], z_mm: float) -> Tuple[float, float]:
    """
    Step 4 helper: return conservative max rotation magnitude (deg) at the nearest sampled z,
    using the 'inscribed sphere' (min over direction samples).

    Returns: (z_used_mm, max_mag_deg)
    """
    z_vals = np.asarray(result["z_vals_mm"], dtype=float).reshape(-1)
    worst = np.asarray(result["worst_deg"], dtype=float).reshape(-1)
    if z_vals.shape[0] != worst.shape[0]:
        raise ValueError("result: z_vals_mm and worst_deg length mismatch")
    i = int(np.argmin(np.abs(z_vals - float(z_mm))))
    return float(z_vals[i]), float(worst[i])


def inscribed_rotation_sphere_deg_worst_over_z(result: Dict[str, np.ndarray], z_min_mm: float, z_max_mm: float) -> float:
    """
    Step 4 helper: a single ultra-conservative rotation magnitude bound (deg) that should hold
    for all sampled z within [z_min_mm, z_max_mm], by taking the minimum inscribed-sphere radius.
    """
    z_vals = np.asarray(result["z_vals_mm"], dtype=float).reshape(-1)
    worst = np.asarray(result["worst_deg"], dtype=float).reshape(-1)
    z0, z1 = float(z_min_mm), float(z_max_mm)
    lo, hi = (z0, z1) if z0 <= z1 else (z1, z0)
    mask = (z_vals >= lo) & (z_vals <= hi)
    if not mask.any():
        raise ValueError("No z samples in the requested interval")
    return float(worst[mask].min())


def _rotmat_to_rotvec(R: np.ndarray) -> np.ndarray:
    """
    Log map SO(3): rotation matrix -> rotation vector (rad), principal angle in [0, pi].
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    tr = float(np.trace(R))
    c = (tr - 1.0) * 0.5
    c = float(np.clip(c, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-12:
        return np.zeros(3, dtype=float)

    s = float(np.sin(th))
    if abs(s) > 1e-8:
        w_hat = (R - R.T) / (2.0 * s)
        k = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]], dtype=float)
        return k * th

    # Near pi: use diagonal-based extraction (good enough for saturation use).
    A = (R + np.eye(3)) * 0.5
    k = np.array([np.sqrt(max(float(A[0, 0]), 0.0)),
                  np.sqrt(max(float(A[1, 1]), 0.0)),
                  np.sqrt(max(float(A[2, 2]), 0.0))], dtype=float)
    if float(R[2, 1] - R[1, 2]) < 0.0:
        k[0] = -k[0]
    if float(R[0, 2] - R[2, 0]) < 0.0:
        k[1] = -k[1]
    if float(R[1, 0] - R[0, 1]) < 0.0:
        k[2] = -k[2]
    n = float(np.linalg.norm(k))
    if n < 1e-9:
        return np.zeros(3, dtype=float)
    return (k / n) * th


def arduino_rpy_to_rotmat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """
    Build rotation matrix equivalent to Arduino's:
      p' = Rz(yaw) * Ry(-roll) * Rx(-pitch) * p
    """
    # Use the same elemental rotations as `_rotate_x/_rotate_y/_rotate_z`, but as matrices.
    d2r = np.pi / 180.0
    a = -float(pitch_deg) * d2r  # Rx
    b = -float(roll_deg) * d2r   # Ry
    c = float(yaw_deg) * d2r     # Rz

    ca, sa = float(np.cos(a)), float(np.sin(a))
    cb, sb = float(np.cos(b)), float(np.sin(b))
    cc, sc = float(np.cos(c)), float(np.sin(c))

    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, ca, -sa],
                   [0.0, sa,  ca]], dtype=float)
    Ry = np.array([[ cb, 0.0, sb],
                   [0.0, 1.0, 0.0],
                   [-sb, 0.0, cb]], dtype=float)
    Rz = np.array([[cc, -sc, 0.0],
                   [sc,  cc, 0.0],
                   [0.0, 0.0, 1.0]], dtype=float)
    return Rz @ Ry @ Rx


def rotmat_to_arduino_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Inverse of `arduino_rpy_to_rotmat()` for:
      R = Rz(yaw) * Ry(-roll) * Rx(-pitch)
    Returns (roll_deg, pitch_deg, yaw_deg).
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)

    # Standard ZYX extraction for R = Rz(gamma) * Ry(beta) * Rx(alpha)
    # Here: alpha = -pitch, beta = -roll, gamma = yaw
    alpha = float(np.arctan2(R[2, 1], R[2, 2]))
    beta = float(np.arcsin(np.clip(-float(R[2, 0]), -1.0, 1.0)))
    gamma = float(np.arctan2(R[1, 0], R[0, 0]))

    r2d = 180.0 / np.pi
    pitch_deg = -alpha * r2d
    roll_deg = -beta * r2d
    yaw_deg = gamma * r2d

    # Match Arduino yaw normalization used in transformUpperAnchors()
    if yaw_deg > 180.0:
        yaw_deg -= 360.0
    if yaw_deg <= -180.0:
        yaw_deg += 360.0

    return float(roll_deg), float(pitch_deg), float(yaw_deg)


def saturate_rotation_magnitude_arduino_rpy(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    *,
    max_mag_deg: float,
) -> Tuple[float, float, float]:
    """
    Saturate the overall rotation magnitude by scaling in rotation-vector space,
    while accepting/returning Arduino-style (roll,pitch,yaw) (deg).

    This preserves the rotation axis (in SO(3)), unlike clamping each Euler axis.
    """
    R = arduino_rpy_to_rotmat(roll_deg, pitch_deg, yaw_deg)
    r = _rotmat_to_rotvec(R)  # rad
    mag = float(np.linalg.norm(r))
    max_mag = float(max_mag_deg) * np.pi / 180.0
    if mag <= max_mag or mag < 1e-12:
        return float(roll_deg), float(pitch_deg), float(yaw_deg)

    r_sat = r * (max_mag / mag)
    R_sat = _rotvec_to_rotmat(r_sat)
    return rotmat_to_arduino_rpy(R_sat)


def _default_horn_len_mm(bottom: np.ndarray, horn_0deg: np.ndarray) -> float:
    """
    Infer the current horn *perpendicular* length (the rotating radius) from servo 1 (index 0).
    This matches the "hornLength" concept in `ref/coordinates.md`.
    """
    i = 0
    axis = cg.get_servo_axis(i)
    R = cg.build_servo_frame(axis)  # global -> local
    h_local = R @ (horn_0deg[i] - bottom[i])
    return float(np.hypot(h_local[0], h_local[1]))


def rebuild_horns_with_horn_len(
    bottom: np.ndarray,
    horn_0deg: np.ndarray,
    horn_len_mm: float,
) -> np.ndarray:
    """
    Return a new (6,3) horn_0deg array where the horn radius (perpendicular component)
    is set to horn_len_mm, while preserving each servo's axis-parallel offset.
    """
    if horn_len_mm <= 0:
        raise ValueError("horn_len_mm must be > 0")

    horn_new = np.zeros_like(horn_0deg, dtype=float)
    for i in range(6):
        axis = cg.get_servo_axis(i)
        R = cg.build_servo_frame(axis)  # global -> local

        h_local = R @ (horn_0deg[i] - bottom[i])  # in servo-local coords
        r = float(np.hypot(h_local[0], h_local[1]))
        if r < 1e-9:
            raise ValueError(f"Servo {i+1}: current horn radius is ~0; cannot scale.")

        scale = horn_len_mm / r
        h_local_scaled = np.array([h_local[0] * scale, h_local[1] * scale, h_local[2]], dtype=float)

        horn_new[i] = bottom[i] + (R.T @ h_local_scaled)

    return horn_new


def _pose_reachable(
    bottom: np.ndarray,
    horn_0deg: np.ndarray,
    upper_original: np.ndarray,
    linkage_len_mm: float,
    *,
    dx_mm: float = 0.0,
    dy_mm: float = 0.0,
    dz_mm: float = 0.0,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    servo_min_deg: float = -90.0,
    servo_max_deg: float = 90.0,
) -> bool:
    upper_t = cg.transform_upper_anchors(
        upper_original,
        dz=dz_mm,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        dx=dx_mm,
        dy=dy_mm,
    )
    _, _, success = cg.compute_all_servo_angles(
        bottom,
        horn_0deg,
        upper_t,
        linkage_len_mm,
        min_deg=servo_min_deg,
        max_deg=servo_max_deg,
    )
    return bool(all(success))


def _sweep_1d(
    check_fn,
    values: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    ok = np.array([bool(check_fn(v)) for v in values], dtype=bool)
    if not ok.any():
        return None, None
    return float(values[ok].min()), float(values[ok].max())

def auto_home_upper_z(upper_original: np.ndarray, horn_0deg: np.ndarray, linkage_len_mm: float) -> np.ndarray:
    """
    Shift the entire upper plate in Z so that at servo=0° the linkage length is exactly linkage_len_mm.
    Keeps the plate geometry (XY layout) unchanged; only adds a constant Z offset to all upper anchors.
    """
    i = 0  # use leg 0; symmetry should make all legs consistent
    dx = float(upper_original[i, 0] - horn_0deg[i, 0])
    dy = float(upper_original[i, 1] - horn_0deg[i, 1])
    horiz2 = dx * dx + dy * dy
    L2 = float(linkage_len_mm * linkage_len_mm)

    if L2 < horiz2:
        raise ValueError(f"Impossible home pose: L^2 ({L2:.3f}) < horizontal^2 ({horiz2:.3f}). "
                         f"Increase linkage_len or reduce horn/plate radius.")

    dz_needed = float(horn_0deg[i, 2] + np.sqrt(L2 - horiz2) - upper_original[i, 2])
    upper_homed = upper_original.copy()
    upper_homed[:, 2] += dz_needed
    return upper_homed

def compute_actuation_range(
    *,
    linkage_len_mm: float = cg.LINK_LENGTH,
    horn_len_mm: Optional[float] = None,
    servo_min_deg: float = -90.0,
    servo_max_deg: float = 90.0,
    # Sweep resolution
    step_mm: float = 1.0,
    step_deg: float = 0.25,
    # Sweep extents (these are just search bounds; reachable region may be smaller)
    dx_search_mm: float = 120.0,
    dy_search_mm: float = 120.0,
    dz_search_mm: float = 120.0,
    rot_search_deg: float = 60.0,
) -> ActuationRange:
    bottom, horn_0deg, upper_original = cg.generate_coordinates()

    if horn_len_mm is None:
        horn_len_mm = _default_horn_len_mm(bottom, horn_0deg)
    print(cg.LINK_LENGTH)

    horn_0deg = rebuild_horns_with_horn_len(bottom, horn_0deg, horn_len_mm)

    upper_original = auto_home_upper_z(upper_original, horn_0deg, linkage_len_mm)

    dx_vals = np.arange(-dx_search_mm, dx_search_mm + 0.5 * step_mm, step_mm, dtype=float)
    dy_vals = np.arange(-dy_search_mm, dy_search_mm + 0.5 * step_mm, step_mm, dtype=float)
    dz_vals = np.arange(-dz_search_mm, dz_search_mm + 0.5 * step_mm, step_mm, dtype=float)

    rot_vals = np.arange(-rot_search_deg, rot_search_deg + 0.5 * step_deg, step_deg, dtype=float)

    dx_minmax = _sweep_1d(
        lambda v: _pose_reachable(
            bottom,
            horn_0deg,
            upper_original,
            linkage_len_mm,
            dx_mm=float(v),
            servo_min_deg=servo_min_deg,
            servo_max_deg=servo_max_deg,
        ),
        dx_vals,
    )
    dy_minmax = _sweep_1d(
        lambda v: _pose_reachable(
            bottom,
            horn_0deg,
            upper_original,
            linkage_len_mm,
            dy_mm=float(v),
            servo_min_deg=servo_min_deg,
            servo_max_deg=servo_max_deg,
        ),
        dy_vals,
    )
    dz_minmax = _sweep_1d(
        lambda v: _pose_reachable(
            bottom,
            horn_0deg,
            upper_original,
            linkage_len_mm,
            dz_mm=float(v),
            servo_min_deg=servo_min_deg,
            servo_max_deg=servo_max_deg,
        ),
        dz_vals,
    )

    roll_minmax = _sweep_1d(
        lambda v: _pose_reachable(
            bottom,
            horn_0deg,
            upper_original,
            linkage_len_mm,
            roll_deg=float(v),
            servo_min_deg=servo_min_deg,
            servo_max_deg=servo_max_deg,
        ),
        rot_vals,
    )
    pitch_minmax = _sweep_1d(
        lambda v: _pose_reachable(
            bottom,
            horn_0deg,
            upper_original,
            linkage_len_mm,
            pitch_deg=float(v),
            servo_min_deg=servo_min_deg,
            servo_max_deg=servo_max_deg,
        ),
        rot_vals,
    )
    yaw_minmax = _sweep_1d(
        lambda v: _pose_reachable(
            bottom,
            horn_0deg,
            upper_original,
            linkage_len_mm,
            yaw_deg=float(v),
            servo_min_deg=servo_min_deg,
            servo_max_deg=servo_max_deg,
        ),
        rot_vals,
    )

    return ActuationRange(
        dx_mm=dx_minmax,
        dy_mm=dy_minmax,
        dz_mm=dz_minmax,
        roll_deg=roll_minmax,
        pitch_deg=pitch_minmax,
        yaw_deg=yaw_minmax,
    )


def _fmt_minmax(unit: str, mnmx: Tuple[Optional[float], Optional[float]]) -> str:
    mn, mx = mnmx
    if mn is None or mx is None:
        return f"(no solution in search bounds)"
    return f"{mn:+.2f}{unit} to {mx:+.2f}{unit}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute approximate Stewart platform actuation ranges via 1-DOF IK sweeps."
    )
    parser.add_argument("--linkage-len", type=float, default=cg.LINK_LENGTH, help="Linkage length (mm)")
    parser.add_argument(
        "--horn-len",
        type=float,
        default=None,
        help="Servo horn radius (mm). Defaults to current geometry.",
    )
    parser.add_argument("--servo-min", type=float, default=-90.0, help="Servo minimum angle (deg)")
    parser.add_argument("--servo-max", type=float, default=90.0, help="Servo maximum angle (deg)")
    parser.add_argument("--step-mm", type=float, default=1.0, help="Sweep step for translations (mm)")
    parser.add_argument("--step-deg", type=float, default=0.25, help="Sweep step for rotations (deg)")
    parser.add_argument("--dx-search", type=float, default=120.0, help="Search half-range for dx (mm)")
    parser.add_argument("--dy-search", type=float, default=120.0, help="Search half-range for dy (mm)")
    parser.add_argument("--dz-search", type=float, default=120.0, help="Search half-range for dz (mm)")
    parser.add_argument("--rot-search", type=float, default=60.0, help="Search half-range for rotations (deg)")
    args = parser.parse_args()

    bottom, horn_0deg, _ = cg.generate_coordinates()
    inferred = _default_horn_len_mm(bottom, horn_0deg)

    r = compute_actuation_range(
        linkage_len_mm=float(args.linkage_len),
        horn_len_mm=(None if args.horn_len is None else float(args.horn_len)),
        servo_min_deg=float(args.servo_min),
        servo_max_deg=float(args.servo_max),
        step_mm=float(args.step_mm),
        step_deg=float(args.step_deg),
        dx_search_mm=float(args.dx_search),
        dy_search_mm=float(args.dy_search),
        dz_search_mm=float(args.dz_search),
        rot_search_deg=float(args.rot_search),
    )

    horn_len_used = inferred if args.horn_len is None else float(args.horn_len)
    print("=== Stewart Platform Actuation Range (1-DOF sweeps) ===")
    print(f"Linkage length:  {args.linkage_len:.5f} mm")
    print(f"Horn length:     {horn_len_used:.5f} mm (inferred default: {inferred:.5f} mm)")
    print(f"Servo limits:    [{args.servo_min:.1f}°, {args.servo_max:.1f}°]")
    print(f"Sweep steps:     {args.step_mm:.3f} mm, {args.step_deg:.3f} deg")
    print()
    print(f"dx:    {_fmt_minmax('mm', r.dx_mm)}")
    print(f"dy:    {_fmt_minmax('mm', r.dy_mm)}")
    print(f"dz:    {_fmt_minmax('mm', r.dz_mm)}")
    print(f"roll:  {_fmt_minmax('°', r.roll_deg)}")
    print(f"pitch: {_fmt_minmax('°', r.pitch_deg)}")
    print(f"yaw:   {_fmt_minmax('°', r.yaw_deg)}")
    print()
    print("Note: These are per-axis ranges with other DOFs held at 0; combined motion has a smaller feasible region.")


if __name__ == "__main__":
    main()











