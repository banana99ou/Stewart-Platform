from msilib import datasizemask
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

DEG2RAD = np.pi / 180.0

# Geometry constants
LINK_LENGTH = 162.21128  # mm (ball link center to center)

def rotate_z(p, deg):
  """Rotate point about Z axis (Yaw)"""
  rad = deg * DEG2RAD
  c = np.cos(rad)
  s = np.sin(rad)
  x, y, z = p
  return np.array([x * c - y * s, x * s + y * c, z])

def rotate_x(p, deg):
  """Rotate point about X axis (Roll)"""
  rad = deg * DEG2RAD
  c = np.cos(rad)
  s = np.sin(rad)
  x, y, z = p
  return np.array([x, y * c - z * s, y * s + z * c])

def rotate_y(p, deg):
  """Rotate point about Y axis (Pitch)"""
  rad = deg * DEG2RAD
  c = np.cos(rad)
  s = np.sin(rad)
  x, y, z = p
  return np.array([x * c + z * s, y, -x * s + z * c])

def translate(p, dx, dy, dz):
  """Translate point by (dx, dy, dz)"""
  return p + np.array([dx, dy, dz])

def mirror_x(p):
  """Mirror across YZ plane (negate X coordinate)"""
  x, y, z = p
  return np.array([-x, y, z])

def get_servo_axis(index):
  """
  Return the servo rotation axis (unit vector in global coordinates) for leg index 0..5.
  Hardware: axes lie in the XY plane and point radially outward.
  - Servos 1 & 6 (indices 0 & 5): axis along +Y (0, 1, 0)
  - Servos 2 & 3 (indices 1 & 2): axis is +120° yaw from +Y
  - Servos 4 & 5 (indices 3 & 4): axis is -120° yaw from +Y
  """
  base_axis = np.array([0.0, 1.0, 0.0])
  if index in (0, 5):
    angle = 0.0
  elif index in (1, 2):
    angle = 120.0
  elif index in (3, 4):
    angle = -120.0
  else:
    raise ValueError(f"Invalid servo index: {index}")
  
  axis = rotate_z(base_axis, angle)
  norm = np.linalg.norm(axis)
  if norm < 1e-9:
    # Fallback to global Z to avoid division by zero, though this shouldn't happen
    return np.array([0.0, 0.0, 1.0])
  return axis / norm

def build_servo_frame(axis):
  """
  Build an orthonormal basis (rotation matrix) for a servo's local frame.
  Local frame is defined so that:
    - local Z axis aligns with the given 'axis' (servo rotation axis)
    - local X,Y span the plane perpendicular to axis
  Returns a 3x3 matrix R such that:
    v_local = R @ v_global
    v_global = R.T @ v_local
  """
  e_z = axis
  # Choose a reference vector not parallel to e_z
  ref = np.array([0.0, 0.0, 1.0])
  if abs(np.dot(ref, e_z)) > 0.99:
    ref = np.array([1.0, 0.0, 0.0])
  
  e_x = np.cross(ref, e_z)
  e_x_norm = np.linalg.norm(e_x)
  if e_x_norm < 1e-9:
    # Degenerate, fall back to some default
    e_x = np.array([1.0, 0.0, 0.0])
    e_x_norm = 1.0
  e_x /= e_x_norm
  
  e_y = np.cross(e_z, e_x)
  # e_y should already be unit length if e_x and e_z are orthonormal
  
  R = np.vstack((e_x, e_y, e_z))
  return R

# Base geometry for servo 1
B1 = np.array([-50.0, 100.0, 10.0])
H1 = np.array([-85.0, 108.0 + 3.3 / 2.0, 10.0])
U1 = np.array([-7.5, 108.0 + 3.3 / 2.0, 152.5])

def generate_coordinates():
  bottom = np.zeros((6, 3))
  horn = np.zeros((6, 3))
  upper = np.zeros((6, 3))

  angles = [0.0, 120.0, 240.0]

  # Servos 1,3,5: rotations of servo1 about Z axis
  for i, a in enumerate(angles):
    idx = i * 2  # 0,2,4
    bottom[idx] = rotate_z(B1, a)
    horn[idx]   = rotate_z(H1, a)
    upper[idx]  = rotate_z(U1, a)

  # Servos 2,4,6: rotations of X-axis mirror of servo1 about Z axis
  # Servo6 is X-mirror of servo1 (0°), Servo2 is 120°, Servo4 is 240°
  B1m = mirror_x(B1)
  H1m = mirror_x(H1)
  U1m = mirror_x(U1)
  
  # Servo2: 120° rotation of X-mirrored servo1
  bottom[1] = rotate_z(B1m, 120.0)
  horn[1]   = rotate_z(H1m, 120.0)
  upper[1]  = rotate_z(U1m, 120.0)
  
  # Servo4: 240° rotation of X-mirrored servo1
  bottom[3] = rotate_z(B1m, 240.0)
  horn[3]   = rotate_z(H1m, 240.0)
  upper[3]  = rotate_z(U1m, 240.0)
  
  # Servo6: X-mirror of servo1 (0° rotation, just mirror)
  bottom[5] = B1m
  horn[5]   = H1m
  upper[5]  = U1m

  return bottom, horn, upper

def solve_servo_angle_z_axis(B, H0, U_target, L):
  """
  Solve inverse kinematics for one leg.
  
  B: (3,) bottom anchor (servo axle center)
  H0: (3,) horn ball position at 0 deg
  U_target: (3,) desired upper anchor position
  L: linkage length
  
  Returns: (theta1_deg, theta2_deg, success)
  """
  h = H0 - B  # Horn vector at 0°
  s = U_target - B  # Vector from bottom to target upper anchor

  # Coefficients for: a*cos(θ) + b*sin(θ) = γ'
  a = s[0]*h[0] + s[1]*h[1]
  b = -s[0]*h[1] + s[1]*h[0]
  gamma = 0.5 * (np.dot(s, s) + np.dot(h, h) - L*L)
  gamma_p = gamma - s[2]*h[2]  # Account for Z component if horn has Z offset

  r = np.hypot(a, b)
  if r < 1e-9:
    return 0.0, 0.0, False

  x = gamma_p / r
  if x < -1.0 or x > 1.0:
    return 0.0, 0.0, False  # No solution (linkage too short/long)

  phi = np.arctan2(b, a)
  d = np.arccos(x)

  theta1_deg = np.degrees(phi + d)
  theta2_deg = np.degrees(phi - d)
  return theta1_deg, theta2_deg, True

def pick_solution(theta1, theta2, min_deg=-90, max_deg=90, prefer_deg=0.0):
  """
  Pick a solution within servo limits.
  If both valid, choose closest to prefer_deg.
  Returns: selected angle in degrees, or None if no valid solution
  """
  candidates = []
  for t in (theta1, theta2):
    # Normalize to [-180, 180] for comparison
    tn = (t + 180.0) % 360.0 - 180.0
    if min_deg <= tn <= max_deg:
      candidates.append(tn)
  
  if not candidates:
    return None
  
  return min(candidates, key=lambda t: abs(t - prefer_deg))

def compute_horn_position_ik(B, H0, servo_angle_deg):
  """
  Compute horn ball link position given servo angle.
  Rotates horn vector by servo_angle_deg about Z axis.
  """
  h = H0 - B  # Horn vector at 0°
  h_rotated = rotate_z(h, servo_angle_deg)
  return B + h_rotated

def transform_upper_anchors(upper_original, dz=0.0, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0, dx=0.0, dy=0.0):
  """
  Transform upper anchors by translation and rotation.
  Rotations are applied about the center of the upper plate.
  Order: Translate to origin, Roll (X-axis), Pitch (Y-axis), Yaw (Z-axis),
  translate back, then global translation (dx, dy, dz).
  """
  upper = upper_original.copy()
  
  # Find center of upper plate
  center = np.mean(upper, axis=0)
  
  # Translate to origin for rotation
  upper_centered = upper - center
  
  # Apply Roll (rotation about X axis)
  if roll_deg != 0.0:
    upper_centered = np.array([rotate_x(p, roll_deg) for p in upper_centered])
  
  # Apply Pitch (rotation about Y axis)
  if pitch_deg != 0.0:
    upper_centered = np.array([rotate_y(p, pitch_deg) for p in upper_centered])
  
  # Apply Yaw (rotation about Z axis)
  if yaw_deg != 0.0:
    upper_centered = np.array([rotate_z(p, yaw_deg) for p in upper_centered])
  
  # Translate back to original center
  upper = upper_centered + center
  
  # Apply global translation
  if abs(dx) > 1e-9 or abs(dy) > 1e-9 or abs(dz) > 1e-9:
    upper = np.array([translate(p, dx, dy, dz) for p in upper])
  
  return upper

def format_config_title(dz=0.0, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0, dx=0.0, dy=0.0):
  """
  Generate a human-readable title from transform parameters.
  Ensures labels always match dz / roll / yaw / dx / dy values used.
  """
  parts = []
  if abs(dx) > 1e-6:
    parts.append(f"X {dx:+.1f}mm")
  if abs(dy) > 1e-6:
    parts.append(f"Y {dy:+.1f}mm")
  if abs(dz) > 1e-6:
    parts.append(f"Z {dz:+.1f}mm")
  if abs(roll_deg) > 1e-6:
    parts.append(f"Roll {roll_deg:+.1f}°")
  if abs(pitch_deg) > 1e-6:
    parts.append(f"Pitch {pitch_deg:+.1f}°")
  if abs(yaw_deg) > 1e-6:
    parts.append(f"Yaw {yaw_deg:+.1f}°")
  if not parts:
    return "Original Position"
  return ", ".join(parts)

def compute_all_servo_angles(bottom, horn_0deg, upper_target, L, min_deg=-90, max_deg=90):
  """
  Compute servo angles for all 6 legs using inverse kinematics.
  
  Returns:
    angles: list of 6 angles (degrees), None if IK failed
    horn_ik: (6, 3) array of computed horn positions
    success: list of 6 booleans indicating IK success
  """
  angles = []
  horn_ik = np.zeros((6, 3))
  success_flags = []
  
  for i in range(6):
    # Build servo-local frame where the real servo axis is the local Z axis.
    axis = get_servo_axis(i)
    R = build_servo_frame(axis)  # global -> local
    
    B_global = bottom[i]
    H0_global = horn_0deg[i]
    U_global = upper_target[i]
    
    # Express horn and target vectors in the servo-local frame with B at origin.
    h_local = R @ (H0_global - B_global)
    u_local = R @ (U_global - B_global)
    
    B_local = np.zeros(3)
    H0_local = h_local
    U_local = u_local
    
    # Reuse existing Z-axis IK solver in the servo-local frame.
    theta1, theta2, ok = solve_servo_angle_z_axis(B_local, H0_local, U_local, L)
    if not ok:
      angles.append(None)
      horn_ik[i] = horn_0deg[i]  # Fallback to 0° position
      success_flags.append(False)
      continue
    
    theta = pick_solution(theta1, theta2, min_deg, max_deg, prefer_deg=0.0)
    if theta is None:
      angles.append(None)
      horn_ik[i] = horn_0deg[i]
      success_flags.append(False)
    else:
      angles.append(theta)
      # Rotate horn in local frame about local Z, then map back to global.
      h_rot_local = rotate_z(h_local, theta)
      horn_global = B_global + R.T @ h_rot_local
      horn_ik[i] = horn_global
      success_flags.append(True)
  
  return angles, horn_ik, success_flags

def plot_single_platform(ax, bottom, horn, upper, title, show_angles=None):
  """Plot a single platform configuration in a subplot"""
  
  # Bottom anchors
  ax.scatter(bottom[:, 0], bottom[:, 1], bottom[:, 2], c='b', s=30, label='BottomAnchor')

  # Upper anchors
  ax.scatter(upper[:, 0], upper[:, 1], upper[:, 2], c='r', s=30, label='UpperAnchor')

  # Upper plate polygon (projected through the 6 upper anchor points)
  upper_center = np.mean(upper, axis=0)
  # Order anchors around center by angle in XY so edges don't cross
  angles_xy = np.arctan2(upper[:, 1] - upper_center[1],
                         upper[:, 0] - upper_center[0])
  order = np.argsort(angles_xy)
  upper_ordered = upper[order]
  upper_closed = np.vstack([upper_ordered, upper_ordered[0:1]])
  ax.plot(upper_closed[:, 0], upper_closed[:, 1], upper_closed[:, 2],
          'r-', linewidth=1.5, alpha=0.6, label='UpperPlate')

  # Link rods (horn ball link -> upper anchor)
  for i in range(6):
    xs = [horn[i, 0], upper[i, 0]]
    ys = [horn[i, 1], upper[i, 1]]
    zs = [horn[i, 2], upper[i, 2]]
    ax.plot(xs, ys, zs, 'k-', linewidth=1.5, label='Linkage' if i == 0 else '')

  # Horns (bottom -> horn position)
  for i in range(6):
    xs = [bottom[i, 0], horn[i, 0]]
    ys = [bottom[i, 1], horn[i, 1]]
    zs = [bottom[i, 2], horn[i, 2]]
    ax.plot(xs, ys, zs, 'g--', linewidth=1, alpha=0.5)
    
    # Show servo angle if provided
    if show_angles is not None and show_angles[i] is not None:
      ax.text(bottom[i, 0], bottom[i, 1], bottom[i, 2] - 20,
              f"θ={show_angles[i]:.1f}°", fontsize=6, color='purple')

  # Approximate upper plate normal and draw it at the plate center
  # Use two non-collinear edges on the ordered upper polygon
  v1 = upper_ordered[1] - upper_ordered[0]
  v2 = upper_ordered[2] - upper_ordered[0]
  n = np.cross(v1, v2)
  n_norm = np.linalg.norm(n)
  if n_norm > 1e-6:
    n_unit = n / n_norm
    normal_scale = 40.0  # mm, purely for visualization
    ax.quiver(upper_center[0], upper_center[1], upper_center[2],
              n_unit[0] * normal_scale,
              n_unit[1] * normal_scale,
              n_unit[2] * normal_scale,
              color='magenta', linewidth=2, arrow_length_ratio=0.15,
              label='Plate normal')

  # Make aspect roughly equal
  all_pts = np.vstack([bottom, upper, horn])
  x_min, y_min, z_min = all_pts.min(axis=0)
  x_max, y_max, z_max = all_pts.max(axis=0)
  max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
  mid_x = (x_max + x_min) / 2.0
  mid_y = (y_max + y_min) / 2.0
  mid_z = (z_max + z_min) / 2.0

  ax.set_xlim(mid_x - max_range / 2.0, mid_x + max_range / 2.0)
  ax.set_ylim(mid_y - max_range / 2.0, mid_y + max_range / 2.0)
  ax.set_zlim(mid_z - max_range / 2.0, mid_z + max_range / 2.0)

  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Z")
  ax.set_title(title)
  ax.legend(fontsize=6)
  
  # Set top-down view (looking along -Z onto XY plane)
  ax.view_init(elev=45.0, azim=-45.0)

def plot_platform(bottom, horn, upper):
  """Original single plot function (for backward compatibility)"""
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  plot_single_platform(ax, bottom, horn, upper, "Stewart Platform Servo & Link Geometry")
  plt.show()

def simulate_circular_motion(bottom, horn_0deg, upper_original,
                             radius_mm=50.0, dz=0.0, roll_deg=0.0,
                             pitch_deg=0.0, yaw_deg=0.0, n_steps=180,
                             min_deg=-90.0, max_deg=90.0,
                             align_normal_to_origin=True,
                             tilt_deg=10.0):
  """
  Sample a horizontal circular trajectory of the upper plate center in XY.
  Center starts at current upper plate center and is offset by (dx, dy) on
  a circle of given radius in the XY plane.
  
  Returns a list of samples:
    (t_deg, dx, dy, angles, success_flags, horn_ik, upper_transformed)
  """
  thetas = np.linspace(0.0, 360.0, n_steps, endpoint=False)
  samples = []
  for t in thetas:
    rad = t * DEG2RAD
    dx = radius_mm * np.cos(rad)
    dy = radius_mm * np.sin(rad)
    
    # Compute roll/pitch so that the platform normal leans radially outward
    # in the XY plane (i.e. -normal points toward the origin), using only
    # roll and pitch (no yaw).
    if align_normal_to_origin and (abs(dx) > 1e-9 or abs(dy) > 1e-9):
      # For small angles, n_xy ≈ (pitch, -roll). To align n_xy with (dx, dy),
      # choose roll, pitch proportional to dy and dx respectively.
      roll_step = -tilt_deg * (dy / radius_mm)
      pitch_step = tilt_deg * (dx / radius_mm)
      roll_cmd = roll_deg + roll_step
      pitch_cmd = pitch_deg + pitch_step
    else:
      roll_cmd = roll_deg
      pitch_cmd = pitch_deg
    
    upper_t = transform_upper_anchors(
      upper_original,
      dz=dz,
      roll_deg=roll_cmd,
      pitch_deg=pitch_cmd,
      yaw_deg=yaw_deg,
      dx=dx,
      dy=dy,
    )
    
    angles, horn_ik, success = compute_all_servo_angles(
      bottom, horn_0deg, upper_t, LINK_LENGTH,
      min_deg=min_deg, max_deg=max_deg
    )
    samples.append((t, dx, dy, angles, success, horn_ik, upper_t))
  
  return samples

def animate_circular_motion(bottom, horn_0deg, upper_original,
                            radius_mm=50.0, dz=0.0, roll_deg=0.0,
                            pitch_deg=0.0, yaw_deg=0.0, n_steps=180,
                            min_deg=-90.0, max_deg=90.0, pause_s=0.05,
                            align_normal_to_origin=True,
                            tilt_deg=10.0):
  """
  Animate the platform following a horizontal circular motion in XY.
  Close the matplotlib window or interrupt the script to stop the animation.
  """
  samples = simulate_circular_motion(
    bottom, horn_0deg, upper_original,
    radius_mm=radius_mm,
    dz=dz,
    roll_deg=roll_deg,
    pitch_deg=pitch_deg,
    yaw_deg=yaw_deg,
    n_steps=n_steps,
    min_deg=min_deg,
    max_deg=max_deg,
    align_normal_to_origin=align_normal_to_origin,
    tilt_deg=tilt_deg,
  )
  
  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111, projection='3d')
  
  # Use first valid configuration to set consistent axis limits
  for (_, _, _, _, success, horn_ik, upper_t) in samples:
    if all(success):
      plot_single_platform(ax, bottom, horn_ik, upper_t,
                           "Stewart Platform Circular Motion (initial frame)")
      break
  plt.pause(pause_s)
  
  for t_deg, dx, dy, angles, success, horn_ik, upper_t in samples:
    ax.cla()
    title = (f"Circular XY motion r={radius_mm:.1f}mm, "
             f"t={t_deg:6.1f}°, dx={dx:+6.1f}, dy={dy:+6.1f}")
    # Mark frames where any leg fails IK
    if not all(success):
      title += "  [IK FAIL]"
    plot_single_platform(ax, bottom, horn_ik, upper_t, title, show_angles=angles)
    plt.pause(pause_s)
  
  plt.show()

if __name__ == "__main__":
  bottom, horn, upper = generate_coordinates()
  for i in range(6):
    print(f"Servo {i+1}")
    print(f"  Bottom: {bottom[i]}")
    print(f"  Horn0:  {horn[i]}")
    print(f"  Upper:  {upper[i]}")
    print()
  
  # Animate a circular XY path within actuation range
  print("=== Circular XY Motion Animation ===")
  print("Radius = 50 mm, dz = 0, roll = 0, pitch = 0, yaw = 0\n")
  animate_circular_motion(
    bottom,
    horn,
    upper,
    radius_mm=50.0,
    dz=0.0,
    roll_deg=0.0,
    pitch_deg=0.0,
    yaw_deg=0.0,
    n_steps=180,
    min_deg=-90.0,
    max_deg=90.0,
    pause_s=0.05,
    align_normal_to_origin=True,
    tilt_deg=10.0,
  )

