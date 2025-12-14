import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

DEG2RAD = np.pi / 180.0

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

def translate(p, dx, dy, dz):
  """Translate point by (dx, dy, dz)"""
  return p + np.array([dx, dy, dz])

def mirror_x(p):
  """Mirror across YZ plane (negate X coordinate)"""
  x, y, z = p
  return np.array([-x, y, z])

# Base geometry for servo 1 (same as Arduino code)
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

def transform_upper_anchors(upper_original, dz=0.0, roll_deg=0.0, yaw_deg=0.0):
  """
  Transform upper anchors by translation and rotation.
  Rotations are applied about the center of the upper plate.
  Order: Translate to origin, Roll (X-axis), Yaw (Z-axis), translate back, then Z translation.
  """
  upper = upper_original.copy()
  
  # Find center of upper plate
  center = np.mean(upper, axis=0)
  
  # Translate to origin for rotation
  upper_centered = upper - center
  
  # Apply Roll (rotation about X axis)
  if roll_deg != 0.0:
    upper_centered = np.array([rotate_x(p, roll_deg) for p in upper_centered])
  
  # Apply Yaw (rotation about Z axis)
  if yaw_deg != 0.0:
    upper_centered = np.array([rotate_z(p, yaw_deg) for p in upper_centered])
  
  # Translate back to original center
  upper = upper_centered + center
  
  # Apply translation in Z
  if dz != 0.0:
    upper = np.array([translate(p, 0.0, 0.0, dz) for p in upper])
  
  return upper

def plot_single_platform(ax, bottom, horn, upper, title):
  """Plot a single platform configuration in a subplot"""

  # Bottom anchors
  ax.scatter(bottom[:, 0], bottom[:, 1], bottom[:, 2], c='b', s=30, label='BottomAnchor')

  # Upper anchors
  ax.scatter(upper[:, 0], upper[:, 1], upper[:, 2], c='r', s=30, label='UpperAnchor')

  # Link rods (horn ball link -> upper anchor)
  for i in range(6):
    xs = [horn[i, 0], upper[i, 0]]
    ys = [horn[i, 1], upper[i, 1]]
    zs = [horn[i, 2], upper[i, 2]]
    ax.plot(xs, ys, zs, 'k-', linewidth=1.5, label='Linkage' if i == 0 else '')

  # Horns (bottom -> horn@0deg)
  for i in range(6):
    xs = [bottom[i, 0], horn[i, 0]]
    ys = [bottom[i, 1], horn[i, 1]]
    zs = [bottom[i, 2], horn[i, 2]]
    ax.plot(xs, ys, zs, 'g--', linewidth=1, alpha=0.5)

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

def plot_platform(bottom, horn, upper):
  """Original single plot function (for backward compatibility)"""
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  plot_single_platform(ax, bottom, horn, upper, "Stewart Platform Servo & Link Geometry")
  plt.show()

def plot_multiple_configurations(bottom, horn, upper_original):
  """Plot 6 different platform configurations in a 2x3 grid"""
  fig = plt.figure(figsize=(18, 12))
  
  # Define 6 test configurations
  configs = [
    {"title": "Z +50mm", "dz": 50.0, "roll": 0.0, "yaw": 0.0},
    {"title": "Z -50mm", "dz": -50.0, "roll": 0.0, "yaw": 0.0},
    {"title": "Roll +30°", "dz": 0.0, "roll": 30.0, "yaw": 0.0},
    {"title": "Roll -30°", "dz": 0.0, "roll": -30.0, "yaw": 0.0},
    {"title": "Yaw +30°", "dz": 0.0, "roll": 0.0, "yaw": 30.0},
    {"title": "Yaw -30°", "dz": 0.0, "roll": 0.0, "yaw": -30.0},
  ]
  
  for idx, config in enumerate(configs):
    ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
    upper_transformed = transform_upper_anchors(
      upper_original,
      dz=config["dz"],
      roll_deg=config["roll"],
      yaw_deg=config["yaw"]
    )
    plot_single_platform(ax, bottom, horn, upper_transformed, config["title"])
  
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  bottom, horn, upper = generate_coordinates()
  for i in range(6):
    print(f"Servo {i+1}")
    print(f"  Bottom: {bottom[i]}")
    print(f"  Horn0:  {horn[i]}")
    print(f"  Upper:  {upper[i]}")
    print()
  
  # Plot 6 different configurations
  plot_multiple_configurations(bottom, horn, upper)