#pragma once

// Basic 3D vector and 3x3 matrix types used by main.ino.
// Placed in a header so Arduino's auto-generated function prototypes
// (inserted after includes) can see these types.

struct Vec3 {
  float x, y, z;
};

struct Mat3 {
  float m[3][3]; // row-major
};



