"""
Geometry consistency check for the Stewart platform.

Purpose:
  Catch common "accuracy killers" early:
  - mismatched linkage length between Python and Arduino
  - mismatched reference anchor coordinates
  - unit/sign mistakes introduced during edits

Usage:
  python geometry_consistency_check.py
  python geometry_consistency_check.py --tol 1e-3
"""

from __future__ import annotations

import argparse
import ast
import importlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Geometry:
    link_length_mm: Optional[float] = None
    b1: Optional[Tuple[float, float, float]] = None
    h1: Optional[Tuple[float, float, float]] = None
    u1: Optional[Tuple[float, float, float]] = None


def _safe_eval_arith(expr: str) -> float:
    """
    Safely evaluate a tiny arithmetic subset used in constants like:
      108.0f + 3.3f / 2.0f
    Allowed: numbers, + - * /, parentheses, unary +/-.
    """
    expr = expr.strip().replace("f", "")  # strip Arduino float suffix
    if not re.fullmatch(r"[0-9eE\.\+\-\*/\(\)\s]+", expr):
        raise ValueError(f"Disallowed characters in expression: {expr!r}")

    node = ast.parse(expr, mode="eval")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = _eval(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a = _eval(n.left)
            b = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
        raise ValueError(f"Disallowed expression node: {ast.dump(n)}")

    return _eval(node)


def _parse_vec3_literal(s: str) -> Tuple[float, float, float]:
    # Accept formats like: {-50.0f, 100.0f, 10.0f} or per-component expressions.
    s = s.strip()
    if not (s.startswith("{") and s.endswith("}")):
        raise ValueError(f"Expected braces for Vec3 literal, got: {s!r}")
    inner = s[1:-1]
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated components, got {len(parts)} from: {s!r}")
    return (_safe_eval_arith(parts[0]), _safe_eval_arith(parts[1]), _safe_eval_arith(parts[2]))


def read_ref_coordinates_md(path: Path) -> Geometry:
    text = path.read_text(encoding="utf-8", errors="replace")

    # Linkage 162.21128 (ball link center to center)
    m_link = re.search(r"Linkage\s+([-+]?\d+(?:\.\d+)?)", text)
    link = float(m_link.group(1)) if m_link else None

    # BottomAchor1 ... -50, 100, 10
    m_b1 = re.search(r"BottomAchor1.*?([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?)", text)
    b1 = (float(m_b1.group(1)), float(m_b1.group(2)), float(m_b1.group(3))) if m_b1 else None

    # servo horn ball link center @(-85, 108+3.3/2, 10)
    # We'll accept either fully-evaluated or the expression form; if expression, we compute it.
    m_h1 = re.search(r"ball link center\s*@\(\s*([-+]?\d+(?:\.\d+)?),\s*([^,]+),\s*([-+]?\d+(?:\.\d+)?)\s*\)", text, flags=re.IGNORECASE)
    h1 = None
    if m_h1:
        x = float(m_h1.group(1))
        y_expr = m_h1.group(2).strip()
        z = float(m_h1.group(3))
        try:
            y = _safe_eval_arith(y_expr)
            h1 = (x, float(y), z)
        except Exception:
            h1 = None

    # UpperAchor1 ... -7.5, 108+3.3/2, 152.5
    m_u1 = re.search(r"UpperAchor1.*?([-+]?\d+(?:\.\d+)?),\s*([^,]+),\s*([-+]?\d+(?:\.\d+)?)", text)
    u1 = None
    if m_u1:
        x = float(m_u1.group(1))
        y_expr = m_u1.group(2).strip()
        z = float(m_u1.group(3))
        try:
            y = _safe_eval_arith(y_expr)
            u1 = (x, float(y), z)
        except Exception:
            u1 = None

    return Geometry(link_length_mm=link, b1=b1, h1=h1, u1=u1)


def read_python_geometry() -> Geometry:
    cg = importlib.import_module("coordinate_generator")
    link = float(getattr(cg, "LINK_LENGTH", math.nan))
    b1 = tuple(float(v) for v in getattr(cg, "B1"))
    h1 = tuple(float(v) for v in getattr(cg, "H1"))
    u1 = tuple(float(v) for v in getattr(cg, "U1"))
    return Geometry(link_length_mm=link, b1=b1, h1=h1, u1=u1)


def read_arduino_geometry(path: Path) -> Geometry:
    text = path.read_text(encoding="utf-8", errors="replace")

    m_link = re.search(r"const\s+float\s+LINK_LENGTH\s*=\s*([-+]?\d+(?:\.\d+)?)", text)
    link = float(m_link.group(1)) if m_link else None

    m_b1 = re.search(r"const\s+Vec3\s+B1_pos\s*=\s*(\{[^}]+\})", text)
    m_h1 = re.search(r"const\s+Vec3\s+H1\s*=\s*(\{[^}]+\})", text)
    m_u1 = re.search(r"const\s+Vec3\s+U1\s*=\s*(\{[^}]+\})", text)

    b1 = _parse_vec3_literal(m_b1.group(1)) if m_b1 else None
    h1 = _parse_vec3_literal(m_h1.group(1)) if m_h1 else None
    u1 = _parse_vec3_literal(m_u1.group(1)) if m_u1 else None

    return Geometry(link_length_mm=link, b1=b1, h1=h1, u1=u1)


def _diff(a: Optional[Tuple[float, float, float]], b: Optional[Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
    if a is None or b is None:
        return None
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _max_abs(t: Optional[Tuple[float, float, float]]) -> Optional[float]:
    if t is None:
        return None
    return max(abs(t[0]), abs(t[1]), abs(t[2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-3, help="tolerance (mm) for coordinate equality checks")
    args = ap.parse_args()

    ref = read_ref_coordinates_md(REPO_ROOT / "ref" / "coordinates.md")
    py = read_python_geometry()
    ino = read_arduino_geometry(REPO_ROOT / "main" / "main.ino")

    def pr(name: str, g: Geometry) -> None:
        print(f"== {name} ==")
        print(f"LINK_LENGTH mm: {g.link_length_mm}")
        print(f"B1: {g.b1}")
        print(f"H1: {g.h1}")
        print(f"U1: {g.u1}")
        print()

    pr("ref/coordinates.md", ref)
    pr("Python (coordinate_generator.py)", py)
    pr("Arduino (main/main.ino)", ino)

    failed = False

    def check_scalar(label: str, a: Optional[float], b: Optional[float], tol: float) -> None:
        nonlocal failed
        if a is None or b is None:
            print(f"[WARN] {label}: missing value(s) (a={a}, b={b})")
            return
        if not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > tol:
            print(f"[FAIL] {label}: {a} vs {b} (diff={a-b:+.6f})")
            failed = True
        else:
            print(f"[ OK ] {label}: match within tol ({tol})")

    def check_vec(label: str, a: Optional[Tuple[float, float, float]], b: Optional[Tuple[float, float, float]], tol: float) -> None:
        nonlocal failed
        if a is None or b is None:
            print(f"[WARN] {label}: missing value(s) (a={a}, b={b})")
            return
        d = _diff(a, b)
        ma = _max_abs(d)
        if ma is None or ma > tol:
            print(f"[FAIL] {label}: {a} vs {b} (diff={d}, maxAbs={ma})")
            failed = True
        else:
            print(f"[ OK ] {label}: match within tol ({tol})")

    print("=== Checks (ref vs Python, ref vs Arduino) ===")
    check_scalar("LINK_LENGTH (ref vs Python)", ref.link_length_mm, py.link_length_mm, tol=args.tol)
    check_scalar("LINK_LENGTH (ref vs Arduino)", ref.link_length_mm, ino.link_length_mm, tol=args.tol)

    check_vec("B1 (ref vs Python)", ref.b1, py.b1, tol=args.tol)
    check_vec("B1 (ref vs Arduino)", ref.b1, ino.b1, tol=args.tol)

    check_vec("H1 (ref vs Python)", ref.h1, py.h1, tol=args.tol)
    check_vec("H1 (ref vs Arduino)", ref.h1, ino.h1, tol=args.tol)

    check_vec("U1 (ref vs Python)", ref.u1, py.u1, tol=args.tol)
    check_vec("U1 (ref vs Arduino)", ref.u1, ino.u1, tol=args.tol)

    if failed:
        print("\nResult: FAIL (geometry mismatch detected).")
        print("Fix by updating constants so ref/Python/Arduino describe the SAME physical build.")
        return 2

    print("\nResult: OK (geometry appears consistent within tolerance).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


