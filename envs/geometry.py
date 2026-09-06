from __future__ import annotations

import numpy as np


def quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def goal_vec(rx: float, ry: float, yaw: float, gx: float, gy: float) -> np.ndarray:
    dx, dy = gx - rx, gy - ry
    dist = float(np.hypot(dx, dy))
    ang = (float(np.arctan2(dy, dx)) - yaw + np.pi) % (2.0 * np.pi) - np.pi
    return np.array([dist, ang / np.pi], dtype=np.float32)
