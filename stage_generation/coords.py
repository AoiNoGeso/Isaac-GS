"""Source up-axis -> Z-up 座標変換 (convert_gs / convert_mesh 共通)"""

import math

import numpy as np
from pxr import Gf

_VALID_AXES = ("Y",)


def rotation_matrix(source_up_axis: str) -> np.ndarray:
    """変換元up-axisをZ-upに直す回転行列を返す(現状"Y"のみ対応)"""
    if source_up_axis == "Y":
        return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    raise ValueError(f"Unsupported source_up_axis: {source_up_axis!r} (expected one of {_VALID_AXES})")


def rotation_quat(source_up_axis: str) -> Gf.Quatf:
    """rotation_matrixと同じ回転をクォータニオンで返す"""
    if source_up_axis == "Y":
        return Gf.Quatf(math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0)
    raise ValueError(f"Unsupported source_up_axis: {source_up_axis!r} (expected one of {_VALID_AXES})")


def apply_matrix_to_vec3_array(arr: np.ndarray, R: np.ndarray) -> np.ndarray:
    """(N,3)配列の各点に回転行列Rを適用する"""
    return (R @ np.array(arr).T).T


def compute_extent(points: np.ndarray):
    """点群(N,3)の最小値・最大値を返す"""
    return points.min(axis=0), points.max(axis=0)
