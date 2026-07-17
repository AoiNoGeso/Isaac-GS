"""Source up-axis -> Z-up 座標変換 (convert_gs / convert_mesh 共通)"""

import math

import numpy as np
from pxr import Gf

_VALID_AXES = ("Y", "Z", "Yinv")


def rotation_matrix(source_up_axis: str) -> np.ndarray:
    """source_up_axis の座標系を Z-up に変換する回転行列を返す.

    "Y": -Y-up -> Z-up の -90°X 回転 ((x,y,z) -> (x,z,-y))
    "Z": 既に Z-up のため恒等行列
    "Yinv": "Y" の逆回転 (+90°X) ((x,y,z) -> (x,-z,y)).
        GS とメッシュの .ply が互いに上下逆の座標系でエクスポートされている場合など,
        "Y" では合わないときに使う.
    """
    if source_up_axis == "Y":
        return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    if source_up_axis == "Z":
        return np.eye(3, dtype=np.float64)
    if source_up_axis == "Yinv":
        return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    raise ValueError(f"Unsupported source_up_axis: {source_up_axis!r} (expected one of {_VALID_AXES})")


def rotation_quat(source_up_axis: str) -> Gf.Quatf:
    """rotation_matrix と同じ回転を表すクォータニオンを返す."""
    if source_up_axis == "Y":
        return Gf.Quatf(math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0)
    if source_up_axis == "Z":
        return Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    if source_up_axis == "Yinv":
        return Gf.Quatf(math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
    raise ValueError(f"Unsupported source_up_axis: {source_up_axis!r} (expected one of {_VALID_AXES})")


def apply_matrix_to_vec3_array(arr: np.ndarray, R: np.ndarray) -> np.ndarray:
    """(N,3) 配列に回転行列 R を適用する."""
    return (R @ np.array(arr).T).T


def compute_extent(points: np.ndarray):
    """points (N,3) から extent (min, max) を計算する."""
    return points.min(axis=0), points.max(axis=0)
