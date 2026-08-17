"""エピソード動画の収録(ロボット視点 + 任意で俯瞰カメラ)。train.py / test.pyが共用する"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OVERHEAD_CAMERA_PRIM_PATH = "/World/OverheadCamera"
OVERHEAD_CAMERA_RESOLUTION = (854, 480)  # 480p相当。ロボット観測用84x84とは独立

_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


def make_overhead_camera(stage_preset, resolution: tuple[int, int] = OVERHEAD_CAMERA_RESOLUTION):
    """stage_presetに俯瞰カメラ座標が設定されていればRGBCameraを生成する(未設定ならNone)"""
    if stage_preset.overhead_camera_translation is None:
        return None

    from envs.sensors.camera_sensor import RGBCamera

    return RGBCamera(
        camera_prim_path=OVERHEAD_CAMERA_PRIM_PATH,
        resolution=resolution,
        translation=np.array(stage_preset.overhead_camera_translation, dtype=np.float32),
        orientation=(
            np.array(stage_preset.overhead_camera_orientation, dtype=np.float32)
            if stage_preset.overhead_camera_orientation is not None
            else None
        ),
    )


def write_frame(writer: cv2.VideoWriter, rgb: np.ndarray) -> None:
    """RGBフレームをBGRへ直してwriterへ書き込む
    (3,H,W) float32 [0,1] の観測形式と (H,W,3) uint8 のカメラ出力形式の両方を受け付ける"""
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = (rgb.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


class EpisodeRecorder:
    """1エピソード分の動画を書き出す

    ロボット視点は`{out_dir}/robot/`、俯瞰カメラ(渡した場合のみ)は`{out_dir}/overhead/`へ
    それぞれ同じファイル名で保存する。`finish()`で両方のファイル名末尾へ終了理由タグ(s/h/w/t)
    を付けてリネームする(最終的なファイル名は`{stem}_{タグ}.mp4`)

    使い方:
        rec.start("50000_3", obs)   # エピソード開始時(初期観測を1フレーム目として記録)
        rec.capture(obs)            # 毎ステップ
        rec.finish("s")             # エピソード終了時 -> robot/50000_3_s.mp4 等にリネーム
    """

    def __init__(
        self,
        out_dir: str | Path,
        fps: float,
        robot_resolution: tuple[int, int],
        overhead_camera=None,
    ):
        self._dir = Path(out_dir)
        self._fps = fps
        self._robot_resolution = tuple(robot_resolution)  # (W, H)
        self._overhead_camera = overhead_camera
        self._robot_writer: cv2.VideoWriter | None = None
        self._overhead_writer: cv2.VideoWriter | None = None
        self._robot_path: Path | None = None
        self._overhead_path: Path | None = None

    def start(self, stem: str, obs: dict) -> None:
        """`robot/{stem}.mp4`(・`overhead/{stem}.mp4`)の収録を開始し、初期観測を1フレーム目として書き込む"""
        robot_dir = self._dir / "robot"
        robot_dir.mkdir(parents=True, exist_ok=True)
        self._robot_path = robot_dir / f"{stem}.mp4"
        self._robot_writer = cv2.VideoWriter(
            str(self._robot_path), _FOURCC, self._fps, self._robot_resolution
        )
        if self._overhead_camera is not None:
            overhead_dir = self._dir / "overhead"
            overhead_dir.mkdir(parents=True, exist_ok=True)
            self._overhead_path = overhead_dir / f"{stem}.mp4"
            self._overhead_writer = cv2.VideoWriter(
                str(self._overhead_path),
                _FOURCC,
                self._fps,
                tuple(self._overhead_camera.resolution),
            )
        self.capture(obs)

    def capture(self, obs: dict) -> None:
        """1フレーム分を書き込む(収録中でなければ何もしない)"""
        if self._robot_writer is None:
            return
        write_frame(self._robot_writer, obs["rgb"])
        if self._overhead_writer is not None:
            write_frame(self._overhead_writer, self._overhead_camera.get_rgb())

    def finish(self, tag: str) -> None:
        """writerを閉じ、ファイル名末尾に終了理由タグを付ける(`{stem}_{タグ}.mp4`)"""
        if self._robot_writer is None:
            return
        for writer, path in (
            (self._robot_writer, self._robot_path),
            (self._overhead_writer, self._overhead_path),
        ):
            if writer is None:
                continue
            writer.release()
            path.rename(path.with_stem(f"{path.stem}_{tag}"))
        self._robot_writer = self._overhead_writer = None
        self._robot_path = self._overhead_path = None
