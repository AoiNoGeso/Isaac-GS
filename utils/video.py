"""test.py / train.py が共用する動画書き出しヘルパー"""

import cv2
import numpy as np

OVERHEAD_CAMERA_PRIM_PATH = "/World/OverheadCamera"


def write_frame(writer: cv2.VideoWriter, rgb: np.ndarray) -> None:
    """rgb: (3,H,W) float32 [0,1] -> BGR uint8 (H,W,3) をvideo writerへ書き込む"""
    frame = (rgb.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def write_overhead_frame(writer: cv2.VideoWriter, rgb_hwc: np.ndarray) -> None:
    """rgb_hwc: (H,W,3) uint8 をvideo writerへ書き込む"""
    writer.write(cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR))


def make_overhead_camera(env_cfg, stage_preset):
    """stage_presetに俯瞰カメラ座標が設定されていればRGBDCameraを生成する(未設定ならNone)"""
    if stage_preset.overhead_camera_translation is None:
        return None

    import numpy as np

    from envs.sensors.camera_sensor import RGBDCamera

    return RGBDCamera(
        camera_prim_path=OVERHEAD_CAMERA_PRIM_PATH,
        resolution=env_cfg.camera_resolution,
        translation=np.array(stage_preset.overhead_camera_translation, dtype=np.float32),
        orientation=(
            np.array(stage_preset.overhead_camera_orientation, dtype=np.float32)
            if stage_preset.overhead_camera_orientation is not None
            else None
        ),
    )
