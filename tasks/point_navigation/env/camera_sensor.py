"""
RGB-D カメラセンサーのセットアップ（IsaacSim 6.0 対応）

omni.isaac.sensor → isaacsim.sensors.camera に移行済み．
Carter V1 の chassis_link 前方に固定カメラを配置する．
"""

from __future__ import annotations

import numpy as np

# カメラの取り付け位置（chassis_link ローカル座標, Y-up）
# Y-up 座標系: X=右, Y=上, Z=後ろ → 前方は -Z 方向
_CAMERA_TRANSLATE = np.array([0.0, 0.3, -0.3])  # 前方0.3m, 高さ0.3m
# 前方(-Z)向き: カメラのデフォルト向きは -Z なので orientation は identity で良い
_CAMERA_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)


class RGBDCamera:
    """RGB と Depth を同時に取得できるカメララッパー．"""

    def __init__(self, camera_prim_path: str, resolution: tuple[int, int]):
        from isaacsim.sensors.camera import Camera
        import omni.replicator.core as rep

        self._resolution = resolution  # (W, H)

        # カメラ prim を作成・初期化
        self._cam = Camera(
            prim_path=camera_prim_path,
            translation=_CAMERA_TRANSLATE,   # 親 prim からのローカル translation
            orientation=_CAMERA_ORIENTATION,
            frequency=30,
            resolution=resolution,
        )
        self._cam.initialize()

        # Replicator annotator で RGB と Depth を取得
        self._rp = rep.create.render_product(
            camera_prim_path,
            resolution=resolution,
        )
        self._rgb_ann   = rep.AnnotatorRegistry.get_annotator("rgb")
        self._depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self._rgb_ann.attach([self._rp])
        self._depth_ann.attach([self._rp])

    def get_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            rgb:   (H, W, 3) uint8
            depth: (H, W)    float32 [m]
        """
        import omni.replicator.core as rep
        rep.orchestrator.step(rt_subframes=4, pause_timeline=False)

        rgb_data   = self._rgb_ann.get_data()
        depth_data = self._depth_ann.get_data()

        W, H = self._resolution

        if rgb_data is not None and rgb_data.size > 0:
            # annotator は RGBA (H,W,4) を返すことがある
            rgb = rgb_data[..., :3] if rgb_data.shape[-1] == 4 else rgb_data
        else:
            rgb = np.zeros((H, W, 3), dtype=np.uint8)

        if depth_data is not None and depth_data.size > 0:
            depth = depth_data.astype(np.float32)
        else:
            depth = np.zeros((H, W), dtype=np.float32)

        return rgb, depth


def setup_rgbd_camera(
    prim_path: str,
    resolution: tuple[int, int] = (84, 84),
) -> RGBDCamera:
    """
    カメラを作成して返す．

    Args:
        prim_path:  カメラ prim のパス（親 prim に子として作成される）
        resolution: (W, H)
    """
    return RGBDCamera(camera_prim_path=prim_path, resolution=resolution)
