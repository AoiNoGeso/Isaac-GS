import numpy as np


class RGBCamera:
    """USDカメラprimからRGB画像を取得するセンサー`Camera`初期化時にアタッチされる"""

    def __init__(
        self,
        camera_prim_path: str,
        resolution: tuple[int, int],
        translation: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
    ):
        from isaacsim.sensors.camera import Camera

        self._resolution = resolution

        kwargs = {}
        if translation is not None:
            kwargs["translation"] = translation
        if orientation is not None:
            kwargs["orientation"] = orientation

        self._cam = Camera(
            prim_path=camera_prim_path,
            frequency=30,
            resolution=resolution,
            **kwargs,
        )
        self._cam.initialize()

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    def get_rgb(self) -> np.ndarray:
        """(H,W,3) uint8のRGB画像を返す(まだ描画結果が無い場合はゼロ画像)
        呼び出し前に`world.step(render=True)`が実行済みであること"""
        rgb_data = self._cam.get_rgb()
        if rgb_data is None or rgb_data.size == 0:
            W, H = self._resolution
            return np.zeros((H, W, 3), dtype=np.uint8)
        return rgb_data
