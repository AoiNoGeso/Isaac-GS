import numpy as np


class RGBCamera:
    """USDカメラprimからRGB画像を取得するセンサー"""

    def __init__(
        self,
        camera_prim_path: str,
        resolution: tuple[int, int],
        translation: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
    ):
        import omni.replicator.core as rep
        from isaacsim.sensors.camera import Camera

        self._resolution = resolution
        self._rep = rep

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

        rp = rep.create.render_product(camera_prim_path, resolution=resolution)
        self._rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb_ann.attach([rp])

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    def get_rgb(self) -> np.ndarray:
        """(H,W,3) uint8のRGB画像を返す(まだ描画結果が無い場合はゼロ画像)"""
        self._rep.orchestrator.step(rt_subframes=4, pause_timeline=False)

        rgb_data = self._rgb_ann.get_data()
        if rgb_data is None or rgb_data.size == 0:
            W, H = self._resolution
            return np.zeros((H, W, 3), dtype=np.uint8)
        return rgb_data[..., :3] if rgb_data.shape[-1] == 4 else rgb_data
