import numpy as np


class RGBDCamera:
    def __init__(self, camera_prim_path: str, resolution: tuple[int, int]):
        import omni.replicator.core as rep
        from isaacsim.sensors.camera import Camera

        self._resolution = resolution
        self._rep = rep

        self._cam = Camera(
            prim_path=camera_prim_path,
            translation=np.array([-1.0, 0.0, 1.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            frequency=30,
            resolution=resolution,
        )
        self._cam.initialize()

        rp = rep.create.render_product(camera_prim_path, resolution=resolution)
        self._rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
        self._depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self._rgb_ann.attach([rp])
        self._depth_ann.attach([rp])

    def get_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        self._rep.orchestrator.step(rt_subframes=4, pause_timeline=False)

        W, H = self._resolution
        rgb_data = self._rgb_ann.get_data()
        depth_data = self._depth_ann.get_data()

        if rgb_data is not None and rgb_data.size > 0:
            rgb = rgb_data[..., :3] if rgb_data.shape[-1] == 4 else rgb_data
        else:
            rgb = np.zeros((H, W, 3), dtype=np.uint8)

        if depth_data is not None and depth_data.size > 0:
            depth = depth_data.astype(np.float32)
        else:
            depth = np.zeros((H, W), dtype=np.float32)

        return rgb, depth
