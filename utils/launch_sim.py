"""SimulationApp起動の定型処理(train.py / test.py / teleop.pyで共用)"""

from __future__ import annotations

_BASE_EXTRA_ARGS = ["--/rtx/scenedb/maxHistoryTransformCount=256"]
# gpu指定時はCUDA_VISIBLE_DEVICESで見えるGPUが1枚だけになるためレンダラも0番に固定する
_PINNED_GPU_ARG = "--/renderer/activeGpu=0"


def launch_sim(headless: bool, gpu: int | None = None):
    """SimulationAppを起動して返す

    gpuの固定はCUDA初期化(isaacsim/torch)より前でなければ効かないため、isaacsimのimportは
    必ずこの関数の内側で行う。これが無いと共用マシンでIsaacSim側とPyTorch側が別の物理GPUを
    掴み`weight is on cuda:N, different from other tensors`のようなクラッシュが起きる
    """
    import os

    extra_args = list(_BASE_EXTRA_ARGS)
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        extra_args.append(_PINNED_GPU_ARG)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": headless, "extra_args": extra_args})

    import omni.log

    omni.log.get_log().set_channel_level(
        "omni.physx.plugin", omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE
    )
    return app
