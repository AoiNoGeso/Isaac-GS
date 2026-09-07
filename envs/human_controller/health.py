"""人物モーション(座標/回転/コマンド/関節)の異常検知。Isaac Simに非依存。

tests/reIRA/health.pyから移設(元のtests/reIRA/health.pyはこのモジュールを再importするだけ)。
「異常を隠さないため自動リセットは行わない」という設計方針(tests/reIRA/README.md参照)に基づき、
異常検知時は例外を送出するだけで、内部で回復処理は行わない。呼び出し側(HumanManager/isaac_env.py)が
エピソード終了などの形で明示的にハンドリングすること。
"""

from collections import deque

import numpy as np


class HumanLocomotionAnomaly(RuntimeError):
    """人物モーションにNaN/Infが混入した、またはフリーズが疑われる場合に送出される。
    RuntimeErrorのサブクラスなので、既存のRuntimeErrorベースの捕捉コードとの後方互換性がある。"""


class HealthMonitor:
    """位置/回転/コマンドの非有限値検知と、コマンドありなのに動いていない(フリーズ疑い)検知を行う。

    使い方:
        health = HealthMonitor(num_agents)
        health.check(positions, rotations, commands)  # 呼び出しごとにNaN/Inf/フリーズを検査
    """

    def __init__(self, agents, window=300):
        self.positions = deque(maxlen=window)
        self.commands = deque(maxlen=window)
        self.window = window
        self.agents = agents

    def check(self, positions, rotations, commands, joints=None):
        for name, value in [
            ("position", positions),
            ("rotation", rotations),
            ("command", commands),
            ("joints", joints),
        ]:
            if value is not None and not np.isfinite(value).all():
                raise HumanLocomotionAnomaly(f"Nonfinite {name}")
        self.positions.append(np.asarray(positions).copy())
        self.commands.append(np.asarray(commands).copy())
        if len(self.positions) == self.window:
            span = np.ptp(np.asarray(self.positions)[:, :, :2], axis=0)
            requested = np.min(np.linalg.norm(np.asarray(self.commands), axis=2), axis=0)
            frozen = np.flatnonzero((np.linalg.norm(span, axis=1) < 0.01) & (requested > 0.15))
            if len(frozen):
                raise HumanLocomotionAnomaly(
                    f"Suspected freeze: agents {frozen.tolist()} stationary under nonzero commands"
                )
