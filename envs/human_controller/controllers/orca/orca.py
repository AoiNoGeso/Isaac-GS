"""ORCA(Optimal Reciprocal Collision Avoidance)による衝突回避。RVO2
(https://github.com/sybrenstuvel/Python-RVO2, importは`rvo2`)のPythonバインディングをラップし、
`envs/human_controller/base.py`のCrowdController契約を満たす。

インストール:
  uv run python -m pip install --no-build-isolation "git+https://github.com/sybrenstuvel/Python-RVO2.git"
"""

from dataclasses import dataclass
from typing import Tuple

import rvo2


@dataclass
class ORCAConfig:
    time_step: float = 1.0 / 30.0
    neighbor_dist: float = 3.0  # 近傍探索半径(m)
    max_neighbors: int = 10
    time_horizon: float = 2.0  # 他エージェントとの衝突を先読みする時間(s)
    time_horizon_obst: float = 1.0  # 静的障害物との衝突を先読みする時間(s)
    radius: float = 0.4  # デフォルトのエージェント半径(m)
    max_speed: float = 1.5  # デフォルトの最大速度(m/s)


class ORCASimulator:
    """複数エージェントの衝突回避を毎ステップ計算するラッパー。

    使い方:
        sim = ORCASimulator(config)
        aid = sim.add_agent((x, y))
        ...
        sim.set_preferred_velocity(aid, (vx, vy))
        sim.step()
        vx, vy = sim.get_velocity(aid)
        x, y = sim.get_position(aid)
    """

    def __init__(self, config: ORCAConfig = None):
        self.config = config or ORCAConfig()
        self._sim = rvo2.PyRVOSimulator(
            self.config.time_step,
            self.config.neighbor_dist,
            self.config.max_neighbors,
            self.config.time_horizon,
            self.config.time_horizon_obst,
            self.config.radius,
            self.config.max_speed,
        )

    def add_agent(
        self,
        position: Tuple[float, float],
        radius: float = None,
        max_speed: float = None,
        velocity: Tuple[float, float] = (0.0, 0.0),
    ) -> int:
        """position/velocityは呼び出し側が定義する2D水平面座標。戻り値はagent_id
        (rvo2が返すhandleをそのままagent_idとして使う)。"""
        return self._sim.addAgent(
            tuple(position),
            self.config.neighbor_dist,
            self.config.max_neighbors,
            self.config.time_horizon,
            self.config.time_horizon_obst,
            radius if radius is not None else self.config.radius,
            max_speed if max_speed is not None else self.config.max_speed,
            tuple(velocity),
        )

    def add_static_obstacle(self, vertices: list):
        """静的障害物(時計回りの頂点列, [(h0,h1), ...])を追加する。
        追加後、必ずprocess_obstacles()を1回呼ぶこと(rvo2の仕様)。"""
        return self._sim.addObstacle([tuple(v) for v in vertices])

    def process_obstacles(self):
        self._sim.processObstacles()

    def set_preferred_velocity(self, agent_id: int, velocity: Tuple[float, float]):
        self._sim.setAgentPrefVelocity(agent_id, tuple(velocity))

    def set_position(self, agent_id: int, position: Tuple[float, float]):
        """外部(物理シミュレーション側)の実位置とORCA内部の位置を同期させたい場合に使う。"""
        self._sim.setAgentPosition(agent_id, tuple(position))

    def step(self):
        self._sim.doStep()

    def get_velocity(self, agent_id: int) -> Tuple[float, float]:
        """衝突回避後の目標速度ベクトル(次ステップに使うべき速度)。"""
        return self._sim.getAgentVelocity(agent_id)

    def get_position(self, agent_id: int) -> Tuple[float, float]:
        """ORCA内部で積分された位置(外部で別途位置管理する場合はset_positionで同期すること)。"""
        return self._sim.getAgentPosition(agent_id)
