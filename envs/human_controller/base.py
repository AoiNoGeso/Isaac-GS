"""歩行モデル(衝突回避アルゴリズム)を差し替え可能にするための共通インターフェース"""

from typing import Protocol, Tuple


class CrowdController(Protocol):
    """複数エージェントの衝突回避後速度を毎ステップ計算するアルゴリズムの共通インターフェース"""

    def add_agent(
        self,
        position: Tuple[float, float],
        radius: float | None = None,
        max_speed: float | None = None,
        velocity: Tuple[float, float] = (0.0, 0.0),
    ) -> int:
        """エージェントを追加し、agent_idを返す"""
        ...

    def set_preferred_velocity(self, agent_id: int, velocity: Tuple[float, float]) -> None:
        """このエージェントが(他者を無視すれば)本来出したい速度を設定する"""
        ...

    def set_position(self, agent_id: int, position: Tuple[float, float]) -> None:
        """外部(モーション生成側)の実位置とコントローラ内部の位置を同期させる"""
        ...

    def set_velocity(self, agent_id: int, velocity: Tuple[float, float]) -> None:
        """外部(モーション生成側)の実速度をコントローラ内部の速度推定に同期させる
        (近傍エージェントからの回避計算に使われる)"""
        ...

    def step(self) -> None:
        """1ステップ分、全エージェントの衝突回避後速度を計算する"""
        ...

    def get_velocity(self, agent_id: int) -> Tuple[float, float]:
        """衝突回避後の目標速度ベクトル(次ステップに使うべき速度)"""
        ...

    def get_position(self, agent_id: int) -> Tuple[float, float]:
        """コントローラ内部で管理している位置(set_positionで同期している前提)"""
        ...
