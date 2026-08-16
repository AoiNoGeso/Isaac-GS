from __future__ import annotations

import numpy as np


def derive_outcome(info: dict) -> tuple[bool, bool, bool, bool]:
    """envのinfoから(success, wall_collision, human_collision, timeout)を導出する"""
    success = bool(info.get("success", False))
    human_collision = bool(info.get("human_collision", False))
    wall_collision = bool(info.get("collision", False)) and not human_collision
    timeout = bool(info.get("timeout", False))
    return success, wall_collision, human_collision, timeout


def compute_spl(success: bool, init_dist: float, path_len: float) -> float:
    """SPL(Success weighted by Path Length)を計算する"""
    if init_dist <= 0:
        return 0.0
    return float(success) * (init_dist / max(path_len, init_dist))


def outcome_suffix(success: bool, wall_collision: bool, human_collision: bool) -> str:
    """動画ファイル名末尾に付ける結果サフィックス(成功/人物衝突/壁衝突/タイムアウト)"""
    if success:
        return "_s"
    if human_collision:
        return "_h"
    if wall_collision:
        return "_w"
    return "_t"


class EpisodeTracker:
    """学習ループ用。1エピソード分の統計(SPL含む)を蓄積し、wandbログ用の辞書を返す"""

    def __init__(self):
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_path_len = 0.0
        self._ep_init_dist = 0.0
        self._prev_xy = None
        self._ep_count = 0
        self._total_success = 0
        self._total_wall_collision = 0
        self._total_human_collision = 0
        self._total_timeout = 0

    def step(self, reward: float, info: dict):
        """1ステップ分の報酬・移動距離を積算する"""
        self._ep_reward += reward
        self._ep_steps += 1
        xy = info.get("robot_xy")
        if xy is not None:
            xy = np.array(xy, dtype=np.float32)
            if self._prev_xy is not None:
                self._ep_path_len += float(np.linalg.norm(xy - self._prev_xy))
            self._prev_xy = xy

    def reset(self, obs: dict, info: dict | None = None):
        """新しいエピソード開始時に内部状態を初期化する"""
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_path_len = 0.0
        self._prev_xy = None
        if "goal" in obs:
            self._ep_init_dist = float(obs["goal"][0])
        elif info is not None and "dist" in info:
            self._ep_init_dist = float(info["dist"])

    def finish(self, info: dict) -> dict:
        """エピソード終了時に統計をまとめてwandbログ用の辞書を返す"""
        self._ep_count += 1
        success, wall_collision, human_collision, timeout = derive_outcome(info)

        self._total_success += int(success)
        self._total_wall_collision += int(wall_collision)
        self._total_human_collision += int(human_collision)
        self._total_timeout += int(timeout)

        return {
            "episode/reward": self._ep_reward,
            "episode/steps": self._ep_steps,
            "episode/success": float(success),
            "episode/wall_collision": float(wall_collision),
            "episode/human_collision": float(human_collision),
            "episode/timeout": float(timeout),
            "episode/spl": compute_spl(success, self._ep_init_dist, self._ep_path_len),
            "episode/dist_final": float(info.get("dist", 0.0)),
            "episode/success_rate": self._total_success / self._ep_count,
            "episode/wall_collision_rate": self._total_wall_collision / self._ep_count,
            "episode/human_collision_rate": self._total_human_collision / self._ep_count,
            "episode/timeout_rate": self._total_timeout / self._ep_count,
            "episode/count": self._ep_count,
        }


class EvalStats:
    """評価ループ用。複数エピソードの結果を集計する
    (train.pyのバリデーションログとtest.pyの結果表示が共用する)"""

    def __init__(self):
        self.episodes = 0
        self.successes = 0
        self.wall_collisions = 0
        self.human_collisions = 0
        self.timeouts = 0
        self.total_reward = 0.0
        self.total_dist_final = 0.0
        self.total_spl = 0.0

    def add(
        self,
        info: dict,
        ep_reward: float = 0.0,
        init_dist: float = 0.0,
        path_len: float = 0.0,
    ) -> tuple[bool, bool, bool, bool]:
        """1エピソード分を加算し(success, wall_collision, human_collision, timeout)を返す"""
        success, wall_collision, human_collision, timeout = derive_outcome(info)
        self.episodes += 1
        self.successes += int(success)
        self.wall_collisions += int(wall_collision)
        self.human_collisions += int(human_collision)
        self.timeouts += int(timeout)
        self.total_reward += ep_reward
        self.total_dist_final += float(info.get("dist", 0.0))
        self.total_spl += compute_spl(success, init_dist, path_len)
        return success, wall_collision, human_collision, timeout

    @property
    def collisions(self) -> int:
        """壁衝突と人物衝突の合計"""
        return self.wall_collisions + self.human_collisions

    def rates(self, prefix: str = "val/") -> dict:
        """成功率・衝突率・タイムアウト率をwandbログ用の辞書で返す"""
        n = max(self.episodes, 1)
        return {
            f"{prefix}success_rate": self.successes / n,
            f"{prefix}human_collision_rate": self.human_collisions / n,
            f"{prefix}wall_collision_rate": self.wall_collisions / n,
            f"{prefix}timeout_rate": self.timeouts / n,
        }

    def report_lines(self) -> list[str]:
        """人が読む結果表の各行を返す"""
        n = max(self.episodes, 1)
        return [
            f"Episodes      : {self.episodes}",
            f"Success Rate  : {self.successes / n:.3f}  ({self.successes}/{n})",
            f"Collision Rate: {self.collisions / n:.3f}  ({self.collisions}/{n})",
            f"Wall Collision Rate : {self.wall_collisions / n:.3f}  ({self.wall_collisions}/{n})",
            f"Human Collision Rate: {self.human_collisions / n:.3f}  ({self.human_collisions}/{n})",
            f"Timeout Rate  : {self.timeouts / n:.3f}  ({self.timeouts}/{n})",
            f"Avg Reward    : {self.total_reward / n:.2f}",
            f"Avg Dist Final: {self.total_dist_final / n:.3f} m",
            f"SPL           : {self.total_spl / n:.3f}",
        ]
