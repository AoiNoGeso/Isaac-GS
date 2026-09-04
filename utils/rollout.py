"""greedy方策による評価ロールアウト。train.pyのバリデーションとtest.pyの評価が共用する"""

from __future__ import annotations

import sys
from collections import deque
from typing import Callable

import numpy as np
from tqdm import tqdm

from utils.metrics import EvalStats, outcome_suffix
from utils.recorder import EpisodeRecorder

# ReplayBuffer.encode_recent_observation()相当を、バッファを持たないeval専用ロールアウトで
# 再現するために結合対象とするキー(models/*/policy.py::ReplayBuffer._STACK_KEYと合わせる)
_STACKABLE_KEYS = ("rgb", "dino_feat")


class _FrameHistory:
    """act()用のスタック観測を組み立てる軽量ヘルパー(stack_size=1なら実質no-op)。
    不足時は先頭フレームの繰り返しでパディングする(ObservationManagerと同じ規則)"""

    def __init__(self, stack_size: int):
        self._n = stack_size
        self._hist: dict[str, deque] = {}

    def reset(self, obs: dict) -> dict:
        if self._n == 1:
            return obs
        self._hist = {k: deque([obs[k]] * self._n, maxlen=self._n) for k in _STACKABLE_KEYS if k in obs}
        return self._stacked(obs)

    def push(self, obs: dict) -> dict:
        if self._n == 1:
            return obs
        for k, hist in self._hist.items():
            hist.append(obs[k])
        return self._stacked(obs)

    def _stacked(self, obs: dict) -> dict:
        out = dict(obs)
        for k, hist in self._hist.items():
            out[k] = np.concatenate(list(hist), axis=0)
        return out


def evaluate(
    env,
    agent,
    num_episodes: int,
    recorder: EpisodeRecorder | None = None,
    video_episodes: int = 0,
    stem_fn: Callable[[int], str] | None = None,
    desc: str | None = None,
    leave: bool = True,
    obs_transform: Callable[[dict], dict] | None = None,
    stack_size: int = 1,
) -> EvalStats:
    """決定論的方策でnum_episodes回エピソードを回し、集計結果を返す

    recorderを渡すと先頭video_episodes本だけ動画を収録する(ファイル名はstem_fn(ep)で決める)
    obs_transformを渡すとenv観測をモデル観測へ変換してagent.act()へ渡す(recorder/goal参照はenv観測のまま)
    stack_sizeはact()へ渡す観測の直近フレーム結合数(学習時のReplayBufferと揃えること)
    """
    stats = EvalStats()
    bar = tqdm(range(num_episodes), desc=desc, leave=leave, dynamic_ncols=True, file=sys.stdout)

    for ep in bar:
        env_obs, _ = env.reset()
        history = _FrameHistory(stack_size)
        model_obs = history.reset(obs_transform(env_obs) if obs_transform else env_obs)
        recording = recorder is not None and ep < video_episodes
        if recording:
            recorder.start(stem_fn(ep) if stem_fn is not None else f"ep{ep:04d}", env_obs)

        init_dist = float(env_obs["goal"][0]) if "goal" in env_obs else 0.0
        ep_reward = 0.0
        path_len = 0.0
        prev_xy = None

        while True:
            action = agent.act(model_obs, deterministic=True)
            env_obs, reward, terminated, truncated, info = env.step(action)
            model_obs = history.push(obs_transform(env_obs) if obs_transform else env_obs)
            ep_reward += reward
            if recording:
                recorder.capture(env_obs)

            xy = info.get("robot_xy")
            if xy is not None:
                if prev_xy is not None:
                    path_len += float(np.linalg.norm(np.asarray(xy) - np.asarray(prev_xy)))
                prev_xy = xy

            if terminated or truncated:
                break

        success, wall_collision, human_collision, _ = stats.add(
            info, ep_reward=ep_reward, init_dist=init_dist, path_len=path_len
        )
        if recording:
            recorder.finish(outcome_suffix(success, wall_collision, human_collision))

        bar.set_postfix(
            success=f"{stats.successes}/{stats.episodes}",
            wall=f"{stats.wall_collisions}/{stats.episodes}",
            human=f"{stats.human_collisions}/{stats.episodes}",
        )

    return stats
