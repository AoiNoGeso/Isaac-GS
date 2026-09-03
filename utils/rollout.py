"""greedy方策による評価ロールアウト。train.pyのバリデーションとtest.pyの評価が共用する"""

from __future__ import annotations

import sys
from typing import Callable

import numpy as np
from tqdm import tqdm

from utils.metrics import EvalStats, outcome_suffix
from utils.recorder import EpisodeRecorder


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
) -> EvalStats:
    """決定論的方策でnum_episodes回エピソードを回し、集計結果を返す

    recorderを渡すと先頭video_episodes本だけ動画を収録する(ファイル名はstem_fn(ep)で決める)
    obs_transformを渡すとenv観測をモデル観測へ変換してagent.act()へ渡す(recorder/goal参照はenv観測のまま)
    """
    stats = EvalStats()
    bar = tqdm(range(num_episodes), desc=desc, leave=leave, dynamic_ncols=True, file=sys.stdout)

    for ep in bar:
        env_obs, _ = env.reset()
        model_obs = obs_transform(env_obs) if obs_transform else env_obs
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
            model_obs = obs_transform(env_obs) if obs_transform else env_obs
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
