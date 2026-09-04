"""SAC学習ループ本体。models/sac・models/sac_DINOv3で共通のtrain()/validation()を提供する。

ReplayBufferの遅延スタッキング(store_frame/store_effect/encode_recent_observation)を
前提としたロールアウト構成になっている(models/*/policy.py::ReplayBuffer参照)。
"""

import wandb
from tqdm import tqdm

from utils.metrics import EpisodeTracker
from utils.recorder import EpisodeRecorder
from utils.rollout import evaluate


def validation(env, agent, train_cfg, recorder, step: int, obs_transform, out, stack_size: int = 1) -> dict:
    """greedy方策で評価し成功率・衝突率・タイムアウト率をwandbログ用の辞書で返す"""
    stats = evaluate(
        env,
        agent,
        train_cfg.val_episodes,
        recorder=recorder,
        video_episodes=train_cfg.val_video_episodes,
        stem_fn=lambda ep: f"{step}_{ep}",
        desc="[val]",
        leave=False,
        obs_transform=obs_transform,
        stack_size=stack_size,
    )
    return stats.rates("val/")


def _store_current_frame(env_obs, buffer, obs_transform):
    """env観測を保存し、(保存インデックス, act()用のスタック観測)を返す"""
    model_obs = obs_transform(env_obs) if obs_transform else env_obs
    idx = buffer.store_frame(model_obs)
    return idx, buffer.encode_recent_observation()


def train(
    *,
    env,
    agent,
    buffer,
    tracker: EpisodeTracker,
    obs: dict,
    recorder: EpisodeRecorder | None,
    use_wandb: bool,
    ckpt_dir: str,
    obs_transform,
    train_cfg,
    out,
    stack_size: int = 1,
) -> None:
    """学習ループ本体。ロールアウト・SAC更新・定期バリデーション・チェックポイント保存を行う

    ReplayBufferは1フレームだけを保存し、agent.act()用のスタック観測は
    buffer.encode_recent_observation()から都度取得する(env観測とモデル観測の変換は
    obs_transformで1step1回だけ行い、二重適用を避ける)"""
    metrics: dict = {}
    idx, stacked = _store_current_frame(obs, buffer, obs_transform)
    pbar = tqdm(range(1, train_cfg.total_timesteps + 1), dynamic_ncols=True, file=out)
    for step in pbar:
        if len(buffer) < train_cfg.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.act(stacked, deterministic=False)

        next_env_obs, reward, terminated, truncated, info = env.step(action)
        # タイムアウトによる終了はdone=0として扱う(Bellman targetがnext_obsを使うため、
        # reset前の真の継続フレームを行動選択とは無関係に保存しておく必要がある)
        buffer.store_effect(idx, action, reward, float(terminated))
        if truncated and not terminated:
            next_model_obs = obs_transform(next_env_obs) if obs_transform else next_env_obs
            buffer.store_frame(next_model_obs)
        tracker.step(reward, info)

        if terminated or truncated:
            ep_metrics = tracker.finish(info)
            tqdm.write(
                f"episode={ep_metrics['episode/count']:4d}"
                f"  reward={ep_metrics['episode/reward']:+7.1f}"
                f"  success={ep_metrics['episode/success_rate']:.2f}"
                f"  wall={ep_metrics['episode/wall_collision_rate']:.2f}"
                f"  human={ep_metrics['episode/human_collision_rate']:.2f}"
                f"  timeout={ep_metrics['episode/timeout_rate']:.2f}",
                file=out,
            )
            if use_wandb:
                wandb.log(ep_metrics, step=step)
            env_obs, reset_info = env.reset()
            buffer.reset_episode()
            tracker.reset(env_obs, reset_info)
        else:
            env_obs = next_env_obs

        idx, stacked = _store_current_frame(env_obs, buffer, obs_transform)

        if step % train_cfg.val_interval == 0:
            val_metrics = validation(env, agent, train_cfg, recorder, step, obs_transform, out, stack_size)
            tqdm.write(f"[val] step={step} {val_metrics}", file=out)
            if use_wandb:
                wandb.log(val_metrics, step=step)
            env_obs, reset_info = env.reset()
            buffer.reset_episode()
            tracker.reset(env_obs, reset_info)
            idx, stacked = _store_current_frame(env_obs, buffer, obs_transform)

        if len(buffer) >= train_cfg.learning_starts and step % train_cfg.train_freq == 0:
            metrics = agent.update(buffer)
            if use_wandb and step % train_cfg.log_interval == 0:
                wandb.log(metrics, step=step)

        if step % train_cfg.checkpoint_interval == 0:
            ckpt_path = f"{ckpt_dir}/sac_{step}.pt"
            agent.save(ckpt_path)
            tqdm.write(f"[train] Checkpoint saved: {ckpt_path}", file=out)

    final_path = f"{train_cfg.log_dir}/sac_final.pt"
    agent.save(final_path)
    print(f"[train] Final model saved: {final_path}")
