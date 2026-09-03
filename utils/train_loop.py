"""SAC学習ループ本体。models/sac・models/sac_DINOv3で共通のtrain()/validation()を提供する。"""

import wandb
from tqdm import tqdm

from utils.metrics import EpisodeTracker
from utils.recorder import EpisodeRecorder
from utils.rollout import evaluate


def validation(env, agent, train_cfg, recorder, step: int, obs_transform, out) -> dict:
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
    )
    return stats.rates("val/")


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
) -> None:
    """学習ループ本体。ロールアウト・SAC更新・定期バリデーション・チェックポイント保存を行う

    env観測(rgb等)とモデル観測(dino_feat等)を別々に保持する。
    obs_transformの二重適用を避けるため、変換結果は次イテレーションへそのまま持ち回る"""
    metrics: dict = {}
    env_obs = obs
    model_obs = obs_transform(env_obs) if obs_transform else env_obs
    pbar = tqdm(range(1, train_cfg.total_timesteps + 1), dynamic_ncols=True, file=out)
    for step in pbar:
        if len(buffer) < train_cfg.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.act(model_obs, deterministic=False)

        next_env_obs, reward, terminated, truncated, info = env.step(action)
        next_model_obs = obs_transform(next_env_obs) if obs_transform else next_env_obs

        # タイムアウトによる終了はdone=0として扱う
        buffer.add(model_obs, action, reward, next_model_obs, float(terminated))
        tracker.step(reward, info)
        env_obs, model_obs = next_env_obs, next_model_obs

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
            model_obs = obs_transform(env_obs) if obs_transform else env_obs
            tracker.reset(env_obs, reset_info)

        if step % train_cfg.val_interval == 0:
            val_metrics = validation(env, agent, train_cfg, recorder, step, obs_transform, out)
            tqdm.write(f"[val] step={step} {val_metrics}", file=out)
            if use_wandb:
                wandb.log(val_metrics, step=step)
            env_obs, reset_info = env.reset()
            model_obs = obs_transform(env_obs) if obs_transform else env_obs
            tracker.reset(env_obs, reset_info)

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
