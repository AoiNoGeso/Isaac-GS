"""
Point Navigation 学習スクリプト
IsaacSim 6.0 + 自前 SAC + wandb ログ

実行方法:
  cd ~/Programs/Isaac-GS
  uv run tasks/point_navigation/train.py --headless
  uv run tasks/point_navigation/train.py --headless --run-name my_run
  uv run tasks/point_navigation/train.py --headless --checkpoint runs/point_nav/checkpoints/sac_10000.pt
"""

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--run-name", type=str, default=None)
parser.add_argument("--no-wandb", action="store_true", default=False)
args = parser.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": args.headless})

import omni.log

omni.log.get_log().set_channel_level(
    "omni.physx.plugin", omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE
)

import numpy as np
import torch
import wandb
from tqdm import tqdm

# Isaac Sim が stderr を横取りするため stdout に固定
_OUT = sys.stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.gym_wrapper import PointNavGymEnv
from tasks.point_navigation.config import PointNavEnvCfg, PointNavTrainCfg
from tasks.point_navigation.policy.network import PointNavEncoder
from tasks.point_navigation.policy.policy import ReplayBuffer, SACAgent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# エピソード統計トラッカー
# ---------------------------------------------------------------------------


class EpisodeTracker:
    def __init__(self, window: int = 100):
        self._window = window
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_path_len = 0.0
        self._ep_init_dist = 0.0
        self._prev_xy = None
        self._ep_count = 0
        self._total_success = 0
        self._total_collision = 0
        self._success_buf: list[float] = []
        self._collision_buf: list[float] = []

    def step(self, reward: float, info: dict):
        self._ep_reward += reward
        self._ep_steps += 1
        xy = info.get("robot_xz")
        if xy is not None:
            xy = np.array(xy, dtype=np.float32)
            if self._prev_xy is not None:
                self._ep_path_len += float(np.linalg.norm(xy - self._prev_xy))
            self._prev_xy = xy

    def reset(self, obs: dict):
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_path_len = 0.0
        self._prev_xy = None
        if "goal" in obs:
            self._ep_init_dist = float(obs["goal"][0])  # dist [m]

    def finish(self, info: dict) -> dict:
        self._ep_count += 1
        success = bool(info.get("success", False))
        collision = bool(info.get("collision", False))
        timeout = bool(info.get("timeout", False))

        self._total_success += int(success)
        self._total_collision += int(collision)
        self._success_buf.append(float(success))
        self._collision_buf.append(float(collision))
        if len(self._success_buf) > self._window:
            self._success_buf.pop(0)
        if len(self._collision_buf) > self._window:
            self._collision_buf.pop(0)

        l = self._ep_init_dist
        p = self._ep_path_len
        spl = float(success) * (l / max(p, l)) if l > 0 else 0.0

        return {
            "episode/reward": self._ep_reward,
            "episode/steps": self._ep_steps,
            "episode/success": float(success),
            "episode/collision": float(collision),
            "episode/timeout": float(timeout),
            "episode/spl": spl,
            "episode/dist_final": float(info.get("dist", 0.0)),
            "episode/success_rate": self._total_success / self._ep_count,
            "episode/collision_rate": self._total_collision / self._ep_count,
            "episode/count": self._ep_count,
        }


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------


def main():
    train_cfg = PointNavTrainCfg()
    if args.run_name is not None:
        train_cfg.run_name = args.run_name

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project="isaac-gs-point-nav",
            name=train_cfg.run_name,
            config={
                "total_timesteps": train_cfg.total_timesteps,
                "input_rgb": train_cfg.input_rgb,
                "input_goal": train_cfg.input_goal,
                **{f"sac/{k}": v for k, v in train_cfg.sac.model_dump().items()},
            },
            dir=train_cfg.log_dir,
        )

    # ── 環境 ─────────────────────────────────────────────────────────────
    env_cfg = PointNavEnvCfg(
        input_rgb=train_cfg.input_rgb,
        input_goal=train_cfg.input_goal,
    )

    env = PointNavGymEnv(cfg=env_cfg)
    obs, _ = env.reset()

    # ── obs_spec（バッファ構築用）────────────────────────────────────────
    obs_spec = {k: v.shape for k, v in env.observation_space.spaces.items()}
    action_dim = env.action_space.shape[0]

    # ── リプレイバッファ ──────────────────────────────────────────────────
    buffer = ReplayBuffer(
        capacity=train_cfg.sac.buffer_size,
        obs_spec=obs_spec,
        action_dim=action_dim,
        device=DEVICE,
    )

    # ── SAC エージェント ──────────────────────────────────────────────────
    img_size = env_cfg.camera_resolution[0]

    def encoder_factory():
        return PointNavEncoder(
            input_rgb=train_cfg.input_rgb,
            input_goal=train_cfg.input_goal,
            img_size=img_size,
        )

    agent = SACAgent(
        encoder_factory=encoder_factory,
        action_dim=action_dim,
        cfg=train_cfg.sac,
        device=DEVICE,
    )

    if args.checkpoint:
        agent.load(args.checkpoint)

    # ── 学習ループ ────────────────────────────────────────────────────────
    log_dir  = train_cfg.log_dir
    ckpt_dir = f"{log_dir}/checkpoints"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    tracker = EpisodeTracker(window=100)
    tracker.reset(obs)

    modality_str = f"rgb={train_cfg.input_rgb}, goal={train_cfg.input_goal}"
    print(
        f"[train] modality=({modality_str})  device={DEVICE}  total={train_cfg.total_timesteps:,}"
    )

    metrics: dict = {}
    pbar = tqdm(range(1, train_cfg.total_timesteps + 1), dynamic_ncols=True, file=_OUT)
    for step in pbar:
        # 行動選択（warmup 中はランダム）
        if len(buffer) < train_cfg.sac.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # バッファに格納（timeout による truncation では done=0 とする）
        buffer.add(obs, action, reward, next_obs, float(terminated))
        tracker.step(reward, info)
        obs = next_obs

        # エピソード終了
        if done:
            ep_metrics = tracker.finish(info)
            tqdm.write(
                f"episode={ep_metrics['episode/count']:4d}"
                f"  reward={ep_metrics['episode/reward']:+7.1f}"
                f"  success={ep_metrics['episode/success_rate']:.2f}"
                f"  collision={ep_metrics['episode/collision_rate']:.2f}",
                file=_OUT,
            )
            if use_wandb:
                wandb.log(ep_metrics, step=step)
            obs, _ = env.reset()
            tracker.reset(obs)

        # 学習ステップ
        if (
            len(buffer) >= train_cfg.sac.learning_starts
            and step % train_cfg.sac.train_freq == 0
        ):
            metrics = agent.update(buffer)
            if use_wandb and step % train_cfg.log_interval == 0:
                wandb.log(metrics, step=step)

        # チェックポイント保存
        if step % train_cfg.checkpoint_interval == 0:
            ckpt_path = f"{ckpt_dir}/sac_{step}.pt"
            agent.save(ckpt_path)
            tqdm.write(f"[train] Checkpoint saved: {ckpt_path}", file=_OUT)

    # ── 最終保存 ─────────────────────────────────────────────────────────
    final_path = f"{log_dir}/sac_final.pt"
    agent.save(final_path)
    print(f"[train] Final model saved: {final_path}")

    if use_wandb:
        wandb.finish()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
