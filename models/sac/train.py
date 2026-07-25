"""Point Navigation 学習スクリプト。
実行例: uv run models/sac/train.py --headless [--checkpoint PATH] [--num-humans N]
stage/run_name/log_dirはmodels/sac/config.pyのTrainConfigで実験ごとに書き換える。"""

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--no-wandb", action="store_true", default=False)
parser.add_argument("--num-humans", type=int, default=None)
args = parser.parse_args()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# SimulationApp起動前に検証する(stage未指定なら即座にエラーにする)
from models.sac.config import TrainConfig

train_cfg = TrainConfig(stage="corridor2")
if Path(f"{train_cfg.log_dir}/sac_final.pt").exists():
    print(
        f"[train] 警告: {train_cfg.log_dir}/sac_final.pt が既に存在します。"
        f"このまま実行すると前回の実験結果を上書きします (log_dir/run_nameを変更してください)"
    )

from isaacsim import SimulationApp

app = SimulationApp({
    "headless": args.headless,
    "extra_args": ["--/rtx/scenedb/maxHistoryTransformCount=256"],
})

import omni.log

omni.log.get_log().set_channel_level(
    "omni.physx.plugin", omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE
)

import cv2
import torch
import wandb
from tqdm import tqdm

_OUT = sys.stdout

from envs import PointNavGymEnv
from envs.config import EnvConfig
from models.sac.config import ModelConfig, replay_buffer_spec
from models.sac.network import make_encoder
from models.sac.policy import ReplayBuffer, SACAgent
from utils.metrics import EpisodeTracker
from utils.video import write_frame, write_overhead_frame

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def validation(
    env,
    agent,
    num_episodes: int,
    video_episodes: int = 0,
    env_cfg: EnvConfig | None = None,
    overhead_camera=None,
    video_dir: str | None = None,
    step: int = 0,
) -> dict:
    """greedy方策でnum_episodes回評価し成功率・衝突率・タイムアウト率を返す。
    先頭video_episodes件はローカルに動画を保存する(wandbへは送らない)。"""
    successes = wall_collisions = human_collisions = timeouts = 0
    val_bar = tqdm(range(num_episodes), desc="[val]", leave=False, dynamic_ncols=True, file=_OUT)
    for ep_idx in val_bar:
        record = ep_idx < video_episodes and env_cfg is not None
        video_writer = overhead_writer = None
        video_path = overhead_path = None

        obs, _ = env.reset()

        if record:
            H, W = env_cfg.camera_resolution[1], env_cfg.camera_resolution[0]
            fps = 1.0 / env_cfg.rendering_dt
            out_dir = Path(video_dir) if video_dir else Path("videos") / "val"
            out_dir.mkdir(parents=True, exist_ok=True)
            video_path = out_dir / f"step{step}_ep{ep_idx}.mp4"
            video_writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
            )
            write_frame(video_writer, obs["rgb"])
            if overhead_camera is not None:
                overhead_path = video_path.with_stem(video_path.stem + "_overhead")
                overhead_writer = cv2.VideoWriter(
                    str(overhead_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
                )
                orgb, _ = overhead_camera.get_rgbd()
                write_overhead_frame(overhead_writer, orgb)

        while True:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if video_writer is not None:
                write_frame(video_writer, obs["rgb"])
            if overhead_writer is not None:
                orgb, _ = overhead_camera.get_rgbd()
                write_overhead_frame(overhead_writer, orgb)
            if terminated or truncated:
                break

        if video_writer is not None:
            video_writer.release()
            if overhead_writer is not None:
                overhead_writer.release()

        success, wall_collision, human_collision, timeout = EpisodeTracker.derive_outcome(info)
        successes += int(success)
        human_collisions += int(human_collision)
        wall_collisions += int(wall_collision)
        timeouts += int(timeout)
        val_bar.set_postfix(success=successes, human=human_collisions, wall=wall_collisions)
    n = num_episodes
    return {
        "val/success_rate": successes / n,
        "val/human_collision_rate": human_collisions / n,
        "val/wall_collision_rate": wall_collisions / n,
        "val/timeout_rate": timeouts / n,
    }


def main():
    """環境・エージェント・リプレイバッファを構築し、train()に処理を渡す"""
    model_cfg = ModelConfig()
    env_cfg = EnvConfig.from_preset(train_cfg.stage)
    if args.num_humans is not None:
        env_cfg.num_humans = args.num_humans

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project="Isaac-GS_PointNav",
            name=train_cfg.run_name,
            config={
                "total_timesteps": train_cfg.total_timesteps,
                "input_rgb": model_cfg.input_rgb,
                "input_goal": model_cfg.input_goal,
                "num_humans": env_cfg.num_humans,
                **{
                    f"sac/{k}": v
                    for k, v in train_cfg.model_dump().items()
                    if k
                    in (
                        "buffer_size",
                        "batch_size",
                        "gamma",
                        "tau",
                        "learning_rate",
                        "learning_starts",
                        "train_freq",
                        "gradient_steps",
                        "target_entropy",
                    )
                },
            },
            dir=train_cfg.log_dir,
        )

    env = PointNavGymEnv(env_cfg=env_cfg)
    obs, reset_info = env.reset()

    overhead_camera = None
    if train_cfg.val_video_episodes > 0 and not args.headless:
        from envs.config import get_preset
        from utils.video import make_overhead_camera

        overhead_camera = make_overhead_camera(env_cfg, get_preset(train_cfg.stage))

    obs_spec, obs_dtypes = replay_buffer_spec(model_cfg, env_cfg.camera_resolution)
    action_dim = env.action_space.shape[0]

    buffer = ReplayBuffer(
        capacity=train_cfg.buffer_size,
        obs_spec=obs_spec,
        obs_dtypes=obs_dtypes,
        action_dim=action_dim,
        device=DEVICE,
    )

    agent = SACAgent(
        encoder_factory=lambda: make_encoder(model_cfg, img_size=env_cfg.camera_resolution[0]),
        action_dim=action_dim,
        cfg=train_cfg,
        device=DEVICE,
    )

    if args.checkpoint:
        agent.load(args.checkpoint)

    log_dir = train_cfg.log_dir
    ckpt_dir = f"{log_dir}/checkpoints"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    tracker = EpisodeTracker()
    tracker.reset(obs, reset_info)

    modality_str = (
        f"rgb={model_cfg.input_rgb}, goal={model_cfg.input_goal}, "
        f"humans={env_cfg.num_humans}"
    )
    print(
        f"[train] modality=({modality_str})  device={DEVICE}  total={train_cfg.total_timesteps:,}"
    )

    train(
        env=env,
        agent=agent,
        buffer=buffer,
        tracker=tracker,
        obs=obs,
        reset_info=reset_info,
        train_cfg=train_cfg,
        env_cfg=env_cfg,
        overhead_camera=overhead_camera,
        use_wandb=use_wandb,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
    )

    if use_wandb:
        wandb.finish()
    env.close()
    app.close()


def train(
    env,
    agent: SACAgent,
    buffer: ReplayBuffer,
    tracker: EpisodeTracker,
    obs: dict,
    reset_info: dict,
    train_cfg: TrainConfig,
    env_cfg: EnvConfig,
    overhead_camera,
    use_wandb: bool,
    log_dir: str,
    ckpt_dir: str,
) -> None:
    """学習ループ本体。ロールアウト・SAC更新・定期バリデーション・チェックポイント保存を行う"""
    metrics: dict = {}
    pbar = tqdm(range(1, train_cfg.total_timesteps + 1), dynamic_ncols=True, file=_OUT)
    for step in pbar:
        if len(buffer) < train_cfg.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # タイムアウトによる終了はdone=0として扱う(TD学習でブートストラップを継続)
        buffer.add(obs, action, reward, next_obs, float(terminated))
        tracker.step(reward, info)
        obs = next_obs

        if done:
            ep_metrics = tracker.finish(info)
            tqdm.write(
                f"episode={ep_metrics['episode/count']:4d}"
                f"  reward={ep_metrics['episode/reward']:+7.1f}"
                f"  success={ep_metrics['episode/success_rate']:.2f}"
                f"  wall={ep_metrics['episode/wall_collision_rate']:.2f}"
                f"  human={ep_metrics['episode/human_collision_rate']:.2f}"
                f"  timeout={ep_metrics['episode/timeout_rate']:.2f}",
                file=_OUT,
            )
            if use_wandb:
                wandb.log(ep_metrics, step=step)
            obs, reset_info = env.reset()
            tracker.reset(obs, reset_info)

        if step % train_cfg.val_interval == 0:
            val_metrics = validation(
                env,
                agent,
                train_cfg.val_episodes,
                video_episodes=train_cfg.val_video_episodes,
                env_cfg=env_cfg,
                overhead_camera=overhead_camera,
                video_dir=f"{log_dir}/val_videos",
                step=step,
            )
            tqdm.write(f"[val] step={step} {val_metrics}", file=_OUT)
            if use_wandb:
                wandb.log(val_metrics, step=step)
            obs, reset_info = env.reset()
            tracker.reset(obs, reset_info)

        if (
            len(buffer) >= train_cfg.learning_starts
            and step % train_cfg.train_freq == 0
        ):
            metrics = agent.update(buffer)
            if use_wandb and step % train_cfg.log_interval == 0:
                wandb.log(metrics, step=step)

        if step % train_cfg.checkpoint_interval == 0:
            ckpt_path = f"{ckpt_dir}/sac_{step}.pt"
            agent.save(ckpt_path)
            tqdm.write(f"[train] Checkpoint saved: {ckpt_path}", file=_OUT)

    final_path = f"{log_dir}/sac_final.pt"
    agent.save(final_path)
    print(f"[train] Final model saved: {final_path}")


if __name__ == "__main__":
    main()
