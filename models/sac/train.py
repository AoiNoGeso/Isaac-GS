"""Point Navigation 学習スクリプト。
実行例: uv run models/sac/train.py --headless [--checkpoint PATH] [--num-humans N]
stage/run_name/log_dirはmodels/sac/config.pyのTrainConfigで実験ごとに書き換える。"""

import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--no-wandb", action="store_true", default=False)
parser.add_argument("--num-humans", type=int, default=None)
parser.add_argument("--profile", type=int, default=0)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

# CUDA初期化(isaacsim/torch)より前に設定する必要がある。
# 共用マシンで複数GPUが見える場合、これが無いとIsaac Sim側とPyTorch側で
# 異なる物理GPUが選ばれ、`weight is on cuda:N, different from other tensors`
# のようなクラッシュが起きる。
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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
    "extra_args": [
        "--/rtx/scenedb/maxHistoryTransformCount=256",
        "--/renderer/activeGpu=0"
        # "--/app/runLoops/main/rateLimitEnabled=false",
    ],
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
    """greedy方策でnum_episodes回評価し成功率・衝突率・タイムアウト率を返す"""
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
                ow, oh = overhead_camera.resolution
                overhead_path = video_path.with_stem(video_path.stem + "_overhead")
                overhead_writer = cv2.VideoWriter(
                    str(overhead_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh)
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

        success, wall_collision, human_collision, timeout = EpisodeTracker.derive_outcome(info)

        if video_writer is not None:
            video_writer.release()
            if success:
                suffix = "_s"
            elif human_collision:
                suffix = "_h"
            elif wall_collision:
                suffix = "_w"
            else:
                suffix = "_t"
            video_path.rename(video_path.with_stem(video_path.stem + suffix))
            if overhead_writer is not None:
                overhead_writer.release()
                overhead_path.rename(overhead_path.with_stem(overhead_path.stem + suffix))

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


def _print_profile_summary(profiler) -> None:
    """区間ごとの平均所要時間[ms]と全体に占める割合を表示する(--profile用、デバッグ機能)"""
    summary = profiler.summary()
    total = sum(summary.values())
    print(f"\n{'=' * 50}\n[profile] 区間別の平均所要時間 (1step換算)\n{'=' * 50}")
    for name, avg_ms in sorted(summary.items(), key=lambda kv: -kv[1]):
        pct = (avg_ms / total * 100.0) if total > 0 else 0.0
        print(f"  {name:16s}: {avg_ms:8.3f} ms  ({pct:5.1f}%)")
    print(f"  {'合計':16s}: {total:8.3f} ms")
    print(f"{'=' * 50}\n")


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

    profiler = None
    if args.profile > 0:
        from tests.manual.profiling import StepProfiler

        profiler = StepProfiler()

    env = PointNavGymEnv(env_cfg=env_cfg, profiler=profiler)
    obs, reset_info = env.reset()

    overhead_camera = None
    if train_cfg.val_video_episodes > 0 and not args.headless:
        from envs.config import get_preset
        from utils.video import make_overhead_camera

        overhead_camera = make_overhead_camera(get_preset(train_cfg.stage))

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
        profiler=profiler,
        profile_steps=args.profile,
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
    profiler=None,
    profile_steps: int = 0,
) -> None:
    """学習ループ本体。ロールアウト・SAC更新・定期バリデーション・チェックポイント保存を行う。
    profile_steps>0の場合はそのstep数だけ実行し、区間ごとの計測結果を表示して終了する"""
    from contextlib import nullcontext

    def span(name: str):
        return profiler.span(name) if profiler is not None else nullcontext()

    metrics: dict = {}
    total_steps = profile_steps if profile_steps > 0 else train_cfg.total_timesteps
    # profile_steps指定時はlearning_startsを待たずagent.update()の計測も取れるようにする
    learning_starts = min(train_cfg.learning_starts, 10) if profile_steps > 0 else train_cfg.learning_starts
    pbar = tqdm(range(1, total_steps + 1), dynamic_ncols=True, file=_OUT)
    for step in pbar:
        with span("agent_act"):
            if len(buffer) < learning_starts:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # タイムアウトによる終了はdone=0として扱う
        with span("buffer_add"):
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

        if len(buffer) >= learning_starts and step % train_cfg.train_freq == 0:
            with span("agent_update"):
                metrics = agent.update(buffer)
            if use_wandb and step % train_cfg.log_interval == 0:
                wandb.log(metrics, step=step)

        if not profile_steps and step % train_cfg.checkpoint_interval == 0:
            ckpt_path = f"{ckpt_dir}/sac_{step}.pt"
            agent.save(ckpt_path)
            tqdm.write(f"[train] Checkpoint saved: {ckpt_path}", file=_OUT)

    if profile_steps:
        _print_profile_summary(profiler)
        return

    final_path = f"{log_dir}/sac_final.pt"
    agent.save(final_path)
    print(f"[train] Final model saved: {final_path}")


if __name__ == "__main__":
    main()
