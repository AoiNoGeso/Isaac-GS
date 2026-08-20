"""Point Navigation 学習スクリプト(DINOv3視覚エンコーダ版)。
実行例: uv run models/sac_DINOv3/train.py --headless [--checkpoint PATH] [--num-humans N]
stage/run_name/log_dirはmodels/sac_DINOv3/config.pyのTrainConfigで実験ごとに書き換える。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--no-wandb", action="store_true", default=False)
parser.add_argument("--num-humans", type=int, default=None)
parser.add_argument("--gpu", type=int, default=0, help="共用マシンでの複数GPU分散を防ぐため、使用するGPU番号を固定する")
args = parser.parse_args()

# SimulationApp起動前に検証する(stage未指定なら即座にエラーにする)
from models.sac_DINOv3.config import TrainConfig

train_cfg = TrainConfig(stage="corridor2")
if Path(f"{train_cfg.log_dir}/sac_final.pt").exists():
    print(
        f"[train] 警告: {train_cfg.log_dir}/sac_final.pt が既に存在します。"
        f"このまま実行すると前回の実験結果を上書きします (log_dir/run_nameを変更してください)"
    )

from utils.launch_sim import launch_sim

app = launch_sim(headless=args.headless, gpu=args.gpu)

import torch
import wandb
from tqdm import tqdm

_OUT = sys.stdout

from envs import PointNavGymEnv
from envs.config import EnvConfig, get_preset
from models.sac_DINOv3.config import ModelConfig, replay_buffer_spec
from models.sac_DINOv3.dino_backbone import DINOBackbone, DINOEnvWrapper
from models.sac_DINOv3.network import make_encoder
from models.sac_DINOv3.policy import ReplayBuffer, SACAgent
from utils.metrics import EpisodeTracker
from utils.recorder import EpisodeRecorder, make_overhead_camera
from utils.rollout import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# wandbのconfigに残すSACハイパーパラメータ
_LOGGED_SAC_KEYS = (
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


def validation(env, agent, train_cfg: TrainConfig, recorder, step: int) -> dict:
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
    )
    return stats.rates("val/")


def main():
    """環境・エージェント・リプレイバッファを構築し、train()に処理を渡す"""
    model_cfg = ModelConfig()
    backbone = DINOBackbone(device=DEVICE)
    # DINOv3の学習解像度(224x224)にIsaacSim側のレンダリング解像度を合わせる
    env_cfg = EnvConfig.from_preset(
        train_cfg.stage, camera_resolution=(backbone.image_size, backbone.image_size)
    )
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
                **{f"sac/{k}": getattr(train_cfg, k) for k in _LOGGED_SAC_KEYS},
            },
            dir=train_cfg.log_dir,
        )

    env = DINOEnvWrapper(PointNavGymEnv(env_cfg=env_cfg), backbone)
    obs, reset_info = env.reset()

    recorder = None
    if train_cfg.val_video_episodes > 0:
        # ロボット搭載カメラと同じrep.create.render_product機構のため俯瞰カメラもheadlessで動作する
        recorder = EpisodeRecorder(
            out_dir=f"{train_cfg.log_dir}/val_videos",
            # 動画1フレーム=env.step()1回(decimation物理サブステップぶんの時間経過)なので、
            # fpsはrendering_dtではなくenv.step()の呼び出し頻度に合わせる
            fps=1.0 / (env_cfg.physics_dt * env_cfg.decimation),
            robot_resolution=env_cfg.camera_resolution,
            overhead_camera=make_overhead_camera(get_preset(train_cfg.stage)),
        )

    obs_spec, obs_dtypes = replay_buffer_spec(model_cfg, backbone.hidden_size, backbone.grid_size)
    action_dim = env.action_space.shape[0]

    buffer = ReplayBuffer(
        capacity=train_cfg.buffer_size,
        obs_spec=obs_spec,
        obs_dtypes=obs_dtypes,
        action_dim=action_dim,
        device=DEVICE,
    )

    agent = SACAgent(
        encoder_factory=lambda: make_encoder(model_cfg, backbone.hidden_size, backbone.grid_size),
        action_dim=action_dim,
        cfg=train_cfg,
        device=DEVICE,
    )

    if args.checkpoint:
        agent.load(args.checkpoint)

    ckpt_dir = f"{train_cfg.log_dir}/checkpoints"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    tracker = EpisodeTracker()
    tracker.reset(obs, reset_info)

    print(
        f"[train] modality=(rgb={model_cfg.input_rgb}, goal={model_cfg.input_goal}, "
        f"humans={env_cfg.num_humans})  device={DEVICE}  total={train_cfg.total_timesteps:,}"
    )

    train(
        env=env,
        agent=agent,
        buffer=buffer,
        tracker=tracker,
        obs=obs,
        recorder=recorder,
        use_wandb=use_wandb,
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
    recorder: EpisodeRecorder | None,
    use_wandb: bool,
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

        # タイムアウトによる終了はdone=0として扱う
        buffer.add(obs, action, reward, next_obs, float(terminated))
        tracker.step(reward, info)
        obs = next_obs

        if terminated or truncated:
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
            val_metrics = validation(env, agent, train_cfg, recorder, step)
            tqdm.write(f"[val] step={step} {val_metrics}", file=_OUT)
            if use_wandb:
                wandb.log(val_metrics, step=step)
            obs, reset_info = env.reset()
            tracker.reset(obs, reset_info)

        if len(buffer) >= train_cfg.learning_starts and step % train_cfg.train_freq == 0:
            metrics = agent.update(buffer)
            if use_wandb and step % train_cfg.log_interval == 0:
                wandb.log(metrics, step=step)

        if step % train_cfg.checkpoint_interval == 0:
            ckpt_path = f"{ckpt_dir}/sac_{step}.pt"
            agent.save(ckpt_path)
            tqdm.write(f"[train] Checkpoint saved: {ckpt_path}", file=_OUT)

    final_path = f"{train_cfg.log_dir}/sac_final.pt"
    agent.save(final_path)
    print(f"[train] Final model saved: {final_path}")


if __name__ == "__main__":
    main()
