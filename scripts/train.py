"""Point Navigation 学習スクリプト(モデル共通)。
実行例:
  uv run scripts/train.py --model sac --headless
  uv run scripts/train.py --model sac_DINOv3 --headless [--checkpoint PATH] [--num-humans N]
stage/run_name/log_dirはmodels/<model>/config.pyのTrainConfigで実験ごとに書き換える。"""

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_MODELS = sorted(p.name for p in (ROOT / "models").iterdir() if (p / "config.py").exists())

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=_MODELS, required=True)
parser.add_argument("--stage", type=str, default="corridor2")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--no-wandb", action="store_true", default=False)
parser.add_argument("--num-humans", type=int, default=None)
parser.add_argument("--gpu", type=int, default=0, help="共用マシンでの複数GPU分散を防ぐため、使用するGPU番号を固定する")
parser.add_argument("--total-timesteps", type=int, default=None)
parser.add_argument("--run-name", type=str, default=None)
parser.add_argument("--log-dir", type=str, default=None)
args = parser.parse_args()

# SimulationApp起動前に検証する(stage未指定なら即座にエラーにする)
# launch_sim()前: pydantic/numpyのみに依存するconfigだけを引き込む(models/*/__init__.pyは空)
mcfg = importlib.import_module(f"models.{args.model}.config")

_overrides = {
    k: v
    for k, v in (
        ("total_timesteps", args.total_timesteps),
        ("run_name", args.run_name),
        ("log_dir", args.log_dir),
    )
    if v is not None
}
train_cfg = mcfg.TrainConfig(stage=args.stage, **_overrides)
if Path(f"{train_cfg.log_dir}/sac_final.pt").exists():
    print(
        f"[train] 警告: {train_cfg.log_dir}/sac_final.pt が既に存在します。"
        f"このまま実行すると前回の実験結果を上書きします (log_dir/run_nameを変更してください)"
    )

_OUT = sys.stdout  # launch_sim()前のstdoutを捕捉(tqdm.writeの出力先)
assert "torch" not in sys.modules, "launch_sim()前にtorchがimportされている"

# wandbのコンソールキャプチャはinit()呼び出し時点のsys.stdout/stderrをラップする。
# launch_sim()(IsaacSimのSimulationApp)はKit自身のロギング機構に繋ぐためグローバルな
# sys.stdout/stderrを差し替えるので、launch_sim()より後にwandb.init()すると
# 上記_OUT(差し替え前の本物のstdout)と別物をラップしてしまい、_OUTへ書いているtqdmの
# 出力がwandbの「Logs」タブに一切届かなくなる。必ずlaunch_sim()より前にinit()すること
# (env構築後にしか分からないconfig項目は後段でwandb.config.update()する)。
import wandb

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
use_wandb = not args.no_wandb
if use_wandb:
    wandb.init(
        project=train_cfg.project_name,
        name=train_cfg.run_name,
        config={
            "total_timesteps": train_cfg.total_timesteps,
            **({"num_humans": args.num_humans} if args.num_humans is not None else {}),
            **{f"sac/{k}": getattr(train_cfg, k) for k in _LOGGED_SAC_KEYS},
        },
        dir=train_cfg.log_dir,
    )

from utils.launch_sim import launch_sim

app = launch_sim(headless=args.headless, gpu=args.gpu)

# ここから先はtorch依存モジュールをimportしてよい
import torch

from envs import PointNavGymEnv
from envs.config import EnvConfig, get_preset
from utils.metrics import EpisodeTracker
from utils.recorder import EpisodeRecorder, make_overhead_camera, robot_resolution_from_space
from utils.train_loop import train

mnet = importlib.import_module(f"models.{args.model}.network")
mpol = importlib.import_module(f"models.{args.model}.policy")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    """環境・エージェント・リプレイバッファを構築し、train()に処理を渡す"""
    env_cfg = EnvConfig.from_preset(train_cfg.stage, **mcfg.env_overrides())
    if args.num_humans is not None:
        env_cfg.num_humans = args.num_humans

    # observation_spaceは__init__で確定するため、重いreset()を待たずに参照できる
    env = PointNavGymEnv(env_cfg=env_cfg)
    model_obs_space, obs_transform = mnet.build_obs_pipeline(env.observation_space, DEVICE)

    if use_wandb:
        # env構築後にしか分からない項目をここで追記する(wandb.init()自体はlaunch_sim()より前、
        # ファイル先頭で済ませてある。理由は_OUT/wandbコンソールキャプチャのコメント参照)
        wandb.config.update({
            "obs_keys": list(model_obs_space.spaces),
            "stack_size": train_cfg.stack_size,
            "num_humans": env_cfg.num_humans,
        })

    obs, reset_info = env.reset()

    recorder = None
    if train_cfg.val_video_episodes > 0:
        # ロボット搭載カメラと同じrep.create.render_product機構のため俯瞰カメラもheadlessで動作する
        recorder = EpisodeRecorder(
            out_dir=f"{train_cfg.log_dir}/val_videos",
            # 動画1フレーム=env.step()1回(decimation物理サブステップぶんの時間経過)なので、
            # fpsはrendering_dtではなくenv.step()の呼び出し頻度に合わせる
            fps=1.0 / (env_cfg.physics_dt * env_cfg.decimation),
            robot_resolution=robot_resolution_from_space(env.observation_space),
            overhead_camera=make_overhead_camera(get_preset(train_cfg.stage)),
        )

    obs_spec, obs_dtypes = mcfg.replay_buffer_spec(model_obs_space)
    action_dim = env.action_space.shape[0]

    buffer = mpol.ReplayBuffer(
        capacity=train_cfg.buffer_size,
        obs_spec=obs_spec,
        obs_dtypes=obs_dtypes,
        action_dim=action_dim,
        device=DEVICE,
        stack_size=train_cfg.stack_size,
    )

    agent = mpol.SACAgent(
        encoder_factory=lambda: mnet.make_encoder(model_obs_space, stack_size=train_cfg.stack_size),
        action_dim=action_dim,
        cfg=train_cfg,
        device=DEVICE,
    )

    if args.checkpoint:
        print("[train] 警告: optimizer stateは復元されません(target networkは再構築されます)")
        agent.load(args.checkpoint)

    ckpt_dir = f"{train_cfg.log_dir}/checkpoints"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    tracker = EpisodeTracker()
    tracker.reset(obs, reset_info)

    print(
        f"[train] obs_keys={list(model_obs_space.spaces)}  "
        f"humans={env_cfg.num_humans}  device={DEVICE}  total={train_cfg.total_timesteps:,}"
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
        obs_transform=obs_transform,
        train_cfg=train_cfg,
        out=_OUT,
        stack_size=train_cfg.stack_size,
    )

    if use_wandb:
        wandb.finish()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
