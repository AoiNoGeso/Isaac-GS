"""Point Navigation テストスクリプト(DINOv3視覚エンコーダ版)。
指定ディレクトリ内の全チェックポイント(sac_{step}.pt, sac_final.pt)を、
Isaac Simを1度だけ起動した単一プロセス内で順に評価し、wandbの同一runへ
stepごとにログする(run_test.shから呼び出す想定)。

実行例:
    uv run models/sac_DINOv3/test.py \
        --checkpoint-dir runs_forSI/corridor2/S1-RGB+G/0824/checkpoints \
        --log-dir runs_forSI/corridor2/S1-RGB+G/0824/test/S1/log \
        --headless \
        --stage-index 2 --num-humans 1 --video --video-dir runs_forSI/corridor2/S1-RGB+G/0824/test/S1/test_videos
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# envs/models配下はtorchを巻き込むためimportしない
# (launch_sim()がCUDA_VISIBLE_DEVICESを設定するより前にCUDAが初期化されるのを防ぐ)
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument(
    "--checkpoint-dir", type=str, required=True, help="評価するチェックポイント群が入ったディレクトリ"
)
parser.add_argument(
    "--log-dir", type=str, required=True, help="結果ログ・wandbデータの出力先ディレクトリ"
)
parser.add_argument(
    "--stage-index", type=int, default=0, help="評価するステージのインデックス"
)
parser.add_argument("--num-humans", type=int, default=0)
parser.add_argument(
    "--video", action="store_true", default=False, help="各エピソードの映像をmp4で保存する"
)
parser.add_argument(
    "--video-dir", type=str, default="videos", help="動画の出力先ディレクトリ (default: videos)"
)
parser.add_argument("--gpu", type=int, default=0, help="共用マシンでの複数GPU分散を防ぐため、使用するGPU番号を固定する")
parser.add_argument("--no-wandb", action="store_true", default=False)
parser.add_argument(
    "--run-name", type=str, default=None, help="wandbのrun名 (省略時は checkpoint-dir名+日時から自動生成)"
)
args = parser.parse_args()

from utils.launch_sim import launch_sim

app = launch_sim(headless=args.headless, gpu=args.gpu)

import torch
from tqdm import tqdm

_OUT = sys.stdout

from envs import PointNavGymEnv
from envs.config import EnvConfig, get_preset, stage_names
from models.sac_DINOv3.config import TestConfig, TrainConfig
from models.sac_DINOv3.dino_backbone import get_backbone
from models.sac_DINOv3.network import build_obs_pipeline, make_encoder
from models.sac_DINOv3.policy import SACAgent
from utils.recorder import EpisodeRecorder, make_overhead_camera
from utils.rollout import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_CKPT_STEP_RE = re.compile(r"sac_(\d+)\.pt$")


def discover_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    """checkpoint_dir内の sac_{step}.pt / sac_final.pt を step 昇順で列挙する。

    sac_final.pt はwandb上でstep軸の続きに置けるよう、(最大step + 1)として扱う。
    """
    numbered: list[tuple[int, Path]] = []
    final_path = None
    for path in checkpoint_dir.glob("sac_*.pt"):
        m = _CKPT_STEP_RE.match(path.name)
        if m:
            numbered.append((int(m.group(1)), path))
        elif path.name == "sac_final.pt":
            final_path = path
    numbered.sort(key=lambda item: item[0])
    if final_path is not None:
        last_step = numbered[-1][0] if numbered else 0
        numbered.append((last_step + 1, final_path))
    return numbered


def main():
    """checkpoint_dir内の全チェックポイントを、単一のIsaac Simプロセス内で順に評価する"""
    test_cfg = TestConfig()
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(checkpoint_dir)
    if not checkpoints:
        print(f"[test] Error: {checkpoint_dir} にチェックポイント(sac_*.pt)が見つかりません")
        app.close()
        return

    names = stage_names()
    if args.stage_index >= len(names):
        print(f"[test] Error: stage-index {args.stage_index} は範囲外です (stages={len(names)})")
        app.close()
        return

    stage_name = names[args.stage_index]
    stage_cfg = get_preset(stage_name)

    print(f"[test] Stage {args.stage_index}: {stage_cfg.stage_path}")
    print(f"[test] Checkpoints: {len(checkpoints)} 個 in {checkpoint_dir}")
    print(f"[test] Episodes per checkpoint: {test_cfg.episodes_per_stage}")
    print(f"[test] num_humans: {args.num_humans}")

    use_wandb = not args.no_wandb
    run_name = args.run_name or f"test_{checkpoint_dir.name}_{datetime.now():%Y%m%d_%H%M%S}"
    if use_wandb:
        import wandb

        wandb.init(
            project="Isaac-GS-test",
            name=run_name,
            config={
                "checkpoint_dir": str(checkpoint_dir),
                "stage": stage_name,
                "num_humans": args.num_humans,
                "episodes_per_checkpoint": test_cfg.episodes_per_stage,
            },
            dir=str(log_dir),
        )

    backbone = get_backbone(DEVICE)
    # DINOv3の学習解像度(224x224)にIsaacSim側のレンダリング解像度を合わせる
    env_cfg = EnvConfig.from_preset(
        stage_name,
        show_camera_viewport=not args.headless,
        num_humans=args.num_humans,
        camera_resolution=(backbone.image_size, backbone.image_size),
    )

    env = PointNavGymEnv(env_cfg=env_cfg)
    model_obs_space, obs_transform = build_obs_pipeline(env.observation_space, DEVICE)
    env.reset()  # 俯瞰カメラ生成にはステージのロードが済んでいる必要がある
    action_dim = env.action_space.shape[0]

    # agentはstate_dictの再ロードだけで使い回せる(policy.SACAgent.load参照)ため、
    # チェックポイントごとに再構築せず、Isaac Simの起動は1度だけで済ませる。
    agent = SACAgent(
        encoder_factory=lambda: make_encoder(model_obs_space),
        action_dim=action_dim,
        cfg=TrainConfig(stage=stage_name),
        device=DEVICE,
    )

    overhead_camera = make_overhead_camera(stage_cfg) if args.video else None
    if args.video:
        print(f"[test] video: enabled -> {args.video_dir}/<checkpoint>/")
        if overhead_camera is not None:
            print("[test] overhead camera: enabled")

    log_path = log_dir / "results.log"
    with open(log_path, "a") as log_file:
        for step, ckpt_path in checkpoints:
            print(f"[test] Loading checkpoint: {ckpt_path} (step={step})")
            agent.load(str(ckpt_path))

            recorder = None
            if args.video:
                # EpisodeRecorderはコンストラクタでout_dirを確定するため、
                # チェックポイントごとにサブディレクトリを分けて作り直す。
                recorder = EpisodeRecorder(
                    out_dir=Path(args.video_dir) / ckpt_path.stem,
                    fps=1.0 / (env_cfg.physics_dt * env_cfg.decimation),
                    robot_resolution=env_cfg.camera_resolution,
                    overhead_camera=overhead_camera,
                )

            stats = evaluate(
                env,
                agent,
                test_cfg.episodes_per_stage,
                recorder=recorder,
                video_episodes=test_cfg.episodes_per_stage if args.video else 0,
                stem_fn=lambda ep: f"{ep}",
                desc=f"[{ckpt_path.stem}]",
                obs_transform=obs_transform,
            )

            separator = "=" * 60
            lines = [
                separator,
                f"Checkpoint: {ckpt_path.name} (step={step})",
                f"Stage {args.stage_index}: {stage_cfg.stage_path}",
                separator,
                *stats.report_lines(),
                separator,
                "",
            ]
            for line in lines:
                tqdm.write(line, file=_OUT)
                log_file.write(line + "\n")
            log_file.flush()

            if use_wandb:
                wandb.log(stats.rates("test/"), step=step)

    if use_wandb:
        wandb.finish()

    env.close()
    app.close()


if __name__ == "__main__":
    main()
