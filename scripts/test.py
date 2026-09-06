"""Point Navigation テストスクリプト(モデル共通)。
指定チェックポイント(ファイル)または全チェックポイント(ディレクトリ)を、
Isaac Simを1度だけ起動した単一プロセス内で順に評価し、wandbの同一runへ
stepごとにログする。
実行例:
  uv run scripts/test.py --model sac --checkpoint runs/S1-RGB+G/corridor2/sac_final.pt --stage-index 0
  uv run scripts/test.py --model sac_DINOv3 --checkpoint runs_forSI/.../checkpoints --log-dir runs_forSI/.../test/S1/log --headless"""

import argparse
import importlib
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_MODELS = sorted(p.name for p in (ROOT / "models").iterdir() if (p / "config.py").exists())

_CKPT_STEP_RE = re.compile(r"sac_(\d+)\.pt$")


def discover_checkpoints(checkpoint_path: Path) -> list[tuple[int, Path]]:
    """checkpoint_pathを掃引してstep昇順のリストを返す。

    ファイルなら単一チェックポイントを1要素で返す(stepはファイル名から拾う)。
    ディレクトリなら sac_{step}.pt / sac_final.pt を step 昇順で列挙する
    (sac_final.pt はwandb上でstep軸の続きに置けるよう、最大step+1として末尾に置く)。
    """
    if checkpoint_path.is_file():
        m = _CKPT_STEP_RE.match(checkpoint_path.name)
        return [(int(m.group(1)) if m else 0, checkpoint_path)]

    numbered: list[tuple[int, Path]] = []
    final_path = None
    for path in checkpoint_path.glob("sac_*.pt"):
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


parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=_MODELS, required=True)
parser.add_argument("--checkpoint", type=str, required=True, help="チェックポイントのファイルまたはディレクトリ")
parser.add_argument("--log-dir", type=str, required=True, help="結果ログ・wandbデータの出力先ディレクトリ")
parser.add_argument("--stage-index", type=int, default=0, help="評価するステージのインデックス")
parser.add_argument("--num-humans", type=int, default=0)
parser.add_argument("--episodes", type=int, default=None, help="TestConfig.episodes_per_stageを上書き(スモーク用)")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--no-wandb", action="store_true", default=False)
parser.add_argument("--gpu", type=int, default=0, help="共用マシンでの複数GPU分散を防ぐため、使用するGPU番号を固定する")
parser.add_argument("--video", action="store_true", default=False, help="各エピソードの映像をmp4で保存する")
parser.add_argument("--video-dir", type=str, default="videos", help="動画の出力先ディレクトリ (default: videos)")
parser.add_argument("--run-name", type=str, default=None, help="wandbのrun名 (省略時はcheckpoint名+日時から自動生成)")
args = parser.parse_args()

# launch_sim()前: pydantic/numpyのみに依存するconfigだけを引き込む(models/*/__init__.pyは空)
mcfg = importlib.import_module(f"models.{args.model}.config")

_OUT = sys.stdout  # launch_sim()前のstdoutを捕捉(tqdm.writeの出力先)
assert "torch" not in sys.modules, "launch_sim()前にtorchがimportされている"

from utils.launch_sim import launch_sim

app = launch_sim(headless=args.headless, gpu=args.gpu)

# ここから先はtorch依存モジュールをimportしてよい
import torch
from tqdm import tqdm

from envs import PointNavGymEnv
from envs.config import EnvConfig, get_preset, stage_names
from utils.recorder import EpisodeRecorder, make_overhead_camera, robot_resolution_from_space
from utils.rollout import evaluate

mnet = importlib.import_module(f"models.{args.model}.network")
mpol = importlib.import_module(f"models.{args.model}.policy")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    """checkpoint(単一または複数)を、単一のIsaac Simプロセス内で順に評価する"""
    test_cfg = mcfg.TestConfig()
    if args.episodes is not None:
        test_cfg.episodes_per_stage = args.episodes

    checkpoint_path = Path(args.checkpoint)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(checkpoint_path)
    if not checkpoints:
        print(f"[test] Error: {checkpoint_path} にチェックポイント(sac_*.pt)が見つかりません")
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
    print(f"[test] Checkpoints: {len(checkpoints)} 個 in {checkpoint_path}")
    print(f"[test] Episodes per checkpoint: {test_cfg.episodes_per_stage}")
    print(f"[test] num_humans: {args.num_humans}")

    use_wandb = not args.no_wandb
    run_name = args.run_name or f"test_{checkpoint_path.name}_{datetime.now():%Y%m%d_%H%M%S}"
    if use_wandb:
        import wandb

        wandb.init(
            project="Isaac-GS-test",
            name=run_name,
            config={
                "checkpoint": str(checkpoint_path),
                "stage": stage_name,
                "num_humans": args.num_humans,
                "episodes_per_checkpoint": test_cfg.episodes_per_stage,
            },
            dir=str(log_dir),
        )

    env_cfg = EnvConfig.from_preset(
        stage_name,
        num_humans=args.num_humans,
        **mcfg.env_overrides(),
    )

    env = PointNavGymEnv(env_cfg=env_cfg)
    model_obs_space, obs_transform = mnet.build_obs_pipeline(env.observation_space, DEVICE)
    env.reset()  # 俯瞰カメラ生成にはステージのロードが済んでいる必要がある
    action_dim = env.action_space.shape[0]

    # stack_sizeは学習時のTrainConfigと一致している前提(checkpointは学習時の設定を保存しないため)
    ref_train_cfg = mcfg.TrainConfig(stage=stage_name)

    # agentはstate_dictの再ロードだけで使い回せる(policy.SACAgent.load参照)ため、
    # チェックポイントごとに再構築せず、Isaac Simの起動は1度だけで済ませる。
    agent = mpol.SACAgent(
        encoder_factory=lambda: mnet.make_encoder(model_obs_space, stack_size=ref_train_cfg.stack_size),
        action_dim=action_dim,
        cfg=ref_train_cfg,
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
                    robot_resolution=robot_resolution_from_space(env.observation_space),
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
                stack_size=ref_train_cfg.stack_size,
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
