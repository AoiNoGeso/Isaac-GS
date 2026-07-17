# Isaac-GS

[Recon-GS](https://github.com/AoiNoGeso/Recon-GS) によって生成した 3D Gaussian Splatting (.ply) と床・壁メッシュ (.ply) を Isaac Sim 6.0 で使用し，ビジョンベース強化学習 (Point Navigation) を実行するためのパイプラインです

## 概要

```
Recon-GS による 3D 再構成
        ↓
gaussian.ply + floor.ply + wall.ply
        ↓
stage_generation convert-gs     →  gs.usdc        (視覚: 3DGS スプラット)
stage_generation convert-mesh   →  floor_mesh.usd (物理: 床コライダ)
stage_generation convert-mesh   →  wall_mesh.usd  (物理: 壁コライダ)
        ↓
stage_generation compose        →  stage.usda  (CollisionAPI + NavMeshVolume)
        ↓
RL 学習 (models/sac/train.py)  ← 起動時に NavMesh をランタイム bake
```

生成される `stage.usda` は 3DGS を視覚表現とし，床・壁メッシュを不可視コライダとして重ねることで，リアルな見た目と正確な物理コリジョンを両立します

## 前提条件

- uv がインストールされていること
- CUDA 12.8 対応の GPU 環境
- [Recon-GS](https://github.com/AoiNoGeso/Recon-GS) で生成した `gaussian.ply`・`floor.ply`・`wall.ply`

## ディレクトリ構成

```
Isaac-GS/
├── setup.sh                    # 環境構築スクリプト
├── pyproject.toml              # Isaac-GS 追加依存パッケージ管理
├── stage_generation/            # 統合 CLI: python -m stage_generation {convert-gs,convert-mesh,compose}
│   ├── __main__.py              # argparse エントリポイント（サブコマンド定義）
│   ├── convert_gs.py            # GS (.ply) → .usdc 変換
│   ├── convert_mesh.py          # メッシュ (.ply) → .usd 変換
│   ├── compose.py                # gs.usdc + floor/wall → stage.usda 合成
│   └── coords.py                 # source-up-axis → Z-up 座標変換（共通）
├── stages/                       # ステージファイル置き場
│   ├── room1/stage.usda
│   └── corridor1/stage.usda
├── envs/
│   ├── config.py                # EnvConfig / RobotConfig(+JACKAL) / STAGE_PRESETS
│   ├── geometry.py               # quat_to_yaw / goal_vec（isaacsim 非依存、env と deploy で共用）
│   ├── isaac_env.py              # IsaacSim 環境コア (PointNavIsaacEnv) + gymnasium ラッパー (PointNavGymEnv)
│   └── sensors/
│       └── camera_sensor.py      # RGB+Depth カメラセンサ
├── models/
│   └── sac/                      # 自前実装 SAC（参照実装）
│       ├── config.py             # ModelConfig / TrainConfig
│       ├── network.py            # エンコーダ (make_encoder)
│       ├── policy.py             # ReplayBuffer / Actor / Critic / SACAgent
│       ├── train.py              # 学習スクリプト
│       └── test.py               # 評価スクリプト
│   # 外部モデル（NoMaD, NavDP 等）は models/<Name>/ に git clone して配置する
├── utils/
│   └── wandb_utils.py            # EpisodeTracker（SPL 計算・wandb ログ、train/test 共用）
├── deploy/
│   └── deploy.py                 # 実機 policy 推論ノード (ROS2)
├── debug/
│   └── teleop.py                 # WASD テレオペ
└── runs/                        # 学習ログ・チェックポイント
    └── point_nav/
```

## 環境構築

```bash
git clone https://github.com/AoiNoGeso/Isaac-GS.git
cd Isaac-GS
```

### 1. システム依存ライブラリ

```bash
sudo apt install python3.12-dev libgl1-mesa-dev libx11-dev \
    libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev
```

### 2. Python 環境のセットアップ

`setup.sh` で順番にインストールします

```bash
zsh setup.sh
```

`setup.sh` は以下を順番に実行します：

1. リポジトリ直下に `.venv` (Python 3.12) 仮想環境を作成
2. IsaacSim 6.0 をインストール
3. PyTorch 2.10.0 (CUDA 12.8) をインストール
4. `pyproject.toml` のパッケージ (gymnasium / wandb / Pillow / pydantic) をインストール

## パイプライン実行手順

### 1. 3DGS (.ply) → .usdc

```bash
uv run python -m stage_generation convert-gs \
    -i path/to/gaussian.ply \
    -o stages/corridor1
```

`--source-up-axis {Y,Z}`（既定 `Y` = Recon-GS の -Y-up 出力）で変換元データの up-axis を指定できます

### 2. 床・壁メッシュ (.ply) → .usd

```bash
uv run python -m stage_generation convert-mesh \
    -i path/to/floor.ply \
    -o stages/corridor1/floor_mesh.usd

uv run python -m stage_generation convert-mesh \
    -i path/to/wall.ply \
    -o stages/corridor1/wall_mesh.usd
```

### 3. stage.usda の合成

```bash
uv run python -m stage_generation compose -i stages/corridor1
```

出力: `stages/corridor1/stage.usda`

  ### 4. ナビゲーションタスク学習

学習設定は `envs/config.py` の `EnvConfig` / `RobotConfig`、エンコーダ入力設定は `models/sac/config.py` の `ModelConfig`、SAC ハイパーパラメータは同ファイルの `TrainConfig` で編集します

```bash
# 学習開始
uv run models/sac/train.py

# 実験名を指定
uv run models/sac/train.py --run-name my_run

# チェックポイントから再開
uv run models/sac/train.py \
    --checkpoint runs/PointNav-SAC-RGB/checkpoints/sac_10000.pt

# wandb なし
uv run models/sac/train.py --no-wandb
```


## テスト

学習済みモデルを評価します。ステージと評価設定は `test.py` 冒頭の `TestCfg` で定義します

```bash
# 単一ステージ（index 0）
uv run models/sac/test.py \
    --model runs/PointNav-RGB+Goal/0627/sac_final.pt \
    --stage-index 0 \
   
```

| 引数 | 説明 |
|---|---|
| `--model` | チェックポイントパス（省略時は `TestCfg.model_path`） |
| `--stage-index` | 評価するステージのインデックス（デフォルト 0） |
| `--headless` | ヘッドレス実行 |

出力例:
```
============================================================
Stage 0: stages/room1/stage.usda
============================================================
Episodes      : 100
Success Rate  : 0.720  (72/100)
Collision Rate: 0.150  (15/100)
Timeout Rate  : 0.130  (13/100)
Avg Reward    : 45.32
Avg Dist Final: 0.381 m
SPL           : 0.613
============================================================
```

## デバッグ

```bash
uv run debug/teleop.py
uv run debug/teleop.py --num-humans 2            # IRA 人物キャラを注入して衝突判定を確認
```

| キー | 動作 |
|---|---|
| W / S | 前進 / 後退 |
| A / D | 左回転 / 右回転 |
| P | 現在のワールド座標を表示（人物ありの場合は各人物の座標も） |
| R | 手動リセット |
| Q | 終了 |

## システム構成

| 項目 | 内容 |
|---|---|
| シミュレータ | Isaac Sim 6.0 |
| RL アルゴリズム | 自前実装 SAC (Soft Actor-Critic) |
| ロボット | Clearpath Jackal (4輪スキッドステア) |
| 観測 | 環境は常に RGB 84×84 px・Depth・ゴールベクトル (2,) を発行。どれをエンコーダ入力に使うかは `models/sac/config.py` の `ModelConfig` で選択 |
| 行動 | [v_x_norm, ω_z_norm]（`RobotConfig` の `wheel_base`/`v_linear_max`/`v_angular_max` を介しロボット非依存に左右輪速度へ変換） |
| 衝突判定 | ContactSensor (Newton エンジン) による wall_mesh 接触検出 |
| 座標系 | Z-up、ロボット前方 = +X |

外部のナビゲーションモデル（NoMaD, NavDP 等）を試す場合は `models/<Name>/` に git clone して配置してください

## 注意事項

- `stage_generation/convert_gs.py` の GSPLAT 変換パス（`GSPLAT_DIR` / `USD_LIBS`）は `.venv` 内から自動検出されます。自動検出できない場合は環境変数で上書きしてください (以下のコマンドで検索できます)
  ```bash
  find .venv -type d -name "omni.kit.converter.gsplat-*"
  find .venv -type d -name "omni.usd.libs-*"
  export GSPLAT_DIR=...
  export USD_LIBS=...
  ```
