# Isaac-GS

[Recon-GS](https://github.com/AoiNoGeso/Recon-GS) によって生成した 3D Gaussian Splatting (.ply) と床・壁メッシュ (.ply) を Isaac Sim 6.0 で使用し，ビジョンベース強化学習 (Point Navigation) を実行するためのパイプラインです

## 概要

```
Recon-GS による 3D 再構成
        ↓
gaussian.ply + floor.ply + wall.ply
        ↓
convert_gs.py     →  gs.usdc        (視覚: 3DGS スプラット)
convert_mesh.py   →  floor_mesh.usd (物理: 床コライダ)
convert_mesh.py   →  wall_mesh.usd  (物理: 壁コライダ)
        ↓
compose_stage.py  →  stage.usda  (CollisionAPI + NavMeshVolume)
        ↓
RL 学習 (tasks/point_navigation/train.py)  ← 起動時に NavMesh をランタイム bake
```

生成される `stage.usda` は 3DGS を視覚表現とし，床・壁メッシュを不可視コライダとして重ねることで，リアルな見た目と正確な物理コリジョンを両立します

## 前提条件

- uv がインストールされていること ([公式](https://docs.astral.sh/uv/getting-started/installation/))
- CUDA 12.8 対応の GPU 環境
- [Recon-GS](https://github.com/AoiNoGeso/Recon-GS) で生成した `gaussian.ply`・`floor.ply`・`wall.ply`

## ディレクトリ構成

```
Isaac-GS/
├── setup.sh                    # 環境構築スクリプト
├── pyproject.toml              # Isaac-GS 追加依存パッケージ管理
├── stage_generation/
│   ├── convert_gs.py           # GS (.ply) → .usdc 変換
│   ├── convert_mesh.py         # メッシュ (.ply) → .usd 変換
│   └── compose_stage.py        # gs.usdc + floor/wall → stage.usda 合成
├── stages/                     # ステージファイル置き場
│   └── corridor1/
│       └── stage.usda
├── envs/
│   ├── isaac_env.py            # IsaacSim 環境コア
│   ├── gym_wrapper.py          # gymnasium.Env ラッパー
│   └── sensors/
│       └── camera_sensor.py    # RGB カメラセンサ
├── tasks/
│   └── point_navigation/
│       ├── config.py           # 環境・報酬・学習設定 (Pydantic)
│       ├── train.py            # 学習スクリプト (SAC)
│       └── policy/
│           ├── network.py      # CNNEncoder / GoalEncoder / PointNavEncoder
│           └── policy.py       # SAC (ReplayBuffer / Actor / Critic / SACAgent)
├── deploy/
│   ├── sim_ros2_bridge.py      # Isaac Sim 内 ROS2 ブリッジ
│   └── deploy.py              # 実機・シミュレータ共通 policy 推論ノード
├── debug/
│   └── teleop.py               # WASD テレオペ
└── runs/                       # 学習ログ・チェックポイント
    └── point_nav/
```

## 環境構築

```bash
git clone https://github.com/AoiNoGeso/Isaac-GS.git
cd Isaac-GS
```

### 1. システム依存ライブラリ (初回のみ)

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

1. `~/env_Isaac-GS` (Python 3.12) 仮想環境を作成
2. IsaacSim 6.0 をインストール
3. PyTorch 2.10.0 (CUDA 12.8) をインストール
4. `pyproject.toml` のパッケージ (gymnasium / wandb / Pillow / pydantic) をインストール

## パイプライン実行手順

### 1. 3DGS (.ply) → .usdc

```bash
python stage_generation/convert_gs.py \
    -i path/to/gaussian.ply \
    -o stages/corridor1
```

### 2. 床・壁メッシュ (.ply) → .usd

```bash
python stage_generation/convert_mesh.py \
    -i path/to/floor.ply \
    -o stages/corridor1/floor_mesh.usd

python stage_generation/convert_mesh.py \
    -i path/to/wall.ply \
    -o stages/corridor1/wall_mesh.usd
```

### 3. stage.usda の合成

```bash
python stage_generation/compose_stage.py -i stages/corridor1
```

出力: `stages/corridor1/stage.usda`

### 4. RL 学習

学習設定は `tasks/point_navigation/config.py` の `PointNavTrainCfg` / `SACCfg` で編集します

```bash
# 学習開始
python tasks/point_navigation/train.py --headless

# 実験名を指定
python tasks/point_navigation/train.py --headless --run-name my_run

# チェックポイントから再開
python tasks/point_navigation/train.py \
    --headless --checkpoint runs/point_nav/checkpoints/sac_10000.pt

# wandb なし
python tasks/point_navigation/train.py --headless --no-wandb
```

### 5. シミュレータでのデプロイ (ROS2)

```bash
# ターミナル 1: Isaac Sim ROS2 ブリッジ起動
python deploy/sim_ros2_bridge.py

# ターミナル 2: policy 推論ノード起動
python deploy/deploy.py --model runs/point_nav/sac_final.pt
```

## デバッグ

```bash
python debug/teleop.py
```

| キー | 動作 |
|---|---|
| W / S | 前進 / 後退 |
| A / D | 左回転 / 右回転 |
| P | 現在のワールド座標を表示 |
| Q | 終了 |

## システム構成

| 項目 | 内容 |
|---|---|
| シミュレータ | Isaac Sim 6.0 |
| RL アルゴリズム | 自前実装 SAC (Soft Actor-Critic) |
| ロボット | Carter V1 (差動二輪) |
| 観測 | RGB 84×84 px / ゴールベクトル (2,) / 両方 (`config.py` で切替) |
| 行動 | [v_x_norm, ω_z_norm] (ユニサイクルモデル) |
| 衝突判定 | ContactSensor (Newton エンジン) による wall_mesh 接触検出 |
| 座標系 | Z-up |

## 注意事項

- `convert_gs.py` 内の `GSPLAT_DIR` / `USD_LIBS` パスは環境に合わせて修正してください (以下のコマンドで検索できます)
  ```bash
  find ~/env_Isaac-GS -type d -name "omni.kit.converter.gsplat-*"
  find ~/env_Isaac-GS -type d -name "omni.usd.libs-*"
  ```
