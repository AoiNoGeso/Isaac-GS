# Isaac-GS

[Recon-GS](https://github.com/AoiNoGeso/Recon-GS) によって生成した 3D Gaussian Splatting (.ply) と床・壁メッシュ (.ply) を Isaac Sim 6.0 で使用できるようにするためのパイプラインです．

## 概要

```
Recon-GSによる3D再構成
        ↓
gaussian.ply（3DGS点群）  +  floor.ply（床メッシュ）  +  wall.ply（壁メッシュ）
        ↓
convert_gs.py          →   gs.usdc        （視覚：3DGSスプラット，Y-up，X軸180度回転済み）
convert_mesh.py (床)   →   floor_mesh.usd （物理：床コライダ）
convert_mesh.py (壁)   →   wall_mesh.usd  （物理：壁コライダ・ContactSensor対象）
        ↓
compose_stage.py → stage.usda （Isaac Sim用ステージ）
```

生成される `stage.usda` では，3DGSを視覚表現として参照し，床・壁メッシュを不可視のコライダとして重ねることで，リアルな見た目と正確な物理コリジョンを両立します．床と壁を別プリムとして管理することで，強化学習タスクでの壁衝突判定（ContactSensor）を床接触と分離して扱えます．

## ディレクトリ構成

```
Isaac-GS/
├── stage_generation/
│   ├── convert_gs.py      # GS (.ply) → .usdc 変換・回転補正
│   ├── convert_mesh.py    # メッシュ (.ply) → .usd 変換（汎用）
│   └── compose_stage.py   # gs.usdc + floor_mesh.usd + wall_mesh.usd → stage.usda 合成
├── stages/                # 生成済みステージの保存先
│   ├── corridor1/
│   │   ├── stage.usda     ← compose_stage.py の出力
│   │   ├── gs.usdc        ← convert_gs.py の出力
│   │   ├── floor_mesh.usd ← convert_mesh.py の出力（床）
│   │   └── wall_mesh.usd  ← convert_mesh.py の出力（壁）
│   └── room1/
│       ├── stage.usda
│       ├── gs.usdc
│       ├── floor_mesh.usd
│       └── wall_mesh.usd
└── tasks/                 # Isaac Lab 強化学習タスク (未実装)
    └── point_navigation/
        ├── __init__.py
        ├── point_navigation_env_cfg.py
        ├── point_navigation_env.py
        └── agents/
            ├── __init__.py
            └── rsl_rl_ppo_cfg.py

```

## 前提条件

- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) がインストールされた仮想環境（`~/env_isaaclab`）
- [Recon-GS](https://github.com/AoiNoGeso/Recon-GS) で生成した `gaussian.ply`・`floor.ply`・`wall.ply`
- Isaac Sim 6.0 および Isaac Lab のインストールは[公式ドキュメント](https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/pip_installation.html)を参照してください

## パイプライン実行手順
ここでは「corridor1」という名称のstageを作ることを想定しています．
stage名は適宜変更して下さい．

**1. 仮想環境を有効化**

```bash
source ~/env_isaaclab/bin/activate
```

**2. 3DGS (.ply) を Isaac Sim 用 .usdc に変換**

`usd_convert_gsplat` により変換後，X軸180度の回転補正を適用します．

```bash
uv run stage_generation/convert_gs.py \
    -i path/to/gaussian.ply \
    -o stages/corridor1
```

出力: `stages/corridor1/gs.usdc`

**3. 床メッシュ (.ply) を .usd に変換**

```bash
uv run stage_generation/convert_mesh.py \
    -i path/to/floor.ply \
    -o stages/corridor1/floor_mesh.usd
```

出力: `stages/corridor1/floor_mesh.usd`

**4. 壁メッシュ (.ply) を .usd に変換**

```bash
uv run stage_generation/convert_mesh.py \
    -i path/to/wall.ply \
    -o stages/corridor1/wall_mesh.usd
```

出力: `stages/corridor1/wall_mesh.usd`

**5. stage.usda の合成**

`gs.usdc`・`floor_mesh.usd`・`wall_mesh.usd` が揃ったディレクトリを `-i` に指定します．

```bash
uv run stage_generation/compose_stage.py \
    -i stages/corridor1
```

出力: `stages/corridor1/stage.usda`

## 出力ファイルの構成

生成される `stage.usda` のプリム構成は以下の通りです．

```
/World                   （Xform，defaultPrim）
└── /World/env           （Xform，環境ルート）
    ├── /World/env/gs          （3DGSスプラット参照: gs.usdc，visible）
    ├── /World/env/floor_mesh  （床メッシュ参照: floor_mesh.usd，invisible）
    └── /World/env/wall_mesh   （壁メッシュ参照: wall_mesh.usd，invisible）
```

| プリム | 役割 | Visible | コリジョン | ContactSensor対象 |
|---|---|---|---|---|
| `env/gs` | 視覚表現（3DGS） | ✓ | なし | ✗ |
| `env/floor_mesh` | 床コライダ | ✗ | `PhysicsCollisionAPI` / 近似: `none` | ✗ |
| `env/wall_mesh` | 壁コライダ | ✗ | `PhysicsCollisionAPI` / 近似: `none` | ✓ |

ステージ設定: `upAxis = Y`，`metersPerUnit = 1.0`

## 注意事項

- `convert_gs.py` 内の `GSPLAT_DIR` / `USD_LIBS` は環境に合わせてパスを変更してください．
- `compose_stage.py` は `gs.usdc`・`floor_mesh.usd`・`wall_mesh.usd` の3ファイルが全て揃っていない場合エラーで終了します．
