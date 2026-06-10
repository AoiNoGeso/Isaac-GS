# Isaac-GS

[Recon-GS](https://github.com/AoiNoGeso/Recon-GS) によって生成した 3D Gaussian Splatting (.ply) とメッシュ (.ply) を Isaac Sim で使用できるようにするためのパイプラインです

## 概要

```
Recon-GSによる3D再構成
        ↓
gaussian.ply（3DGS点群）  +  tsdf_fusion_post.ply（メッシュ）
        ↓
convert_gs.py   →   gs.usdc   （視覚：3DGSスプラット，Y-up，X軸180度回転済み）
convert_mesh.py →   mesh.usd  （物理：メッシュコライダ）
        ↓
compose_stage.py → stage.usda （Isaac Sim用ステージ）
```

生成される `stage.usda` では，3DGSを視覚表現として参照し，メッシュを不可視のコライダとして重ねることで，リアルな見た目と正確な物理コリジョンを両立します

## ディレクトリ構成

```
Isaac-GS/
├── stage_generation/
│   ├── convert_gs.py      # GS (.ply) → .usdc 変換・回転補正
│   ├── convert_mesh.py    # メッシュ (.ply) → .usd 変換
│   └── compose_stage.py   # gs.usdc + mesh.usd → stage.usda 合成
└── sample_data/           # サンプルデータ（.gitignore対象）
    ├── gaussian.ply
    ├── tsdf_fusion_post.ply
    └── outputs/
        ├── gs.usdc
        ├── mesh.usd
        └── stage.usda
```

## 前提条件

- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) がインストールされたuv仮想環境（`~/env_isaaclab`）
- [Recon-GS](https://github.com/AoiNoGeso/Recon-GS) で生成した `gaussian.ply` と `tsdf_fusion_post.ply`
- Isaac Sim 6.0 およびIsaac Labのインストールは[公式ドキュメント](https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/pip_installation.html)を参照してください

## パイプライン実行手順

**1. Isaac Sim の uv 環境を有効化**

```bash
source ~/env_isaaclab/bin/activate
```

**2. 3DGS (.ply) を Isaac Sim 用 .usdc に変換**

`usd_convert_gsplat` により変換後，X軸180度の回転補正を適用します．

```bash
uv run stage_generation/convert_gs.py \
    -i path/to/your/gaussian.ply \
    -o dir/to/your/outputs
```

出力: `dir/to/your/outputs/gs.usdc`

**3. メッシュ (.ply) を Isaac Sim 用 .usd に変換**

`omni.kit.asset_converter` を使用して変換します．`-o` には手順2と同じ出力先ディレクトリを指定してください．

```bash
uv run stage_generation/convert_mesh.py \
    -i path/to/your/tsdf_fusion_post.ply \
    -o dir/to/your/outputs
```

出力: `dir/to/your/outputs/mesh.usd`

**4. stage.usda の合成**

`gs.usdc` と `mesh.usd` が置かれたディレクトリを `-i` に指定します．

```bash
uv run stage_generation/compose_stage.py \
    -i dir/to/your/outputs
```

出力: `dir/to/your/outputs/stage.usda`

## 出力ファイルの構成

生成される `stage.usda` のプリム構成は以下の通りです．

```
/World                        （Xform, defaultPrim）
├── /gs_splat           （3DGSスプラット参照: gs.usdc, visible）
└── /mesh_collider      （メッシュ参照: mesh.usd, invisible）
    └── /mesh_          （PhysicsCollisionAPI + PhysicsMeshCollisionAPI 付与）
```

| プリム | 役割 | Visible | コリジョン |
|---|---|---|---|
| `gs_splat` | 視覚表現（3DGS） | ✓ | なし |
| `mesh_collider` | 物理コライダ（メッシュ） | ✗ | `PhysicsCollisionAPI` / 近似: `none` |

ステージ設定: `upAxis = Y`，`metersPerUnit = 1.0`

## 注意事項

- `convert_gs.py` 内の `GSPLAT_DIR` / `USD_LIBS` は環境に合わせてパスを変更してください．
