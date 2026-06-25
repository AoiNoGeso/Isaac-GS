#!/usr/bin/env bash
# Isaac-GS 環境セットアップスクリプト
# 実行: bash setup.sh
set -e

ENV_DIR="$HOME/env_Isaac-GS"

# 1. 仮想環境作成
uv venv --python 3.12 --seed "$ENV_DIR"
source "$ENV_DIR/bin/activate"

# 2. IsaacSim インストール
uv pip install \
    "isaacsim[all,extscache]==6.0.0.1" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease=allow

# 3. PyTorch (CUDA 12.8) インストール
#    isaacsim のメタデータ宣言より新しいバージョンを上書きインストール
uv pip install -U \
    "torch==2.10.0" \
    "torchvision==0.25.0" \
    --index-url https://download.pytorch.org/whl/cu128

# 4. Isaac-GS 追加依存インストール（gymnasium / wandb / Pillow / pydantic）
uv pip install -e .

echo ""
echo "セットアップ完了！"
echo "以降は以下で環境を有効化してください:"
echo "  source $ENV_DIR/bin/activate"
