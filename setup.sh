#!/usr/bin/env bash
# Isaac-GS 環境セットアップスクリプト

set -e

ENV_DIR=".venv"

# 1. 仮想環境作成
uv venv --python 3.12 --seed "$ENV_DIR"
source "$ENV_DIR/bin/activate"

# 2. IsaacSim インストール（torch/torchvisionはCUDA対応版が同梱される）
uv pip install \
    "isaacsim[all,extscache]==6.0.1.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease=allow

# 3. Isaac-GS 追加依存インストール（gymnasium / wandb / Pillow / pydantic / opencv-python）
uv pip install -e .

echo ""
echo "セットアップ完了！"
