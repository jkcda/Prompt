#!/usr/bin/env bash
# 不调用模型，只验证 ffmpeg / 镜头检测 / 选帧 / 音频分析
# 用法：./scripts/selfcheck.sh path/to/video.mp4
set -euo pipefail

cd "$(dirname "$0")/../backend"

if [ $# -lt 1 ]; then
  echo "用法：$0 path/to/video.mp4"
  exit 2
fi

python -m app.selfcheck "$1"
