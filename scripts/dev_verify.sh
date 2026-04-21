#!/usr/bin/env bash
# 仓库根一键：语法编译 + Phase1 核心烟雾（默认不拉 HF）。
# 用法：bash scripts/dev_verify.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m compileall -q .
python3 experiments/phase1/run_phase1_smoke_all.py
echo "dev_verify.sh: OK"
