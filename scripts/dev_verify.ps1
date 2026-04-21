# 仓库根一键：语法编译 + Phase1 核心烟雾（默认不拉 HF）。
# 用法：在仓库根执行  powershell -File scripts/dev_verify.ps1  （或 pwsh -File ...）
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
py -3 -m compileall -q .
py -3 experiments/phase1/run_phase1_smoke_all.py
Write-Host "dev_verify.ps1: OK"
