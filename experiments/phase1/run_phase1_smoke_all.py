# -*- coding: utf-8 -*-
"""
Phase1 端到端烟雾（默认 **不依赖 HF / 网络**）：几何自检、Concat 等价、toy CL、Phase0 smoke。

可选 HF 最小步：环境变量 ``EVIDENCE_HF_SMOKE=1`` 时再跑 ``train_distilgpt2_minimal.py``
（需已安装 requirements_hf.txt 且能拉模型或本地缓存）。

在仓库根执行:
  py -3 experiments\\phase1\\run_phase1_smoke_all.py
  set EVIDENCE_HF_SMOKE=1
  py -3 experiments\\phase1\\run_phase1_smoke_all.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _run(rel: str, extra_args: list[str] | None = None) -> int:
    script = _ROOT / rel
    cmd = [sys.executable, str(script)] + (extra_args or [])
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(_ROOT))


def main() -> int:
    steps = [
        ("experiments/phase1/verify_geometry.py", []),
        ("experiments/phase1/test_concat_optimizer.py", []),
        ("experiments/phase1/run_toy_cl.py", []),
        ("smoke/run_smoke.py", []),
    ]
    for rel, xa in steps:
        code = _run(rel, xa)
        if code != 0:
            print("phase1_smoke_all: FAIL at", rel, "exit", code)
            return code

    if os.environ.get("EVIDENCE_HF_SMOKE", "").strip() in ("1", "true", "yes", "on"):
        code = _run("experiments/phase1/train_distilgpt2_minimal.py", [])
        if code != 0:
            print("phase1_smoke_all: HF minimal FAIL exit", code)
            return code
        print("phase1_smoke_all: HF minimal step OK (or SKIP inside script)")
    else:
        print("phase1_smoke_all: skip HF (set EVIDENCE_HF_SMOKE=1 to include train_distilgpt2_minimal.py)")

    print("phase1_smoke_all: OK (core + optional HF)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
