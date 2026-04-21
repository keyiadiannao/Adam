# -*- coding: utf-8 -*-
"""读取 ``run_hf_real_two_task_cl.py`` 的 JSONL：打印 meta、eval 行首尾、训练 loss 均值。"""
from __future__ import annotations

import json
import statistics
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: py -3 summarize_hf_real_cl_log.py <path.jsonl>")
        return 1
    path = sys.argv[1]
    meta = None
    train_losses: list[float] = []
    eval_rows: list[dict] = []
    with open(path, encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if i == 0 and isinstance(o, dict) and "lora_D" in o:
                meta = o
            if o.get("kind") == "eval":
                eval_rows.append(o)
            if "loss" in o and "task" in o and "step" in o and o.get("kind") != "eval":
                train_losses.append(float(o["loss"]))
    if meta:
        s = json.dumps(meta, ensure_ascii=False)
        print("meta:", s if len(s) <= 800 else s[:800] + "...")
    if train_losses:
        print("train steps:", len(train_losses), "mean loss:", statistics.mean(train_losses))
    if eval_rows:
        print("eval rows:", len(eval_rows))
        print("  first:", eval_rows[0])
        print("  last: ", eval_rows[-1])
    else:
        print("no eval rows (use --holdout-per-class >0 and --eval-every >0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
