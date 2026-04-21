# -*- coding: utf-8 -*-
"""
汇总 run_joint_geometry_cl.py 产出的 JSONL，生成一段可直接粘贴到聊天/ issue 的短报告。

用法（服务器）:
  python scripts/summarize_joint_geom_jsonl.py experiments/phase1/logs/joint_geom*.jsonl
  python scripts/summarize_joint_geom_jsonl.py experiments/phase1/logs/ --glob 'joint_geom*.jsonl'

落盘 + 终端同时有（便于下载一个文件）:
  python scripts/summarize_joint_geom_jsonl.py experiments/phase1/logs/ --glob 'joint_geom*.jsonl' \\
    --out /root/autodl-tmp/work/Adam/JOINT_SUMMARY.txt && cat /root/autodl-tmp/work/Adam/JOINT_SUMMARY.txt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def _try_git_head(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _collect_paths(inputs: list[str], glob_pat: str | None) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_dir():
            pat = glob_pat or "joint_geom*.jsonl"
            paths.extend(sorted(p.glob(pat)))
        elif p.is_file():
            paths.append(p)
        else:
            print("warn: skip missing path", raw, file=sys.stderr)
    # de-dupe preserve order
    seen: set[str] = set()
    out: list[Path] = []
    for q in paths:
        s = str(q.resolve())
        if s not in seen:
            seen.add(s)
            out.append(q)
    return out


def _parse_jsonl(path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if isinstance(o, dict) and (
                o.get("kind") == "joint_geometry_cl_meta"
                or ("lora_D" in o and "adamw_lora" in o and "anchor_steps" in o)
            ):
                meta = o
            elif isinstance(o, dict) and "step" in o and "loss" in o and "task" in o:
                rows.append(o)
    return meta, rows


def _mean_tail(rows: list[dict], task: int, k: int, key: str) -> float:
    sub = [r for r in rows if int(r["task"]) == task]
    if not sub:
        return float("nan")
    tail = sub[-k:] if len(sub) >= k else sub
    return sum(float(r[key]) for r in tail) / len(tail)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize joint_geometry_cl JSONL logs")
    ap.add_argument(
        "inputs",
        nargs="+",
        help="JSONL 文件路径，或目录（与 --glob 联用）",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default=None,
        help="当 inputs 为目录时使用的 glob，默认 joint_geom*.jsonl",
    )
    ap.add_argument(
        "--tail",
        type=int,
        default=50,
        help="按 task 统计「最后 N 条」训练步的平均 loss / vtm",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="可选：把报告写入该文件（UTF-8）",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    paths = _collect_paths(args.inputs, args.glob)
    if not paths:
        print("No JSONL files found.", file=sys.stderr)
        return 1

    lines: list[str] = []
    lines.append("=== SubGeo joint_geometry_cl 汇总（可整段粘贴）===")
    lines.append(f"repo_git_short: {_try_git_head(root)}")
    lines.append(f"tail_n: {args.tail}")
    lines.append("")

    summaries: list[dict] = []
    for path in paths:
        meta, rows = _parse_jsonl(path)
        if not rows:
            lines.append(f"file: {path}")
            lines.append("  ERROR: no train rows")
            lines.append("")
            continue
        adamw = bool(meta.get("adamw_lora", False))
        seed = meta.get("seed", "n/a")
        m0 = _mean_tail(rows, 0, args.tail, "loss")
        m1 = _mean_tail(rows, 1, args.tail, "loss")
        v0 = _mean_tail(rows, 0, args.tail, "vtm_lora")
        v1 = _mean_tail(rows, 1, args.tail, "vtm_lora")
        last = rows[-1]
        summaries.append(
            {
                "path": str(path),
                "stem": path.name,
                "seed": seed,
                "adamw_lora": adamw,
                "n_steps": len(rows),
                "mean_loss_t0_tail": m0,
                "mean_loss_t1_tail": m1,
                "mean_vtm_t0_tail": v0,
                "mean_vtm_t1_tail": v1,
                "last_step": int(last["step"]),
                "last_loss": float(last["loss"]),
                "last_task": int(last["task"]),
            }
        )
        lines.append(f"file: {path.name}")
        lines.append(f"  seed: {seed}  adamw_lora: {adamw}")
        lines.append(f"  post_steps: {len(rows)}  last_step task: {last['step']} task={last['task']}")
        lines.append(f"  mean_loss_task0_last{args.tail}: {m0:.6g}")
        lines.append(f"  mean_loss_task1_last{args.tail}: {m1:.6g}")
        lines.append(f"  mean_vtm_task0_last{args.tail}:   {v0:.6g}")
        lines.append(f"  mean_vtm_task1_last{args.tail}:   {v1:.6g}")
        lines.append("")

    # 按 seed 粗配对提示
    by_seed: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        sk = str(s.get("seed", "?"))
        by_seed[sk].append(s)
    lines.append("--- 同 seed 下 SubGeo(False) vs AdamW(True) 尾窗 mean_loss 差（task0 / task1）---")
    for sk, group in sorted(by_seed.items(), key=lambda x: x[0]):
        subgeo = [x for x in group if not x["adamw_lora"]]
        adamw = [x for x in group if x["adamw_lora"]]
        if len(subgeo) != 1 or len(adamw) != 1:
            lines.append(f"  seed {sk}: 条目数 subgeo={len(subgeo)} adamw={len(adamw)}（跳过自动对比）")
            continue
        a, b = subgeo[0], adamw[0]
        d0 = a["mean_loss_t0_tail"] - b["mean_loss_t0_tail"]
        d1 = a["mean_loss_t1_tail"] - b["mean_loss_t1_tail"]
        lines.append(
            f"  seed {sk}: Δloss0(subgeo-adamw)={d0:+.6g}  Δloss1={d1:+.6g}  "
            f"(负表示 SubGeo 尾窗更低)"
        )
    lines.append("")
    lines.append("=== 以上结束 ===")

    text = "\n".join(lines)
    print(text)
    if args.out.strip():
        outp = Path(args.out).expanduser()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text + "\n", encoding="utf-8")
        print(f"(also wrote {outp})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
