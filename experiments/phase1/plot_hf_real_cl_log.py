# -*- coding: utf-8 -*-
"""
从 ``run_hf_real_two_task_cl.py`` 的 JSONL 绑图：训练 loss（按 step）、eval 曲线（若有）。

依赖: pip install -r experiments/phase1/requirements_plot.txt

示例:
  py -3 experiments\\phase1\\plot_hf_real_cl_log.py experiments\\phase1\\logs\\ag_news_real_subgeo_xxx.jsonl --out plot_subgeo.png
  py -3 experiments\\phase1\\plot_hf_real_cl_log.py subgeo.jsonl adamw.jsonl --out compare.png --labels SubGeo AdamW
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse(path: str) -> tuple[dict, list[dict], list[dict]]:
    meta: dict = {}
    train: list[dict] = []
    evals: list[dict] = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if isinstance(o, dict) and "lora_D" in o:
                meta = o
            elif o.get("kind") == "eval":
                evals.append(o)
            elif "step" in o and "loss" in o and "task" in o:
                train.append(o)
    return meta, train, evals


def main() -> int:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("SKIP: matplotlib not installed. pip install -r experiments/phase1/requirements_plot.txt")
        return 0

    ap = argparse.ArgumentParser(description="Plot HF real two-task JSONL logs")
    ap.add_argument("jsonl", nargs="+", help="One or two JSONL paths")
    ap.add_argument("--out", type=str, required=True, help="Output .png path")
    ap.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Legend labels for each jsonl (default: file stem)",
    )
    args = ap.parse_args()

    paths = args.jsonl
    labels = list(args.labels) if args.labels else []
    if not labels:
        labels = [Path(p).stem for p in paths]
    while len(labels) < len(paths):
        labels.append(Path(paths[len(labels)]).stem)

    first_meta: dict = {}
    for p in paths:
        m, _, _ = _parse(p)
        if m:
            first_meta = m
            break

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_loss, ax_eval = axes[0], axes[1]
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")

    for i, p in enumerate(paths):
        meta, train, evals = _parse(p)
        c = colors[i % len(colors)]
        lab = labels[i]
        if train:
            xs = [int(r["step"]) for r in train]
            ys = [float(r["loss"]) for r in train]
            ax_loss.plot(xs, ys, color=c, alpha=0.85, linewidth=1.2, label=lab + " train")
        if evals:
            ex = [int(r["step"]) for r in evals]
            ey0 = [float(r["eval_loss_task0"]) for r in evals]
            ey1 = [float(r["eval_loss_task1"]) for r in evals]
            ax_eval.plot(ex, ey0, color=c, marker="o", linestyle="--", markersize=4, label=lab + " eval task0")
            ax_eval.plot(ex, ey1, color=c, marker="s", linestyle=":", markersize=3, alpha=0.8, label=lab + " eval task1")

    ax_loss.set_ylabel("train batch loss")
    ax_loss.legend(loc="upper right", fontsize=8)
    ax_loss.grid(True, alpha=0.3)
    ax_eval.set_xlabel("step")
    ax_eval.set_ylabel("eval mean loss (holdout)")
    ax_eval.legend(loc="upper right", fontsize=7)
    ax_eval.grid(True, alpha=0.3)
    ds = first_meta.get("dataset", "") if first_meta else ""
    tau = first_meta.get("tau", "") if first_meta else ""
    kind = str(first_meta.get("kind", "")) if first_meta else ""
    if "joint_geometry" in kind:
        fig.suptitle(f"Joint geometry CL  {ds}  tau={tau}", fontsize=11)
    else:
        fig.suptitle(f"HF two-task CL  {ds}  tau={tau}", fontsize=11)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("plot_hf_real_cl_log: wrote", out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
