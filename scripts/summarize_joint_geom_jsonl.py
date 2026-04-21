# -*- coding: utf-8 -*-
"""
汇总 run_joint_geometry_cl.py 产出的 JSONL，生成一段可直接粘贴到聊天/ issue 的短报告。

用法（服务器）:
  python scripts/summarize_joint_geom_jsonl.py experiments/phase1/logs/ --glob 'joint_geom*.jsonl'
  python scripts/summarize_joint_geom_jsonl.py experiments/phase1/logs/ --glob 'joint_geom*.jsonl' --min-rows 200

落盘 + 终端同时有:
  python scripts/summarize_joint_geom_jsonl.py experiments/phase1/logs/ --glob 'joint_geom*.jsonl' \\
    | tee JOINT_SUMMARY.txt
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
    seen: set[str] = set()
    out: list[Path] = []
    for q in paths:
        s = str(q.resolve())
        if s not in seen:
            seen.add(s)
            out.append(q)
    return out


def _parse_jsonl(path: Path) -> tuple[dict, list[dict], list[dict]]:
    meta: dict = {}
    rows: list[dict] = []
    evals: list[dict] = []
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
            elif isinstance(o, dict) and o.get("kind") == "eval":
                evals.append(o)
            elif isinstance(o, dict) and "step" in o and "loss" in o and "task" in o:
                rows.append(o)
    return meta, rows, evals


def _mean_tail(rows: list[dict], task: int, k: int, key: str) -> float:
    sub = [r for r in rows if int(r["task"]) == task]
    if not sub:
        return float("nan")
    tail = sub[-k:] if len(sub) >= k else sub
    return sum(float(r[key]) for r in tail) / len(tail)


def _config_group_key(meta: dict, n_rows: int) -> tuple:
    """同 key 视为同一实验配置 + 同长度曲线（可配对 SubGeo vs AdamW）。"""
    seed = meta.get("seed", "n/a")
    ho = int(meta.get("holdout_per_class") or 0)
    ev = int(meta.get("eval_every") or 0)
    mpc = int(meta.get("max_per_class") or -1)
    post_mode = str(meta.get("post_task_mode") or "alternate")
    return (
        str(seed),
        str(meta.get("dataset", "?")),
        mpc,
        ho,
        ev,
        post_mode,
        int(meta.get("anchor_steps", -1)),
        int(meta.get("post_steps", -1)),
        int(meta.get("B_grad", -1)),
        int(meta.get("r_sub", -1)),
        float(meta.get("tau", 0.0)),
        int(meta.get("lora_D", -1)),
        int(n_rows),
    )


def _tail_metrics_sig(s: dict) -> tuple:
    return (
        round(float(s["mean_loss_t0_tail"]), 6),
        round(float(s["mean_loss_t1_tail"]), 6),
        round(float(s["mean_vtm_t0_tail"]), 6),
        round(float(s["mean_vtm_t1_tail"]), 6),
    )


def _dedupe_by_metrics(entries: list[dict], label: str) -> tuple[list[dict], list[str]]:
    """指标完全相同的重复文件只保留一条，并记录说明。"""
    notes: list[str] = []
    by_sig: dict[tuple, dict] = {}
    order: list[tuple] = []
    for e in entries:
        sig = _tail_metrics_sig(e)
        if sig not in by_sig:
            by_sig[sig] = e
            order.append(sig)
        else:
            prev = by_sig[sig]["stem"]
            notes.append(f"{label}: 合并重复指标 {prev} ≈ {e['stem']}")
    return [by_sig[s] for s in order], notes


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize joint_geometry_cl JSONL logs")
    ap.add_argument("inputs", nargs="+", help="JSONL 文件或目录（与 --glob）")
    ap.add_argument("--glob", type=str, default=None, help="目录下 glob，默认 joint_geom*.jsonl")
    ap.add_argument("--tail", type=int, default=50, help="每 task 尾窗步数平均")
    ap.add_argument(
        "--min-rows",
        type=int,
        default=0,
        help="忽略训练行数少于此值的文件（例如烟雾用 --min-rows 50）",
    )
    ap.add_argument("--out", type=str, default="", help="写入 UTF-8 文本路径")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    paths = _collect_paths(args.inputs, args.glob)
    if not paths:
        print("No JSONL files found.", file=sys.stderr)
        return 1

    lines: list[str] = []
    lines.append("=== SubGeo joint_geometry_cl 汇总（可整段粘贴）===")
    lines.append(f"repo_git_short: {_try_git_head(root)}")
    lines.append(f"tail_n: {args.tail}  min_rows_filter: {args.min_rows or 'off'}")
    lines.append("")

    summaries: list[dict] = []
    skipped_short: list[str] = []

    for path in paths:
        meta, rows, evals = _parse_jsonl(path)
        if not rows:
            lines.append(f"file: {path.name}")
            lines.append("  ERROR: no train rows")
            lines.append("")
            continue
        if len(rows) < args.min_rows:
            skipped_short.append(f"{path.name} (rows={len(rows)})")
            continue

        adamw = bool(meta.get("adamw_lora", False))
        geo_reg_lambda = float(meta.get("geo_reg_lambda") or 0.0)
        if geo_reg_lambda > 0.0:
            method_bucket = "treatment"
        elif adamw:
            method_bucket = "baseline"
        else:
            method_bucket = "treatment"
        seed = meta.get("seed", "n/a")
        m0 = _mean_tail(rows, 0, args.tail, "loss")
        m1 = _mean_tail(rows, 1, args.tail, "loss")
        v0 = _mean_tail(rows, 0, args.tail, "vtm_lora")
        v1 = _mean_tail(rows, 1, args.tail, "vtm_lora")
        last = rows[-1]
        cfg_key = _config_group_key(meta, len(rows))
        last_eval = evals[-1] if evals else None
        summaries.append(
            {
                "path": str(path),
                "stem": path.name,
                "seed": seed,
                "adamw_lora": adamw,
                "method_bucket": method_bucket,
                "n_steps": len(rows),
                "n_evals": len(evals),
                "cfg_key": cfg_key,
                "meta": meta,
                "mean_loss_t0_tail": m0,
                "mean_loss_t1_tail": m1,
                "mean_vtm_t0_tail": v0,
                "mean_vtm_t1_tail": v1,
                "last_step": int(last["step"]),
                "last_loss": float(last["loss"]),
                "last_task": int(last["task"]),
                "last_eval_e0": float(last_eval["eval_loss_task0"]) if last_eval else None,
                "last_eval_e1": float(last_eval["eval_loss_task1"]) if last_eval else None,
            }
        )
        lines.append(f"file: {path.name}")
        lines.append(f"  seed: {seed}  adamw_lora: {adamw}")
        lines.append(
            f"  meta: dataset={meta.get('dataset')} post_mode={meta.get('post_task_mode', 'alternate')} "
            f"subgeo_mode={meta.get('subgeo_mode', 'asym')} geo_reg_lambda={meta.get('geo_reg_lambda', 0)} "
            f"anchor={meta.get('anchor_steps')} "
            f"post={meta.get('post_steps')} B={meta.get('B_grad')} r={meta.get('r_sub')} tau={meta.get('tau')}"
        )
        lines.append(f"  train_rows: {len(rows)}  last_step={last['step']} task={last['task']}")
        lines.append(f"  mean_loss_task0_last{args.tail}: {m0:.6g}")
        lines.append(f"  mean_loss_task1_last{args.tail}: {m1:.6g}")
        lines.append(f"  mean_vtm_task0_last{args.tail}:   {v0:.6g}")
        lines.append(f"  mean_vtm_task1_last{args.tail}:   {v1:.6g}")
        if last_eval is not None:
            lines.append(
                f"  last_eval(step={last_eval['step']}): "
                f"eval_loss_task0={float(last_eval['eval_loss_task0']):.6g} "
                f"eval_loss_task1={float(last_eval['eval_loss_task1']):.6g}"
            )
        else:
            lines.append("  last_eval: (无 eval 行；可加 --holdout-per-class 与 --eval-every)")
        lines.append("")

    if skipped_short:
        lines.append("--- 已跳过（行数 < min_rows）---")
        for s in skipped_short:
            lines.append(f"  {s}")
        lines.append("")

    if not summaries:
        lines.append("--- (无文件进入统计：提高 min_rows 门槛过严，或目录下无 joint_geom JSONL) ---")
        lines.append("")
        lines.append("=== 以上结束 ===")
        text = "\n".join(lines)
        print(text)
        if args.out.strip():
            Path(args.out).expanduser().write_text(text + "\n", encoding="utf-8")
        return 0

    # 按 (seed, 超参, 行数) 分组配对
    by_cfg: dict[tuple, list[dict]] = defaultdict(list)
    for s in summaries:
        by_cfg[s["cfg_key"]].append(s)

    lines.append("--- Treatment vs Baseline 尾窗 mean_loss 差（同配置、同 train_rows）---")
    lines.append("  dloss = treatment - baseline (负: treatment 尾窗 loss 更低)")
    any_pair = False
    merge_notes: list[str] = []

    for cfg_key in sorted(by_cfg.keys(), key=lambda k: (k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[12])):
        group = by_cfg[cfg_key]
        subgeo = [x for x in group if x["method_bucket"] == "treatment"]
        adamw = [x for x in group if x["method_bucket"] == "baseline"]
        subgeo, n1 = _dedupe_by_metrics(subgeo, "Treatment")
        adamw, n2 = _dedupe_by_metrics(adamw, "Baseline")
        merge_notes.extend(n1)
        merge_notes.extend(n2)

        seed, ds, mpc, ho, ev_every, post_mode, anch, post, bg, rsub, tau, lora_d, n_rows = cfg_key
        hdr = (
            f"  cfg seed={seed} {ds} max_pc={mpc} holdout={ho} eval_every={ev_every} post_mode={post_mode} "
            f"rows={n_rows} anchor={anch} post={post} B={bg} r={rsub} tau={tau} lora_D={lora_d}"
        )

        if len(subgeo) == 1 and len(adamw) == 1:
            any_pair = True
            a, b = subgeo[0], adamw[0]
            d0 = a["mean_loss_t0_tail"] - b["mean_loss_t0_tail"]
            d1 = a["mean_loss_t1_tail"] - b["mean_loss_t1_tail"]
            lines.append(hdr)
            lines.append(f"    treatment_file: {a['stem']}")
            lines.append(f"    baseline_file:  {b['stem']}")
            lines.append(f"    dloss0={d0:+.6g}  dloss1={d1:+.6g}")
            if (
                a.get("last_eval_e0") is not None
                and b.get("last_eval_e0") is not None
                and a.get("last_eval_e1") is not None
                and b.get("last_eval_e1") is not None
            ):
                de0 = a["last_eval_e0"] - b["last_eval_e0"]
                de1 = a["last_eval_e1"] - b["last_eval_e1"]
                lines.append(
                    f"    last_eval d(treatment-baseline): de0={de0:+.6g} de1={de1:+.6g} "
                    f"(holdout 遗忘 proxy；负表示 treatment 更低)"
                )
            lines.append("")
        elif len(subgeo) == 0 or len(adamw) == 0:
            lines.append(hdr)
            lines.append(f"    仅一侧: subgeo={len(subgeo)} adamw={len(adamw)}（跳过）")
            lines.append("")
        else:
            lines.append(hdr)
            lines.append(f"    仍ambiguous: subgeo={len(subgeo)} adamw={len(adamw)}（需删旧日志或指定文件路径）")
            for x in subgeo:
                lines.append(f"      [SubGeo] {x['stem']}  m0={x['mean_loss_t0_tail']:.6g} m1={x['mean_loss_t1_tail']:.6g}")
            for x in adamw:
                lines.append(f"      [AdamW]  {x['stem']}  m0={x['mean_loss_t0_tail']:.6g} m1={x['mean_loss_t1_tail']:.6g}")
            lines.append("")

    if merge_notes:
        lines.append("--- 合并说明（同配置下指标全同的重复文件）---")
        for n in merge_notes:
            lines.append(f"  {n}")
        lines.append("")

    if not any_pair and summaries:
        lines.append(
            "  hint: 若均为 seed n/a，已按「meta+train_rows」分组；"
            "可用 --min-rows 200 忽略短烟雾；或清理 logs 里旧 joint_geom 再汇总。"
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
