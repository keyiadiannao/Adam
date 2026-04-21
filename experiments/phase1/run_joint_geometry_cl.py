# -*- coding: utf-8 -*-
"""
锚点训练（仅任务 0）→ 在锚点提取联合几何 (V_k, gamma) → 双任务交替训练（SubGeo 或 AdamW）。

用于在 **真实 Hessian 截面 + 梯度子空间** 下初步观察相对随机 V 的差异；适合单机/服务器 Phase1。

示例:
  py -3 experiments\\phase1\\run_joint_geometry_cl.py --anchor-steps 30 --post-steps 40 --B-grad 8 --r-sub 8 --tau 50
  py -3 experiments\\phase1\\run_joint_geometry_cl.py --adamw-lora --post-steps 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402

from hf_hub_endpoint import configure_hf_hub_endpoint  # noqa: E402
from hf_lora_model import load_distilgpt2_peft_lora_seqcls  # noqa: E402
from hf_metrics import concat_adamw_exp_avg_flat  # noqa: E402
from real_two_task_data import batch_indices, build_two_task_pools, tokenize_batch  # noqa: E402
from subgeo.joint_geometry import extract_joint_vk_gamma  # noqa: E402
from subgeo.optimizer import (  # noqa: E402
    ConcatSubGeoAdam,
    concat_subgeo_m_flat,
    lora_trainable_parameters,
    momentum_energy_in_subspace,
)


def _make_batch(tok, rows, idxs, device, max_length):
    return tokenize_batch(tok, rows, idxs, device, max_length)


def main() -> int:
    parser = argparse.ArgumentParser(description="Joint geometry at anchor + two-task CL")
    parser.add_argument("--dataset", type=str, default="ag_news", choices=("ag_news", "dbpedia_14"))
    parser.add_argument("--max-per-class", type=int, default=120)
    parser.add_argument("--holdout-per-class", type=int, default=0)
    parser.add_argument("--anchor-steps", type=int, default=25)
    parser.add_argument("--post-steps", type=int, default=50)
    parser.add_argument("--B-grad", type=int, default=8, help="收集 G 的 batch 数（≥2，建议 ≥max(2r,64) 时改大）")
    parser.add_argument("--r-sub", type=int, default=8, help="梯度子空间截断秩")
    parser.add_argument("--tau", type=float, default=80.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--adamw-lora", action="store_true", help="提取后 LoRA 仍用 AdamW 基线")
    parser.add_argument("--save-vk", type=str, default="", help="可选：保存 V_k/gamma/lambdas 的 .pt 路径")
    parser.add_argument("--log", type=str, default="", help="JSONL 路径")
    args = parser.parse_args()

    ep = configure_hf_hub_endpoint()
    if ep:
        print("HF hub endpoint:", ep)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    try:
        train_t0, train_t1, _, _ = build_two_task_pools(
            args.dataset,
            args.max_per_class,
            args.holdout_per_class,
            args.seed,
        )
    except Exception as e:
        print("SKIP: pools:", repr(e))
        return 0

    Bg = max(2, args.B_grad)
    if len(train_t0) < Bg + args.batch_size:
        print("SKIP: train_t0 too small for B_grad and training")
        return 0

    try:
        model, tok, _ = load_distilgpt2_peft_lora_seqcls(device)
    except Exception as e:
        print("SKIP: model:", repr(e))
        return 0

    model.train()
    lora_params = lora_trainable_parameters(model)
    lora_set = set(lora_params)
    other_params = [p for p in model.parameters() if p.requires_grad and p not in lora_set]
    if not lora_params:
        print("SKIP: no lora")
        return 0

    opt_anchor = torch.optim.AdamW(
        lora_params + other_params,
        lr=args.lr,
        weight_decay=0.01 if other_params else 0.0,
    )
    L = args.max_length
    bs = args.batch_size

    # --- 锚点：仅 task0 ---
    for s in range(args.anchor_steps):
        opt_anchor.zero_grad()
        idxs = batch_indices(len(train_t0), bs, s)
        batch = _make_batch(tok, train_t0, idxs, device, L)
        loss = model(**batch).loss
        loss.backward()
        opt_anchor.step()

    # --- 构造 Bg 个互不重叠的 task0 batch 用于 G（环形取样）---
    batches_g = []
    for j in range(Bg):
        idxs = batch_indices(len(train_t0), bs, args.anchor_steps + j)
        batches_g.append(_make_batch(tok, train_t0, idxs, device, L))
    hvp_batch = batches_g[0]

    closures_g: list = []
    for b in batches_g:
        closures_g.append(lambda b=b: model(**b).loss)

    hb = hvp_batch
    loss_hvp = lambda: model(**hb).loss

    try:
        V_k, lambdas, gamma = extract_joint_vk_gamma(closures_g, loss_hvp, lora_params, args.r_sub, float(args.tau))
    except Exception as e:
        print("SKIP: extract_joint_vk_gamma:", repr(e))
        return 0

    V_k = V_k.detach().to(device=device, dtype=next(iter(lora_params)).dtype)
    gamma = gamma.detach().to(device=device, dtype=V_k.dtype)
    V_log = V_k.clone()
    D = V_k.shape[0]
    if args.save_vk:
        payload = {
            "V_k": V_k.cpu(),
            "gamma": gamma.cpu(),
            "lambdas": lambdas.detach().cpu(),
            "tau": float(args.tau),
            "r_sub": int(args.r_sub),
            "B_grad": int(Bg),
        }
        vk_path = Path(args.save_vk).expanduser()
        vk_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(vk_path))
        print("saved vk payload ->", vk_path)

    if args.adamw_lora:
        opt_lora = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.0)
    else:
        opt_lora = ConcatSubGeoAdam(
            lora_params,
            lr=args.lr,
            weight_decay=0.0,
            V=V_k,
            gamma=gamma,
            mode="asym",
        )
    opt_other = torch.optim.AdamW(other_params, lr=args.lr, weight_decay=0.01) if other_params else None

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = args.log.strip() or str(log_dir / f"joint_geom_{args.dataset}_{ts}.jsonl")

    meta = {
        "kind": "joint_geometry_cl_meta",
        "dataset": args.dataset,
        "anchor_steps": args.anchor_steps,
        "post_steps": args.post_steps,
        "B_grad": Bg,
        "r_sub": args.r_sub,
        "tau": float(args.tau),
        "lora_D": D,
        "adamw_lora": bool(args.adamw_lora),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
    }

    with open(log_path, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for step in range(args.post_steps):
            task_id = step % 2
            pool = train_t0 if task_id == 0 else train_t1
            t_local = step // 2
            idxs = batch_indices(len(pool), bs, t_local)
            batch = _make_batch(tok, pool, idxs, device, L)
            opt_lora.zero_grad()
            if opt_other is not None:
                opt_other.zero_grad()
            out = model(**batch)
            lv = float(out.loss.detach().cpu().item())
            out.loss.backward()
            opt_lora.step()
            if opt_other is not None:
                opt_other.step()
            if args.adamw_lora:
                m_flat = concat_adamw_exp_avg_flat(opt_lora, lora_params)
            else:
                m_flat = concat_subgeo_m_flat(opt_lora)
            vtm = momentum_energy_in_subspace(V_log, m_flat)
            fp.write(
                json.dumps({"step": step, "task": task_id, "loss": lv, "vtm_lora": vtm}, ensure_ascii=False)
                + "\n"
            )

    print("joint_geometry_cl: OK ->", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
