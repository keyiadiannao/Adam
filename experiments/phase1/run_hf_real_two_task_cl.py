# -*- coding: utf-8 -*-
"""
AGNews 或 DBpedia-14 子集 + DistilGPT2 LoRA：双任务交替训练，JSONL 记录训练 loss / vtm，
可选 **holdout eval**（遗忘 proxy：任务 0/1 各自 holdout 上的平均 loss，``--eval-every`` 步写一行）。

``--tau`` 控制 ``make_weak_pd_operators`` 中 gamma 强度（越小阻尼越强，与 AdamW 差异更易显现）。

示例:
  py -3 experiments\\phase1\\run_hf_real_two_task_cl.py --dataset ag_news --run-both --steps 120 --tau 50
  py -3 experiments\\phase1\\run_hf_real_two_task_cl.py --dataset dbpedia_14 --max-per-class 200 --holdout-per-class 30 --eval-every 10
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
from subgeo.optimizer import (  # noqa: E402
    ConcatSubGeoAdam,
    concat_subgeo_m_flat,
    lora_trainable_parameters,
    make_weak_pd_operators,
    momentum_energy_in_subspace,
)


def _eval_mean_loss(
    model: torch.nn.Module,
    tok: object,
    rows: list,
    device: torch.device,
    max_length: int,
    eval_batch_size: int,
) -> float:
    if not rows:
        return float("nan")
    was_training = model.training
    model.eval()
    tot, n = 0.0, 0
    try:
        with torch.no_grad():
            for i in range(0, len(rows), eval_batch_size):
                sub = rows[i : i + eval_batch_size]
                batch = tokenize_batch(tok, sub, list(range(len(sub))), device, max_length)
                out = model(**batch)
                bs = int(batch["labels"].shape[0])
                tot += float(out.loss.detach().cpu().item()) * bs
                n += bs
    finally:
        if was_training:
            model.train()
    return tot / max(n, 1)


def _train_one_run(
    train_t0: list,
    train_t1: list,
    eval_t0: list,
    eval_t1: list,
    args: argparse.Namespace,
    log_path: str,
    adamw_lora: bool,
) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model, tok, _pad_id = load_distilgpt2_peft_lora_seqcls(device)
    model.train()

    lora_params = lora_trainable_parameters(model)
    lora_set = set(lora_params)
    other_params = [p for p in model.parameters() if p.requires_grad and p not in lora_set]
    D = sum(p.numel() for p in lora_params)
    if D == 0:
        raise RuntimeError("no lora_* parameters")

    dtype = next(iter(lora_params)).dtype
    k_sub = min(64, max(1, D // 32))
    V, gamma = make_weak_pd_operators(D, k_sub, device, dtype, tau=float(args.tau))
    V_log = V.detach().clone()

    if adamw_lora:
        opt_lora = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.0)
    else:
        opt_lora = ConcatSubGeoAdam(
            lora_params,
            lr=args.lr,
            weight_decay=0.0,
            V=V,
            gamma=gamma,
            mode="asym",
        )
    opt_other = (
        torch.optim.AdamW(other_params, lr=args.lr, weight_decay=0.01) if other_params else None
    )

    meta = {
        "kind": "hf_real_two_task_cl_meta",
        "dataset": args.dataset,
        "train_per_class": args.max_per_class,
        "holdout_per_class": args.holdout_per_class,
        "tau": float(args.tau),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "eval_every": args.eval_every,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "adamw_lora": adamw_lora,
        "lora_D": D,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "len_train_task0": len(train_t0),
        "len_train_task1": len(train_t1),
        "len_eval_task0": len(eval_t0),
        "len_eval_task1": len(eval_t1),
    }

    B, L = args.batch_size, args.max_length
    if len(train_t0) < B or len(train_t1) < B:
        raise RuntimeError("train pool smaller than batch_size")

    eval_every = int(args.eval_every)
    ebs = int(args.eval_batch_size)

    with open(log_path, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(meta, ensure_ascii=False) + "\n")

        def maybe_eval(step_after: int) -> None:
            if eval_every <= 0 or (not eval_t0 and not eval_t1):
                return
            if (step_after + 1) % eval_every != 0:
                return
            e0 = _eval_mean_loss(model, tok, eval_t0, device, L, ebs)
            e1 = _eval_mean_loss(model, tok, eval_t1, device, L, ebs)
            fp.write(
                json.dumps(
                    {
                        "kind": "eval",
                        "step": step_after,
                        "eval_loss_task0": e0,
                        "eval_loss_task1": e1,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fp.flush()

        for step in range(args.steps):
            task_id = step % 2
            pool = train_t0 if task_id == 0 else train_t1
            t_local = step // 2
            idxs = batch_indices(len(pool), B, t_local)
            batch = tokenize_batch(tok, pool, idxs, device, L)

            opt_lora.zero_grad()
            if opt_other is not None:
                opt_other.zero_grad()
            out = model(**batch)
            loss_val = float(out.loss.detach().cpu().item())
            out.loss.backward()
            opt_lora.step()
            if opt_other is not None:
                opt_other.step()

            if adamw_lora:
                m_flat = concat_adamw_exp_avg_flat(opt_lora, lora_params)
            else:
                m_flat = concat_subgeo_m_flat(opt_lora)
            vtm = momentum_energy_in_subspace(V_log, m_flat)

            fp.write(
                json.dumps(
                    {"step": step, "task": task_id, "loss": loss_val, "vtm_lora": vtm},
                    ensure_ascii=False,
                )
                + "\n"
            )
            maybe_eval(step)


def main() -> int:
    parser = argparse.ArgumentParser(description="HF real-data two-task CL (AGNews / DBpedia-14)")
    parser.add_argument("--dataset", type=str, default="ag_news", choices=("ag_news", "dbpedia_14"))
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=250,
        help="每个原始类别用于 **训练池** 的条数（不含 holdout）",
    )
    parser.add_argument(
        "--holdout-per-class",
        type=int,
        default=0,
        help="每个原始类别划给 **eval** 的条数；>0 且 --eval-every>0 时写 eval 行",
    )
    parser.add_argument("--eval-every", type=int, default=0, help="每 N 步写 eval（0=关闭）")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--tau", type=float, default=1.0e9, help="make_weak_pd_operators 的 tau（越小 gamma 越大）")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--adamw-lora", action="store_true")
    parser.add_argument("--run-both", action="store_true")
    parser.add_argument("--log", type=str, default="")
    args = parser.parse_args()

    ep = configure_hf_hub_endpoint()
    if ep:
        print("HF hub endpoint:", ep)

    try:
        train_t0, train_t1, ev_t0, ev_t1 = build_two_task_pools(
            args.dataset,
            args.max_per_class,
            args.holdout_per_class,
            args.seed,
        )
    except Exception as e:
        print("SKIP: cannot build pools:", repr(e))
        return 0

    ds_tag = args.dataset.replace("/", "_")
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    try:
        if args.run_both:
            p0 = str(log_dir / f"{ds_tag}_real_subgeo_{ts}.jsonl")
            p1 = str(log_dir / f"{ds_tag}_real_adamw_{ts}.jsonl")
            _train_one_run(train_t0, train_t1, ev_t0, ev_t1, args, p0, adamw_lora=False)
            print("hf_real_two_task_cl: SubGeo OK ->", p0)
            _train_one_run(train_t0, train_t1, ev_t0, ev_t1, args, p1, adamw_lora=True)
            print("hf_real_two_task_cl: AdamW OK ->", p1)
        else:
            log_path = args.log.strip()
            if not log_path:
                suffix = "adamw" if args.adamw_lora else "subgeo"
                log_path = str(log_dir / f"{ds_tag}_real_{suffix}_{ts}.jsonl")
            _train_one_run(train_t0, train_t1, ev_t0, ev_t1, args, log_path, adamw_lora=bool(args.adamw_lora))
            print("hf_real_two_task_cl: OK ->", log_path)
    except Exception as e:
        print("SKIP:", repr(e))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
