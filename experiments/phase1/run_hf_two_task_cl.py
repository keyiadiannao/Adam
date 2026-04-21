# -*- coding: utf-8 -*-
"""
DistilGPT2 + LoRA 双任务交替烟雾（合成 batch）：日志 loss 与 LoRA 动量在固定随机子空间 V 上的能量 ||V^T m||。

任务 0/1：不同随机种子生成 input_ids，标签分别为全 0 / 全 1（模拟任务切换，非真实 AGNews）。

运行（仓库根，需 requirements_hf.txt）:
  py -3 experiments\\phase1\\run_hf_two_task_cl.py --steps 30
基线（LoRA 也走 AdamW，仍用同一 V 记录 ||V^T m|| 作对照）:
  py -3 experiments\\phase1\\run_hf_two_task_cl.py --adamw-lora
"""
from __future__ import annotations

import argparse
import json
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
from subgeo.optimizer import (  # noqa: E402
    ConcatSubGeoAdam,
    concat_subgeo_m_flat,
    lora_trainable_parameters,
    make_weak_pd_operators,
    momentum_energy_in_subspace,
)


def _synthetic_batch(tok, device, B: int, L: int, task_id: int, step: int) -> dict:
    g = torch.Generator()
    g.manual_seed(10_000 + task_id * 100_000 + step)
    input_ids = torch.randint(0, tok.vocab_size, (B, L), generator=g).to(device)
    attention_mask = torch.ones(B, L, dtype=torch.long, device=device)
    labels = torch.full((B,), task_id, dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main() -> int:
    parser = argparse.ArgumentParser(description="HF two-task alternating CL smoke + JSONL log")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--adamw-lora", action="store_true", help="LoRA 用 AdamW 基线（仍记录同一 V 下的 ||V^T m||）")
    parser.add_argument(
        "--log",
        type=str,
        default="",
        help="JSONL 路径；默认写入 experiments/phase1/logs/",
    )
    args = parser.parse_args()

    ep = configure_hf_hub_endpoint()
    if ep:
        print("HF hub endpoint:", ep)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    try:
        model, tok, _pad_id = load_distilgpt2_peft_lora_seqcls(device)
    except Exception as e:
        print("SKIP: cannot load model:", repr(e))
        return 0

    model.train()
    lora_params = lora_trainable_parameters(model)
    lora_set = set(lora_params)
    other_params = [p for p in model.parameters() if p.requires_grad and p not in lora_set]
    D = sum(p.numel() for p in lora_params)
    if D == 0:
        print("SKIP: no lora_* trainable parameters")
        return 0

    dtype = next(iter(lora_params)).dtype
    k_sub = min(64, max(1, D // 32))
    V, gamma = make_weak_pd_operators(D, k_sub, device, dtype, tau=1.0e9)
    V_log = V.detach().clone()

    if args.adamw_lora:
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

    log_path = args.log.strip()
    if not log_path:
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_path = str(log_dir / f"two_task_{ts}.jsonl")

    meta = {
        "kind": "hf_two_task_cl_meta",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "adamw_lora": bool(args.adamw_lora),
        "lora_D": D,
        "hf_endpoint": ep,
    }

    B, L = args.batch_size, args.seq_len
    with open(log_path, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for step in range(args.steps):
            task_id = step % 2
            opt_lora.zero_grad()
            if opt_other is not None:
                opt_other.zero_grad()
            batch = _synthetic_batch(tok, device, B, L, task_id, step)
            out = model(**batch)
            loss_val = float(out.loss.detach().cpu().item())
            out.loss.backward()
            opt_lora.step()
            if opt_other is not None:
                opt_other.step()

            if args.adamw_lora:
                m_flat = concat_adamw_exp_avg_flat(opt_lora, lora_params)
            else:
                m_flat = concat_subgeo_m_flat(opt_lora)
            vtm = momentum_energy_in_subspace(V_log, m_flat)

            row = {
                "step": step,
                "task": task_id,
                "loss": loss_val,
                "vtm_lora": vtm,
            }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("hf_two_task_cl: OK, wrote", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
