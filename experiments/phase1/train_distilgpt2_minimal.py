# -*- coding: utf-8 -*-
"""
最小 HF 烟雾：DistilGPT2 + LoRA，在拼接 LoRA 空间上用 ConcatSubGeoAdam（弱阻尼 V,gamma）；
分类头等非 LoRA 可训参数仍用 AdamW。

镜像：见 hf_hub_endpoint.configure_hf_hub_endpoint（EVIDENCE_HF_MIRROR / HF_ENDPOINT 等）。

无网络时：若本地无缓存权重会失败，脚本捕获后 exit 0 并提示。

运行（需先 pip install -r experiments/phase1/requirements_hf.txt）:
  conda run -n base --cwd d:\\cursor_try\\Evidence python experiments\\phase1\\train_distilgpt2_minimal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from hf_hub_endpoint import configure_hf_hub_endpoint  # noqa: E402
from hf_lora_model import load_distilgpt2_peft_lora_seqcls  # noqa: E402


def main() -> int:
    ep = configure_hf_hub_endpoint()
    if ep:
        print("HF hub endpoint:", ep)

    try:
        import torch
        from subgeo.optimizer import ConcatSubGeoAdam, lora_trainable_parameters, make_weak_pd_operators
    except ImportError as e:
        print("SKIP: torch/subgeo not installed:", e)
        return 0

    device = torch.device("cpu")
    try:
        model, tok, _pad_id = load_distilgpt2_peft_lora_seqcls(device)
    except ImportError as e:
        print("SKIP: transformers/peft not installed:", e)
        return 0
    except Exception as e:
        print("SKIP: cannot load pretrained weights (offline/SSL/cache):", repr(e))
        return 0

    model.train()

    lora_params = lora_trainable_parameters(model)
    lora_set = set(lora_params)
    other_params = [p for p in model.parameters() if p.requires_grad and p not in lora_set]
    D = sum(p.numel() for p in lora_params)
    if D == 0:
        print("SKIP: no lora_* trainable parameters found")
        return 0
    dtype = next(iter(lora_params)).dtype
    k_sub = min(64, max(1, D // 32))
    V, gamma = make_weak_pd_operators(D, k_sub, device, dtype, tau=1.0e9)
    opt_lora = ConcatSubGeoAdam(
        lora_params,
        lr=3e-4,
        weight_decay=0.0,
        V=V,
        gamma=gamma,
        mode="asym",
    )
    opt_other = (
        torch.optim.AdamW(other_params, lr=3e-4, weight_decay=0.01) if other_params else None
    )

    B, L = 4, 32
    for step in range(3):
        opt_lora.zero_grad()
        if opt_other is not None:
            opt_other.zero_grad()
        g = torch.Generator()
        g.manual_seed(step)
        batch = {
            "input_ids": torch.randint(0, tok.vocab_size, (B, L), generator=g).to(device),
            "attention_mask": torch.ones(B, L, dtype=torch.long, device=device),
            "labels": torch.randint(0, 2, (B,), device=device),
        }
        out = model(**batch)
        out.loss.backward()
        opt_lora.step()
        if opt_other is not None:
            opt_other.step()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("distilgpt2_lora_minimal: OK, trainable_params=%d, lora_D=%d" % (n_trainable, D))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
