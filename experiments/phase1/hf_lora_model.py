# -*- coding: utf-8 -*-
"""DistilGPT2 + PEFT LoRA 序列分类：供 Phase1 HF 脚本复用。"""
from __future__ import annotations

from typing import Any

import torch


def load_distilgpt2_peft_lora_seqcls(
    device: torch.device,
    model_id: str = "distilgpt2",
) -> tuple[torch.nn.Module, Any, int]:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = int(tok.pad_token_id) if tok.pad_token_id is not None else int(tok.eos_token_id)

    cfg = AutoConfig.from_pretrained(model_id)
    cfg.num_labels = 2
    cfg.id2label = {0: "neg", 1: "pos"}
    cfg.label2id = {"neg": 0, "pos": 1}
    cfg.pad_token_id = pad_id
    load_kw: dict = {
        "config": cfg,
        "torch_dtype": torch.float32,
        "ignore_mismatched_sizes": True,
    }
    # 避免 SDPA 融合路径在 Hvp 二阶反传上缺失算子；旧版 transformers 无此参数则忽略。
    try:
        base = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            attn_implementation="eager",
            **load_kw,
        )
    except TypeError:
        base = AutoModelForSequenceClassification.from_pretrained(model_id, **load_kw)
    base.config.pad_token_id = pad_id

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["c_attn", "c_proj"],
    )
    model = get_peft_model(base, lora)
    model.config.pad_token_id = pad_id
    model.to(device)
    return model, tok, pad_id
