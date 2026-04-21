# -*- coding: utf-8 -*-
"""HF 训练日志用标量：拼接动量、子空间能量 ||V^T m||。"""
from __future__ import annotations

from typing import List

import torch
from torch.optim import Optimizer


def concat_adamw_exp_avg_flat(opt: Optimizer, params: List[torch.nn.Parameter]) -> torch.Tensor:
    """Adam / AdamW 一阶矩 exp_avg，按 params 顺序拼接为 (D,)。"""
    chunks: List[torch.Tensor] = []
    for p in params:
        st = opt.state[p]
        if "exp_avg" not in st:
            raise RuntimeError("concat_adamw_exp_avg_flat: run at least one optimizer.step first")
        chunks.append(st["exp_avg"].detach().reshape(-1))
    return torch.cat(chunks)
