# -*- coding: utf-8 -*-
"""
锚点处联合几何：梯度矩阵 G → V_grad（SVD+QR）→ 投影 Hessian H_proj → 特征分解 → V_k 与 gamma。

``loss_batches_for_G``：长度 B 的可调用对象，各调用一次返回当前 batch 的标量损失（已接入 model）。
``loss_for_hvp``：每次调用返回 **同一** batch 的损失，供 r 次 Hvp 估计 H（与文档约定一致）。
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import torch

from .geometry import build_gamma, grad_flat_from_params, projected_hessian, subspace_from_G


def collect_G_columns(
    loss_batches_for_G: Sequence[Callable[[], torch.Tensor]],
    params: List[torch.nn.Parameter],
    zero_grad_fn: Callable[[], None],
) -> torch.Tensor:
    cols: List[torch.Tensor] = []
    for fn in loss_batches_for_G:
        zero_grad_fn()
        loss = fn()
        if not loss.requires_grad:
            raise RuntimeError("loss must require grad for gradient collection")
        loss.backward()
        cols.append(grad_flat_from_params(params))
    return torch.stack(cols, dim=1)


def extract_joint_vk_gamma(
    loss_batches_for_G: Sequence[Callable[[], torch.Tensor]],
    loss_for_hvp: Callable[[], torch.Tensor],
    params: List[torch.nn.Parameter],
    r_sub: int,
    tau: float,
    lam_eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回:
      V_k: (D, r_eff) 列正交
      lambdas: (r_eff,) Ritz 特征值（升序或需调用方排序；此处为 eigh 升序后取反序排列为降序）
      gamma: (r_eff,) 阻尼系数
    """
    if len(loss_batches_for_G) < 1:
        raise ValueError("need at least one batch for G")

    def zg() -> None:
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

    G = collect_G_columns(loss_batches_for_G, params, zg)
    V_grad = subspace_from_G(G, r_sub)
    r_eff = V_grad.shape[1]
    shapes = [p.shape for p in params]

    def hvp_loss_fn() -> torch.Tensor:
        return loss_for_hvp()

    H_proj = projected_hessian(hvp_loss_fn, params, V_grad, shapes)
    # 对称阵特征分解；eigh 返回升序特征值
    evals, evecs = torch.linalg.eigh(H_proj.to(torch.float64))
    idx = torch.arange(evals.numel() - 1, -1, -1, device=evals.device)
    lambdas = evals[idx].to(dtype=V_grad.dtype)
    Q = evecs[:, idx].to(dtype=V_grad.dtype)
    V_k = V_grad @ Q
    gamma = build_gamma(lambdas, tau=tau, lam_eps=lam_eps)
    return V_k, lambdas, gamma
