# -*- coding: utf-8 -*-
"""
梯度子空间 + 投影 Hessian（Ritz）工具。

约定：
- 仅对传入的 `params`（一般为 LoRA 或子集参数）展平为 D 维向量。
- Hvp 使用一次标量损失对 `params` 的图；每次调用 `loss_fn()` 应重新前向得到新计算图。
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import torch


def apply_pd(g_flat: torch.Tensor, V: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """P_d g = g - V (gamma ⊙ (V^T g))。g_flat: (D,), V: (D, k), gamma: (k,)。"""
    coeff = V.T @ g_flat
    return g_flat - V @ (gamma * coeff)


def flatten_params(params: Iterable[torch.nn.Parameter]) -> Tuple[torch.Tensor, List[torch.Size]]:
    """把参数列表展平为一维向量，并记录各张量形状。"""
    plist = [p for p in params]
    shapes = [p.shape for p in plist]
    parts = [p.reshape(-1) for p in plist]
    if not parts:
        raise ValueError("empty params")
    return torch.cat(parts, dim=0), shapes


def split_flat_to_tensors(flat: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
    out = []
    i = 0
    for sh in shapes:
        n = int(torch.tensor(sh).prod().item())
        out.append(flat[i : i + n].view(sh))
        i += n
    return out


def grad_flat_from_params(params: List[torch.nn.Parameter]) -> torch.Tensor:
    """把各 param.grad 展平拼接（调用方需已 backward）。"""
    parts = []
    for p in params:
        if p.grad is None:
            raise RuntimeError("param has no grad")
        parts.append(p.grad.detach().reshape(-1))
    return torch.cat(parts, dim=0)


def split_flat_to_param_grads(flat_g: torch.Tensor, params: List[torch.nn.Parameter], shapes: List[torch.Size]) -> None:
    """把展平梯度写回 p.grad。"""
    tensors = split_flat_to_tensors(flat_g, shapes)
    for p, g in zip(params, tensors):
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)


def collect_grad_flat_matrix(
    loss_fn: Callable[[], torch.Tensor],
    params: List[torch.nn.Parameter],
    n_batches: int,
    zero_grad_fn: Callable[[], None] | None = None,
) -> torch.Tensor:
    """
    连续调用 n 次 loss_fn（外部负责换 batch），每次 backward，收集一列梯度。
    返回 G: (D, n_batches)。
    """
    shapes = [p.shape for p in params]
    cols: List[torch.Tensor] = []
    for _ in range(n_batches):
        if zero_grad_fn is not None:
            zero_grad_fn()
        else:
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
        loss = loss_fn()
        if not loss.requires_grad:
            raise RuntimeError("loss does not require grad; check model / params linkage")
        loss.backward()
        cols.append(grad_flat_from_params(params))
    return torch.stack(cols, dim=1)


def subspace_from_G(G: torch.Tensor, r: int) -> torch.Tensor:
    """
    对 G 做截断 SVD，取前 r 个左奇异向量，QR 得到列正交 V_grad: (D, r)。
    """
    if r < 1:
        raise ValueError("r >= 1")
    D, B = G.shape
    r_eff = min(r, D, B)
    U, _, _ = torch.linalg.svd(G, full_matrices=False)
    U_r = U[:, :r_eff]
    Q, _ = torch.linalg.qr(U_r, mode="reduced")
    return Q[:, : min(r, Q.shape[1])]


def hvp_flat(
    loss: torch.Tensor,
    params: List[torch.nn.Parameter],
    v_flat: torch.Tensor,
    shapes: List[torch.Size],
) -> torch.Tensor:
    """对展平方向 v_flat 计算 Hvp（向量与 params 展平同维）。"""
    v_parts = split_flat_to_tensors(v_flat, shapes)
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    dot = torch.zeros((), device=loss.device, dtype=loss.dtype)
    for g, vi in zip(grads, v_parts):
        dot = dot + (g * vi).sum()
    hv = torch.autograd.grad(dot, params, retain_graph=False, allow_unused=False)
    return torch.cat([h.reshape(-1) for h in hv])


def projected_hessian(
    loss_fn: Callable[[], torch.Tensor],
    params: List[torch.nn.Parameter],
    V: torch.Tensor,
    shapes: List[torch.Size],
) -> torch.Tensor:
    """
    计算 H_proj = V^T H V，大小 (r, r)。
    第 j 列为 V^T (H @ v_j)，其中 v_j = V[:, j] 展平后按 shapes 拆开参与 hvp。
    """
    r = V.shape[1]
    cols = []
    for j in range(r):
        v_flat = V[:, j].contiguous()
        loss = loss_fn()
        hv_flat = hvp_flat(loss, params, v_flat, shapes)
        cols.append(V.T @ hv_flat)
    return torch.stack(cols, dim=1)


def build_gamma(lambdas: torch.Tensor, tau: float, lam_eps: float = 1e-12) -> torch.Tensor:
    """gamma_i = min(1, lambda_i/tau) 若 lambda_i > lam_eps，否则 0（含非正特征值）。"""
    out = torch.zeros_like(lambdas)
    pos = lambdas > lam_eps
    out[pos] = torch.clamp(lambdas[pos] / tau, max=1.0)
    return out
