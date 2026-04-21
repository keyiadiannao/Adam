# -*- coding: utf-8 -*-
"""
在 **多个 LoRA 参数张量** 按固定顺序展平后的 D 维空间上应用同一组 (V, gamma)，
实现与单张量 SubGeo-Adam 等价的 g_m / g_v 分路（Concat 版）。

典型用法：PEFT 注入后，按 name 排序收集所有 `lora_*` 可训练参数，构造 D=sum(numel) 的 V:(D,k)。
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
from torch.optim import Optimizer

from .geometry import apply_pd


def make_weak_pd_operators(
    D: int,
    k: int,
    device: torch.device,
    dtype: torch.dtype,
    tau: float = 1.0e9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    烟雾/预热用：随机列正交 V，gamma = min(1, 1/tau) 全维相同 => 接近恒等阻尼。
    """
    k_eff = max(1, min(k, D))
    V, _ = torch.linalg.qr(torch.randn(D, k_eff, device=device, dtype=dtype), mode="reduced")
    lam = torch.ones(V.shape[1], device=device, dtype=dtype)
    gamma = torch.clamp(lam / tau, max=1.0)
    return V, gamma


class ConcatSubGeoAdam(Optimizer):
    """
    单 param_group，内含按 **固定顺序** 排列的若干 Parameter；梯度先拼接再施加 P_d。
    """

    def __init__(
        self,
        params: Sequence[torch.nn.Parameter],
        lr: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        V: torch.Tensor | None = None,
        gamma: torch.Tensor | None = None,
        mode: str = "asym",
    ):
        if mode not in ("vanilla", "grad_damp", "sym", "asym", "reverse"):
            raise ValueError(mode)
        plist = list(params)
        if not plist:
            raise ValueError("ConcatSubGeoAdam: empty params")
        D = sum(p.numel() for p in plist)
        if V is not None and V.shape[0] != D:
            raise ValueError(f"V rows {V.shape[0]} != flattened D {D}")
        if V is not None and gamma is not None and gamma.numel() != V.shape[1]:
            raise ValueError("gamma length must match V columns")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            V=V,
            gamma=gamma,
            mode=mode,
        )
        super().__init__(plist, defaults)
        self._plist: List[torch.nn.Parameter] = plist
        self._numels = [int(p.numel()) for p in plist]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        b1, b2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]
        V, gamma = group["V"], group["gamma"]
        mode = group["mode"]
        plist: List[torch.nn.Parameter] = group["params"]

        chunks: List[torch.Tensor] = []
        for p in plist:
            if p.grad is None:
                raise RuntimeError("ConcatSubGeoAdam: missing grad on a parameter")
            chunks.append(p.grad.detach().view(-1))
        g_total = torch.cat(chunks)

        if V is None or mode == "vanilla":
            gm_total = gv_total = g_total
        elif mode == "grad_damp":
            t = apply_pd(g_total, V, gamma)
            gm_total = gv_total = t
        elif mode == "sym":
            t = apply_pd(g_total, V, gamma)
            gm_total = gv_total = t
        elif mode == "asym":
            gm_total = apply_pd(g_total, V, gamma)
            gv_total = g_total
        elif mode == "reverse":
            gm_total = g_total
            gv_total = apply_pd(g_total, V, gamma)
        else:
            raise RuntimeError(mode)

        offset = 0
        for p in plist:
            n = p.numel()
            sl = slice(offset, offset + n)
            gm = gm_total[sl].view_as(p)
            gv = gv_total[sl].view_as(p)
            offset += n

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["m"] = torch.zeros_like(p)
                state["v"] = torch.zeros_like(p)
            m, v = state["m"], state["v"]
            state["step"] = state["step"] + 1
            t = state["step"]

            m_flat = m.view(-1)
            v_flat = v.view(-1)
            m_flat.mul_(b1).add_(gm.view(-1), alpha=1.0 - b1)
            v_flat.mul_(b2).addcmul_(gv.view(-1), gv.view(-1), value=1.0 - b2)

            m_hat = m_flat / (1.0 - b1**t)
            v_hat = v_flat / (1.0 - b2**t)
            upd = lr * m_hat / (v_hat.sqrt() + eps)

            p_flat = p.data.view(-1)
            p_flat.sub_(upd)
            if wd > 0:
                p_flat.mul_(1.0 - lr * wd)

        return loss


def lora_trainable_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """按 name 排序，收集 requires_grad 且名中含 lora 的参数（PEFT 惯例）。"""
    items = [(n, p) for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower()]
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def concat_subgeo_m_flat(opt: ConcatSubGeoAdam) -> torch.Tensor:
    """拼接 ConcatSubGeoAdam 各参数的一阶动量 m（与梯度拼接顺序一致）。"""
    plist: List[torch.nn.Parameter] = list(opt.param_groups[0]["params"])
    chunks: List[torch.Tensor] = []
    for p in plist:
        st = opt.state.get(p)
        if st is None or "m" not in st:
            raise RuntimeError("concat_subgeo_m_flat: need at least one optimizer.step before reading m")
        chunks.append(st["m"].detach().reshape(-1))
    return torch.cat(chunks)


def momentum_energy_in_subspace(V: torch.Tensor | None, m_flat: torch.Tensor) -> float:
    """返回 ||V^T m||_2；V 为 None 时返回 0.0（无子空间参照）。"""
    if V is None:
        return 0.0
    x = V.T @ m_flat.reshape(-1)
    return float(torch.linalg.norm(x).item())
