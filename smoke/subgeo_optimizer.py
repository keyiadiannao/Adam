# -*- coding: utf-8 -*-
"""SubGeo / Geo-Adam 风格单张量优化步：支持 vanilla、grad_damp、sym、asym、reverse。"""
from __future__ import annotations

import torch
from torch.optim import Optimizer


def apply_pd(g_flat: torch.Tensor, V: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """P_d g = g - V (gamma ⊙ (V^T g)). g_flat: (D,), V: (D, k), gamma: (k,)"""
    coeff = V.T @ g_flat
    return g_flat - V @ (gamma * coeff)


class SubGeoAdam(Optimizer):
    """
    仅优化单个展平向量参数（与 Phase0 烟雾一致）。
    mode:
      - vanilla: 标准 AdamW（无 P_d）
      - grad_damp: g <- P_d g 后 m、v 同用（路径 B）
      - sym: g_m = g_v = P_d g
      - asym: g_m = P_d g, g_v = g
      - reverse: g_m = g, g_v = P_d g
    """

    def __init__(
        self,
        params,
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
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            V=V,
            gamma=gamma,
            mode=mode,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            V, gamma = group["V"], group["gamma"]
            mode = group["mode"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                g_flat = g.view(-1)
                if V is None or mode == "vanilla":
                    gm = gv = g_flat
                elif mode == "grad_damp":
                    gf = apply_pd(g_flat, V, gamma)
                    gm = gv = gf
                elif mode == "sym":
                    gf = apply_pd(g_flat, V, gamma)
                    gm = gv = gf
                elif mode == "asym":
                    gm = apply_pd(g_flat, V, gamma)
                    gv = g_flat
                elif mode == "reverse":
                    gm = g_flat
                    gv = apply_pd(g_flat, V, gamma)
                else:
                    raise RuntimeError(mode)

                m_flat = m.view(-1)
                v_flat = v.view(-1)
                m_flat.mul_(b1).add_(gm, alpha=1.0 - b1)
                v_flat.mul_(b2).addcmul_(gv, gv, value=1.0 - b2)

                m_hat = m_flat / (1.0 - b1**t)
                v_hat = v_flat / (1.0 - b2**t)
                upd = lr * m_hat / (v_hat.sqrt() + eps)

                p_flat = p.data.view(-1)
                p_flat.sub_(upd)
                if wd > 0:
                    p_flat.mul_(1.0 - lr * wd)

        return loss


def momentum_project_energy(m: torch.Tensor, V: torch.Tensor) -> float:
    """||V^T m||_2，m 与 V 同设备。"""
    x = V.T @ m.view(-1)
    return float(torch.linalg.norm(x).item())
