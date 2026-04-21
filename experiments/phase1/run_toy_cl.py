# -*- coding: utf-8 -*-
"""
两任务 toy CL：Task A 收敛到锚点 -> 用 Task B 在锚点的一阶梯度构造 1 维 V 与 gamma
-> Task B 训练对比 vanilla AdamW vs SubGeo asym。

运行（仓库根为 cwd）:
  conda run -n base --cwd d:\\cursor_try\\Evidence python experiments\\phase1\\run_toy_cl.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from smoke.subgeo_optimizer import SubGeoAdam, momentum_project_energy


def train_task_a_to_anchor(D: int, a_star: torch.Tensor, steps: int, lr: float) -> torch.Tensor:
    w = torch.zeros(D, dtype=a_star.dtype, device=a_star.device, requires_grad=True)
    opt = torch.optim.AdamW([w], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.5 * torch.sum((w - a_star) ** 2)
        loss.backward()
        opt.step()
    return w.detach()


def grad_task_b(w: torch.Tensor, b_star: torch.Tensor) -> torch.Tensor:
    """B 损失 0.5||w - b_star||^2 的梯度。"""
    return w - b_star


def build_one_dim_pd_not_parallel_to_b_grad(w_anchor: torch.Tensor, b_star: torch.Tensor, gamma_val: float):
    """
    构造 (D,1) 的 V：与 Task B 在锚点的梯度 **正交**。

    若 V 与 g_B 共线且 γ=1，则 P_d g≈0，Task B 无法更新（这是好的反例，但不适合当「可收敛 toy」）。
    此处模拟「旧任务敏感方向」与当前 B 梯度不完全对齐的常见情形。
    """
    g = grad_task_b(w_anchor, b_star)
    if float(torch.linalg.norm(g).item()) < 1e-12:
        raise RuntimeError("zero gradient at anchor")
    g_unit = g / torch.linalg.norm(g)
    # 随机向量去掉沿 g 的分量，再单位化
    r = torch.randn_like(g)
    v = r - (r @ g_unit) * g_unit
    v = v / torch.linalg.norm(v)
    V = v.unsqueeze(1)
    lam = torch.tensor([1.0], device=w_anchor.device, dtype=w_anchor.dtype)
    gamma = torch.tensor([min(1.0, float(gamma_val))], device=w_anchor.device, dtype=w_anchor.dtype)
    return V, gamma


def run_b(
    w_init: torch.Tensor,
    b_star: torch.Tensor,
    V: torch.Tensor,
    gamma: torch.Tensor,
    mode: str,
    steps: int,
    lr: float,
):
    w = w_init.clone().requires_grad_(True)
    if mode == "vanilla":
        opt: torch.optim.Optimizer = torch.optim.AdamW([w], lr=lr, weight_decay=0.0)
    else:
        opt = SubGeoAdam([w], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, V=V, gamma=gamma, mode="asym")

    losses = []
    energies = []
    for _ in range(steps):
        if mode == "vanilla":
            opt.zero_grad()
            loss = 0.5 * torch.sum((w - b_star) ** 2)
            loss.backward()
            opt.step()
        else:
            loss = 0.5 * torch.sum((w - b_star) ** 2)
            w.grad = grad_task_b(w.detach(), b_star)
            opt.step()
            w.grad = None
        losses.append(float(loss.detach().item()))
        if mode != "vanilla":
            energies.append(momentum_project_energy(opt.state[w]["m"], V))
        else:
            energies.append(0.0)
    return losses, energies


def main():
    torch.manual_seed(7)
    device = torch.device("cpu")
    dtype = torch.float64
    D = 32
    a_star = torch.randn(D, device=device, dtype=dtype)
    b_star = torch.randn(D, device=device, dtype=dtype)
    # 故意让 b 与 a 不完全共线，以便有非平凡子空间
    b_star = b_star + 0.3 * a_star

    w_anchor = train_task_a_to_anchor(D, a_star, steps=600, lr=0.05)
    err_a = float(torch.linalg.norm(w_anchor - a_star).item())
    print("After Task A: ||w - a_star|| = %.4e" % err_a)

    V, gamma = build_one_dim_pd_not_parallel_to_b_grad(w_anchor, b_star, gamma_val=0.85)
    print("gamma[0]=%.4f (damp along V orthog to B-grad at anchor)" % float(gamma[0].item()))

    steps_b = 200
    lr_b = 0.08
    Lv, _ = run_b(w_anchor, b_star, V, gamma, "vanilla", steps_b, lr_b)
    La, Ea = run_b(w_anchor, b_star, V, gamma, "asym", steps_b, lr_b)

    print("Task B loss end: vanilla=%.6e asym=%.6e" % (Lv[-1], La[-1]))
    print("Task B loss best: vanilla=%.6e asym=%.6e" % (min(Lv), min(La)))
    print("||V^T m|| end (asym): %.6e" % Ea[-1])

    # 门控：两者均应明显降低 Task B 损失；asym 不应出现 NaN/爆炸
    assert math.isfinite(La[-1]) and math.isfinite(Lv[-1])
    assert La[-1] < 1e-2 and Lv[-1] < 1e-2, (La[-1], Lv[-1])
    assert Ea[-1] < 1e3, "momentum projection energy exploded"

    print("toy_cl: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
