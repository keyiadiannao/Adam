# -*- coding: utf-8 -*-
"""
Phase0 烟雾：验证 P_d 实现、SubGeoAdam 各 mode 第一步与长轨迹（弱阻尼贴近 vanilla）。

在仓库根目录执行:
  py -3 smoke/run_smoke.py

退出码 0 表示门控通过。说明：在**强阻尼**的纯凸二次型上，刻意压制 Hessian
主方向会破坏收敛，因此**不要求** asym 优于 vanilla/sym；此处用 **弱阻尼 + 解析对照** 做 CI 门控。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from smoke.subgeo_optimizer import SubGeoAdam, apply_pd, momentum_project_energy


def test_apply_pd_matches_bruteforce():
    torch.manual_seed(0)
    D, k = 32, 5
    V, _ = torch.linalg.qr(torch.randn(D, k), mode="reduced")
    g = torch.randn(D)
    gamma = torch.rand(k).clamp(0.01, 0.99)
    y1 = apply_pd(g, V, gamma)
    coeff = V.T @ g
    y2 = g - V @ (gamma * coeff)
    err = float(torch.linalg.norm(y1 - y2).item())
    assert err < 1e-10, err


def test_first_step_asym_differs_from_sym_when_gamma_positive():
    """gamma 非零时，第一步更新量应对 asym / sym 有区分。"""
    torch.manual_seed(1)
    D, k = 16, 3
    V, _ = torch.linalg.qr(torch.randn(D, k), mode="reduced")
    lam = torch.tensor([10.0, 3.0, 1.0])
    tau = 5.0
    gamma = torch.clamp(lam / tau, max=1.0)
    w = torch.randn(D)
    g = torch.randn(D)

    def one_step(mode: str):
        p = w.clone()
        opt = SubGeoAdam([p], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, V=V, gamma=gamma, mode=mode)
        p.grad = g.clone()
        opt.step()
        return p.clone()

    p_sym = one_step("sym")
    p_asym = one_step("asym")
    diff = float(torch.linalg.norm(p_sym - p_asym).item())
    assert diff > 1e-8, "sym and asym first steps unexpectedly identical"


def test_weak_damping_tracks_vanilla():
    """tau 极大 => gamma≈0，asym/sym/grad_damp 轨迹应接近 vanilla。"""
    device, dtype = torch.device("cpu"), torch.float64
    D, k = 48, 4
    gen = torch.Generator(device=device)
    gen.manual_seed(2)
    A = torch.randn(D, D, generator=gen, device=device, dtype=dtype)
    V, _ = torch.linalg.qr(A[:, :k], mode="reduced")
    lam = torch.tensor([200.0, 50.0, 5.0, 1.0], device=device, dtype=dtype)
    c = torch.randn(k, generator=gen, device=device, dtype=dtype)
    b = V @ c
    H = V @ torch.diag(lam) @ V.T

    def loss_fn(w):
        return 0.5 * (w.view(-1) @ (H @ w.view(-1))) - b @ w.view(-1)

    def grad_fn(w):
        return (H @ w.view(-1) - b).view_as(w)

    tau = 1.0e9
    gamma = torch.clamp(lam / tau, max=1.0)
    assert float(gamma.max().item()) < 1e-6

    steps, lr = 400, 0.05
    g0 = torch.Generator(device=device)
    g0.manual_seed(3)
    w0 = torch.randn(D, generator=g0, device=device, dtype=dtype)

    def run(mode: str):
        w = w0.clone()
        opt = SubGeoAdam(
            [w],
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            V=None if mode == "vanilla" else V,
            gamma=None if mode == "vanilla" else gamma,
            mode=mode,
        )
        losses = []
        for _ in range(steps):
            w.grad = grad_fn(w)
            opt.step()
            w.grad = None
            losses.append(float(loss_fn(w).item()))
        return losses[-1]

    lv = run("vanilla")
    for mode in ("grad_damp", "sym", "asym"):
        lm = run(mode)
        rel = abs(lm - lv) / (abs(lv) + 1e-8)
        assert rel < 0.05, (mode, lv, lm, rel)


def test_reverse_differs_from_asym_under_moderate_damping():
    """中等阻尼下，reverse 与 asym 的动量投影能量应有可测差异。"""
    device, dtype = torch.device("cpu"), torch.float64
    D, k = 40, 4
    gen = torch.Generator(device=device)
    gen.manual_seed(4)
    A = torch.randn(D, D, generator=gen, device=device, dtype=dtype)
    V, _ = torch.linalg.qr(A[:, :k], mode="reduced")
    lam = torch.tensor([80.0, 20.0, 4.0, 1.0], device=device, dtype=dtype)
    tau = 25.0
    gamma = torch.clamp(lam / tau, max=1.0)

    c = torch.randn(k, generator=gen, device=device, dtype=dtype)
    b = V @ c
    H = V @ torch.diag(lam) @ V.T

    def grad_fn(w):
        return (H @ w.view(-1) - b).view_as(w)

    g0 = torch.Generator(device=device)
    g0.manual_seed(5)
    w0 = torch.randn(D, generator=g0, device=device, dtype=dtype)

    def run_energy(mode: str, steps: int = 120):
        w = w0.clone()
        opt = SubGeoAdam(
            [w],
            lr=0.02,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            V=V,
            gamma=gamma,
            mode=mode,
        )
        last_e = 0.0
        for _ in range(steps):
            w.grad = grad_fn(w)
            opt.step()
            w.grad = None
            last_e = momentum_project_energy(opt.state[w]["m"], V)
        return last_e

    ea = run_energy("asym")
    er = run_energy("reverse")
    assert abs(ea - er) > 1e-6, (ea, er)


def main():
    test_apply_pd_matches_bruteforce()
    test_first_step_asym_differs_from_sym_when_gamma_positive()
    test_weak_damping_tracks_vanilla()
    test_reverse_differs_from_asym_under_moderate_damping()

    print("Phase0 smoke: all gates OK (P_d, first-step asym!=sym, weak-damping~vanilla, reverse!=asym energy).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as e:
        print("FAIL:", e)
        raise SystemExit(1)
