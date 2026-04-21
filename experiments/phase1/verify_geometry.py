# -*- coding: utf-8 -*-
"""
验证 geometry 模块：Hvp 与解析 Hessian 一致；投影矩阵 V^T H V 一致；G 子空间秩。

运行:
  conda run -n base --cwd d:\\cursor_try\\Evidence python experiments\\phase1\\verify_geometry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from subgeo.geometry import (  # noqa: E402
    build_gamma,
    collect_grad_flat_matrix,
    projected_hessian,
    subspace_from_G,
    hvp_flat,
)


def test_hvp_matches_quadratic_hessian():
    torch.manual_seed(11)
    n, D = 80, 12
    X = torch.randn(n, D, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    w = nn.Parameter(torch.randn(D, dtype=torch.float64) * 0.05)

    def full_loss():
        pred = X @ w
        return 0.5 * ((pred - y) ** 2).mean()

    with torch.no_grad():
        H = (X.T @ X) / n

    v = torch.randn(D, dtype=torch.float64)
    v = v / torch.linalg.norm(v)
    loss = full_loss()
    hv_num = hvp_flat(loss, [w], v, [w.shape])
    hv_ana = H @ v
    err = float(torch.linalg.norm(hv_num - hv_ana).item())
    assert err < 1e-5, err


def test_projected_hessian_matches_analytic():
    torch.manual_seed(12)
    n, D = 100, 16
    X = torch.randn(n, D, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    w = nn.Parameter(torch.randn(D, dtype=torch.float64) * 0.02)

    def loss_fn():
        pred = X @ w
        return 0.5 * ((pred - y) ** 2).mean()

    with torch.no_grad():
        H = (X.T @ X) / n

    r = 5
    G = torch.randn(D, 40, dtype=torch.float64)
    V = subspace_from_G(G, r=r)
    r_eff = V.shape[1]
    Hp_num = projected_hessian(loss_fn, [w], V, [w.shape])
    Hp_ana = V.T @ H @ V
    # 数值对称化后比较特征值集合
    w_sym = 0.5 * (Hp_num + Hp_num.T)
    w_ana = 0.5 * (Hp_ana + Hp_ana.T)
    ev_n, _ = torch.linalg.eigh(w_sym)
    ev_a, _ = torch.linalg.eigh(w_ana)
    err = float(torch.linalg.norm(torch.sort(ev_n)[0] - torch.sort(ev_a)[0]).item())
    assert err < 1e-4 * max(1.0, float(torch.linalg.norm(ev_a).item())), (err, ev_n, ev_a)


def test_collect_grad_and_subspace_rank():
    torch.manual_seed(13)
    n, D, bs = 200, 14, 10
    X = torch.randn(n, D, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    w = nn.Parameter(torch.randn(D, dtype=torch.float64) * 0.02)
    B = 12

    batch = 0

    def loss_fn():
        nonlocal batch
        sl = slice((batch % B) * bs, (batch % B) * bs + bs)
        batch += 1
        pred = X[sl] @ w
        return 0.5 * ((pred - y[sl]) ** 2).mean()

    G = collect_grad_flat_matrix(loss_fn, [w], n_batches=B)
    assert G.shape == (D, B)
    V = subspace_from_G(G, r=min(8, B))
    assert V.shape[0] == D and V.shape[1] >= 1
    # 列正交
    ortho = float(torch.linalg.norm(V.T @ V - torch.eye(V.shape[1], dtype=V.dtype, device=V.device)).item())
    assert ortho < 1e-5, ortho


def test_build_gamma():
    lam = torch.tensor([-1.0, 0.0, 1e-15, 2.0, 10.0])
    g = build_gamma(lam, tau=4.0, lam_eps=1e-12)
    assert g[0].item() == 0 and g[1].item() == 0
    assert abs(g[3].item() - 0.5) < 1e-6
    assert abs(g[4].item() - 1.0) < 1e-6


def main():
    test_hvp_matches_quadratic_hessian()
    test_projected_hessian_matches_analytic()
    test_collect_grad_and_subspace_rank()
    test_build_gamma()
    print("verify_geometry: all checks OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as e:
        print("FAIL:", e)
        raise SystemExit(1)
