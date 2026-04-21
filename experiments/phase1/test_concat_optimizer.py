# -*- coding: utf-8 -*-
"""ConcatSubGeoAdam 与单张量 SubGeoAdam 在拼接空间上等价性自检。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from smoke.subgeo_optimizer import SubGeoAdam  # noqa: E402
from subgeo.optimizer import ConcatSubGeoAdam, make_weak_pd_operators  # noqa: E402


def test_concat_matches_single_tensor():
    torch.manual_seed(42)
    dtype, device = torch.float64, torch.device("cpu")
    a = nn.Parameter(torch.tensor([1.0, -0.5], device=device, dtype=dtype))
    b = nn.Parameter(torch.tensor([0.25], device=device, dtype=dtype))
    w_single = nn.Parameter(torch.cat([a.detach().reshape(-1), b.detach().reshape(-1)]).clone())
    w_single.requires_grad_(True)

    D = 3
    k = 2
    V, gamma = make_weak_pd_operators(D, k, device, dtype, tau=1e8)

    opt_cat = ConcatSubGeoAdam([a, b], lr=0.01, weight_decay=0.0, V=V, gamma=gamma, mode="asym")
    opt_one = SubGeoAdam([w_single], lr=0.01, weight_decay=0.0, V=V, gamma=gamma, mode="asym")

    g_vec = torch.randn(D, device=device, dtype=dtype)
    a.grad = g_vec[:2].clone()
    b.grad = g_vec[2:].clone()
    w_single.grad = g_vec.clone()

    opt_cat.step()
    opt_one.step()

    w_after = torch.cat([a.data.reshape(-1), b.data.reshape(-1)])
    err = float(torch.linalg.norm(w_after - w_single.data.reshape(-1)).item())
    assert err < 1e-10, err


if __name__ == "__main__":
    test_concat_matches_single_tensor()
    print("test_concat_optimizer: OK")
