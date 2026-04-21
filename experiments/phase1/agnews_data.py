# -*- coding: utf-8 -*-
"""兼容层：请优先使用 ``real_two_task_data.build_two_task_pools``。"""
from __future__ import annotations

from typing import List, Tuple

from real_two_task_data import batch_indices, build_two_task_pools, tokenize_batch

Row = Tuple[str, int]


def build_agnews_two_task_pools(
    max_per_class: int,
    seed: int,
) -> Tuple[List[Row], List[Row]]:
    t0, t1, _, _ = build_two_task_pools("ag_news", max_per_class, 0, seed)
    return t0, t1


__all__ = ["batch_indices", "build_agnews_two_task_pools", "tokenize_batch", "Row"]
