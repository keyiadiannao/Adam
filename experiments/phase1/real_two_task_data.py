# -*- coding: utf-8 -*-
"""
双任务二分类数据池：AGNews 与 DBpedia-14，结构一致（任务0=原始类0/1，任务1=原始类2/3→二分类0/1）。

``train_per_class`` / ``holdout_per_class``：四个原始类别各取 ``train_per_class + holdout_per_class`` 条后，
按桶内 shuffle + 切分，前 ``holdout_per_class`` 为 **eval**（不参与训练），后 ``train_per_class`` 为 **train**。
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch

Row = Tuple[str, int]

_DATASET_CFG: Dict[str, Dict] = {
    "ag_news": {"text_key": "text", "split": "train"},
    "dbpedia_14": {"text_key": "content", "split": "train"},
}


def batch_indices(pool_len: int, batch_size: int, task_step: int) -> List[int]:
    start = (task_step * batch_size) % pool_len
    return [(start + j) % pool_len for j in range(batch_size)]


def tokenize_batch(
    tok: object,
    rows: List[Row],
    indices: List[int],
    device: torch.device,
    max_length: int,
) -> dict:
    texts = [rows[i][0] for i in indices]
    labels = [rows[i][1] for i in indices]
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in enc.items()}
    batch["labels"] = torch.tensor(labels, dtype=torch.long, device=device)
    return batch


def _split_bucket(bucket: List[Row], hold_pc: int, train_pc: int, rng: random.Random) -> Tuple[List[Row], List[Row]]:
    rng.shuffle(bucket)
    if hold_pc <= 0:
        return bucket[:train_pc], []
    ev = bucket[:hold_pc]
    tr = bucket[hold_pc : hold_pc + train_pc]
    return tr, ev


def build_two_task_pools(
    dataset_name: str,
    train_per_class: int,
    holdout_per_class: int,
    seed: int,
) -> Tuple[List[Row], List[Row], List[Row], List[Row]]:
    if dataset_name not in _DATASET_CFG:
        raise ValueError(dataset_name, sorted(_DATASET_CFG))
    from datasets import load_dataset

    cfg = _DATASET_CFG[dataset_name]
    ds = load_dataset(dataset_name, split=cfg["split"])
    tk = cfg["text_key"]
    need = train_per_class + holdout_per_class
    t0_c0: List[Row] = []
    t0_c1: List[Row] = []
    t1_c0: List[Row] = []
    t1_c1: List[Row] = []

    for ex in ds:
        lab = int(ex["label"])
        text = str(ex[tk])
        if lab == 0 and len(t0_c0) < need:
            t0_c0.append((text, 0))
        elif lab == 1 and len(t0_c1) < need:
            t0_c1.append((text, 1))
        elif lab == 2 and len(t1_c0) < need:
            t1_c0.append((text, 0))
        elif lab == 3 and len(t1_c1) < need:
            t1_c1.append((text, 1))
        if (
            len(t0_c0) >= need
            and len(t0_c1) >= need
            and len(t1_c0) >= need
            and len(t1_c1) >= need
        ):
            break

    rng = random.Random(seed)
    tr0_c0, ev0_c0 = _split_bucket(t0_c0, holdout_per_class, train_per_class, rng)
    tr0_c1, ev0_c1 = _split_bucket(t0_c1, holdout_per_class, train_per_class, random.Random(seed + 1))
    tr1_c0, ev1_c0 = _split_bucket(t1_c0, holdout_per_class, train_per_class, random.Random(seed + 2))
    tr1_c1, ev1_c1 = _split_bucket(t1_c1, holdout_per_class, train_per_class, random.Random(seed + 3))

    train_task0 = tr0_c0 + tr0_c1
    train_task1 = tr1_c0 + tr1_c1
    eval_task0 = ev0_c0 + ev0_c1
    eval_task1 = ev1_c0 + ev1_c1
    random.Random(seed + 10).shuffle(train_task0)
    random.Random(seed + 11).shuffle(train_task1)
    random.Random(seed + 12).shuffle(eval_task0)
    random.Random(seed + 13).shuffle(eval_task1)
    return train_task0, train_task1, eval_task0, eval_task1
