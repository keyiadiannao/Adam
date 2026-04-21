# -*- coding: utf-8 -*-
"""在导入 huggingface_hub / transformers 之前设置 HF_ENDPOINT（镜像）。"""
from __future__ import annotations

import os


def configure_hf_hub_endpoint() -> str | None:
    """
    若已设置 HF_ENDPOINT 或 HUGGINGFACE_HUB_ENDPOINT，则不覆盖。
    默认（EVIDENCE_HF_MIRROR 未关闭时）使用 EVIDENCE_HF_ENDPOINT 或 https://hf-mirror.com。
    关闭：EVIDENCE_HF_MIRROR=0
    """
    if os.environ.get("HF_ENDPOINT") or os.environ.get("HUGGINGFACE_HUB_ENDPOINT"):
        return None
    flag = os.environ.get("EVIDENCE_HF_MIRROR", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return None
    base = os.environ.get("EVIDENCE_HF_ENDPOINT", "https://hf-mirror.com").strip().rstrip("/")
    if not base.lower().startswith("http"):
        base = "https://" + base.lstrip("/")
    os.environ["HF_ENDPOINT"] = base
    return base
