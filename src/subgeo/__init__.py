# -*- coding: utf-8 -*-
"""SubGeo-Adam 工具包：几何提取、Hvp 等。"""

from .joint_geometry import extract_joint_vk_gamma
from .geometry import (
    apply_pd,
    build_gamma,
    collect_grad_flat_matrix,
    flatten_params,
    hvp_flat,
    projected_hessian,
    split_flat_to_param_grads,
    split_flat_to_tensors,
    subspace_from_G,
)
from .optimizer import (
    ConcatSubGeoAdam,
    concat_subgeo_m_flat,
    lora_trainable_parameters,
    make_weak_pd_operators,
    momentum_energy_in_subspace,
)

__all__ = [
    "apply_pd",
    "flatten_params",
    "split_flat_to_tensors",
    "split_flat_to_param_grads",
    "collect_grad_flat_matrix",
    "subspace_from_G",
    "hvp_flat",
    "projected_hessian",
    "build_gamma",
    "extract_joint_vk_gamma",
    "ConcatSubGeoAdam",
    "concat_subgeo_m_flat",
    "make_weak_pd_operators",
    "momentum_energy_in_subspace",
    "lora_trainable_parameters",
]
