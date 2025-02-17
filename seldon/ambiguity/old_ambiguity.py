import math

import numpy as np


__all__ = [
    "ambiguity"
]


def exp_cant_solve_scaling(p_cs: np.ndarray, p0: float, eta0: float) -> np.ndarray:
    gamma = -math.log(eta0) / p0
    return np.exp(-gamma * p_cs)


def linear_cant_solve_scaling(p_cs: np.ndarray, *args, **kwargs) -> np.ndarray:
    return 1 - p_cs


def base_disambiguity(p: np.ndarray, eps: float) -> np.ndarray:
    # Re-normalize p's
    p_proper = p[..., :-1]
    *_, num_proper_classes = p_proper.shape
    p_proper_normalized = p_proper / np.maximum(
        p_proper.sum(axis=-1, keepdims=True), 1e-4
    )
    dist_to_uniform = (
        0.5
        * num_proper_classes
        / (num_proper_classes - 1)
        * np.sum(np.abs(p_proper_normalized - 1 / num_proper_classes), axis=-1)
    )
    return (1 - eps) * dist_to_uniform + eps


def disambiguity(
    p: np.ndarray,
    eps: float = 0,
    p0: float = 0.2,
    eta0: float = 0.4,
    use_exp_scaling: bool = False,
):
    cs_scaling = linear_cant_solve_scaling
    if use_exp_scaling:
        cs_scaling = exp_cant_solve_scaling
    p_cs = p[..., -1]
    eta = cs_scaling(p_cs, p0=p0, eta0=eta0)
    base_disamb = base_disambiguity(p, eps=eps)
    return eta * base_disamb


def ambiguity(p: np.ndarray, **kwargs):
    return 1 - disambiguity(p, **kwargs)
