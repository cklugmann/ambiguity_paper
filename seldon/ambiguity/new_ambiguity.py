import numpy as np


__all__ = [
    "new_ambiguity",
    "modified_new_ambiguity"
]


def new_ambiguity(p: np.ndarray):
    p_res = p[..., :-1]
    p_cs = p[..., -1]
    return np.where(p_cs  < 1, 1 - 1 / np.maximum(1 - p_cs, 1e-7) * np.sum(p_res ** 2, axis=-1), np.ones_like(p_cs))


def modified_new_ambiguity(p: np.ndarray):
    p_res = p[..., :-1]
    p_cs = p[..., -1]
    *_, num_classes = p_res.shape
    return (
        1 / (num_classes - 1)
        * (num_classes * new_ambiguity(p) - p_cs)
    )
