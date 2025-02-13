from typing import Callable

import numpy as np
from sklearn.utils import Bunch


__all__ = [
    "dirichlet_mvs",
    "repeated_dirichlet_draws",
    "sample_from"
]


def dirichlet_mvs(alphas: np.ndarray) -> np.ndarray:
    """
    Sample from the Dirichlet distribution with multiple alpha vectors.

    :param alphas: An ndarray of shape (K, C).
    :return: An ndarray of shape (K, C) where each entry represents a probability distriution drawn from the
        corresponding Dirichlet distribution.
    """
    r = np.random.standard_gamma(alphas)
    return r / r.sum(-1, keepdims=True)


def repeated_dirichlet_draws(alphas: np.ndarray, repeats: int) -> np.ndarray:
    is_vector = False
    if len(alphas.shape) == 1:
        alphas = alphas.reshape(1, -1)
        is_vector = True
    num_objects, *_ = alphas.shape
    ps = dirichlet_mvs(
        alphas=np.repeat(
            alphas,
            repeats=repeats,
            axis=0
        )
    ).reshape(num_objects, repeats, -1)
    if is_vector:
        ps = np.squeeze(ps, axis=0)
    return ps


def sample_from(f: Callable[[np.ndarray], np.ndarray], alphas: np.ndarray, **kwargs):
    ps = repeated_dirichlet_draws(alphas=alphas, **kwargs)
    return Bunch(vals=f(ps), ps=ps)
