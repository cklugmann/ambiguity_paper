from typing import Callable

import numpy as np
from sklearn.utils import Bunch


__all__ = [
    "multinomial_mvs",
    "repeated_multinomial_mvs",
    "dirichlet_mvs",
    "repeated_dirichlet_draws",
    "sample_from"
]


def multinomial_mvs(n: int, p: np.ndarray) -> np.ndarray:
    """
    Sample from the multinomial distribution with multiple p vectors.

    :param n: A scalar values indicating the number of trials.
    :param p: An array of shape (N, C) containing the probability vectors.
    :return: An array of the same shape as p.
    """
    count = np.full(p.shape[:-1], n)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out


def repeated_multinomial_mvs(R: int, p: np.ndarray, B: int = 2048) -> np.ndarray:
    """
    Similar to the above, but here we assume we know a batch of probability vectors p of the form (N, C) and
    we want to compute repeated (B times) realizations of a corresponding multonimial random variable to R
    repetitions for each of the N distributions.
    :param R: The number of responses we want to sample per distribution.
    :param p: An array of shape (N, C) containing the probability vectors.
    :param B: The number of samples to be drawn per probability distribution. More samples determine the
        quality with which the target probability is approximated.
    :return: An array of the shape (N, B, C) containing the repeated realizations of the sampled response
        frequencies.
    """
    num_objects, num_categories = p.shape
    p = np.repeat(p, axis=0, repeats=B)
    samples = multinomial_mvs(R, p).reshape(num_objects, B, num_categories)
    return samples


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
