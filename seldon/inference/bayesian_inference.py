import abc
from typing import Type

import numpy as np
from sklearn.utils import Bunch

import seldon.utils.sampling as sampling_utils
from seldon.utils.binning import bin_values


__all__ = [
    "BayesianInference",
    "Posterior"
]


class Posterior(abc.ABC):

    inference_object: "BayesianInference"

    def __init__(self, inference_object: "BayesianInference"):
        self.inference_object = inference_object

    @property
    def alphas(self) -> np.ndarray:
        return (
            self.inference_object.ns
            + self.inference_object.beta * np.ones_like(self.inference_object.ns, dtype=float)
        )

    @abc.abstractmethod
    def measure(self, ps: np.ndarray):
        pass

    def sample(self, repeats: int, keep_p: bool = False, **kwargs) -> np.ndarray | Bunch:
        res = sampling_utils.sample_from(
            f=self.measure,
            alphas=self.alphas,
            repeats=repeats,
            **kwargs
        )
        if keep_p:
            return res
        return res.vals

    def apply(self, what: str, repeats: int, *args, **kwargs):
        # Applies function to sampled values
        # `what` can be any numpy function like mean or std
        values = self.sample(repeats=repeats, keep_p=False)
        fn = getattr(np, what)
        return fn(values, *args, axis=-1, **kwargs)

    # Idea: make these functions explicit and allow overriding them

    def mean(self, *args, **kwargs) -> np.ndarray:
        return self.apply("mean", *args, **kwargs)

    def var(self, *args, **kwargs) -> np.ndarray:
        return self.apply("var", *args, **kwargs)

    def std(self, *args, **kwargs) -> np.ndarray:
        return np.sqrt(self.var(*args, **kwargs))

    def mode(self, repeats: int, *args, **kwargs) -> np.ndarray:
        values = self.sample(repeats=repeats, keep_p=False)
        # TODO: just dummy implementation for now
        return bin_values(values, *args, **kwargs).mode



class BayesianInference:

    class RequiresFitException(Exception):
        pass

    beta: float
    ns: np.ndarray | None

    posterior_cls: Type[Posterior] | None = None

    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.ns = None

    def fit(self, ns: np.ndarray) -> "BayesianInference":
        self.ns = ns
        return self

    @classmethod
    def specify(cls, posterior_cls: Type[Posterior]):
        cls.posterior_cls = posterior_cls
        return cls

    @property
    def posterior(self):
        if self.ns is None:
            raise self.RequiresFitException("Call the `fit()` method first.")
        return self.posterior_cls(self)
