import abc
from enum import Enum
from typing import Type

import numpy as np

import seldon.utils.sampling as sampling_utils
from seldon.inference import BayesianInference, Posterior


__all__ = [
    "Estimator",
    "PlugInEstimator",
    "BayesEstimator"
]


class Estimator(abc.ABC):

    class RequiresFitException(Exception):
        pass

    ns: np.ndarray | None

    def fit(self, ns: np.ndarray) -> "Estimator":
        self.ns = ns
        return self

    @abc.abstractmethod
    def value(self, *args, **kwargs) -> np.ndarray:
        pass

    def expectation(self, num_trials: int, q0: np.ndarray, *args, num_samples: int = 10_000, **kwargs) -> np.ndarray:
        """
        Can be used to determine (approximate) the estimator's error value under a Multinomial(num_trials, q0) sampling
            distribution.
        Override this method if an analytical expression for the expected value of the specific estimator exists.
        :param num_trials: The number of multinomial trials
        :param q0: The array of probability vectors
        :param num_samples: The number of repeated samplings per probability vector.
        :return: Numpy array with (approximate) mean values for the estimator.
        """
        ns = sampling_utils.repeated_multinomial_mvs(R=num_trials, p=q0, B=num_samples)
        _ = self.fit(ns=ns)
        values = self.value(*args, **kwargs)
        return values.mean(axis=-1)


class PlugInEstimator(Estimator):

    def value(self, *args, **kwargs):
        if self.ns is None:
            raise self.RequiresFitException("Call the `fit()` method first.")
        qs = self.ns / self.ns.sum(axis=-1, keepdims=True)
        return self.measure(qs)

    @abc.abstractmethod
    def measure(self, ps: np.ndarray) -> np.ndarray:
        pass


class BayesEstimator(Estimator):

    class EstimatorType(Enum):
        MEAN = 0
        MODE = 1

    posterior_cls: Type[Posterior] | None = None

    inference_object: BayesianInference | None
    estimator_type: EstimatorType | None

    def __init__(self, beta: float, estimator_type: EstimatorType):
        self.estimator_type = estimator_type
        self.inference_object = (
            BayesianInference
            .specify(posterior_cls=self.posterior_cls)
        )(beta=beta)

    def fit(self, ns: np.ndarray) -> "BayesEstimator":
        super().fit(ns=ns)
        self.inference_object.fit(ns=ns)
        return self

    @property
    def _posterior(self) -> Posterior:
        return self.inference_object.posterior

    def value(self, *args, **kwargs) -> np.ndarray:
        if self.ns is None:
            raise self.RequiresFitException("Call the `fit()` method first.")
        if self.estimator_type == self.EstimatorType.MODE:
            return self._posterior.mode(*args, **kwargs)
        return self._posterior.mean(*args, **kwargs)

    @classmethod
    def specify(cls, posterior_cls: Type[Posterior]):
        cls.posterior_cls = posterior_cls
        return cls
