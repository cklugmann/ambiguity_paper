import numpy as np

from seldon.estimator import PlugInEstimator, BayesEstimator
from seldon.ambiguity.old_ambiguity import ambiguity
from seldon.ambiguity.posterior import AmbiguityPosterior


__all__ = [
    "AmbiguityPlugInEstimator",
    "AmbiguityBayesianEstimator"
]


class AmbiguityPlugInEstimator(PlugInEstimator):
    def measure(self, ps: np.ndarray) -> np.ndarray:
        return ambiguity(ps)


class AmbiguityBayesianEstimator(BayesEstimator):
    posterior_cls = AmbiguityPosterior
