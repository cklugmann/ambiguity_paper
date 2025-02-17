import numpy as np

from seldon.estimator import PlugInEstimator, BayesEstimator
from seldon.ambiguity.new_ambiguity import new_ambiguity, modified_new_ambiguity
from seldon.ambiguity.posterior import NewAmbiguityPosterior, ModifiedNewAmbiguityPosterior


__all__ = [
    "NewAmbiguityPlugInEstimator",
    "NewAmbiguityBayesianEstimator",
    "ModifiedNewAmbiguityPlugInEstimator",
    "ModifiedNewAmbiguityBayesianEstimator"

]


class NewAmbiguityPlugInEstimator(PlugInEstimator):
    def measure(self, ps: np.ndarray) -> np.ndarray:
        return new_ambiguity(ps)

    def expectation(self, num_trials: int, q0: np.ndarray, *args, **kwargs) -> np.ndarray:
        # For the plug-in estimator of the new ambiguity measure, we calculated the expected value under the sampling
        #   distribution
        R = num_trials
        q_res = q0[..., :-1]
        q_cs = q0[..., -1]
        return 1 - (1 - q_cs ** R) / R - np.sum(q_res ** 2, axis=-1) * (
                1 / (1 - q_cs) - (1 - q_cs ** R) / (R * (1 - q_cs) ** 2)
        )


class NewAmbiguityBayesianEstimator(BayesEstimator):
    posterior_cls = NewAmbiguityPosterior


class ModifiedNewAmbiguityPlugInEstimator(PlugInEstimator):
    def measure(self, ps: np.ndarray) -> np.ndarray:
        return modified_new_ambiguity(ps)


class ModifiedNewAmbiguityBayesianEstimator(BayesEstimator):
    posterior_cls = ModifiedNewAmbiguityPosterior
