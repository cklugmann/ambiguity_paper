import numpy as np

from seldon.inference import Posterior
from seldon.ambiguity.old_ambiguity import ambiguity


__all__ = ["AmbiguityPosterior"]


class AmbiguityPosterior(Posterior):

    def measure(self, ps: np.ndarray):
        return ambiguity(ps)
