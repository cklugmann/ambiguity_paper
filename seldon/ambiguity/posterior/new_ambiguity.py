import numpy as np

from seldon.inference import Posterior
from seldon.ambiguity.new_ambiguity import new_ambiguity, modified_new_ambiguity


__all__ = [
    "expected_amb",
    "var_amb",
    "NewAmbiguityPosterior",
    "ModifiedNewAmbiguityPosterior"
]


def expected_amb(alphas: np.ndarray, modified: bool = False):
    alphas_reduced = alphas[..., :-1]
    alphas_cs = np.expand_dims(alphas[..., -1], axis=-1)
    a_0 = alphas.sum(axis=-1, keepdims=True)
    a_0_r = a_0 - alphas_cs
    e = 1 - 1 / (a_0 * (a_0_r + 1)) * (
        np.sum(alphas_reduced * (alphas_reduced + 1), axis=-1, keepdims=True)
    )
    if modified:
        *_, num_classes = alphas_reduced.shape
        e = 1 / (num_classes - 1) * (
            num_classes * e - alphas_cs / a_0
        )
    return np.squeeze(e, axis=-1)


def var_amb(alphas: np.ndarray, modified: bool = False):
    alphas_reduced = alphas[..., :-1]
    alphas_cs = np.expand_dims(alphas[..., -1], axis=-1)
    a_0 = alphas.sum(axis=-1, keepdims=True)
    b_0 = a_0 - alphas_cs
    denom = a_0 * (a_0 + 1) * (b_0 + 2) * (b_0 + 3)
    nom = (a_0 * (b_0 + 1)) ** 2
    rest = 1 / denom * np.sum(
        alphas_reduced * (alphas_reduced + 1) * (
            (alphas_reduced + 2) * (alphas_reduced + 3)
            - alphas_reduced * (alphas_reduced + 1)
        ),
        axis=-1,
        keepdims=True
    )
    exp_amb = expected_amb(alphas=alphas, modified=False)
    exp_amb = np.expand_dims(exp_amb, axis=-1)
    var = rest + (nom / denom - 1) * (1 - exp_amb) ** 2
    if modified:
        *_, num_classes = alphas_reduced.shape
        var_qcs = alphas_cs * b_0 / (
            (a_0 + 1) * a_0 ** 2
        )
        cov = alphas_cs / (a_0 * (a_0 + 1)) * (1 - exp_amb)
        var = 1 / (num_classes - 1) ** 2 * (
            num_classes ** 2 * var
            + var_qcs
            - 2 * num_classes * cov
        )
    return np.squeeze(var, axis=-1)


class NewAmbiguityPosterior(Posterior):

    def measure(self, ps: np.ndarray):
        return new_ambiguity(ps)

    def mean(self, *args, **kwargs) -> np.ndarray:
        return expected_amb(alphas=self.alphas, modified=False)

    def var(self, *args, **kwargs) -> np.ndarray:
        return var_amb(alphas=self.alphas, modified=False)


class ModifiedNewAmbiguityPosterior(Posterior):

    def measure(self, ps: np.ndarray):
        return modified_new_ambiguity(ps)

    def mean(self, *args, **kwargs) -> np.ndarray:
        return expected_amb(alphas=self.alphas, modified=True)

    def var(self, *args, **kwargs) -> np.ndarray:
        return var_amb(alphas=self.alphas, modified=True)

