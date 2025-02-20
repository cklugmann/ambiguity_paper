import numpy as np

from sklearn.utils import Bunch


__all__ = ["bin_values"]


def bin_values(amb: np.ndarray, bins: int = 256):
    bins = np.linspace(0, 1, bins+1)
    left = bins[:-1]
    right = bins[1:]
    bin_width, *_ = right - left
    mid = (left + right) / 2
    counts = np.logical_and(
        left <= np.expand_dims(amb, axis=-1),
        np.expand_dims(amb, axis=-1) < right
    ).sum(axis=-2)
    amax = np.argmax(counts, axis=-1)
    mode = mid[amax]
    indices = np.indices(amax.shape)
    idx = (*indices, amax)
    max_count = counts[idx]
    return Bunch(
        bin_centers=mid,
        bin_width=bin_width,
        counts=counts,
        mode=mode,
        max_count=max_count
    )
