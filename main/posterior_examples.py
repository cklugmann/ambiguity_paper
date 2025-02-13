import numpy as np

from seldon.inference import BayesianInference
from seldon.ambiguity.posterior import *


def main():

    ns = np.array([
        [10, 1, 1],
        [5, 0, 0]
    ])

    inference_object = (
        BayesianInference
        .specify(NewAmbiguityPosterior)
    )(beta=1.5).fit(ns)
    posterior = inference_object.posterior
    sample = posterior.sample(repeats=4)
    print("some sample\n", sample)
    print("Mean (MC):", posterior.apply("mean", repeats=2048))
    print("Mean (analytical):", posterior.mean())
    # Note: for the mode, we are unable to derive an analytical formula
    print("Mode (MC):", posterior.mode(repeats=2048))


if __name__ == "__main__":
    main()