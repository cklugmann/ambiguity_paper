import numpy as np

from seldon.estimator import BayesEstimator
from seldon.ambiguity.estimator import *


def main():

    ns = np.array([
        [10, 1, 1],
        [5, 0, 0]
    ])

    amb_old_estimators = dict(
        plug_in=AmbiguityPlugInEstimator().fit(ns=ns),
        bayes_mean=AmbiguityBayesianEstimator(beta=1.0, estimator_type=BayesEstimator.EstimatorType.MEAN).fit(ns=ns),
        bayes_mode=AmbiguityBayesianEstimator(beta=1.0, estimator_type=BayesEstimator.EstimatorType.MODE).fit(ns=ns)
    )

    for what, estimator in amb_old_estimators.items():
        print(what)
        print(estimator.value(repeats=256))


if __name__ == "__main__":
    main()