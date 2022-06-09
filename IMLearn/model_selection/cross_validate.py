from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    train_score, validation_score = 0, 0

    X_folds = np.array_split(X, cv)
    y_folds = np.array_split(y, cv)

    estimator_copy = deepcopy(estimator)

    for i in range(cv):
        X_train = np.concatenate(X_folds[:i] + X_folds[i + 1:])
        y_train = np.concatenate(y_folds[:i] + y_folds[i + 1:])
        X_valid = X_folds[i]
        y_valid = y_folds[i]

        estimator_copy.fit(X_train, y_train)

        train_score += scoring(y_train, estimator_copy.predict(X_train))
        validation_score += scoring(y_valid, estimator_copy.predict(X_valid))

    return (train_score / cv), (validation_score / cv)

