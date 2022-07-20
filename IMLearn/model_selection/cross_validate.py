# from __future__ import annotations
# from copy import deepcopy
# from typing import Tuple, Callable
# import numpy as np
# from IMLearn import BaseEstimator
#
#
# def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
#                    scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
#     """
#     Evaluate metric by cross-validation for given estimator
#
#     Parameters
#     ----------
#     estimator: BaseEstimator
#         Initialized estimator to use for fitting the data
#
#     X: ndarray of shape (n_samples, n_features)
#        Input data to fit
#
#     y: ndarray of shape (n_samples, )
#        Responses of input data to fit to
#
#     scoring: Callable[[np.ndarray, np.ndarray, ...], float]
#         Callable to use for evaluating the performance of the cross-validated model.
#         When called, the scoring function receives the true- and predicted values for each sample
#         and potentially additional arguments. The function returns the score for given input.
#
#     cv: int
#         Specify the number of folds.
#
#     Returns
#     -------
#     train_score: float
#         Average train score over folds
#
#     validation_score: float
#         Average validation score over folds
#     """
#     ids = np.arange(X.shape[0])
#
#     # Randomly split samples into `cv` folds
#     folds = np.array_split(ids, cv)
#
#     train_score, validation_score = .0, .0
#     for fold_ids in folds:
#         train_msk = ~np.isin(ids, fold_ids)
#         fit = deepcopy(estimator).fit(X[train_msk], y[train_msk])
#
#         train_score += scoring(y[train_msk], fit.predict(X[train_msk]))
#         validation_score += scoring(y[fold_ids], fit.predict(X[fold_ids]))
#
#     return train_score / cv, validation_score / cv
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

    val_score: float
        Average validation score over folds
    """

    train_score, validation_score = [], []
    combined_data = np.vstack((X.T, y)).T
    folds = [i for i in np.array_split(combined_data, cv)]
    for i in range(len(folds)):
        training_folds = folds[:]
        validate_x = folds[i][:, :-1]
        validate_y = folds[i][:, -1]
        del training_folds[i]
        training = np.vstack([training_folds[j] for j in
                              range(len(training_folds))])
        train_x = training[:, :-1]
        train_y = training[:, -1]
        if train_x.shape[1] == 1:
            train_x = train_x.flatten()
            validate_x = validate_x.flatten()
        estimator.fit(train_x, train_y)
        train_score.append(
            scoring(train_y, estimator.predict(train_x)))
        validation_score.append(scoring(validate_y,
                                        estimator.predict(validate_x)))
    return float(np.mean(train_score)), float(np.mean(validation_score))