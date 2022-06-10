from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
from IMLearn.metrics import misclassification_error
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        best_loss = float('inf')
        for feature in range(X.shape[1]):
            for sign in {-1, 1}:
                feature_thresh, feature_loss = self._find_threshold(
                    X[:, feature], y, sign)
                if feature_loss < best_loss:
                    self.threshold_ = feature_thresh
                    self.j_ = feature
                    self.sign_ = sign
                    best_loss = feature_loss


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        col = X[:,self.j_]
        y_pred = np.ones(len(col))
        for i in range(len(col)):
            if col[i] < self.threshold_:
                y_pred[i] *= -self.sign_
            else:
                y_pred[i] *= self.sign_
        return y_pred

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        self.fitted_ = True
        best_thresh = values[0]
        best_loss = float('inf')
        sorted_ind = np.argsort(values, axis=0)
        sorted_val = np.take_along_axis(values, sorted_ind, axis=0)
        sorted_lab = np.take_along_axis(labels, sorted_ind, axis=0)
        for i in range(len(values)):
            cur_thresh = sorted_val[i]
            less = np.ones(i) * -sign
            more = np.ones(len(values) - i) * sign
            y_pred = np.concatenate((less, more))
            cur_loss = self.loss(y_pred, sorted_lab)
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_thresh = cur_thresh
        return best_thresh, best_loss

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        losses = np.where(np.sign(y)!=X,abs(y),0)
        res = np.sum(losses)
        return res

