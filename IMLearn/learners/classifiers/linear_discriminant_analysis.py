from typing import NoReturn
from IMLearn.metrics.loss_functions import misclassification_error
import numpy as np

from ...base import BaseEstimator
import scipy.stats


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m = len(X)
        self.mu_ = None
        self.pi_ = np.array([])
        self.classes_ = np.unique(y)
        mu_ind_dic = {}
        classes_dic = {}
        self.cov_ = None
        for i in range(m):
            if isinstance(classes_dic.get(y[i]), np.ndarray):
                classes_dic[y[i]] = np.vstack((classes_dic[y[i]], np.array(
                    X[i])))
            else:
                classes_dic[y[i]] = np.array(X[i])
        for cls in self.classes_:
            mu_ind_dic[cls] = len(mu_ind_dic.keys())
            self.pi_ = np.append(self.pi_, len(classes_dic[cls]) / m)
            if isinstance(self.mu_, np.ndarray):
                self.mu_ = np.vstack((self.mu_, classes_dic[cls].mean(
                    axis=0)))
            else:
                self.mu_ = np.array(classes_dic[cls].mean(axis=0))
        for i in range(m):
            if isinstance(self.cov_, np.ndarray):
                self.cov_ += np.outer(X[i] - self.mu_[mu_ind_dic[y[i]]],
                                      X[i] - self.mu_[mu_ind_dic[y[i]]])
            else:
                self.cov_ = np.outer(X[i] - self.mu_[mu_ind_dic[y[i]]],
                                     X[i] - self.mu_[mu_ind_dic[y[i]]])
        self.cov_ = self.cov_ / (m - len(self.classes_))  # unbiased
        self._cov_inv = np.linalg.inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        likely = self.likelihood(X)
        y_hat = np.array([])
        for i in range(len(X)):
            label = np.argmax(likely[i])
            y_hat = np.append(y_hat, label)
        return y_hat

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        likelihood = None
        for i in range(len(X)):
            samp_likely = np.array([])
            for j in range(len(self.classes_)):
                like = scipy.stats.multivariate_normal.pdf(X[i]
                                , self.mu_[j],self.cov_) * self.pi_[j]
                samp_likely = np.append(samp_likely, like)

            if isinstance(likelihood, np.ndarray):
                likelihood = np.vstack((likelihood, samp_likely))
            else:
                likelihood = np.array(samp_likely)
        return likelihood

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
        y_hat = self._predict(X)
        return misclassification_error(y, y_hat)
