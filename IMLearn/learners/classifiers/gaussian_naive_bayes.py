from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # the number of distinct classes, and n_k of each class:
        self.classes_, n_ks = np.unique(y, return_counts=True)
        class_num = len(self.classes_)
        sample_num, feature_num = X.shape

        # calculate pi:
        self.pi_ = n_ks / sample_num

        # calculate mean:
        self.mu_ = np.zeros((class_num, feature_num))
        for i in range(class_num):
            self.mu_[i] = np.mean(X[y == self.classes_[i]], axis=0)
            # self.vars_[i] = np.var(X[y == self.classes_[i]], axis=0, ddof=1)

        # calculate variance:
        self.vars_ = np.zeros((class_num, feature_num))
        for i in range(class_num):
            self.vars_[i] = np.var((X - self.mu_[y])[y == self.classes_[i]], axis=0, ddof=1)

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

        return np.argmax(self.likelihood(X), axis=1)

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
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        class_num = len(self.classes_)
        sample_num, feature_num = X.shape
        log_likelihoods = np.zeros((sample_num, class_num))

        for i in range(class_num):
            log_likelihoods[:, i] = -0.5 * np.sum(
                np.log(self.vars_[i]) + (X - self.mu_[i]) ** 2 / self.vars_[i], axis=1)
        return log_likelihoods + np.log(self.pi_)

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
        from ...metrics import misclassification_error

        return misclassification_error(self.predict(X), y)
