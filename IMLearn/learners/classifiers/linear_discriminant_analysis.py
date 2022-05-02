from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


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
        Estimates gaussian for each label class - Different mean vector,
        same covariance matrix with dependent features.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.classes_, n_k_vector = np.unique(y, return_counts=True)
        classes_num = len(self.classes_)
        sample_num, features_num = X.shape

        # calc pi_
        self.pi_ = n_k_vector / sample_num

        # calc mu_
        self.mu_ = np.zeros((classes_num, features_num))
        for i in range(classes_num):
            self.mu_[i] = np.mean(X[y == self.classes_[i]], axis=0)

        # calc the covariance matrix
        self.cov_ = np.zeros((features_num, features_num))
        for i in range(sample_num):
            self.cov_ += np.outer(X[i] - self.mu_[y[i]], X[i] - self.mu_[y[i]])
        self.cov_ /= sample_num - classes_num

        # calculate the inverse of the covariance matrix
        self._cov_inv = inv(self.cov_)

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
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def _calc_gaussian_pdf(self, X: np.ndarray, mu: np.ndarray) -> float:
        calc = (X - mu) @ self._cov_inv @ (X - mu).T
        return np.exp(-.5 * calc) / np.sqrt((2 * np.pi) ** len(X) * det(self.cov_))

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

        classes_num = len(self.classes_)
        likelihoods = np.zeros((X.shape[0], classes_num))

        for i in range(X.shape[0]):
            for j in range(classes_num):
                likelihoods[i, j] = self.pi_[j] * self._calc_gaussian_pdf(X[i], self.mu_[j])
        return likelihoods

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
