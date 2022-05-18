import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the
    linearly separable and inseparable datasets.

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "../datasets/linearly_separable.npy"),
                 ("Linearly Inseparable", "../datasets/linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        callback = lambda fit, x_i, y_i: losses.append(fit.loss(X, y))
        perceptron = Perceptron(callback=callback)
        perceptron.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure(go.Line(x=list(range(1, perceptron.max_iter_ + 1)), y=losses, name="Loss"))
        fig.update_layout(title=f"Perceptron: {n}", xaxis_title="Iteration", yaxis_title="Loss")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["../datasets/gaussian1.npy", "../datasets/gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        gnb_pred = gnb.predict(X)

        lda = LDA()
        lda.fit(X, y)
        lda_pred = lda.predict(X)

        # Plot a figure with two subplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        # make two subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Gaussian Naive Bayes", "LDA"))

        # Add traces for data-points setting symbols and colors
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                 marker_color=gnb_pred, marker_symbol=class_symbols[y], marker_size=9), 1, 1)
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                 marker_color=lda_pred, marker_symbol=class_symbols[y], marker_size=9), 1, 2)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(x=np.array(gnb.mu_[:, 0]), y=np.array(gnb.mu_[:, 1]),
                                 mode="markers", marker_color="black", marker_symbol="x", marker_size=17), 1, 1)
        fig.add_trace(go.Scatter(x=np.array(lda.mu_[:, 0]), y=np.array(lda.mu_[:, 1]),
                                 mode="markers", marker_color="black", marker_symbol="x", marker_size=17, ), 1, 2, )

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(np.unique(y).size):
            fig.add_trace(get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i])), 1, 1)
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), 1, 2)

        # Add titles
        fig.layout.update(title=f"Gaussian Naive Bayes and LDA Estimators, for {f}", title_x=0.5)
        fig.update_xaxes(title_text="x", row=1, col=1)
        fig.update_yaxes(title_text="y", row=1, col=1)
        fig.update_xaxes(title_text="x", row=1, col=2)
        fig.update_yaxes(title_text="y", row=1, col=2)

        # Add accuracy
        fig.layout.annotations[0].update(text=f"Gaussian Naive Bayes, accuracy: {accuracy(gnb.predict(X), y)}")
        fig.layout.annotations[1].update(text=f"LDA, accuracy: {accuracy(lda.predict(X), y)}")

        fig.show()


def quiz():
    # X = np.array([[0, 0], [1, 0], [2, 1], [3, 1], [4, 1], [5, 1], [6, 2], [7, 2]])
    # y = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    X = np.array([[1, 1], [1, 2], [2, 3], [2, 4], [3, 3], [3, 4]])
    y = np.array([0, 0, 1, 1, 1, 1])

    gnb = GaussianNaiveBayes()
    gnb.fit(X, y)
    # print("mu_:", gnb.mu_)
    # print("pi_:", gnb.pi_)
    print("vars_:", gnb.vars_)


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    # compare_gaussian_classifiers()
    # quiz()
