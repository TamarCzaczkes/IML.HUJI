from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    x = np.linspace(-1.2, 2, n_samples)
    epsilon = np.random.normal(0, noise, size=n_samples)
    y = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    noise_y = y + epsilon

    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(x), pd.Series(noise_y), train_proportion=(2 / 3))

    fig = go.Figure([go.Scatter(x=train_X[0], y=train_y, mode="markers", name="Training Data"),
                     go.Scatter(x=test_X[0], y=test_y, mode="markers", name="Testing Data"),
                     go.Scatter(x=x, y=y, mode="lines", name="True (noiseless) Data")])
    fig.update_layout(title="Noiseless Data (True Model) and Noised Data (Split to Train and Test)",
                      xaxis_title="x", yaxis_title="y")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    k = 11
    degrees = np.arange(k)

    train_score = np.zeros(k)
    validation_score = np.zeros(k)

    for i in range(k):
        model = PolynomialFitting(i)
        train_score[i], validation_score[i] = cross_validate(model, train_X[0].to_numpy(), train_y.to_numpy(),
                                                             scoring=mean_square_error, cv=5)

    fig = go.Figure([go.Scatter(x=degrees, y=train_score, mode="lines", name="Train Error"),
                     go.Scatter(x=degrees, y=validation_score, mode="lines", name="Validation Error")])
    fig.update_layout(title="Cross-Validation for Polynomial Fitting",
                      xaxis_title="Degree", yaxis_title="Error (MSE)")
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error

    k_min = degrees[np.argmin(validation_score)]  # the k with the lowest validation error
    model = PolynomialFitting(k_min)
    model.fit(train_X[0], train_y.to_numpy())
    y_hat = model.predict(test_X[0].to_numpy())
    print(f"Test error for degree {k_min}: {mean_square_error(test_y.to_numpy(), y_hat)}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """

    # Question 6 - Load diabetes dataset and split into training and testing portions

    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    train_X = diabetes_X[:n_samples]
    test_X = diabetes_X[n_samples:]
    train_y = diabetes_y[:n_samples]
    test_y = diabetes_y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    lambda_vals = np.linspace(0.001, 2, n_evaluations)
    ridge_train_score, ridge_validation_score = np.zeros(n_evaluations), np.zeros(n_evaluations)
    lasso_train_score, lasso_validation_score = np.zeros(n_evaluations), np.zeros(n_evaluations)

    for i, val in enumerate(lambda_vals):
        ridge_model = RidgeRegression(lam=val)
        ridge_train_score[i], ridge_validation_score[i] = cross_validate(ridge_model, train_X, train_y,
                                                                         scoring=mean_square_error)

        lasso_model = Lasso(alpha=val)
        lasso_train_score[i], lasso_validation_score[i] = cross_validate(lasso_model, train_X, train_y,
                                                                         scoring=mean_square_error)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Train and Validation Error for Ridge Regression (using CV)",
                                                        "Train and Validation Error for Lasso Regression (using CV)"))
    fig.add_trace(go.Scatter(x=lambda_vals, y=ridge_train_score, name="Ridge Train Error"), 1, 1)
    fig.add_trace(go.Scatter(x=lambda_vals, y=ridge_validation_score, name="Ridge Validation Error"), 1, 1)
    fig.add_trace(go.Scatter(x=lambda_vals, y=lasso_train_score, name="Lasso Train Error"), 1, 2)
    fig.add_trace(go.Scatter(x=lambda_vals, y=lasso_validation_score, name="Lasso Validation Error"), 1, 2)
    fig.update_xaxes(title_text="Lambda", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=1, col=1)
    fig.update_xaxes(title_text="Lambda", row=1, col=2)
    fig.update_yaxes(title_text="Error", row=1, col=2)
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    k_best_ridge = lambda_vals[np.argmin(ridge_validation_score)]
    print(f"Best lambda for Ridge regression: {k_best_ridge}")
    k_best_lasso = lambda_vals[np.argmin(lasso_validation_score)]
    print(f"Best lambda for Lasso regression: {k_best_lasso}")
    models = [("Ridge", RidgeRegression(k_best_ridge)), ("Lasso", Lasso(k_best_lasso)), ("Linear", LinearRegression())]
    for name, estimator in models:
        estimator.fit(train_X, train_y)
        print(f"MSE of {name} Regression: {mean_square_error(test_y, estimator.predict(test_X))}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()  # n_samples=100, noise=5 as default
    select_polynomial_degree(noise=0)  # for q4
    select_polynomial_degree(n_samples=1500, noise=10)  # for q5

    select_regularization_parameter()  # for q7 + q8
