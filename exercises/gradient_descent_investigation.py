import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.utils import split_train_test
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.model_selection import cross_validate
from IMLearn.metrics.loss_functions import misclassification_error
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression

from sklearn.metrics import roc_curve, auc

import plotly.express as px
import plotly.graph_objects as go

MODELS = {"L1": L1, "L2": L2}


def plot_descent_path(module: Type[BaseModule], descent_path: np.ndarray,
                      title: str = "", xrange=(-1.5, 1.5), yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """

    values_lst = []
    weights_lst = []

    def callback(model, weights, val, grad, t, eta, delta):
        values_lst.append(val.copy())
        weights_lst.append(weights.copy())

    return callback, values_lst, weights_lst


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    max_iter = 1000

    for name, model in MODELS.items():

        losses_df = pd.DataFrame(columns=etas)

        for eta in etas:
            callback, values, weights = get_gd_state_recorder_callback()
            GD = GradientDescent(learning_rate=FixedLR(eta), callback=callback, max_iter=max_iter)
            GD.fit(model(weights=init.copy()), None, None)

            losses_df[eta] = values + ([np.nan] * (max_iter - len(values)))

            plot_descent_path(module=model, descent_path=np.array(weights),
                              title=f"{name}: learning rate = {eta}").show()
            print(f"{name} norm, eta = {eta}, minimum loss = {np.min(values)}")

        plot = px.line(losses_df, markers=True)
        plot.update_layout(title=f"Convergence rate for {name}",
                           xaxis_title=f"Iteration", yaxis_title=f"Norm",
                           legend_title="Eta Value:").show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1, gammas: Tuple[float] = (.9, .95, .99, 1)):
    max_iter = 1000
    losses_df = pd.DataFrame(columns=gammas)

    for gamma in gammas:
        callback, values, weights = get_gd_state_recorder_callback()
        GD = GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=callback, max_iter=max_iter)
        GD.fit(L1(weights=init.copy()), None, None)

        losses_df[gamma] = values + ([np.nan] * (max_iter - len(values)))

        fig = plot_descent_path(module=L1, descent_path=np.array(weights),
                                title=f"L1: eta = {eta}, gamma = {gamma}")
        fig.show()
        print(f"L1 norm, gamma = {gamma}, minimum loss = {np.min(values)}")

    val_fig = px.line(losses_df)
    val_fig.update_layout(title=f"Convergence rate for L1",
                          xaxis_title=f"Iteration", yaxis_title=f"Norm",
                          legend_title="Eta Value:").show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data

    lr = 1e-4
    max_iter = 20000

    GD = GradientDescent(learning_rate=FixedLR(lr), max_iter=max_iter)
    LG = LogisticRegression(solver=GD)
    LG.fit(X_train, y_train)

    train_predict_proba = LG.predict_proba(X_train.to_numpy())
    fpr, tpr, thresholds = roc_curve(y_train, train_predict_proba)

    # plot ROC curve
    roc_plot = go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                          line=dict(color="black", dash='dash'), name="Random Class Assignment"),
                               go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="",
                                          showlegend=False,
                                          hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
                         layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                                          xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                                          yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    roc_plot.show()

    opt_alpha = thresholds[np.argmax(tpr - fpr)]
    print(f'Alpha with Optimal ROC - {opt_alpha}')

    temp_lg = LogisticRegression(solver=GD, alpha=opt_alpha)
    temp_lg.fit(X_train.to_numpy(), y_train.to_numpy())
    best_alpha_test_error = temp_lg.loss(X_test.to_numpy(), y_test.to_numpy())
    print(f'Optimal Alpha Test Error - {best_alpha_test_error}')

    # Fitting l1- and l2-regularized logistic regression models,
    # using cross-validation to specify values of regularization parameter

    lambdas = pd.Series([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])

    for penalty in ['l1', 'l2']:
        values = []
        GD = GradientDescent(learning_rate=FixedLR(lr), max_iter=max_iter)

        for lam in lambdas:
            LG = LogisticRegression(solver=GD, penalty=penalty, lam=lam)
            values.append(cross_validate(LG, X_train.to_numpy(), y_train.to_numpy(), misclassification_error)[1])

        # validation_score = pd.Series(values).apply(lambda r: r[1])
        best_lambda = lambdas[np.argmin(values)]
        print(f'Best Lambda for {penalty.upper()} Regularization: {best_lambda}')

        loss = LogisticRegression(
            solver=GD, penalty=penalty, lam=best_lambda).fit(X_train, y_train).loss(X_test, y_test)
        print(f"Beat Lambda Test Error: {loss}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
