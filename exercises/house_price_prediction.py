from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset
    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    df = pd.read_csv(filename)
    print(df.columns)
    print(df.shape)

    # drop all samples with Nan values, and drop duplicate samples
    df = df.dropna().drop_duplicates()
    print(df.shape)

    # df_positive = df[df.price > 0]
    # print(df_positive.shape)

    # print(df_positive.shape)
    # df_positive = df[df.price != np.nan]
    # print(df_positive.shape)
    # df_positive1 = df_positive[df_positive.id > 0]
    # print(df_positive1.shape)
    # print(df)
    # print(np.unique(df.view))  # nan


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X.columns:
        cov = X[feature].cov(y)
        std_X = X[feature].std()
        std_y = y.std()
        corr = cov / (std_X * std_y)
        fig = go.Figure([go.Scatter(x=feature, y=y)],
                        layout=go.Layout(
                            title=f"Pearson Correlation between {feature} (values) and price (response): {corr}",
                            xaxis_title=feature, yaxis_title="price"))
        # height=700, width=1000)
        fig.write_image(f"{output_path}\\pearson_correlation_{feature}.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    load_data("../datasets/house_prices.csv")
    # X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X, y, output_path=".")


    # Question 3 - Split samples into training- and testing sets.
    # train_X, train_y, test_X, test_y = split_train_test(X, y, test_size=0.25)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    # raise NotImplementedError()
