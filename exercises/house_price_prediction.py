from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

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

    # drop all samples with Nan values, and drop duplicated samples
    df = df.dropna().drop_duplicates()
    print(df.shape)
    # remove samples non-negative prices
    df = df[df.price > 0]
    print(df.shape)

    # create an indicator column - is the house renovated in since year 2000
    df["renovated_lately"] = np.where(df["yr_renovated"] >= 2000, 1, 0)

    # turn zipcodes to int values, then to one-hot vector
    df["zipcode"] = df["zipcode"].astype(int)
    df = pd.get_dummies(df, prefix='zipcode', columns=['zipcode'])  # todo ?
    print(df.shape)

    prices = df.price
    df.drop(columns=["id", "price", "yr_renovated", "lat", "long", "date"], inplace=True)
    print(df.shape)

    return df, prices


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
        output = px.scatter(pd.DataFrame({'x': X[feature], 'y': y}), x="x", y="y", trendline="ols",
                            title=f"Pearson Correlation Between {feature} (values) and <br>price (response): {corr}",
                            labels={"x": f"{feature}", "y": "Price (response values)"})
        output.write_image(f"{output_path}/pearson_correlation_{feature}.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    # load_data("../datasets/house_prices.csv")
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X, y, "./outputs")  # todo - uncomment

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)  # default is 75% train, 25% test

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    percentage = np.linspace(10, 100, 91)
    mean_lst, var_lst = np.array([]), np.array([])

    for p in percentage:
        loss_lst = np.array([])
        for test in range(10):
            partial_x = train_X.sample(frac=p / 100)
            partial_y = train_y.loc[partial_x.index]
            estimator = LinearRegression()
            estimator.fit(partial_x, partial_y)
            loss = estimator.loss(test_X, test_y)
            loss_lst = np.append(loss_lst, loss)

        mean_lst = np.append(mean_lst, np.mean(loss_lst))
        var_lst = np.append(var_lst, np.std(loss_lst))

    error_ribbon1 = mean_lst - (2 * var_lst)
    error_ribbon2 = mean_lst + 2 * var_lst
    go.Figure(
        ([go.Scatter(x=percentage, y=mean_lst, mode="markers+lines", name="Mean Prediction", line=dict(dash="dash"),
                     marker=dict(color="green", opacity=.7), ),
          go.Scatter(x=percentage, y=error_ribbon1, mode="lines", line=dict(color="lightgrey"), showlegend=False),
          go.Scatter(x=percentage, y=error_ribbon2, fill='tonexty', mode="lines",
                     line=dict(color="lightgrey"), showlegend=False)]),
        layout=go.Layout(title="Average Test Loss as a Function of Training Proportion",
                         xaxis_title=dict({'text': "Training Proportion"}),
                         yaxis_title=dict({'text': "Average Test Loss"}))).show()
