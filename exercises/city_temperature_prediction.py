import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    # remove invalid samples
    df = df.dropna().drop_duplicates()
    df = df[df.Temp > -70]
    df["DayOfYear"] = df.Date.dt.day_of_year

    return df


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = data[data.Country == "Israel"]
    px.scatter(x=israel_data.DayOfYear, y=israel_data.Temp, color=israel_data.Year.astype(str),
               title=f"Average Daily Temperature as a Function of Date in Israel",
               labels={"x": "Day of Year", "y": "Temperature"}).show()

    std_israel_by_month = israel_data.groupby("Month").agg(np.std).Temp
    px.bar(x=std_israel_by_month.index, y=std_israel_by_month,
           title=f"Standard Deviation of Temperature by Month in Israel",
           labels={"x": "Month", "y": "Standard Deviation of Daily Temperature"}).show()

    # Question 3 - Exploring differences between countries
    country_month_data = data.groupby(["Country", "Month"]).agg([np.mean, np.std]).Temp.reset_index()
    out = px.line(country_month_data, x='Month', y='mean', color='Country', error_y='std',
                  title="Average and Standard Deviation of Temperature by Country")
    out.update_layout(yaxis_title="Average Temperature").show()

    # Question 4 - Fitting model for different values of `k`
    # default is 75% train, 25% test:
    train_X, train_y, test_X, test_y = split_train_test(israel_data.DayOfYear, israel_data.Temp)
    k_range = np.arange(1, 11)
    loss_lst = np.array([])
    lowest_loss, opt_k = np.inf, 0

    for k in k_range:
        model = PolynomialFitting(k)
        model.fit(train_X.to_numpy(), train_y.to_numpy())
        loss = np.round(model.loss(train_X.to_numpy(), train_y.to_numpy()), 2)
        loss_lst = np.append(loss_lst, loss)
        if loss < lowest_loss:
            lowest_loss, opt_k = loss, k

    px.bar(x=k_range, y=loss_lst, text_auto=True,
           title=f"The Loss of PolynomialFitting Model for Different Polynomial Degree",
           labels={"x": "Degree", "y": "Loss (error)"}).show()

    # Question 5 - Evaluating fitted model on different countries
    loss_lst = np.array([])
    model = PolynomialFitting(opt_k)
    model.fit(israel_data.DayOfYear, israel_data.Temp)
    countries = data.Country.unique()
    countries = countries[countries != "Israel"]
    for country in countries:
        country_data = data[data.Country == country]
        loss = np.round(model.loss(country_data.DayOfYear, country_data.Temp), 2)
        loss_lst = np.append(loss_lst, loss)

    px.bar(x=countries, y=loss_lst, title=f"Model's Error in Other Countries",
           labels={"x": "Country", "y": "Error"}).show()
