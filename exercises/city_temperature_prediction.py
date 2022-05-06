import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"



File_PATH = r"C:\Users\galva\PycharmProjects\IML.HUJI\datasets\City_Temperature.csv"



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
    row_rem = df.index.values[(df["Temp"] < -50)]
    row_rem = np.append(row_rem, df.index.values[(df["Temp"] > 50)])
    df = df.drop(row_rem)
    df["dayofyear"] = df["Date"].dt.dayofyear

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data(File_PATH)

    # Question 2 - Exploring data for specific country
    is_df = data.drop(data.index.values[(data["Country"] != "Israel")])
    is_df["Year"] = is_df["Year"].astype(str)
    fig = px.scatter(is_df, x="dayofyear", y="Temp", color="Year",
                     template="seaborn")
    fig.show()
    df = is_df.groupby("Month").Temp.agg("std").to_frame()
    df.rename(columns={"Temp": "Temp Deviation"}, inplace=True)
    fig2 = px.bar(df, x=df.index, y="Temp Deviation")
    fig2.show()

    # # Question 3 - Exploring differences between countries
    grouped_df = data.groupby(["Country", "Month"]).agg({"Temp": ["mean",
                                                                  "std"]}).reset_index()
    grouped_df.columns = ["Country", "Month", "Mean Temp", "Std Temp"]
    fig3 = px.line(grouped_df, x="Month", y="Mean Temp", error_y="Std Temp",
                   color="Country", template="seaborn")
    fig3.show()

    # # Question 4 - Fitting model for different values of `k`
    X = is_df["dayofyear"].to_frame()
    y = is_df["Temp"]
    train_x, train_y, test_x, test_y = split_train_test(X,y, 0.75)
    deg_err = {}
    for k in range(1, 11):
        model = PolynomialFitting(k)
        train_x_reshaped = train_x.to_numpy().reshape(train_x.shape[0])
        train_y_nparr = train_y.to_numpy()
        model.fit(train_x_reshaped,train_y_nparr)
        loss = round(model.loss(test_x.to_numpy().reshape(test_x.shape[0]),
                                test_y.to_numpy()), 2)
        deg_err[k] = loss
    print(deg_err)
    fig4 = px.bar(x=list(deg_err.keys()),y=list(deg_err.values()))
    fig4.update_layout(xaxis_title='Polynomial Degree', yaxis_title='model '
                                                                    'loss',
                       title='MSE with respect to degree')
    fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    is_df = PolynomialFitting(5)
    is_df.fit(X.to_numpy().reshape(X.shape[0]), y.to_numpy())
    countries = list(data["Country"].drop_duplicates(inplace=False))
    error_by_country = {}
    for country in countries:
        if country == "Israel":
            continue
        country_data = data.drop(data[data.Country != country].index)
        X_data = country_data["dayofyear"].to_numpy()
        y_data = country_data["Temp"].to_numpy()
        error_by_country[country] = is_df.loss(X_data.reshape(
            X_data.shape[0]), y_data)
    fig5 = px.bar(x=list(error_by_country.keys()), y=list(error_by_country.values()))
    fig5.update_layout(xaxis_title="Country", yaxis_title="MSE",
                       title='MSE For Israeli Model')
    fig5.show()