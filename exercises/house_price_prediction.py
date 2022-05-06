from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

OUT_PATH = r"graph imgs"
FILE_PATH = r"C:\Users\galva\PycharmProjects\IML.HUJI\datasets\house_prices.csv"


pio.templates.default = "simple_white"
pio.renderers.default = "browser"


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
    df["date"] = pd.to_datetime(df['date'], errors="coerce")
    invalid_rows = df.index.values[pd.isnull(df["date"])]
    df = df.drop(invalid_rows)
    df = df.sort_values(by=['date']).drop_duplicates('id', keep='last')
    invalid_rows = df.index.values[df.isnull().any(1)]
    df = df.drop(invalid_rows, axis=0)
    row_rem = df.index.values[
        (df["price"] < 0)]  # removing rows with negative
    row_rem = np.append(row_rem, df.index.values[(df["bedrooms"] <= 0)])
    row_rem = np.append(row_rem, df.index.values[(df["bathrooms"] <= 0)])
    row_rem = np.append(row_rem, df.index.values[(df["sqft_living"] <= 0)])
    row_rem = np.append(row_rem, df.index.values[(df["sqft_lot"] <= 0)])
    row_rem = np.append(row_rem, df.index.values[(df["floors"] <= 0)])
    row_rem = np.append(row_rem, df.index.values[(df["yr_built"] <= 0)])
    row_rem = np.append(row_rem, df.index.values[(df["condition"] <= 0)])
    row_rem = np.append(row_rem, df.index.values[(df["condition"] > 5)])
    row_rem = np.append(row_rem, df.index.values[(df["view"] < 0)])
    row_rem = np.append(row_rem, df.index.values[(df["grade"] <= 0)])
    row_rem = np.append(row_rem, df.index.values[(df["grade"] > 13)])
    row_rem = np.unique(row_rem)
    df = df.drop(row_rem)

    df = pd.get_dummies(df, columns=["zipcode"])
    prices = df["price"]
    df = df.drop(columns=["price", "date", "id"])

    return df, prices


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
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
    std_y = np.std(y)
    colms = X.columns[:17]  # i excluded the categorical zipcode features
    for col in colms:
        corr_feat, feat = pearson_corr(X, col, std_y, y)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=feat, y=y, name="{}".format(col),
                                 mode="markers"))
        fig.update_layout(title="{} - Prearson Correlation: {}".format(
            col, corr_feat),
            xaxis_title=f"{col}", yaxis_title="Price", template="seaborn")
        fig.show()
        pio.write_html(fig, output_path + f"\{col}")


def pearson_corr(X, col, std_y, y):
    feat = X[col]
    cov_feat = np.cov(feat, y)[0][1]
    std_feat = np.std(feat)
    corr_feat = cov_feat / (std_feat * std_y)
    return corr_feat, feat


if __name__ == '__main__':
    np.random.seed(0)
    lin_reg = LinearRegression()
    # Question 1 - Load and preprocessing of housing prices dataset

    X, y = load_data(FILE_PATH)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, OUT_PATH)

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    mean_dic = {}
    for p in range(10, 101):
        mean_sample_arr = []
        for _ in range(10):
            train_samp = train_x.sample(
                int(np.floor((p / 100) * train_x.shape[
                    0])))
            ind = train_samp.index.values.tolist()
            samp_y = train_y.loc[ind]
            samp_y = samp_y.to_numpy()
            train_samp = train_samp.to_numpy()
            lin_reg.fit(train_samp, samp_y)
            mean_sample_arr.append(
                lin_reg.loss(test_x.to_numpy(), test_y.to_numpy()))
        samp_mean = np.mean(mean_sample_arr)
        mean_dic[p] = [samp_mean, samp_mean + 2 * np.std(mean_sample_arr),
                       samp_mean - 2 * np.std(mean_sample_arr)]

    get_val_arr = lambda arr, ind: [arr[i][ind] for i in range(10, 101)]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(mean_dic.keys()), y=get_val_arr(
        mean_dic, 0),
                              name="mean loss",
                              mode="markers+lines"))
    fig2.add_trace(
        go.Scatter(x=list(mean_dic.keys()), y=get_val_arr(
            mean_dic, 1),
                   name="mean(loss) + 2*std(loss) ",
                   mode="markers+lines"))
    fig2.add_trace(
        go.Scatter(x=list(mean_dic.keys()), y=get_val_arr(
            mean_dic, 2),
                   name="mean(loss) - 2*std(loss) ",
                   mode="markers+lines"))
    fig2.update_layout(title="Mean prediction", template="seaborn")
    fig2.show()
