from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors.polynomial_fitting import PolynomialFitting
from IMLearn.learners.regressors import LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = "browser"
pio.templates.default = "seaborn"


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

    res_func = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    x = np.linspace(-1.2, 2, n_samples)
    y_noiseless = res_func(x)
    y_noise = y_noiseless + np.random.normal(0, noise, n_samples)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(x),
                                                        pd.Series(y_noise).rename("labels")
                                                        , 2 / 3)
    train_X, train_y, = np.array(train_X).reshape(len(
        train_X)), np.array(train_y)
    test_X, test_y = np.array(test_X).reshape(len(
        test_X)), np.array(test_y)
    # train_X, train_y, test_X, test_y = np.array(train_X).reshape(len(
    #     train_X)), np.array(train_y), np.array(test_X).reshape(len(
    #     test_X)), np.array(test_y)
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=x, y=y_noiseless, mode="markers+lines"))
    fig1.add_trace(go.Scatter(x=train_X, y=train_y,
                              mode="markers", marker=dict(color='red'),
                              name="train points"))
    fig1.add_trace(go.Scatter(x=test_X, y=test_y,
                              mode="markers", marker=dict(color='black'),
                              name="test points"))
    fig1.update_layout(xaxis_title="x",
                       yaxis_title="f(x)")
    fig1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    avg_train_err = []
    avg_validation_err = []
    min_validation = float("inf")
    mini_valid_deg = 0
    for deg in range(11):
        err_train, err_validation = cross_validate(PolynomialFitting(deg), train_X, train_y, mean_square_error)
        if err_validation <= min_validation:
            min_validation = err_validation
            mini_valid_deg = deg
        avg_train_err.append(err_train)
        avg_validation_err.append(err_validation)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(11)), y=avg_validation_err,
                              mode="markers+lines", marker=dict(
            color='blue'), name="validation error"))
    fig2.add_trace(go.Scatter(x=list(range(11)), y=avg_train_err,
                              mode="markers+lines", marker=dict(
            color="red"), name="training error"))
    fig2.update_layout(title="training and validation error as a function of "
                             "polynomial "
                             "degree", xaxis_title="polynomial degree",
                       yaxis_title="errors")
    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error

    est_full_train = PolynomialFitting(mini_valid_deg)
    est_full_train.fit(train_X, train_y)
    test_err = est_full_train.loss(test_X, test_y)
    print(
        f"the lowest validation error was achieved by deg {mini_valid_deg}.\n fitting a polynomial of this degree on the "
        f"whole training set returns a error rate of {round(test_err, 2)}")


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
    data = load_diabetes()
    X, y = data.data, data.target
    train_X, train_y, test_X, test_y = X[:50], y[:50], X[50:], y[50:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    rid_avg_train_err, rid_avg_valid_err = [], []
    las_avg_train_err, las_avg_valid_err = [], []
    lambdas = np.linspace(0, 1, n_evaluations)
    for lam in lambdas:
        ridge_model = RidgeRegression(lam)
        lasso_model = Lasso(lam)
        err_train, err_valid = cross_validate(ridge_model, train_X, train_y, mean_square_error)
        rid_avg_train_err.append(err_train)
        rid_avg_valid_err.append(err_valid)
        err_train, err_valid = cross_validate(lasso_model, train_X, train_y, mean_square_error)
        las_avg_train_err.append(err_train)
        las_avg_valid_err.append(err_valid)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=lambdas,y=rid_avg_train_err,mode="markers+lines",name="training errors"))
    fig3.add_trace(go.Scatter(x=lambdas,y=rid_avg_valid_err,mode="markers+lines",name="validation errors"))
    fig3.update_layout(title="Ridge training and validation errors a function of lambda param",xaxis_title="lambda",
                       yaxis_title="errors")
    fig3.show()
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=lambdas, y=las_avg_train_err, mode="markers+lines", name="training errors"))
    fig4.add_trace(go.Scatter(x=lambdas, y=las_avg_valid_err, mode="markers+lines", name="validation errors"))
    fig4.update_layout(title="Lasso training and validation errors a function of lambda param",xaxis_title="lambda",
                       yaxis_title="errors")
    fig4.show()

# Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lasso_lam_pram = np.argmin(las_avg_valid_err)
    best_lasso_lam_pram = lambdas[best_lasso_lam_pram]
    best_ridge_lam_pram = np.argmin(rid_avg_valid_err)
    best_ridge_lam_pram = lambdas[best_ridge_lam_pram]
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(train_X,train_y)
    reg_loss = lin_reg_model.loss(test_X,test_y)
    lasso_model = Lasso(best_lasso_lam_pram)
    lasso_model.fit(train_X,train_y)
    lasso_loss = mean_square_error(test_y,lasso_model.predict(test_X))
    ridge_model = RidgeRegression(best_ridge_lam_pram)
    ridge_model.fit(train_X,train_y)
    ridge_loss = mean_square_error(test_y,ridge_model.predict(test_X))
    print(f"Best lambda param: ridge - {best_ridge_lam_pram} , lasso - {best_lasso_lam_pram}")
    print(f"Model errors : ridge - {ridge_loss} , lasso - {lasso_loss} , linear - {reg_loss}")




if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(100, 5)
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
