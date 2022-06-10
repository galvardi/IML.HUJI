import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

YAXIS_TITLE = "error"

XAXIS_TITLE = "Number of learners"

PLOT1_TITLE = "Train - Test sets error as a function of amount of learners"


def generate_data(n: int, noise_ratio: float) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    boost = AdaBoost(DecisionStump, n_learners)
    boost.fit(train_X, train_y)
    train_err = np.array([])
    test_err = np.array([])
    for i in range(1, n_learners):
        train_err = np.append(train_err,
                              boost.partial_loss(train_X, train_y, i))
        test_err = np.append(test_err, boost.partial_loss(test_X, test_y, i))

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=list(range(1,n_learners)),y=train_err,
                                name="train_error"))
    fig1.add_trace(go.Scatter(x=list(range(1,n_learners)),y=test_err
                     ,name="test_error"))
    fig1.update_layout(title=PLOT1_TITLE, xaxis_title=XAXIS_TITLE,
                       yaxis_title=YAXIS_TITLE)
    fig1.show()

    # # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    fig2 = make_subplots(rows=2, cols=2,
                        subplot_titles=[rf"$\textbf{{{t}}} - Learners$" for
                                        t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)

    symbols = np.array(["circle", "x"])
    y = np.where(test_y == 1, 1, 0)
    for i, t in enumerate(T):
        fig2.add_traces([decision_surface(lambda s:boost.partial_predict(s,t),
                                          lims[0],
                                         lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                   mode="markers",
                                   showlegend=False,
                                   marker=dict(color=y,
                                               symbol=symbols[y],
                                               colorscale=[custom[0],
                                                           custom[-1]],
                                               line=dict(color="black",
                                                         width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig2.update_layout(
        title=rf"$\textbf{{Decision Boundaries Of Models with [5, 50, 100, 250] Learners}}$",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig2.show()

    # Question 3: Decision surface of best performing ensemble

    best_ensemble = np.argmin(test_err) + 1
    best_pred = boost.partial_predict(test_X, best_ensemble)
    best_accuracy = accuracy(test_y, best_pred)

    fig3 = go.Figure()
    symbols = np.array(["circle", "x"])
    y = np.where(test_y == 1, 1, 0)
    fig3.add_traces(
        [decision_surface(lambda s: boost.partial_predict(s, best_ensemble),
                          lims[0],
                          lims[1], showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                    mode="markers",
                    showlegend=False,
                    marker=dict(color=y,
                                symbol=symbols[y],
                                colorscale=[custom[0],
                                            custom[-1]],
                                line=dict(color="black",
                                          width=1)))])
    fig3.update_layout(
        title=rf"$\textbf{{Decision Boundarie Of Best Ensemble size - "
              rf"{best_ensemble} with Accuracy - {best_accuracy}}}$")
    fig3.show()

    # Question 4: Decision surface with weighted samples

    normal = boost.D_ / np.max(boost.D_) * 5
    fig4 = go.Figure().add_traces([decision_surface(
        lambda x: boost.partial_predict(x, T=250),
        lims[0], lims[1], showscale=False),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                   mode="markers",
                   showlegend=False,
                   marker=dict(color=y, size=normal,
                               symbol=symbols[y],
                               colorscale=[
                                   custom[0],
                                   custom[-1]],
                               line=dict(
                                   color="black",
                                   width=1)))])
    fig4.update_layout(title="Training data prediction proportional to point weight")
    fig4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
