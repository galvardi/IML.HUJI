from math import atan2

from numpy import pi

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from IMLearn.metrics import accuracy

from plotly.subplots import make_subplots

pio.templates.default = "simple_white"
pio.renderers.default = "browser"

DIR_PATH = r"../datasets/"

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
    t = np.linspace(0, 2 * np.pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")

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
    data = np.load(DIR_PATH + filename)
    return data[:, :2], data[:, 2]


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    def callback_func(fit: Perceptron, x: np.ndarray, y: int):
        losses.append(fit.loss(X, y_))

    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y_ = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        percy = Perceptron(callback=callback_func)
        percy.fit(X, y_)
        fig1 = go.Figure(go.Scatter(x=list(range(len(losses))), y=losses))
        fig1.update_layout(title=n, xaxis_title="Iteration",
                          yaxis_title="Loss")
        fig1.show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y_ = load_dataset(f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y_)
        lda_pred = lda.predict(X)
        naive = GaussianNaiveBayes()
        naive.fit(X,y_)
        naive_pred = naive.predict(X)

        # Create subplots
        naive_accuracy = accuracy(y_, naive_pred)
        lda_accuracy = accuracy(y_, lda_pred)
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"Gaussian Naive Bayes With Accuracy"
                            f" - {naive_accuracy}",
                            f"Linear discriminant analysis "
                            f"With Accuracy - {lda_accuracy}"))

        # Add traces for data-points setting symbols and colors
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1],
                                 mode="markers", marker=dict(symbol=y_,
                                                             color=naive_pred,
                                                             size=8)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1],
                                 mode="markers", marker=dict(symbol=y_,
                                                             color=lda_pred,
                                                             size=8)),
                      row=1, col=2)

        for i in range(3):

        # Add `X` dots specifying fitted Gaussians' means

            fig.add_trace(go.Scatter(x=[naive.mu_[i][0]], y=[naive.mu_[i][1]],
                                     mode="markers",
                                     marker=dict(color="black",
                                                 symbol="x",
                                                 size=10)),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=[lda.mu_[i][0]], y=[lda.mu_[i][1]],
                                     mode="markers", marker=dict(
                    color="black",
                    symbol="x",
                    size=10)),
                          row=1, col=2)
        # Add ellipses depicting the covariances of the fitted Gaussians

            fig.add_trace(get_ellipse(naive.mu_[i],np.diag(naive.vars_[i])),
                                      row=1,
                          col=1)
            fig.add_trace(get_ellipse(lda.mu_[i],lda.cov_),row=1,col=2)


        fig.update_layout(title=f"{f} Dataset",showlegend=False)
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
