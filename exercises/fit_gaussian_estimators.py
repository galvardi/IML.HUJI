from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

SAMPLE_SIZE = 1000
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    univariate1 = UnivariateGaussian()
    org_mu = 10
    org_var = 1
    sample_arr = np.random.normal(org_mu, org_var, SAMPLE_SIZE)
    univariate1.fit(sample_arr)
    print(univariate1.mu_, univariate1.var_)

    # Question 2 - Empirically showing sample mean is consistent
    dists_arr_y = []
    univariate2 = UnivariateGaussian()
    for i in range(10, SAMPLE_SIZE, 10):
        univariate2.fit((sample_arr[:i]))
        dists_arr_y.append(abs(univariate2.mu_ - org_mu))
    x_axis = np.arange(10, SAMPLE_SIZE, 10)
    go.Figure([go.Scatter(x=x_axis, y=dists_arr_y, mode='markers+lines',
                          name=r'$\expected_dist$')],
              layout=go.Layout(
                  title=r"$\text{Distance between estimated and true mean "
                        r"expectation}$",
                  xaxis_title="$\\text{Sample Size}$",
                  yaxis_title="r$Expected & Actual Delta$",
                  height=800)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    x_axis = np.sort(sample_arr)
    go.Figure([go.Scatter(x=x_axis, y=univariate1.pdf(x_axis),
    mode='markers+lines',
                          name=r'$\pdf$')],
              layout=go.Layout(
                  title=r"$\text{Empirical PDF}$",
                  xaxis_title="$\\text{Sample Value}$",
                  yaxis_title="r$PDF alue$",
                  height=700)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    matrix = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0],
                  [0.5, 0, 0, 1]]
    samp_arr = np.random.multivariate_normal([0, 0, 4, 0], matrix, SAMPLE_SIZE)
    mult_gauss = MultivariateGaussian()

    mult_gauss.fit(samp_arr)
    print(f"expectation:{mult_gauss.mu_}, \n cov matrix: {mult_gauss.cov_}")

    # Question 5 - Likelihood evaluation

    data = np.linspace(-10, 10, 200)
    V = []
    max = -np.inf
    for f1 in data:
        liklyhood = []
        for f3 in data:
            log_like = mult_gauss.log_likelihood([f1, 0, f3, 0], matrix,
                                                  samp_arr)
            liklyhood.append(log_like)
            if log_like > max:
                max = log_like
                cor = (f1, f3)

        V.append(liklyhood)
    V = np.transpose(V)
    go.Figure(go.Heatmap(x=data, y=data, z=V)).update_layout(
        title="log likelihood"
        , xaxis_title="f1 Values", yaxis_title="f3 Values",
        height=900, width=1500).show()
    # Question 6 - Maximum liklyhood

    print(
        f"maximum log-likelihood in : "
        f"{[format(cor[i], '0.3f') for i in [0, 1]]}"
        f" and the value is : {format(max, '0.3f')}")

if __name__ == '__main__':
    np.random.seed(0)

    print("hello")
    test_univariate_gaussian()
    test_multivariate_gaussian()
