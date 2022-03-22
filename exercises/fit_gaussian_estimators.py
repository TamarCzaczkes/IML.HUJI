from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    ug = UnivariateGaussian()

    x = np.random.normal(10, 1, 1000)
    ug.fit(x)
    print("(", ug.mu_, ", ", ug.var_, ")", sep='')

    # Question 2 - Empirically showing sample mean is consistent

    num_of_samples = np.arange(10, 1010, 10)
    dist = []
    for n in num_of_samples:
        temp = x[0:n]
        dist.append(np.abs(10 - ug.fit(temp).mu_))

    go.Figure([go.Scatter(x=num_of_samples, y=dist)],
              layout=go.Layout(title=r"$\text{Absolute Distance Between Estimated and True Value of Expectation of a "
                                     r"Univariate Gaussian}$",
                               xaxis_title=r"$\text{Sample Size}$",
                               yaxis_title=r"$\text{Error (Absolute Distance)}$",
                               height=700, width=1000)).show()

    # Question 3 - Plotting Empirical PDF of fitted model

    go.Figure([go.Scatter(x=x, y=ug.pdf(x), mode='markers')],
              layout=go.Layout(title=r"$\text{Empirical PDF function under the fitted model}$",
                               xaxis_title=r"$\text{Samples}$",
                               yaxis_title=r"$\text{PDF Value}$",
                               height=700, width=1100)).show()

    # for the quiz
    # x1 = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3,
    #                2, -1, -3, 1, -4, 1, 2, 1, -4, -4, 1, 3, 2, 6, -6, 8, 3, -6,
    #                4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    # print(ug.log_likelihood(1, 1, x1))
    # print(ug.log_likelihood(10, 1, x1))


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    x = np.random.multivariate_normal(mu, cov, 1000)

    mg = MultivariateGaussian()
    mg.fit(x)
    print(mg.mu_)
    print(mg.cov_)

    # Question 5 - Likelihood evaluation

    f = np.linspace(-10, 10, 200)
    log_likelihood_calc = np.empty((f.size, f.size))

    for i in range(f.size):
        for j in range(f.size):
            log_likelihood_calc[i, j] = mg.log_likelihood(np.array([f[i], 0, f[j], 0]), cov, x)

    go.Figure(go.Heatmap(x=f, y=f, z=log_likelihood_calc),
              layout=go.Layout(title=r"$\text{log_likelihood for expectation = [f1, 0, f3, 0]}$",
                               xaxis_title=r"$\text{f3}$",
                               yaxis_title=r"$\text{f1}$",
                               height=700, width=1100)).show()

    # Question 6 - Maximum likelihood
    idx = np.unravel_index(log_likelihood_calc.argmax(), log_likelihood_calc.shape)
    print("\nThe model that achieved the maximum log-likelihood value is:\n"
          "f1 = ", f[idx[0]], "\nf3 = ", f[idx[1]], sep='')


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
