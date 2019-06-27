import numpy as np
import nnkcde
import pytest

def cde_loss(cdes, z_grid, true_z):
    n_obs, _ = cdes.shape
    term1 = np.mean(np.trapz(cdes ** 2, z_grid))
    nns = [np.argmin(np.abs(z_grid - true_z[ii])) for ii in range(n_obs)]
    term2 = np.mean(cdes[range(n_obs), nns])
    return term1 - 2 * term2

def test_beta_example_performance():
    def generate_data(n):
        x = 5.0 * np.random.random((n, 2))
        z = np.random.beta(x[:, 0] + 5, x[:, 1] + 5, n)
        return x, z

    np.random.seed(42)
    x_train, z_train = generate_data(100000)
    x_test, z_test = generate_data(1000)

    k = 100
    bandwidth = 0.1

    obj = nnkcde.NNKCDE(k=k)
    obj.fit(x_train, z_train)
    n_grid = 1000
    z_grid = np.linspace(0, 1, n_grid)
    density = obj.predict(x_test, z_grid, bandwidth=bandwidth)
    assert cde_loss(density, z_grid, z_test) < -2.0
