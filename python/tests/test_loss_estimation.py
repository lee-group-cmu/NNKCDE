import numpy as np
import nnkcde
import pytest

def cde_loss(cdes, z_grid, true_z):
    n_obs, _ = cdes.shape
    term1 = np.mean(np.trapz(cdes ** 2, z_grid))
    nns = [np.argmin(np.abs(z_grid - true_z[ii])) for ii in range(n_obs)]
    term2 = np.mean(cdes[range(n_obs), nns])
    return term1 - 2 * term2

def test_loss_estimation():
    def generate_data(n):
        x = np.random.random((n, 1))
        z = np.random.normal(x[:, 0], 1, n).reshape(-1, 1)
        return x, z

    np.random.seed(312)
    x_train, z_train = generate_data(1000)
    x_test, z_test = generate_data(1000)

    obj = nnkcde.NNKCDE()
    obj.fit(x_train, z_train)
    n_grid = 1000
    z_grid = np.linspace(-5.0, 5.0, n_grid)

    for bandwidth in (1.0, 1.0):
        for k in (2, 5, 10, 100):
            cde = obj.predict(x_test, z_grid, k=k, bandwidth=bandwidth)
            expected = cde_loss(cde, z_grid, z_test)
            actual = obj.estimate_loss(x_test, z_test, [k], bandwidth)[0]

            assert abs(expected - actual) <= 1e-2
