import numpy as np
import scipy.stats

import nnkcde

def test_k_equals_1():
    np.random.seed(91)

    z_grid = np.linspace(-1, 1, 100)

    n_train = 1000
    x_train = np.random.uniform(0, 1, n_train)
    z_train = np.random.normal(x_train, 1, n_train)

    obj = nnkcde.NNKCDE(x_train, z_train)

    for bandwidth in (0.1, 0.3, 1.0, 2.0, 3.0):
        x_obs = np.random.uniform(0, 1, 1)
        nearest = np.argmin(np.abs(x_train - x_obs))
        expected = scipy.stats.norm.pdf(z_grid, z_train[nearest], bandwidth).reshape(1, -1)
        preds = obj.predict(x_obs, z_grid, k = 1, bandwidth = bandwidth)
        np.testing.assert_almost_equal(expected, preds)

def test_k_equals_n():
    np.random.seed(92)

    z_grid = np.linspace(-1, 1, 100)

    n_train = 1000
    x_train = np.random.uniform(0, 1, n_train)
    z_train = np.random.normal(x_train, 1, n_train)

    obj = nnkcde.NNKCDE(x_train, z_train)

    for bandwidth in (0.1, 0.3, 1.0, 2.0, 3.0):
        x_obs = np.random.uniform(0, 1, 1)
        expected = nnkcde.kde.kde(z_train, z_grid, bandwidth).reshape(1, -1)
        preds = obj.predict(x_obs, z_grid, k = n_train, bandwidth = bandwidth)
        np.testing.assert_almost_equal(expected, preds)
