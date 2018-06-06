import sklearn.neighbors
import scipy.spatial
import scipy.stats
import numpy as np
from .kde import kde

class NNKCDE(object):
    def __init__(self, x_train, z_train, k=None):
        self.k = k

        if len(z_train.shape) == 1:
            z_train = z_train.reshape(-1, 1)
        if len(x_train.shape) == 1:
            x_train = x_train.reshape(-1, 1)

        self.z_train = z_train
        self.tree = sklearn.neighbors.BallTree(x_train)

    def tune(self, x_validation, z_validation, k_grid):
        raise NotImplementedError

    def estimate_loss(self, x_validation, z_validation, k_grid, bandwidth):
        k_max = max(k_grid)
        n_k = len(k_grid)

        if len(z_validation.shape) == 1:
            z_validation = z_validation.reshape(-1, 1)
        if len(x_validation.shape) == 1:
            x_validation = x_validation.reshape(-1, 1)

        n_dim = z_validation.shape[1]

        losses = np.zeros(n_k)

        n_validation = x_validation.shape[0]

        if n_dim == 1:
            invh = np.array([[1.0 / bandwidth ** 2]])
            det = bandwidth ** 2
        else:
            invh = np.diag(1.0 / np.array(bandwidth ** 2))
            det = 1.0 / np.linalg.det(invh)

        term1const = (2 * np.pi) ** (-n_dim / 2.0) / (det * np.sqrt(2))

        ids = self.tree.query(x_validation, k=k_max, return_distance=False)
        for idx in range(n_validation):
            for ik, k in enumerate(k_grid):
                z_close = self.z_train[ids[idx][:k], :]
                dkk = scipy.spatial.distance.pdist(z_close, "mahalanobis",
                                                   VI=invh) ** 2

                diag_sum = k
                lower_sum = np.sum(np.exp(-dkk / 4.0))
                tot_sum = 2 * lower_sum + diag_sum
                term1 = term1const * tot_sum / (k ** 2)

                dk1 = scipy.spatial.distance.cdist(z_close,
                                                   z_validation[idx:(idx+1), :],
                                                   "mahalanobis", VI=invh)
                term2 = np.mean(scipy.stats.norm.pdf(dk1, 0.0, 1.0)) / det

                losses[ik] += term1 - 2 * term2

        return losses / n_validation

    def predict(self, x_test, z_grid, k=None, bandwidth=None):
        n_test = x_test.shape[0]

        if k is None:
            k = self.k

        if len(x_test.shape) == 1:
            x_test = x_test.reshape(-1, 1)
        if len(z_grid.shape) == 1:
            z_grid = z_grid.reshape(-1, 1)
        n_grid = z_grid.shape[0]

        ids = self.tree.query(x_test, k=k, return_distance=False)

        cdes = np.empty((n_test, n_grid))
        for idx in range(n_test):
            cdes[idx, :] = kde(self.z_train[ids[idx], :], z_grid, bandwidth)

        return cdes
