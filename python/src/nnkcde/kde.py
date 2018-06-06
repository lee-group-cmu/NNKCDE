"""Functions for kernel density estimation."""

import numpy as np
import statsmodels.api as sm
import scipy.stats

def kde(responses, grid, bandwidth):
    """Calculates the kernel density estimate.

    Arguments
    ---------
    responses : numpy matrix
       The training responses; each row corresponds to an observation,
       each column corresponds to a variable.
    grid : numpy matrix
        The grid points at which the KDE is evaluated.
    bandwidth : numpy array or string
        The bandwidth for the kernel density estimate; array specifies
        the diagonal of the bandwidth matrix. Strings include
        "scott", "silverman", and "normal_reference" for univariate densities and
        "normal_reference", "cv_ml", and "cv_ls" for multivariate densities.

    Returns
    -------
    numpy array
       The density evaluated at the grid points.

    """

    if len(grid.shape) == 1:
        grid = grid.reshape(-1, 1)
    if len(responses.shape) == 1:
        responses = responses.reshape(-1, 1)


    n_grid, n_dim = grid.shape
    n_obs, _ = responses.shape
    density = np.zeros(n_grid)

    if n_dim == 1:
        kde = sm.nonparametric.KDEUnivariate(responses[:, 0])
        kde.fit(bw = bandwidth, fft = False)
        return kde.evaluate(grid[:, 0])
    else:
        if isinstance(bandwidth, (float, int)):
            bandwidth = [bandwidth] * n_dim
        kde = sm.nonparametric.KDEMultivariate(responses, var_type = "c" * n_dim,
                                               bw = bandwidth)
        return kde.pdf(grid)
