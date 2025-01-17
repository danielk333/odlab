from typing import Optional
import numpy as np


def autocovariance(
    chain: np.ndarray,
    max_k: Optional[int] = None,
    min_k: Optional[int] = None,
) -> np.ndarray:
    """Calculate the natural estimator of the autocovariance function of a Markov chain.

    The autocovariance function is defined as
    $$
        \\gamma_k = cov(g(X_i), g(X_{i + k}))
    $$
    where $g$ is a real-valued function on the state space and $X_j$ is a sample of the
    Markov chain in that state space.

    The natural estimator of the autocovariance function and the mean is
    $$
        \\gamma_k = \\frac{1}{n} \\sum_{i=1}^{n - k} 
            (g(X_i) - \\hat{\\mu}_n)(g(X_{i + k) - \\hat{\\mu}_n), \\\\
        \\hat{\\mu}_n = \\frac{1}{n} \\sum_{i=1}^{n} g(X_i)
    $$

    Parameters
    ----------
    chain : np.ndarray
        The Markov chain represented as a (N, M) matrix where N is the number of dimensions
        and M is the number of steps
    max_k : optional, int
        The maximum autocorrelation length
    min_k : optional, int
        The minimum autocorrelation length

    Returns
    -------
    float or numpy.ndarray
        autocovariance function between $k_min$ and $k_max$ [1]

    """
    _n = chain.shape[1]
    dims = chain.shape[0]

    if max_k is None:
        max_k = _n
    else:
        if max_k >= _n:
            max_k = _n

    if min_k is None:
        min_k = 0
    else:
        if min_k >= _n:
            min_k = _n - 1

    gamma = np.empty(
        (
            dims,
            max_k - min_k,
        ),
        dtype=chain.dtype,
    )
    mu = np.mean(chain, axis=1)

    for vari in range(dims):
        for k in range(min_k, max_k):
            covi = chain[vari, :(_n - k)] - mu[vari]
            covik = chain[vari, k:_n] - mu[vari]
            gamma[vari, k] = np.sum(covi * covik) / float(_n)

    return gamma


def batch_mean(chain, batch_size):
    _n = chain.shape[1]
    dims = chain.shape[0]
    if batch_size > _n:
        raise Exception("Not enough samples to calculate batch statistics")

    _max = batch_size
    batches = _n // batch_size
    batch_mean = np.empty((dims, batches,), dtype=chain.dtype)
    for ind in range(batches):
        batch = chain[:, (_max - batch_size):_max]
        _max += batch_size
        batch_mean[:, ind] = np.mean(batch, axis=1)

    return batch_mean
