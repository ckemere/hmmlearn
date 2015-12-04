import numpy as np
from scipy.special import gammaln


def normalize(a, axis=None):
    """Normalize the input array so that it sums to 1.

    Parameters
    ----------
    a: array_like
        Non-normalized input data.
    axis: int
        Dimension along which normalization is performed.

    Returns
    -------
    res: array, shape (n_samples, n_features)
        A with values normalized (summing to 1) along the prescribed axis

    WARNING: Modifies the array inplace.
    """
    a += np.finfo(float).eps

    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum

    # TODO: should return nothing, since the operation is inplace.
    return a


def exp_mask_zero(a):
    """Computes the exponent of input elements masking underflows."""
    with np.errstate(under="ignore"):
        out = np.exp(a)
    out[out == 0] = np.finfo(float).eps
    return out


def log_mask_zero(a):
    """Computes the log of input elements masking underflows."""
    with np.errstate(divide="ignore"):
        out = np.log(a)
    out[np.isnan(out)] = 0.0
    return out


def logsumexp(a, axis=0):
    """Compute the log of the sum of exponentials of input elements.

    Notes
    -----
    Unlike the versions implemented in ``scipy.misc`` and
    ``sklearn.utils.extmath`` this version explicitly masks the underflows
    occured during ``np.exp``.

    Examples
    --------
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107
    """
    a = np.rollaxis(a, axis)
    a_max = a.max(axis=0)
    out = np.log(exp_mask_zero(a - a_max).sum(axis=0))
    out += a_max
    return out


def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {0:d} samples in lengths array {1!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield start[i], end[i]


class assert_raises(object):
    """A backport of the ``assert_raises`` context manager for Python2.6."""
    def __init__(self, expected):
        self.expected = expected

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            exc_name = getattr(self.expected, "__name__", str(self.expected))
            raise AssertionError("{0} is not raised".format(exc_name))

        # propagate the unexpected exception if any.
        return issubclass(exc_type, self.expected)

def log_multivariate_poisson_density(X, means) :
  # modeled on log_multivariate_normal_density from sklearn.mixture
  n_samples, n_dim = X.shape
  # -lambda + k log(lambda) - log(k!)
  log_means = np.where(means > 0, np.log(means), -1e3)
  lpr =  np.dot(X, log_means.T)
  lpr += -means # means vector is broadcast across the observation dimenension
  log_factorial = np.sum(gammaln(X + 1), axis=1)
  lpr += -log_factorial[:,None] # logfactobs vector broad cast across the state dimension
  return lpr

