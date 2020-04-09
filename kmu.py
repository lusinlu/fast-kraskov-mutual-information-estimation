""""Implementation based on
B.C.Ross "Mutual Information between Discrete and Continuous Data Sets"
"""
import numpy as np
from scipy.special import digamma
from pyitlib import discrete_random_variable as drv
import faiss
from bisect import bisect
import sys
np.set_printoptions(threshold=sys.maxsize)


def compute_kmu(x, y, per_filter=True, avarage=True, n_neighbors=3):
    """Compute mutual information between continious and discrete variables
    :parameter
    x : ndarray, shape (batch_size, n_filters, height, width)
         4d  continious variable,
    y : ndarray,  shape (batch_size, )
        1d discrete variable
        per_filter : bool,
        Whether to calculate mu between each 3d filter and discrete variable, or full 4d tensor and discrete variable
    avarage : bool,
        In case of per_filter=True, avarage the result or no
    n_neighbors: int,
        Number of nearest neighbors to search for each point
     :returns
     kmu : float, or list of floats (depends on per_filter parameter),
        Estimated mutual information
     """
    if per_filter:
        filters_count = x.shape[1]
        x = x.reshape(x.shape[0],x.shape[1], -1)
        kmu = [mu_approximate(x[:, i, :], y, n_neighbors=n_neighbors) for i in range(filters_count)]
    else:
        x = x.reshape(x.shape[0], -1)
        kmu = mu_approximate(x, y, n_neighbors=n_neighbors)

    if avarage:
        kmu = np.mean(kmu)

    return kmu


def nn_faiss(x, k):
    """Compute nearest neighbors for the each point in the given set
    :parameter
    x : ndarray, shape (n_samples, )
        Set of points
    k : int,
        Number of nearest neighbors to search for each point
    :returns
    d : ndarray, shape (n_samples, n_neighbors)
        Distances between the point and each neighbor for each point,
    """
    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)
    d, i = index.search(x, k + 1)
    return d


def mu_approximate(c, d, n_neighbors):
    """Mutual information calculation based on approximate nearest neighbors
      :parameter
      c : ndarray, shape (n_samples,)
          Samples of a continuous random variable.
      d : ndarray, shape (n_samples,)
          Samples of a discrete random variable.
      n_neighbors : int
          Number of nearest neighbors to search for each point, see [1]_.
      :returns
      mi : float
          Estimated mutual information. If it turned out to be negative it is
          replace by 0.
      Notes
      -----
      True mutual information can't be negative. If its estimate by a numerical
      method is negative, it means (providing the method is adequate) that the
      mutual information is close to 0 and replacing it by 0 is a reasonable
      strategy.
      ----------
    """
    n_samples = c.shape[0]
    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)

    for label in np.unique(d):
        mask = d == label
        count = np.sum(mask)
        if count > n_neighbors + 1:

            k = min(n_neighbors, count - 1)
            try:
                dist = nn_faiss(c[mask, :], k=k)
            except FloatingPointError as ex:
                print(ex)
            radius[mask] = np.nextafter(dist[:, -1], 0)

            k_all[mask] = k
        label_counts[mask] = count

    # Ignore points with unique labels.
    mask = label_counts > 1
    n_samples = np.sum(mask)
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c = c[mask]
    radius_faiss = radius[mask]

    index = faiss.IndexFlatL2(c.shape[1])
    index.add(c)
    # find at max 100 neighbors. Can lead to inaccuracies, but will speed up the process
    D = nn_faiss(c, k=100)
    idc_counts = np.array([max(0, (bisect(D[i], radius_faiss[i]))) for i in range(c.shape[0])])

    mi = (digamma(n_samples) + np.mean(digamma(k_all)) -
          np.mean(digamma(label_counts)) -
          np.mean(digamma(idc_counts + 1)))

    # mutual information can not be too high. It means that approximation gave bad results
    if mi > 100: mi = -1
    return max(0, mi)










