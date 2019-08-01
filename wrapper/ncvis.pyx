# distutils: language=c++

from wrapper cimport cncvis
import numpy as np
cimport numpy as cnp
import ctypes
from multiprocessing import cpu_count

from scipy.optimize import curve_fit
def find_ab_params(spread=1., min_dist=0.1):
    """
    https://github.com/lmcinnes/umap/blob/834184f9c0455f26db13ab148c0abd2d3767d968/umap/umap_.py#L1049

    Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]

cdef class NCVisWrapper:
    cdef cncvis.NCVis* c_ncvis
    cdef size_t d

    def __cinit__(self, size_t d, size_t n_threads, size_t n_neighbors, size_t M, size_t ef_construction, size_t random_seed, int n_epochs, int n_init_epochs, float spread, float min_dist, float alpha, float alpha_Q, object n_noise):
        a, b = find_ab_params(spread, min_dist)
        cdef size_t[:] n_noise_arr
        if isinstance(n_noise, int):
            n_noise_arr = np.full(n_epochs, n_noise, dtype=np.uint)
        elif isinstance(n_noise, np.ndarray):
            if len(n_noise.shape) > 1:
                raise ValueError("Expected 1D n_noise array.")
            n_epochs = n_noise.shape[0]
            n_noise_arr = n_noise.astype(np.uint)
        self.c_ncvis = new cncvis.NCVis(d, n_threads, n_neighbors, M, ef_construction, random_seed, n_epochs, n_init_epochs, a, b, alpha, alpha_Q, &n_noise_arr[0])
        self.d = d

    def __dealloc__(self):
        del self.c_ncvis

    def fit(self, float[:, :] X):
        return np.asarray(<float[:X.shape[0], :self.d]>self.c_ncvis.fit(&X[0, 0], X.shape[0], X.shape[1]))

class NCVis:
    def __init__(self, d=2, n_threads=-1, n_neighbors=15, M=16, ef_construction=200, random_seed=42, n_epochs=30, n_init_epochs=10, spread=1., min_dist=0.5, alpha=1., alpha_Q=1., n_noise=None):
    """
    Creates new NCVis instance.

    Parameters
    ----------
    d : int
        Desired dimensionality of the embedding.
    n_threads : int
        The maximum number of threads to use. In case n_threads < 1, it defaults to the number of available CPUs.
    n_neighbors : int
        Number of nearest neighbours in the high dimensional space to consider.
    M : int
        The number of bi-directional links created for every new element during construction of HNSW.
        See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
    ef_construction : int
        The size of the dynamic list for the nearest neighbors (used during the search) in HNSW.
        See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
    random_seed : int
        Random seed to initialize the generators. Notice, however, that the result may still depend on the number of threads.
    n_epochs : int
        The total number of epochs to run. During one epoch the positions of each nearest neighbors pair are updated.
    n_init_epochs : int
        The number of epochs used for initialization. During one epoch the positions of each nearest neighbors pair are updated.
    spread : float
        The effective scale of embedded points. In combination with ``min_dist``
        this determines how clustered/clumped the embedded points are.
        See https://github.com/lmcinnes/umap/blob/834184f9c0455f26db13ab148c0abd2d3767d968/umap/umap_.py#L1143
    min_dist : float
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.
        See https://github.com/lmcinnes/umap/blob/834184f9c0455f26db13ab148c0abd2d3767d968/umap/umap_.py#L1135
    alpha : float
        Learning rate for the embedding positions.
    alpha_Q : float
        Learning rate for the normalization constant.
    n_noise : int or ndarray of ints
        Number of noise samples to use per data sample. If ndarray is provided, n_epochs is set to its length. If n_noise is None, it is set to dynamic sampling with noise level gradually increasing from 0 to fixed value. 
    """
        if n_noise is None:
            n_negative = 5

            negative_plan = np.linspace(0, 1, n_epochs)
            negative_plan = negative_plan**3

            negative_plan /= negative_plan.sum()
            negative_plan *= n_epochs*n_negative
            negative_plan = negative_plan.round().astype(np.int)
            negative_plan[negative_plan < 1] = 1

        if n_threads < 1:
            n_threads = cpu_count()

        self.model = NCVisWrapper(d, n_threads, n_neighbors, M, ef_construction, random_seed, n_epochs, n_init_epochs, spread, min_dist, alpha, alpha_Q, negative_plan)

    def fit(self, X):
        """
        Builds an embedding for given points.

        Parameters
        ----------
        X : ndarray of size [n_samples, n_high_dimensions]
            The data samples. Will be converted to float by default.

        Returns:
        --------
        out : ndarray of floats of size [n_samples, m_low_dimensions]
            The embedding of the data samples.
        """
        return self.model.fit(X.astype(np.float32))