from wrapper cimport cncvis
import numpy as np
cimport numpy as cnp
import ctypes
from multiprocessing import cpu_count
from libcpp cimport bool
from vis_utils.utils import NCE_loss_keops, KL_divergence, compute_normalization
import scipy.sparse


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
    cdef long d

    def __cinit__(self, long d, long n_threads, long n_neighbors, long M, long ef_construction, long random_seed, int n_epochs, int n_init_epochs, float a, float b, float alpha, float alpha_Q, object n_noise, cncvis.Distance distance, bool fix_Q, bool fix_noise):
        cdef long[:] n_noise_arr
        if isinstance(n_noise, int):
            n_noise_arr = np.full(n_epochs, n_noise, dtype=np.long)
        elif isinstance(n_noise, np.ndarray):
            if len(n_noise.shape) > 1:
                raise ValueError("Expected 1D n_noise array.")
            n_epochs = n_noise.shape[0]
            n_noise_arr = n_noise.astype(np.long)
        self.c_ncvis = new cncvis.NCVis(d, n_threads, n_neighbors, M, ef_construction, random_seed, n_epochs, n_init_epochs, a, b, alpha, alpha_Q, &n_noise_arr[0], distance, fix_Q, fix_noise)
        self.d = d

    def __dealloc__(self):
        del self.c_ncvis

    # no longer void
    # returns data from the iterations now



    def fit_transform(self, float[:, :] X, float[:, :] Y):
        aux_data_cpp = self.c_ncvis.fit_transform(&X[0, 0],
                                                  X.shape[0],
                                                  X.shape[1],
                                                  &Y[0, 0])

        qs_vector = aux_data_cpp[b"qs"]

        n_q = qs_vector.size()
        qs = []
        for i in range(n_q):
            qs.append(qs_vector[i])

        qs = np.array(qs)

        embds_vector = aux_data_cpp[b"embds"]
        embds = []
        n_embds = embds_vector.size()
        for i in range(n_embds):
           embds.append(embds_vector[i])

        embds = np.array(embds)



        edges_vector = aux_data_cpp[b"edges"]
        edges = []
        n_edges = edges_vector.size()
        for i in range(n_edges):
            edges.append(int(edges_vector[i]))

        edges = np.array(edges)

        aux_data = {"qs": qs, "embds": embds, "edges": edges}

        return aux_data


class NCVis:
    def __init__(self, d=2, n_threads=-1, n_neighbors=15, M=16, ef_construction=200, random_seed=42, n_epochs=50, n_init_epochs=20, spread=1., min_dist=0.4, a=None, b=None, alpha=1., alpha_Q=1., n_noise=None, distance="euclidean", fix_Q=False, fix_noise=False):
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
        a : (optional, default None)
            More specific parameters controlling the embedding. If None these values
            are set automatically as determined by ``min_dist`` and ``spread``.
            See https://github.com/lmcinnes/umap/blob/834184f9c0455f26db13ab148c0abd2d3767d968/umap/umap_.py#L1179
        b : (optional, default None)
            More specific parameters controlling the embedding. If None these values
            are set automatically as determined by ``min_dist`` and ``spread``.
            See https://github.com/lmcinnes/umap/blob/834184f9c0455f26db13ab148c0abd2d3767d968/umap/umap_.py#L1183
        alpha : float
            Learning rate for the embedding positions.
        alpha_Q : float
            Learning rate for the normalization constant.
        n_noise : int or ndarray of ints
            Number of noise samples to use per data sample. If ndarray is provided, n_epochs is set to its length. If n_noise is None, it is set to dynamic sampling with noise level gradually increasing from 0 to fixed value. 
        distance : str {'euclidean', 'cosine', 'correlation', 'inner_product'}
            Distance to use for nearest neighbors search.
        """
        self.d = d
        self.n_epochs = n_epochs
        self.random_seed = random_seed
        self.distance = distance
        self.fix_noise = fix_noise
        self.fix_Q = fix_Q
        self.n_noise = n_noise
        self.n_neighbors = n_neighbors

        if n_noise is None:
            n_negative = 5

            negative_plan = np.linspace(0, 1, n_epochs)
            negative_plan = negative_plan**3

            negative_plan /= negative_plan.sum()
            negative_plan *= n_epochs*n_negative
            negative_plan = negative_plan.round().astype(np.int)
            negative_plan[negative_plan < 1] = 1
        elif type(n_noise) is np.ndarray:
            if len(n_noise.shape) != 1:
                raise ValueError("n_noise should have exactly one dimension, but shape {} was passed".format(n_noise.shape))
            negative_plan = n_noise.astype(np.int)
            n_epochs = negative_plan.size
        elif type(n_noise) is int:
            if n_noise < 1:
                raise ValueError("n_noise should be at least 1, but {} was passed".format(n_noise))
            negative_plan = np.full(n_epochs, n_noise).astype(np.int)
        else:
            raise ValueError("n_noise has unsupported type")

        if n_threads < 1:
            n_threads = cpu_count()

        distances = {
            'euclidean': cncvis.squared_L2,
            'cosine': cncvis.cosine_similarity, 
            'correlation': cncvis.correlation,
            'inner_product': cncvis.inner_product 
        }
        if distance not in distances:
            raise ValueError(f"Unsupported distance, expected one of: {'euclidean', 'cosine', 'correlation', 'inner_product'}, but got {distance}")

        if (a is None) or (b is None):
            if (a is None) and (b is None):
                a, b = find_ab_params(spread, min_dist)
            else:
                raise ValueError(f'Expected (a, b) to be (float, float) or (None, None),con but got (a, b) = ({a}, {b})')
        self.a = a
        self.b = b
        self.model = NCVisWrapper(d, n_threads, n_neighbors, M, ef_construction, random_seed, n_epochs, n_init_epochs, a, b, alpha, alpha_Q, negative_plan, distances[distance], fix_Q, fix_noise)

    def fit_transform(self,
                      X,
                      log_norm=True,
                      log_nce=True,
                      log_nce_no_noise=True,
                      log_nce_norm=True,
                      log_kl=True,
                      log_embds=True):
        """
        Builds an embedding for given points.

        Parameters
        ----------
        X : ndarray of size [n_samples, n_high_dimensions]
            The data samples. Will be converted to float by default.


        Returns:
        --------
        Y : ndarray of floats of size [n_samples, m_low_dimensions]
            The embedding of the data samples.
        """

        Y = np.empty((X.shape[0], self.d), dtype=np.float32)
        aux_data = self.model.fit_transform(np.ascontiguousarray(X,
                                                                  dtype=np.float32),
                                             np.ascontiguousarray(Y,
                                                                  dtype=np.float32))
        self.embd = Y
        self.aux_data = aux_data

        # reshape logged embeddings and edges
        self.aux_data["embds"] = self.aux_data["embds"].reshape(self.n_epochs+1,
                                                                len(X),
                                                                self.d)

        self.aux_data["edges"] = self.aux_data["edges"].reshape(-1, 2)

        # log various parameters in self.aux_data
        self.aux_data["n_epochs"] = self.n_epochs
        self.aux_data["random_seed"] = self.random_seed
        self.aux_data["distance"] = self.distance
        self.aux_data["fix_noise"] = self.fix_noise
        self.aux_data["fix_Q"] = self.fix_Q
        self.aux_data["n_noise"] = self.n_noise
        self.aux_data["n_neighbors"] = self.n_neighbors

        if log_norm or log_nce_norm:
            norm = []
            for embd in self.aux_data["embds"]:
                norm.append(compute_normalization(embd, a=self.a, b=self.b).cpu().numpy())
            norm = np.array(norm)
            self.aux_data["normalization"] = norm.flatten()

        if log_nce:
            knn_graph = scipy.sparse.coo_matrix((np.ones(len(self.aux_data["edges"])),
                                                (self.aux_data["edges"][:, 0],
                                                 self.aux_data["edges"][:, 1])),
                                               shape=(len(X), len(X)))

            zs = np.exp(self.aux_data["qs"])

        if log_nce:
            assert isinstance(self.n_noise, int)
            nce_loss = []
            for embd, z in zip(self.aux_data["embds"], zs):
                nce_loss.append(NCE_loss_keops(knn_graph,
                                               embd,
                                               m=self.n_noise,
                                               Z=z,
                                               a=self.a,
                                               b=self.b))
            self.aux_data["nce_loss"] = np.array(nce_loss)

        if log_nce_no_noise:
            nce_loss_no_noise = []
            for embd, z in zip(self.aux_data["embds"], zs):
                nce_loss_no_noise.append(NCE_loss_keops(knn_graph,
                                                        embd,
                                                        m=5,
                                                        Z=z,
                                                        a=self.a,
                                                        b=self.b,
                                                        noise_log_arg=False))
            self.aux_data["nce_loss_no_noise"] = np.array(nce_loss_no_noise)

        if log_nce_norm:
            nce_loss_norm = []
            for embd, z in zip(self.aux_data["embds"], norm):
                nce_loss_norm.append(NCE_loss_keops(knn_graph,
                                               embd,
                                               m=self.n_noise,
                                               Z=z,
                                               a=self.a,
                                               b=self.b))
            self.aux_data["nce_loss_norm"] = np.array(nce_loss_norm)

        if log_kl:
            kl_div = []
            for embd in self.aux_data["embds"]:
                kl_div.append(KL_divergence(high_sim=knn_graph,
                                            embedding=embd,
                                            a=self.a,
                                            b=self.b,
                                            norm_over_pos=False))
            self.aux_data["kl_div"] = np.array(kl_div)

        if not log_embds:
            del self.aux_data["embds"]

        return Y

