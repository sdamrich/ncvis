cdef extern from "../src/ncvis.hpp" namespace "ncvis":
    cdef enum Distance:
        squared_L2,
        inner_product,
        cosine_similarity,
        correlation

from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.string cimport string


cdef extern from "../src/ncvis.hpp" namespace "ncvis":
    cdef cppclass NCVis:
        NCVis(long d, long n_threads, long n_neighbors, long M, long ef_construction, long random_seed, int n_epochs, int n_init_epochs, float a, float b, float alpha, float alpha_Q, long* n_noise, Distance dist, bint fix_Q, bint noise_in_ratio, float noise_in_ratio_val, bint learn_Q) except +
        map[string, vector[float]] fit_transform(const float *const X, long N, long D, float* Y)
        map[string, vector[float]] fit_transform_edges(const float *const X, long N, long D, float* Y, long* precomp_edges, long n_edges)
        bint precomp_init
