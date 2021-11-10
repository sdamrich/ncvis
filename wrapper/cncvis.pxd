cdef extern from "../src/ncvis.hpp" namespace "ncvis":
    cdef enum Distance:
        squared_L2,
        inner_product,
        cosine_similarity,
        correlation

from libcpp.vector cimport vector
from libcpp.pair cimport pair


cdef extern from "../src/ncvis.hpp" namespace "ncvis":
    cdef cppclass NCVis:
        NCVis(long d, long n_threads, long n_neighbors, long M, long ef_construction, long random_seed, int n_epochs, int n_init_epochs, float a, float b, float alpha, float alpha_Q, long* n_noise, Distance dist, fix_Q) except +
        pair[vector[float], vector[vector[float]]] fit_transform(const float *const X, long N, long D, float* Y)

