# distutils: language = c++
# distutils: sources = knn.cxx

import numpy as np
cimport numpy as np
import cython

cdef extern from "knn_.h":
    void cpp_knn(const float* points, const size_t npts, const size_t dim, 
			const float* queries, const size_t nqueries,
			const size_t K, long* indices)

    void cpp_knn_omp(const float* points, const size_t npts, const size_t dim, 
                const float* queries, const size_t nqueries,
                const size_t K, long* indices)

    void cpp_knn_batch(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim,
                const float* queries, const size_t nqueries,
                const size_t K, long* batch_indices)

    void cpp_knn_batch_omp(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, 
                    const float* queries, const size_t nqueries,
                    const size_t K, long* batch_indices)

    void cpp_knn_batch_distance_pick(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, 
                    float* queries, const size_t nqueries,
                    const size_t K, long* batch_indices)
        
    void cpp_knn_batch_distance_pick_omp(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, 
				float* batch_queries, const size_t nqueries,
				const size_t K, long* batch_indices)

def knn(pts, queries, K, omp=False):

    # define shape parameters
    cdef int npts
    cdef int dim
    cdef int K_cpp
    cdef int nqueries

    # define tables
    cdef np.ndarray[np.float32_t, ndim=2] pts_cpp
    cdef np.ndarray[np.float32_t, ndim=2] queries_cpp
    cdef np.ndarray[np.int64_t, ndim=2] indices_cpp

    # set shape values
    npts = pts.shape[0]
    nqueries = queries.shape[0]
    dim = pts.shape[1]
    K_cpp = K

    # create indices tensor
    indices = np.zeros((queries.shape[0], K), dtype=np.int64)

    pts_cpp = np.ascontiguousarray(pts, dtype=np.float32)
    queries_cpp = np.ascontiguousarray(queries, dtype=np.float32)
    indices_cpp = indices

    # normal estimation
    if omp:
        cpp_knn_omp(<float*> pts_cpp.data, npts, dim, 
                <float*> queries_cpp.data, nqueries,
                K_cpp, <long*> indices_cpp.data)
    else:
        cpp_knn(<float*> pts_cpp.data, npts, dim,
                <float*> queries_cpp.data, nqueries,
                K_cpp, <long*> indices_cpp.data)

    return indices

def knn_batch(pts, queries, K, omp=False):

    # define shape parameters
    cdef int batch_size
    cdef int npts
    cdef int nqueries
    cdef int K_cpp
    cdef int dim

    # define tables
    cdef np.ndarray[np.float32_t, ndim=3] pts_cpp
    cdef np.ndarray[np.float32_t, ndim=3] queries_cpp
    cdef np.ndarray[np.int64_t, ndim=3] indices_cpp

    # set shape values
    batch_size = pts.shape[0]
    npts = pts.shape[1]
    dim = pts.shape[2]
    nqueries = queries.shape[1]
    K_cpp = K

    # create indices tensor
    indices = np.zeros((pts.shape[0], queries.shape[1], K), dtype=np.int64)

    pts_cpp = np.ascontiguousarray(pts, dtype=np.float32)
    queries_cpp = np.ascontiguousarray(queries, dtype=np.float32)
    indices_cpp = indices

    # normal estimation
    if omp:
        cpp_knn_batch_omp(<float*> pts_cpp.data, batch_size, npts, dim, 
                <float*> queries_cpp.data, nqueries,
                K_cpp, <long*> indices_cpp.data)
    else:
        cpp_knn_batch(<float*> pts_cpp.data, batch_size, npts, dim,
                <float*> queries_cpp.data, nqueries,
                K_cpp, <long*> indices_cpp.data)

    return indices

def knn_batch_distance_pick(pts, nqueries, K, omp=False):

    # define shape parameters
    cdef int batch_size
    cdef int npts
    cdef int nqueries_cpp
    cdef int K_cpp
    cdef int dim

    # define tables
    cdef np.ndarray[np.float32_t, ndim=3] pts_cpp
    cdef np.ndarray[np.float32_t, ndim=3] queries_cpp
    cdef np.ndarray[np.int64_t, ndim=3] indices_cpp

    # set shape values
    batch_size = pts.shape[0]
    npts = pts.shape[1]
    dim = pts.shape[2]
    nqueries_cpp = nqueries
    K_cpp = K

    # create indices tensor
    indices = np.zeros((pts.shape[0], nqueries, K), dtype=np.long)
    queries = np.zeros((pts.shape[0], nqueries, dim), dtype=np.float32)

    pts_cpp = np.ascontiguousarray(pts, dtype=np.float32)
    queries_cpp = np.ascontiguousarray(queries, dtype=np.float32)
    indices_cpp = indices

    if omp:
        cpp_knn_batch_distance_pick_omp(<float*> pts_cpp.data, batch_size, npts, dim,
            <float*> queries_cpp.data, nqueries,
            K_cpp, <long*> indices_cpp.data)
    else:
        cpp_knn_batch_distance_pick(<float*> pts_cpp.data, batch_size, npts, dim,
            <float*> queries_cpp.data, nqueries,
            K_cpp, <long*> indices_cpp.data)

    return indices, queries