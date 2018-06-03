#!python
#cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np
from numpy cimport ndarray as nd_arr
from cython.parallel cimport prange

# don't use np.sqrt - the sqrt function from the C standard library is much
# faster
from libc.math cimport sqrt


def distance_matrix(double[:, :] A):

    cdef:
        Py_ssize_t nrow = A.shape[0]
        Py_ssize_t ncol = A.shape[1]
        Py_ssize_t ii, jj, kk
        np.ndarray[np.float64_t, ndim=2] D = np.zeros((nrow, nrow), np.double)
        double tmpss, diff

    for ii in range(nrow):
        for jj in range(ii + 1, nrow):
            tmpss = 0
            for kk in range(ncol):
                diff = A[ii, kk] - A[jj, kk]
                tmpss += diff * diff
            tmpss = sqrt(tmpss)
            D[ii, jj] = tmpss
            D[jj, ii] = tmpss
    return D