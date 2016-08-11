from __future__ import division

from numpy.testing import run_module_suite

from cuda_cffi.cusolver import *

from cuda_cffi import cusolver

# import pycuda.autoinit

cusolver.init()


cusolver_real_dtypes = [np.float32, np.float64]
cusolver_complex_dtypes = [np.complex64, np.complex128]
cusolver_dtypes = cusolver_real_dtypes + cusolver_complex_dtypes


def test_Dn_create_destroy():
    handle = cusolverDnCreate()
    cusolverDnDestroy(handle)


def test_Sp_create_destroy():
    handle = cusolverSpCreate()
    cusolverSpDestroy(handle)


def test_Rf_create_destroy():
    handle = cusolverRfCreate()
    cusolverRfDestroy(handle)


if __name__ == "__main__":
    run_module_suite()
