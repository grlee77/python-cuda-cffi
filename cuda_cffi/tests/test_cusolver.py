from __future__ import division

from cuda_cffi.cusolver import *

from cuda_cffi import cusolver
cusolver.init()


import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_almost_equal

from unittest import skipIf

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv

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
