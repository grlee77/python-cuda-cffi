#!/usr/bin/env python
from __future__ import division, print_function, absolute_import
import functools
import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv

from cuda_cffi.misc import init

toolkit_version = drv.get_version()

if toolkit_version < (7, 0, 0):
    raise ImportError("cuSOLVER not present prior to v7.0 of the CUDA toolkit")

"""
Python interface to cuSOLVER functions.

Note: You may need to set the environment variable CUDA_ROOT to the base of
your CUDA installation.
"""
# import low level cuSOLVER python wrappers and constants

try:
    from cuda_cffi._cusolver_cffi import *
except Exception as e:
    print(repr(e))
    estr = "autogeneration and import of cuSOLVER wrappers failed\n"
    estr += ("Try setting the CUDA_ROOT environment variable to the base of "
             "your CUDA installation.  The autogeneration script tries to "
             "find the CUSOLVER headers in CUDA_ROOT/include/\n")
    raise ImportError(estr)

# TODO: