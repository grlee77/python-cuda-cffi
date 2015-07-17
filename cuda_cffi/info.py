#!/usr/bin/env python

"""
cuda_cffi
=========

cuda_cffi provides python interfaces to a subset of functions defined in the
cuSPARSE and cuSOLVER libraries distributed as part of NVIDIA's CUDA
Programming Toolkit [1]. It is meant to complement the existing scikits.cuda
package [2] which wraps cuBLAS, CULA, etc.  This package uses PyCUDA [3]_ to
provide high-level functions comparable to those in the NumPy package [4]_.


Low-level modules
------------------

- cusparse       cuSPARSE functions
- cusolver       cuSOLVER functions
- misc           Miscellaneous support functions.

High-level modules
------------------

- cusparse       higher-level cuSPARSE CSR class


.. [1] http://www.nvidia.com/cuda
.. [2] http://scikits.appspot.com/cuda
.. [3] http://mathema.tician.de/software/pycuda/
.. [4] http://numpy.scipy.org/

"""
