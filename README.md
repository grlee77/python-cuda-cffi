python-cuda-cffi
================

This is an experimental package using CFFI to automatically wrap all C
functions in NVIDIA's cuSPARSE and cuSOLVER libraries.

cuSPARSE requires NVIDIA CUDA v3.2+

cuSOLVER requires NVIDIA CUDA v7.0+

This software is still at the experimental/alpha stage.  At this point CSR
routines from cuSPARSE are the only functions with higher-level python wrappers
and tests.

No official documentation exists yet, but there is a short example below and
more can be seen in the functions in tests/test_cusparse.py.

This routines provided here are intended to be complementary to those provided
in the scikit-cuda package:
https://github.com/lebedov/scikit-cuda

If you are interested in cuDNN, there is a set of wrappers for that here:
https://github.com/hannes-brt/cudnn-python-wrappers

requirements
------------
- CUDA v3.2+  (developed with CUDA v6.0-v7.0)
- pyCUDA
- numpy
- scipy

installation
------------
python setup.py install

After installation, the first time cuda\_cffi.cusparse or cuda_cffi.\_cusolver
are imported, CFFI will attempt to automatically generate all the low-level
wrappers.  This takes several seconds, but should only need to be done once
after installation.

usage
-----

```python
from cuda_cffi import cusparse
cusparse.init()
```

If the above succeeds, all wrappers should have been generated.  Low-level
functions can be accessed by the C function names

For example single-precision complex matrix-matrix multiplication corresponds
to ``cusparse.cusparseCcsrmm``.  There is a slightly more friendly python
wrapper for most of the CSR-based routines that will take any GPUarrays as
input and call the appropriate precision variant as needed.  In this case, the
slightly higher level python wrapper is ``cusparse.csrmm``.

Getting the input argument types to these low level wrappers correct can be
tricky, so Ideally there would be higher level python interfaces to all of
these to take care of that for you.

Currently there is only a higher level CSR class that simplifies use of many
of the sparse matrix CSR routines.  Here is a simple example of generating a
GPU CSR matrix object and performing CSR matrix-matrix multiplication on the
GPU.

```python
import numpy as np
from numpy.testing import assert_almost_equal
import scipy.sparse

import pycuda.autoinit
import pycuda.gpuarray as gpuarray

from cuda_cffi import cusparse
cusparse.init()

# An m x m sparse matrix will be multiplied by m x n dense matrix
m = 64
n = 32
A = 2*np.eye(m)
dtype = np.float32
A = A.astype(dtype)
A_scipy_csr = scipy.sparse.csr_matrix(A)

# generate a CSR matrix object on the GPU from the scipy CSR matrix
# Note: .to_SCR() can also take a dense numpy, GPUarray, or other scipy.sparse
#       matrix type as input.
A_CSR = cusparse.CSR.to_CSR(A_scipy_csr)
print(A_CSR)

# convert cusparseCSR back to a dense matrix
A_dense = A_CSR.todense(to_cpu=False)
assert type(A_dense) == gpuarray.GPUArray

# we can get the pyCUDA GPUarray back to the CPU in the usual way
A_dense_cpu = A_dense.get()
B_cpu = np.ones((m, n), dtype=dtype, order='F')
B = gpuarray.to_gpu(B_cpu.astype(dtype))


# The following is A * B computed on the GPU
C = A_CSR.mm(B)
assert_almost_equal(C.get(), np.dot(A, B_cpu))

# can also multiply A.T by B if transA is specified appropriately and so forth
C = A_CSR.mm(B_cpu.astype(dtype), transA=cusparse.CUSPARSE_OPERATION_TRANSPOSE)
assert_almost_equal(C, np.dot(A.T, B_cpu))

```
See tests/test_cusparse.py for other examples using either the CSR matrix or
the lower level wrappers.  The tests for the high level CSR object are all
names ``test_CSR_*``

The cuSOLVER wrappers are almost entirely untested at this point.