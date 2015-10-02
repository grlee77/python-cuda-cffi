#!/usr/bin/env python

"""
Miscellaneous functions.  Similar implementation to scikit.cuda.misc
"""

from __future__ import absolute_import

import pycuda.driver as drv

global _global_cusparse_handle
_global_cusparse_handle = None
global _global_cusparse_allocator
_global_cusparse_allocator = None
def init(allocator=drv.mem_alloc):
    """
    Initialize libraries used by cuda_cffi.

    Initialize the cuSPARSE and cuSOLVER libraries used by high-level functions
    provided by cuda_cffi.

    Parameters
    ----------
    allocator : an allocator used internally by some of the high-level
        functions.

    Notes
    -----
    This function does not initialize PyCUDA; it uses whatever device
    and context were initialized in the current host thread.
    """

    # CUBLAS & CUSPARSE use whatever device is being used by the host thread:
    global _global_cusparse_handle, _global_cusparse_allocator

    if not _global_cusparse_handle:
        from . import cusparse  # nest to avoid requiring cublas e.g. for FFT
        _global_cusparse_handle = cusparse.cusparseCreate()
        # set so that scalar values are passed by reference on the host
        cusparse.cusparseSetPointerMode(_global_cusparse_handle,
                                        cusparse.CUSPARSE_POINTER_MODE_HOST)

    if _global_cusparse_allocator is None:
        _global_cusparse_allocator = allocator


def shutdown():
    """
    Shutdown libraries used by cuda_cffi.

    Shutdown the cuSPARSE libraries used by high-level functions provided by
    cuda_cffi

    Notes
    -----
    This function does not shutdown PyCUDA.
    """

    global _global_cusparse_handle
    if _global_cusparse_handle:
        from . import cusparse
        cusparse.cusparseDestroy(_global_cusparse_handle)
        _global_cusparse_handle = None


