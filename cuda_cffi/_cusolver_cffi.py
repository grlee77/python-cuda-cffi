"""
Autogenerate Python interface to cuSOLVER functions.

"""
from __future__ import absolute_import, print_function

import re
from os.path import join as pjoin

import numpy as np  # don't remove!  is used during call to exec() below

import cuda_cffi
from cuda_cffi._cffi_autogen_common import wrap_library
from cuda_cffi._cusolver_cffi_autogen import (
    generate_cffi_cdef,
    ffi_init_cusolver,
    build_func_body,
    generate_func_descriptions_json)

base_dir = cuda_cffi.__path__[0]
python_wrapper_file = pjoin(base_dir, '_cusolver_python.py')

try:
    from cuda_cffi._cusolver_ffi import ffi
    from cuda_cffi._cusolver_ffi import lib as ffi_lib
except ImportError:
    print("BUILDING CUSOLVER INTERFACE")

    """Call wrap_library to wrap cuSOLVER.  This should only be slow the first
    it is called.  After that the already compiled wrappers should be found."""
    ffi, ffi_lib = wrap_library(
        lib_name='_cusolver_ffi',
        cffi_file=pjoin(base_dir, '_cusolver.cffi'),
        python_wrapper_file=python_wrapper_file,
        build_body_func=build_func_body,
        ffi_init_func=ffi_init_cusolver,
        cdef_generator_func=generate_cffi_cdef,
        variable_defs_json=pjoin(base_dir,
                                 'cusolver_variable_descriptions.json'),
        func_defs_json=pjoin(base_dir, 'cusolver_func_descriptions.json'),
        func_description_generator_func=generate_func_descriptions_json,
        force_update=False,
        verbose=True)
    from cuda_cffi._cusolver_ffi import ffi
    from cuda_cffi._cusolver_ffi import lib as ffi_lib

class CUSOLVER_ERROR(Exception):
    """CUSOLVER error"""
    pass

# Use CUSOLVER_STATUS* definitions to dynamically create corresponding
# exception classes and populate dictionary used to raise appropriate
# exception in response to the corresponding CUSOLVER error code:
CUSOLVER_STATUS_SUCCESS = ffi_lib.CUSOLVER_STATUS_SUCCESS
CUSOLVER_EXCEPTIONS = {-1: CUSOLVER_ERROR}
for k, v in ffi_lib.__dict__.items():
    # Skip CUSOLVER_STATUS_SUCCESS:
    if re.match('CUSOLVER_STATUS.*', k) and v != CUSOLVER_STATUS_SUCCESS:
        CUSOLVER_EXCEPTIONS[v] = vars()[k] = type(k, (CUSOLVER_ERROR,), {})


# Import various other enum values into module namespace:
regex = 'CUSOLVER(?!STATUS).*'
for k, v in ffi_lib.__dict__.items():
    if re.match(regex, k):
        # print("k={}, v={}".format(k,v))
        vars()[k] = v
regex = 'CUBLAS(?!STATUS).*'
for k, v in ffi_lib.__dict__.items():
    if re.match(regex, k):
        # print("k={}, v={}".format(k,v))
        vars()[k] = v

# enums from libraryPropertyType (added for CUDA 8.0)
if 'MAJOR_VERSION' in ffi_lib.__dict__:
    vars()['MAJOR_VERSION'] = ffi_lib.__dict__['MAJOR_VERSION']
if 'MINOR_VERSION' in ffi_lib.__dict__:
    vars()['MINOR_VERSION'] = ffi_lib.__dict__['MINOR_VERSION']
if 'PATCH_LEVEL' in ffi_lib.__dict__:
    vars()['PATCH_LEVEL'] = ffi_lib.__dict__['PATCH_LEVEL']

# match enums corresponding to cudaDataType (added for CUDA 8.0)
regex = 'CUDA_[RC]_.'
for k, v in ffi_lib.__dict__.items():
    if re.match(regex, k):
        # print("k={}, v={}".format(k,v))
        vars()[k] = v


def cusolverCheckStatus(status):
    """
    Raise CUSOLVER exception

    Raise an exception corresponding to the specified CUSOLVER error
    code.

    Parameters
    ----------
    status : int
        CUSOLVER error code.

    See Also
    --------
    CUSOLVER_EXCEPTIONS
    """

    if status != 0:
        try:
            raise CUSOLVER_EXCEPTIONS[status]
        except KeyError:
            raise CUSOLVER_ERROR

# execute the python wrapper code
with open(python_wrapper_file) as f:
    code = compile(f.read(), python_wrapper_file, 'exec')
    exec(code)


__all__ = [k for k, v in ffi_lib.__dict__.items()]
__all__.remove('cusolverRfBatchResetValues') # TODO: fix (pointer to array)
__all__.remove('cusolverRfBatchSetupHost')   # TODO: fix (pointer to array)
__all__.remove('cusolverRfBatchSolve')       # TODO: fix (pointer to array)
__all__.remove('cusolverDnSpotrfBatched')       # TODO: fix (pointer to array)
__all__.remove('cusolverDnDpotrfBatched')       # TODO: fix (pointer to array)
__all__.remove('cusolverDnCpotrfBatched')       # TODO: fix (pointer to array)
__all__.remove('cusolverDnZpotrfBatched')       # TODO: fix (pointer to array)
__all__.remove('cusolverDnSpotrsBatched')       # TODO: fix (pointer to array)
__all__.remove('cusolverDnDpotrsBatched')       # TODO: fix (pointer to array)
__all__.remove('cusolverDnCpotrsBatched')       # TODO: fix (pointer to array)
__all__.remove('cusolverDnZpotrsBatched')       # TODO: fix (pointer to array)
__all__.append('CUSOLVER_ERROR')
__all__.append('CUSOLVER_EXCEPTIONS')
__all__.append('cusolverCheckStatus')
__all__.append('ffi')
__all__.append('ffi_lib')

#__all__.remove('cusolverDnSsytrd_bufferSize')
#__all__.remove('cusolverDnCheevd')
#__all__.remove('cusolverDnSsyevd')