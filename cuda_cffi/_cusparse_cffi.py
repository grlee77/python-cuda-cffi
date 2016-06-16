"""
Autogenerate Python interface to cuSPARSE functions.

"""
from __future__ import absolute_import, print_function

import re
from os.path import join as pjoin

import numpy as np  # don't remove!  is used during call to exec() below

import cuda_cffi
from cuda_cffi._cffi_autogen_common import wrap_library
from cuda_cffi._cusparse_cffi_autogen import (
    generate_cffi_cdef,
    ffi_init_cusparse,
    build_func_body,
    generate_func_descriptions_json)

base_dir = cuda_cffi.__path__[0]
python_wrapper_file = pjoin(base_dir, '_cusparse_python.py')

try:
    from cuda_cffi._cusparse_ffi import ffi
    from cuda_cffi._cusparse_ffi import lib as ffi_lib
except:
    """ Call wrap_library to wrap cuSPARSE.  This should only be slow the first
    it is called.  After that the already compiled wrappers should be found. """
    ffi, ffi_lib = wrap_library(
        lib_name='_cusparse_ffi',
        cffi_file=pjoin(base_dir, '_cusparse.cffi'),
        python_wrapper_file=python_wrapper_file,
        build_body_func=build_func_body,
        ffi_init_func=ffi_init_cusparse,
        cdef_generator_func=generate_cffi_cdef,
        variable_defs_json=pjoin(base_dir, 'cusparse_variable_descriptions.json'),
        func_defs_json=pjoin(base_dir, 'cusparse_func_descriptions.json'),
        func_description_generator_func=generate_func_descriptions_json,
        force_update=False,
        verbose=True)
    from cuda_cffi._cusparse_ffi import ffi
    from cuda_cffi._cusparse_ffi import lib as ffi_lib

class CUSPARSE_ERROR(Exception):
    """CUSPARSE error"""
    pass

# Use CUSPARSE_STATUS* definitions to dynamically create corresponding
# exception classes and populate dictionary used to raise appropriate
# exception in response to the corresponding CUSPARSE error code:
CUSPARSE_STATUS_SUCCESS = ffi_lib.CUSPARSE_STATUS_SUCCESS
CUSPARSE_EXCEPTIONS = {-1: CUSPARSE_ERROR}
for k, v in ffi_lib.__dict__.items():
    # Skip CUSPARSE_STATUS_SUCCESS:
    if re.match('CUSPARSE_STATUS.*', k) and v != CUSPARSE_STATUS_SUCCESS:
        CUSPARSE_EXCEPTIONS[v] = vars()[k] = type(k, (CUSPARSE_ERROR,), {})


# Import various other enum values into module namespace:
regex = 'CUSPARSE_(?!STATUS).*'
for k, v in ffi_lib.__dict__.items():
    if re.match(regex, k):
        # print("k={}, v={}".format(k,v))
        vars()[k] = v


def cusparseCheckStatus(status):
    """
    Raise CUSPARSE exception

    Raise an exception corresponding to the specified CUSPARSE error
    code.

    Parameters
    ----------
    status : int
        CUSPARSE error code.

    See Also
    --------
    CUSPARSE_EXCEPTIONS
    """

    if status != 0:
        try:
            raise CUSPARSE_EXCEPTIONS[status]
        except KeyError:
            raise CUSPARSE_ERROR

# execute the python wrapper code
with open(python_wrapper_file) as f:
    code = compile(f.read(), python_wrapper_file, 'exec')
    exec(code)


__all__ = [k for k, v in ffi_lib.__dict__.items()]
__all__.append('CUSPARSE_ERROR')
__all__.append('CUSPARSE_EXCEPTIONS')
__all__.append('cusparseCheckStatus')
__all__.append('ffi')
__all__.append('ffi_lib')
