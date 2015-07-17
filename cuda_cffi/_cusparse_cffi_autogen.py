"""
Support functions for autogenerating Python interface to the cuSPARSE library.

"""

"""
Developed on linux for CUDA v6.5, but should support any version >3.1 where
cusparse.h or cusparse_v2.h can be found in CUDA_ROOT/include.

Set the environment variable CUDA_ROOT to the base of your CUDA installation

Note from the NVIDIA CUSPARSE release notes:

"The csr2csc() and bsr2bsc() routines contain a bug in the CUDA 6.0 and 6.5
releases. As a consequence, csrsv(), csrsv2(), csrsm(), bsrsv2(), bsrsm2(),
and csrgemm() may produce incorrect results when working with transpose
(CUSPARSE_OPERATION_TRANSPOSE) or conjugate-transpose
(CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE) operations. These routines work
correctly when the non-transpose (CUSPARSE_OPERATION_NON_TRANSPOSE) operation
is selected. The bug has been fixed in the CUDA 7.0 release."
"""

import os
import json

from ._cffi_autogen_common import (
    get_variable_descriptions, get_function_descriptions, CUDA_ROOT, ffi_init,
    cuda_include_path, reindent, split_line, generate_cffi_python_wrappers)


# on versions where cusparse_v2.h exists, use it
cusparse_header = os.path.join(cuda_include_path, 'cusparse_v2.h')
if not os.path.exists(cusparse_header):
    # on old versions there was only cusparse.h
    cusparse_header = os.path.join(cuda_include_path, 'cusparse.h')
    if not os.path.exists(cusparse_header):
        raise IOError("cusparse header file not found in expected "
                      "location.  Try defining CUDA_ROOT")


def ffi_init_cusparse(cffi_cdef):
    return ffi_init(cffi_cdef, headers=[cusparse_header],
                    libraries=['cusparse'])


def generate_cffi_cdef(
        cuda_include_path=cuda_include_path, cusparse_header=cusparse_header,
        cffi_out_file=None):
    """ generate the CUSPARSE FFI definition

    Parameters
    ----------
    cuda_include_path : str
        CUDA include path
    cffi_out_file : str, optional
        if provided, write the definition out to a file

    Returns
    -------
    cffi_cdef : str
        function definitions for use with cffi.  e.g. input to FFI.verify()

    """

    with open(cusparse_header, 'r') as f:
        cusparse_hdr = f.readlines()

    # in some version cusparse_v2.h just points to cusparse.h, so read it
    # instead
    for line in cusparse_hdr:
        # if v2 header includes cusparse.h, read that one instead
        if line.startswith('#include "cusparse.h"'):
            cusparse_header = os.path.join(cuda_include_path, 'cusparse.h')
            with open(cusparse_header, 'r') as f:
                cusparse_hdr = f.readlines()

    # skip lines leading up to first typedef
    for idx, line in enumerate(cusparse_hdr):
        if line.startswith('typedef'):
            start_line = idx
            break

    # skip closing #if defined logic
    for idx, line in enumerate(cusparse_hdr[start_line:]):
        if line.startswith('#if defined(__cplusplus)') or \
           'Define the following symbols for the new API' in line:
            # second match is to avoid CFFI compilation errror due to the final
            # define statements in v4.1 through v5.5
            end_line = start_line + idx
            break

    # define other data types needed by FFI
    # ... will be filled in from cuComplex.h by the C compiler
    cffi_cdef = """
    typedef struct CUstream_st *cudaStream_t;

    typedef struct float2 {
        ...;
    } float2;
    typedef float2 cuFloatComplex;
    typedef float2 cuComplex;

    typedef struct double2 {
        ...;
    } double2;
    typedef double2 cuDoubleComplex;

    typedef float cufftReal;
    typedef double cufftDoubleReal;

    typedef cuComplex cufftComplex;
    typedef cuDoubleComplex cufftDoubleComplex;

    /* definitions from cusparse header below this point */
    """

    cffi_cdef += ''.join(cusparse_hdr[start_line:end_line])

    """
    don't use the _v2 versions of the function names defined in CUDA v4.1
    through v5.5
    """
    cffi_cdef = cffi_cdef.replace('_v2(', '(')

    if os.name == 'nt':  # Win
        cffi_cdef = cffi_cdef.replace('CUSPARSEAPI', '__stdcall')
    else:  # posix, etc
        cffi_cdef = cffi_cdef.replace('CUSPARSEAPI', '')

    if cffi_out_file is not None:
        # create specified output directory if it doesn't already exist
        out_dir = os.path.dirname(cffi_out_file)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(cffi_out_file, 'w') as f:
            f.write(cffi_cdef)

    return cffi_cdef


def build_func_body(func_name, arg_dict, return_type):
    """ generate python_wrapper function body

    Note: this code is highly specific to the particulars of the cuSPARSE
    library

    """
    body = ""
    arg_list = ""

    # the following are pointers to scalar outputs
    # Note: pBufferSize was renamed pBufferSizeInBytes in v6.5
    scalar_ptr_outputs = ['nnzTotalDevHostPtr',
                          'pBufferSize',
                          'pBufferSizeInBytes',
                          'resultDevHostPtr']

    is_creator = 'cusparseCreate' in func_name
    is_getter = 'cusparseGet' in func_name

    if return_type == 'cusparseStatus_t' and not (is_creator or is_getter):
        is_return = False
    else:
        is_return = True

    # else:
    return_str = ''
    for k, v in arg_dict.items():

        """
        set some flags based on the name/type of the argument
        will use these flags to determine whether and how to call ffi.new or
        ffi.cast on each variable
        """
        is_ptr = '*' in v
        is_cusparse_type = '_t' in v
        is_cusparse_ptr = is_ptr and is_cusparse_type
        is_output_scalar = k in scalar_ptr_outputs
        if k in ['alpha', 'beta']:
            is_scalar = True
        else:
            is_scalar = False
        if is_getter:
            is_gpu_array = False
        else:
            is_gpu_array = is_ptr and (not is_cusparse_ptr) and (not is_scalar)
        if 'Complex' in v:
            is_complex = True
        else:
            is_complex = False

        # convert variable to appropriate type for the FFI
        if is_output_scalar:
            # for scalar outputs make a new pointer
            body += "%s = ffi.cast('%s', %s)\n" % (k, v, k)
        elif is_getter and is_ptr and (return_type == 'cusparseStatus_t'):
            # any pointers in cusparseGet* are new outputs to be created
            body += "%s = ffi.new('%s')\n" % (k, v)
        elif is_gpu_array:
            # pass pointer to GPU array data (use either .ptr or .gpudata)
            body += "%s = ffi.cast('%s', %s.ptr)\n" % (k, v, k)
        elif is_cusparse_ptr:
            if is_creator:
                # generate custom cusparse type
                body += "%s = ffi.new('%s')\n" % (k, v)
            else:
                # cast to the custom cusparse type
                body += "%s = ffi.cast('%s', %s)\n" % (k, v, k)
        elif is_ptr and is_scalar:
            # create new pointer, with value initialized to scalar
            if is_complex:
                # complex case is a bit tricky.  requires ffi.buffer
                body += "%sffi = ffi.new('%s')\n" % (k, v)
                if 'cusparseC' in func_name:
                    body += "ffi.buffer(%sffi)[:] = \
                        np.complex64(%s).tostring()\n" % (k, k)
                elif 'cusparseZ' in func_name:
                    body += "ffi.buffer(%sffi)[:] = \
                        np.complex128(%s).tostring()\n" % (k, k)
            else:
                body += "%s = ffi.new('%s', %s)\n" % (k, v, k)
        elif is_ptr or v == 'cudaStream_t':
            # case non-scalar pointer to appropriate type
            body += "%s = ffi.cast('%s', %s)\n" % (k, v, k)
        else:
            # don't need explicit cast for plain int, float, etc
            pass

        # build the list of arguments to pass to the API
        if is_ptr and is_scalar and is_complex:
            # take into account modified argument name for complex scalars
            arg_list += "%sffi, " % k
        else:
            arg_list += "%s, " % k

    # add the function call and optionally return the result
    last_key = k
    arg_list = arg_list[:-2]  # remove trailing ", "
    if is_getter and return_type != 'cusparseStatus_t':
        body += "return ffi_lib.%s(%s)\n" % (func_name, arg_list)
    else:
        # check cusparseStatus_t state before returning
        call_str = "status = ffi_lib.%s(%s)\n" % (func_name, arg_list)
        body += split_line(call_str, break_pattern=', ', nmax=76)
        body += "cusparseCheckStatus(status)\n"
        if is_return:
            # len(arg_dict) == 2) is to avoid return for cusparseGetLevelInfo
            if is_creator or (is_getter and (len(arg_dict) == 2)):
                body += "return %s[0]\n" % last_key
            else:
                body += "#TODO: return the appropriate result"
    body += '\n\n'
    return reindent(body, numSpaces=4, lstrip=False)


def generate_func_descriptions_json(ffi_lib, json_file):
    func_descriptions = {}
    for t in ['S', 'D', 'C', 'Z']:
        # functions introduced in CUDA 3.2
        func_descriptions['cusparse' + t + 'axpyi'] = 'scalar multiply and add: y = y + alpha * x'
        func_descriptions['cusparse' + t + 'coo2csr'] = 'convert sparse matrix formats: COO to CSR'
        func_descriptions['cusparse' + t + 'csc2dense'] = 'convert sparse matrix formats: CSC to dense'
        func_descriptions['cusparse' + t + 'csr2coo'] = 'convert sparse matrix formats: CSR to COO'
        func_descriptions['cusparse' + t + 'csr2csc'] = 'convert sparse matrix formats: CSR to CSC'
        func_descriptions['cusparse' + t + 'csr2dense'] = 'convert sparse matrix formats: CSR to dense'
        func_descriptions['cusparse' + t + 'csrmv'] = 'sparse CSR matrix vector product:  y = alpha * op(A)*x + beta * y'
        func_descriptions['cusparse' + t + 'csrmm'] = 'sparse CSR matrix-matrix product:  C = alpha * op(A) * B + beta * C'
        func_descriptions['cusparse' + t + 'dense2csc'] = 'convert sparse matrix formats: dense to CSC'
        func_descriptions['cusparse' + t + 'dense2csr'] = 'convert sparse matrix formats: dense to CSR'
        func_descriptions['cusparse' + t + 'doti'] = 'dot product: result = y.T * x'
        func_descriptions['cusparse' + t + 'dotci'] = 'complex conjugate dot product: result = y.H * x'
        func_descriptions['cusparse' + t + 'gthr'] = 'gather elements of y at locations xInd into data array xVal'
        func_descriptions['cusparse' + t + 'gthrz'] = 'gather elements of y at locations xInd into data array xVal.  Also zeros the gathered elements in y'
        func_descriptions['cusparse' + t + 'nnz'] = 'compute number of nonzero elements per row or column and the total number of nonzero elements'
        func_descriptions['cusparse' + t + 'roti'] = 'applies Givens rotation matrix to sparse x and dense y'
        func_descriptions['cusparse' + t + 'sctr'] = 'scatters elements of vector x into the vector y (at locations xInd)'
        # for CUDA 4.0+ below here
        func_descriptions['cusparse' + t + 'csrsv_analysis'] = 'perform analysis phase of csrsv'
        func_descriptions['cusparse' + t + 'csrsv_solve'] = 'perform solve phase of csrsv'
        # for CUDA 4.1+ below here
        func_descriptions['cusparse' + t + 'csr2hyb'] = 'convert sparse matrix formats: CSR to HYB'
        func_descriptions['cusparse' + t + 'csrsm_analysis'] = 'perform analysis phase of csrsm'
        func_descriptions['cusparse' + t + 'csrsm_solve'] = 'perform solve phase of csrsm'
        func_descriptions['cusparse' + t + 'dense2hyb'] = 'convert sparse matrix formats: dense to HYB'
        func_descriptions['cusparse' + t + 'gtsv'] = 'solve tridiagonal system with multiple right-hand sides: A * Y = alpha * X'
        func_descriptions['cusparse' + t + 'gtsvStridedBatch'] = 'solve multiple tridiagonal systems for i = 0, ..., batchCount: A_i * y_i = alpha * x_i'
        func_descriptions['cusparse' + t + 'hyb2dense'] = 'convert sparse matrix formats: HYB to dense'
        func_descriptions['cusparse' + t + 'hybmv'] = 'sparse HYB matrix vector product:  y = alpha * op(A)*x + beta * y'
        func_descriptions['cusparse' + t + 'hybsv_analysis'] = 'perform analysis phase of hybsv'
        func_descriptions['cusparse' + t + 'hybsv_solve'] = 'perform solve phase of hybsv'
        # for CUDA 5.0+ below here
        func_descriptions['cusparse' + t + 'bsr2csr'] = 'convert sparse matrix formats: BSR to CSR'
        func_descriptions['cusparse' + t + 'bsrmv'] = 'sparse BSR matrix vector product:  y = alpha * op(A)*x + beta * y'
        func_descriptions['cusparse' + t + 'bsrxmv'] = 'sparse BSRX matrix vector product:  y = alpha * op(A)*x + beta * y'
        func_descriptions['cusparse' + t + 'csr2bsr'] = 'convert sparse matrix formats: CSR to BSR'
        func_descriptions['cusparse' + t + 'csrgeam'] = 'sparse CSR matrix-matrix operation:  C = alpha * A + beta * B'
        func_descriptions['cusparse' + t + 'csrgemm'] = 'sparse CSR matrix-matrix operation:  C = op(A) * op(B)'
        func_descriptions['cusparse' + t + 'csric0'] = 'CSR incomplete-Cholesky factorization:  op(A) ~= R.T * R'
        func_descriptions['cusparse' + t + 'csrilu0'] = 'CSR incomplete-LU factorization:  op(A) ~= LU'
        func_descriptions['cusparse' + t + 'hyb2csr'] = 'convert sparse matrix formats: HYB to CSR'
        # for CUDA 5.5+ below here
        func_descriptions['cusparse' + t + 'csc2hyb'] = 'convert sparse matrix formats: CSC to HYB'
        func_descriptions['cusparse' + t + 'csrmm2'] = 'sparse CSR matrix-matrix product type 2:  C = alpha * op(A) * op(B) + beta * C'
        func_descriptions['cusparse' + t + 'gtsv_nopivot'] = 'solve tridiagonal system with multiple right-hand sides: A * Y = alpha * X'
        func_descriptions['cusparse' + t + 'hyb2csc'] = 'convert sparse matrix formats: HYB to CSC'
        # for CUDA 6.0+ below here
        func_descriptions['cusparse' + t + 'bsric02_bufferSize'] = 'return bsric02 (A ~= L * L.H) buffer size'
        func_descriptions['cusparse' + t + 'bsric02_analysis'] = 'perform bsric02 (A ~= L * L.H) analysis phase'
        func_descriptions['cusparse' + t + 'bsric02'] = 'perform bsric02 (A ~= L * L.H) solve phase'
        func_descriptions['cusparse' + t + 'bsrilu02'] = 'perform bsrilu02 (A ~= LU) solve phase'
        func_descriptions['cusparse' + t + 'bsrilu02_analysis'] = 'perform bsrilu02 (A ~= LU) analysis phase'
        func_descriptions['cusparse' + t + 'bsrilu02_bufferSize'] = 'return bsrilu02 (A ~= LU) buffer size'
        func_descriptions['cusparse' + t + 'bsrilu02_numericBoost'] = 'use a boost value to replace a numerical value in incomplete LU factorization'
        func_descriptions['cusparse' + t + 'bsrsv2_analysis'] = 'perform analysis phase of bsrsv2'
        func_descriptions['cusparse' + t + 'bsrsv2_bufferSize'] = 'return size of buffer used in bsrsv2'
        func_descriptions['cusparse' + t + 'bsrsv2_solve'] = 'perform solve phase of bsrsv2'
        func_descriptions['cusparse' + t + 'csr2gebsr'] = 'convert sparse matrix formats: CSR to GEBSR'
        func_descriptions['cusparse' + t + 'csr2gebsr_bufferSize'] = 'return csr2gebsr buffer size'
        func_descriptions['cusparse' + t + 'csrsv2_analysis'] = 'perform analysis phase of csrsv2'
        func_descriptions['cusparse' + t + 'csrsv2_bufferSize'] = 'return size of buffer used in csrsv2'
        func_descriptions['cusparse' + t + 'csrsv2_solve'] = 'perform solve phase of csrsv2'
        func_descriptions['cusparse' + t + 'csric02_analysis'] = 'perform csric02 (A ~= L * L.H) analysis phase'
        func_descriptions['cusparse' + t + 'csric02_bufferSize'] = 'return csric02 (A ~= L * L.H) buffer size'
        func_descriptions['cusparse' + t + 'csric02'] = 'perform csric02 (A ~= L * L.H) solve phase'
        func_descriptions['cusparse' + t + 'csrilu02'] = 'perform csrilu02 (A ~= LU) solve phase'
        func_descriptions['cusparse' + t + 'csrilu02_analysis'] = 'perform csrilu02 (A ~= LU) analysis phase'
        func_descriptions['cusparse' + t + 'csrilu02_bufferSize'] = 'return csrilu02 (A ~= LU) buffer size'
        func_descriptions['cusparse' + t + 'csrilu02_numericBoost'] = 'use a boost value to replace a numerical value in incomplete LU factorization'
        func_descriptions['cusparse' + t + 'gebsr2csr'] = 'convert sparse matrix formats: GEBSR to CSR'
        func_descriptions['cusparse' + t + 'gebsr2gebsc'] = 'convert sparse matrix formats: GEBSR to GEBSC'
        func_descriptions['cusparse' + t + 'gebsr2gebsc_bufferSize'] = 'return gebsr2gebsc buffer size'
        func_descriptions['cusparse' + t + 'gebsr2gebsr'] = 'convert sparse matrix formats: GEBSR to GEBSR'
        func_descriptions['cusparse' + t + 'gebsr2gebsr_bufferSize'] = 'return gebsr2gebsr or gebsr2gebsrNnz buffer size'
        # for CUDA 6.5+ below here
        func_descriptions['cusparse' + t + 'bsrmm'] = 'sparse BSR matrix-matrix product:  C = alpha * op(A) * B + beta * C'
        func_descriptions['cusparse' + t + 'bsrsm2_analysis'] = 'perform analysis phase of bsrsm2'
        func_descriptions['cusparse' + t + 'bsrsm2_bufferSize'] = 'return size of buffer used in bsrsm2'
        func_descriptions['cusparse' + t + 'bsrsm2_solve'] = 'perform solve phase of bsrsm2'
        # for CUDA 7.0+ below here
        func_descriptions['cusparse' + t + 'coosort_bufferSizeExt'] = 'determine buffer size for coosort'
        func_descriptions['cusparse' + t + 'coosortByColumn'] = 'in-place sort of COO format by column'
        func_descriptions['cusparse' + t + 'coosortByRow'] = 'in-place sort of COO format by row'
        func_descriptions['cusparse' + t + 'cscsort'] = 'in-place sort of CSC'
        func_descriptions['cusparse' + t + 'cscsort_bufferSizeExt'] = 'determine buffer size for cscsort'
        func_descriptions['cusparse' + t + 'csr2csru'] = 'convert sorted CSR to unsorted CSR'
        func_descriptions['cusparse' + t + 'csrcolor'] = 'performs the coloring of the adjacency graph associated with the matrix A stored in CSR format'
        func_descriptions['cusparse' + t + 'csrgemm2'] = 'generalization of the csrgemm. a matrix-matrix operation: C=alpha*A*B + beta*D'
        func_descriptions['cusparse' + t + 'csrgemm2_bufferSizeExt'] = 'returns the buffer size required by csrgemm2'
        func_descriptions['cusparse' + t + 'csrsort'] = 'in-place sort of CSR'
        func_descriptions['cusparse' + t + 'csrsort_bufferSizeExt'] = 'determine buffer size for csrsort'
        func_descriptions['cusparse' + t + 'csru2csr_bufferSizeExt'] = 'determine buffer size for csru2csr and csr2csru'
        func_descriptions['cusparse' + t + 'csru2csr'] = 'convert unsorted CSR to sorted CSR'
        # CUDA 6.0 functions renamed in CUDA 7.0+ below here
        func_descriptions['cusparse' + t + 'bsric02_bufferSizeExt'] = 'return bsric02 (A ~= L * L.H) buffer size'
        func_descriptions['cusparse' + t + 'bsrilu02_bufferSizeExt'] = 'return bsrilu02 (A ~= LU) buffer size'
        func_descriptions['cusparse' + t + 'bsrsm2_bufferSizeExt'] = 'return size of buffer used in bsrsm2'
        func_descriptions['cusparse' + t + 'bsrsv2_bufferSizeExt'] = 'return size of buffer used in bsrsv2'
        func_descriptions['cusparse' + t + 'csr2gebsr_bufferSizeExt'] = 'return csr2gebsr buffer size'
        func_descriptions['cusparse' + t + 'csric02_bufferSizeExt'] = 'return csric02 (A ~= L * L.H) buffer size'
        func_descriptions['cusparse' + t + 'csrilu02_bufferSizeExt'] = 'return csrilu02 (A ~= LU) buffer size'
        func_descriptions['cusparse' + t + 'csrsv2_bufferSizeExt'] = 'return size of buffer used in csrsv2'
        func_descriptions['cusparse' + t + 'gebsr2gebsc_bufferSizeExt'] = 'return gebsr2gebsc buffer size'
        func_descriptions['cusparse' + t + 'gebsr2gebsr_bufferSizeExt'] = 'return gebsr2gebsr or gebsr2gebsrNnz buffer size'


    # operations common across all precisions below here
    # for CUDA 5.0+
    func_descriptions['cusparseXcsr2bsrNnz'] = 'determine the number of nonzero block columns per block row for csr2bsr'
    func_descriptions['cusparseXcsrgemmNnz'] = 'determine csrRowPtrC and the total number of nonzero elements for gemm'
    func_descriptions['cusparseXcsrgeamNnz'] = 'determine csrRowPtrC and the total number of nonzero elements for geam'
    # for CUDA 6.0+
    func_descriptions['cusparseXbsrsv2_zeroPivot'] = 'return numerical zero location for bsrsv2'
    func_descriptions['cusparseXbsric02_zeroPivot'] = 'return numerical zero location for bsric02'
    func_descriptions['cusparseXbsrilu02_zeroPivot'] = 'return numerical zero location for bsrilu02'
    func_descriptions['cusparseXcsr2gebsrNnz'] = 'determine the number of nonzero block columns per block row for csr2gebsr'
    func_descriptions['cusparseXcsric02_zeroPivot'] = 'return numerical zero location for csric02'
    func_descriptions['cusparseXcsrilu02_zeroPivot'] = 'return numerical zero location for csrilu02'
    func_descriptions['cusparseXcsrsv2_zeroPivot'] = 'return numerical zero location for csrsv2'
    func_descriptions['cusparseXgebsr2gebsrNnz'] = 'determine the number of nonzero block columns per block row for gebsr2gebsr'
    # for CUDA 6.5+
    func_descriptions['cusparseXbsrsm2_zeroPivot'] = 'return numerical zero location for bsrsm2'
    # for CUDA 7.0+
    # func_descriptions['cusparseCreateIdentityPermutation'] = 'creates an identity map for use with coosort, csrsort, cscsort, csr2csc_indexOnly'
    func_descriptions['cusparseXcsrgemm2Nnz'] = 'determine csrRowPtrC and the number of non-zero elements for csrgemm2'
    # func_descriptions['csrgemm2Info'] = "Opaque structures holding sparse gemm information"
    # func_descriptions['csru2csrInfo'] = "Opaque structures holding sorting information"
    # func_descriptions['cusparseColorInfo'] = "Opaque structures holding coloring information"
    # func_descriptions['cusparseCreateCsru2csrInfo'] = "Opaque structures holding sorting information"
    # func_descriptions['cusparseDestroyCsru2csrInfo'] = "Opaque structures holding sorting information"


    create_funcs = [cdef for cdef in ffi_lib.__dict__ if 'Create' in cdef]
    for func in create_funcs:
        tmp, obj = func.split('Create')
        if obj:
            func_descriptions[func] = "Create cuSPARSE {} structure.".format(obj)
        else:
            func_descriptions[func] = "Create cuSPARSE context."
    destroy_funcs = [cdef for cdef in ffi_lib.__dict__ if 'Destroy' in cdef]
    for func in destroy_funcs:
        tmp, obj = func.split('Destroy')
        if obj:
            func_descriptions[func] = "Destroy cuSPARSE {} structure.".format(obj)
        else:
            func_descriptions[func] = "Destroy cuSPARSE context."
    get_funcs = [cdef for cdef in ffi_lib.__dict__ if 'Get' in cdef]
    for func in get_funcs:
        tmp, obj = func.split('Get')
        func_descriptions[func] = "Get cuSPARSE {}.".format(obj)

    set_funcs = [cdef for cdef in ffi_lib.__dict__ if 'Set' in cdef]
    for func in set_funcs:
        tmp, obj = func.split('Set')
        func_descriptions[func] = "Set cuSPARSE {}.".format(obj)

    # prune any of the above that aren't in ffi_lib:
    func_descriptions = dict(
        (k, v) for k, v in func_descriptions.items(
            ) if k in ffi_lib.__dict__)

    with open(json_file, 'w') as fid:
        json.dump(func_descriptions, fid, sort_keys=True, indent=4)
