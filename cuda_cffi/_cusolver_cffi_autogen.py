"""
Support functions for autogenerating Python interface to the cuSOLVER library.

"""

"""
Developed on linux for CUDA v7.0, but should support any version where
cusolver.h can be found in CUDA_ROOT/include with minor modifications.

Set the environment variable CUDA_ROOT to the base of your CUDA installation

"""

import os
import json

from ._cffi_autogen_common import (
    get_variable_descriptions, get_function_descriptions, CUDA_ROOT, ffi_init,
    cuda_include_path, reindent, split_line, generate_cffi_python_wrappers)

cusolver_headers = [os.path.join(cuda_include_path, 'cusolver_common.h'),
                    os.path.join(cuda_include_path, 'cusolverDn.h'),
                    os.path.join(cuda_include_path, 'cusolverRf.h'),
                    os.path.join(cuda_include_path, 'cusolverSp.h')]
for hdr in cusolver_headers:
    if not os.path.exists(hdr):
        raise IOError("cusolver header files not found in expected location. "
                      "Try defining CUDA_ROOT.")


def ffi_init_cusolver(lib_name, cffi_cdef):
    return ffi_init(lib_name, cffi_cdef, headers=cusolver_headers,
                    libraries=['cusolver'])


def generate_cffi_cdef(
        cuda_include_path=cuda_include_path,
        cusolver_common_header=cusolver_headers[0],
        cusolverDn_header=cusolver_headers[1],
        cusolverSp_header=cusolver_headers[2],
        cusolverRf_header=cusolver_headers[3],
        cffi_out_file=None):
    """ generate the CUSOLVER FFI definition

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

    all_hdr = []
    for hdr in [cusolver_common_header, cusolverDn_header, cusolverSp_header,
                cusolverRf_header]:
        with open(hdr, 'r') as f:
            cusolver_hdr = f.readlines()

        # skip lines leading up to first typedef or struct
        for idx, line in enumerate(cusolver_hdr):
            if line.startswith('typedef') or line.startswith('struct'):
                start_line = idx
                break

        # skip closing #if defined logic
        for idx, line in enumerate(cusolver_hdr[start_line:]):
            if line.startswith('#endif') or \
               line.startswith('#if defined(__cplusplus)'):
                end_line = start_line + idx
                break

        all_hdr.append(cusolver_hdr[start_line:end_line])

    # define other data types needed by FFI
    # ... will be filled in from cuComplex.h, etc by the C compiler
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

    /* Declarations from cusparse.h or cusparse_v2.h */
    struct cusparseMatDescr;
    typedef struct cusparseMatDescr *cusparseMatDescr_t;

    /* Declarations from cublas_api.h */
    typedef enum {
        CUBLAS_FILL_MODE_LOWER=0,
        CUBLAS_FILL_MODE_UPPER=1
    } cublasFillMode_t;

    typedef enum {
        CUBLAS_SIDE_LEFT =0,
        CUBLAS_SIDE_RIGHT=1
    } cublasSideMode_t;

    typedef enum {
        CUBLAS_OP_N=0,
        CUBLAS_OP_T=1,
        CUBLAS_OP_C=2
    } cublasOperation_t;

    /* definitions from cusolver header below this point */
    """

    for hdr in all_hdr:
        cffi_cdef += ''.join(hdr)

    for api_string in ['CUSOLVERAPI', 'CUDENSEAPI', 'CRFWINAPI']:
        if os.name == 'nt':  # Win
            cffi_cdef = cffi_cdef.replace(api_string, '__stdcall')
        else:  # posix, etc
            cffi_cdef = cffi_cdef.replace(api_string, '')

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

    Note: this code is highly specific to the particulars of the cuSOLVER
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

    is_creator = 'Create' in func_name
    is_getter = 'Get' in func_name

    if return_type == 'cusolverStatus_t' and not (is_creator or is_getter):
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
        is_cusolver_type = '_t' in v
        is_cusolver_ptr = is_ptr and is_cusolver_type
        is_output_scalar = k in scalar_ptr_outputs
        if k in ['alpha', 'beta']:
            is_scalar = True
        else:
            is_scalar = False
        if is_getter:
            is_gpu_array = False
        else:
            is_gpu_array = is_ptr and (not is_cusolver_ptr) and (not is_scalar)
        if 'Complex' in v:
            is_complex = True
        else:
            is_complex = False

        # convert variable to appropriate type for the FFI
        if is_output_scalar:
            # for scalar outputs make a new pointer
            body += "%s = ffi.cast('%s', %s)\n" % (k, v, k)
        elif is_getter and is_ptr and (return_type == 'cusolverStatus_t'):
            # any pointers in cusolverGet* are new outputs to be created
            body += "%s = ffi.new('%s')\n" % (k, v)
        elif is_gpu_array:
            # pass pointer to GPU array data (use either .ptr or .gpudata)
            body += "%s = ffi.cast('%s', %s.ptr)\n" % (k, v, k)
        elif is_cusolver_ptr:
            if is_creator:
                # generate custom cusolver type
                body += "%s = ffi.new('%s')\n" % (k, v)
            else:
                # cast to the custom cusolver type
                body += "%s = ffi.cast('%s', %s)\n" % (k, v, k)
        elif is_ptr and is_scalar:
            # create new pointer, with value initialized to scalar
            if is_complex:
                # complex case is a bit tricky.  requires ffi.buffer
                body += "%sffi = ffi.new('%s')\n" % (k, v)
                if 'cusolverC' in func_name:
                    body += "ffi.buffer(%sffi)[:] = \
                        np.complex64(%s).tostring()\n" % (k, k)
                elif 'cusolverZ' in func_name:
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
    if is_getter and return_type != 'cusolverStatus_t':
        body += "return ffi_lib.%s(%s)\n" % (func_name, arg_list)
    else:
        # check cusolverStatus_t state before returning
        call_str = "status = ffi_lib.%s(%s)\n" % (func_name, arg_list)
        body += split_line(call_str, break_pattern=', ', nmax=76)
        body += "cusolverCheckStatus(status)\n"
        if is_return:
            # len(arg_dict) == 2) is to avoid return for cusolverGetLevelInfo
            if is_creator or (is_getter and (len(arg_dict) == 2)):
                body += "return %s[0]\n" % last_key
            else:
                body += "#TODO: return the appropriate result"
    body += '\n\n'
    return reindent(body, numSpaces=4, lstrip=False)


def generate_func_descriptions_json(ffi_lib, json_file):
    func_descriptions = {}
    for t in ['S', 'D']:  # real only
        # Dense linear solver routines
        func_descriptions['cusolverDn' + t + 'ormqr'] = 'overwrite mxn matrix C by either op(Q)*C  or C*op(Q)'
        # added in CUDA 8.0
        func_descriptions['cusolverDn' + t + 'ormqr_bufferSize'] = 'buffer size of ormqr'
        func_descriptions['cusolverDn' + t + 'orgqr'] = 'TODO'
        func_descriptions['cusolverDn' + t + 'orgqr_bufferSize'] = 'buffer size of orgqr'
        func_descriptions['cusolverDn' + t + 'orgbr'] = 'TODO'
        func_descriptions['cusolverDn' + t + 'orgbr_bufferSize'] = 'buffer size of orgbr'
        func_descriptions['cusolverDn' + t + 'orgtr'] = 'TODO'
        func_descriptions['cusolverDn' + t + 'orgtr_bufferSize'] = 'buffer size of orgtr'
        func_descriptions['cusolverDn' + t + 'ormtr'] = 'TODO'
        func_descriptions['cusolverDn' + t + 'ormtr_bufferSize'] = 'buffer size of ormtr'
        func_descriptions['cusolverDn' + t + 'sytrd'] = 'TODO'
        func_descriptions['cusolverDn' + t + 'sytrd_bufferSize'] = 'buffer size of sytrd'
        func_descriptions['cusolverDn' + t + 'syevd'] = 'TODO'
        func_descriptions['cusolverDn' + t + 'syevd_bufferSize'] = 'buffer size of syevd'
        func_descriptions['cusolverDn' + t + 'sygvd'] = 'TODO'
        func_descriptions['cusolverDn' + t + 'sygvd_bufferSize'] = 'buffer size of sygvd'

    for t in ['C', 'Z']:  # complex only
        # Dense linear solver routines
        func_descriptions['cusolverDn' + t + 'unmqr'] = 'overwrite mxn matrix C by either op(Q)*C  or C*op(Q)'

        # added in CUDA 8.0
        func_descriptions['cusolverDn' + t + 'unmqr_bufferSize'] = 'buffer size of ormqr'
        func_descriptions['cusolverDn' + t + 'ungqr'] = 'TODO'
        func_descriptions['cusolverDn' + t + 'ungqr_bufferSize'] = 'buffer size of ungqr'
        func_descriptions['cusolverDn' + t + 'ungbr'] = 'TODO'
        func_descriptions['cusolverDn' + t + 'ungbr_bufferSize'] = 'buffer size of ungbr'
        func_descriptions['cusolverDn' + t + 'ungtr'] = 'TODO'
        func_descriptions['cusolverDn' + t + 'ungtr_bufferSize'] = 'buffer size of ungtr'
        func_descriptions['cusolverDn' + t + 'unmtr'] = 'TODO'
        func_descriptions['cusolverDn' + t + 'unmtr_bufferSize'] = 'buffer size of unmtr'
        func_descriptions['cusolverDn' + t + 'hetrd'] = 'TODO'
        func_descriptions['cusolverDn' + t + 'hetrd_bufferSize'] = 'buffer size of hetrd'
        func_descriptions['cusolverDn' + t + 'heevd'] = 'TODO'
        func_descriptions['cusolverDn' + t + 'heevd_bufferSize'] = 'buffer size of heevd'
        func_descriptions['cusolverDn' + t + 'hegvd'] = 'TODO'
        func_descriptions['cusolverDn' + t + 'hegvd_bufferSize'] = 'buffer size of hegvd'

    for t in ['S', 'D', 'C', 'Z']:  # general routines
        # Dense linear solver routines
        func_descriptions['cusolverDn' + t + 'potrf'] = 'compute Cholesky factorization of a Hermitian positive-definite matrix'
        func_descriptions['cusolverDn' + t + 'potrf_bufferSize'] = 'buffer size for potrf'
        func_descriptions['cusolverDn' + t + 'potrs'] = 'solves a system of linear equations A*X = B where A is nxn Hermitian'
        func_descriptions['cusolverDn' + t + 'getrf'] = 'compute LU factorization of an mxn matrix. P*A = L*U'
        func_descriptions['cusolverDn' + t + 'getrf_bufferSize'] = 'buffer size for getrf'
        func_descriptions['cusolverDn' + t + 'geqrf'] = 'compute QR factorization of an mxn matrix. A = Q*R'
        func_descriptions['cusolverDn' + t + 'geqrf_bufferSize'] = 'buffer size for geqrf'
        func_descriptions['cusolverDn' + t + 'getrs'] = 'solves a linear system with multiple right hand sides:  op(A)*X = B'
        func_descriptions['cusolverDn' + t + 'sytrf'] = 'compute the Bunch-Kaufman factorization of an nxn symmetric indefinite matrix'
        func_descriptions['cusolverDn' + t + 'sytrf_bufferSize'] = 'buffer size for sytrf'
        # Dense eigenvalue routines
        func_descriptions['cusolverDn' + t + 'gebrd'] = 'reduce a general real matrix A to lower bidiagonal form B by an orthogonal transformation: Q^H*A*P=B'
        func_descriptions['cusolverDn' + t + 'gebrd_bufferSize'] = 'calculate buffer size for gebrd'
        func_descriptions['cusolverDn' + t + 'gesvd'] = 'compute the SVD of a mxn matrix A corresponding to the left and/or right singular vectors:  A=U*S*V^H'
        func_descriptions['cusolverDn' + t + 'gesvd_bufferSize'] = 'calculate buffer size for gesvd'
        # Undocumented Dense routines
        func_descriptions['cusolverDn' + t + 'laswp'] = ''

        # Sparse high level routines
        # func_descriptions['cusolverDn' + t + 'csrlsvlu'] = 'solves the linear system A*x=b on the GPU by sparse LU with partial pivoting'
        func_descriptions['cusolverDn' + t + 'csrlsvluHost'] = 'solves the linear system A*x=b on the CPU by sparse LU with partial pivoting'
        func_descriptions['cusolverDn' + t + 'csrlsvqr'] = 'solves the linear system A*x=b on the GPU by sparse QR factorization'
        func_descriptions['cusolverDn' + t + 'csrlsvqrHost'] = 'solves the linear system A*x=b on the CPU by sparse QR factorization'
        func_descriptions['cusolverDn' + t + 'csrlsvchol'] = 'solves the linear system A*x=b on the GPU by sparse Cholesky factorization'
        func_descriptions['cusolverDn' + t + 'csrlsvcholHost'] = 'solves the linear system A*x=b on the CPU by sparse Cholesky factorization'
        # func_descriptions['cusolverDn' + t + 'csrlsqvqr'] = 'solves the least-square problem x = argmin_z||A*z-b|| on the GPU by sparse QR factorization'
        func_descriptions['cusolverDn' + t + 'csrlsqvqrHost'] = 'solves the least-square problem x = argmin_z||A*z-b|| on the CPU by sparse QR factorization'
        func_descriptions['cusolverDn' + t + 'csreigvsi'] = 'solves the eigenvalue problem A*x=l*x on the GPU by the shift-inverse method'
        func_descriptions['cusolverDn' + t + 'csreigvsiHost'] = 'solves the eigenvalue problem A*x=l*x on the CPU by the shift-inverse method'
        # func_descriptions['cusolverDn' + t + 'csreigs'] = ''
        func_descriptions['cusolverDn' + t + 'csreigsHost'] = ' computes the number of algebraic eigenvalues in a given box B by a contour integral'

        # Sparse low level routines
        func_descriptions['cusolverSp' + t + 'csrqrBufferInfoBatched'] = 'calculate buffer information for csrqrsvBatched'
        func_descriptions['cusolverSp' + t + 'csrqrsvBatched'] = 'batched sparse QR factorization for solving either a set of least-squares problems or a set of linear systems'





    # operations common across all precisions
    func_descriptions['cusolverSpXcsrissymHost'] = 'Check if A has a symmetric pattern'
    func_descriptions['cusolverSpXcsrsymrcmHost'] = 'Symmetric reverse Cuthill McKee permutation'
    func_descriptions['cusolverSpXcsrperm_bufferSizeHost'] = 'calculate buffer size for csrpermHost'
    func_descriptions['cusolverSpXcsrpermHost'] = 'given left perumation vector P and right permutation vector Q, compute the permutation of matrix A:  B = P*A*Q^T'
    func_descriptions['cusolverSpXcsrqrAnalysisBatched'] = 'analyzes the sparity pattern of matrices Q and R of a QR factorization.  for use with csrqrsvBatched'
    # TODO: Fill descriptions for Refactorization routines
    func_descriptions['cusolverRfAccessBundledFactorsDevice'] = ''
    func_descriptions['cusolverRfAccessBundledFactorsHost'] = ''
    func_descriptions['cusolverRfAnalyze'] = ''
    func_descriptions['cusolverRfBatchAnalyze'] = ''
    func_descriptions['cusolverRfBatchRefactor'] = ''
    func_descriptions['cusolverRfBatchResetValues'] = ''
    func_descriptions['cusolverRfBatchSetupHost'] = ''
    func_descriptions['cusolverRfBatchSolve'] = ''
    func_descriptions['cusolverRfBatchZeroPivot'] = ''
    func_descriptions['cusolverRfExtractSplitFactorsHost'] = ''
    func_descriptions['cusolverRfRefactor'] = ''
    func_descriptions['cusolverRfResetValues'] = ''
    func_descriptions['cusolverRfSetupDevice'] = ''
    func_descriptions['cusolverRfSetupHost'] = ''
    func_descriptions['cusolverRfSolve'] = ''

    create_funcs = [cdef for cdef in ffi_lib.__dict__ if 'Create' in cdef]
    for func in create_funcs:
        tmp, obj = func.split('Create')
        if obj:
            func_descriptions[func] = "Create cuSOLVER {} structure.".format(obj)
        else:
            func_descriptions[func] = "Create cuSOLVER context."
    destroy_funcs = [cdef for cdef in ffi_lib.__dict__ if 'Destroy' in cdef]
    for func in destroy_funcs:
        tmp, obj = func.split('Destroy')
        if obj:
            func_descriptions[func] = "Destroy cuSOLVER {} structure.".format(obj)
        else:
            func_descriptions[func] = "Destroy cuSOLVER context."
    get_funcs = [cdef for cdef in ffi_lib.__dict__ if 'Get' in cdef]
    for func in get_funcs:
        tmp, obj = func.split('Get')
        func_descriptions[func] = "Get cuSOLVER {}.".format(obj)

    set_funcs = [cdef for cdef in ffi_lib.__dict__ if 'Set' in cdef]
    for func in set_funcs:
        tmp, obj = func.split('Set')
        func_descriptions[func] = "Set cuSOLVER {}.".format(obj)

    # prune any of the above that aren't in ffi_lib:
    func_descriptions = dict(
        (k, v) for k, v in func_descriptions.items(
            ) if k in ffi_lib.__dict__)

    with open(json_file, 'w') as fid:
        json.dump(func_descriptions, fid, sort_keys=True, indent=4)
