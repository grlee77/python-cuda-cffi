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
import re
import json
import warnings
import importlib
import numpy as np

import cffi

# TODO: improve autodetection of necessary cuda paths
CUDA_ROOT = os.environ.get('CUDA_ROOT', None) or '/usr/local/cuda'
if not os.path.isdir(CUDA_ROOT):
    raise ValueError("specified CUDA_ROOT is not a valid directory")

cuda_include_path = os.path.join(CUDA_ROOT, 'include')
if not os.path.isdir(cuda_include_path):
    raise ValueError("cuda include path not found.  please specify CUDA_ROOT.")

cuda_lib_path = os.path.join(CUDA_ROOT, 'lib64')
if not os.path.isdir(cuda_lib_path):
    cuda_lib_path = os.path.join(CUDA_ROOT, 'lib')
    if not os.path.isdir(cuda_lib_path):
        raise ValueError("cuda library path not found.  please specify "
                         "CUDA_ROOT.")


def ffi_init(lib_name, cffi_cdef, headers=[], libraries=[],
             include_dirs=[cuda_include_path], library_dirs=[cuda_lib_path],
             **kwargs):
    """ initialize FFI and FFILibrary objects using cffi

    Parameters
    ----------
    lib_name : str
        name of the cffi Out-of-Line module to be passed to ffi.set_source()
    cffi_cdef : str
        cffi cdef string
    headers : list of str, optional
        list of additional headers to include in the call to ffi.verify
    libraries : list of str, optional
        list of library names to pass to ffi.verify
    include_dirs : list of str, optional
        list of include paths for ffi.verify
    library_dirs : list of str, optional
        list of library paths for ffi.verify

    Returns
    -------
    ffi : cffi.api.FFI object
        cffi FFI object returned by ffi.verify
    ffi_lib : cffi.vengine_cpy.FFILibrary
        cffi FFILibrary object returned by ffi.verify

    """
    ffi = cffi.FFI()
    ffi.cdef(cffi_cdef)

    c_header_source = ''
    for hdr in headers:
        c_header_source += "#include <{}>\n".format(hdr)

    if cffi.__version__ > '1':
        use_verify = False  # now depricated, so no longer use it
    else:
        use_verify = True  # now depricated, so no longer use it

    if use_verify:
        ffi_lib = ffi.verify(c_header_source, libraries=libraries,
                             include_dirs=include_dirs,
                             library_dirs=library_dirs,
                             **kwargs)

        return ffi, ffi_lib
    else:
        previous_dir = os.getcwd()
        os.chdir(os.path.dirname(__file__))
        # try:
        #     from ._cusparse_ffi import ffi, lib
        # except:
        ffi.set_source(lib_name,
                       c_header_source,
                       libraries=libraries,
                       include_dirs=include_dirs,
                       library_dirs=library_dirs,
                       **kwargs)
        ffi.compile()
        os.chdir(previous_dir)
        mod = importlib.import_module('.' + lib_name, package='cuda_cffi')
        return mod.ffi, mod.lib


def get_cffi_filenames(ffi):
    """ returns the source and module filenames """
    sourcefilename = ffi.verifier.sourcefilename
    modulefilename = ffi.verifier.modulefilename
    return (sourcefilename, modulefilename)


def reindent(s, numSpaces=4, lstrip=True):
    """add indentation to a multiline string.

    Parameters
    ----------
    s : str
        string to reformat
    numSpaces : str, optional
        number of spaces to indent each line by
    lstrip : bool, optional
        if True, lstrip() prior to adding numSpaces

    Returns
    -------
    s : str
        reformatted str
    """
    s = s.split('\n')
    if lstrip:
        s = [line.lstrip() for line in s]

    for idx, line in enumerate(s):
        if line.strip() == '':
            # don't indent empty lines
            s[idx] = ''
        else:
            s[idx] = (numSpaces * ' ') + line
    s = '\n'.join(s)
    return s


def split_line(line, break_pattern=', ', nmax=80, pad_char='('):
    """ split a line (repeatedly) until length < nmax chars.

    split will occur at last occurence of break_pattern occuring before nmax
    characters

    subsequent lines will be indented until the first occurance of pad_char
    in the initial line

    Parameters
    ----------
    line : str
        line to reformat
    break_pattern : str, optional
        break line only where this pattern occurs
    nmax : int, optional
        max number of characters to allow
    pad_char : str, optional
        auto-indent subsequent lines up to the first occurance of pad_char

    Returns
    -------
    new_line : str
        reformatted line
    """
    if len(line) < nmax:
        return line.rstrip() + '\n'
    locs, break_loc = _find_breakpoint(line,
                                       break_pattern=break_pattern,
                                       nmax=nmax)
    if break_loc is None:
        return line.rstrip() + '\n'
    if pad_char is not None:
        npad = line.find(pad_char) + 1
    else:
        npad = 0

    lines = []
    lines.append(line[:break_loc].rstrip())
    line = ' ' * npad + line[break_loc:]
    while (len(line) > nmax - 1) and (break_loc is not None):
        locs, break_loc = _find_breakpoint(line,
                                           break_pattern=break_pattern,
                                           nmax=nmax)
        lines.append(line[:break_loc].rstrip())
        line = ' ' * npad + line[break_loc:]
    lines.append(line.rstrip())
    return '\n'.join(lines) + '\n'


def _find_breakpoint(line, break_pattern=', ', nmax=80):
    """ determine where to break the line """
    locs = [m.start() for m in re.finditer(break_pattern, line)]
    if len(locs) > 0:
        break_loc = locs[np.where(
            np.asarray(locs) < (nmax - len(break_pattern)))[0][-1]]
        break_loc += len(break_pattern)
    else:
        break_loc = None
    return locs, break_loc


def _build_func_sig(func_name, arg_dict, return_type):
    """ generate the python wrapper function signature line(s). """

    if 'Create' in func_name:
        # don't pass in any argument to creation functions
        return "def %s():\n" % func_name

    if ('Get' in func_name) and (return_type == 'cusparseStatus_t') and \
       len(arg_dict) == 2:
        basic_getter = True
    else:
        basic_getter = False

    sig = "def %s(" % func_name
    for k, v in arg_dict.items():
        is_ptr = '*' in v
        if is_ptr and basic_getter:
            continue
        sig += k + ", "
    sig = sig[:-2] + "):\n"
    # wrap to 2nd line if too long
    return split_line(sig, break_pattern=', ', nmax=79)


def _build_doc_str(arg_dict, func_description='', variable_descriptions={}):
    """ generate python wrapper docstring """
    docstr = '"""' + func_description + '\n'
    docstr += 'Parameters\n----------\n'
    for k, v in arg_dict.items():
        docstr += k + " : " + v + "\n"
        if k in variable_descriptions:
            docstr += reindent(variable_descriptions[k],
                               numSpaces=4,
                               lstrip=True)
        else:
            print("no variable description provided for {}".format(k))
    docstr += '"""\n'
    return reindent(docstr, numSpaces=4, lstrip=False)


def _func_str(func_name, arg_dict, return_type, build_func_body,
              variable_descriptions={}, func_description=''):
    """ build a single python wrapper """
    fstr = _build_func_sig(func_name, arg_dict, return_type)
    fstr += _build_doc_str(arg_dict, func_description=func_description,
                           variable_descriptions=variable_descriptions)
    fstr += build_func_body(func_name, arg_dict, return_type)
    return fstr


def build_python_func(cdef, build_func_body, variable_descriptions={},
                      func_descriptions={}):
    """ wrap a single python function corresponding to the given cdef C
    function string.

    Parameters
    ----------
    cdef : str
        single line string containing a C function definition
    build_func_body : callable
        function to build a string corresponding to the body of the python
        wrapper function
    variable_descriptions : dict
        dictionary of variable descriptions for the docstring
    func_descriptions : dict
        dictionary of function descriptions for the docstring

    Returns
    -------
    str corresponding to the python_wrapper
    """


    cdef_regex = "(\w*)\s*(\w*)\s*\((.*)\).*"
    p = re.compile(cdef_regex)
    match = p.search(cdef)
    (return_type, func_name, func_args) = match.group(1, 2, 3)
    func_args = func_args.split(', ')

    if '[]' in cdef:
        warnings.warn("[] within cdef not currently supported. "
                      "skipping function: {}".format(func_name))
        return ''

    from collections import OrderedDict
    arg_dict = OrderedDict()
    for arg in func_args:
        substr = arg.split()
        if len(substr) == 2:
            val = substr[0]
        else:
            val = substr[-2]
        key = substr[-1]
        # handle pointer
        if key[0] == '*':
            val += ' *'
            key = key[1:]
        # handle pointer to pointer
        if key[0] == '*':
            val += '*'
            key = key[1:]
        arg_dict[key] = val

    func_description = func_descriptions.get(func_name, '')
    return _func_str(func_name, arg_dict, return_type,
                     build_func_body=build_func_body,
                     variable_descriptions=variable_descriptions,
                     func_description=func_description)


def get_variable_descriptions(var_def_json):
    """ load variable description dictionary from .json file"""
    with open(var_def_json, 'r') as fid:
        variable_descriptions = json.load(fid)
    for k, v in variable_descriptions.items():
        variable_descriptions[k] = split_line(v, break_pattern=' ', nmax=72,
                                              pad_char=None)
    return variable_descriptions


def get_function_descriptions(func_def_json):
    """ load function description dictionary from .json file"""
    with open(func_def_json, 'r') as fid:
        func_descriptions = json.load(fid)
    for k, v in func_descriptions.items():
        func_descriptions[k] = split_line(v, break_pattern=' ', nmax=72,
                                          pad_char=None)
    return func_descriptions


def generate_cffi_python_wrappers(cffi_cdef,
                                  build_func_body,
                                  variable_defs_json='',
                                  func_defs_json='',
                                  python_wrapper_file=None):
    """ generate python wrappers for all functions within cffi_cdef.

    Parameters
    ----------
    cffi_cdef : str
        cffi definition string as generated by `generate_cffi_cdef`
    build_func_body : callable
        function to build a string corresponding to the body of the python
        wrapper function
    variable_defs_json : str, optional
        filename of .json file containing dictionary of variable descriptions
    func_defs_json : str, optional
        filename of .json file containing dictionary of function descriptions
    python_wrapper_file : str, optional
        file to output the generated python wrappers to

    Returns
    -------
    python_wrappers : str
        string containing all of the python wrappers

    """
    cffi_cdef_list = cffi_cdef.split('\n')

    # find lines containing a function definition
    func_def_lines = []
    for idx, line in enumerate(cffi_cdef_list):
        if line.startswith('cusolver') or line.startswith('cusparse'):  # TODO: fix this!
            func_def_lines.append(idx)

    # reformat each definition into a single line for easier string processing
    n_funcs = len(func_def_lines)
    cdef_list = []
    for i in range(len(func_def_lines)):
        loc1 = func_def_lines[i]
        if i < n_funcs - 1:
            loc2 = func_def_lines[i + 1]
            cdef = ' '.join([l.strip() for l in cffi_cdef_list[loc1:loc2]])
        else:
            cdef = ' '.join([l.strip() for l in cffi_cdef_list[loc1:]])
        # strip any remaining comments after the semicolon
        cdef = cdef[:cdef.find(';') + 1]
        cdef_list.append(cdef)

    # read function and variable definition strings to use when building the
    # the Python doc strings
    if variable_defs_json:
        variable_descriptions = get_function_descriptions(variable_defs_json)
    if func_defs_json:
        func_descriptions = get_function_descriptions(func_defs_json)

    # build the wrappers
    python_wrappers = ''
    for cdef in cdef_list:
        python_wrappers += build_python_func(
            cdef,
            build_func_body,
            variable_descriptions=variable_descriptions,
            func_descriptions=func_descriptions)

    if python_wrapper_file is not None:
        with open(python_wrapper_file, 'w') as f:
            f.write(python_wrappers)
    return python_wrappers


def wrap_library(lib_name, cffi_file, python_wrapper_file, build_body_func,
                 ffi_init_func, cdef_generator_func,
                 variable_defs_json=None, func_defs_json=None,
                 func_description_generator_func=None,
                 force_update=False, verbose=True):
    if not os.path.exists(cffi_file) or force_update:
        if verbose:
            print("first import:  cffi interface file being created. This may "
                  "take several seconds")
        cffi_cdef = cdef_generator_func(cffi_out_file=cffi_file)
    else:
        with open(cffi_file, 'r') as f:
            cffi_cdef = f.read()

    ffi, ffi_lib = ffi_init_func(lib_name, cffi_cdef)

    if not os.path.exists(python_wrapper_file):
        if verbose:
            print("first import:  python wrappers being created.")
        if not os.path.exists(func_defs_json):
            if func_description_generator_func is not None:
                func_description_generator_func(ffi_lib,
                                                json_file=func_defs_json)
            else:
                raise ValueError(
                    "specified func_defs_json file doesn't exist and the "
                    "generate_func_descriptions_json callable was not given")
        generate_cffi_python_wrappers(
            cffi_cdef,
            build_body_func,
            variable_defs_json=variable_defs_json,
            func_defs_json=func_defs_json,
            python_wrapper_file=python_wrapper_file)

    return ffi, ffi_lib