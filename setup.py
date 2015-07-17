#!/usr/bin/env python
'''
Installation script for cuda_cffi

Note:
To make a source distribution:
python setup.py sdist

To make an RPM distribution:
python setup.py bdist_rpm

To Install:
python setup.py install --prefix=/usr/local

See also:
python setup.py bdist --help-formats
'''
import os
#from distutils.core import setup
from setuptools import setup

install_requires = ['numpy',
                    'scipy >= 0.9.0',
                    'pycuda >= 2013.1']
extras_require = {}

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name='cuda_cffi',
          version='0.1',
          description='NVIDIA cuSPARSE and cuSOLVER wrappers',
          author='Gregory Lee',
          author_email='grlee77@gmail.com',
          url='https://github.com/grlee77/cuda_cffi',
          license='BSD',
          packages=['cuda_cffi'],
          # Force installation of __init__.py in namespace package:
          data_files = [('cuda_cffi',
            ['cuda_cffi/cusparse_variable_descriptions.json',
             'cuda_cffi/cusolver_variable_descriptions.json'])],
          install_requires = install_requires,
          extras_require = extras_require)
