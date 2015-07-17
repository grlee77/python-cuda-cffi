try:
    import pkg_resources
    __version__ = pkg_resources.require('cuda_cffi')[0].version
except:
    __version__ = '0.1'
