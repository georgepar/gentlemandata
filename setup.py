from numpy.distutils.misc_util import Configuration

import numpy
import os
from Cython.Build import cythonize


def configuration(parent_package="", top_path=None):
    config = Configuration('gentlemandata', parent_package, top_path)
    libraries = []
    if os.name == 'posix':
        libraries.append('m')
    config.add_extension('_fast_utils',
                         sources=["gentlemandata/_fast_utils.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries)

    config.ext_modules[-1] = cythonize(config.ext_modules[-1], language='c')[0]
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration().todict())
