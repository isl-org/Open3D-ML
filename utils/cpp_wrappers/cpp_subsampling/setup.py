from distutils.core import setup, Extension
import numpy.distutils.misc_util

# Adding OpenCV to project
# ************************

# Adding sources of the project
# *****************************

m_name = "grid_subsampling"

SOURCES = ["../cpp_utils/cloud/cloud.cpp",
           "grid_subsampling/grid_subsampling.cpp",
           "wrapper.cpp"]

module = Extension(m_name,
                   sources=SOURCES,
                   extra_compile_args=['-std=c++11',
                                       '-D_GLIBCXX_USE_CXX11_ABI=0'])

setup(ext_modules=[module], include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs())








