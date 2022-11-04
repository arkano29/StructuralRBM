from setuptools import setup
from Cython.Build import cythonize
import numpy

print(numpy.get_include())
setup(
    ext_modules = cythonize("sparseupdate.pyx"),include_dirs=[numpy.get_include()]
)