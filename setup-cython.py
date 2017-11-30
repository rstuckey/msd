#!/usr/bin/env python

# from distutils.core import setup
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


# setup(
#     ext_modules = cythonize("msdc.pyx", include_path=[numpy.get_include()])
# )
setup(
	name="MSD_CYTHON",
	version="1.0",
    description="Mass-Spring-Damper Simulation",
    author="Roger Stuckey",
    author_email="Roger.Stuckey@gmail.com",
    license="BSD",
    url="http://rogerstuckey.com/",
    ext_modules=cythonize(Extension("msd.msdc",
                                    ["msdc.pyx"],
                                    include_dirs=[numpy.get_include()],
                                    )),
)
