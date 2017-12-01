#!/usr/bin/env python

import os


def main():
    # from aksetup_helper import hack_distutils, setup, NumpyExtension
    # hack_distutils()

    # from setuptools import setup, Extension

    from distutils.core import setup
    from distutils.extension import Extension

    try:
        from distutils.command.build_py import build_py_2to3 as build_py
    except ImportError:
        # 2.x
        from distutils.command.build_py import build_py

    INCLUDE_DIRS = [ os.path.join(os.environ.get("HOME"), "pool", "include") ]
    LIBRARY_DIRS = [ os.path.join(os.environ.get("HOME"), "pool", "lib") ]

    LIBRARIES = ['boost_python3']
    EXTRA_DEFINES = { }
    conf = { }
    conf["CXXFLAGS"] = ['-Wno-sign-compare']
    conf["LDFLAGS"] = [ ]

    setup(
            name="MSD_BOOST",
            version="1.0",
            description="Mass-Spring-Damper Simulation",
            author="Roger Stuckey",
            author_email="Roger.Stuckey@gmail.com",
            license="BSD",
            url="http://rogerstuckey.com/",
            ext_modules=[
                    Extension(
                        "msd.msdbx",
                        sources=["msdbx.cpp"],
                        include_dirs=INCLUDE_DIRS,
                        library_dirs=LIBRARY_DIRS,
                        libraries=LIBRARIES,
                        define_macros=list(EXTRA_DEFINES.items()),
                        extra_compile_args=conf["CXXFLAGS"],
                        extra_link_args=conf["LDFLAGS"],
                        )
                    ],

            zip_safe=False,

            # 2to3 invocation
            cmdclass={'build_py': build_py})


if __name__ == '__main__':
    main()
