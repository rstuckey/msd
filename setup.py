#!/usr/bin/env python

import os


def main():
    from aksetup_helper import hack_distutils, setup, NumpyExtension

    hack_distutils()

    try:
        from distutils.command.build_py import build_py_2to3 as build_py
    except ImportError:
        # 2.x
        from distutils.command.build_py import build_py

    # INCLUDE_DIRS = ['../pyublas/pyublas/include']
    # INCLUDE_DIRS = ["/usr/local/lib/python2.7/dist-packages/PyUblas-2013.1-py2.7-linux-i686.egg/pyublas/include"]
    INCLUDE_DIRS = [ os.path.join(os.environ.get("HOME"), "pool", "include") ]

    LIBRARY_DIRS = [ ]
    LIBRARY_DIRS = [ os.path.join(os.environ.get("HOME"), "pool", "lib") ]
#    LIBRARIES = ['boost_python-py27']
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

            # numpy is often under the setuptools radar.
            #setup_requires=[
                    #"numpy>=1.0.4",
                    #],
#            install_requires=[
                    #"numpy>=1.0.4",
#                    "pytest>=2",
#                    ],

#            packages=["pyublas"],
#            ext_package="pyublas",
            ext_modules=[
                    # NumpyExtension(
                    #     "_internal",
                    #     ext_src,
                    #     include_dirs=INCLUDE_DIRS,
                    #     library_dirs=LIBRARY_DIRS,
                    #     libraries=LIBRARIES,
                    #     define_macros=list(EXTRA_DEFINES.items()),
                    #     extra_compile_args=conf["CXXFLAGS"],
                    #     extra_link_args=conf["LDFLAGS"],
                    #     ),
                    NumpyExtension(
                        "msd.msde",
                        ["msde.cpp"],
                        include_dirs=INCLUDE_DIRS,
                        library_dirs=LIBRARY_DIRS,
                        libraries=LIBRARIES,
                        define_macros=list(EXTRA_DEFINES.items()),
                        extra_compile_args=conf["CXXFLAGS"],
                        extra_link_args=conf["LDFLAGS"],
                        )
                    ],

            # include_package_data=True,
            # package_data={
            #         "pyublas": [
            #             "include/pyublas/*.hpp",
            #             ]
            #         },

            zip_safe=False,

            # 2to3 invocation
            cmdclass={'build_py': build_py})


if __name__ == '__main__':
    main()
