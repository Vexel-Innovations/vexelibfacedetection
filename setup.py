# VexelLibFaceDetection - Python Package Setup
# Copyright (c) 2026, Vexel Innovations. All rights reserved.

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'pyfacedetect',
        sources=[
            'python/pyfacedetect.cpp',
            'src/vexelfacedetect.cpp',
            'src/facedetectcnn.cpp',
            'src/facedetectcnn-data.cpp',
            'src/facedetectcnn-model.cpp',
        ],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            'src',
        ],
        language='c++',
    ),
]

class BuildExt(build_ext):
    c_opts = {
        'msvc': ['/EHsc', '/std:c++20', '/O2', '/arch:AVX2', '-D_ENABLE_AVX2'],
        'unix': ['-std=c++20', '-O3', '-mavx2', '-mfma', '-D_ENABLE_AVX2'],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, ['-std=c++20'])
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(
    name='vexelfacedetect',
    version='1.0.0',
    author='Vexel Innovations',
    author_email='info@vexelinnovations.com',
    description='VexelLibFaceDetection — High-Performance CNN Face Detection by Vexel Innovations',
    url='https://github.com/Vexel-Innovations/libfacedetection',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.6.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    license='BSD-3-Clause',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
)
