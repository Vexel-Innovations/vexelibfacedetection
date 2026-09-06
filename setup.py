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
            'src/facedetect20.cpp',
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

def has_flag(compiler, flagname):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except Exception:
            return False
    return True

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
    name='pyfacedetect',
    version='0.2.0',
    author='Shiqi Yu / Antigravity Modernization',
    description='Modern C++20 High-Performance CNN Face Detection Python Library',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.6.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
