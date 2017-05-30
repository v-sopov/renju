from distutils.core import setup, Extension
import numpy


mcts_utils_module = Extension('mcts_utils',
                    define_macros = [('PYTHON_C_EXTENTION', None)],
                    sources = ['mcts_utils.cpp'],
                    extra_compile_args = ['-std=c++11'])

setup (name = 'mcts_utils',
       version = '1.0',
       description = 'Special methods for renju MCTS',
       include_dirs = [numpy.get_include()],
       ext_modules = [mcts_utils_module])
