from distutils.core import setup,Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

inc_dirs = []
inc_dirs.append(numpy.get_include())
lib_dirs = []
#lib_dirs.append(numpy.get_lib())
libs = []

cmdclass = {'build_ext': build_ext}

extensions = [Extension("cFloodFillSearch",["floodFillSearch.py"],libraries=libs,library_dirs=lib_dirs,include_dirs=inc_dirs,runtime_library_dirs=lib_dirs)]

revision = "1.0"

#            0packages = ['floodfillsearch'], \
setup(
            name = 'floodfillsearch', \
            version = revision, \
            description = 'A tool for finding connected components', \
            author = "Travis A. O'Brien", \
            author_email = "TAOBrien@lbl.gov", \
            url = "https://bitbucket.org/lbl-cascade/floodfillsearch", \
            download_url = "https://bitbucket.org/lbl-cascade/floodfillsearch/get/v{}.tar.gz".format(revision), \
            keywords = ['connected component','flood'], \
            classifiers = [], \
            cmdclass = cmdclass,\
            ext_modules = extensions, \
            )
