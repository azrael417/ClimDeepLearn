.PHONY: default
default: build

PYTHON = python
PYTHON_INCLUDE = ${shell ${PYTHON} -c 'from distutils import sysconfig; print( sysconfig.get_python_inc() )'}
NUMPY_INCLUDE = ${shell ${PYTHON} -c 'import numpy; print( numpy.get_include() )'}

TIMEIT = ${shell ${PYTHON} -c 'import timeit; print(timeit.__file__)'}

CYTHON = cython
src: cFloodFillSearch.c
floodFillSearch.c: floodFillSearch.py floodFillSearch.pyd
	${CYTHON} -I${NUMPY_INCLUDE} $<


CC = cc -shared -fPIC
CFLAGS = -shared -fPIC
SO = ${shell ${PYTHON} -c 'import imp; print (imp.get_suffixes()[0][0])'}
.PHONY: build
build: cFloodFillSearch${SO}
cFloodFillSearch${SO}: floodFillSearch.py floodFillSearch.pyd
	${PYTHON} setup.py build_ext --inplace


.PHONY: test testTiming.py
test: cFloodFillSearch${SO}
	#${PYTHON} ${TIMEIT} -n2 -r3 "import testTiming as test; test.runTest()"
	${PYTHON} -c "import testTiming as test; test.runTest()"

.PHONY: clean
clean:
	${RM} floodFillSearch.c floodFillSearch${SO}
