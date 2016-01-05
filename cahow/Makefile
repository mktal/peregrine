## pass it as argument to make, e.g., make USE_ACCELERATE=true
## USE_ACCELERATE=no

PYTHON_ROOT = /Users/xtang/anaconda
PYTHON_LINK = ${PYTHON_ROOT}/lib
PYTHON_INCLUDE = ${PYTHON_ROOT}/include/python2.7

OPENBLAS_LINK = /usr/local/opt/openblas/lib

CXX=clang++

CXX_INCLUDES = -I ./include -I${PYTHON_INCLUDE}
LDBLAS = -llapack -lblas
CXXBLAS = -DUSE_CBLAS

ifeq ($(USE_ACCELERATE), true)
	LDBLAS = -framework Accelerate -framework CoreFoundation
    $(info **use accelerate BLAS....)
endif
ifeq ($(USE_OPENBLAS), true)
	LDBLAS = -llapack -lopenblas -L$(OPENBLAS_LINK)
    $(info **use openBLAS....)
endif

LDPYTHON = -Wl,-rpath,$(PYTHON_ROOT) -L${PYTHON_LINK} -lpython2.7
LDFLAGS = -Wall -shared ${LDPYTHON} ${LDBLAS}
CXXOPTFLAGS = -Wall -fpic -m64 -fno-omit-frame-pointer -std=c++11 -O3 -DNDEBUG ${CXXBLAS} $(CXX_INCLUDES)

%.so: src/cahow.cpp src/matrix.cpp src/array.cpp src/lhac-py-gen.cpp
	$(CXX) $(CXXOPTFLAGS) $(LDFLAGS) $^ -o $@

clean :
	rm -f *.so





