cimport cython
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int multiply(int a, int b):
    return a * b