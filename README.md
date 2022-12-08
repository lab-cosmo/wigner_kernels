A program to benchmark Wigner kernels.

In addition to the requirements in requirements.txt, rascaline and the sparse accumulation package are needed.

To compile the Fortran file: python -m numpy.f2py -c fortran.f -m fortran
(A Fortran compiler is also required)
