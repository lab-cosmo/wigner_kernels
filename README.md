# Wigner Kernels

This a collection of programs to benchmark Wigner kernels on the dipole moments of the QM9 dataset. 
We recommend running them on GPU, as they have not been tested on CPU.


## Prerequisites

The python requirements are listed in requirements.txt

In our programs, we also make use of a Fortran subroutine to calculate scaled modified spherical Bessel functions.
To compile the Fortran file: 

`python -m numpy.f2py -c fortran.f -m fortran`

(A Fortran compiler is required)


## Usage

This directory contains three scripts: `run_wk_dipoles.py`, `dipoles_large.py`, and `grid.py`.

### run_wk.py

The `run_wk_dipoles.py` is appropriate for small or medium-sized training sets. It can be invoked, for instance, as follows:

`python run_wk_dipoles.py dipoles.json 500 0`

Here, `dipoles.json` is an input file, `500` refers to the number of training points, and `0` is the random seed used to shuffle
the training part of the dataset before extracting the train structures.

The main piece of output of the script is the MAE test set error.

### large.py and grid.py 

These two scripts can be used to fit large training sets.

`large.py` can be invoked as

`python large_dipoles.py 1 2`

This will calculate a 5000x5000 chunk of the Wigner kernel matrix for the whole dataset.
The inputs 1 and 2 refer to the location of the chunk within the Wigner kernel matrix. In this case, the output will be a dumped PyTorch
tensor corresponding to all elements of the Wigner kernel matrix in rows 5000 to 10000 and columns 10000 to 15000. 

Once all chunks in the upper triangular part of the matrix have been calculated (the kernel is symmetric), the fit can be carried out
with `grid.py`. This script executes a grid search over the kernel mixing hyperparameters and it returns the test set error.

`grid.py` is the only script that is supposed to run on CPU (preferrably with large memory), and it can be invoked as 

`python grid.py 20000 0`

Here, the first argument is the number of training points, while the second number is the random seed used to shuffle the dataset,
thereby changing the composition of the training and test sets. The second argument is ignored if, like in this case, all 20000
training set structures are used.


## Examples

For reproducibility purposes, an output example is provided for the `run_wk_dipoles.py` script. This is `dipoles.out`,
which corresponds to the output of `python run_wk_dipoles.py dipoles.json 500 0`
