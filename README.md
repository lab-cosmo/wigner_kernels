# Wigner Kernels

This a collection of programs to benchmark Wigner kernels. We recommend running them on GPU, as they have not been tested on CPU.


## Prerequisites

The python requirements are listed in requirements.txt

In our programs, we also make use of a Fortran subroutine to calculate scaled modified spherical Bessel functions.
To compile the Fortran file: 

`python -m numpy.f2py -c fortran.f -m fortran`

(A Fortran compiler is required)


## Usage

This directory contains three scripts: `run_wk.py`, `large.py`, and `grid.py`.

### run_wk.py

The `run_wk.py` script will be appropriate in the vast majority of cases. It can be invoked, for instance, as follows:

`python run_wk.py methane.json 500 0`

Here, `methane.json` is an input file, `500` refers to the number of training points, and `0` is the random seed used to shuffle
the dataset before extracting the train and test structures.

The main piece of output of the script is a test set error (MAE or RMSE according to what was selected in the input file).

### large.py and grid.py 

These two scripts can be used to apply the Wigner kernel model to large datasets. The dataset is hardcoded to be QM9.

`large.py` can be invoked as 

`python large.py 2 3`

This will calculate a 10000x10000 chunk of the Wigner kernel matrix for the whole dataset.
The inputs 2 and 3 refer to the location of the chunk within the Wigner kernel matrix. In this case, the output will be a dumped PyTorch
tensor corresponding to all elements of the Wigner kernel matrix in rows 20000 to 30000 and columns 30000 to 40000. 

Once all chunks in the upper triangular part of the matrix have been calculated (the kernel is symmetric), the fit can be executed
with `grid.py`. This script executes a grid search over the kernel mixing hyperparameters and it returns the test set error.

`grid.py` can be invoked as 

`python grid.py 110000 0`

Here, the first argument is the number of training points, while the second number is the random seed used to shuffle the dataset,
thereby changing the composition of the training and test sets.


## Examples

For reproducibility purposes, three examples of the outputs are provided for the `run_wk.py` script. These are

`methane.out`, which is the output of `python run_wk.py methane.json 500 0`
`gold.out`, which is the output of `python run_wk.py gold.json 500 0`
`qm9.out`, which is the output of `python run_wk.py qm9.json 500 0`
