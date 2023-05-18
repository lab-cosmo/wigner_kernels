import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from math import factorial

from validation import get_validation_predictions
import argparse
from error_measures import get_dipole_sae, get_dipole_mae

import sys
n_train = int(sys.argv[1])
random_seed = int(sys.argv[2])
print(n_train, random_seed)

TARGET_KEY = "dipole_b3lyp"
DATASET_PATH = "qm9_dipoles.xyz"
n_test = 1000

import ase
from ase import io
all_structures = ase.io.read(DATASET_PATH, ":")
train_structures = all_structures[:20000]
test_structures = all_structures[20000:]

if n_train != 20000:
    np.random.seed(random_seed)
    permutation = np.random.permutation(20000)
    permutation = permutation[:n_train]
    train_structures = [train_structures[index] for index in permutation]
    permutation = torch.LongTensor(permutation)

train_dipoles = [structure.info[TARGET_KEY] for structure in train_structures]
train_dipoles = torch.tensor(np.array(train_dipoles), dtype = torch.get_default_dtype())
train_dipoles = train_dipoles[:, [1, 2, 0]]  # real spherical harmonics ordering: y, z, x

test_dipoles = [structure.info[TARGET_KEY] for structure in test_structures]
test_dipoles = torch.tensor(np.array(test_dipoles), dtype = torch.get_default_dtype())
test_dipoles = test_dipoles[:, [1, 2, 0]]  # real spherical harmonics ordering: y, z, x

NU_MAX = 4
n_validation_splits = 10
assert n_train % n_validation_splits == 0
n_validation = n_train // n_validation_splits
n_train_sub = n_train - n_validation

best_validation_error = 1e30
best_coefficients = None
a_exp_best = None
b_best = None
best_test_error = None

a_exp = 14.92788162
b = 5.78247607

if n_train == 5000:
    min_a_exp = 3.0
    max_a_exp = 4.5
    min_b = 5.0
    max_b = 15.0

elif n_train == 10000:
    min_a_exp = 3.5
    max_a_exp = 5.0
    min_b = 5.0
    max_b = 15.0

elif n_train == 20000:
    min_a_exp = 3.5
    max_a_exp = 5.0 #?????????????????????
    min_b = 10.0
    max_b = 20.0

else:
    exit()

for a_exp in np.linspace(min_a_exp, max_a_exp, 7):
    for b in np.linspace(min_b, max_b, 11):
        train_train_kernel = torch.zeros((20000, 20000, 3, 3), dtype = torch.get_default_dtype())
        test_train_kernel = torch.zeros((n_test, 20000, 3, 3), dtype = torch.get_default_dtype())

        # define coefficients
        print()
        print(a_exp, b) 
        a = np.exp(np.log(10.0)*a_exp)
        coefficients = torch.tensor(
            [
                a*b**nu/factorial(nu) for nu in range(1, NU_MAX+1)
            ]
        )
        print(coefficients)
        
        # load train kernels
        for i in range(4):
            for j in range(i, 4):
                chunk_primitive = torch.load(f"wks_{i}_{j}.pt")
                chunk = chunk_primitive @ coefficients
                train_train_kernel[5000*i:5000*(i+1), 5000*j:5000*(j+1)] = chunk
                train_train_kernel[5000*j:5000*(j+1), 5000*i:5000*(i+1)] = chunk.swapaxes(0, 1).swapaxes(2, 3)

        # load test kernels
        for i in range(4):
            chunk_primitive = torch.load(f"wks_{i}_4.pt")
            chunk = chunk_primitive @ coefficients
            test_train_kernel[:, 5000*i:5000*(i+1)] = chunk.swapaxes(0, 1).swapaxes(2, 3)

        # scramble kernels:
        if n_train != 20000:
            train_train_kernel = train_train_kernel[permutation][:, permutation]
            test_train_kernel = test_train_kernel[:, permutation]
        
        # calculate validation error:
        validation_error = 0.0
        for i_validation_split in range(n_validation_splits):
            index_validation_start = i_validation_split*n_validation
            index_validation_stop = index_validation_start + n_validation

            K_train_sub = torch.empty((n_train_sub, n_train_sub, 3, 3), device = train_train_kernel.device)
            K_train_sub[:index_validation_start, :index_validation_start] = train_train_kernel[:index_validation_start, :index_validation_start]
            if i_validation_split != n_validation_splits - 1:
                K_train_sub[:index_validation_start, index_validation_start:] = train_train_kernel[:index_validation_start, index_validation_stop:]
                K_train_sub[index_validation_start:, :index_validation_start] = train_train_kernel[index_validation_stop:, :index_validation_start]
                K_train_sub[index_validation_start:, index_validation_start:] = train_train_kernel[index_validation_stop:, index_validation_stop:]
            y_train_sub = train_dipoles[:index_validation_start]
            if i_validation_split != n_validation_splits - 1:
                y_train_sub = torch.concat([y_train_sub, train_dipoles[index_validation_stop:]], dim=0)

            K_validation = train_train_kernel[index_validation_start:index_validation_stop, :index_validation_start, :]
            if i_validation_split != n_validation_splits - 1:
                K_validation = torch.concat([K_validation, train_train_kernel[index_validation_start:index_validation_stop, index_validation_stop:, :]], dim = 1)
            y_validation = train_dipoles[index_validation_start:index_validation_stop] 

            try:
                validation_predictions = get_validation_predictions(K_train_sub, y_train_sub, K_validation)
                validation_error += get_dipole_sae(validation_predictions, y_validation.flatten()).item()
            except Exception as e:
                print("WARNING:", e)
                validation_error += 10e30

        validation_error = validation_error/n_train
        c = torch.linalg.solve(
        train_train_kernel.swapaxes(1, 2).reshape(n_train*3, n_train*3) +  # nu = 1, ..., 4 kernels
        torch.eye(3*n_train, dtype=torch.get_default_dtype())  # regularization
        , 
        train_dipoles.flatten())
        test_predictions = test_train_kernel.swapaxes(1, 2).reshape(n_test*3, n_train*3) @ c
        test_error = (get_dipole_mae(test_predictions, test_dipoles.flatten()).item()) 

        print(validation_error, test_error)

        if validation_error < best_validation_error:
            best_validation_error = validation_error
            best_coefficients = coefficients 
            best_test_error = test_error
            a_exp_best = a_exp
            b_best = b

print()
print(a_exp_best, b_best)
print(n_train, best_test_error)
