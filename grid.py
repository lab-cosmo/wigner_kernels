import torch
torch.set_default_dtype(torch.float64)
import ase
from ase import io
import math
import sys

import numpy as np
np.random.seed(int(sys.argv[1]))

structures = ase.io.read("datasets/qm9.xyz", ":")
permutation = np.random.permutation(len(structures))
structures = [structures[index] for index in permutation]
permutation = torch.LongTensor(permutation)

y = torch.tensor(
        [
                structure.info["U0"] for structure in structures                
        ],
        dtype = torch.get_default_dtype()
)*627.5
n_structures = y.shape[0]

n_train = 110000
n_validation = 10000
K = torch.zeros((n_structures, n_structures), dtype = torch.get_default_dtype())

C_exp_best = 0
exp_best = 0
validation_best = 1e30
test_best = 1e30
for C_exp in np.linspace(5.25, 6.75, 7):
        for exp in np.linspace(9, 20, 12):
                C = 10.0**C_exp
                print(C_exp, exp) 
                coefficients = torch.tensor([1.0e9] + # You could decrease it even more... 
                        [C * exp**nu / math.factorial(nu) for nu in range(1, 5)], 
                        dtype = torch.get_default_dtype())
                """
                coefficients = torch.tensor( 
                        [C * exp**nu / math.factorial(nu) for nu in range(5)], 
                        dtype = torch.get_default_dtype()
                )
                """
                print(coefficients)

                for i in range(14):
                        for j in range(i, 14):
                                chunk_primitive = torch.load(f"wks_{i}_{j}.pt")
                                chunk = chunk_primitive @ coefficients
                                K[10000*i:10000*(i+1), 10000*j:10000*(j+1)] = chunk
                                K[10000*j:10000*(j+1), 10000*i:10000*(i+1)] = chunk.T

                print("Kernels loaded")
                print("Average: ", torch.mean(K))

                K = K[permutation][:, permutation]

                K_train_train = K[:n_train, :n_train]
                K_validation_train = K[n_train:n_train+n_validation, :n_train]
                K_test_train = K[n_train+n_validation:, :n_train]

                y_train = y[:n_train]
                y_validation = y[n_train:n_train+n_validation]
                y_test = y[n_train+n_validation:]

                c = torch.linalg.solve(K_train_train+torch.eye(n_train, dtype = torch.get_default_dtype()), y_train)

                validation_error = torch.mean(torch.abs(y_validation-K_validation_train@c)).item()
                test_error = torch.mean(torch.abs(y_test-K_test_train@c)).item()

                print()
                print("Validation", validation_error, "kcal/mol")
                print("Validation", validation_error*43.36411531, "meV")
                print()
                print("Test", test_error, "kcal/mol")
                print("Test", test_error*43.36411531, "meV")
                print()
                if validation_error < validation_best:
                        C_exp_best = C_exp
                        exp_best = exp
                        validation_best = validation_error
                        test_best = test_error

print()
print("FINAL")
print(C_exp_best, exp_best)
print()
print("Validation", validation_best, "kcal/mol")
print("Validation", validation_best*43.36411531, "meV")
print()
print("Test", test_best, "kcal/mol")
print("Test", test_best*43.36411531, "meV")
print()




