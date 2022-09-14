import torch
import copy
import numpy as np
import ase.io

from pytorch_prototype.code_pytorch import *
from pytorch_prototype.utilities import *
from pytorch_prototype.clebsch_gordan import ClebschGordan

from equistore import TensorBlock, TensorMap
from rascaline import SphericalExpansion

# torch.set_default_dtype(torch.float64)
# torch.manual_seed(0)

METHANE_PATH = 'methane.extxyz'
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCALMOL = 627.5

n_test = 1000
n_train = 5000
n_validation = int(n_train/10)

test_slice = str(0) + ":" + str(n_test)
train_slice = str(n_test) + ":" + str(n_test+n_train)
validation_slice = str(n_test+n_train) + ":" + str(n_test+n_train+n_validation)

L_MAX = 3
clebsch = ClebschGordan(L_MAX)

hypers_spherical_expansion = {
    "cutoff": 6.5,
    "max_radial": 22,
    "max_angular": L_MAX,
    "atomic_gaussian_width": 0.2,
    "center_atom_weight": 0.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "Step": {},
    },
    # "radial_scaling":  {"Willatt2018": { "scale": 2.0, "rate": 0.8, "exponent": 2}},
}

BATCH_SIZE = 10000
DEVICE = 'cuda'

calculator = SphericalExpansion(**hypers_spherical_expansion)

def initialize_wigner_single_l(first, second):
    first_b_size, first_m_size = first.shape[0], first.shape[1]
    second_b_size, second_m_size = second.shape[0], second.shape[1]
    first = first.reshape([-1, first.shape[2]])
    second = second.reshape([-1, second.shape[2]])
    result = torch.matmul(first, second.transpose(0, 1))
    result = result.reshape(first_b_size, first_m_size, second_b_size, second_m_size)
    return result.transpose(1, 2)

def initialize_wigner_single_species_batched(first, second, center_species, idx_1, idx_2):
    idx_1_begin, idx_1_end = idx_1
    idx_2_begin, idx_2_end = idx_2
    lmax = np.max(first.keys["spherical_harmonics_l"])
    result = {}
    for l in range(lmax+1):
        result[str(l) + "_" + str(1)] = initialize_wigner_single_l(
                first.block(spherical_harmonics_l=l, species_center=center_species).values[idx_1_begin:idx_1_end, :, :], 
                second.block(spherical_harmonics_l=l, species_center=center_species).values[idx_2_begin:idx_2_end, :, :]
                )
    return result

class WignerKernel(torch.nn.Module):
    def __init__(self, clebsch, lambda_max, num_iterations):
        super(WignerKernel, self).__init__()
        main = [WignerCombiningUnrolled(clebsch.precomputed_, lambda_max, algorithm = 'vectorized') 
                for _ in range(num_iterations)]
        self.main = nn.ModuleList(main)
        self.last = WignerCombiningUnrolled(clebsch.precomputed_, 0, algorithm = 'vectorized')
    
    #############################
    # Need to try what happens when combining e.g. nu = 2 and nu = 2 kernels
    #############################
            
    def forward(self, X):
        result = []
        wig_now = X
        result.append(wig_now['0_1'][:, 0, 0, None])
        for block in self.main:
            wig_now = block(wig_now, X)
            result.append(wig_now['0_1'][:, 0, 0, None])
        wig_now = self.last(wig_now, X)
        result.append(wig_now['0_1'][:, 0, 0, None])
        result = torch.cat(result, dim = -1)
        return result

def compute_kernel(model, first, second, batch_size = 1000, device = 'cpu'):
    all_species = np.unique(np.concatenate([first.keys["species_center"], second.keys["species_center"]]))

    n_first = len(np.unique(first.block(0).samples["structure"]))
    n_second = len(np.unique(second.block(0).samples["structure"]))
    nu_max = len(model.main) + 2  # nu = 1 and last iteration not counted in model.main
    wigner_invariants = torch.zeros((n_first, n_second, nu_max))
    batch_size_each = int(np.sqrt(batch_size))  # A batch size for each of the two tensor maps involved.
  
    for center_species in all_species:
        # if center_species == 1: continue  # UNCOMMENT FOR C-ONLY VERSION
        structures_first = first.block(spherical_harmonics_l=0, species_center=center_species).samples["structure"]
        structures_second = second.block(spherical_harmonics_l=0, species_center=center_species).samples["structure"]
        len_first = structures_first.shape[0]
        len_second = structures_second.shape[0]

        # Batched calculation, starting from nu = 1 kernel initialization:
        for idx_1_begin in tqdm.tqdm(range(0, len_first, batch_size_each)):
            idx_1_end = min(idx_1_begin+batch_size_each, len_first)
            dimension_1 = idx_1_end - idx_1_begin

            for idx_2_begin in range(0, len_second, batch_size_each):
                idx_2_end = min(idx_2_begin+batch_size_each, len_second)
                dimension_2 = idx_2_end - idx_2_begin

                wigner_c = initialize_wigner_single_species_batched(
                    first,
                    second, 
                    center_species,
                    (idx_1_begin, idx_1_end),
                    (idx_2_begin, idx_2_end),
                    )            

                for key in wigner_c.keys():
                    wigner_c[key] = wigner_c[key].reshape([dimension_1*dimension_2, wigner_c[key].shape[2], wigner_c[key].shape[3]])               
                now = {}
                for key in wigner_c.keys():
                    now[key] = wigner_c[key].to(device)
                result_now = model(now).to('cpu')
                result_now = result_now.reshape([dimension_1, dimension_2, nu_max])

                for i_1 in range(idx_1_begin, idx_1_end):
                    for i_2 in range(idx_2_begin, idx_2_end):
                        wigner_invariants[structures_first[i_1], structures_second[i_2]] += result_now[i_1-idx_1_begin, i_2-idx_2_begin]

    return wigner_invariants

train_structures = ase.io.read(METHANE_PATH, index = train_slice)
validation_structures = ase.io.read(METHANE_PATH, index = validation_slice)
test_structures = ase.io.read(METHANE_PATH, index = test_slice)

def move_to_torch(rust_map: TensorMap) -> TensorMap:
    torch_blocks = []
    for _, block in rust_map:
        torch_block = TensorBlock(
            values=torch.tensor(block.values).to(dtype=torch.get_default_dtype()),
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )
        torch_blocks.append(torch_block)
    return TensorMap(
            keys = rust_map.keys,
            blocks = torch_blocks
            )

print("Calculating expansion coefficients", flush = True)

train_coefs = calculator.compute(train_structures)
train_coefs.keys_to_properties("species_neighbor")
train_coefs = move_to_torch(train_coefs)

validation_coefs = calculator.compute(validation_structures)
validation_coefs.keys_to_properties("species_neighbor")
validation_coefs = move_to_torch(validation_coefs)

test_coefs = calculator.compute(test_structures)
test_coefs.keys_to_properties("species_neighbor")
test_coefs = move_to_torch(test_coefs)

print("Expansion coefficients done", flush = True)

'''
L2_mean = get_L2_mean(train_coefs)
#print(L2_mean)
for key in train_coefs.keys():
    train_coefs[key] /= np.sqrt(L2_mean)
    validation_coefs[key] /= np.sqrt(L2_mean)
    test_coefs[key] /= np.sqrt(L2_mean)
'''

model = WignerKernel(clebsch, L_MAX, 2)
model = model.to(DEVICE)

train_train_kernel = compute_kernel(model, train_coefs, train_coefs, batch_size = BATCH_SIZE, device = DEVICE)
train_validation_kernel = compute_kernel(model, train_coefs, validation_coefs, batch_size = BATCH_SIZE, device = DEVICE)
train_test_kernel = compute_kernel(model, train_coefs, test_coefs, batch_size = BATCH_SIZE, device = DEVICE)

train_train_kernel = train_train_kernel.data.cpu()
train_validation_kernel = train_validation_kernel.data.cpu()
train_test_kernel = train_test_kernel.data.cpu()

for i in range(10):
    print(train_train_kernel[i, i])

def get_rmse(first, second):
    return torch.sqrt(torch.mean((first - second)**2))

def get_sse(first, second):
    return torch.sum((first - second)**2)

train_energies = [structure.info['energy'] for structure in train_structures]
train_energies = torch.tensor(train_energies, dtype = torch.get_default_dtype()) * HARTREE_TO_KCALMOL

validation_energies = [structure.info['energy'] for structure in validation_structures]
validation_energies = torch.tensor(validation_energies, dtype = torch.get_default_dtype()) * HARTREE_TO_KCALMOL

test_energies = [structure.info['energy'] for structure in test_structures]
test_energies = torch.tensor(test_energies, dtype = torch.get_default_dtype()) * HARTREE_TO_KCALMOL

mean_e = torch.mean(train_energies)
train_energies -= mean_e
validation_energies -= mean_e
test_energies -= mean_e

alpha = 0.0
c = torch.linalg.solve(train_train_kernel[:, :, -1] + alpha * torch.eye(train_train_kernel.shape[0]), train_energies)
validation_predictions = train_validation_kernel[:, :, -1].T @ c
best_alpha = 0.0
print(f"Validation set RMSE (before kernel mixing): {get_rmse(validation_predictions, validation_energies).item()}")

c = torch.linalg.solve(train_train_kernel[:, :, -1] + best_alpha * torch.eye(train_train_kernel.shape[0]), train_energies)
test_predictions = train_test_kernel[:, :, -1].T @ c
print("Test set RMSE (before kernel mixing): ", get_rmse(test_predictions, test_energies).item())

class ValidationCycle(torch.nn.Module):
    # Evaluates the model on the validation set so that derivatives 
    # of an arbitrary loss with respect to the continuous
    # hyperparameters can be used to minimize the validation loss.

    def __init__(self):
        super().__init__()

        # Kernel regularization:
        self.sigma_exponent = torch.nn.Parameter(
            torch.tensor([-6], dtype = torch.get_default_dtype())
            )

        # Coefficients for mixing kernels of different body-orders:
        self.coefficients = torch.nn.Parameter(torch.zeros((3,)))
        self.nu_zero_coefficient = torch.nn.Parameter(torch.zeros((1,)))

    def forward(self, K_train, y_train, K_val):
        coefficients = torch.cat([self.coefficients, torch.ones((1,))], dim = -1)
        sigma = torch.exp(self.sigma_exponent)
        n_train = K_train.shape[0] 
        n_val = K_val.shape[1]
        c = torch.linalg.solve(
        self.nu_zero_coefficient * torch.ones((n_train, n_train)) +  # very dirty nu = 0 kernel
        K_train @ coefficients +  # nu = 1, ..., 4 kernels
        sigma * torch.eye(n_train)  # regularization
        , 
        y_train)
        y_val_predictions = (
            self.nu_zero_coefficient * torch.ones((n_val, n_train)) + 
            (K_val @ coefficients).T) @ c

        return y_val_predictions

validation_cycle = ValidationCycle()
optimizer = torch.optim.Adam(validation_cycle.parameters(), lr = 1e-2)

print("Beginning hyperparameter optimization")
best_rmse = 1e20
for i in range(10000):
    optimizer.zero_grad()
    validation_predictions = validation_cycle(train_train_kernel, train_energies, train_validation_kernel)

    validation_rmse = get_rmse(validation_predictions, validation_energies).item()
    if validation_rmse < best_rmse: 
        best_rmse = validation_rmse
        best_nu_zero_coefficient = copy.deepcopy(validation_cycle.nu_zero_coefficient.data)
        best_coefficients = copy.deepcopy(validation_cycle.coefficients.data)
        best_sigma = copy.deepcopy(torch.exp(validation_cycle.sigma_exponent.data))

    validation_loss = get_sse(validation_predictions, validation_energies)
    validation_loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(best_rmse, best_nu_zero_coefficient, best_coefficients, best_sigma, flush = True)

best_coefficients = torch.cat([best_coefficients, torch.ones((1,))], dim = -1)
n_train = train_train_kernel.shape[0] 
n_val = train_validation_kernel.shape[1]
n_test = train_test_kernel.shape[1]

c = torch.linalg.solve(
    best_nu_zero_coefficient * torch.ones((n_train, n_train)) +  # very dirty nu = 0 kernel
    train_train_kernel @ best_coefficients +  # nu = 1, ..., 4 kernels
    best_sigma * torch.eye(n_train)  # regularization
    , 
    train_energies)

test_predictions = (
    best_nu_zero_coefficient * torch.ones((n_test, n_train)) + 
    (train_test_kernel @ best_coefficients).T) @ c

print(f"Test set RMSE (after kernel mixing): {get_rmse(test_predictions, test_energies).item()}")
print()
print(n_train, get_rmse(test_predictions, test_energies).item())