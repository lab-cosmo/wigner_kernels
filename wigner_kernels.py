import torch
#torch.set_default_dtype(torch.float64)
import numpy as np
import ase.io

from pytorch_prototype.code_pytorch import *
from pytorch_prototype.utilities import *
from pytorch_prototype.clebsch_gordan import ClebschGordan

from rascal.neighbourlist.structure_manager import mask_center_atoms_by_species
from rascal.utils import (get_radial_basis_covariance, get_radial_basis_pca, 
                          get_radial_basis_projections, get_optimal_radial_basis_hypers)

METHANE_PATH = 'methane.extxyz'
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCALMOL = 627.5

train_slice = '0:800'
validation_slice = '800:900'
test_slice = '900:1000'

L_MAX = 6
clebsch = ClebschGordan(L_MAX)

HYPERS_INITIAL = {
    'interaction_cutoff': 6.3,
    'max_radial': 20,
    'max_angular': L_MAX,
    'gaussian_sigma_type': 'Constant',
    'gaussian_sigma_constant': 0.2,
    'cutoff_smooth_width': 0.3,
    'radial_basis': 'DVR'
}

BATCH_SIZE = 2000
DEVICE = 'cuda'



structures = process_structures(ase.io.read(METHANE_PATH, index = train_slice))
HYPERS = get_optimal_radial_basis_hypers(HYPERS_INITIAL,
                                           structures,
                                           expanded_max_radial=100)

def initialize_wigner_single(first, second):
    first_b_size, first_m_size = first.shape[0], first.shape[2]
    second_b_size, second_m_size = second.shape[0], second.shape[2]
    first = first.transpose(1, 2)
    second = second.transpose(1, 2)
    first = first.reshape([-1, first.shape[2]])
    second = second.reshape([-1, second.shape[2]])
    result = torch.matmul(first, second.transpose(0, 1))
    result = result.reshape(first_b_size, first_m_size, second_b_size, second_m_size)
    return result.transpose(1, 2)

def initialize_wigner(first, second):
    result = {}
    for key in first.keys():
        result[str(key) + "_" + str(1)] = initialize_wigner_single(first[key], second[key])
    return result

class WignerKernel(torch.nn.Module):
    def __init__(self, clebsch, lambda_max, num_iterations):
        super(WignerKernel, self).__init__()
        main = [WignerCombiningUnrolled(clebsch.precomputed_, lambda_max, algorithm = 'vectorized') 
                for _ in range(num_iterations)]
        self.main = nn.ModuleList(main)
        self.last = WignerCombiningUnrolled(clebsch.precomputed_, 0, algorithm = 'vectorized')
       
            
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
    wigner = initialize_wigner(first, second)
   
    for key in wigner.keys():
        initial_shape = [wigner[key].shape[0], wigner[key].shape[1]]
        wigner[key] = wigner[key].reshape([-1, wigner[key].shape[2], wigner[key].shape[3]])
    ''' for key in wigner.keys():
        print(key, wigner[key].shape)'''
    
    total = initial_shape[0] * initial_shape[1]
    result = []
    #print(total, batch_size)
    #print(initial_shape)
    for ind in tqdm.tqdm(range(0, total, batch_size)):
        now = {}
        for key in wigner.keys():
            now[key] = wigner[key][ind : ind + batch_size].to(device)
        result_now = model(now).to('cpu')
        result.append(result_now)
        
        
    result = torch.cat(result, dim = 0)
    return result.reshape(initial_shape + [-1])

train_structures = ase.io.read(METHANE_PATH, index = train_slice)
validation_structures = ase.io.read(METHANE_PATH, index = validation_slice)
test_structures = ase.io.read(METHANE_PATH, index = test_slice)
all_species = get_all_species(train_structures + validation_structures + test_structures)

for struc in train_structures:
    mask_center_atoms_by_species(struc, species_select=["C"])
for struc in validation_structures:
    mask_center_atoms_by_species(struc, species_select=["C"])
for struc in test_structures:
    mask_center_atoms_by_species(struc, species_select=["C"])


train_coefs = get_coefs(train_structures, HYPERS, all_species)
validation_coefs = get_coefs(validation_structures, HYPERS, all_species)
test_coefs = get_coefs(test_structures, HYPERS, all_species)

L2_mean = get_L2_mean(train_coefs)
#print(L2_mean)
for key in train_coefs.keys():
    train_coefs[key] /= np.sqrt(L2_mean)
    validation_coefs[key] /= np.sqrt(L2_mean)
    test_coefs[key] /= np.sqrt(L2_mean)

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

'''
print(train_train_kernel.shape)
print(train_validation_kernel.shape)
print(train_test_kernel.shape)
train_train_kernel = train_train_kernel[:, :, -1]
train_validation_kernel = train_validation_kernel[:, :, -1]
train_test_kernel = train_test_kernel[:, :, -1]
'''

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

alpha_grid = np.logspace(5, -15, 100)
rmse = []
for alpha in tqdm.tqdm(alpha_grid):
    c = torch.linalg.solve(train_train_kernel[:, :, -1] + alpha * torch.eye(train_train_kernel.shape[0]), train_energies)
    validation_predictions = train_validation_kernel[:, :, -1].T @ c
    rmse.append(get_rmse(validation_predictions, validation_energies).item())

best_alpha = alpha_grid[np.argmin(rmse)]
print(f"Validation set RMSE (before kernel mixing): {np.min(rmse)}")

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
        self.sigma = (
            # torch.nn.Parameter(
            torch.tensor([best_alpha], dtype = torch.get_default_dtype())
            )

        # Coefficients for mixing kernels of different body-orders:
        self.coefficients = torch.nn.Linear(4, 1, bias = False)
        self.coefficients.weight = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))

    def forward(self, K_train, y_train, K_val):
        n_train = K_train.shape[0] 
        n_val = K_val.shape[1]
        c = torch.linalg.solve(
        # torch.ones((n_train, n_train)) +  # very dirty nu = 0 kernel
        self.coefficients(K_train).squeeze(dim = -1) +  # nu = 1, ..., 4 kernels
        self.sigma * torch.eye(n_train)  # regularization
        , 
        y_train)
        y_val_predictions = (
            # torch.ones((n_val, n_train)) + 
            self.coefficients(K_val).squeeze(dim = -1).T) @ c

        return y_val_predictions

validation_cycle = ValidationCycle()
optimizer = torch.optim.Adam(validation_cycle.parameters(), lr = 1e-2)

for i in range(10000):
    optimizer.zero_grad()
    validation_predictions = validation_cycle(train_train_kernel, train_energies, train_validation_kernel)
    validation_loss = get_sse(validation_predictions, validation_energies)
    validation_loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(validation_cycle.sigma.item(), validation_cycle.coefficients.weight, get_rmse(validation_predictions, validation_energies).item())

best_coefficients = validation_cycle.coefficients.weight

c = torch.linalg.solve((train_train_kernel @ best_coefficients.T).squeeze(dim = -1) + best_alpha * torch.eye(train_train_kernel.shape[0]), train_energies)
test_predictions = (train_test_kernel @ best_coefficients.T).squeeze(dim = -1).T @ c
print("Test set RMSE (after kernel mixing): ", get_rmse(test_predictions, test_energies).item())


