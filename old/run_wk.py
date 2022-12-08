import torch
import copy
import numpy as np
import scipy as sp

from pytorch_prototype.clebsch_gordan import ClebschGordan

from equistore import Labels, TensorBlock, TensorMap
from rascaline import SphericalExpansion

from wigner_kernels import WignerKernel, compute_kernel
from dataset_processing import get_dataset_slice
from error_measures import get_sse, get_rmse, get_mae
from validation import ValidationCycle

# torch.set_default_dtype(torch.float64)
# torch.manual_seed(1234)
RANDOM_SEED = 2
np.random.seed(RANDOM_SEED)
print(f"Random seed: {RANDOM_SEED}", flush = True)

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCALMOL = 627.5
EV_TO_KCALMOL = HARTREE_TO_KCALMOL/HARTREE_TO_EV

DATASET_PATH = 'datasets/qm9.xyz'
TARGET_KEY = "U0" # "elec. Free Energy [eV]" # "U0"
CONVERSION_FACTOR = HARTREE_TO_KCALMOL

print("SOTA")

n_test = 500
n_train = 100

n_validation_splits = 10
assert n_train % n_validation_splits == 0
n_validation = n_train // n_validation_splits
n_train_sub = n_train - n_validation

test_slice = str(0) + ":" + str(n_test)
train_slice = str(n_test) + ":" + str(n_test+n_train)

BATCH_SIZE = 10000
DEVICE = 'cuda'
NU_MAX = 4 # 4!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
L_MAX = 3
clebsch = ClebschGordan(L_MAX)

# Spherical expansion and composition

def get_composition_features(frames, all_species):
    species_dict = {s: i for i, s in enumerate(all_species)}
    data = torch.zeros((len(frames), len(species_dict)))
    for i, f in enumerate(frames):
        for s in f.numbers:
            data[i, species_dict[s]] += 1
    properties = Labels(
        names=["atomic_number"],
        values=np.array(list(species_dict.keys()), dtype=np.int32).reshape(
            -1, 1
        ),
    )

    frames_i = np.arange(len(frames), dtype=np.int32).reshape(-1, 1)
    samples = Labels(names=["structure"], values=frames_i)

    block = TensorBlock(
        values=data, samples=samples, components=[], properties=properties
    )
    composition = TensorMap(Labels.single(), blocks=[block])
    return composition.block().values

if "methane" in DATASET_PATH:
    hypers_spherical_expansion = {
        "cutoff": 6.5,
        "max_radial": 22,
        "max_angular": L_MAX,
        "atomic_gaussian_width": 0.4,
        "center_atom_weight": 0.0,
        "radial_basis": {"Gto": {"spline_accuracy": 1e-8}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
        "radial_scaling":  {"Willatt2018": { "scale": 2.0, "rate": 2.0, "exponent": 6}}, 
        # A radial scaling (but much smoother) could also help for methane
    }
else:
    hypers_spherical_expansion = {
        "cutoff": 4.5,
        "max_radial": 22,
        "max_angular": L_MAX,
        "atomic_gaussian_width": 0.15,
        "center_atom_weight": 0.0,
        "radial_basis": {"Gto": {"spline_accuracy": 1e-8}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
        "radial_scaling":  {"Willatt2018": { "scale": 1.5, "rate": 2.0, "exponent": 6}},
    }

calculator = SphericalExpansion(**hypers_spherical_expansion)

train_structures = get_dataset_slice(DATASET_PATH, train_slice)
test_structures = get_dataset_slice(DATASET_PATH, test_slice)

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
train_coefs = move_to_torch(train_coefs)

test_coefs = calculator.compute(test_structures)
test_coefs = move_to_torch(test_coefs)

all_species = np.unique(np.concatenate([train_coefs.keys["species_center"], test_coefs.keys["species_center"]]))

all_neighbor_species = Labels(
        names=["species_neighbor"],
        values=np.array(all_species, dtype=np.int32).reshape(-1, 1),
    )

train_coefs.keys_to_properties(all_neighbor_species)
test_coefs.keys_to_properties(all_neighbor_species)

print("Expansion coefficients done", flush = True)

'''
L2_mean = get_L2_mean(train_coefs)
#print(L2_mean)
for key in train_coefs.keys():
    train_coefs[key] /= np.sqrt(L2_mean)
    test_coefs[key] /= np.sqrt(L2_mean)
'''

# Kernel computation

model = WignerKernel(clebsch, L_MAX, NU_MAX-2)
model = model.to(DEVICE)

print("Computing train-train-kernels", flush = True)
train_train_kernel = compute_kernel(model, train_coefs, train_coefs, batch_size = BATCH_SIZE, device = DEVICE)
print("Computing train-test-kernels", flush = True)
train_test_kernel = compute_kernel(model, train_coefs, test_coefs, batch_size = BATCH_SIZE, device = DEVICE)

train_train_kernel = train_train_kernel.data.cpu()
train_test_kernel = train_test_kernel.data.cpu()

print("Calculating composition features", flush = True)
X_train = get_composition_features(train_structures, all_species)
X_test = get_composition_features(test_structures, all_species)
print("Composition features done", flush = True)

train_energies = [structure.info[TARGET_KEY] for structure in train_structures]
train_energies = torch.tensor(train_energies, dtype = torch.get_default_dtype()) * CONVERSION_FACTOR

test_energies = [structure.info[TARGET_KEY] for structure in test_structures]
test_energies = torch.tensor(test_energies, dtype = torch.get_default_dtype()) * CONVERSION_FACTOR


# nu = 0 contribution

if "methane" in DATASET_PATH:
    mean_train_energy = torch.mean(train_energies)
    train_energies -= mean_train_energy
    test_energies -= mean_train_energy
else:
    c_comp = torch.linalg.solve(X_train.T @ X_train, X_train.T @ train_energies)
    train_energies -= X_train @ c_comp
    test_energies -= X_test @ c_comp

train_train_nu0_kernel = X_train @ X_train.T
train_test_nu0_kernel = X_train @ X_test.T

train_train_kernel = torch.concat([train_train_nu0_kernel.unsqueeze(dim = 2), train_train_kernel], dim = 2) 
train_test_kernel = torch.concat([train_test_nu0_kernel.unsqueeze(dim = 2), train_test_kernel], dim = 2)


'''
    # Kernel regularization estimation (useful if gradient descent used):
    i_validation_split = 0
    index_validation_start = i_validation_split*n_validation
    index_validation_stop = index_validation_start + n_validation

    K_train_sub = torch.empty((n_train_sub, n_train_sub, NU_MAX+1))
    K_train_sub[:index_validation_start, :index_validation_start , :] = train_train_kernel[:index_validation_start, :index_validation_start , :]
    if i_validation_split != n_validation_splits - 1:
        K_train_sub[:index_validation_start, index_validation_start: , :] = train_train_kernel[:index_validation_start, index_validation_stop: , :]
        K_train_sub[index_validation_start:, :index_validation_start , :] = train_train_kernel[index_validation_stop:, :index_validation_start , :]
        K_train_sub[index_validation_start:, index_validation_start: , :] = train_train_kernel[index_validation_stop:, index_validation_stop: , :]
    y_train_sub = train_energies[:index_validation_start]
    if i_validation_split != n_validation_splits - 1:
        y_train_sub = torch.concat([y_train_sub, train_energies[index_validation_stop:]])

    K_validation = train_train_kernel[index_validation_start:index_validation_stop, :index_validation_start, :]
    if i_validation_split != n_validation_splits - 1:
        K_validation = torch.concat([K_validation, train_train_kernel[index_validation_start:index_validation_stop, index_validation_stop:, :]], dim = 1)
    y_validation = train_energies[index_validation_start:index_validation_stop] 
    ##############
    '''

'''
    rmse_list = []
    alpha_exp_list = np.linspace(-3, 5, 100)
    for alpha_exp in alpha_exp_list:
        c_comp = torch.linalg.solve(
            K_train_sub @ torch.concat([torch.ones((1,)), torch.zeros((NU_MAX,))]) +
            10.0**alpha_exp * torch.eye(n_train_sub), 
            y_train_sub)

        validation_predictions = K_validation @ torch.concat([torch.ones((1,)), torch.zeros((NU_MAX,))]) @ c_comp
        rmse_list.append(get_rmse(validation_predictions, y_validation).item())
    alpha_exp_initial_guess = alpha_exp_list[np.argmin(rmse_list)]
    print("Result of preliminary sigma optimization: ", alpha_exp_initial_guess, min(rmse_list)) 
    '''


# Validation cycles to optimize kernel regularization and kernel mixing

validation_cycle = ValidationCycle(nu_max = NU_MAX, alpha_exp_initial_guess = -5) # alpha_exp_initial_guess)
optimizer = torch.optim.Adam(validation_cycle.parameters(), lr = 1e-3)

print("Beginning hyperparameter optimization")

'''
# Gradient-based version:
best_rmse = 1e20
for i in range(1000):
    optimizer.zero_grad()
    validation_rmse = 0.0

    for i_validation_split in range(n_validation_splits):
        index_validation_start = i_validation_split*n_validation
        index_validation_stop = index_validation_start + n_validation

        K_train_sub = torch.empty((n_train_sub, n_train_sub, NU_MAX+1))
        K_train_sub[:index_validation_start, :index_validation_start , :] = train_train_kernel[:index_validation_start, :index_validation_start , :]
        if i_validation_split != n_validation_splits - 1:
            K_train_sub[:index_validation_start, index_validation_start: , :] = train_train_kernel[:index_validation_start, index_validation_stop: , :]
            K_train_sub[index_validation_start:, :index_validation_start , :] = train_train_kernel[index_validation_stop:, :index_validation_start , :]
            K_train_sub[index_validation_start:, index_validation_start: , :] = train_train_kernel[index_validation_stop:, index_validation_stop: , :]
        y_train_sub = train_energies[:index_validation_start]
        if i_validation_split != n_validation_splits - 1:
            y_train_sub = torch.concat([y_train_sub, train_energies[index_validation_stop:]])

        K_validation = train_train_kernel[index_validation_start:index_validation_stop, :index_validation_start, :]
        if i_validation_split != n_validation_splits - 1:
            K_validation = torch.concat([K_validation, train_train_kernel[index_validation_start:index_validation_stop, index_validation_stop:, :]], dim = 1)
        y_validation = train_energies[index_validation_start:index_validation_stop] 

        validation_predictions = validation_cycle(K_train_sub, y_train_sub, K_validation)

        with torch.no_grad():
            validation_rmse += get_sse(validation_predictions, y_validation).item()

        validation_loss = get_sse(validation_predictions, y_validation)
        validation_loss.backward()
    
    validation_rmse = np.sqrt(validation_rmse/n_train)
    if validation_rmse < best_rmse: 
            best_rmse = validation_rmse
            best_coefficients = copy.deepcopy(validation_cycle.coefficients.weight)
            best_sigma = copy.deepcopy(torch.exp(validation_cycle.sigma_exponent.data*np.log(10.0)))
    optimizer.step()

    if i % 100 == 0:
        print(best_rmse, best_coefficients, best_sigma, flush = True)

'''
def validation_loss_for_global_optimization(x):

    validation_cycle.sigma_exponent = torch.nn.Parameter(
            torch.tensor(x[-1], dtype = torch.get_default_dtype())
            )
    validation_cycle.coefficients.weight = torch.nn.Parameter(torch.tensor(x[0:NU_MAX+1], dtype = torch.get_default_dtype()).reshape(1, -1))

    validation_loss = 0.0
    for i_validation_split in range(n_validation_splits):
        index_validation_start = i_validation_split*n_validation
        index_validation_stop = index_validation_start + n_validation

        K_train_sub = torch.empty((n_train_sub, n_train_sub, NU_MAX+1))
        K_train_sub[:index_validation_start, :index_validation_start , :] = train_train_kernel[:index_validation_start, :index_validation_start , :]
        if i_validation_split != n_validation_splits - 1:
            K_train_sub[:index_validation_start, index_validation_start: , :] = train_train_kernel[:index_validation_start, index_validation_stop: , :]
            K_train_sub[index_validation_start:, :index_validation_start , :] = train_train_kernel[index_validation_stop:, :index_validation_start , :]
            K_train_sub[index_validation_start:, index_validation_start: , :] = train_train_kernel[index_validation_stop:, index_validation_stop: , :]
        y_train_sub = train_energies[:index_validation_start]
        if i_validation_split != n_validation_splits - 1:
            y_train_sub = torch.concat([y_train_sub, train_energies[index_validation_stop:]])

        K_validation = train_train_kernel[index_validation_start:index_validation_stop, :index_validation_start, :]
        if i_validation_split != n_validation_splits - 1:
            K_validation = torch.concat([K_validation, train_train_kernel[index_validation_start:index_validation_stop, index_validation_stop:, :]], dim = 1)
        y_validation = train_energies[index_validation_start:index_validation_stop] 

        with torch.no_grad():
            validation_predictions = validation_cycle(K_train_sub, y_train_sub, K_validation)
            validation_loss += get_sse(validation_predictions, y_validation).item()
    '''
    with open("log.txt", "a") as out:
        out.write(str(np.sqrt(validation_loss/n_train)) + "\n")
        out.flush()
    '''
    print(validation_loss)
    return validation_loss

bounds = [(-1.0, 1.0) for i in range(NU_MAX+1)]
bounds.append((-6.0, 2.0))  # -8 2
x0 = [0.0] * (NU_MAX+1)
x0.append(-5.0)  # -5
x0 = np.array(x0)
solution = sp.optimize.dual_annealing(validation_loss_for_global_optimization, bounds = bounds, x0 = x0, no_local_search = True)
print(solution.x)
print(np.sqrt(solution.fun/n_train)) # n_train

best_coefficients = torch.tensor(solution.x[0:NU_MAX+1], dtype = torch.get_default_dtype())
best_sigma = np.exp(solution.x[-1]*np.log(10.0))


c = torch.linalg.solve(
    train_train_kernel @ best_coefficients +  # nu = 1, ..., 4 kernels
    best_sigma * torch.eye(n_train)  # regularization
    , 
    train_energies)

test_predictions = (train_test_kernel @ best_coefficients).T @ c
print(f"Test set RMSE (after kernel mixing): {get_rmse(test_predictions, test_energies).item()}")

print()
print("Final result (test MAE):")
print(n_train, get_mae(test_predictions, test_energies).item())

'''
# Version for gradient-based local optimization
c = torch.linalg.solve(
    train_train_kernel @ best_coefficients.squeeze(dim = 0) +  # nu = 1, ..., 4 kernels
    best_sigma * torch.eye(n_train)  # regularization
    , 
    train_energies)

test_predictions = (train_test_kernel @ best_coefficients.squeeze(dim = 0)).T @ c
print(f"Test set RMSE (after kernel mixing): {get_rmse(test_predictions, test_energies).item()}")

print()
print("Final result (test MAE):")
print(n_train, get_mae(test_predictions, test_energies).item())
'''