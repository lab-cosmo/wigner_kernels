import torch
import copy
import numpy as np
import ase.io

from pytorch_prototype.clebsch_gordan import ClebschGordan

from equistore import Labels, TensorBlock, TensorMap
from rascaline import SphericalExpansion

from wigner_kernels import WignerKernel, compute_kernel
from error_measures import get_sse, get_rmse, get_mae
from validation import ValidationCycle

# torch.set_default_dtype(torch.float64)
# torch.manual_seed(0)

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCALMOL = 627.5

DATASET_PATH = 'datasets/qm9.xyz'
TARGET_KEY = "U0"
CONVERSION_FACTOR = HARTREE_TO_KCALMOL

n_test = 100
n_train = 100
n_validation = int(n_train/10)

test_slice = str(0) + ":" + str(n_test)
test_slice = str(2000) + ":" + str(2000+n_test)
train_slice = str(n_test) + ":" + str(n_test+n_train)
validation_slice = str(n_test+n_train) + ":" + str(n_test+n_train+n_validation)

BATCH_SIZE = 10000
DEVICE = 'cuda'
NU_MAX = 4
L_MAX = 4
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

hypers_spherical_expansion = {
    "cutoff": 4.5,
    "max_radial": 22,
    "max_angular": L_MAX,
    "atomic_gaussian_width": 0.2,
    "center_atom_weight": 0.0,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_scaling":  {"Willatt2018": { "scale": 2.0, "rate": 0.8, "exponent": 2}},
}

calculator = SphericalExpansion(**hypers_spherical_expansion)

train_structures = ase.io.read(DATASET_PATH, index = train_slice)
validation_structures = ase.io.read(DATASET_PATH, index = validation_slice)
test_structures = ase.io.read(DATASET_PATH, index = test_slice)

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

validation_coefs = calculator.compute(validation_structures)
validation_coefs = move_to_torch(validation_coefs)

test_coefs = calculator.compute(test_structures)
test_coefs = move_to_torch(test_coefs)

all_species = np.unique(np.concatenate([train_coefs.keys["species_center"], validation_coefs.keys["species_center"], test_coefs.keys["species_center"]]))

all_neighbor_species = Labels(
        names=["species_neighbor"],
        values=np.array(all_species, dtype=np.int32).reshape(-1, 1),
    )

train_coefs.keys_to_properties(all_neighbor_species)
validation_coefs.keys_to_properties(all_neighbor_species)
test_coefs.keys_to_properties(all_neighbor_species)

print("Expansion coefficients done", flush = True)

'''
L2_mean = get_L2_mean(train_coefs)
#print(L2_mean)
for key in train_coefs.keys():
    train_coefs[key] /= np.sqrt(L2_mean)
    validation_coefs[key] /= np.sqrt(L2_mean)
    test_coefs[key] /= np.sqrt(L2_mean)
'''

# Kernel computation

model = WignerKernel(clebsch, L_MAX, NU_MAX-2)
model = model.to(DEVICE)

print("Computing train-train-kernels", flush = True)
train_train_kernel = compute_kernel(model, train_coefs, train_coefs, batch_size = BATCH_SIZE, device = DEVICE)
print("Computing train-validation-kernels", flush = True)
train_validation_kernel = compute_kernel(model, train_coefs, validation_coefs, batch_size = BATCH_SIZE, device = DEVICE)
print("Computing train-test-kernels", flush = True)
train_test_kernel = compute_kernel(model, train_coefs, test_coefs, batch_size = BATCH_SIZE, device = DEVICE)

train_train_kernel = train_train_kernel.data.cpu()
train_validation_kernel = train_validation_kernel.data.cpu()
train_test_kernel = train_test_kernel.data.cpu()

print("Calculating composition features", flush = True)
X_train = get_composition_features(train_structures, all_species)
X_validation = get_composition_features(validation_structures, all_species)
X_test = get_composition_features(test_structures, all_species)
print("Composition features done", flush = True)

train_energies = [structure.info[TARGET_KEY] for structure in train_structures]
train_energies = torch.tensor(train_energies, dtype = torch.get_default_dtype()) * CONVERSION_FACTOR

validation_energies = [structure.info[TARGET_KEY] for structure in validation_structures]
validation_energies = torch.tensor(validation_energies, dtype = torch.get_default_dtype()) * CONVERSION_FACTOR

test_energies = [structure.info[TARGET_KEY] for structure in test_structures]
test_energies = torch.tensor(test_energies, dtype = torch.get_default_dtype()) * CONVERSION_FACTOR


# nu = 0 contribution

nu0_method = "nu_0_kernel"
if nu0_method == "nu_0_kernel":
    train_train_nu0_kernel = X_train @ X_train.T
    train_validation_nu0_kernel = X_train @ X_validation.T
    train_test_nu0_kernel = X_train @ X_test.T

    rmse_list = []
    alpha_exp_list = np.linspace(-5, 5, 100)
    for alpha_exp in alpha_exp_list:
        c_comp = torch.linalg.solve(
            train_train_nu0_kernel +
            10.0**alpha_exp * torch.eye(n_train), 
            train_energies)

        validation_predictions = train_validation_nu0_kernel.T @ c_comp
        rmse_list.append(get_rmse(validation_predictions, validation_energies).item())
    alpha_exp_initial_guess = alpha_exp_list[np.argmin(rmse_list)]
    print("Result of preliminary sigma optimization: ", alpha_exp_initial_guess, min(rmse_list))

    train_train_kernel = torch.concat([train_train_nu0_kernel.unsqueeze(dim = 2), train_train_kernel], dim = -1) 
    train_validation_kernel = torch.concat([train_validation_nu0_kernel.unsqueeze(dim = 2), train_validation_kernel], dim = -1) 
    train_test_kernel = torch.concat([train_test_nu0_kernel.unsqueeze(dim = 2), train_test_kernel], dim = -1) 
    

# Need to implement linear pathway rigorously.
'''
else if ...
    if "methane" in DATASET_PATH:
        mean_train_energy = torch.mean(train_energies)
        train_energies -= mean_train_energy
        validation_energies -= mean_train_energy
        test_energies -= mean_train_energy
    else:
        c_comp = torch.linalg.solve(X_train.T @ X_train, X_train.T @ train_energies)
        # train_energies -= X_train @ c_comp
        # validation_energies -= X_validation @ c_comp
        # test_energies -= X_test @ c_comp
        test_predictions = X_test @ c_comp
        print(f"Test set RMSE (nu = 0) LINEAR: {get_rmse(test_predictions, test_energies).item()}")
        ...
'''


# Validation cycles to optimize kernel regularization and kernel mixing

validation_cycle = ValidationCycle(nu_max = NU_MAX, alpha_exp_initial_guess = alpha_exp_initial_guess)
optimizer = torch.optim.Adam(validation_cycle.parameters(), lr = 1e-3)

print("Beginning hyperparameter optimization")
best_rmse = 1e20
for i in range(10000):
    optimizer.zero_grad()
    validation_predictions = validation_cycle(train_train_kernel, train_energies, train_validation_kernel)

    validation_rmse = get_rmse(validation_predictions, validation_energies).item()
    if validation_rmse < best_rmse: 
        best_rmse = validation_rmse
        best_coefficients = copy.deepcopy(validation_cycle.coefficients.weight)
        best_sigma = copy.deepcopy(torch.exp(validation_cycle.sigma_exponent.data*np.log(10.0)))

    validation_loss = get_sse(validation_predictions, validation_energies)
    validation_loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(best_rmse, best_coefficients, best_sigma, flush = True)

c = torch.linalg.solve(
    train_train_kernel @ best_coefficients.squeeze(dim = 0) +  # nu = 1, ..., 4 kernels
    best_sigma * torch.eye(n_train)  # regularization
    , 
    train_energies)

test_predictions = (train_test_kernel @ best_coefficients.squeeze(dim = 0)).T @ c
print(f"Test set RMSE (after kernel mixing): {get_rmse(test_predictions, test_energies).item()}")

print()
print("Final result:")
print(n_train, get_mae(test_predictions, test_energies).item())