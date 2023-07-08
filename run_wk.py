import torch
import numpy as np
import scipy as sp
from scipy import optimize
from math import factorial

import rascaline
rascaline._c_lib._get_library()

from equistore import Labels
from rascaline import SphericalExpansion

from utils.clebsch_gordan import ClebschGordan
from utils.wigner_kernels import WignerKernelFullIterations, WignerKernelReducedCost, compute_kernel
from utils.dataset_processing import get_dataset_slice, get_composition_features, move_to_torch
from utils.error_measures import get_sse, get_rmse, get_mae, get_sae
from utils.validation import ValidationCycle
from utils.LE_maths import get_LE_calculator

import argparse
import json

parser = argparse.ArgumentParser(description="?")

parser.add_argument(
    "parameters",
    type=str,
    help="The file containing the parameters. JSON formatted dictionary.",
)

parser.add_argument(
    "n_train",
    type=str,
    help="The file containing the parameters. JSON formatted dictionary.",
)

parser.add_argument(
    "random_seed",
    type=str,
    help="The file containing the parameters. JSON formatted dictionary.",
)

args = parser.parse_args()
parameters = args.parameters

param_dict = json.load(open(parameters, "r"))
DTYPE = param_dict["data type"]
print(f"data type: {DTYPE}")
RANDOM_SEED = int(args.random_seed) # param_dict["random seed"]
print(f"random seed: {RANDOM_SEED}")
BATCH_SIZE = param_dict["batch size"]
print(f"batch size: {BATCH_SIZE}")
CONVERSION_FACTOR = param_dict["conversion factor"]
print(f"conversion factor: {CONVERSION_FACTOR}")
TARGET_KEY = param_dict["target key"]  # TARGET_KEY is "U0" for QM9, "elec. Free Energy [eV]" for gold, "energy" for methane
print(f"target key: {TARGET_KEY}")
DATASET_PATH = param_dict["dataset path"]
print(f"dataset path: {DATASET_PATH}")
n_test = param_dict["n_test"]
print(f"n_test: {n_test}")
n_train = int(args.n_train) # param_dict["n_train"]
print(f"n_train: {n_train}")
r_cut = param_dict["r_cut"]
print(f"r_cut: {r_cut}")
NU_MAX = param_dict["nu_max"]
print(f"nu_max: {NU_MAX}")
L_MAX = param_dict["L_max"]
print(f"l_max: {L_MAX}")
opt_target_name = param_dict["optimization target"] 
print(f"optimization target: {opt_target_name}")
if opt_target_name != "mae" and opt_target_name != "rmse": raise NotImplementedError
C = param_dict["C"]
L_NU = param_dict["L_NU"]
L_R = param_dict["L_R"]
print(f"Density parameters: C={C}, L_NU={L_NU}, L_R={L_R}")
optimization_mode = param_dict["optimization mode"]
if optimization_mode != "full" and optimization_mode != "kernel_exp": raise NotImplementedError
if optimization_mode == "kernel_exp" and L_NU != 0: print("WARNING: Cannot interpret final kernel as a kernel exponential")
cg_mode = param_dict["cg mode"]


if DTYPE == "double": torch.set_default_dtype(torch.float64)
np.random.seed(RANDOM_SEED)

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCALMOL = 627.5
EV_TO_KCALMOL = HARTREE_TO_KCALMOL/HARTREE_TO_EV

conversions = {}
conversions["HARTREE_TO_EV"] = 27.211386245988
conversions["NO_CONVERSION"] = 1.0
conversions["HARTREE_TO_KCAL_MOL"] = 627.509608030593
conversions["EV_TO_KCAL_MOL"] = conversions["HARTREE_TO_KCAL_MOL"]/conversions["HARTREE_TO_EV"]
conversions["KCAL_MOL_TO_MEV"] = 0.0433641153087705*1000.0
conversions["METHANE_FORCE"] = conversions["HARTREE_TO_KCAL_MOL"]/0.529177

CONVERSION_FACTOR = conversions[CONVERSION_FACTOR]

n_validation_splits = 10
assert n_train % n_validation_splits == 0
n_validation = n_train // n_validation_splits
n_train_sub = n_train - n_validation

if "rmd17" in DATASET_PATH:
    train_slice = str(0) + ":" + str(n_train)
    test_slice = str(0) + ":" + str(n_test)
else:
    test_slice = str(0) + ":" + str(n_test)
    train_slice = str(n_test) + ":" + str(n_test+n_train)

DEVICE = ('cuda' if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
clebsch = ClebschGordan(L_MAX)

print("Gaussian smoothing map for r = 1, 2, 3, 4 A:")
for nu in range(1, NU_MAX+1):
    print(f"nu = {nu}: {C*np.exp(L_NU*nu+1.0/L_R)} {C*np.exp(L_NU*nu+2.0/L_R)} {C*np.exp(L_NU*nu+3.0/L_R)} {C*np.exp(L_NU*nu+4.0/L_R)}")

train_structures, test_structures = get_dataset_slice(DATASET_PATH, train_slice, test_slice)
all_species = np.sort(np.unique(np.concatenate([train_structure.numbers for train_structure in train_structures] + [test_structure.numbers for test_structure in test_structures])))

train_train_kernel = torch.zeros((n_train, n_train, NU_MAX+1), device=DEVICE)
test_train_kernel = torch.zeros((n_test, n_train, NU_MAX+1), device=DEVICE)

print("Calculating composition kernels", flush = True)
comp_train = get_composition_features(train_structures, all_species)
comp_test = get_composition_features(test_structures, all_species)
train_train_nu0_kernel = comp_train @ comp_train.T
test_train_nu0_kernel = comp_test @ comp_train.T
train_train_kernel[:, :, 0] = train_train_nu0_kernel.to(DEVICE)
test_train_kernel[:, :, 0] = test_train_nu0_kernel.to(DEVICE)
print("Composition kernels done", flush = True)

train_energies = [structure.info[TARGET_KEY] for structure in train_structures]
train_energies = torch.tensor(train_energies, dtype = torch.get_default_dtype(), device = DEVICE) * CONVERSION_FACTOR

test_energies = [structure.info[TARGET_KEY] for structure in test_structures]
test_energies = torch.tensor(test_energies, dtype = torch.get_default_dtype(), device = DEVICE) * CONVERSION_FACTOR

if L_NU == 0.0:
    """
    if "methane" in DATASET_PATH or "ch4" in DATASET_PATH:
        hypers_spherical_expansion = {
            "cutoff": r_cut,
            "max_radial": 22,
            "max_angular": L_MAX,
            "atomic_gaussian_width": C*np.exp(L_NU*nu), 
            "center_atom_weight": 0.0,
            "radial_basis": {"Gto": {"spline_accuracy": 1e-8}},
            "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
            "radial_scaling":  {"Willatt2018": { "scale": 2.0, "rate": 2.0, "exponent": 6}}, 
        }
    else:
        hypers_spherical_expansion = {
            "cutoff": r_cut,
            "max_radial": 22,
            "max_angular": L_MAX,
            "atomic_gaussian_width": C*np.exp(L_NU*nu),
            "center_atom_weight": 0.0,
            "radial_basis": {"Gto": {"spline_accuracy": 1e-8}},
            "cutoff_function": {"Step": {}}
            # "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
            # "radial_scaling":  {"Willatt2018": {"scale": 1.5, "rate": 2.0, "exponent": 6}},
        }
    calculator = SphericalExpansion(**hypers_spherical_expansion)
    """

    calculator = get_LE_calculator(l_max=L_MAX, n_max=25, a=r_cut, nu=NU_MAX, CS=C, l_nu=L_NU, l_r=L_R)

    print("Calculating expansion coefficients", flush = True)

    train_coefs = calculator.compute(train_structures)
    train_coefs = move_to_torch(train_coefs, device=DEVICE)

    test_coefs = calculator.compute(test_structures)
    test_coefs = move_to_torch(test_coefs, device=DEVICE)

    neighbor_species_labels = Labels(
        names=["species_neighbor"],
        values=np.array(all_species, dtype=np.int32).reshape(-1, 1),
    )
    train_coefs = train_coefs.keys_to_properties(neighbor_species_labels)
    test_coefs = test_coefs.keys_to_properties(neighbor_species_labels)

    print("Expansion coefficients done", flush = True)

    if cg_mode == "full":
        model = WignerKernelFullIterations(clebsch, L_MAX, NU_MAX, DEVICE)
    else:
        model = WignerKernelReducedCost(clebsch, L_MAX, NU_MAX, DEVICE)
    model = model.to(DEVICE)

    print("Computing train-train-kernels", flush = True)
    train_train_kernel[:, :, 1:NU_MAX+1] = compute_kernel(model, train_coefs, train_coefs, batch_size = BATCH_SIZE, device = DEVICE)
    print("Computing train-test-kernels", flush = True)
    test_train_kernel[:, :, 1:NU_MAX+1] = compute_kernel(model, test_coefs, train_coefs, batch_size = BATCH_SIZE, device = DEVICE)

else:
    for nu in range(1, NU_MAX+1):
        print(f"Calculating nu = {nu} kernels")
        """
        if "methane" in DATASET_PATH or "ch4" in DATASET_PATH:
            hypers_spherical_expansion = {
                "cutoff": r_cut,
                "max_radial": 22,
                "max_angular": L_MAX,
                "atomic_gaussian_width": C*np.exp(L_NU*nu), 
                "center_atom_weight": 0.0,
                "radial_basis": {"Gto": {"spline_accuracy": 1e-8}},
                "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
                "radial_scaling":  {"Willatt2018": { "scale": 2.0, "rate": 2.0, "exponent": 6}}, 
            }
        else:
            hypers_spherical_expansion = {
                "cutoff": r_cut,
                "max_radial": 22,
                "max_angular": L_MAX,
                "atomic_gaussian_width": C*np.exp(L_NU*nu),
                "center_atom_weight": 0.0,
                "radial_basis": {"Gto": {"spline_accuracy": 1e-8}},
                "cutoff_function": {"Step": {}}
                # "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
                # "radial_scaling":  {"Willatt2018": {"scale": 1.5, "rate": 2.0, "exponent": 6}},
            }
        calculator = SphericalExpansion(**hypers_spherical_expansion)
        """
        calculator = get_LE_calculator(l_max=L_MAX, n_max=25, a=r_cut, nu=nu, CS=C, l_nu=L_NU, l_r=L_R)

        print("Calculating expansion coefficients", flush = True)

        train_coefs = calculator.compute(train_structures)
        train_coefs = move_to_torch(train_coefs)

        test_coefs = calculator.compute(test_structures)
        test_coefs = move_to_torch(test_coefs)

        neighbor_species_labels = Labels(
            names=["species_neighbor"],
            values=np.array(all_species, dtype=np.int32).reshape(-1, 1),
        )

        train_coefs.keys_to_properties(neighbor_species_labels)
        test_coefs.keys_to_properties(neighbor_species_labels)

        print("Expansion coefficients done", flush = True)

        # Kernel computation

        if cg_mode == "full":
            model = WignerKernelFullIterations(clebsch, L_MAX, nu, DEVICE)
        else:
            model = WignerKernelReducedCost(clebsch, L_MAX, nu, DEVICE)
        model = model.to(DEVICE)

        print("Computing train-train-kernels", flush = True)
        train_train_kernel[:, :, nu] = compute_kernel(model, train_coefs, train_coefs, batch_size = BATCH_SIZE, device = DEVICE)[nu-1]
        print("Computing train-test-kernels", flush = True)
        test_train_kernel[:, :, nu] = compute_kernel(model, test_coefs, train_coefs, batch_size = BATCH_SIZE, device = DEVICE)[nu-1]



print("Printing a few representative kernels:")
for iota in range(NU_MAX+1):
    print(f"nu = {iota}:")
    print(train_train_kernel[:6, :6, iota])



print("Beginning hyperparameter optimization")

train_train_kernel = train_train_kernel
test_train_kernel = test_train_kernel

validation_cycle = ValidationCycle(nu_max = NU_MAX, alpha_exp = 0.0).to(DEVICE)

def validation_loss_for_global_optimization(x):

    if optimization_mode == "kernel_exp":
        C = np.exp(np.log(10.0)*x[0])
        alpha = x[1]
        C0 = np.exp(np.log(10.0)*x[2])
        validation_cycle.coefficients.weight = torch.nn.Parameter(
            torch.tensor(
            [C0] + 
            [
            C*alpha**nu/factorial(nu) for nu in range(1, NU_MAX+1)
            ], device = DEVICE).reshape(1, -1)
        )
    else:
        validation_cycle.coefficients.weight = torch.nn.Parameter(torch.exp(np.log(10.0)*torch.tensor(x[0:NU_MAX+1], dtype = torch.get_default_dtype()).reshape(1, -1)))

    validation_loss = 0.0
    for i_validation_split in range(n_validation_splits):
        index_validation_start = i_validation_split*n_validation
        index_validation_stop = index_validation_start + n_validation

        K_train_sub = torch.empty((n_train_sub, n_train_sub, NU_MAX+1), device = train_train_kernel.device)
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
            try:
                validation_predictions = validation_cycle(K_train_sub, y_train_sub, K_validation)
                if opt_target_name == "mae":
                    validation_loss += get_sae(validation_predictions, y_validation).item()
                else:
                    validation_loss += get_sse(validation_predictions, y_validation).item()
            except:
                print("WARNING: tried singular matrix in global optimization")
                validation_loss += 10e30

    with torch.no_grad():
        if opt_target_name == "mae":
            validation_loss = validation_loss/n_train
        else:
            validation_loss = np.sqrt(validation_loss/n_train)    

    # print(x, validation_loss)
    return validation_loss

if optimization_mode == "kernel_exp":
    
    bounds = [(0.0, 12.0), (0.0, 100.0), (1.0, 15.0)]
    x0 = np.array([1.0, 0.5, 5])
    solution = sp.optimize.dual_annealing(validation_loss_for_global_optimization, bounds = bounds, x0 = x0, no_local_search = True)
    print(solution.x)

    if solution.x[0] < bounds[0][0]+0.2 or solution.x[0] > bounds[0][1]-0.2:
        print("solution[0] hit a boundary")
    if solution.x[1] > bounds[1][1]-0.2:
        print("solution[1] hit a boundary")
    if solution.x[2] < bounds[2][0]+0.2 or solution.x[2] > bounds[2][1]-0.2:
        print("solution[2] hit a boundary")

    print(solution.fun)
    best_coefficients = torch.tensor([np.exp(np.log(10.0)*solution.x[2])] + [
            np.exp(np.log(10.0)*solution.x[0])*solution.x[1]**nu/factorial(nu) for nu in range(1, NU_MAX+1)
        ], device = DEVICE)
    print("Adaptive equivalent:", best_coefficients)
    """
    # Version with only two parameters, grid search. 
    # Sacrifices 5-10% accuracy
    validation_best = 1e30
    for C_exp in np.linspace(4, 10, 7):#range(5, 10):#
        for exp in np.linspace(0, 0.5, 11):#range(5, 13):#
            C = 10.0**C_exp
            print(C_exp, exp)
            
            #coefficients = torch.tensor([1.0e9] + # You could decrease it even more... 
            #    [C * exp**nu / math.factorial(nu) for nu in range(1, 5)], 
            #    dtype = torch.get_default_dtype())
            
            coefficients = torch.tensor( 
                [C * exp**nu / factorial(nu) for nu in range(NU_MAX+1)], 
                dtype = torch.get_default_dtype(),
                device = train_train_kernel.device
            )
            print(coefficients)
            x = np.array([C_exp, exp, -1000000.0])

            validation_error = validation_loss_for_global_optimization(x)
            print("Validation error", validation_error, "kcal/mol")
            #print("Validation", validation_error*43.36411531, "meV")

            if validation_error < validation_best:
                best_coefficients = coefficients
                validation_best = validation_error
                best_C_exp = C_exp
                best_exp = exp

    print(best_C_exp, best_exp)     
    print("Final validation error", validation_best)
    """

else:
    bounds = [(-5.0, 14.0) for _ in range(NU_MAX+1)]
    x0 = [0.0] * (NU_MAX+1)
    x0 = np.array(x0)
    solution = sp.optimize.dual_annealing(validation_loss_for_global_optimization, bounds = bounds, x0 = x0, no_local_search = True)
    print(solution.x)
    print(solution.fun)
    best_coefficients = torch.exp(np.log(10.0)*torch.tensor(solution.x, dtype = torch.get_default_dtype()))

c = torch.linalg.solve(
    train_train_kernel @ best_coefficients +  # nu = 1, ..., 4 kernels
    torch.eye(n_train, device = train_train_kernel.device)  # regularization
    , 
    train_energies)

test_predictions = test_train_kernel @ best_coefficients @ c

print()
print("Final result (test RMSE or MAE):")
final_result = (get_mae(test_predictions, test_energies).item() if opt_target_name == "mae" else get_rmse(test_predictions, test_energies).item())
print(n_train, final_result)
