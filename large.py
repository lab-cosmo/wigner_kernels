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

import ase 
from ase import io

import sys
i = int(sys.argv[1])
j = int(sys.argv[2])
print(i, j)

DTYPE = "double"
BATCH_SIZE = 10000
CONVERSION_FACTOR = "HARTREE_TO_KCAL_MOL"
TARGET_KEY = "U0"
DATASET_PATH = "datasets/qm9.xyz"
r_cut = 5.0
NU_MAX = 4
L_MAX = 3
opt_target_name = "mae"
if opt_target_name != "mae" and opt_target_name != "rmse": raise NotImplementedError
C = 0.03
L_NU = 0.0
L_R = 1.0
print(f"Density parameters: C={C}, L_NU={L_NU}, L_R={L_R}")
optimization_mode = "kernel_exp"
if optimization_mode != "full" and optimization_mode != "kernel_exp": raise NotImplementedError
if optimization_mode == "kernel_exp" and L_NU != 0: print("WARNING: Cannot interpret final kernel as a kernel exponential")
cg_mode = "reduced"

if DTYPE == "double": torch.set_default_dtype(torch.float64)

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCALMOL = 627.5
EV_TO_KCALMOL = HARTREE_TO_KCALMOL/HARTREE_TO_EV

conversions = {}
conversions["HARTREE_TO_EV"] = 27.211386245988
conversions["HARTREE_TO_KCAL_MOL"] = 627.509608030593
conversions["EV_TO_KCAL_MOL"] = conversions["HARTREE_TO_KCAL_MOL"]/conversions["HARTREE_TO_EV"]
conversions["KCAL_MOL_TO_MEV"] = 0.0433641153087705*1000.0
conversions["METHANE_FORCE"] = conversions["HARTREE_TO_KCAL_MOL"]/0.529177

CONVERSION_FACTOR = conversions[CONVERSION_FACTOR]

test_slice = str(i*10000) + ":" + str((i+1)*10000)
train_slice = str(j*10000) + ":" + str((j+1)*10000)

DEVICE = ('cuda' if torch.cuda.is_available() else "cpu")
clebsch = ClebschGordan(L_MAX)

print("Gaussian smoothing map for r = 1, 2, 3, 4 A:")
for nu in range(1, NU_MAX+1):
    print(f"nu = {nu}: {C*np.exp(L_NU*nu+L_R*1)} {C*np.exp(L_NU*nu+L_R*2)} {C*np.exp(L_NU*nu+L_R*3)} {C*np.exp(L_NU*nu+L_R*4)}")

test_structures = ase.io.read(DATASET_PATH, test_slice)
train_structures = ase.io.read(DATASET_PATH, train_slice)
all_species = np.sort(np.unique(np.concatenate([train_structure.numbers for train_structure in train_structures] + [test_structure.numbers for test_structure in test_structures])))

test_train_kernel = torch.zeros((len(test_structures), len(train_structures), NU_MAX+1))

print("Calculating composition kernels", flush = True)
comp_train = get_composition_features(train_structures, all_species)
comp_test = get_composition_features(test_structures, all_species)
test_train_nu0_kernel = comp_test @ comp_train.T
test_train_kernel[:, :, 0] = test_train_nu0_kernel
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
        model = WignerKernelFullIterations(clebsch, L_MAX, NU_MAX)
    else:
        model = WignerKernelReducedCost(clebsch, L_MAX, NU_MAX)
    model = model.to(DEVICE)

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

        # Kernel computation

        if cg_mode == "full":
            model = WignerKernelFullIterations(clebsch, L_MAX, nu)
        else:
            model = WignerKernelReducedCost(clebsch, L_MAX, nu)
        model = model.to(DEVICE)

        print("Computing train-test-kernels", flush = True)
        test_train_kernel[:, :, nu] = compute_kernel(model, test_coefs, train_coefs, batch_size = BATCH_SIZE, device = DEVICE)[nu-1]

torch.save(test_train_kernel, f'wks_{i}_{j}.pt')
