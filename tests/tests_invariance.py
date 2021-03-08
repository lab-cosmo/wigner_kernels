import sys
sys.path.append("..")
from code_pytorch import *
from nice.blocks import *
from nice.utilities import *
import numpy as np
import ase.io
import copy
import os

def convert_to_torch(coefficients):
    result = {}
    for lambd in range(coefficients.shape[2]):
        result[lambd] = torch.FloatTensor(coefficients[:, :, lambd, : 2 * lambd + 1])
        result[lambd] = result[lambd].transpose(0, -1)
        #print("shape: ", lambd, result[lambd].shape)
    return result

class Powerspectrum(torch.nn.Module):
    def __init__(self, clebsch, lambda_max = None):
        super(Powerspectrum, self).__init__()
        self.first = ClebschCombining(clebsch, 0)

    def forward(self, X):
        ps_invariants = self.first(X, X)
        return ps_invariants

class Bispectrum(torch.nn.Module):
    def __init__(self, clebsch, lambda_max):
        super(Bispectrum, self).__init__()
        self.first = ClebschCombining(clebsch, lambda_max)
        self.second = ClebschCombining(clebsch, 0)

    def forward(self, X):
        ps_covariants = self.first(X, X)
        bs_invariants = self.second(ps_covariants, X)
        return bs_invariants

class Trispectrum(torch.nn.Module):
    def __init__(self, clebsch, lambda_max):
        super(Trispectrum, self).__init__()
        self.first = ClebschCombining(clebsch, lambda_max)
        self.second = ClebschCombining(clebsch, lambda_max)
        self.third = ClebschCombining(clebsch, 0)

    def forward(self, X):
        ps_covariants = self.first(X, X)
        bs_covariants = self.second(ps_covariants, X)
        ts_invariants = self.third(bs_covariants, X)
        return ts_invariants

def get_torch_invariants(structures, Model, N_MAX, L_MAX):
    HYPERS = {
        'interaction_cutoff': 6.3,
        'max_radial': N_MAX,
        'max_angular': L_MAX,
        'gaussian_sigma_type': 'Constant',
        'gaussian_sigma_constant': 0.3,
        'cutoff_smooth_width': 0.3,
        'radial_basis': 'GTO'
    }


    all_species = get_all_species(structures)

    coefficients = get_spherical_expansion(structures, HYPERS,
                                                 all_species, show_progress = False)


    for key in coefficients.keys():
        coefficients[key] = convert_to_torch(coefficients[key])

    clebsch = nice.clebsch_gordan.ClebschGordan(L_MAX)
    model = Model(clebsch.precomputed_, L_MAX)

    result = {}
    for key in coefficients.keys():
        result[key] = np.array(model(coefficients[key])[0].squeeze())
    return result

def single_test_invariance(structures, Model, N_MAX, L_MAX, epsilon, verbose = False):
    first = get_torch_invariants(structures, Model, N_MAX, L_MAX)
    rotated_structures = copy.deepcopy(structures)
    for structure in rotated_structures:
        structure.euler_rotate(np.random.rand() * 360, np.random.rand() * 360, np.random.rand() * 360)

    second = get_torch_invariants(rotated_structures, Model, N_MAX, L_MAX)

    for key in first.keys():
        total = np.sum(np.abs(first[key]))
        diff = np.sum(np.abs(first[key] - second[key]))
        if verbose:
            print("total: ", total, "diff: ", diff)
        assert (diff <= total * epsilon)

def test_powerspectrum_invariance(epsilon = 1e-5, verbose = False):

    structures = ase.io.read('../structures/methane.extxyz', index='0:20')
    N_MAX = 5
    L_MAX = 5
    N_TESTS = 10
    for _ in range(N_TESTS):
        single_test_invariance(structures, Powerspectrum, N_MAX, L_MAX, epsilon, verbose = verbose)


def test_bispectrum_invariance(epsilon = 1e-5, verbose = False):

    structures = ase.io.read('../structures/methane.extxyz', index='0:20')
    N_MAX = 3
    L_MAX = 3
    N_TESTS = 10
    for _ in range(N_TESTS):
        single_test_invariance(structures, Bispectrum, N_MAX, L_MAX, epsilon, verbose = verbose)


def test_trispectrum_invariance(epsilon = 1e-5, verbose = False):

    structures = ase.io.read('../structures/methane.extxyz', index='0:5')
    N_MAX = 3
    L_MAX = 3
    N_TESTS = 10
    for _ in range(N_TESTS):
        single_test_invariance(structures, Trispectrum, N_MAX, L_MAX, epsilon, verbose = verbose)
