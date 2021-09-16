import torch
#torch.set_default_dtype(torch.float64)
import numpy as np
import ase.io

from pytorch_prototype.code_pytorch import *
from pytorch_prototype.utilities import *
from pytorch_prototype.miscellaneous import ClebschGordan

METHANE_PATH = '../structures/methane.extxyz'

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
        result[key] = initialize_wigner_single(first[key], second[key])
    return result


class WignerKernel(torch.nn.Module):
    def __init__(self, clebsch, lambda_max, num_iterations):
        super(WignerKernel, self).__init__()
        main = [WignerCombiningUnrolled(clebsch.precomputed_, lambda_max, algorithm = 'vectorized') 
                for _ in range(num_iterations)]
        self.main = nn.ModuleList(main)
        self.last = WignerCombiningUnrolled(clebsch.precomputed_, 0, algorithm = 'vectorized')
       
            
    def forward(self, X):
        wig_now = X
        for block in self.main:
            wig_now = block(wig_now, X)
        result = self.last(wig_now, X)
        return result['0'][:, 0, 0]
    
    
def compute_kernel(model, first, second):
    wigner = initialize_wigner(first, second)
   
    for key in wigner.keys():
        initial_shape = [wigner[key].shape[0], wigner[key].shape[1]]
        wigner[key] = wigner[key].reshape([-1, wigner[key].shape[2], wigner[key].shape[3]])
    ''' for key in wigner.keys():
        print(key, wigner[key].shape)'''
    result = model(wigner)
    return result.reshape(initial_shape)

def get_coefs_both(hypers, first_structures, second_structures):
    all_species = get_all_species(first_structures + second_structures)
    first_coefs = get_coefs(first_structures, hypers, all_species)
    second_coefs = get_coefs(second_structures, hypers, all_species)
    return first_coefs, second_coefs

def scale_coefs(first, second, multiplier):
    result_first, result_second = {}, {}
    for key in first.keys():
        result_first[key] = first[key] * multiplier
    for key in second.keys():
        result_second[key] = second[key] * multiplier
    return result_first, result_second

def get_rotated(first_structures, second_structures):
    first_rotated_structures = copy.deepcopy(first_structures)
    for struc in first_rotated_structures:
        struc.euler_rotate(np.random.rand() * 360, np.random.rand() * 360, np.random.rand() * 360)
    
    second_rotated_structures = copy.deepcopy(second_structures)
    for struc in second_rotated_structures:
        struc.euler_rotate(np.random.rand() * 360, np.random.rand() * 360, np.random.rand() * 360)
    return first_rotated_structures, second_rotated_structures


class Powerspectrum(torch.nn.Module):
    def __init__(self, clebsch):
        super(Powerspectrum, self).__init__()
        self.first = ClebschCombining(clebsch, 0)

    def forward(self, X):
        ps_invariants = self.first(X, X)
        return ps_invariants['0'][:, :, 0]

class Bispectrum(torch.nn.Module):
    def __init__(self, clebsch, lambda_max):
        super(Bispectrum, self).__init__()
        self.first = ClebschCombining(clebsch, lambda_max)
        self.second = ClebschCombining(clebsch, 0)

    def forward(self, X):
        ps_covariants = self.first(X, X)
        bs_invariants = self.second(ps_covariants, X)
        return bs_invariants['0'][:, :, 0]
    
def get_matching_relative_error(n_first, n_second, model_wigner, model_invariants, n_max, l_max):
    HYPERS = {
        'interaction_cutoff': 6.3,
        'max_radial': n_max,
        'max_angular': l_max,
        'gaussian_sigma_type': 'Constant',
        'gaussian_sigma_constant': 0.3,
        'cutoff_smooth_width': 0.3,
        'radial_basis': 'GTO'

    }
    
    first_structures = ase.io.read(METHANE_PATH, index = f'0:{n_first}')
    second_structures = ase.io.read(METHANE_PATH, index = f'0:{n_second}')

    first_coefs, second_coefs = get_coefs_both(HYPERS, first_structures, second_structures)
    
   
    L2_mean = get_L2_mean(first_coefs)
    first_coefs, second_coefs = scale_coefs(first_coefs, second_coefs, 1.0 / L2_mean)
    
    kernel_wigner = compute_kernel(model_wigner, first_coefs, second_coefs).data.cpu().numpy()
    
    invariants_first = model_invariants(first_coefs).data.cpu().numpy()
    invariants_second = model_invariants(second_coefs).data.cpu().numpy()
    kernel_invariants = np.dot(invariants_first, invariants_second.T)
    
    delta = kernel_wigner - kernel_invariants
    error = np.mean(delta * delta)
    relative_error = error / np.mean(kernel_invariants * kernel_invariants)
    return relative_error
    
def test_ps_kernel_matching(epsilon = 1e-8):
    L_MAX = 5
    N_MAX = 12
    N_FIRST = 7
    N_SECOND = 11
    clebsch = ClebschGordan(L_MAX)
    model_wigner = WignerKernel(clebsch, L_MAX, 0)
    model_invariants = Powerspectrum(clebsch.precomputed_)
    relative_error = get_matching_relative_error(N_FIRST, N_SECOND, model_wigner, model_invariants, N_MAX, L_MAX)
    assert relative_error <= epsilon
   
def test_bs_kernel_matching(epsilon = 1e-8):
    L_MAX = 4
    N_MAX = 4
    N_FIRST = 7
    N_SECOND = 11
    clebsch = ClebschGordan(L_MAX)
    model_wigner = WignerKernel(clebsch, L_MAX, 1)
    model_invariants = Bispectrum(clebsch.precomputed_, L_MAX)
    relative_error = get_matching_relative_error(N_FIRST, N_SECOND, model_wigner, model_invariants, N_MAX, L_MAX)
    assert relative_error <= epsilon

