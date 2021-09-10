import torch
#torch.set_default_dtype(torch.float64)
import numpy as np
import ase.io

from pytorch_prototype.code_pytorch import *
from pytorch_prototype.utilities import *
from pytorch_prototype.miscellaneous import ClebschGordan

from sklearn.linear_model import Ridge

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
        return result[0][:, 0, 0]
    
    
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


def get_wigner_invariance_error(n_first, n_second, model, n_max, l_max):
    

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
    first_rotated_structures, second_rotated_structures = get_rotated(first_structures, second_structures)

    first_coefs, second_coefs = get_coefs_both(HYPERS, first_structures, second_structures)
    first_coefs_rotated, second_coefs_rotated = get_coefs_both(HYPERS, first_rotated_structures, second_rotated_structures)
    L2_mean = get_L2_mean(first_coefs)
    first_coefs, second_coefs = scale_coefs(first_coefs, second_coefs, 1.0 / L2_mean)
    first_coefs_rotated, second_coefs_rotated = scale_coefs(first_coefs_rotated, second_coefs_rotated,
                                                            1.0 / L2_mean)

    
    kernel_initial = compute_kernel(model, first_coefs, second_coefs).data.cpu().numpy()
    kernel_rotated = compute_kernel(model, first_coefs_rotated, second_coefs_rotated).data.cpu().numpy()

    norm = np.sqrt(np.mean(kernel_initial ** 2))
    delta = kernel_initial - kernel_rotated
    delta = np.sqrt(np.mean(delta ** 2))
    relative_error = delta / norm
    #print(relative_error, delta, norm)
    return relative_error
   
def test_wigner_ps_kernel_invariance(epsilon = 1e-5):
    NUM_EXPERIMENTS = 3
    N_MAX = 10
    L_MAX = 4
    N_FIRST = 7
    N_SECOND = 11
    clebsch = ClebschGordan(L_MAX)
    model = WignerKernel(clebsch, L_MAX, 0)
    for _ in range(NUM_EXPERIMENTS):
        relative_error = get_wigner_invariance_error(N_FIRST, N_SECOND, model, N_MAX, L_MAX)
        assert relative_error <= epsilon

def test_wigner_bs_kernel_invariance(epsilon = 1e-5):
    NUM_EXPERIMENTS = 3
    N_MAX = 10
    L_MAX = 4
    N_FIRST = 7
    N_SECOND = 11
    clebsch = ClebschGordan(L_MAX)
    model = WignerKernel(clebsch, L_MAX, 1)
    for _ in range(NUM_EXPERIMENTS):
        relative_error = get_wigner_invariance_error(N_FIRST, N_SECOND, model, N_MAX, L_MAX)
        assert relative_error <= epsilon
    
def test_wigner_ts_kernel_invariance(epsilon = 1e-5):
    NUM_EXPERIMENTS = 3
    N_MAX = 10
    L_MAX = 4
    N_FIRST = 7
    N_SECOND = 11
    clebsch = ClebschGordan(L_MAX)
    model = WignerKernel(clebsch, L_MAX, 2)
    for _ in range(NUM_EXPERIMENTS):
        relative_error = get_wigner_invariance_error(N_FIRST, N_SECOND, model, N_MAX, L_MAX)
        assert relative_error <= epsilon
    

    