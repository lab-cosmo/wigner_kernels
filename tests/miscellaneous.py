import sys
sys.path.append("../code")
from code_pytorch import *
from utilities import *
import ase.io
import numpy as np
from nice.utilities import *

def test_central_splitter_uniter(epsilon = 1e-12):  
    
    LAMBDA_MAX = 5

    HYPERS = {
        'interaction_cutoff': 6.3,
        'max_radial': 5,
        'max_angular': LAMBDA_MAX,
        'gaussian_sigma_type': 'Constant',
        'gaussian_sigma_constant': 0.3,
        'cutoff_smooth_width': 0.3,
        'radial_basis': 'GTO'
    }


    structures = ase.io.read('../structures/methane.extxyz', index='0:20')
    all_species = get_all_species(structures)
    
    block = CentralSplitter()
    central_species = get_central_species(structures)
    coefficients = get_spherical_expansion(structures, HYPERS,
                                                 all_species, split_by_central_specie= False, show_progress=False)
    coefficients = torch.FloatTensor(coefficients)
    #print(coefficients.shape)
    #print(central_species.shape)
    result = block([coefficients, coefficients], central_species)
    #for key in result.keys():
    #    print(key, len(result[key]), result[key][0].shape)

    block = CentralUniter()
    result = block(result, central_species)
    #print(len(result))
    #print(result[0].shape)
    delta = result[0] - coefficients
    #print(torch.sum(torch.abs(delta)))
    assert torch.sum(torch.abs(delta)) < epsilon