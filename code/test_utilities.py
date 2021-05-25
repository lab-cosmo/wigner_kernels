import numpy as np
from nice.utilities import *

def get_numerical_derivatives(structures, all_species, hypers, epsilon = 1e-6, show_progress = True):
    result = []
    
    
    for direction_ind in [0, 1, 2]:
        first_structures, second_structures = [], []
        central_indices, derivative_indices = [], []
        pos = 0
        for struc_ind in tqdm.tqdm(range(len(structures)), disable = not show_progress):
            for atom_ind in range(len(structures[struc_ind].positions)):
                
                first = copy.deepcopy(structures[struc_ind])
                first.positions[atom_ind][direction_ind] += epsilon * 0.5
                first_structures.append(first)
                
                second = copy.deepcopy(structures[struc_ind])                
                second.positions[atom_ind][direction_ind] -= epsilon * 0.5
                second_structures.append(second)
                
                for _ in range(len(first.positions)):
                    derivative_indices.append(pos + atom_ind)
                    
                for i in range(len(first.positions)):
                    central_indices.append(pos + i)
                    
            pos += len(structures[struc_ind].positions)
                
        first_coefs =  get_spherical_expansion(first_structures, hypers, all_species,
                                                      split_by_central_specie = False,
                                                      show_progress = False)
        second_coefs = get_spherical_expansion(second_structures, hypers, all_species,
                                                      split_by_central_specie = False,
                                                      show_progress = False)
        derivative = (first_coefs - second_coefs) / epsilon
        derivative = derivative[:, np.newaxis, :, :, :]
        result.append(derivative)
    result = np.concatenate(result, axis = 1)

    result_shaped = {}
    for l in range(result.shape[3]):
        result_shaped[l] = result[:, :, :, l, :(2 * l + 1)]
    
    return result_shaped, np.array(central_indices), np.array(derivative_indices)
        
                
                
                      