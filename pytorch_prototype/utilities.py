import numpy as np
import torch
import copy
from rascal.representations import SphericalExpansion as SPH
from nice.rascal_coefficients import process_structures

def get_L2_mean(covariants):
    L2 = 0.0
    for key in covariants.keys():
        L2 = L2 + torch.sum(covariants[key] * covariants[key], dim = (1, 2))
        
    return torch.mean(L2)

def L2_normalize(covariants, epsilon = 1e-15):
    L2 = 0.0
    for key in covariants.keys():
        L2 = L2 + torch.sum(covariants[key] * covariants[key], dim = (1, 2))
        
    #print(torch.sqrt(L2 + epsilon))
    result = {}
    for key in covariants.keys():
        result[key] = covariants[key] / torch.sqrt(L2 + epsilon)[:, None, None]
    return result
        
def get_all_species(structures):
    result = []
    for structure in structures:
        result.append(structure.get_atomic_numbers())
    result = np.concatenate(result, axis = 0)
    return np.sort(np.unique(result))

def get_central_species(structures):
    result = []
    for structure in structures:
        result.append(structure.get_atomic_numbers())
    return np.concatenate(result, axis = 0)

def get_structural_indices(structures):
    result = []
    now = 0
    for structure in structures:
        num = len(structure.get_atomic_numbers())
        for _ in range(num):
            result.append(now)
        now += 1
    return np.array(result)

def get_n_atoms(structures):
    result = []
    for structure in structures:
        result.append(len(structure.get_atomic_numbers()))
    return np.array(result)

def nested_map(obj, func):
    if type(obj) is dict:
        result = {}
        for key in obj.keys():
            result[key] = nested_map(obj[key], func)
        return result
    if type(obj) is list:
        result = []
        for value in obj:
            result.append(nested_map(value, func))
        return result
    return func(obj)


def apply_slice(obj, start, finish):
    def func(value):
        return value[start:finish]
    return nested_map(obj, func)

def to_device(obj, device):
    def func(value):
        if torch.is_tensor(value):
            return value.to(device)
        else:
            return value
    return nested_map(obj, func)

def iterate_minibatches(atomic_der_data, central_indices, derivative_indices,
                        atomic_data, structural_data, structures,
                        batch_size, target_device):
    structural_indices = get_structural_indices(structures)
    beginnings = []
    n_structures = np.max(structural_indices) + 1
    
    beginnings.append(0)
    now = 0
    for i in range(len(structural_indices)):
        if (structural_indices[i] != now):           
            beginnings.append(i)
            now = structural_indices[i]
    beginnings.append(len(structural_indices))
    beginnings = np.array(beginnings)
    
    n_atoms = get_n_atoms(structures)
    
    beginnings_der = []
    now = 0
    beginnings_der.append(0)
    for i in range(len(n_atoms)):
        now = now + n_atoms[i] * n_atoms[i]
        beginnings_der.append(now)
    beginnings_der = np.array(beginnings_der)
    
    
    for start_struc in range(0, n_structures, batch_size):
        next_struc = min(start_struc + batch_size, n_structures)
        
        if (atomic_data is not None):
            atomic_batch = apply_slice(atomic_data, beginnings[start_struc],
                                       beginnings[next_struc])
            atomic_batch = to_device(atomic_batch, target_device)
        else:
            atomic_batch = None
       
            
        if (structural_data is not None):
            
            structural_batch = apply_slice(structural_data, start_struc, next_struc)
            structural_batch = to_device(structural_batch, target_device)
        else:
            structural_batch = None
         
        if (atomic_der_data is not None):
            atomic_der_batch = apply_slice(atomic_der_data,
                                           beginnings_der[start_struc],
                                           beginnings_der[next_struc])
            atomic_der_batch = to_device(atomic_der_batch, target_device)
                
            #print("iterate from to: ", beginnings_der[start_struc] , beginnings_der[next_struc])
            central_indices_batch = central_indices[beginnings_der[start_struc] : beginnings_der[next_struc]]
            derivative_indices_batch = derivative_indices[beginnings_der[start_struc] :
                                                          beginnings_der[next_struc]]
            central_indices_batch = central_indices_batch - np.min(central_indices_batch)
            derivative_indices_batch = derivative_indices_batch - np.min(derivative_indices_batch)
        else:
            atomic_der_batch = None
            central_indices_batch = None
            derivative_indices_batch = None
        
        current_structures = structures[start_struc : next_struc]
        yield atomic_der_batch, central_indices_batch, derivative_indices_batch, atomic_batch, structural_batch, current_structures
        
def convert_rascal_coefficients(features, n_max, n_types, l_max):
    n_radial = n_max * n_types
    
    result = {}
    for l in range(l_max + 1):
        result[str(l)] = np.empty([features.shape[0], n_radial, 2 * l + 1])
    now = 0
    for n in range(n_radial):
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                result[str(l)][:, n, m + l] = features[:, now]                
                now += 1
    #for l in range(l_max + 1):
    #    result[l] = torch.from_numpy(result[l]).type(torch.get_default_dtype())        
    
    return result

def get_coefs(structures, hypers, all_species):
    structures = process_structures(structures)
    hypers = copy.deepcopy(hypers)
    hypers['global_species'] = [int(specie) for specie in all_species]
    hypers['expansion_by_species_method'] = 'user defined'
    
    n_max = hypers['max_radial']
    l_max = hypers['max_angular']
    n_types = len(all_species)
    soap = SPH(**hypers)
    features = soap.transform(structures)
    coefficients = features.get_features(soap)
    coefficients = convert_rascal_coefficients(coefficients, n_max, n_types, l_max)
    for key in coefficients.keys():
        coefficients[key] = torch.from_numpy(coefficients[key]).type(torch.get_default_dtype())
    return coefficients

def get_coef_ders(structures, hypers, all_species):
    structures = process_structures(structures)
    hypers = copy.deepcopy(hypers)
    hypers['global_species'] = [int(specie) for specie in all_species]
    hypers['expansion_by_species_method'] = 'user defined'
    hypers['compute_gradients'] = True
    
    n_max = hypers['max_radial']
    l_max = hypers['max_angular']
    n_types = len(all_species)
    soap = SPH(**hypers)
    features = soap.transform(structures)
    gradients = features.get_features_gradient(soap)    
    grad_info = features.get_gradients_info()
    
    gradients = convert_rascal_coefficients(gradients, n_max, n_types, l_max)
    
    for key in gradients.keys():
        gradients[key] = np.reshape(gradients[key], [-1, 3, gradients[key].shape[1], gradients[key].shape[2]])
        
    hashes = np.array(grad_info[:, 2], dtype = np.int64) * (np.int64(np.max(grad_info[:, 1])) + 1) + np.array(grad_info[:, 1], dtype = np.int64)
    indices = np.argsort(hashes)
    grad_info[:, 1] = grad_info[indices, 1]
    grad_info[:, 2] = grad_info[indices, 2]
    for key in gradients.keys():
        gradients[key] = gradients[key][indices]
   
    for key in gradients.keys():
        gradients[key] = torch.from_numpy(gradients[key]).type(torch.get_default_dtype())
    return gradients, grad_info[:, 1], grad_info[:, 2]
    