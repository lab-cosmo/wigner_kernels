import numpy as np

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

def iterate_minibatches(atomic_data, structural_data, structural_indices, batch_size):
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
    
    '''print(beginnings.shape)
    print(ends.shape)
    print(n_structures)
    print(beginnings[0], beginnings[-1])
    print(ends[0], ends[-1])
    print(beginnings[0:20])
    print(ends[0:20])'''
    for start_struc in range(0, n_structures, batch_size):
        next_struc = min(start_struc + batch_size, n_structures)
        atomic_batch = {}
        for key, el in atomic_data.items():
            atomic_batch[key] = el[beginnings[start_struc] : beginnings[next_struc]]
       
            
        structural_batch = {}
        for key, el in structural_data.items():
            structural_batch[key] = el[start_struc : next_struc]
        
        current_structural_indices = structural_indices[beginnings[start_struc] : beginnings[next_struc]]
        current_structural_indices = current_structural_indices - np.min(current_structural_indices)
        yield atomic_batch, structural_batch, current_structural_indices