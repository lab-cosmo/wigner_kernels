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