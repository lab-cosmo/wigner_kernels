import numpy as np
import ase
from ase import io

def get_dataset_slice(dataset_path, slice):
    
    if "methane" in dataset_path:
        structures = ase.io.read(dataset_path, index = slice)
    else:  # QM7 and QM9 don't seem to be shuffled randomly 
        print("Shuffling and extracting from dataset")
        all_structures = ase.io.read(dataset_path, index = ":")
        np.random.shuffle(all_structures)
        index_begin = int(slice.split(":")[0])
        index_end = int(slice.split(":")[1])
        structures = all_structures[index_begin:index_end]
        print("Shuffling and extraction done")

    return structures