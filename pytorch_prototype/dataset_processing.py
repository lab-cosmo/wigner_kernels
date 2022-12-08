import numpy as np
import torch
import ase
from ase import io
from equistore import Labels, TensorBlock, TensorMap

def get_dataset_slice(dataset_path, slice):
    
    if "methane" in dataset_path:
        structures = ase.io.read(dataset_path, index = slice)
    else:  # QM7 and QM9 don't seem to be shuffled randomly 
        print("Shuffling and extracting from dataset")
        all_structures = ase.io.read(dataset_path, index = ":")
        print("Total length:", len(all_structures))
        np.random.shuffle(all_structures)
        index_begin = int(slice.split(":")[0])
        index_end = int(slice.split(":")[1])
        structures = all_structures[index_begin:index_end]
        print("Shuffling and extraction done")

    return structures

def get_composition_features(frames, all_species):
    species_dict = {s: i for i, s in enumerate(all_species)}
    data = torch.zeros((len(frames), len(species_dict)))
    for i, f in enumerate(frames):
        for s in f.numbers:
            data[i, species_dict[s]] += 1
    properties = Labels(
        names=["atomic_number"],
        values=np.array(list(species_dict.keys()), dtype=np.int32).reshape(
            -1, 1
        ),
    )

    frames_i = np.arange(len(frames), dtype=np.int32).reshape(-1, 1)
    samples = Labels(names=["structure"], values=frames_i)

    block = TensorBlock(
        values=data, samples=samples, components=[], properties=properties
    )
    composition = TensorMap(Labels.single(), blocks=[block])
    return composition.block().values
