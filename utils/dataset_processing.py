import numpy as np
import torch
import ase
from ase import io
from equistore import Labels, TensorBlock, TensorMap

def get_dataset_slice(dataset_path, train_slice, test_slice):

    if "rmd17" in dataset_path: # or "ch4" in dataset_path: or methane??
        print("Reading dataset")
        train_structures = ase.io.read(dataset_path, index = "0:1000")
        test_structures = ase.io.read(dataset_path, index = "1000:2000")
        print("Shuffling and extracting from dataset")
        np.random.shuffle(train_structures)
        np.random.shuffle(test_structures)
        train_index_begin = int(train_slice.split(":")[0])
        train_index_end = int(train_slice.split(":")[1])
        train_structures = train_structures[train_index_begin:train_index_end]
        test_index_begin = int(test_slice.split(":")[0])
        test_index_end = int(test_slice.split(":")[1])
        test_structures = test_structures[test_index_begin:test_index_end]
        print("Shuffling and extraction done")

    elif "methane" in dataset_path:
        print("Reading dataset")
        all_structures = ase.io.read(dataset_path, index = ":100000")
        print("Shuffling and extracting from dataset")
        np.random.shuffle(all_structures)
        train_index_begin = int(train_slice.split(":")[0])
        train_index_end = int(train_slice.split(":")[1])
        train_structures = all_structures[train_index_begin:train_index_end]
        test_index_begin = int(test_slice.split(":")[0])
        test_index_end = int(test_slice.split(":")[1])
        test_structures = all_structures[test_index_begin:test_index_end]
        print("Shuffling and extraction done")

    elif "3bpa" in dataset_path:
        print("Reading dataset")
        train_structures = ase.io.read("datasets/3bpa/train_300K.xyz", index = ":500")
        test_structures = ase.io.read("datasets/3bpa/test_1200K.xyz", index = ":2139")
        print("Shuffling and extraction done")

    else:  # QM7 and QM9 don't seem to be shuffled randomly 
        print("Reading dataset")
        all_structures = ase.io.read(dataset_path, index = ":")
        print("Shuffling and extracting from dataset")
        np.random.shuffle(all_structures)
        train_index_begin = int(train_slice.split(":")[0])
        train_index_end = int(train_slice.split(":")[1])
        train_structures = all_structures[train_index_begin:train_index_end]
        test_index_begin = int(test_slice.split(":")[0])
        test_index_end = int(test_slice.split(":")[1])
        test_structures = all_structures[test_index_begin:test_index_end]
        print("Shuffling and extraction done")

    return train_structures, test_structures

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

def move_to_torch(rust_map: TensorMap, device: str) -> TensorMap:
    torch_blocks = []
    for _, block in rust_map.items():
        torch_block = TensorBlock(
            values=torch.tensor(block.values, dtype=torch.get_default_dtype(), device=device),
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )
        torch_blocks.append(torch_block)
    return TensorMap(
            keys = rust_map.keys,
            blocks = torch_blocks
        )
