from .torch_impl import torch_neighbor_list

import torch
from torch_geometric.data import Data

def ase2data(frame, energy_tag=None, force_tag=None):
    z = torch.from_numpy(frame.get_atomic_numbers())
    pos = torch.from_numpy(frame.get_positions())
    pbc  = torch.from_numpy(frame.get_pbc())
    cell = torch.tensor(frame.get_cell().tolist(), dtype=torch.float64)
    n_atoms = torch.tensor([len(frame)])
    data = Data(z=z, pos=pos, pbc=pbc, cell=cell, n_atoms=n_atoms)

    if energy_tag is not None:
        E = torch.tensor(frame.info[energy_tag])
        data.energy = E
    if force_tag is not None:
        forces = torch.from_numpy(frame.arrays[force_tag])
        data.forces = forces

    return data