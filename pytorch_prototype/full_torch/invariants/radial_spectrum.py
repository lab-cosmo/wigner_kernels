import torch
import torch.nn as nn
from typing import List, Union

from ..spherical_expansion import SphericalExpansion


class RadialSpectrum(nn.Module):
    def __init__(self, max_radial: int, interaction_cutoff: float,
                                gaussian_sigma_constant: float, species: Union[List[int], torch.Tensor], normalize: bool =True, smooth_width: float=0.5):
        super(RadialSpectrum, self).__init__()
        self.nmax = max_radial
        self.lmax = 0
        self.rc = interaction_cutoff
        self.sigma = gaussian_sigma_constant
        self.normalize = normalize

        if isinstance(species, list):
            species = torch.tensor(species, dtype=torch.long)
        self.species, _ = torch.sort(species)
        self.n_species = len(species)
        self.species2idx = -1*torch.ones(torch.max(species)+1,dtype=torch.int32)
        for isp, sp in enumerate(self.species):
            self.species2idx[sp] = isp

        self.se = SphericalExpansion(max_radial, 0, interaction_cutoff, gaussian_sigma_constant, species, smooth_width)

        self.D = self.n_species*self.nmax

    def size(self):
        return int(self.n_species*self.nmax)

    def forward(self, data):
        ci_an = self.se(data)[0]

        if self.normalize:
            return torch.nn.functional.normalize(
                ci_an.view(-1, self.D), dim=1)
        else:
            return ci_an.view(-1, self.D)