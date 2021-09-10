import torch
import torch.nn as nn
import math
from ..spherical_expansion import SphericalExpansion
from typing import List, Union

@torch.jit.script
def powerspectrum(se_: List[torch.Tensor]) -> torch.Tensor:
    lmax = len(se_)
    J, nsp, nmax, _ = se_[0].shape
    dtype = se_[0].dtype
    device = se_[0].device

    PS = torch.zeros(J, lmax, int(nsp* nmax*(nsp* nmax+1)/2), dtype=dtype, device=device)
    diag_ids = torch.arange(nsp* nmax, dtype=torch.long, device=device)
    upper_tri = torch.triu_indices(nsp * nmax, nsp * nmax, offset=1, dtype=torch.long, device=device)
    for l in range(lmax):
        se = se_[l].view(J, nsp* nmax, 2*l+1)
        PS[:, l, :nsp * nmax] = torch.einsum('iam,iam->ia', se[:, diag_ids, :], se[:, diag_ids, :]) / math.sqrt(2*l+1)
        PS[:, l, nsp * nmax:] = math.sqrt(2.) * torch.einsum(
            'iam,iam->ia', se[:, upper_tri[0], :], se[:, upper_tri[1], :]) / math.sqrt(2*l+1)

    return PS.view(J, -1)

class PowerSpectrum(nn.Module):
    def __init__(self, max_radial: int, max_angular: int,
                    interaction_cutoff: float,
                                gaussian_sigma_constant: float, species: Union[List[int], torch.Tensor], normalize: bool =True, smooth_width: float=0.5):
        super(PowerSpectrum, self).__init__()
        self.nmax = max_radial
        self.lmax = max_angular
        self.rc = interaction_cutoff
        self.sigma = gaussian_sigma_constant
        self.normalize = normalize
        if isinstance(species, list):
            species = torch.tensor(species, dtype=torch.long)
        self.species, _ = torch.sort(species)

        self.n_species = len(species)
        self.species2idx = -1*torch.ones(torch.max(species)+1,dtype=torch.long)
        for isp, sp in enumerate(self.species):
            self.species2idx[sp] = isp

        self.se = SphericalExpansion(max_radial, max_angular, interaction_cutoff, gaussian_sigma_constant, species, smooth_width=smooth_width)

        self.D = int(self.n_species*self.nmax*(self.n_species*self.nmax+1)/2) * (self.lmax+1)

    def size(self):
        return int(self.n_species*self.nmax*(self.n_species*self.nmax+1)/2) * (self.lmax+1)

    def forward(self, data):
        cl_ianm = self.se(data)
        pi_anbml = powerspectrum(cl_ianm)
        if self.normalize:
            return torch.nn.functional.normalize(
                pi_anbml.view(-1, self.D), dim=1)
        else:
            return pi_anbml.view(-1, self.D)
