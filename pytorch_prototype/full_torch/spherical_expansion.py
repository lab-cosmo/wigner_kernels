import torch
import torch.nn as nn

from e3nn import o3
from torch_scatter import scatter

from ._spherical_harmonics import SphericalHarmonics
from .radial_basis import splined_radial_integrals
from .cutoff import ShiftedCosineCutoff
from .neighbor_list import torch_neighbor_list
from .math_utils import safe_norm, safe_normalization
from typing import List


@torch.jit.script
def mult_ljn_ljm(x_ljn: torch.Tensor, x_ljm: List[torch.Tensor]) -> List[torch.Tensor]:

    lmax, J, nmax = x_ljn.shape
    x_out = [torch.empty((J, nmax, 2*l+1), dtype=x_ljn.dtype,
                                            device=x_ljn.device) for l in range(lmax)]

    for l in range(lmax):
        x_out[l] = torch.einsum(
            'jn,jm->jnm', x_ljn[l], x_ljm[l])

    return x_out

@torch.jit.script
def species_dependant_reduction_ids(z: torch.Tensor, idx_j: torch.Tensor, idx_i: torch.Tensor, species2idx: torch.Tensor):
    zj = z[idx_j]
    n_species = len(torch.unique(species2idx))-1
    ids = species2idx[zj]+ n_species*idx_i
    return ids


class SphericalExpansion(nn.Module):
    """
    Computes the spherical expansion of an atomic density. For a given atom :math:`i` in an atomic structure, the expansion coefficients are computed as:

    .. math::

        c^i_{anlm} = \sum_{(j,b) \in i} \delta_{ab} I_{nl}(r_{ij} Y_l^m(\hat{\mathbf{r}}_{ij})) f_c(r_{ij})

    where the sum runs over the neighbors :math:`j` of atom i with atomic type :math:`b`, :math:`f_c` is a cutoff function, :math:`I_{nl}` is the radial integral and :math:`Y_l^m` are spherical harmonics.

    """
    def __init__(self, nmax: int, lmax: int, rc: float, sigma: float, species: torch.Tensor, smooth_width: float=0.5):
        super(SphericalExpansion, self).__init__()
        self.nmax = nmax
        self.lmax = lmax
        self.rc = rc
        self.sigma = sigma
        self.cutoff = ShiftedCosineCutoff(rc, smooth_width)

        self.species, _ = torch.sort(species)
        self.n_species = len(species)
        species2idx = -1*torch.ones(torch.max(species)+1,dtype=torch.long)
        for isp, sp in enumerate(self.species):
            species2idx[sp] = isp
        self.register_buffer("species2idx", species2idx)

        self.Rln = splined_radial_integrals(self.nmax, self.lmax+1,
                                self.rc, self.sigma, mesh_size=600)

        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=self.lmax)
        self.Ylm = SphericalHarmonics(self.irreps_sh, normalization='integral',
                                            normalize=False)

    def forward(self, data):
        if 'idx_j' not in data or 'idx_i' not in data or 'cell_shifts' not in data:
            idx_i, idx_j, cell_shifts, _ = torch_neighbor_list(data, self.rc, self_interaction=True)
            data.idx_i = idx_i
            data.idx_j = idx_j
            data.cell_shifts = cell_shifts

        if 'direction_vectors' not in data or 'distances' not in data:
            rij = (data.pos[data.idx_j] - data.pos[data.idx_i] + data.cell_shifts)
            data.distances = safe_norm(rij, dim=1)
            data.direction_vectors = safe_normalization(rij, data.distances)

        reduction_ids = species_dependant_reduction_ids(data.z, data.idx_j, data.idx_i, self.species2idx)

        Yljm = self.Ylm(data.direction_vectors)
        RIjln = self.Rln(data.distances).view(-1,self.lmax+1,self.nmax) * self.cutoff(data.distances)[:, None, None]

        n_atoms = torch.sum(data.n_atoms)

        c_ljnm = mult_ljn_ljm(RIjln.transpose(0, 1).contiguous(), Yljm)
        cl_ianm = [
            torch.empty((self.n_species*n_atoms, self.nmax, 2*l+1))
                                        for l in range(self.lmax+1)
        ]
        for l in range(self.lmax+1):
            cl_ianm[l] = scatter(c_ljnm[l], reduction_ids,
                            dim_size=self.n_species*n_atoms, dim=0, reduce='sum').view(n_atoms, self.n_species, self.nmax, 2*l+1)
        return cl_ianm

