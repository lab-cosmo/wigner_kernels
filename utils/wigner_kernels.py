import numpy as np
import torch
from utils.wigner_iterations import WignerCombiningUnrolled
import tqdm

def initialize_wigner_single_l(first, second):
    """
    Evaluates the nu=1, single-l Wigner kernels as a scalar product of density expansion coefficients.
    """
    first_b_size, first_m_size = first.shape[0], first.shape[1]
    second_b_size, second_m_size = second.shape[0], second.shape[1]
    first = first.reshape([-1, first.shape[2]])
    second = second.reshape([-1, second.shape[2]])
    result = torch.matmul(first, second.transpose(0, 1))
    result = result.reshape(first_b_size, first_m_size, second_b_size, second_m_size)
    return result.transpose(1, 2)

def initialize_wigner_single_species_batched(first, second, center_species, idx_1, idx_2):
    """
    Evaluates the nu=1 Wigner kernels for a single species.
    """
    idx_1_begin, idx_1_end = idx_1
    idx_2_begin, idx_2_end = idx_2
    lmax = np.max(first.keys["spherical_harmonics_l"])
    result = {}
    for l in range(lmax+1):
        result[str(l) + "_" + str(1)] = ((8.0*np.pi**2)/(2*l+1))*initialize_wigner_single_l(
                first.block(spherical_harmonics_l=l, species_center=center_species).values[idx_1_begin:idx_1_end, :, :], 
                second.block(spherical_harmonics_l=l, species_center=center_species).values[idx_2_begin:idx_2_end, :, :]
                )
    return result

class WignerKernelFullIterations(torch.nn.Module):
    """
    Performs full Wigner iterations.
    """
    def __init__(self, clebsch, lambda_max, nu_max, device):
        super(WignerKernelFullIterations, self).__init__()
        self.nu_max = nu_max
        equivariant_iterators = {
            str(nu) : WignerCombiningUnrolled(clebsch.precomputed_, lambda_max,
            algorithm = 'dense' if device == "cuda" else "fast_wigner",
            device = device)  
            for nu in range(2, nu_max)
        }
        self.equivariant_iterators = torch.nn.ModuleDict(equivariant_iterators)
        self.invariant_iterator = WignerCombiningUnrolled(clebsch.precomputed_, 0,
        algorithm = 'dense' if device == "cuda" else "fast_wigner",
        device = device)
            
    def forward(self, X):
        result = []
        wig_nu = X
        result.append(wig_nu['0_1'][:, 0, 0, None])  # nu = 1 kernel
        for nu in range(2, self.nu_max):  # nu = 2 to nu = nu_max-1
            wig_nu = self.equivariant_iterators[str(nu)](wig_nu, X)
            result.append(wig_nu['0_1'][:, 0, 0, None])
        wig_nu = self.invariant_iterator(wig_nu, X)  # only calculate invariants for nu = nu_max
        result.append(wig_nu['0_1'][:, 0, 0, None])   
        result = torch.cat(result, dim = -1)
        return result

class WignerKernelReducedCost(torch.nn.Module):
    """
    Calculates Wigner kernels, but it calculates equivariant kernels only up to approx.
    nu/2, and then only invariant kernels.
    """
    def __init__(self, clebsch, lambda_max, nu_max, device):
        super(WignerKernelReducedCost, self).__init__()
        self.nu_max = nu_max
        equivariant_iterators = {
            str(nu): WignerCombiningUnrolled(clebsch.precomputed_, lambda_max, 
            algorithm = 'dense' if device == "cuda" else "fast_wigner", 
            device = device) 
            for nu in range(2, nu_max-nu_max//2+1)
            }
        self.equivariant_iterators = torch.nn.ModuleDict(equivariant_iterators)
        invariant_iterators = {
            str(nu): WignerCombiningUnrolled(clebsch.precomputed_, 0,
            algorithm = 'dense' if device == "cuda" else "fast_wigner",
            device = device)
            for nu in range(nu_max-nu_max//2+1, nu_max+1)
            }
        self.invariant_iterators = torch.nn.ModuleDict(invariant_iterators)

    def forward(self, X):
        equivariant_kernels = [X]
        result = [equivariant_kernels[1-1]['0_1'][:, 0, 0, None]]
        for nu in range(2, self.nu_max+1):
            if (nu <= self.nu_max-self.nu_max//2): 
                equivariant_kernels.append(self.equivariant_iterators[str(nu)](equivariant_kernels[-1], equivariant_kernels[0]))
                result.append(equivariant_kernels[-1]['0_1'][:, 0, 0, None])
            else:
                result.append(self.invariant_iterators[str(nu)](equivariant_kernels[self.nu_max-self.nu_max//2-1], equivariant_kernels[nu-self.nu_max+self.nu_max//2-1])['0_1'][:, 0, 0, None])
        result = torch.cat(result, dim = -1)
        return result

def compute_kernel(model, first, second, batch_size = 1000, device = 'cpu'):
    """
    Concatenates invariant Wigner kernels of different body orders and sums them over atoms in the same structure.
    """
    all_species = np.unique(np.concatenate([first.keys["species_center"], second.keys["species_center"]]))
    nu_max = model.nu_max

    n_first = len(np.unique(
        np.concatenate(
            [first.block(spherical_harmonics_l=0, species_center=center_species).samples["structure"] for center_species in np.unique(first.keys["species_center"].values[:, 0])]
            )))
    n_second = len(np.unique(
        np.concatenate(
            [second.block(spherical_harmonics_l=0, species_center=center_species).samples["structure"] for center_species in second.keys["species_center"].values[:, 0]]
            )))
    
    wigner_invariants = torch.zeros((n_first, n_second, nu_max), device=device)
    batch_size_each = int(np.sqrt(batch_size))  # A batch size for each of the two tensor maps involved.
  
    for center_species in all_species:
        # if center_species == 1: continue  # UNCOMMENT FOR METHANE DATASET C-ONLY VERSION
        print(f"     Calculating kernels for center species {center_species}", flush = True)
        try:
            structures_first = torch.tensor(first.block(spherical_harmonics_l=0, species_center=center_species).samples["structure"].values[:, 0], dtype=torch.long, device=wigner_invariants.device)
        except ValueError:
            print("First does not contain the above center species")
            continue
        try:
            structures_second = torch.tensor(second.block(spherical_harmonics_l=0, species_center=center_species).samples["structure"].values[:, 0], dtype=torch.long, device=wigner_invariants.device)
        except ValueError:
            print("Second does not contain the above center species")
            continue
        len_first = structures_first.shape[0]
        len_second = structures_second.shape[0]

        # Batched calculation, starting from nu = 1 kernel initialization:
        for idx_1_begin in tqdm.tqdm(range(0, len_first, batch_size_each)):
            idx_1_end = min(idx_1_begin+batch_size_each, len_first)
            dimension_1 = idx_1_end - idx_1_begin

            for idx_2_begin in range(0, len_second, batch_size_each):
                idx_2_end = min(idx_2_begin+batch_size_each, len_second)
                dimension_2 = idx_2_end - idx_2_begin

                wigner_c = initialize_wigner_single_species_batched(
                    first,
                    second, 
                    center_species,
                    (idx_1_begin, idx_1_end),
                    (idx_2_begin, idx_2_end),
                )            

                for key in wigner_c.keys():
                    wigner_c[key] = wigner_c[key].reshape([dimension_1*dimension_2, wigner_c[key].shape[2], wigner_c[key].shape[3]])               
                now = {}
                for key in wigner_c.keys():
                    now[key] = wigner_c[key]
                result_now = model(now)
                result_now = result_now.reshape([dimension_1, dimension_2, nu_max])

                temp = torch.zeros((wigner_invariants.shape[0], result_now.shape[1], nu_max), device = result_now.device)
                temp.index_add_(dim=0, index=structures_first[idx_1_begin:idx_1_end], source=result_now)
                wigner_invariants.index_add_(dim=1, index=structures_second[idx_2_begin:idx_2_end], source=temp)

                """
                # Old (and slow) version
                for i_1 in range(idx_1_begin, idx_1_end):
                    for i_2 in range(idx_2_begin, idx_2_end):
                        wigner_invariants[structures_first[i_1], structures_second[i_2]] += result_now[i_1-idx_1_begin, i_2-idx_2_begin]
                """
    return wigner_invariants
