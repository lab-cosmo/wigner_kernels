import torch
import torch.nn as nn
import numpy as np
from pytorch_prototype.code_pytorch import *
from pytorch_prototype.miscellaneous import ClebschGordan
from sklearn.decomposition import TruncatedSVD
from pytorch_prototype.thresholding import get_thresholded_tasks
from sklearn.linear_model import Ridge

class Compressor(torch.nn.Module):
    def __init__(self, n_components = None):
        super(Compressor, self).__init__()
        self.n_components = n_components
    
    def get_n_components(self, tensor):
        if self.n_components is None:
            return min(tensor.shape[1], tensor.shape[0])
        else:
            return min(tensor.shape[1], tensor.shape[0], self.n_components)
    
    def get_linear(self, covs):
        in_shape = {key : value.shape[1] for key, value in covs.items()}
        out_shape = {key : self.get_n_components(value) for key, value in covs.items()}
        linear = CovLinear(in_shape, out_shape)
        
        for key in covs.keys():
            n_components = self.get_n_components(covs[key])
            initial_shape = covs[key].shape
            now = covs[key].data.cpu().numpy().transpose(0, 2, 1)
            now = now.reshape(-1, now.shape[2])
            svd = TruncatedSVD(n_components = n_components)
            if (now.shape[1] == n_components):
                #print("shape now: ", now.shape)
                now = np.concatenate([now, np.zeros([now.shape[0], 1])], axis = 1)
                #print("shape after: ", now.shape)
                svd.fit(now)
                #print(svd.components_[:, -1])
                with torch.no_grad():
                    weight = torch.from_numpy(svd.components_[:, :-1])
                    linear.linears[key].weight.copy_(weight)
            else:
                svd.fit(now)
                with torch.no_grad():
                    weight = torch.from_numpy(svd.components_)
                    #print("torch weight shape: ", linear.linears[key].weight.shape)
                    #print("ridge shape: ", weight.shape)
                    linear.linears[key].weight.copy_(weight)
        return linear
            
    def fit(self, even, odd):
        
        self.even_linear = self.get_linear(even)
        self.odd_linear = self.get_linear(odd)
        
    def forward(self, even, odd):
        return self.even_linear(even), self.odd_linear(odd)
    
class Purifier(torch.nn.Module):
    def __init__(self, alpha):
        super(Purifier, self).__init__()
        self.regressor = Ridge(alpha = alpha, fit_intercept = False)
        self.cov_cat = CovCat()
        
    def get_active_keys(self, old_covs, new_covs):
        old_covs = self.cov_cat(old_covs)
        return [key for key in old_covs.keys() if key in new_covs.keys()]
    
    def get_linear(self, old_covs, new_covs):
        old_covs = self.cov_cat(old_covs)
        in_shape = {key : value.shape[1] for key, value in old_covs.items() if key in new_covs.keys()}
        out_shape = {key : value.shape[1] for key, value in new_covs.items() if key in old_covs.keys()}
        
        linear = CovLinear(in_shape, out_shape)
        
        for key in in_shape.keys():
            features = old_covs[key].data.cpu().numpy().transpose(0, 2, 1)
            targets = new_covs[key].data.cpu().numpy().transpose(0, 2, 1)
            features = features.reshape(-1, features.shape[2])
            targets = targets.reshape(-1, targets.shape[2])
            #print("features: ", features.shape)
            #print("targets: ", targets.shape)
            self.regressor.fit(features, targets)
            with torch.no_grad():
                weight = torch.from_numpy(self.regressor.coef_)
                linear.linears[key].weight.copy_(weight)
        return linear
            
    def fit(self, even_old, even_new, odd_old, odd_new):
        self.even_linear = self.get_linear(even_old, even_new)
        self.odd_linear = self.get_linear(odd_old, odd_new)
        
        self.even_active_keys = self.get_active_keys(even_old, even_new)
        self.odd_active_keys = self.get_active_keys(odd_old, odd_new)
        
    def forward(self, even_old, even_new, odd_old, odd_new):
        even_old = self.cov_cat(even_old)
        odd_old = self.cov_cat(odd_old)
        
        even_old = {key : even_old[key] for key in self.even_active_keys}
        odd_old = {key : odd_old[key] for key in self.odd_active_keys}
        
        even_purifying = self.even_linear(even_old)
        odd_purifying = self.odd_linear(odd_old)
           
        result_even = {}
        for key in even_new.keys():
            if key in even_purifying.keys():
                result_even[key] = even_new[key] - even_purifying[key]
            else:
                result_even[key] = even_new[key]
                
        result_odd = {}
        for key in odd_new.keys():
            if key in odd_purifying.keys():
                result_odd[key] = odd_new[key] - odd_purifying[key]
            else:
                result_odd[key] = odd_new[key]
        return result_even, result_odd
    
def convert_task(task, l_max, lambda_max, first_indices, second_indices):
    for key in first_indices.keys():
        if (type(key) != str):
            raise ValueError("wrong key")

    for key in second_indices.keys():
        if (type(key) != str):
            raise ValueError("wrong key")
    
    result = {}
    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):
            for lambd in range(abs(l1 - l2), min(l1 + l2, lambda_max) + 1):
                result[f'{l1}_{l2}_{lambd}'] = [[], []]
        
    for index in range(task.shape[0]):
        first_ind, l1, second_ind, l2, lambd = task[index]
        key = f'{l1}_{l2}_{lambd}'

        # new[i] = old[indices[i]]
        # old[i] = new[inverted_indices[i]]

        # need that new[task[i]] = old[processed_task[i]]
        # we have that new[task[i]] = old[indices[task[i]]]
        # so, put processed_task[i] <- indices[task[i]]

        first_ind = first_indices[str(l1)][first_ind]
        second_ind = second_indices[str(l2)][second_ind]

        result[key][0].append(first_ind)
        result[key][1].append(second_ind)
    return result

def get_sorting_indices(covariants):
    indices = {}
    for key in covariants.keys():
        squares = covariants[key].data.cpu().numpy() ** 2
        amplitudes = np.mean(squares.sum(axis = 2), axis = 0)
        indices_now = np.argsort(amplitudes)[::-1].copy()
        indices[key] = torch.LongTensor(indices_now, device = covariants[key].device)
    return indices
    
def apply_indices(covariants, indices):
    result = {}
    for key in covariants.keys():
        result[key] = covariants[key][:, indices[key]]
    return result

class Expansioner(torch.nn.Module):
    def __init__(self, lambda_max, num_expand):
        super(Expansioner, self).__init__()
        self.lambda_max = lambda_max
        self.num_expand = num_expand
        
    def fit(self, first_even,
            first_odd,
            second_even,
            second_odd,
            clebsch = None):
        
        all_keys = list(first_even.keys()) + list(first_odd.keys()) + \
                   list(second_even.keys()) + list(second_odd.keys())
        all_keys = [int(el) for el in all_keys]
        
        self.l_max = np.max(all_keys + [self.lambda_max])
        
        if clebsch is None:
            self.clebsch = ClebschGordan(self.l_max).precomputed_
        else:
            self.clebsch = clebsch
            
        if self.num_expand is not None:
            first_even_idx = get_sorting_indices(first_even)
            first_odd_idx = get_sorting_indices(first_odd)
            second_even_idx = get_sorting_indices(second_even)
            second_odd_idx = get_sorting_indices(second_odd)
    
            first_even = apply_indices(first_even, first_even_idx)
            first_odd = apply_indices(first_odd, first_odd_idx)
            second_even = apply_indices(second_even, second_even_idx)
            second_odd = apply_indices(second_odd, second_odd_idx)
          
            task_even_even, task_odd_odd, task_even_odd, task_odd_even = \
                get_thresholded_tasks(first_even, first_odd, second_even, second_odd, self.num_expand,
                                      self.l_max, self.lambda_max)
            
            
            task_even_even = convert_task(task_even_even, self.l_max, self.lambda_max,
                                         first_even_idx, second_even_idx)
            task_odd_odd = convert_task(task_odd_odd, self.l_max, self.lambda_max,
                                        first_odd_idx, second_odd_idx)
            task_even_odd = convert_task(task_even_odd, self.l_max, self.lambda_max,
                                        first_even_idx, second_odd_idx)
            task_odd_even = convert_task(task_odd_even, self.l_max, self.lambda_max,
                                        first_odd_idx, second_even_idx)
            
            self.has_tasks = True
        else:
            self.has_tasks = False
            
        if self.has_tasks:
            self.even_even_comb = ClebschCombining(self.clebsch, self.lambda_max, task =
                                                   task_even_even)
            self.odd_odd_comb = ClebschCombining(self.clebsch, self.lambda_max, task =
                                                   task_odd_odd)
            
            self.even_odd_comb = ClebschCombining(self.clebsch, self.lambda_max, task =
                                                   task_even_odd)
            
            self.odd_even_comb = ClebschCombining(self.clebsch, self.lambda_max, task =
                                                   task_odd_even)
            
        else:
            self.comb = ClebschCombining(self.clebsch, self.lambda_max)
            
        self.cov_cat = CovCat()
        
    def forward(self, first_even, first_odd, second_even, second_odd):
        if self.has_tasks:
            even_even = self.even_even_comb(first_even, second_even)
            odd_odd = self.odd_odd_comb(first_odd, second_odd)
            even_odd = self.even_odd_comb(first_even, second_odd)
            odd_even = self.odd_even_comb(first_odd, second_even)
        else:
            even_even = self.comb(first_even, second_even)
            odd_odd = self.comb(first_odd, second_odd)
            even_odd = self.comb(first_even, second_odd)
            odd_even = self.comb(first_odd, second_even)
        
        res_even = self.cov_cat([even_even, odd_odd])
        res_odd = self.cov_cat([even_odd, odd_even])
        return res_even, res_odd
        
        
class BodyOrderIteration(torch.nn.Module):
    def __init__(self, expansioner, purifier = None, compressor = None, clebsch = None):
        super(BodyOrderIteration, self).__init__()
        self.expansioner = expansioner
        self.purifier = purifier
        self.compressor = compressor
        self.clebsch = clebsch
        
    def fit(self, even_now, odd_now, even_initial, odd_initial,
                  even_old = None, odd_old = None):
        self.expansioner.fit(even_now, odd_now, even_initial, odd_initial, 
                             clebsch = self.clebsch)
        res_even, res_odd = self.expansioner(even_now, odd_now, even_initial, odd_initial)
        
        if (self.purifier is not None):
            if (even_old is None) or (odd_old is None):
                raise ValueError("old covariants should be provided for purifier")
            self.purifier.fit(even_old, res_even, odd_old, res_odd)
            res_even, res_odd = self.purifier(even_old, res_even, odd_old, res_odd)
        
        if (self.compressor is not None):
            self.compressor.fit(res_even, res_odd)
            res_even, res_odd = self.compressor(res_even, res_odd)
            
    def forward(self, even_now, odd_now, even_initial, odd_initial,
                      even_old = None, odd_old = None):
        res_even, res_odd = self.expansioner(even_now, odd_now, even_initial, odd_initial)
        
        if self.purifier is not None:
            if (even_old is None) or (odd_old is None):
                raise ValueError("old covariants should be provided for purifier")
                
            res_even, res_odd = self.purifier(even_old, res_even, odd_old, res_odd)
            
        if self.compressor is not None:
            res_even, res_odd = self.compressor(res_even, res_odd)
        
        return res_even, res_odd

def filter_invariants(covs):
    if '0' in covs.keys():
        return covs['0'][:, :, 0]
    else:
        return None

class NICE(torch.nn.Module):
    
    @staticmethod
    def convert_block(block):
        if isinstance(block, list):
            cov_block = block[0]
            inv_block = block[1]
        else:
            cov_block = block
            inv_block = None
        return cov_block, inv_block
    
    @staticmethod
    def is_empty(block):
        cov_block, inv_block = NICE.convert_block(block)
        return (inv_block is None) and (cov_block is None)
    
    @staticmethod
    def computes_covariants(block):
        cov_block, inv_block = NICE.convert_block(block)
        return cov_block is not None
    
    def __init__(self, blocks, initial_compressor):
        super(NICE, self).__init__()
        
        self.blocks = blocks
        self.initial_compressor = initial_compressor
        for block in self.blocks:
            if NICE.is_empty(block):
                raise ValueError("fully empty blocks should not present")
        
        for i, block in enumerate(self.blocks):
            if (not NICE.computes_covariants(block)) and (i + 1 < len(self.blocks)):
                raise ValueError("all inner blocks should compute covariants (should have covariant branch)")
                
        
        
    def fit(self, even_initial, odd_initial):
        if self.initial_compressor is not None:
            self.initial_compressor.fit(even_initial, odd_initial)
            even_initial, odd_initial = self.initial_compressor(even_initial, odd_initial)

        even_now = even_initial
        odd_now = odd_initial
        
        even_old = [even_initial]
        odd_old = [odd_initial]
        
        for block in self.blocks:
            cov_block, inv_block = NICE.convert_block(block)
            if cov_block is not None:
                cov_block.fit(even_now, odd_now, even_initial, odd_initial,
                      even_old, odd_old)
            if inv_block is not None:
                inv_block.fit(even_now, odd_now, even_initial, odd_initial,
                      even_old, odd_old)
                
            if cov_block is not None:
                even_now, odd_now = cov_block(even_now, odd_now, even_initial, odd_initial,
                                          even_old, odd_old)
            else:
                even_now, odd_now = None, None
            
            even_old.append(even_now)
            odd_old.append(odd_now)
            
    def forward(self, even_initial, odd_initial, return_covariants = False):
        if self.initial_compressor is not None:
            even_initial, odd_initial  = self.initial_compressor(even_initial, odd_initial)
        
        even_now = even_initial
        odd_now = odd_initial
        
        even_old = [even_initial]
        odd_old = [odd_initial]
        
        body_order_now = 1
        even_invs = {str(body_order_now) : filter_invariants(even_initial)}
        odd_invs = {str(body_order_now) : filter_invariants(odd_initial)}
      
        
        for current in self.blocks:
            body_order_now += 1
            cov_block, inv_block = NICE.convert_block(current)
            
            if inv_block is not None:
                even_inv_now, odd_inv_now = inv_block(even_now, odd_now, even_initial, odd_initial,
                                          even_old, odd_old)
                even_invs[body_order_now] = filter_invariants(even_inv_now)
                odd_invs[body_order_now] = filter_invariants(odd_inv_now)
            else:
                even_invs[body_order_now] = filter_invariants(even_now)
                odd_invs[body_order_now] = filter_invariants(odd_now)
                
            
            if cov_block is not None:
                even_now, odd_now = cov_block(even_now, odd_now, even_initial, odd_initial,
                                          even_old, odd_old)
            else:
                even_now, odd_now = None, None
                
            
                
            even_old.append(even_now)
            odd_old.append(odd_now)
            
        if return_covariants:
            even_dict, odd_dict = {}, {}
            for body_order in range(len(even_old)):
                even_dict[str(body_order + 1)] = even_old[body_order]
            
            for body_order in range(len(odd_old)):
                odd_dict[str(body_order + 1)] = odd_old[body_order]
                
            return even_invs, odd_invs, even_dict, odd_dict
        else:
            return even_invs, odd_invs
       