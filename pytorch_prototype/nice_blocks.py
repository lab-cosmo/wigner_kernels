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
    
class Expansioner(torch.nn.Module):
    def __init__(self, lambda_max):
        super(Expansioner, self).__init__()
        self.lambda_max = lambda_max
        
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
            
        self.clebsch_comb = ClebschCombining(self.clebsch, self.lambda_max)
        self.cov_cat = CovCat()
        
    def forward(self, first_even, first_odd, second_even, second_odd):
        even_even = self.clebsch_comb(first_even, second_even)
        odd_odd = self.clebsch_comb(first_odd, second_odd)
        even_odd = self.clebsch_comb(first_even, second_odd)
        odd_even = self.clebsch_comb(first_odd, second_even)
        
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
            
        

class NICE(torch.nn.Module):
    def __init__(self, blocks):
        super(NICE, self).__init__()
        self.blocks = blocks
        self.initial_compressor = Compressor()
        
    def fit(self, even_initial, odd_initial):
        self.initial_compressor.fit(even_initial, odd_initial)
        even_initial, odd_initial = self.initial_compressor(even_initial, odd_initial)
        
        even_now = even_initial
        odd_now = odd_initial
        
        even_old = [even_initial]
        odd_old = [odd_initial]
        
        for block in self.blocks:
            block.fit(even_now, odd_now, even_initial, odd_initial,
                      even_old, odd_old)
            even_now, odd_now = block(even_now, odd_now, even_initial, odd_initial,
                                      even_old, odd_old)
            even_old.append(even_now)
            odd_old.append(odd_now)
            
    def forward(self, even_initial, odd_initial):
        even_initial, odd_initial  = self.initial_compressor(even_initial, odd_initial)
        even_now = even_initial
        odd_now = odd_initial
        
        even_old = [even_initial]
        odd_old = [odd_initial]
        
        for block in self.blocks:
            even_now, odd_now = block(even_now, odd_now, even_initial, odd_initial,
                                      even_old, odd_old)
            even_old.append(even_now)
            odd_old.append(odd_now)
        return even_old, odd_old
       