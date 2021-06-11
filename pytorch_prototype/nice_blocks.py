import torch
import torch.nn as nn
import numpy as np
from code_pytorch import *

from sklearn.decomposition import TruncatedSVD

class Compressor(torch.nn.Module):
    def __init__(self, n_components = None):
        super(Compressor, self).__init__()
        self.n_components = n_components
    
    def get_n_components(self, tensor):
        if self.n_components is None:
            return tensor.shape[1]
        else:
            return min(tensor.shape[1], self.n_components)
    
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
            svd.fit(now)
            
            with torch.no_grad():
                weight = torch.from_numpy(svd.components_)
                linear.linears[key].weight.copy_(weight)
        return linear
            
    def fit(self, even, odd):
        
        self.even_linear = self.get_linear(even)
        self.odd_linear = self.get_linear(odd)
        
    def forward(self, even, odd):
        return self.even_linear(even), self.odd_linear(odd)
    
class Purifier(torch.nn.Module):
    def __init__(self, regressor):
        super(Purifier, self).__init__()
        self.regressor = regressor
        self.regressor.set_params(**{"fit_intercept": False})
        self.cov_cat = CovCat()
        
    def get_linear(self, old_covs, new_covs):
        old_covs = self.cov_cat(old_covs)
        in_shape = {key : value.shape[1] for key, value in old_covs.items()}
        out_shape = {key : value.shape[1] for key, value in new_covs.items()}
        
        linear = CovLinear(in_shape, out_shape)
        
        for key in new_covs.keys():
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
        
    def forward(self, even_old, even_new, odd_old, odd_new):
        even_old = self.cov_cat(even_old)
        odd_old = self.cov_cat(odd_old)
        
        even_purifying = self.even_linear(even_old)
        odd_purifying = self.odd_linear(odd_old)
        
        result_even = {}
        for key in even_new.keys():
            result_even[key] = even_new[key] - even_purifying[key]
        result_odd = {}
        for key in odd_new.keys():
            result_odd[key] = odd_new[key] - odd_purifying[key]
        return result_even, result_odd
        
       