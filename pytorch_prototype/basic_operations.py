import torch
from torch import nn

class CovLinear(torch.nn.Module):
    def __init__(self, in_shape, out_shape):
        super(CovLinear, self).__init__()
        self.in_shape = in_shape
        if type(out_shape) is dict:
            self.out_shape = out_shape
        else:
            self.out_shape = {}
            for key in self.in_shape.keys():
                self.out_shape[key] = out_shape
        if set(in_shape.keys()) != set(out_shape.keys()):
            raise ValueError("sets of keys of in_shape and out_shape must be the same")
            
        linears = {}
        for key in self.in_shape.keys():
            linears[key] = torch.nn.Linear(self.in_shape[key], 
                                           self.out_shape[key], bias = False)
        self.linears = nn.ModuleDict(linears)
        
    def forward(self, features):
        result = {}
        for key in features.keys():
            if key not in self.linears.keys():
                raise ValueError(f"key {key} in the features was not present in the initialization")
            now = features[key].transpose(1, 2)
            now = self.linears[key](now)
            now = now.transpose(1, 2)
            result[key] = now
        return result
    
class CovCat(torch.nn.Module):
    def __init__(self):
        super(CovCat, self).__init__()
    def forward(self, covariants):
        all_keys = set()
        for el in covariants:
            all_keys.update(set(el.keys()))
        result = {}
        for key in all_keys:
            now = []
            for el in covariants:
                if key in el.keys():
                    now.append(el[key])
            result[key] = torch.cat(now, dim = 1)
        return result
    