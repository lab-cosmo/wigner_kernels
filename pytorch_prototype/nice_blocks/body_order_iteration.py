import torch
from typing import Dict, List, Optional


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
            
    def forward(self, even_now : Dict[str, torch.Tensor], odd_now : Dict[str, torch.Tensor],
                  even_initial : Dict[str, torch.Tensor], odd_initial : Dict[str, torch.Tensor],
                  even_old : Optional[List[Dict[str, torch.Tensor]]] = None, 
                  odd_old : Optional[List[Dict[str, torch.Tensor]]] = None):
        res_even, res_odd = self.expansioner(even_now, odd_now, even_initial, odd_initial)
        
        if self.purifier is not None:
            if (even_old is None) or (odd_old is None):
                raise ValueError("old covariants should be provided for purifier")
                
            res_even, res_odd = self.purifier(even_old, res_even, odd_old, res_odd)
            
        if self.compressor is not None:
            res_even, res_odd = self.compressor(res_even, res_odd)
        
        return res_even, res_odd