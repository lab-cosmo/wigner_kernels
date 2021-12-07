import torch
from typing import Dict

def _filter_invariants(covs : Dict[str, torch.Tensor]):
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
        
        for block in blocks:
            if NICE.is_empty(block):
                raise ValueError("fully empty blocks should not present")
        
        for i, block in enumerate(blocks):
            if (not NICE.computes_covariants(block)) and (i + 1 < len(blocks)):
                raise ValueError("all inner blocks should compute covariants (should have covariant branch)")
                
        cov_blocks, inv_blocks = [], []
        for block in blocks:
            cov_block, inv_block = NICE.convert_block(block)
            cov_blocks.append(cov_block)
            inv_blocks.append(inv_block)
        
        self.cov_blocks = torch.nn.ModuleList(cov_blocks)
        self.inv_blocks = torch.nn.ModuleList(inv_blocks)
        
      
        self.initial_compressor = initial_compressor
        
                
    def fit(self, even_initial, odd_initial):
        if self.initial_compressor is not None:
            self.initial_compressor.fit(even_initial, odd_initial)
            even_initial, odd_initial = self.initial_compressor(even_initial, odd_initial)

        even_now = even_initial
        odd_now = odd_initial
        
        even_old = [even_initial]
        odd_old = [odd_initial]
        
        for cov_block, inv_block in zip(self.cov_blocks, self.inv_blocks):
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
            
    def forward(self, even_initial : Dict[str, torch.Tensor], odd_initial : Dict[str, torch.Tensor]):
        if self.initial_compressor is not None:
            even_initial, odd_initial  = self.initial_compressor(even_initial, odd_initial)
        
        even_now = even_initial
        odd_now = odd_initial
        
        even_old = [even_initial]
        odd_old = [odd_initial]
        
        body_order_now = 1
        even_invs = {str(body_order_now) : _filter_invariants(even_initial)}
        odd_invs = {str(body_order_now) : _filter_invariants(odd_initial)}
      
        #print("len blocks: ", len(self.cov_blocks), len(self.inv_blocks))
        #for i in range(len(self.cov_blocks)):
        #    cov_block = self.cov_blocks[i]
        #    inv_block = self.inv_blocks[i]
        for cov_block, inv_block in zip(self.cov_blocks, self.inv_blocks):
            body_order_now += 1
            #print(body_order_now)
            
           
            
            if cov_block is not None:
                even_new, odd_new = cov_block(even_now, odd_now, even_initial, odd_initial,
                                          even_old, odd_old)
            else:
                even_new, odd_new = None, None
            
            if inv_block is not None:
                even_inv_now, odd_inv_now = inv_block(even_now, odd_now, even_initial, odd_initial,
                                          even_old, odd_old)
                even_invs[str(body_order_now)] = _filter_invariants(even_inv_now)
                odd_invs[str(body_order_now)] = _filter_invariants(odd_inv_now)
            else:
                even_invs[str(body_order_now)] = _filter_invariants(even_new)
                odd_invs[str(body_order_now)] = _filter_invariants(odd_new)                 
            
                
            even_now = even_new
            odd_now = odd_new
            
            even_old.append(even_now)
            odd_old.append(odd_now)
            
        #if return_covariants:
        even_dict : Dict[str, Dict[str, torch.Tensor]] = {}
        odd_dict : Dict[str, Dict[str, torch.Tensor]] = {}


        for body_order in range(len(even_old)):
            even_dict[str(body_order + 1)] = even_old[body_order]

        for body_order in range(len(odd_old)):
            odd_dict[str(body_order + 1)] = odd_old[body_order]

        return even_invs, odd_invs, even_dict, odd_dict
        #else:
        #    return even_invs, odd_invs