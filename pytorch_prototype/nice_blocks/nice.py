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
            
            if cov_block is not None:
                even_new, odd_new = cov_block(even_now, odd_now, even_initial, odd_initial,
                                          even_old, odd_old)
            else:
                even_new, odd_new = None, None
            
            if inv_block is not None:
                even_inv_now, odd_inv_now = inv_block(even_now, odd_now, even_initial, odd_initial,
                                          even_old, odd_old)
                even_invs[str(body_order_now)] = filter_invariants(even_inv_now)
                odd_invs[str(body_order_now)] = filter_invariants(odd_inv_now)
            else:
                even_invs[str(body_order_now)] = filter_invariants(even_new)
                odd_invs[str(body_order_now)] = filter_invariants(odd_new)                 
            
                
            even_now = even_new
            odd_now = odd_new
            
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