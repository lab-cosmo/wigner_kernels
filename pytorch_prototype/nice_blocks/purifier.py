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