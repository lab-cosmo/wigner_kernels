import torch
import torch.nn as nn

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