import torch
import torch.nn as nn
import numpy as np
from utilities import get_central_species, get_structural_indices

def grad_dict(outputs, inputs, **kwargs):
    outputs = list(outputs.items())
    inputs = list(inputs.items())
    
    outputs_tensors = [element[1] for element in outputs]
    inputs_tensors = [element[1] for element in inputs]
    outputs_ones = [torch.ones_like(element) for element in outputs_tensors]
    
    derivatives = torch.autograd.grad(outputs = outputs_tensors,
                                     inputs = inputs_tensors,
                                     grad_outputs = outputs_ones,
                                     **kwargs)
    result = {}
    for i in range(len(derivatives)):
        result[inputs[i][0]] = derivatives[i]
    return result


def get_forces(structures, target_X_der, X_pos_der, central_indices, derivative_indices, device):
    structural_indices = get_structural_indices(structures)
    
    central_indices = torch.IntTensor(central_indices).to(device)
    derivative_indices = torch.IntTensor(derivative_indices).to(device)
        
    derivatives_aligned = {}
    for key in target_X_der.keys():
        derivatives_aligned[key] = torch.index_select(target_X_der[key],
                                              0, central_indices)
    contributions = {}        
    for key in X_pos_der.keys():
        #print("derivatives_aligned shape:", torch.unsqueeze(derivatives_aligned[key], 1).shape)
        #print("X_der shape: ", X_der[key].shape)
        dims_sum = list(range(len(X_pos_der[key].shape)))[2:]
        #print("dims_sum: ", dims_sum)
        contributions[key] = -torch.sum(torch.unsqueeze(derivatives_aligned[key], 1)\
                               * X_pos_der[key], dim = dims_sum)
    forces_predictions = torch.zeros([structural_indices.shape[0], 3],
                                  device = device, dtype = torch.get_default_dtype())

    for key in contributions.keys():
        forces_predictions.index_add_(0, derivative_indices, contributions[key])       
    return forces_predictions

class Atomistic(torch.nn.Module):
    def __init__(self, models, accumulate = True):
        super(Atomistic, self).__init__()
        self.accumulate = accumulate
        if type(models) == dict:
            self.central_specific = True
            self.splitter = CentralSplitter()
            self.uniter = CentralUniter()
            self.models = nn.ModuleDict(models)
        else:
            self.central_specific = False
            self.model = models
        
        
        if self.accumulate:
            self.accumulator = Accumulator()
        
    def forward(self, X, structures):
        if self.central_specific:
            central_species = get_central_species(structures)           

            splitted = self.splitter(X, central_species)
            result = {}
            for key in splitted.keys():            
                result[key] = self.models[str(key)](splitted[key])
            result = self.uniter(result, central_species)
        else:
            result = self.model(X)
            
        if self.accumulate:
            structural_indices = get_structural_indices(structures)
            result = self.accumulator(result, structural_indices)
        return result
    
    def get_forces(self, X, structures,
                   X_der, central_indices, derivative_indices):
        key = list(X.keys())[0]
        device = X[key].device
        
        for key in X.keys():
            if not X[key].requires_grad:
                raise ValueError("input should require grad for calculation of forces")
        predictions = self.forward(X, structures)
        derivatives = grad_dict(predictions, X)
        return get_forces(structures, derivatives, X_der, central_indices, derivative_indices, device)
    
    
class Accumulator(torch.nn.Module):
    def __init__(self): 
        super(Accumulator, self).__init__()
        
    def forward(self, features, structural_indices):
        n_structures = np.max(structural_indices) + 1
        shapes = {}
        device = None
        
        for key, value in features.items():
            now = list(value.shape)
            now[0] = n_structures
            shapes[key] = now
            device = value.device            
       
        result = {key : torch.zeros(shape, dtype = torch.get_default_dtype()).to(device) for key, shape in shapes.items()}
       
        structural_indices = torch.IntTensor(structural_indices).to(device)
        
        for key, value in features.items():
            result[key].index_add_(0, structural_indices, features[key])       
        return result       
        
        
class CentralSplitter(torch.nn.Module):
    def __init__(self): 
        super(CentralSplitter, self).__init__()
        
    def forward(self, features, central_species):
        all_species = np.unique(central_species)
        result = {}
        for specie in all_species:
            result[specie] = {}
            
        for key, value in features.items():
            for specie in all_species:
                mask_now = (central_species == specie)
                result[specie][key] = value[mask_now]       
        return result
        
class CentralUniter(torch.nn.Module):
    def __init__(self):
        super(CentralUniter, self).__init__()
        
    def forward(self, features, central_species):
        all_species = np.unique(central_species)
        specie = all_species[0]
        
        shapes = {}
        for key, value in features[specie].items():
            now = list(value.shape)
            now[0] = 0
            shapes[key] = now       
            
        device = None
        for specie in all_species:
            for key, value in features[specie].items():
                num = features[specie][key].shape[0]
                device = features[specie][key].device
                shapes[key][0] += num
                
          
        result = {key : torch.empty(shape, dtype = torch.get_default_dtype()).to(device) for key, shape in shapes.items()}        
        
        for specie in features.keys():
            for key, value in features[specie].items():
                mask = (specie == central_species)
                result[key][mask] = features[specie][key]
            
        return result
    

    
class ClebschCombiningSingleUnrolled(torch.nn.Module):
    def __init__(self, clebsch, lambd): 
        super(ClebschCombiningSingleUnrolled, self).__init__()
        self.register_buffer('clebsch', torch.from_numpy(clebsch).type(torch.get_default_dtype()))        
        self.lambd = lambd
        self.l1 = (self.clebsch.shape[0] - 1) // 2
        self.l2 = (self.clebsch.shape[1] - 1) // 2
        index = []
        mask = []
        for m1 in range(clebsch.shape[0]):
            for m2 in range(clebsch.shape[1]):
                if (m1+ m2 < (2 * lambd + 1)):
                    index.append(m1 + m2)
                    mask.append(True)
                else:
                    mask.append(False)
        self.register_buffer('mask', torch.tensor(mask, dtype = torch.bool))
        self.register_buffer('index', torch.IntTensor(index))
        self.sqrt_2 = np.sqrt(2.0)
        self.sqrt_2_inv = 1.0 / np.sqrt(2.0)
        
    
    def forward(self, X1, X2):
        #print("here:", X1.shape, X2.shape)
        X1 = X1.transpose(0, 2).contiguous()
        X2 = X2.transpose(0, 2).contiguous()
        if self.index.is_cuda:
            result = torch.zeros([2 * self.lambd + 1, X1.shape[1], X2.shape[2]], device = 'cuda')
        else:
            result = torch.zeros([2 * self.lambd + 1, X1.shape[1], X2.shape[2]])
        
        for mu in range(0, self.lambd + 1):
            real_now = 0.0
            imag_now = 0.0
            for m2 in range(max(-self.l2, mu-self.l1), min(self.l2,mu+self.l1)+1):
                m1 = mu - m2
                #print(m1, m2, mu)
                if (m1 < 0):
                    X1_re = X1[abs(m1) + self.l1] * self.sqrt_2_inv
                    X1_im = -X1[m1 + self.l1] * self.sqrt_2_inv

                if (m1 == 0):
                    X1_re = X1[self.l1]
                    X1_im = torch.zeros_like(X1[self.l1])
                if (m1 > 0):
                    if (m1 % 2 == 0):
                        X1_re = X1[m1 + self.l1] * self.sqrt_2_inv
                        X1_im = X1[-m1 + self.l1] * self.sqrt_2_inv
                    else:
                        X1_re = -X1[m1 + self.l1] * self.sqrt_2_inv
                        X1_im = -X1[-m1 + self.l1] * self.sqrt_2_inv
                        
                        
                if (m2 < 0):
                    X2_re = X2[abs(m2) + self.l2] * self.sqrt_2_inv
                    X2_im = -X2[m2 + self.l2] * self.sqrt_2_inv

                if (m2 == 0):
                    X2_re = X2[self.l2]
                    X2_im = torch.zeros_like(X2[self.l2])
                if (m2 > 0):
                    if (m2 % 2 == 0):
                        X2_re = X2[m2 + self.l2] * self.sqrt_2_inv
                        X2_im = X2[-m2 + self.l2] * self.sqrt_2_inv
                    else:
                        X2_re = -X2[m2 + self.l2] * self.sqrt_2_inv
                        X2_im = -X2[-m2 + self.l2] * self.sqrt_2_inv
                        
                real_now += self.clebsch[m1 + self.l1, m2 + self.l2] * \
                (X1_re * X2_re - X1_im * X2_im)
                
                imag_now += self.clebsch[m1 + self.l1, m2 + self.l2] * \
                    (X1_re * X2_im + X1_im * X2_re)
                '''print(real_now.shape)
                print(self.clebsch[m1 + self.l1, m2 + self.l2].shape)
                print(X1_re.shape)
                print(X2_re.shape)'''
               
            if ((self.l1 + self.l2 - self.lambd) % 2 == 1):
                imag_now, real_now = real_now, -imag_now      
            
            #if (mu == 0):
            #    print(self.l1 + self.l2 - self.lambd, real_now.abs().sum(), imag_now.abs().sum())
                      
        
            if (mu > 0):
                if mu % 2 == 0:
                    result[mu + self.lambd] = self.sqrt_2 * real_now
                    result[-mu + self.lambd] = self.sqrt_2 * imag_now
                else:
                    result[mu + self.lambd] = -self.sqrt_2 * real_now
                    result[-mu + self.lambd] = -self.sqrt_2 * imag_now
            else:
                #print(real_now)
                result[self.lambd] = real_now
        result = result.transpose(0, 2)      
        return result
    
        '''for m1 in range(self.clebsch.shape[0]):
            for m2 in range(self.clebsch.shape[1]):
                destination = m1 + m2 - self.l1 - self.l2 + self.lambd
                if (destination >= 0) and (destination < 2 * self.lambd + 1):                    
                    result[destination, :, :] += X1[m1] * X2[m2] * self.clebsch[m1, m2]
                    
        return result'''
        '''#print("clebsch grad:", self.clebsch.requires_grad, self.mask.requires_grad, self.index.requires_grad)
        X1 = X1[:, :, :, None]
        X2 = X2[:, :, None, :]
        #print(self.l1, self.l2, X1.shape, X2.shape)
        mult = X1 * X2
        mult = mult * self.clebsch
       
        mult = mult.reshape(mult.shape[0], mult.shape[1], -1)
        if self.index.is_cuda:
            result = torch.zeros([mult.shape[0], mult.shape[1], 2 * self.lambd + 1], device = 'cuda')
        else:
            result = torch.zeros([mult.shape[0], mult.shape[1], 2 * self.lambd + 1])
        
        result = result.index_add_(2, self.index, mult[:, :, self.mask])        
        return result'''
    
    
def get_each_with_each_task(first_size, second_size):
    task = []
    for i in range(first_size):
        for j in range(second_size):
            task.append([i, j])
    return np.array(task, dtype = int)

class ClebschCombiningSingle(torch.nn.Module):
    def __init__(self, clebsch, lambd, task = None):
        super(ClebschCombiningSingle, self).__init__()
        self.register_buffer('clebsch', torch.from_numpy(clebsch).type(torch.get_default_dtype()))
        self.lambd = lambd
        self.unrolled = ClebschCombiningSingleUnrolled(clebsch, lambd)
        if task is None:
            self.task = None
        else:
            self.register_buffer('task', torch.IntTensor(task))
            
    def forward(self, X1, X2):
        if self.task is None:
            first = X1
            second = X2
            
            first = first[:, :, None, :].repeat(1, 1, second.shape[1], 1)
            second = second[:, None, :, :].repeat(1, first.shape[1], 1, 1)

            first = first.reshape(first.shape[0], -1, first.shape[3])
            second = second.reshape(second.shape[0], -1, second.shape[3])            
            return self.unrolled(first, second)
        else:
            first = torch.index_select(first, 1, self.task[:, 0])
            second = torch.index_select(second, 1, self.task[:, 1])
            return self.unrolled(first, second)
        
           

class ClebschCombining(torch.nn.Module):
    def __init__(self, clebsch, lambd_max):
        super(ClebschCombining, self).__init__()
        self.register_buffer('clebsch', torch.from_numpy(clebsch).type(torch.get_default_dtype()))  
        self.lambd_max = lambd_max
         
        self.single_combiners = torch.nn.ModuleDict()
        for l1 in range(self.clebsch.shape[0]):
            for l2 in range(self.clebsch.shape[1]):
                for lambd in range(abs(l1 - l2), min(l1 + l2, self.lambd_max) + 1):
                    key = '{}_{}_{}'.format(l1, l2, lambd)
                    
                    if lambd >= clebsch.shape[2]:
                        raise ValueError("insufficient lambda max in precomputed Clebsch Gordan coefficients")
                        
                    self.single_combiners[key] = ClebschCombiningSingle(
                        clebsch[l1, l2, lambd, :2 * l1 + 1, :2 * l2 + 1], lambd)                
        
            
        
    def forward(self, X1, X2):
        result = {}
        for lambd in range(self.lambd_max + 1):
            result[lambd] = []
        
        for key1 in X1.keys():
            for key2 in X2.keys():
                l1 = int(key1)
                l2 = int(key2)
                for lambd in range(abs(l1 - l2), min(l1 + l2, self.lambd_max) + 1):                   
                    combiner = self.single_combiners['{}_{}_{}'.format(l1, l2, lambd)]                   
                    result[lambd].append(combiner(X1[key1], X2[key2]))
                    #print('{}_{}_{}'.format(l1, l2, lambd), result[lambd][-1].sum())
                    #print(X1[key1].shape, X2[key2].shape, result[str(lambd)][-1].shape)
                    
        for key in result.keys():
            result[key] = torch.cat(result[key], dim = 1)
        return result