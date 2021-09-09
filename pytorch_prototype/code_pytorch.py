import torch
import torch.nn as nn
import numpy as np
from pytorch_prototype.utilities import get_central_species, get_structural_indices
from torch import vmap

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


def get_forces(structural_indices, target_X_der, X_pos_der, central_indices, derivative_indices, device):
   
    
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

def get_jacobians(output_shape, structural_indices, target_X_der, X_pos_der,
                  central_indices, derivative_indices, device):
    X_pos_der_prepared = {}
    for key in X_pos_der.keys():
        X_pos_der_prepared[key] = X_pos_der[key]
        for _ in range(len(output_shape)):
            X_pos_der_prepared[key] = torch.unsqueeze(X_pos_der_prepared[key], 2)
    

    central_indices = torch.IntTensor(central_indices).to(device)
    derivative_indices = torch.IntTensor(derivative_indices).to(device)

    derivatives_aligned = {}
    for key in target_X_der.keys():
        derivatives_aligned[key] = torch.index_select(target_X_der[key],
                                              0, central_indices)

        derivatives_aligned[key] = torch.unsqueeze(derivatives_aligned[key], 1)
    contributions = {}        
    for key in X_pos_der.keys():
        dims_sum = list(range(len(X_pos_der_prepared[key].shape)))[2 + len(output_shape):]
        contributions[key] = torch.sum(derivatives_aligned[key]\
                               * X_pos_der_prepared[key], dim = dims_sum)
   

    jacobians = torch.zeros([structural_indices.shape[0], 3] + output_shape,
                                  device = device, dtype = torch.get_default_dtype())

    for key in contributions.keys():
        jacobians.index_add_(0, derivative_indices, contributions[key])       
    return jacobians

def batched_jacobian_single_output(y, x, retain_graph):
    
    x = list(x.items())
    x_tensors = [element[1] for element in x]
    x_keys = [element[0] for element in x]
    y_batch_size = y.shape[0]
    x_batch_size = x_tensors[0].shape[0]
    
    output_size = 1
    for el in y.shape[1:]:
        output_size *= el
        
    def vjp(v):
        grads = torch.autograd.grad(y, x_tensors, v, retain_graph = retain_graph)
        '''print(len(grads))
        for el in grads:
            print(el)'''
        return grads
    
    output_grads = torch.eye(output_size, device = x_tensors[0].device)
    #print(output_grads.shape)
    output_grads = output_grads.repeat(y_batch_size, 1)
    #print(output_grads.shape)
    output_grads = output_grads.reshape([y_batch_size, -1] + list(y.shape[1:]))
    #print(output_grads.shape)
    output_grads = output_grads.transpose(0, 1)
    #print(output_grads.shape)
    result = list(vmap(vjp)(output_grads))
    
    for i in range(len(result)):
        result[i] = result[i].transpose(0, 1)
        q = [x_batch_size] + list(y.shape[1:]) + list(x_tensors[i].shape[1:])
        result[i] = result[i].reshape(q)
       
    result_dict = {}
    for i in range(len(x_keys)):
        result_dict[x_keys[i]] = result[i]
    return result_dict

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
        
    def forward(self, X, central_species = None, structural_indices = None):
        if self.central_specific:
            if central_species is None:
                raise ValueError("central species should be provided for central specie specific model")
                      

            splitted = self.splitter(X, central_species)
            result = {}
            for key in splitted.keys():            
                result[key] = self.models[str(key)](splitted[key])
            result = self.uniter(result, central_species)
        else:
            result = self.model(X)
            
        if self.accumulate:
            if structural_indices is None:
                raise ValueError("structural indices should be provided to accumulate structural targets")
            result = self.accumulator(result, structural_indices)
        return result
    
    def get_jacobians(self, X_der, central_indices, derivative_indices,
                      X, central_species = None, structural_indices = None):
        key = list(X.keys())[0]
        device = X[key].device
        
        for key in X.keys():
            if not X[key].requires_grad:
                raise ValueError("input should require grad for calculation of jacobians")
        predictions = self.forward(X, central_species = central_species, structural_indices = structural_indices)
        result = {}
        total = 0
        for key in predictions.keys():
            total += 1
            if total == len(predictions.keys()):
                derivatives = batched_jacobian_single_output(predictions[key], X, False)
            else:
                derivatives = batched_jacobian_single_output(predictions[key], X, True)
            
            result[key] = get_jacobians(list(predictions[key].shape[1:]), structural_indices, derivatives, X_der, 
                                        central_indices, derivative_indices, device)
        return result
        
    def get_forces(self, X_der, central_indices, derivative_indices, 
                   X, central_species = None, structural_indices = None):
        key = list(X.keys())[0]
        device = X[key].device
        
        for key in X.keys():
            if not X[key].requires_grad:
                raise ValueError("input should require grad for calculation of forces")
        predictions = self.forward(X, central_species = central_species, structural_indices = structural_indices)
        print("predictions")
        for key in predictions.keys():
            print(key, predictions[key].shape)
        derivatives = grad_dict(predictions, X)
        print("derivatives")
        for key in derivatives.keys():
            print(key, derivatives[key].shape)
        return get_forces(structural_indices, derivatives, X_der, central_indices, derivative_indices, device)
    
    
class Accumulator(torch.nn.Module):
    def __init__(self): 
        super(Accumulator, self).__init__()
        
    def forward(self, features, structural_indices):
        
        key = list(features.keys())[0]
        device = features[key].device
        
        if not torch.is_tensor(structural_indices):
            structural_indices = torch.IntTensor(structural_indices).to(device)
        else:
            structural_indices = structural_indices.to(device)
            
        n_structures = torch.max(structural_indices) + 1
        shapes = {}
        
        for key, value in features.items():
            now = list(value.shape)
            now[0] = n_structures
            shapes[key] = now
                    
       
        result = {key : torch.zeros(shape, dtype = torch.get_default_dtype()).to(device) for key, shape in shapes.items()}
        
        for key, value in features.items():
            result[key].index_add_(0, structural_indices, features[key])       
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
        linears = {}
        for key in self.in_shape.keys():
            linears[key] = torch.nn.Linear(self.in_shape[key], 
                                           self.out_shape[key], bias = False)
        self.linears = nn.ModuleDict(linears)
        
    def forward(self, features):
        result = {}
        for key in self.linears.keys():
            now = features[key].transpose(1, 2)
            now = self.linears[key](now)
            now = now.transpose(1, 2)
            result[key] = now
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
    

    
def multiply(first, second, multiplier):
    return [first[0], second[0], first[1] * second[1] * multiplier]

def multiply_sequence(sequence, multiplier):
    result = []
    
    for el in sequence:
        #print(el)
        #print(len(el))
        result.append([el[0], el[1], el[2] * multiplier])
    return result

def get_conversion(l, m):
    if (m < 0):
        X_re = [abs(m) + l, 1.0 / np.sqrt(2)]
        X_im = [m + l, -1.0 / np.sqrt(2)]
    if m == 0:
        X_re = [l, 1.0]
        X_im = [l, 0.0]
    if m > 0:
        if m % 2 == 0:
            X_re = [m + l, 1.0 / np.sqrt(2)]
            X_im = [-m + l, 1.0 / np.sqrt(2)]
        else:
            X_re = [m + l, -1.0 / np.sqrt(2)]
            X_im = [-m + l, -1.0 / np.sqrt(2)]
    return X_re, X_im

def compress(sequence, epsilon = 1e-15):
    result = []
    for i in range(len(sequence)):
        m1, m2, multiplier = sequence[i][0], sequence[i][1], sequence[i][2]
        already = False
        for j in range(len(result)):
            if (m1 == result[j][0]) and (m2 == result[j][1]):
                already = True
                break
                
        if not already:
            multiplier = 0.0
            for j in range(i, len(sequence)):
                if (m1 == sequence[j][0]) and (m2 == sequence[j][1]):
                    multiplier += sequence[j][2]
            if (np.abs(multiplier) > epsilon):
                result.append([m1, m2, multiplier])
    #print(len(sequence), '->', len(result))
    return result

def precompute_transformation(clebsch, l1, l2, lambd):
    result = [[] for _ in range(2 * lambd + 1)]
    for mu in range(0, lambd + 1):
        real_now = []
        imag_now = []
        for m2 in range(max(-l2, mu-l1), min(l2,mu+l1)+1):
            m1 = mu - m2
            X1_re, X1_im = get_conversion(l1, m1)
            X2_re, X2_im = get_conversion(l2, m2)

            real_now.append(multiply(X1_re, X2_re, clebsch[m1 + l1, m2 + l2]))
            real_now.append(multiply(X1_im, X2_im, -clebsch[m1 + l1, m2 + l2]))


            imag_now.append(multiply(X1_re, X2_im, clebsch[m1 + l1, m2 + l2]))
            imag_now.append(multiply(X1_im, X2_re, clebsch[m1 + l1, m2 + l2]))
        #print(real_now)
        if (l1 + l2 - lambd) % 2 == 1:
            imag_now, real_now = real_now, multiply_sequence(imag_now, -1)
        if mu > 0:
            if mu % 2 == 0:
                result[mu + lambd] = multiply_sequence(real_now, np.sqrt(2))
                result[-mu + lambd] = multiply_sequence(imag_now, np.sqrt(2))
            else:
                result[mu + lambd] = multiply_sequence(real_now, -np.sqrt(2))
                result[-mu + lambd] = multiply_sequence(imag_now, -np.sqrt(2))
        else:
            result[lambd] = real_now
            
    for i in range(len(result)):
        result[i] = compress(result[i])
    return result

class ClebschCombiningSingleUnrolled(torch.nn.Module):
    def __init__(self, clebsch, lambd): 
        super(ClebschCombiningSingleUnrolled, self).__init__()
        self.register_buffer('clebsch', torch.from_numpy(clebsch).type(torch.get_default_dtype()))        
        self.lambd = lambd
        self.l1 = (self.clebsch.shape[0] - 1) // 2
        self.l2 = (self.clebsch.shape[1] - 1) // 2
        self.transformation = precompute_transformation(clebsch, self.l1, self.l2, lambd)
        self.m1_aligned, self.m2_aligned = [], []
        self.multipliers, self.mu = [], []
        for mu in range(0, 2 * self.lambd + 1):
            for el in self.transformation[mu]:
                m1, m2, multiplier = el
                self.m1_aligned.append(m1)
                self.m2_aligned.append(m2)
                self.multipliers.append(multiplier)
                self.mu.append(mu)
        self.m1_aligned = torch.LongTensor(self.m1_aligned)
        self.m2_aligned = torch.LongTensor(self.m2_aligned)
        self.mu = torch.LongTensor(self.mu)
        self.multipliers = torch.tensor(self.multipliers).type(torch.get_default_dtype())
        
    
    def forward(self, X1, X2):
        #print("here:", X1.shape, X2.shape)
        if (self.lambd == 0):
            m1, m2, multiplier = self.transformation[0][0]
            return (torch.sum(X1 * X2, dim = 2) * multiplier)[:, :, None]
            #return (torch.sum(X1 * X2, dim = 2))[:, :, None]
        
        device = X1.device
        if str(device).startswith('cuda'): #the fastest algorithm depends on device
            multipliers = self.multipliers.to(device)
            mu = self.mu.to(device)
            contributions = X1[:, :, self.m1_aligned] * X2[:, :, self.m2_aligned] * multipliers

            result = torch.zeros([X1.shape[0], X2.shape[1], 2 * self.lambd + 1], device = device)
            result.index_add_(2, mu, contributions)
            return result
        else:
            result = torch.zeros([X1.shape[0], X2.shape[1], 2 * self.lambd + 1], device = device)
            for mu in range(0, 2 * self.lambd + 1):
                for m1, m2, multiplier in self.transformation[mu]:
                    #print("l1 l2 lambd multiplier", self.l1, self.l2, self.lambd, multiplier)
                    result[:, :, mu] += X1[:, :, m1] * X2[:, :, m2] * multiplier
           
            return result
    
    
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
            if (self.lambd == 0):
                 self.l1 = (self.clebsch.shape[0] - 1) // 2
                 self.l2 = (self.clebsch.shape[1] - 1) // 2
                 self.transformation = precompute_transformation(clebsch, self.l1, self.l2, lambd)
            self.task = None
        else:
            self.register_buffer('task', torch.IntTensor(task))
           
            
    def forward(self, X1, X2):
        #print("new")
        if self.task is None:
            if self.lambd == 0:
                first = X1
                second = X2
                #print("inside:", X1.shape, X2.shape)
                first = torch.transpose(first, 1, 2)
                result = torch.bmm(second, first) * self.transformation[0][0][2]
                #print(result.shape)
                return(result.reshape(result.shape[0], -1, 1))
                #print(result.shape)
                #print("first: ", result[0, 0:50])
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
        lists = {}
        for lambd in range(self.lambd_max + 1):
            lists[str(lambd)] = []
        
        for key1 in X1.keys():
            for key2 in X2.keys():
                l1 = int(key1)
                l2 = int(key2)
                for lambd in range(abs(l1 - l2), min(l1 + l2, self.lambd_max) + 1):                   
                    combiner = self.single_combiners['{}_{}_{}'.format(l1, l2, lambd)]                   
                    lists[str(lambd)].append(combiner(X1[key1], X2[key2]))
                    #print('{}_{}_{}'.format(l1, l2, lambd), result[lambd][-1].sum())
                    #print(X1[key1].shape, X2[key2].shape, result[str(lambd)][-1].shape)
                    
        result = {}
        for key in lists.keys():
            if len(lists[key]) > 0:
                result[key] = torch.cat(lists[key], dim = 1)
        
        return result
    
class WignerCombiningSingleUnrolled(torch.nn.Module):
    def __init__(self, clebsch, lambd, algorithm = 'vectorized'):
        super(WignerCombiningSingleUnrolled, self).__init__()
        self.algorithm = algorithm
        self.lambd = lambd
        self.l1 = (clebsch.shape[0] - 1) // 2
        self.l2 = (clebsch.shape[1] - 1) // 2
        self.transformation = precompute_transformation(clebsch, self.l1, self.l2, lambd)
        
        mu_both_now = 0
        mu_both = np.zeros([2 * self.lambd + 1, 2 * self.lambd + 1], dtype = int)
        for mu in range(0, 2 * self.lambd + 1):
            for mup in range(0, 2 * self.lambd + 1):
                mu_both[mu, mup] = mu_both_now
                mu_both_now += 1
                
        
        
        
        m1_aligned, m2_aligned, mu_aligned = [], [], []
        m1p_aligned, m2p_aligned, mup_aligned = [], [], []
        multiplier_total_aligned = []
        mu_both_aligned = []
        
        for mu in range(0, 2 * self.lambd + 1):
            for m1, m2, multiplier in self.transformation[mu]:
                for mup in range(0, 2 * self.lambd + 1):
                    for m1p, m2p, multiplierp in self.transformation[mup]:
                        m1_aligned.append(m1)
                        m2_aligned.append(m2)
                        mu_aligned.append(mu)
                        m1p_aligned.append(m1p)
                        m2p_aligned.append(m2p)
                        mup_aligned.append(mup)
                        multiplier_total_aligned.append(multiplier * multiplierp)
                        mu_both_aligned.append(mu_both[mu, mup])
        
        self.register_buffer('m1_aligned', torch.LongTensor(m1_aligned))
        self.register_buffer('m2_aligned', torch.LongTensor(m2_aligned))
        self.register_buffer('mu_aligned', torch.LongTensor(mu_aligned)) 
        
        self.register_buffer('m1p_aligned', torch.LongTensor(m1p_aligned))
        self.register_buffer('m2p_aligned', torch.LongTensor(m2p_aligned))
        self.register_buffer('mup_aligned', torch.LongTensor(mup_aligned))
        
        self.register_buffer('mu_both_aligned', torch.LongTensor(mu_both_aligned))
        self.register_buffer('mu_both', torch.LongTensor(mu_both))
        
        self.register_buffer('multiplier_total_aligned',
                             torch.tensor(multiplier_total_aligned).type(torch.get_default_dtype()))
        
                    
        
    def forward(self, X1, X2):
        #X1[*, m1, mp1]
        #X2[*, m2, mp2]
        #result[*, mu, mup2] <-
        device = X1.device
        
        algorithm_now = self.algorithm
        
        if algorithm_now == 'vectorized':
            contributions = X1[:, self.m1_aligned, self.m1p_aligned] * X2[:, self.m2_aligned, self.m2p_aligned] \
                            * self.multiplier_total_aligned
            result = torch.zeros([X1.shape[0], (2 * self.lambd + 1) ** 2], device = device)
            result.index_add_(1, self.mu_both_aligned, contributions)
            return result[:, self.mu_both]
            
            '''multipliers = self.multipliers.to(device)
            mu = self.mu.to(device)
            contributions = X1[:, :, self.m1_aligned] * X2[:, :, self.m2_aligned] * multipliers

            result = torch.zeros([X1.shape[0], X2.shape[1], 2 * self.lambd + 1], device = device)
            result.index_add_(2, mu, contributions)
            return result
        
            result = torch.zeros([X1.shape[0], 2 * self.lambd + 1, 2 * self.lambd + 1], device = device)'''
           
        if algorithm_now == 'loops':
            result = torch.zeros([X1.shape[0], 2 * self.lambd + 1, 2 * self.lambd + 1], device = device)
            for mu in range(0, 2 * self.lambd + 1):
                for m1, m2, multiplier in self.transformation[mu]:
                    for mup in range(0, 2 * self.lambd + 1):
                        for m1p, m2p, multiplierp in self.transformation[mup]:
                            result[:, mu, mup] += X1[:, m1, m1p] * X2[:, m2, m2p] * multiplier * multiplierp

            return result
    
class WignerCombiningUnrolled(torch.nn.Module):
    def __init__(self, clebsch, lambd_max, algorithm = 'vectorized'):
        super(WignerCombiningUnrolled, self).__init__()
        self.algorithm = algorithm
        self.lambd_max = lambd_max
        self.single_combiners = torch.nn.ModuleDict()
        for l1 in range(clebsch.shape[0]):
            for l2 in range(clebsch.shape[1]):
                for lambd in range(abs(l1 - l2), min(l1 + l2, self.lambd_max) + 1):  
                    key = '{}_{}_{}'.format(l1, l2, lambd)

                    if lambd >= clebsch.shape[2]:
                        raise ValueError("insufficient lambda max in precomputed Clebsch Gordan coefficients")

                    self.single_combiners[key] = WignerCombiningSingleUnrolled(
                        clebsch[l1, l2, lambd, :2 * l1 + 1, :2 * l2 + 1], lambd, algorithm = self.algorithm)
                
    def forward(self, X1, X2):
        result = {}
        for key1 in X1.keys():
            for key2 in X2.keys():
                l1 = int(key1)
                l2 = int(key2)
                for lambd in range(abs(l1 - l2), min(l1 + l2, self.lambd_max) + 1):                   
                    combiner = self.single_combiners['{}_{}_{}'.format(l1, l2, lambd)] 
                    if str(lambd) not in result.keys():
                        result[lambd] = combiner(X1[key1], X2[key2])
                    else:
                        result[lambd] +=  combiner(X1[key1], X2[key2])
        return result
        