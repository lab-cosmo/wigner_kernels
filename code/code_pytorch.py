import torch
import torch.nn
import numpy as np


class Accumulator(torch.nn.Module):
    def __init__(self): 
        super(Accumulator, self).__init__()
        
    def forward(self, features, structural_indices):
        n_structures = torch.max(structural_indices) + 1
        shapes = []
        for el in features:
            now = list(el.shape)
            now[0] = n_structures
            shapes.append(now)
       
        result = [torch.zeros(shape, dtype = torch.float32) for shape in shapes]
        for i in range(len(features)):
            result[i].index_add_(0, structural_indices, features[i])
        return result       
        
        
class CentralSplitter(torch.nn.Module):
    def __init__(self): 
        super(CentralSplitter, self).__init__()
        
    def forward(self, features, central_species):
        all_species = np.unique(central_species)
        result = {}
        for specie in all_species:
            result[specie] = []
        for feature in features:
            for specie in all_species:
                mask_now = (central_species == specie)
                result[specie].append(feature[mask_now])
        return result
        
class CentralUniter(torch.nn.Module):
    def __init__(self):
        super(CentralUniter, self).__init__()
        
    def forward(self, features, central_species):
        all_species = np.unique(central_species)
        key = all_species[0]
        
        shapes = []
        for el in features[key]:
            now = list(el.shape)
            now[0] = 0
            shapes.append(now)
            
        for key in all_species:
            for i in range(len(features[key])):
                num = features[key][i].shape[0]
                shapes[i][0] += num
        #print(shapes)
        
        result = [torch.empty(shape, dtype = torch.float32) for shape in shapes]
        
        for key in features.keys():
            for i in range(len(features[key])):
                mask = (key == central_species)
                result[i][mask] = features[key][i]
            
        return result
    
    
class ClebschCombiningSingleUnrolledOld(torch.nn.Module):
    def __init__(self, clebsch, lambd): 
        super(ClebschCombiningSingleUnrolledOld, self).__init__()
        self.register_buffer('clebsch', torch.FloatTensor(clebsch))        
        self.lambd = lambd
        
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
        self.register_buffer('index', torch.LongTensor(index))        
        
    def forward(self, X1, X2):
       
        #print("clebsch grad:", self.clebsch.requires_grad, self.mask.requires_grad, self.index.requires_grad)
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
        return result
    
class ClebschCombiningSingleUnrolled(torch.nn.Module):
    def __init__(self, clebsch, lambd): 
        super(ClebschCombiningSingleUnrolled, self).__init__()
        self.register_buffer('clebsch', torch.FloatTensor(clebsch))        
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
        self.register_buffer('index', torch.LongTensor(index))
        self.sqrt_2 = np.sqrt(2.0)
        self.sqrt_2_inv = 1.0 / np.sqrt(2.0)
        
    
    def forward(self, X1, X2):
        #print("here:", X1.shape, X2.shape)
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
        self.register_buffer('clebsch', torch.FloatTensor(clebsch))
        self.lambd = lambd
        self.unrolled = ClebschCombiningSingleUnrolled(clebsch, lambd)
        if task is None:
            self.task = None
        else:
            self.register_buffer('task', torch.LongTensor(task))
            
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
            first = torch.index_select(first, 1, torch.LongTensor(self.task[:, 0]))
            second = torch.index_select(second, 1, torch.LongTensor(self.task[:, 1]))
            return self.unrolled(first, second)
        
           

class ClebschCombining(torch.nn.Module):
    def __init__(self, clebsch, lambd_max):
        super(ClebschCombining, self).__init__()
        self.register_buffer('clebsch', torch.FloatTensor(clebsch))  
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