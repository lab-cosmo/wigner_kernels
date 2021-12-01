import torch
import numpy as np

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
        if (task is None):
            self.has_task = False
            if (self.lambd == 0):
                 self.l1 = (self.clebsch.shape[0] - 1) // 2
                 self.l2 = (self.clebsch.shape[1] - 1) // 2
                 self.transformation = precompute_transformation(clebsch, self.l1, self.l2, lambd)
        else:
            if len(task[0]) == 0:
                raise ValueError("task shouldn't be empty")
            self.register_buffer('task_first', torch.LongTensor(task[0]))
            self.register_buffer('task_second', torch.LongTensor(task[1]))
            self.has_task = True
            
            
           
            
    def forward(self, X1, X2):
        #print("new")
        if not self.has_task:
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
            first = X1[:, self.task_first, :]
            second = X2[:, self.task_second, :]
            return self.unrolled(first, second)
        
           

class ClebschCombining(torch.nn.Module):
    def __init__(self, clebsch, lambd_max, task = None):
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
                    if task is None:
                        task_now = None
                    else:
                        task_now = task[key]
                    if (task_now is None) or (len(task_now[0]) > 0):
                        self.single_combiners[key] = ClebschCombiningSingle(
                            clebsch[l1, l2, lambd, :2 * l1 + 1, :2 * l2 + 1], lambd, task_now)                
        
            
        
    def forward(self, X1, X2):
        lists = {}
        for lambd in range(self.lambd_max + 1):
            lists[str(lambd)] = []
        
        for key1 in X1.keys():
            for key2 in X2.keys():
                l1 = int(key1)
                l2 = int(key2)
                for lambd in range(abs(l1 - l2), min(l1 + l2, self.lambd_max) + 1):
                    key = '{}_{}_{}'.format(l1, l2, lambd)
                    if key in self.single_combiners.keys():
                        combiner = self.single_combiners[key]                   
                        lists[str(lambd)].append(combiner(X1[key1], X2[key2]))
                    #print('{}_{}_{}'.format(l1, l2, lambd), result[lambd][-1].sum())
                    #print(X1[key1].shape, X2[key2].shape, result[str(lambd)][-1].shape)
                    
        result = {}
        for key in lists.keys():
            if len(lists[key]) > 0:
                result[key] = torch.cat(lists[key], dim = 1)
        
        return result