import torch
import numpy as np
from typing import Dict
from pytorch_prototype.clebsch_gordan import get_real_clebsch_gordan

class ClebschCombiningSingleUnrolled(torch.nn.Module):
    def __init__(self, clebsch, lambd): 
        super(ClebschCombiningSingleUnrolled, self).__init__()
        self.register_buffer('clebsch', torch.from_numpy(clebsch).type(torch.get_default_dtype()))        
        self.lambd = lambd
        self.l1 = (self.clebsch.shape[0] - 1) // 2
        self.l2 = (self.clebsch.shape[1] - 1) // 2
        transformation = get_real_clebsch_gordan(clebsch, self.l1, self.l2, lambd)
        self.m1_aligned, self.m2_aligned = [], []
        self.multipliers, self.mu = [], []
        for mu in range(0, 2 * self.lambd + 1):
            for el in transformation[mu]:
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
            multiplier = self.multipliers[0]
            return (torch.sum(X1 * X2, dim = 0) * multiplier)[None, :, :]
            #return (torch.sum(X1 * X2, dim = 2))[:, :, None]
        
        device = X1.device
        #if str(device).startswith('cuda'): #the fastest algorithm depends on device
        '''multipliers = self.multipliers.to(device)
        mu = self.mu.to(device)
        contributions = X1[:, :, self.m1_aligned] * X2[:, :, self.m2_aligned] * multipliers

        result = torch.zeros([X1.shape[0], X2.shape[1], 2 * self.lambd + 1], device = device)
        result.index_add_(2, mu, contributions)
        return result'''
        
        multipliers = self.multipliers.to(device)
        mu = self.mu.to(device)
        #print("x1 shape: ", X1.shape)
        #print("x2 shape: ", X2.shape)
        #print("multipliers shape: ", multipliers.shape)
        contributions = X1[self.m1_aligned, :, :] * X2[self.m2_aligned, :, :] * multipliers[:, None, None]

        result = torch.zeros([2 * self.lambd + 1, X2.shape[1], X1.shape[2]], device = device)
        result.index_add_(0, mu, contributions)
        #print("result shape: ", result.shape)
        return result
        
        '''result = torch.zeros([X1.shape[0], X2.shape[1], 2 * self.lambd + 1], device = device)
        for mu in range(0, 2 * self.lambd + 1):
            for m1, m2, multiplier in self.transformation[mu]:
                #print("l1 l2 lambd multiplier", self.l1, self.l2, self.lambd, multiplier)
                result[:, :, mu] += X1[:, :, m1] * X2[:, :, m2] * multiplier

        return result'''
    
    

class ClebschCombiningSingle(torch.nn.Module):
    def __init__(self, clebsch, lambd, task = None):
        super(ClebschCombiningSingle, self).__init__()
        self.register_buffer('clebsch', torch.from_numpy(clebsch).type(torch.get_default_dtype()))
        self.lambd = lambd
        self.unrolled = ClebschCombiningSingleUnrolled(clebsch, lambd)
        
        self.l1 = (self.clebsch.shape[0] - 1) // 2
        self.l2 = (self.clebsch.shape[1] - 1) // 2
        
        transformation = get_real_clebsch_gordan(clebsch, self.l1, self.l2, lambd)
        self.multiplier = transformation[0][0][2]
        
        if (task is None):
            self.has_task = False
            '''if (self.lambd == 0):
                 self.l1 = (self.clebsch.shape[0] - 1) // 2
                 self.l2 = (self.clebsch.shape[1] - 1) // 2
                 transformation = precompute_transformation(clebsch, self.l1, self.l2, lambd)
                 self.multiplier = transformation[0][0][2]'''
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
                #print("first")
                first = X1.transpose(0, 2)
                second = X2.transpose(0, 2)
                #print("inside:", X1.shape, X2.shape)
                first = torch.transpose(first, 1, 2).transpose(0, 1)
                result = torch.bmm(second, first) * self.multiplier
                #print(result.shape)
                return(result.reshape(result.shape[0], -1, 1)).transpose(0, 2)
                #print(result.shape)
                #print("first: ", result[0, 0:50])
            #print("second")
            first = X1
            second = X2
            
            first = first[:, :, None, :].repeat(1, 1, second.shape[1], 1)
            second = second[:, None, :, :].repeat(1, first.shape[1], 1, 1)

            first = first.reshape(first.shape[0], -1, first.shape[3])
            second = second.reshape(second.shape[0], -1, second.shape[3]) 
            
            return self.unrolled(first, second)
        else:
            #print("third")
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
        
            
        
    def forward(self, X1_initial : Dict[str, torch.Tensor], X2_initial : Dict[str, torch.Tensor]):
        X1 = {}
        for key in X1_initial.keys():
            X1[key] = X1_initial[key].transpose(0, 2)
            
        X2 = {}
        for key in X2_initial.keys():
            X2[key] = X2_initial[key].transpose(0, 2)
        
        lists = {str(lambd) : [] for lambd in range(self.lambd_max + 1)}
        
        for key, combiner in self.single_combiners.items():
            l1, l2, lambd = key.split('_')
            if (l1 in X1.keys()) and (l2 in X2.keys()):
                lists[lambd].append(combiner(X1[l1], X2[l2]))
        '''for key1 in X1.keys():
            for key2 in X2.keys():
                l1 = int(key1)
                l2 = int(key2)
                for lambd in range(abs(l1 - l2), min(l1 + l2, self.lambd_max) + 1):
                    key = '{}_{}_{}'.format(l1, l2, lambd)
                    if key in self.single_combiners.keys():
                        combiner = self.single_combiners[key]                   
                        lists[str(lambd)].append(combiner(X1[key1], X2[key2]))
                    #print('{}_{}_{}'.format(l1, l2, lambd), result[lambd][-1].sum())
                    #print(X1[key1].shape, X2[key2].shape, result[str(lambd)][-1].shape)'''
                    
        result = {}
        for key in lists.keys():
            if len(lists[key]) > 0:
                result[key] = torch.cat(lists[key], dim = 1).transpose(0, 2)
        
        return result