import torch
import torch.nn as nn
import numpy as np
import tqdm
import sparse_accumulation

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

    # DANGER ZONE

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
    
    
class WignerCombiningSingleUnrolled(torch.nn.Module):
    """
    Performs a Wigner iteration for a single l1, l2, lambda.
    """
    def __init__(self, clebsch, lambd, algorithm = 'vectorized', device = "cuda"):
        super(WignerCombiningSingleUnrolled, self).__init__()
        self.algorithm = algorithm
        self.lambd = lambd
        self.l1 = (clebsch.shape[0] - 1) // 2
        self.l2 = (clebsch.shape[1] - 1) // 2
        self.transformation = precompute_transformation(clebsch, self.l1, self.l2, lambd)

        # if algorithm == "vectorized" or algorithm == "fast_wigner" or algorithm == "loops":
        if True:
        
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
            
            # Create indices for fast CG iterations:
            m1_fast = (2*self.l1+1)*self.m1_aligned+self.m1p_aligned
            m2_fast = (2*self.l2+1)*self.m2_aligned+self.m2p_aligned
            mu_fast = (2*self.lambd+1)*self.mu_aligned+self.mup_aligned

            sort_indices = torch.argsort(mu_fast)

            self.m1_fast = m1_fast[sort_indices].to(device)
            self.m2_fast = m2_fast[sort_indices].to(device)
            self.mu_fast = mu_fast[sort_indices].to(device)
            self.multipliers_fast = self.multiplier_total_aligned[sort_indices].to(device)

        if algorithm == "dense":

            dense_transformation = torch.zeros((2*self.l1+1, 2*self.l2+1, 2*self.lambd+1), dtype=torch.get_default_dtype(), device=device)
            for mu in range(0, 2 * self.lambd + 1):
                for m1, m2, multiplier in self.transformation[mu]:
                    dense_transformation[m1, m2, mu] = multiplier
            self.dense_transformation = dense_transformation.reshape((2*self.l1+1)*(2*self.l2+1), 2*self.lambd+1)
                    
        
    def forward(self, X1, X2):
        #X1[*, m1, mp1]
        #X2[*, m2, mp2]
        #result[*, mu, mup2] <-
        device = X1.device
        
        algorithm_now = self.algorithm
        
        if algorithm_now == 'fast_wigner':  # definitiely doesn't work for l>2
            X1 = X1.reshape(-1, 1, (2*self.l1+1)**2)
            X2 = X2.reshape(-1, 1, (2*self.l2+1)**2)
            result = sparse_accumulation.accumulate(X1, X2, self.mu_fast, (2*self.lambd+1)**2, self.m1_fast, self.m2_fast, self.multipliers_fast)
            if torch.allclose(result, torch.zeros_like(result)) and device.type == "cuda":
                raise ValueError(f"You probably overflowed the GPU's cache. l1={self.l1}, l2={self.l2}, lambda={self.lambd}")
            return result.reshape(-1, 2*self.lambd+1, 2*self.lambd+1)

        if algorithm_now == 'dense':
            product = torch.einsum("iab, icd -> iacbd", X1, X2)
            result = product.reshape((product.shape[0], (2*self.l1+1)*(2*self.l2+1), (2*self.l1+1)*(2*self.l2+1)))
            result = result @ self.dense_transformation
            result = result.swapaxes(1, 2)
            result = result @ self.dense_transformation
            result = result.swapaxes(1, 2)

            return result
        
        if algorithm_now == 'vectorized':
            contributions = X1[:, self.m1_aligned, self.m1p_aligned] * X2[:, self.m2_aligned, self.m2p_aligned] \
                            * self.multiplier_total_aligned
            result = torch.zeros([X1.shape[0], (2 * self.lambd + 1) ** 2], device = device)
            result.index_add_(1, self.mu_both_aligned, contributions)
            return result[:, self.mu_both]
           
        if algorithm_now == 'loops':
            result = torch.zeros([X1.shape[0], 2 * self.lambd + 1, 2 * self.lambd + 1], device = device)
            for mu in range(0, 2 * self.lambd + 1):
                for m1, m2, multiplier in self.transformation[mu]:
                    for mup in range(0, 2 * self.lambd + 1):
                        for m1p, m2p, multiplierp in self.transformation[mup]:
                            result[:, mu, mup] += X1[:, m1, m1p] * X2[:, m2, m2p] * multiplier * multiplierp

            return result
    
class WignerCombiningUnrolled(torch.nn.Module):
    """
    Performs a Wigner iteration for all l1, l2, lambda within a batch of chemical environments.
    """
    def __init__(self, clebsch, lambd_max, algorithm = 'vectorized', device = 'cuda'):
        super(WignerCombiningUnrolled, self).__init__()
        self.algorithm = algorithm
        self.device = device
        self.lambd_max = lambd_max
        self.single_combiners = torch.nn.ModuleDict()
        for l1 in range(clebsch.shape[0]):
            for l2 in range(clebsch.shape[1]):
                for lambd in range(abs(l1 - l2), min(l1 + l2, self.lambd_max) + 1):  
                    key = '{}_{}_{}'.format(l1, l2, lambd)

                    if lambd >= clebsch.shape[2]:
                        raise ValueError("insufficient lambda max in precomputed Clebsch Gordan coefficients")

                    self.single_combiners[key] = WignerCombiningSingleUnrolled(
                        clebsch[l1, l2, lambd, :2 * l1 + 1, :2 * l2 + 1], lambd, algorithm = self.algorithm, device = self.device)
                
    def forward(self, X1, X2):
        result = {}
        for key1 in X1.keys():
            for key2 in X2.keys():
                l1 = int(key1.split("_")[0])
                l2 = int(key2.split("_")[0])
                sigma1 = int(key1.split("_")[1])
                sigma2 = int(key2.split("_")[1])
                for lambd in range(abs(l1 - l2), min(l1 + l2, self.lambd_max) + 1):
                    sigma = sigma1 * sigma2 * (-1)**(l1+l2+lambd)                   
                    combiner = self.single_combiners['{}_{}_{}'.format(l1, l2, lambd)] 
                    #print("Calling combiner on", '{}_{}_{}'.format(l1, l2, lambd))
                    if str(lambd) + "_" + str(sigma) not in result.keys():
                        result[str(lambd) + "_" + str(sigma)] = combiner(X1[key1], X2[key2])
                    else:
                        result[str(lambd) + "_" + str(sigma)] += combiner(X1[key1], X2[key2])
        return result
        