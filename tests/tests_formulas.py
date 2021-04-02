import sys
sys.path.append("../code")
from code_pytorch import *
from nice.blocks import *
from nice.utilities import *
import numpy as np

def convert_to_complex(values):
    result = []
    lambd = (len(values) - 1) // 2
    for m in range(-lambd, lambd + 1):
        if (m < 0):
            now = 1 / np.sqrt(2.0) * (values[abs(m) + lambd] - 1j * values[-abs(m) + lambd])
        if m == 0:
            now = values[lambd]
        if (m > 0):
            if (m % 2 == 0):                
                now =  1 / np.sqrt(2.0) * (values[abs(m) + lambd] + 1j * values[-abs(m) + lambd])
            else:
                now = -1 / np.sqrt(2.0) * (values[abs(m) + lambd] + 1j * values[-abs(m) + lambd])
        result.append(now)
    return np.array(result)

def calculate_ps_single(first, second, clebsch):
    first = convert_to_complex(first)
    second = convert_to_complex(second)
    lambd = (len(first) - 1) // 2
    result = 0.0
    for m in range(-lambd, lambd + 1):
        result += first[m + lambd] * second[-m + lambd] * \
                            clebsch[lambd, lambd, 0, m + lambd, -m + lambd]
    #print(result.imag)
    return result.real
    
def calculate_ps(coefficients, clebsch):
    result = []
    for l in coefficients.keys():
        for n1 in range(coefficients[l].shape[1]):
            for n2 in range(coefficients[l].shape[1]):
                now = calculate_ps_single(coefficients[l][: 2 * l + 1, n1],
                                          coefficients[l][: 2 * l + 1, n2],
                                          clebsch)               
                result.append(now)
    return np.array(result)


class Powerspectrum(torch.nn.Module):
    def __init__(self, clebsch):
        super(Powerspectrum, self).__init__()
        self.first = ClebschCombining(clebsch, 0)

    def forward(self, X):
        ps_invariants = self.first(X, X)
        return ps_invariants
    
    
def single_test_ps_formula(clebsch, N_MAX, L_MAX, epsilon, verbose):
   
    coefficients = {}
    for l in range(L_MAX):
        coefficients[l] = np.random.rand(2 * l + 1, N_MAX)
    ps = calculate_ps(coefficients, clebsch.precomputed_)

    coefficients_torch = {}
    for l in range(L_MAX):
        coefficients_torch[l] = torch.FloatTensor(coefficients[l])[:, :, None]
            
    model_ps = Powerspectrum(clebsch.precomputed_)
    ps_torch = model_ps(coefficients_torch)
    ps_torch = np.array(ps_torch[0]).squeeze()

    total = np.sum(np.abs(np.sort(ps)))
    diff = np.sum(np.abs(np.sort(ps) - np.sort(ps_torch)))
    if verbose:
        print("total: ", total, "diff: ", diff)
    assert (diff <= total * epsilon)
    
def test_ps_formula(epsilon = 1e-5, verbose = False):
    L_MAX = 5
    N_MAX = 10
    N_TESTS = 10
    
    clebsch = nice.clebsch_gordan.ClebschGordan(L_MAX)
    np.random.seed(0)
    for _ in range(N_TESTS):
        single_test_ps_formula(clebsch, N_MAX, L_MAX, epsilon, verbose)
        
def calculate_bs_single(first, second, third, clebsch, l1, l2, l3):
    first = convert_to_complex(first)
    second = convert_to_complex(second)
    third = convert_to_complex(third)
    result = 0
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            for m3 in range(-l3, l3 + 1):
                if (m1 + m2 + m3 == 0):
                    now = first[l1 + m1] * second[l2 + m2] * third[l3 + m3]
                    now = now * clebsch[l1, l2, l3, m1 + l1, m2 + l2]
                    now = now * clebsch[l3, l3, 0, m1 + m2 + l3, -m1 - m2 + l3]
                    result += now
   
    if ((l1 + l2 + l3) % 2 == 0):
        return result.real
    else:
        return result.imag   


def calculate_bs(coefficients, clebsch):

    result = []
    for l1 in coefficients.keys():
        for l2 in coefficients.keys():
            for l3 in coefficients.keys():
                if (l3 >= abs(l1 - l2)) and (l3 <= l1 + l2):
                    for n1 in range(coefficients[l1].shape[1]):
                        for n2 in range(coefficients[l2].shape[1]):
                            for n3 in range(coefficients[l3].shape[1]):
                                result.append(
                                    calculate_bs_single(coefficients[l1][:2 * l1 + 1, n1],
                                                        coefficients[l2][:2 * l2 + 1, n2],
                                                        coefficients[l3][:2 * l3 + 1, n3],
                                                        clebsch, l1, l2, l3))
    return np.array(result)

class Bispectrum(torch.nn.Module):
    def __init__(self, clebsch, lambda_max):
        super(Bispectrum, self).__init__()
        self.first = ClebschCombining(clebsch, lambda_max)
        self.second = ClebschCombining(clebsch, 0)
            
    def forward(self, X):
        ps_covariants = self.first(X, X)
        #for key in ps_covariants.keys():
        #    print(key, ps_covariants[key].shape)
        bs_invariants = self.second(ps_covariants, X)
        return bs_invariants
    
def single_test_bs_formula(clebsch, N_MAX, L_MAX, epsilon, verbose):
    clebsch = nice.clebsch_gordan.ClebschGordan(L_MAX)
    coefficients = {}
    for l in range(L_MAX):
        coefficients[l] = np.random.rand(2 * l + 1, N_MAX)
    bs = calculate_bs(coefficients, clebsch.precomputed_)
    
    coefficients_torch = {}
    for l in range(L_MAX):
        coefficients_torch[l] = torch.FloatTensor(coefficients[l])[:, :, None]
        
    model_bs = Bispectrum(clebsch.precomputed_, L_MAX)
    bs_torch = model_bs(coefficients_torch)
    bs_torch = np.array(bs_torch[0]).squeeze()   
    
    total = np.sum(np.abs(np.sort(bs)))
    diff = np.sum(np.abs(np.sort(bs) - np.sort(bs_torch)))
    if verbose:
        print("total: ", total, "diff: ", diff)
    assert (diff <= total * epsilon)
    
    
def test_bs_formula(epsilon = 1e-5, verbose = False):
    L_MAX = 4
    N_MAX = 4
    N_TESTS = 10
    
    clebsch = nice.clebsch_gordan.ClebschGordan(L_MAX)
    np.random.seed(0)
    for _ in range(N_TESTS):
        single_test_bs_formula(clebsch, N_MAX, L_MAX, epsilon, verbose)