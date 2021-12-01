    
def convert_task(task, l_max, lambda_max, first_indices, second_indices):
    for key in first_indices.keys():
        if (type(key) != str):
            raise ValueError("wrong key")

    for key in second_indices.keys():
        if (type(key) != str):
            raise ValueError("wrong key")
    
    result = {}
    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):
            for lambd in range(abs(l1 - l2), min(l1 + l2, lambda_max) + 1):
                result[f'{l1}_{l2}_{lambd}'] = [[], []]
        
    for index in range(task.shape[0]):
        first_ind, l1, second_ind, l2, lambd = task[index]
        key = f'{l1}_{l2}_{lambd}'

        # new[i] = old[indices[i]]
        # old[i] = new[inverted_indices[i]]

        # need that new[task[i]] = old[processed_task[i]]
        # we have that new[task[i]] = old[indices[task[i]]]
        # so, put processed_task[i] <- indices[task[i]]

        first_ind = first_indices[str(l1)][first_ind]
        second_ind = second_indices[str(l2)][second_ind]

        result[key][0].append(first_ind)
        result[key][1].append(second_ind)
    return result

def get_sorting_indices(covariants):
    indices = {}
    for key in covariants.keys():
        squares = covariants[key].data.cpu().numpy() ** 2
        amplitudes = np.mean(squares.sum(axis = 2), axis = 0)
        indices_now = np.argsort(amplitudes)[::-1].copy()
        indices[key] = torch.LongTensor(indices_now, device = covariants[key].device)
    return indices
    
def apply_indices(covariants, indices):
    result = {}
    for key in covariants.keys():
        result[key] = covariants[key][:, indices[key]]
    return result

class Expansioner(torch.nn.Module):
    def __init__(self, lambda_max, num_expand):
        super(Expansioner, self).__init__()
        self.lambda_max = lambda_max
        self.num_expand = num_expand
        
    def fit(self, first_even,
            first_odd,
            second_even,
            second_odd,
            clebsch = None):
        
        all_keys = list(first_even.keys()) + list(first_odd.keys()) + \
                   list(second_even.keys()) + list(second_odd.keys())
        all_keys = [int(el) for el in all_keys]
        
        self.l_max = np.max(all_keys + [self.lambda_max])
        
        if clebsch is None:
            self.clebsch = ClebschGordan(self.l_max).precomputed_
        else:
            self.clebsch = clebsch
            
        if self.num_expand is not None:
            first_even_idx = get_sorting_indices(first_even)
            first_odd_idx = get_sorting_indices(first_odd)
            second_even_idx = get_sorting_indices(second_even)
            second_odd_idx = get_sorting_indices(second_odd)
    
            first_even = apply_indices(first_even, first_even_idx)
            first_odd = apply_indices(first_odd, first_odd_idx)
            second_even = apply_indices(second_even, second_even_idx)
            second_odd = apply_indices(second_odd, second_odd_idx)
          
            task_even_even, task_odd_odd, task_even_odd, task_odd_even = \
                get_thresholded_tasks(first_even, first_odd, second_even, second_odd, self.num_expand,
                                      self.l_max, self.lambda_max)
            
            
            task_even_even = convert_task(task_even_even, self.l_max, self.lambda_max,
                                         first_even_idx, second_even_idx)
            task_odd_odd = convert_task(task_odd_odd, self.l_max, self.lambda_max,
                                        first_odd_idx, second_odd_idx)
            task_even_odd = convert_task(task_even_odd, self.l_max, self.lambda_max,
                                        first_even_idx, second_odd_idx)
            task_odd_even = convert_task(task_odd_even, self.l_max, self.lambda_max,
                                        first_odd_idx, second_even_idx)
            
            self.has_tasks = True
        else:
            self.has_tasks = False
            
        if self.has_tasks:
            self.even_even_comb = ClebschCombining(self.clebsch, self.lambda_max, task =
                                                   task_even_even)
            self.odd_odd_comb = ClebschCombining(self.clebsch, self.lambda_max, task =
                                                   task_odd_odd)
            
            self.even_odd_comb = ClebschCombining(self.clebsch, self.lambda_max, task =
                                                   task_even_odd)
            
            self.odd_even_comb = ClebschCombining(self.clebsch, self.lambda_max, task =
                                                   task_odd_even)
            
        else:
            self.comb = ClebschCombining(self.clebsch, self.lambda_max)
            
        self.cov_cat = CovCat()
        
    def forward(self, first_even, first_odd, second_even, second_odd):
        if self.has_tasks:
            even_even = self.even_even_comb(first_even, second_even)
            odd_odd = self.odd_odd_comb(first_odd, second_odd)
            even_odd = self.even_odd_comb(first_even, second_odd)
            odd_even = self.odd_even_comb(first_odd, second_even)
        else:
            even_even = self.comb(first_even, second_even)
            odd_odd = self.comb(first_odd, second_odd)
            even_odd = self.comb(first_even, second_odd)
            odd_even = self.comb(first_odd, second_even)
        
        res_even = self.cov_cat([even_even, odd_odd])
        res_odd = self.cov_cat([even_odd, odd_even])
        return res_even, res_odd