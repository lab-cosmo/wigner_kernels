from libc.math cimport sin, M_PI, sqrt, fmax
cimport cython
import numpy as np
#from cython.parallel cimport prange
cdef double sqrt_2 = sqrt(2.0)
#from nice_utilities import Data


    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int min_c(int a, int b) nogil:
    if (a < b):
        return a
    else:
        return b
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int max_c(int a, int b) nogil:
    if (a > b):
        return a
    else:
        return b
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int abs_c(int a) nogil:
    if (a >= 0):
        return a
    else:
        return -a


cpdef get_thresholded_task(double[:, :] first_importances, int[:] first_actual_sizes,
                           double[:, :] second_importances, int[:] second_actual_sizes,
                           double threshold, int known_num, int l_max, int lambda_max):
    ans = np.empty([known_num, 4], dtype = np.int32)
    
    raw_importances = np.empty([known_num])
    
    cdef int[:, :] ans_view = ans
    
    cdef int l1, l2, first_ind, second_ind, lambd
    cdef int pos = 0
   
    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):
            if (abs_c(l1 - l2) <= lambda_max):
                for first_ind in range(first_actual_sizes[l1]):
                    for second_ind in range(second_actual_sizes[l2]):
                        if (first_importances[first_ind, l1] * second_importances[second_ind, l2] >= threshold):                     
                            ans_view[pos, 0] = first_ind
                            ans_view[pos, 1] = l1
                            ans_view[pos, 2] = second_ind
                            ans_view[pos, 3] = l2                           
                            raw_importances[pos] = first_importances[first_ind, l1] * second_importances[second_ind, l2]
                            pos += 1
   
    return [ans[:pos], raw_importances[:pos]]      
    
    
def get_amplitudes(covariants, l_max):
    max_num = None
    for l in covariants.keys():
        if int(l) > l_max:
            raise ValueError("insufficient l_max")
        if (max_num is None) or (covariants[l].shape[1] > max_num):
            max_num = covariants[l].shape[1]
    result = -1.0 * np.ones([max_num, l_max + 1])
    nums = np.zeros(l_max + 1, dtype = np.int32)
    for l in covariants.keys():
        squares = covariants[l].data.cpu().numpy() ** 2
        amplitudes = np.mean(squares.sum(axis = 2), axis = 0)
        for index in range(1, len(amplitudes)):
            if amplitudes[index] > amplitudes[index - 1]:
                raise ValueError("covariants should be sorted in descending order of their variance")
        result[:covariants[l].shape[1], int(l)] = amplitudes
        
        nums[int(l)] = covariants[l].shape[1]
    return result, nums

cpdef get_thresholded_tasks(first_even, first_odd, second_even, second_odd, int desired_num, int l_max, int lambda_max):
    
    
  
    cdef double threshold_even
    cdef int num_even_even, num_odd_odd
    threshold_even, num_even_even, num_odd_odd = get_threshold(*get_amplitudes(first_even, l_max),
                                                               *get_amplitudes(second_even, l_max),
                                                               *get_amplitudes(first_odd, l_max),
                                                               *get_amplitudes(second_odd, l_max),
                                                               desired_num, lambda_max)
    
    cdef double threshold_odd
    cdef int num_even_odd, num_odd_even
    threshold_odd, num_even_odd, num_odd_even = get_threshold(*get_amplitudes(first_even, l_max),
                                                              *get_amplitudes(second_odd, l_max),
                                                              *get_amplitudes(first_odd, l_max),
                                                              *get_amplitudes(second_even, l_max),
                                                              desired_num, lambda_max)        
      
    
    
    task_even_even = get_thresholded_task(*get_amplitudes(first_even, l_max),
                                          *get_amplitudes(second_even, l_max),
                                          threshold_even, num_even_even, l_max, lambda_max)
    
    task_odd_odd = get_thresholded_task(*get_amplitudes(first_odd, l_max),
                                        *get_amplitudes(second_odd, l_max),
                                        threshold_even, num_odd_odd, l_max, lambda_max)
    
    task_even_odd = get_thresholded_task(*get_amplitudes(first_even, l_max),
                                         *get_amplitudes(second_odd, l_max),
                                         threshold_odd, num_even_odd, l_max, lambda_max)
    
    task_odd_even = get_thresholded_task(*get_amplitudes(first_odd, l_max),
                                         *get_amplitudes(second_even, l_max),
                                         threshold_odd, num_odd_even, l_max, lambda_max)
    
    return task_even_even, task_odd_odd, task_even_odd, task_odd_even
                           
                           
                           
cpdef get_threshold(double[:, :] first_importances_1, int[:] first_actual_sizes_1,
                   double[:, :] second_importances_1, int[:] second_actual_sizes_1,
                   double[:, :] first_importances_2, int[:] first_actual_sizes_2,
                   double[:, :] second_importances_2, int[:] second_actual_sizes_2,
                   int desired_num, int lambda_max, int min_iterations = 50):
    
    
    if (desired_num == -1):
        num_1_1 = get_total_num_full(first_importances_1, first_actual_sizes_1, second_importances_1, second_actual_sizes_1, -1.0, lambda_max)  
        num_2_2 = get_total_num_full(first_importances_2, first_actual_sizes_2, second_importances_2, second_actual_sizes_2, -1.0, lambda_max)  
        return -1.0, num_1_1, num_2_2
    
    cdef double left = -1.0
    cdef double first = get_upper_threshold(first_importances_1, first_actual_sizes_1, second_importances_1, second_actual_sizes_1, lambda_max) + 1.0
    cdef double second = get_upper_threshold(first_importances_2, first_actual_sizes_2, second_importances_2, second_actual_sizes_2, lambda_max) + 1.0
    
    cdef double right = fmax(first, second)
    cdef double middle = (left + right) / 2.0
    cdef int num_now, num_previous = -1
    cdef int num_it_no_change = 0
    while (True):
        middle = (left + right) / 2.0
        num_now = get_total_num_full(first_importances_1, first_actual_sizes_1, second_importances_1, second_actual_sizes_1, middle, lambda_max) + get_total_num_full(first_importances_2, first_actual_sizes_2, second_importances_2, second_actual_sizes_2, middle, lambda_max)
        
        if (num_now == desired_num):
            left = middle
            break
        if (num_now > desired_num):
            left = middle
        if (num_now < desired_num):
            right = middle
            
        if (num_now == num_previous):
            num_it_no_change += 1
            if (num_it_no_change > min_iterations):
                break
        else:
            num_it_no_change = 0
        num_previous = num_now
            
    num_1_1 = get_total_num_full(first_importances_1, first_actual_sizes_1, second_importances_1, second_actual_sizes_1, left, lambda_max)  
    num_2_2 = get_total_num_full(first_importances_2, first_actual_sizes_2, second_importances_2, second_actual_sizes_2, left, lambda_max)  
    return left, num_1_1, num_2_2


cdef double get_upper_threshold(double[:, :] first_importances, int[:] first_actual_sizes, 
                             double[:, :] second_importances, int[:] second_actual_sizes,
                                int lambda_max):
    cdef double ans = 0.0
    cdef int l1, l2
        
    cdef int second_size = second_importances.shape[1]
    for l1 in range(first_importances.shape[1]):
        for l2 in range(second_size):
            if (abs_c(l1 - l2) <= lambda_max):
                if (first_actual_sizes[l1] > 0) and (second_actual_sizes[l2] > 0):
                    if (first_importances[0, l1] * second_importances[0, l2] > ans):
                        ans = first_importances[0, l1] * second_importances[0, l2]
                    
    return ans
                                 
cdef int get_total_num_full(double[:, :] first_importances, int[:] first_actual_sizes,
                            double[:, :] second_importances, int[:] second_actual_sizes,
                            double threshold, int lambda_max):
    
    cdef int result = 0
    cdef int l1, l2
    cdef int second_size = second_importances.shape[1]
    cdef int lambd
    for l1 in range(first_importances.shape[1]):
        for l2 in range(second_size):
            if (first_actual_sizes[l1] > 0) and (second_actual_sizes[l2] > 0):
                if (abs_c(l1 - l2) <= lambda_max):
                    result += get_total_num(first_importances[:first_actual_sizes[l1], l1],
                                     second_importances[:second_actual_sizes[l2], l2], threshold)
             
    return result
    

        
cdef int get_total_num(double[:] a, double[:] b, double threshold):
    cdef int b_size = b.shape[0]
    cdef int i, j, ans
    i = 0
    j = b_size
    ans = 0
    for i in range(a.shape[0]):
        while ((j > 0) and (a[i] * b[j - 1] < threshold)):
            j -= 1
        ans += j
    return ans
