# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 14:44:22 2022

@author: pcadm
"""

# Import relevant modules

import numpy as np
import time

## General methods

def gauss_noise_array(std_dev_array, num_replicas = 1):
    
    if num_replicas == 1:
        D = len(std_dev_array)
        S = std_dev_array
    else:
        D = (num_replicas, len(std_dev_array))
        S = np.tile(std_dev_array, (num_replicas, 1))
    
    return np.random.normal(0, S, D)

def SSE(x, y, model, theta, num_replicas = 1):
    if num_replicas == 1:
        y_model = model(x, theta)
        return sum((y_model - y) ** 2)
    else:
        y_model = np.zeros((num_replicas, len(y)))
        for ii in range(num_replicas):
            y_model[ii, :] = model(x, theta[ii, :])
        
        return np.sum((y_model-y) ** 2, axis=1)

# Function to give the probability of the measured data occurring given
# a certan parameter set

def P_D_theta(x, y, model, theta, sig_noise, num_replicas = 1):
    
    return -1 * SSE(x, y, model, theta, num_replicas) / (2 * sig_noise ** 2)

# Function to give the prior probability (which in our case is 0 whenever our
# parameter falls outside of an acceptable range and 1 otherwise)

def P_theta(theta, theta_low, theta_high, num_replicas = 1):
    if num_replicas == 1:
        answer = 1
        for ii in range(len(theta)):
            if theta[ii] < theta_low[ii] or theta[ii] > theta_high[ii]:
                answer = 0
    else:
        answer = np.ones(num_replicas)
        for jj in range(num_replicas):
            for ii in range(len(theta[0, :])):
                if theta[jj, ii] < theta_low[ii] or theta[jj, ii] > theta_high[ii]:
                    answer[jj] = 0
    
    return answer

# Function to give the probability of a set of parameters given a measured
# distribution
    
def P_theta_D(x, y, model, theta, sig_noise, theta_low, theta_high,
              num_replicas = 1):
    return P_D_theta(x, y, model, theta, sig_noise, num_replicas) *\
                P_theta(theta, theta_low, theta_high, num_replicas)

## Implement algorithm
    
def metropolis_step(x, y, model, theta_curr, sig_theta, theta_lower_bound,
                    theta_upper_bound, noise_level, num_replicas = 1,
                    output_diagnostics = False):
    
    # Generate guess parameters
    theta_p = theta_curr + gauss_noise_array(sig_theta, num_replicas)
    
    # Probability of original parameters
    p1 = P_theta_D(x, y, model, theta_p, noise_level, theta_lower_bound,
                   theta_upper_bound, num_replicas = num_replicas)
    
    # Probability of guess parameters
    p2 = P_theta_D(x, y, model, theta_curr, noise_level, theta_lower_bound,
                   theta_upper_bound, num_replicas = num_replicas)
    
    if num_replicas == 1:
        u = np.random.rand()
        if p1 > p2 and p1 != 0:
            theta_new = theta_p
        elif p1 < p2 and p1 != 0 and u < np.exp(p1 - p2):
            theta_new = theta_p
        else:
            theta_new = theta_curr
    else:
        u = np.random.rand(num_replicas)
        theta_new = np.zeros(np.shape(theta_curr))
        diagnostic = np.zeros(num_replicas)
        for ii in range(num_replicas):
            if p1[ii] > p2[ii] and p1[ii] != 0:
                theta_new[ii,:] = theta_p[ii,:]
                diagnostic[ii] = 1
            elif p1[ii] < p2[ii] and p1[ii] != 0 and u[ii] < np.exp(p1[ii] - p2[ii]):
                theta_new[ii,:] = theta_p[ii,:]
                diagnostic[ii] = 1
            else:
                theta_new[ii,:] = theta_curr[ii,:]
    
    if output_diagnostics:
        return theta_new, diagnostic
    else:
        return theta_new

def metropolis(x, y, model, theta_init, sig_theta, noise_level,
               theta_lower_bound, theta_upper_bound, num_iter = 100000):
    
    # Initialize array of model parameters
    theta_array = np.zeros([num_iter, len(theta_init)])
    theta_array[0,:] = theta_init
    
    for ii in range(num_iter - 1):
        theta_curr = theta_array[ii,:]
        theta_array[ii + 1,:] = metropolis_step(x, y, model, theta_curr,
                                sig_theta, theta_lower_bound, theta_upper_bound,
                                noise_level)
        
    return theta_array


### INPUTS
#   x - The x values from your data
#   y - the y values from your data
#   model - The model you are trying to fit your data with, given
#           as a function
#   theta_init - Your initial guesses at your fit parameters
#   sig_theta - The standard deviations of your fit parameters
#   theta_low - The minimum values for your fit parameters
#   theta_high - The maximum values for your fit parameters
#   sig_array - An array containing the background noise levels you want to
#               test (not to be confused with sig_theta)
#   num_iter - The number of iterations
#
### OUTPUTS
#   theta_array_array - A 3D numpy array containing your fit parameters
#                       from every iteration. The first index of the array
#                       corresponds to wish replica you wish to access. The
#                       second corresponds to which iteration you wish to
#                       access. The third corresponds to which fit parameter
#                       you wish to access. For example, if you wanted to
#                       access the 5th fit parameter from the 48000th
#                       iteration of the 70th replica, you would put
#                       theta_array_array[70-1, 48000-1, 5-1]
#   p_D_theta_array_arry - A 2D numpy array giving the probabilities of a
#                          set of fit parameters occurring for each
#                          iteration of each replica. The first index
#                          indicates the replica, the second indicates the
#                          iteration. To access the probability of the
#                          80000th iteration of the 10th replica, you would put
#                          p_D_theta_array_array[10-1, 80000-1]
def replica_exchange(x, y, model, theta_init, sig_theta, theta_low, theta_high,
                     sig_array, num_iter = 100000):
    
    num_replicas = len(sig_array)
    num_params = len(theta_init)
    
    num_true_exchanges = 0
    num_rand_exchanges = 0
    num_exchange_opportunities = 0
    
    # Time diagnostics
    time_met_stepping = 0
    time_rep_exchanging = 0
    time_E_calcing = 0
    
    theta_array_array = np.zeros([num_replicas, num_iter, num_params])
    p_D_theta_array_array = np.zeros([num_replicas, num_iter])
    
    # Initialize parameter history and probability arrays
    theta_array_array[:,0,:] = np.tile(theta_init, (num_replicas, 1))
    p_D_theta_array_array[:,0] = SSE(x, y, model, theta_array_array[:,0,:],
                                     num_replicas) / 2
    num_acceptances = np.zeros(num_replicas)
    
    for idx in range(num_iter - 1):
        
        # Metropolis step
        met_step_ref = time.time()
        theta_curr = theta_array_array[:, idx, :].copy()
        theta_temp, temp_acceptance = metropolis_step(x, y, model, theta_curr,
                            sig_theta, theta_low, theta_high, sig_array,
                            len(sig_array), output_diagnostics = True)
        theta_array_array[:,idx+1,:] = theta_temp.copy()
        time_met_stepping += time.time() - met_step_ref
        num_acceptances += temp_acceptance
        
        # Calculate SSE's
        rep_exchange_ref = time.time()
        SSE_array = SSE(x, y, model, 
                                theta_array_array[:,idx+1,:],
                                num_replicas)
        # Replica exchange
        SSE1 = SSE_array[0]
        for ii in range(len(sig_array) - 1):
            theta_curr_1 = theta_array_array[ii, idx+1, :].copy()
            theta_curr_2 = theta_array_array[ii+1, idx+1, :].copy()
            SSE2 = SSE_array[ii+1]
            log_p12 = SSE1 / (2 * sig_array[ii + 1] ** 2)
            log_p21 = SSE2 / (2 * sig_array[ii] ** 2)
            log_p11 = SSE1 / (2 * sig_array[ii] ** 2)
            log_p22 = SSE2 / (2 * sig_array[ii + 1] ** 2)
            u = np.random.rand()
            
            num_exchange_opportunities += 1
            if log_p12 + log_p21 < log_p11 + log_p22:
                theta_array_array[ii, idx + 1,:] = theta_curr_2.copy()
                theta_array_array[ii+1, idx+1,:] = theta_curr_1.copy()
                SSE_array[ii] = SSE2
                SSE_array[ii+1] = SSE1
                num_true_exchanges += 1
            elif u < np.exp(log_p11 + log_p22 - log_p12 - log_p21):
                theta_array_array[ii, idx + 1,:] = theta_curr_2.copy()
                theta_array_array[ii+1, idx+1,:] = theta_curr_1.copy()
                num_rand_exchanges += 1
                SSE_array[ii] = SSE2
                SSE_array[ii+1] = SSE1
            else:
                SSE1 = SSE2
        time_rep_exchanging += time.time() - rep_exchange_ref
        
        # Calculate probabilities for new parameters
        E_calc_ref = time.time()
        p_D_theta_array_array[:,idx+1] = SSE_array / 2
        time_E_calcing += time.time() - E_calc_ref
    
    acceptance_ratios = np.round(100 * num_acceptances / num_iter, 2)
    
    def time_str(sec):
        if sec > 3600:
            h = int(sec // 3600)
            m = int((sec - h * 3600) // 60)
            s = round((sec - h * 3600 - m * 60))
            return f'{h} hours, {m} minutes and {s} seconds'
        elif sec > 60:
            m = int(sec // 60)
            s = round(sec - m * 60)
            return f'{m} minutes and {s} seconds'
        else:
            return f'{round(sec)} seconds'
    
    true_exchange_rat = round(100 * num_true_exchanges / num_exchange_opportunities, 2)
    rand_exchange_rat = round(100 * num_rand_exchanges / num_exchange_opportunities, 2)
    print(f'True exchanges: {true_exchange_rat}%')
    print(f'Random exchanges: {rand_exchange_rat}%')
    print('')
    print('Time Diagnostics:')
    print('----------------------------------')
    print(f'Time Spent Metropolis Stepping: {time_str(time_met_stepping)}')
    print(f'Time Spent Replica Exchanging: {time_str(time_rep_exchanging)}')
    print(f'Time Spent E Calculating: {time_str(time_E_calcing)}')
    print('')
    return theta_array_array, p_D_theta_array_array, acceptance_ratios
    
def Z_approx(p_D_theta_array_array, b_array, num_points, num_iter):
    Z_array = np.zeros(len(b_array))
    
    for ii in range(len(b_array)):
        Z_conj = np.mean(p_D_theta_array_array[ii][num_iter//2::])
        Z_array[ii] = Z_conj
    
    return Z_array

# INPUTS
#   E_array_array - This input will be the P_d_theta_array_array output from
#                   your replica exchange method. In the thermodynamic analogy,
#                   it corresponds to the "energy" of each possible "state."
#   b_array - An array of your b values, where b = 1 / sigma^2, with sigma
#             being the assumed background noise of a certain replica.
#   num_points - The number of points in your data set
#   num_iter - The number of iterations you used in your replica exchange
#              method
#
# OUTPUTS
#   answer - A 1D numpy array giving the value of F for each one of your
#            replicas.
def F_approx(E_array_array, b_array, num_points, num_iter):
    answer = np.zeros(len(b_array))
    for ii in range(len(b_array)):
        c = (num_points / 2) * np.log(b_array[ii] / (2 * np.pi))
        temp = 0
        for lp in range(0, ii):
            temp_arr = np.exp((b_array[lp] - b_array[lp+1]) * E_array_array[lp, num_iter//2::])
            temp += np.log(np.mean(temp_arr))
        answer[ii] = -1 * temp - c
    
    return answer

# Function to analyze the output of a replica exchange simulation.
# It takes the output theta_array_array and returns a list of the means
# and standard deviations for each fit parameter. The input replica_idx
# indicates the particular replica whose parameters you would like to
# summarize (i.e., the replica with the lowest F value)
def param_sum(theta_array_array, replica_idx, burn_in_ratio = 0.5):
    num_reps, num_iters, num_params = np.shape(theta_array_array)
    cutoff = int(round(burn_in_ratio * num_iters, 0))
    
    mean_lst = np.zeros(num_params)
    std_lst = np.zeros(num_params)
    
    for ii in range(num_params):
        mean_lst[ii] = np.mean(theta_array_array[replica_idx, cutoff::, ii])
        std_lst[ii] = np.std(theta_array_array[replica_idx, cutoff::, ii])
    
    return mean_lst, std_lst
    
    
    