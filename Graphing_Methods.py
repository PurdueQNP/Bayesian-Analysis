# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from datetime import date
from scipy.optimize import curve_fit
from scipy.special import wofz
from readSpe import readSpe

# Function to handle generic curve fitting

def get_fit(x, y, model, guess):
    best_vals, covar = curve_fit(model, x, y, p0 = guess)
    
    return best_vals, covar


# Function to interpret data for a given directory and file name
def get_integrated_info(directory, name, normalized = False,
                        offline = True):
    
    if offline:
        divider = "\\"
    else:
        divider = "/"
    
    path = directory + divider + name + ".spe"
    data_struct = readSpe(path)
    lambdas = data_struct.wavelengths
    counts = np.zeros(len(lambdas))
    
    num_frames = np.shape(data_struct.data)[1]
    for ii in range(num_frames):
        counts = counts + data_struct.data[0][ii][0][:]
    
    if normalized:
        counts = counts / max(counts)
    
    return lambdas, counts

# Function to take moving average of a data set
def moving_avg(x_array, y_array, screen_size = 200):
    screen_x = np.zeros(len(x_array) - screen_size - 1)
    screen_y = np.zeros(len(x_array) - screen_size - 1)
    for ii in range(len(x_array) - screen_size - 1):
        screen_x[ii] = x_array[ii + screen_size // 2]
        screen_y[ii] = np.mean(y_array[ii:ii + screen_size + 1])
    
    return screen_x, screen_y

# Function to plot data (can take up to 3 full data sets)
def plot_data(wavelength_lst, intensity_lst, screened_wavelength_lst,
                        screened_intensity_lst,
                        cutoff_lams = np.array([]), plot_mode = 'nm',
                        xlim = None, title = None, legend_lst = None,
                        screened_legend_lst = None,
                        discont_linewidth = 1.5, cont_linewidth = 2,
                        ylabel_text = 'Intensity (arb.)',
                        custom_lam_ticks = None, custom_ev_ticks = None,
                        custom_yticks = None):
    
    cutoff_lams_act = cutoff_lams.copy()
    
    # Unpack legend
    label_lst = [0] * (len(wavelength_lst) + len(screened_wavelength_lst))
    c = 0
    if legend_lst is not None:
        for ii in range(len(legend_lst)):
            label_lst[ii] = legend_lst[ii]
            c += 1
    else:
        for ii in range(len(wavelength_lst)):
            label_lst[ii] = ''
            c += 1
    
    if screened_legend_lst is not None:
        for ii in range(len(screened_legend_lst)):
            label_lst[c + ii] = screened_legend_lst[ii]
    else:
        for ii in range(len(screened_wavelength_lst)):
            label_lst[c + ii] = ''
        
    
    # Determine x limits
    if xlim is None:
        xlim_min = 1000000
        xlim_max = -1
        for ii in range(len(wavelength_lst)):
            if wavelength_lst[ii][0] < xlim_min:
                xlim_min = wavelength_lst[ii][0]
            if wavelength_lst[ii][-1] > xlim_max:
                xlim_max = wavelength_lst[ii][-1]
        for ii in range(len(screened_wavelength_lst)):
            if screened_wavelength_lst[ii][0] < xlim_min:
                xlim_min = screened_wavelength_lst[ii][0]
            if screened_wavelength_lst[ii][-1] > xlim_max:
                xlim_max = screened_wavelength_lst[ii][-1]
        
        xlim_act = np.array([xlim_min, xlim_max])
    else:
        xlim_min = xlim[0]
        xlim_max = xlim[-1]
        xlim_act = np.array([xlim_min, xlim_max])
    
    # Determine y limits
    
    low_lam = np.min(xlim_act)
    high_lam = np.max(xlim_act)
    y_min = 100000
    y_max = -1
    low_idx = 0
    high_idx = 0
    for ii in range(len(wavelength_lst)):
        temp_low_idx = 0
        while wavelength_lst[ii][temp_low_idx] < low_lam and temp_low_idx < len(wavelength_lst[ii]) - 1:
            temp_low_idx += 1
        
        temp_high_idx = 0
        while wavelength_lst[ii][temp_high_idx] < high_lam and temp_high_idx < len(wavelength_lst[ii]) - 1:
            temp_high_idx += 1
        
        temp_min = np.min(intensity_lst[ii][temp_low_idx:temp_high_idx])
        temp_max = np.max(intensity_lst[ii][temp_low_idx:temp_high_idx])
        
        if temp_min < y_min:
            y_min = temp_min
        if temp_max > y_max:
            y_max = temp_max
    
    for ii in range(len(screened_wavelength_lst)):
        temp_low_idx = 0
        while screened_wavelength_lst[ii][temp_low_idx] < low_lam and temp_low_idx < len(screened_wavelength_lst[ii]) - 1:
            temp_low_idx += 1
        
        temp_high_idx = 0
        while screened_wavelength_lst[ii][temp_high_idx] < high_lam and temp_high_idx < len(screened_wavelength_lst[ii]) - 1:
            temp_high_idx += 1
        
        temp_min = np.min(screened_intensity_lst[ii][temp_low_idx:temp_high_idx])
        temp_max = np.max(screened_intensity_lst[ii][temp_low_idx:temp_high_idx])
        
        if temp_min < y_min:
            y_min = temp_min
        if temp_max > y_max:
            y_max = temp_max

    yrange = abs(y_max - y_min)
    ylim_min = y_min - 0.05 * yrange
    ylim_max = y_max + 0.05 * yrange
    
    ylim_act = np.array([ylim_min, ylim_max])
    
    # Convert to eV's if necessary
    wavelength_lst_act = wavelength_lst.copy()
    screened_wavelength_lst_act = screened_wavelength_lst.copy()
    if plot_mode.lower() != 'nm':
        xlim_act = np.flip(6.626e-34 * 2.998e8 / (1.602e-19 * xlim_act * 1e-9))
        cutoff_lams_act = 6.626e-34 * 2.998e8 / (1.602e-19 * cutoff_lams_act * 1e-9)
        for ii in range(len(wavelength_lst_act)):
            wavelength_lst_act[ii] = 6.626e-34 * 2.998e8 / (1.602e-19 * wavelength_lst_act[ii] * 1e-9)
        for jj in range(len(screened_wavelength_lst_act)):
            screened_wavelength_lst_act[jj] = 6.626e-34 * 2.998e8 / (1.602e-19 * screened_wavelength_lst_act[jj] * 1e-9)
    
    
    # List of useable colors
    colors = ['blue', 'green', 'black', 'magenta', 'red']
    screened_colors = ['red', 'magenta', 'cyan', 'goldenrod']
    
    fig = plt.figure(figsize = (10,10))
    ax1 = fig.add_subplot(1,1,1)
    if plot_mode == 'multi':
        ax2 = ax1.twiny()
        ax2_range = 6.626e-34 * 2.998e8 / (1.602e-19 * xlim_act * 1e-9)
        ax2.plot(ax2_range, [0,0], linestyle = 'none')
        ax2.set_xlim(ax2_range)
    
    # Plot data and cutoff wavelengths
    c = 0
    for ii in range(len(wavelength_lst)):
        ax1.plot(wavelength_lst_act[ii], intensity_lst[ii],
                 linewidth = discont_linewidth, color = colors[ii],
                 label = label_lst[ii])
        c += 1
    for ii in range(len(screened_wavelength_lst)):
        ax1.plot(screened_wavelength_lst_act[ii], screened_intensity_lst[ii],
                 linewidth = cont_linewidth, color = screened_colors[ii],
                 label = label_lst[c + ii])
    for cutoff_lam in cutoff_lams_act:
        ax1.plot([cutoff_lam, cutoff_lam], [-10000, 100000], color = 'orange', linewidth = 2, linestyle = 'dashed')
    
    # Format graph
    if plot_mode != 'multi':
        ax1.grid(True)
    if custom_yticks is not None:
        ax1.set_yticks(custom_yticks)
    ax1.tick_params(axis = 'both', labelsize = 20)
    ax1.set_xlim(xlim_act)
    ax1.set_ylim(ylim_act)
    if plot_mode == 'multi':
        if custom_lam_ticks is not None:
            ax2.set_xticks(custom_lam_ticks)
        if custom_ev_ticks is not None:
            ax1.set_xticks(custom_ev_ticks)
        ax2.tick_params(axis = 'x', labelsize = 20)
    
    # labels
    if plot_mode.lower() != 'nm':
        ax1.set_xlabel('Energy (eV)', size = 24)
        if plot_mode == 'multi':
            ax2.set_xlabel('Wavelength (nm)', size = 24, labelpad = 10)
    else:
        ax1.set_xlabel('Wavelength (nm)', size = 24)
    
    ax1.set_ylabel(ylabel_text, size = 24)
    
    if title is not None:
        if plot_mode != 'multi':
            ax1.set_title(title, size = 28)
        else:
            ax2.set_title(title, size = 28, pad = 20)
    
    if legend_lst is not None or screened_legend_lst is not None:
        ax1.legend(prop={"size":20})
    
    if plot_mode == 'multi':
        return fig, ax1, ax2
    else:
        return fig, ax1

## Functions for peak finding

# Function for estimating the center wavelength of a point (based on minimum of moving average)
def center_lam_seek(true_wavelength, true_intensity, screened_wavelength, screened_intensity, min_lam, max_lam, mode = 'nm',
                   cutoff_lams = []):
    
    true_wavelength_temp = true_wavelength.copy()
    true_intensity_temp = true_intensity.copy()
    wavelength_temp = screened_wavelength.copy()
    intensity_temp = screened_intensity.copy()
    
    if mode.lower() == "ev":
        wavelength_temp = 6.626e-34 * 2.998e8 / (1.602e-19 * wavelength_temp * 1e-9)
        true_wavelength_temp = 6.626e-34 * 2.998e8 / (1.602e-19 * true_wavelength_temp * 1e-9)
    
    if mode.lower() == "ev":
        low_idx = 0
        while wavelength_temp[low_idx] > min_lam:
            low_idx += 1
    
        high_idx = 0
        while wavelength_temp[high_idx] > max_lam:
            high_idx += 1
        
        wavelength_roi = wavelength_temp[high_idx:low_idx]
        intensity_roi = intensity_temp[high_idx:low_idx]
        
        low_idx = 0
        while true_wavelength_temp[low_idx] > min_lam:
            low_idx += 1
    
        high_idx = 0
        while true_wavelength_temp[high_idx] > max_lam:
            high_idx += 1
        
        true_wavelength_roi = true_wavelength_temp[high_idx:low_idx]
        true_intensity_roi = true_intensity_temp[high_idx:low_idx]

    else:
        low_idx = 0
        while wavelength_temp[low_idx] < min_lam:
            low_idx += 1
    
        high_idx = 0
        while wavelength_temp[high_idx] < max_lam:
            high_idx += 1
        
        wavelength_roi = wavelength_temp[low_idx:high_idx]
        intensity_roi = intensity_temp[low_idx:high_idx]
        
        low_idx = 0
        while true_wavelength_temp[low_idx] < min_lam:
            low_idx += 1
    
        high_idx = 0
        while true_wavelength_temp[high_idx] < max_lam:
            high_idx += 1
        
        true_wavelength_roi = true_wavelength_temp[low_idx:high_idx]
        true_intensity_roi = true_intensity_temp[low_idx:high_idx]
    
    min_seek = np.min(intensity_roi)
    seeker = 0
    while intensity_roi[seeker] != min_seek:
        seeker += 1
    
    peak_lam = wavelength_roi[seeker]
    
    if mode.lower() == "ev":
        title_str = "Peak Energy = " + str(round(peak_lam, 5)) + " eV"
    else:
        title_str = "Peak Wavelength = " + str(round(peak_lam, 2)) + " nm"
    
    # Create output plot
    
    if mode.lower() == "ev":
        plt_xlim = [6.626e-34 * 2.998e8 / (1.602e-19 * max_lam * 1e-9), 
                   6.626e-34 * 2.998e8 / (1.602e-19 * min_lam * 1e-9)]
    else:
        plt.xlim = [min_lam, max_lam]
    
    plot_data([true_wavelength],
                     [true_intensity],
                     [screened_wavelength],
                     [screened_intensity],
                        cutoff_lams = cutoff_lams, plot_mode = mode, xlim = plt_xlim, legend_lst = ['Absorption Spectrum'],
                        screened_legend_lst = ['Moving Average'],
                        title = title_str)
    
