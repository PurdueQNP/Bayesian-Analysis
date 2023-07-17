# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 09:49:48 2022

@author: pcadm
"""

## Import relevant modules

import numpy as np
from scipy.special import wofz
import scipy.interpolate as interpolate

kb = 1.3806e-23
c = 2.998e8
e = 1.602e-19
h = 6.626e-34
hbar = h / (2 * np.pi)

# Lorentzian model
def lorentz(x, theta):
    
    x0 = theta[0]
    FWHM = theta[1]
    mult = theta[2]
    offset = theta[3]
    
    gam  = FWHM / 2
    return offset + mult * (1 / (np.pi * gam * (1 + ((x-x0)/gam)**2)))

# Voigt model
def voigt(x, theta):
    
    x0 = theta[0]
    sigma = theta[1]
    FWHM = theta[2]
    mult = theta[3]
    offset = theta[4]
    
    z = (x - x0 + 1j * FWHM/2) / (np.sqrt(2) * sigma)
    return offset + mult * np.real(wofz(z)) / (np.sqrt(2 * np.pi) * sigma)

# Fano-voigt model
def fano_voigt(x, theta):
    
    x0 = theta[0]
    q = theta[1]
    sigma = theta[2]
    FWHM = theta[3]
    height = theta[4]
    offset = theta[5]
    
    mult = height / (q ** 2 + 1)
    
    z = (x - x0 + 1j * FWHM/2) / (np.sqrt(2) * sigma)
    return offset + mult * (1 / (sigma * np.sqrt(2 * np.pi))) * ((q**2 - 1) * np.real(wofz(z)) - 2*q * np.imag(wofz(z)))

# Urbach Tail
def urbach(x, a0, x0, xu):
    return a0 * np.exp((x - x0) / xu)

# Maxwell-Boltzmann
def max_boltz(x, height, x0, T):
    norm = 1 / np.sqrt(0.5 * kb * T / np.exp(1))
    out = height * norm * np.sqrt((x - x0) * (x > x0) * e)
    out *= np.exp(-(x - x0) * (x > x0) * e / (kb * T))
    out *= np.sqrt(x)
    return out

# Normalized Gaussian
def gauss(x, x0, sig):
    return np.exp(-(x - x0) ** 2 / (2 * sig ** 2)) / (sig * np.sqrt(2 * np.pi))

# Gaussian convolved with Maxwell-Boltzmann dist
def gauss_max_boltz(x, height, x0, T, sig):
    # Find point where the Maxwell Boltzmann distribution drops below
    # 1-billionth of its peak value
    xmax = x0 + kb * T / (2 * e)
    step = -1 * kb * T * np.log(0.99) / e
    while max_boltz(xmax, height, x0, T) > height * 1e-9:
        xmax += step
    
    # Find R (how far we need to ensure we have captured the tail of both
    # the Maxwell-Boltzmann and Gaussian distributions)
    R = max(xmax - x0, 10 * sig)
    
    # Put a margin of error to ensure that there is adequate room for the
    # convolution to be performed for each point of interest
    x_lower = np.min(x) - R
    x_upper = np.max(x) + R
    x_mid = 0.5 * (x_lower + x_upper)
    
    # Find number of subdivisions (N) such that the distance between divisions
    # (dx) is < sig / 100
    N = int(round(1 + (x_upper - x_lower) * 100 / sig))
    dx = (x_upper - x_lower) / N
    
    # Generate x points
    x_temp = np.linspace(x_lower, x_upper, N)
    
    # Generate y points for each distribution and then convolve them
    max_boltz_temp = max_boltz(x_temp, height, x0, T)
    gauss_temp = gauss(x_temp, x_mid, sig)
    conv_temp = np.convolve(max_boltz_temp, gauss_temp, 'same') * dx
    
    conv_interp = interpolate.interp1d(x_temp, conv_temp, kind = 'cubic')
    
    return conv_interp(x)