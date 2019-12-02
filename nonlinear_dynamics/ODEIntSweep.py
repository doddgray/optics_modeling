# -*- coding: utf-8 -*-

################################################################################
################################################################################
##
##            Function library for concurrent ODE integration
##            for modeling of nonlinear optical cavity dynamics
##                     Copyright Dodd Gray 2019
##
################################################################################
################################################################################

from os import path, makedirs, chmod
import stat
import subprocess as subp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import  Normalize
from datetime import datetime
from time import time
import pickle
from glob import glob
import os
from multiprocessing import Pool
from instrumental import u
from math import log10, floor
import socket
from pathlib import Path
import shutil
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from scipy.integrate import solve_ivp

hostname = socket.gethostname()
if hostname=='dodd-laptop':
    data_dir = "/home/dodd/google-drive/notebooks/IMEC V1 1550nm ring measurements/Thermal SOI ring cavity stability analysis"
    n_proc_def = 7
else: # assume I'm on a MTL server or something
    home = str( Path.home() )
    data_dir = home+'/data/'
    n_proc_def = 30


def n_sig_figs(x,n):
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))




def f_DrivenCavity(t,y,p):
    # unpack paramater dict "p"
    Δ = p['Δ']
    s = p['s']
    δ_r = p['γ']
    μ = p['μ']
    r = p['r']
    ζ = p['ζ']
    η = p['η']
    χ = p['χ']
    τ_fc = p['τ_fc']
    τ_th = p['τ_th']
    a_c, a_c_star, a_s, a_s_star, n, T = y
    d_a_c = ( ( -0.5 + 1j*Δ + 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + (1j - r) * (a_c * a_c_star) + 1j*T ) * a_c + s
    d_a_c_star = np.conjugate( ( -0.5 + 1j*Δ + 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + (1j - r) * (a_c * a_c_star) + 1j*T ) * a_c_star + s
    d_a_s = ( ( -0.5 + 1j*Δ - 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + (1j - r) * (a_s * a_s_star) + 1j*T ) * a_s + s
    d_a_s_star = np.conjugate( ( -0.5 + 1j*Δ - 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + (1j - r) * (a_s * a_s_star) + 1j*T ) * a_s_star + s
    d_n = -n / τ_fc + χ * ( (a_c*a_c_star)**2 + (a_s*a_s_star)**2 )
    d_T = -T / τ_th + ζ * ( η * r * ( (a_c*a_c_star)**2 + (a_s*a_s_star)**2 ) + n / μ * ( (a_c*a_c_star) + (a_s*a_s_star) ) )
    return [d_a_c, d_a_c_star, d_a_s, d_a_s_star , d_n, d_T]

def jac_DrivenCavity(t,y,p):
    # unpack paramater dict "p"
    Δ = p['Δ']
    s = p['s']
    δ_r = p['γ']
    μ = p['μ']
    r = p['r']
    ζ = p['ζ']
    η = p['η']
    χ = p['χ']
    τ_fc = p['τ_fc']
    τ_th = p['τ_th']
    a_c, a_c_star, a_s, a_s_star, n, T = y
    d_d_a_c = [ ( ( -0.5 + 1j*Δ + 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + 2 * (1j - r) * (a_c * a_c_star) + 1j*T ), (1j - r) * a_c**2, 0, 0, ( -1/μ - 1j )*a_c, 1j*T*a_c ]
    d_d_a_c_star = [ (-1j - r) * a_c_star**2, np.conjugate( ( -0.5 + 1j*Δ + 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + 2 * (1j - r) * (a_c * a_c_star) + 1j*T ), 0, 0, ( -1/μ + 1j )*a_c_star, -1j*T*a_c_star ]
    d_d_a_s = [ 0, 0, ( ( -0.5 + 1j*Δ - 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + 2 * (1j - r) * (a_s * a_s_star) + 1j*T ), (1j - r) * a_s**2, ( -1/μ - 1j )*a_s, 1j*T*a_s ]
    d_d_a_s_star = [ 0, 0, (-1j - r) * a_s_star**2, np.conjugate( ( -0.5 + 1j*Δ - 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + 2 * (1j - r) * (a_s * a_s_star) + 1j*T ), ( -1/μ + 1j )*a_s_star, -1j*T*a_s_star ]
    d_d_n = [ 2 * χ * a_c_star**2 * a_c , 2 * χ * a_c**2 * a_c_star , 2 * χ * a_s_star**2 * a_s , 2 * χ * a_s**2 * a_s_star , -1/τ_fc, 0 ]
    d_d_T = [ ζ*(2*η*r*a_c_star**2*a_c + n/μ*a_c_star) , ζ*(2*η*r*a_c_star*a_c**2 + n/μ*a_c) , ζ*(2*η*r*a_s_star**2*a_s + n/μ*a_s_star) , ζ*(2*η*r*a_s_star*a_s**2 + n/μ*a_s), ζ/μ * ( (a_c*a_c_star) + (a_s*a_s_star) ), -1./τ_th ]
    return [d_d_a_c, d_d_a_c_star, d_d_a_s, d_d_a_s_star, d_d_n, d_d_T]

def f_tuning(t,y,p,dΔdt):
    # unpack paramater dict "p"
    s = p['s']
    δ_r = p['γ']
    μ = p['μ']
    r = p['r']
    ζ = p['ζ']
    η = p['η']
    χ = p['χ']
    τ_fc = p['τ_fc']
    τ_th = p['τ_th']
    # dΔdt = p['dΔdt']
    # define state vector
    Δ, a_c, a_c_star, a_s, a_s_star, n, T = y
    # define equation system
    d_Δ = dΔdt
    d_a_c = ( ( -0.5 + 1j*Δ + 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + (1j - r) * (a_c * a_c_star) + 1j*T ) * a_c + s
    d_a_c_star = np.conjugate( ( -0.5 + 1j*Δ + 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + (1j - r) * (a_c * a_c_star) + 1j*T ) * a_c_star + s
    d_a_s = ( ( -0.5 + 1j*Δ - 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + (1j - r) * (a_s * a_s_star) + 1j*T ) * a_s + s
    d_a_s_star = np.conjugate( ( -0.5 + 1j*Δ - 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + (1j - r) * (a_s * a_s_star) + 1j*T ) * a_s_star + s
    d_n = -n / τ_fc + χ * ( (a_c*a_c_star)**2 + (a_s*a_s_star)**2 )
    d_T = -T / τ_th + ζ * ( η * r * ( (a_c*a_c_star)**2 + (a_s*a_s_star)**2 ) + n / μ * ( (a_c*a_c_star) + (a_s*a_s_star) ) )
    return [d_Δ, d_a_c, d_a_c_star, d_a_s, d_a_s_star , d_n, d_T]

def jac_tuning(t,y,p,dΔdt):
    # not yet fixed to incorporate tuning, don't use
    # unpack paramater dict "p"
    Δ = p['Δ']
    s = p['s']
    δ_r = p['γ']
    μ = p['μ']
    r = p['r']
    ζ = p['ζ']
    η = p['η']
    χ = p['χ']
    τ_fc = p['τ_fc']
    τ_th = p['τ_th']
    # dΔdt = p['dΔdt']
    Δ, a_c, a_c_star, a_s, a_s_star, n, T = y
    d_d_Δ = [ 0., 0., 0., 0., 0., 0., 0.]
    d_d_a_c = [ 0., ( ( -0.5 + 1j*Δ + 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + 2 * (1j - r) * (a_c * a_c_star) + 1j*T ), (1j - r) * a_c**2, 0, 0, ( -1/μ - 1j )*a_c, 1j*T*a_c ]
    d_d_a_c_star = [ 0., (-1j - r) * a_c_star**2, np.conjugate( ( -0.5 + 1j*Δ + 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + 2 * (1j - r) * (a_c * a_c_star) + 1j*T ), 0, 0, ( -1/μ + 1j )*a_c_star, -1j*T*a_c_star ]
    d_d_a_s = [ 0., 0, 0, ( ( -0.5 + 1j*Δ - 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + 2 * (1j - r) * (a_s * a_s_star) + 1j*T ), (1j - r) * a_s**2, ( -1/μ - 1j )*a_s, 1j*T*a_s ]
    d_d_a_s_star = [ 0., 0, 0, (-1j - r) * a_s_star**2, np.conjugate( ( -0.5 + 1j*Δ - 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + 2 * (1j - r) * (a_s * a_s_star) + 1j*T ), ( -1/μ + 1j )*a_s_star, -1j*T*a_s_star ]
    d_d_n = [ 0., 2 * χ * a_c_star**2 * a_c , 2 * χ * a_c**2 * a_c_star , 2 * χ * a_s_star**2 * a_s , 2 * χ * a_s**2 * a_s_star , -1/τ_fc, 0 ]
    d_d_T = [ 0., ζ*(2*η*r*a_c_star**2*a_c + n/μ*a_c_star) , ζ*(2*η*r*a_c_star*a_c**2 + n/μ*a_c) , ζ*(2*η*r*a_s_star**2*a_s + n/μ*a_s_star) , ζ*(2*η*r*a_s_star*a_s**2 + n/μ*a_s), ζ/μ * ( (a_c*a_c_star) + (a_s*a_s_star) ), -1./τ_th ]
    return [d_d_Δ, d_d_a_c, d_d_a_c_star, d_d_a_s, d_d_a_s_star, d_d_n, d_d_T]



################################################################################
##
##           functions for parameter calculations and
##          normalized <--> unitful parameter conversion
##
################################################################################

def Vrb2τ_fc(V_rb,τ_fc0=250*u.ps,τ_fc_sat=3*u.ps,V_bi=1.1*u.volt):
    τ = τ_fc0 * ( V_bi /(V_bi + V_rb))**2 + τ_fc_sat
    return τ.to(u.ps)

def Vrb2η(V_rb,λ=1.55*u.um):
    c = u.speed_of_light
    q = u.elementary_charge
    h = u.planck_constant
    f_l = (c/λ).to(u.THz) # laser frequency in radians/sec
    E_ph = (h*f_l).to(u.eV)
    η = ( ( q * V_rb + 2 * E_ph ) / ( 2 * E_ph) ).to(u.dimensionless).m
    return η



# default parameter dictionaries for exp2norm_params defined below
p_si = {
    'r': 0.189, # nonlinear refraction 2π * n_2 / λ
    'γ': 3.1e-9 * u.cm/u.watt, # nonlinear refraction 2π * n_2 / λ,
    'μ': 25, # FCD/FCA ratio
    'σ': 1.45e-17 * u.cm**2, # FCA cross section (electron-hole average)
    'c_v': 1./(0.6 * u.degK / u.joule * u.cm**3) # silicon volumetric heat capacity near room temp
}

p_expt_def = {
    'λ': 1.55 * u.um, # free space laser wavelength
    'FSR': 601 * u.GHz, # measured ring FSR
    'd_ring': 40 * u.um, # microring diameter
    'FWHM': 340 * u.MHz, # measured Lorentzian linewidth of single resonance
    'FWHM_i': 190 * u.MHz, # measured "intrinsic" Lorentzian linewidth of single resonance
    'splitting': 1.1*u.GHz, # measured double-lorentzian splitting
    'Δ_min': -60*u.GHz, # f_cavity,0 - f_laser tuning minimum
    'Δ_max': 5*u.GHz, # f_cavity,0 - f_laser tuning maximum
    'P_bus_max': 1 * u.mW, # max input power in bus waveguide
    'τ_th': 1.5 * u.us, # thermal "time constant" to fit
    'df_dT': -9.7 * u.GHz / u.degK, # measured thermal tuning rate
    'τ_fc0': 250 * u.ps, # measured/modeled free carrier lifetime at Vrb=0
    'τ_fc_sat': 3 * u.ps, # measured/modeled minimum free carrier lifetime at Vrb>~15V
    'V_bi': 1.1 * u.volt, # measured/modeled diode built-in voltage
    'α_dB': 0.7/u.cm, # fit waveguide loss inside ring in units of dB/cm
    'A': 0.1 * u.um**2, # mode effective area, from mode solver
    'β_2': 2 * u.ps**2/u.m, # GVD roughly measured, expected to be ~ 1 ps^2 / m
    'n_sf': 2, # number of significant figures to leave in the normalized parameters passed to mathematica. the fewer, the faster
    'δs': 0.05, # s step size (sqrt normalized input power)
    'δΔ': 0.2,  # Δ step size (cold cavity detuning)
    'dΔdt': 1e-6,
}

def expt2norm_params(p_expt=p_expt_def,p_mat=p_si,verbose=True):
    """Following Ryan's conventions in Conditions for Free Carrier Oscillations and... (conference, not JLT version) and 'Optical bistability, self-pulsing and XY optimization in silicon micro-rings with active carrier removal' from photonics west 2017"""
    ##### useful constants
    π = np.pi
    c = u.speed_of_light
    hbar = u.planck_constant / (2*π)
    q = u.elementary_charge

    #### unpack input dictionaries ####
    # p_expt
    λ = p_expt['λ']
    FSR = p_expt['FSR']
    d_ring = p_expt['d_ring']
    FWHM = p_expt['FWHM']
    FWHM_i = p_expt['FWHM_i']
    Δ_min = p_expt['Δ_min']
    Δ_max = p_expt['Δ_max']
    P_bus_max = p_expt['P_bus_max']
    τ_th = p_expt['τ_th']
    df_dT = p_expt['df_dT']
    τ_fc0 = p_expt['τ_fc0']
    τ_fc_sat = p_expt['τ_fc_sat']
    V_bi = p_expt['V_bi']
    V_rb = p_expt['V_rb']
    α_dB = p_expt['α_dB']
    A = p_expt['A']
    splitting = p_expt['splitting']
    β_2 = p_expt['β_2']
    # mathematica solver Parameters
    n_sf = p_expt['n_sf'] # number of significant figures to leave in the normalized parameters passed to mathematica. the fewer, the faster
    δs = p_expt['δs'] # s step size (sqrt normalized input power)
    δΔ = n_sig_figs(p_expt['δΔ'],2) # Δ step size (cold cavity detuning)
    # p_mat
    r = p_mat['r']
    γ = p_mat['γ']
    μ = p_mat['μ']
    σ = p_mat['σ']
    c_v = p_mat['c_v']

    ### Waveguide/Resonator Parameters ###
    t_R = (1/FSR).to(u.ps) # round trip time
    n_g = (c / ( π * d_ring * FSR )).to(u.dimensionless).m # group index from measured FSR
    v_g = c / n_g # group velocity
    κ = 2*π * FWHM # αL + θ, full measured linewidth
    τ_ph = (1 / κ).to(u.ps) # photon lifetime in microring
    κ_min = 2*π * FWHM_i # linewidth limit in super undercoupled regime
#     αL = (κ_min*t_R).to(u.dimensionless).m # fractional power lost per round trip, should be ~0.2%
#     α = α_dB/(10*np.log(10)) # measured loss in units of 1/cm
#     κ_min =  α * d_ring / t_R # intrinsic linewidth due to loss
    κ_c = κ - κ_min # coupling rate to bus waveguide in radians/sec
    κ_c_Hz = κ_c/(2*π) # coupling rate to bus waveguide in GHz
    δ_r = (splitting / FWHM).to(u.dimensionless).m # normalized splitting in units of individual mode linewidth
    θ = (κ_c*t_R).to(u.dimensionless).m # fractional power loss to coupling per round trip
    ω_l = (2*π*c/λ).to(u.THz) # laser frequency in radians/sec
    E_ph = (hbar*ω_l).to(u.eV)
    # fractional increase in thermal energy per TPA due to carrier removal work
    η = Vrb2η(V_rb,λ=λ)
    # reduced free carrier lifetime
    τ_fc = Vrb2τ_fc(V_rb,τ_fc0=τ_fc0,τ_fc_sat=τ_fc_sat,V_bi=V_bi)
    τ_fc_norm = (τ_fc/τ_ph).to(u.dimensionless).m
    # thermal/TO properties
    τ_th_norm = (τ_th/τ_ph).to(u.dimensionless).m
    dneff_dT = -(n_g * df_dT * 2 * π / ω_l).to(1/u.degK)
    δ_T = ( ( ω_l / c ) * dneff_dT ).to(1/u.cm/u.degK) # δ_T = dβ/dT = (ω/c)*dneff/dT
    ζ = (δ_T / ( 2 * c_v * γ * v_g )).to(u.dimensionless).m
    # FC generation rate per |a|^4
    χ = ( ( r * μ * σ ) / ( 2 * E_ph * γ * v_g ) ).to(u.dimensionless).m

    #### Normalization Constants ####
    # circulating field, a = ξ_a * \bar{a} in sqrt{watt/cm^2}
    ξ_a = (1/np.sqrt(2*γ*v_g*τ_ph)).to(u.watt**(0.5)/u.cm)
    # input/output fields, a = ξ_in * \bar{a} in sqrt{watt/cm^2}
    ξ_in = np.sqrt( t_R**2 / ( 8 * γ * v_g * τ_ph**3 * θ ) ).to(u.watt**(0.5)/u.cm)
    # carrier density, n = ξ_n * \bar{n} in cm^{-3}
    ξ_n = (1 / ( μ * σ * v_g * τ_ph )).to(u.cm**-3)
    # fast time, for KCG consideration
    ξ_t = np.sqrt( β_2 * v_g * τ_ph ).to(u.ps)
    # thermal scaling between cavity linewidths/kelvin
    ξ_T = ( 1 / ( δ_T * v_g * τ_ph ) ).to(u.degK)

    #### normalized input power range ####
    I_bus_max = P_bus_max / A
    s_max = int( np.ceil( 10 * ( np.sqrt( I_bus_max) / ξ_in ).to(u.dimensionless).m ) ) / 10.
    s = np.arange(δs,s_max,δs)
    P_bus = ( ( s * ξ_in )**2 * A ).to(u.mW)


    #### normalized tuning range ####
    Δ_min_norm = int( np.floor( ( 2*π * Δ_min * τ_ph ).to(u.dimensionless).m ) )
    Δ_max_norm = int( np.ceil( ( 2*π * Δ_max * τ_ph ).to(u.dimensionless).m ) )
    Δ_norm = np.arange(Δ_min_norm,Δ_max_norm,δΔ)
    Δ = ( Δ_norm / ( 2*π * τ_ph ) ).to(u.GHz)

    # load newly calculated unitful and normalized values into p_expt to export
    p_expt['s_max'] = s_max
    p_expt['Δ_min_norm'] = Δ_min_norm
    p_expt['Δ_max_norm'] = Δ_max_norm
    p_expt['η'] = η
    p_expt['τ_fc_norm'] = τ_fc_norm
    p_expt['τ_th_norm'] = τ_th_norm
    p_expt['ζ'] = ζ
    p_expt['χ'] = χ
    p_expt['E_ph'] = E_ph
    p_expt['δ_T'] = δ_T
    p_expt['n_g'] = n_g
    p_expt['P_bus'] = P_bus
    p_expt['s'] = s
    p_expt['κ_c_Hz'] = κ_c_Hz
    p_expt['τ_fc'] = τ_fc
    p_expt['τ_th'] = τ_th
    p_expt['τ_ph'] = τ_ph
    p_expt['t_R'] = t_R
    p_expt['θ'] = θ
    p_expt['ξ_a'] = ξ_a
    p_expt['ξ_in'] = ξ_in
    p_expt['ξ_n'] = ξ_n
    p_expt['ξ_T'] = ξ_T
    p_expt['ξ_t'] = ξ_t

    if verbose:
        print(f"s_max: {s_max:1.3g}")
        print(f"Δ_min_norm: {Δ_min_norm:1.3g}")
        print(f"Δ_max_norm: {Δ_max_norm:1.3g}")
        print(f"η: {η:1.3g}")
        print(f"τ_fc_norm: {τ_fc_norm:1.3g}")
        print(f"τ_th_norm: {τ_th_norm:1.3g}")
        print(f"ζ: {ζ:1.3g}")
        print(f"χ: {χ:1.3g}")

        print(f"τ_fc: {τ_fc:1.3g}")
        print(f"τ_th: {τ_th:1.3g}")
        print(f"τ_ph: {τ_ph:1.3g}")
        print(f"t_R: {t_R:1.3g}")

        print(f"E_ph: {E_ph:1.3g}")
        print(f"δ_T: {δ_T:1.3g}")
        print(f"n_g: {n_g:1.3g}")
        print(f"θ: {θ:1.3g}")

        print(f"ξ_a: {ξ_a:1.3g}")
        print(f"ξ_in: {ξ_in:1.3g}")
        print(f'ξ_n: {ξ_n:1.3e}')
        print(f"ξ_T: {ξ_T:1.3g}")
        print(f"ξ_t: {ξ_t:1.3g}")

### Old manual p_norm values for reference
#     p_norm = {'Δ':np.arange(-35,7,0.2),
#               's':np.arange(0,0.2,0.02),
#               'γ':2., # splitting over linewidth
#               'μ':30.,
#               'r':0.2,
#               'ζ':0.07,
#               'τ_th':3200,
#               'η':10,
#               'τ_fc':0.01,
#               'χ':5.}

    p_norm = {'Δ':Δ_norm,
              's':s,
              'γ':n_sig_figs(δ_r,n_sf), # splitting over linewidth, old variable name convention.
              'μ':n_sig_figs(p_mat['μ'],n_sf),
              'r':n_sig_figs(p_mat['r'],n_sf),
              'ζ':n_sig_figs(ζ,n_sf),
              'τ_th': n_sig_figs(τ_th_norm,n_sf),
              'η': n_sig_figs(η,n_sf),
              'τ_fc': n_sig_figs(τ_fc_norm,n_sf),
              'χ': n_sig_figs(χ,n_sf),
             }

    p_expt['p_norm'] = p_norm

################################################################################
##
##           functions for power+Vrb+Δ sweeps
##          for comparison with experiment
##
################################################################################

def ODEInt_Dsweep_trace(p):
    timestamp_str = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')

    data_dir = p['data_dir']
    fname_b2r = p['name'] + '_b2r.dat'
    fname_r2b = p['name'] + '_r2b.dat'
    fpath_b2r = path.join(data_dir,fname_b2r)
    fpath_r2b = path.join(data_dir,fname_r2b)

    Δ0_b2r = p['Δ_max_norm']
    Δ0_r2b = p['Δ_min_norm']
    t_max = (Δ0_b2r - Δ0_r2b) / p['dΔdt']
    a_0 = b_0 = 0.0 + 0.0j
    a_star_0 = b_star_0 = np.conjugate(a_0)
    n0 = 0.0
    T0 = 0.0
    # Δ, a_c, a_c_star, a_s, a_s_star, n, T = y
    y0_b2r = [Δ0_b2r,a_0,a_star_0,a_0,a_star_0,n0,T0]
    y0_r2b = [Δ0_r2b,a_0,a_star_0,a_0,a_star_0,n0,T0]
    t_span = (0.0, t_max)
    sol_b2r = solve_ivp(fun=lambda t, y: f_tuning(t, y, p, -p['dΔdt']),
                            t_span=t_span,
                            y0=y0_b2r,
                            method='BDF',
                            jac=lambda t, y: jac_tuning(t, y, p, -p['dΔdt']),
                           )
    with open(fpath_b2r, 'wb') as f:
        pickle.dump(sol_b2r, f,fix_imports=True,protocol=pickle.HIGHEST_PROTOCOL)
    sol_r2b = solve_ivp(fun=lambda t, y: f_tuning(t, y, p, p['dΔdt']),
                            t_span=t_span,
                            y0=y0_r2b,
                            method='BDF',
                            jac=lambda t, y: jac_tuning(t, y, p, p['dΔdt']),
                           )
    with open(fpath_r2b, 'wb') as f:
        pickle.dump(sol_r2b, f,fix_imports=True,protocol=pickle.HIGHEST_PROTOCOL)
    return 1


def collect_ODEInt_PVΔ_sweep(p_expt=p_expt_def,p_mat=p_si,sweep_name='test',nEq=6,n_proc=n_proc_def,data_dir=data_dir,verbose=True,return_data=False):
    """
    Find steady state solutions and corresponding Jacobian eigenvalues
    for the specified normalized 2-mode+free-carrier+thermal microring model
    parameters over the range of cold-cavity detuning (Δ), input power (s^2) and Vrb
    values supplied.
    """
    timestamp_str = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')
    sweep_dir_name = 'ODEInt_PVDsweep_' + sweep_name + '_' + timestamp_str
    sweep_dir = path.normpath(path.join(data_dir,sweep_dir_name))
    makedirs(sweep_dir)
    sweep_data_fname = 'data.dat'
    sweep_data_fpath = path.join(sweep_dir,sweep_data_fname)
    mfname = 'metadata.dat'
    mfpath = path.join(sweep_dir,mfname)
    # add a couple other things to kwargs and save as a metadata dict
    metadata = p_expt.copy()
    metadata.update(p_mat)
    nV = len(p_expt['V_rb'])
    metadata['nV'] = nV
    t_start = time()
    start_timestamp_str = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')
    metadata['t_start'] = start_timestamp_str
    with open(mfpath, 'wb') as f:
        pickle.dump(metadata,f)
    for Vind, VV in enumerate(p_expt['V_rb']):
        V_dir_name = f'V{Vind}'
        V_dir = path.normpath(path.join(sweep_dir,V_dir_name))
        makedirs(V_dir)
        V_data_fname = f'V{Vind}_data.dat'
        V_data_fpath = path.join(V_dir,V_data_fname)
        V_params_fname = f'V{Vind}_params.dat'
        V_params_fpath = path.join(V_dir,V_params_fname)
        # add a couple other things to kwargs and save as a metadata dict
        V_params = p_expt.copy()
        V_params['V_rb'] = VV
        V_params.update(p_mat)
        start_timestamp_str = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')
        V_params['t_start'] = start_timestamp_str
        if verbose:
            print('##########################################################')
            print(f'V: {VV.m:2.2f} V, Vind: {Vind}')
        expt2norm_params(p_expt=V_params,p_mat=p_mat,verbose=verbose)
        if Vind==0:
            V_params_list = [[] for x in range(nV)]
            nΔ = len(V_params['p_norm']['Δ'])
            ns = len(V_params['p_norm']['s'])
        V_params['nΔ'] = nΔ
        V_params['ns'] = ns
        V_params_list[Vind] = V_params
        V_params_ss_list = [[] for sind in range(ns)]
        for sind, ss in enumerate(V_params['p_norm']['s']):
            # timestamp_str = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')
            V_params_ss_list[sind] = V_params['p_norm'].copy()
            V_params_ss_list[sind]['s'] = ss
            V_params_ss_list[sind]['Δ_max_norm'] =  V_params['Δ_max_norm']
            V_params_ss_list[sind]['Δ_min_norm'] =  V_params['Δ_min_norm']
            V_params_ss_list[sind]['dΔdt'] =  V_params['dΔdt']
            # ss_fname = f's{sind}_' + timestamp_str + '.csv'
            V_params_ss_list[sind]['name'] = f's{sind}'
            V_params_ss_list[sind]['data_dir'] = V_dir
            # run_mm_script(params=V_params_ss,data_dir=V_dir,data_fname=ss_fname)
            # Pa_ss,Pb_ss = import_mm_data(path.join(V_dir,ss_fname))
            # a_ss,b_ss,n_ss,T_ss,eigvals_ss,det_j_ss,L_ss = jac_eigvals_sweep(Pa_ss,Pb_ss,**V_params_ss)
        # map function onto pool of mathematica processes
        with Pool(processes=n_proc) as pool:
            res = pool.map(ODEInt_Dsweep_trace,V_params_ss_list)
            # res = pool.map(process_mm_data_parallel,V_params_ss_list)
        # # fill in dataset arrays, creating them first if Vind==0
        # if Vind==0:
        #     Pa = np.zeros((ns,nV,nΔ,nEq),dtype=np.float64)
        #     Pb = np.zeros((ns,nV,nΔ,nEq),dtype=np.float64)
        #     a = np.zeros((ns,nV,nΔ,nEq),dtype=np.complex128)
        #     b = np.zeros((ns,nV,nΔ,nEq),dtype=np.complex128)
        #     n = np.zeros((ns,nV,nΔ,nEq),dtype=np.float64)
        #     T = np.zeros((ns,nV,nΔ,nEq),dtype=np.float64)
        #     Δ = np.zeros((ns,nV,nΔ,nEq),dtype=np.float64)
        # for sind in range(ns):
        #     Pa[sind,Vind,:] = res[sind]['Pa']
        #     Pb[sind,Vind,:] = res[sind]['Pb']
        #     a[sind,Vind,:] = res[sind]['a']
        #     b[sind,Vind,:] = res[sind]['b']
        #     n[sind,Vind,:] = res[sind]['n']
        #     T[sind,Vind,:] = res[sind]['T']
        #     eigvals[sind,Vind,:] = res[sind]['eigvals']
        #     det_j[sind,Vind,:] = res[sind]['det_j']
        #     L[sind,Vind,:] = res[sind]['L']
        # # save data
        # V_data = {'Pa':Pa[:,Vind],'Pb':Pb[:,Vind],'a':a[:,Vind],'b':b[:,Vind],'n':n[:,Vind],'T':T[:,Vind],
        #     'eigvals':eigvals[:,Vind],'det_j':det_j[:,Vind],'L':L[:,Vind],**V_params}
        # with open(V_data_fpath, 'wb') as f:
        #     pickle.dump(V_data,f)
        # data = {'Pa':Pa,'Pb':Pb,'a':a,'b':b,'n':n,'T':T,
        #     'eigvals':eigvals,'det_j':det_j,'L':L,'V_params':V_params_list,**metadata}
        # with open(sweep_data_fpath, 'wb') as f:
        #     pickle.dump(data,f)
    stop_timestamp_str = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')
    t_elapsed_sec = time()-t_start
    if verbose:
        print('PVΔ ODEInt sweep finished at ' + stop_timestamp_str)
    metadata['t_stop'] = stop_timestamp_str
    metadata['t_elapsed_sec'] = t_elapsed_sec
    metadata['V_params_list'] = V_params_list
    with open(mfpath, 'wb') as f:
        pickle.dump(metadata,f)
    if return_data:
        return data

def load_ODEInt_PVΔ_sweep(sweep_name='test',data_dir=data_dir,verbose=True,fpath=None):
    if not(fpath):
        file_list =  glob(path.normpath(data_dir)+path.normpath('/MM_PVDsweep_' + sweep_name + '*'))
        latest_file = max(file_list,key=path.getctime) + '/data.dat'
    else:
        latest_file = fpath + '/data.dat'
    if verbose:
        print('Loading ' + sweep_name +' sweep data from path: ' + path.basename(path.normpath(latest_file)))
    with open( latest_file, "rb" ) as f:
        data = pickle.load(f)
    latest_metadata_file = max(file_list,key=path.getctime) + '/metadata.dat'
    with open( latest_metadata_file, "rb" ) as f:
        metadata = pickle.load(f)
    metadata.update(data)
    return metadata

################################################################################
#
#     functions from model for one-mode, non-thermal FCO behavior
#              for comparison with Ryan's results
#
################################################################################

# define amplitude function to detect/measure free carrier oscillations in integrated solution
def fco_amp(t,U,t_probe=30):
    U_probe = U[t>t_probe]
    return (U_probe.max() - U_probe.min())/U_probe.mean()

# def f_DrivenCavity(t,y,s,Delta):
#     a, a_star , n = y
#     d_a = ( ( -0.5 + 1j*Delta) + ( -1/mu - 1j ) * n + (1j - r) * (a * a_star) ) * a + s
#     d_a_star = np.conjugate( ( -0.5 + 1j*Delta) + ( -1/mu - 1j ) * n + (1j - r) * (a * a_star) ) * a_star + s
#     d_n = -n / tau + chi * (a*a_star)**2
#     return [d_a, d_a_star, d_n]
#
# def jac_DrivenCavity(t,y,s,Delta):
#     a , a_star , n = y
#     d_d_a = [ ( ( -0.5 + 1j*Delta) + ( -1/mu - 1j ) * n + (1j - r) * (a * a_star) ), (1j - r) * a**2, ( -1/mu - 1j ) * a ]
#     d_d_a_star = [ -1*np.conjugate((1j - r) * a**2), np.conjugate( ( -0.5 + 1j*Delta) + ( -1/mu - 1j ) * n + (1j - r) * (a * a_star) ) , np.conjugate(( -1/mu - 1j ) * a) ]
#     d_d_n = [ 2 * chi * a_star**2 * a , 2 * chi * a**2 * a_star , -1/tau ]
#     return [d_d_a, d_d_a_star, d_d_n]



def ode_int_sweep(n_s,n_Delta,s_sq_min=1e-5,s_sq_max=10,Delta_min=-3,Delta_max=3):
    #n_s = 8
    #n_Delta = 10
    s_sq = np.linspace(s_sq_min,s_sq_max,n_s)
    Delta = np.linspace(Delta_min,Delta_max,n_Delta)
    traces = []
    amp = np.empty([n_s,n_Delta])
    s = np.sqrt(s_sq)
    t_max = 50.0
    a_0 = 0.0 + 0.0j
    a_star_0 = np.conjugate(a_0)
    n0 = 0.0
    y0 = [a_0,a_star_0,n0]
    t_span = (0.0, t_max)
    for sind,ss in enumerate(s):
        for Dind,DD in enumerate(Delta):
            sol = solve_ivp(fun=lambda t, y: f_DrivenCavity(t, y, ss, DD),
                            t_span=t_span,
                            y0=y0,
                            method='BDF',
                            jac=lambda t, y: jac_DrivenCavity(t, y, ss, DD),
                           )
            t = sol.t
            a = sol.y[0, :]
            n = sol.y[2, :]
            trace = np.stack((t,a,n))
            traces.append(trace)
            U = np.abs(a)**2
            amp[sind,Dind] = fco_amp(t,U)
    return s_sq, Delta, traces, amp

def collect_ode_int_sweep(n_s,n_Delta,s_sq_min=1e-5,s_sq_max=10,Delta_min=-3,Delta_max=3,name='',data_dir='/Users/doddgray/Google Drive/data/',return_data=True):
    timestamp_str = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')
    fname = 'ode_int_sweep_' + name + '_' + timestamp_str
    fpath = path.join(data_dir,fname)
    s_sq,Delta,traces,amp = ode_int_sweep(n_s,n_Delta,s_sq_min=s_sq_min,s_sq_max=s_sq_max,
                                Delta_min=Delta_min,Delta_max=Delta_max)
    data = {'s_sq':s_sq,'Delta':Delta,'traces':traces,'amp':amp}
    print('saving to '+fpath)
    with open(fpath, 'wb') as f:
        pickle.dump(data, f,fix_imports=True,protocol=pickle.HIGHEST_PROTOCOL)
    if return_data:
        return data


def load_ode_int_sweep(name='',data_dir='/Users/doddgray/Google Drive/data/',verbose=False):
    file_list =  glob(path.normpath(data_dir)+path.normpath('/ode_int_sweep_' + name + '_*'))
    latest_file = max(file_list,key=path.getctime)
    if verbose:
        print('Loading ' + name +' trace from file: ' + path.basename(path.normpath(latest_file)))
    with open(latest_file, 'rb') as handle:
        data = pickle.load(handle)
    return data

def plot_FCO_amplitude(ds):
    # plot s, Delta sweep results
    # ds = load_ode_int_sweep(name='Delta_s_sweep_tau0.2_mu30_r0.2_chi10_nonthermal')
    fig = plt.figure(figsize=(12,6))
    gs = GridSpec(1,1)
    ax0 = fig.add_subplot(gs[0])
    p0 = ax0.pcolormesh(loaded_data['s_sq'],loaded_data['Delta'],loaded_data['amp'].T)
    ax0.set_xlabel('$|s|^2$')
    ax0.set_ylabel('$\Delta$')
    plt.colorbar(p0,label='free carrier\noscllation amplitude')
