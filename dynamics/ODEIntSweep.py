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
    data_dir = "/home/dodd/data/ODEInt_PVDsweep"
    n_proc_def = 6
else: # assume I'm on a MTL server or something
    home = str( Path.home() )
    data_dir = home+'/data/'
    n_proc_def = 30

from FixedPointSweep import n_sig_figs, expt2norm_params, p_si, p_expt_def


def f_DrivenCavity(t,y,p):
    # unpack paramater dict "p"
    Δ = p['Δ']
    s = p['s']
    δ_r = p['γ']
    μ = p['μ']
    r = p['r']
    ζ = p['ζ']
    η = p['η']
    η2 = p['η2']
    χ = p['χ']
    τ_fc = p['τ_fc']
    τ_th = p['τ_th']
    sqrt_eta = sqrt(η)
    a_c, a_c_star, a_s, a_s_star, n, T = y
    d_a_c = ( ( -0.5 + 1j*Δ + 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + (1j - r) * (a_c * a_c_star) + 1j*T ) * a_c + sqrt_eta*s
    d_a_c_star = np.conjugate( ( -0.5 + 1j*Δ + 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + (1j - r) * (a_c * a_c_star) + 1j*T ) * a_c_star + sqrt_eta*s
    d_a_s = ( ( -0.5 + 1j*Δ - 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + (1j - r) * (a_s * a_s_star) + 1j*T ) * a_s + sqrt_eta*s
    d_a_s_star = np.conjugate( ( -0.5 + 1j*Δ - 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + (1j - r) * (a_s * a_s_star) + 1j*T ) * a_s_star + sqrt_eta*s
    d_n = -n / τ_fc + χ * ( (a_c*a_c_star)**2 + (a_s*a_s_star)**2 )
    d_T = -T / τ_th + ζ * ( η2 * r * ( (a_c*a_c_star)**2 + (a_s*a_s_star)**2 ) + n / μ * ( (a_c*a_c_star) + (a_s*a_s_star) ) )
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

def f_tuning_old(t,y,p,dΔdt):
    # unpack paramater dict "p"
    s = p['s']
    δ_r = p['γ']
    μ = p['μ']
    r = p['r']
    ζ = p['ζ']
    η1 = p['η1']
    η2 = p['η2']
    χ = p['χ']
    α = p['α']
    τ_fc = p['τ_fc']
    τ_xfc = p['τ_xfc']
    τ_th = p['τ_th']
    # dΔdt = p['dΔdt']
    # define state vector
    Δ, a_c, a_c_star, a_s, a_s_star, n_c, n_s, T = y
    # define equation system
    d_Δ = dΔdt
    d_a_c = ( ( -0.5 + 1j*Δ + 1j*δ_r/2.0) + ( -1/μ - 1j ) * n_c + (1j - r) * (a_c * a_c_star) + 1j*T ) * a_c + s
    d_a_c_star = np.conjugate( ( -0.5 + 1j*Δ + 1j*δ_r/2.0) + ( -1/μ - 1j ) * n_c + (1j - r) * (a_c * a_c_star) + 1j*T ) * a_c_star + s
    d_a_s = ( ( -0.5 + 1j*Δ - 1j*δ_r/2.0) + ( -1/μ - 1j ) * n_s + (1j - r) * (a_s * a_s_star) + 1j*T ) * a_s + s
    d_a_s_star = np.conjugate( ( -0.5 + 1j*Δ - 1j*δ_r/2.0) + ( -1/μ - 1j ) * n_s + (1j - r) * (a_s * a_s_star) + 1j*T ) * a_s_star + s
    d_n_c = -n_c / τ_fc + α * (a_c*a_c_star) + χ * (a_c*a_c_star)**2 + ( n_s - n_c ) / τ_xfc
    d_n_s = -n_s / τ_fc + α * (a_s*a_s_star) + χ * (a_s*a_s_star)**2 + ( n_c - n_s ) / τ_xfc
    d_T = -T / τ_th + ζ * ( η2 * r * ( (a_c*a_c_star)**2 + (a_s*a_s_star)**2 ) + η1 * r * α / χ * ( (a_c*a_c_star) + (a_s*a_s_star) ) + 1 / μ * ( (a_c*a_c_star) * n_c + (a_s*a_s_star) * n_s ) )
    return [d_Δ, d_a_c, d_a_c_star, d_a_s, d_a_s_star , d_n_c, d_n_s, d_T]

def f_tuning(t,y,p,dΔdt):
    # unpack paramater dict "p"
    s = p['s']
    δ_r = p['γ']
    μ = p['μ']
    r = p['r']
    ζ = p['ζ']
    η = p['η']
    η2 = p['η2']
    χ = p['χ']
    τ_fc = p['τ_fc']
    τ_th = p['τ_th']
    sqrt_eta = np.sqrt(η)
    # dΔdt = p['dΔdt']
    # define state vector
    Δ, a_c, a_c_star, a_s, a_s_star, n, T = y
    # define equation system
    d_Δ = dΔdt
    d_a_c = ( ( -0.5 + 1j*Δ + 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + (1j - r) * (a_c * a_c_star) + 1j*T ) * a_c + sqrt_eta*s
    d_a_c_star = np.conjugate( ( -0.5 + 1j*Δ + 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + (1j - r) * (a_c * a_c_star) + 1j*T ) * a_c_star + sqrt_eta*s
    d_a_s = ( ( -0.5 + 1j*Δ - 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + (1j - r) * (a_s * a_s_star) + 1j*T ) * a_s + sqrt_eta*s
    d_a_s_star = np.conjugate( ( -0.5 + 1j*Δ - 1j*δ_r/2.0) + ( -1/μ - 1j ) * n + (1j - r) * (a_s * a_s_star) + 1j*T ) * a_s_star + sqrt_eta*s
    d_n = -n / τ_fc + χ * ( (a_c*a_c_star)**2 + (a_s*a_s_star)**2 )
    d_T = -T / τ_th + ζ * ( η2 * r * ( (a_c*a_c_star)**2 + (a_s*a_s_star)**2 ) + n / μ * ( (a_c*a_c_star) + (a_s*a_s_star) ) )
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


# # default parameter dictionaries for exp2norm_params defined below
# p_si = {
#     'r': 0.189, # nonlinear refraction 2π * n_2 / λ
#     'γ': 3.1e-9 * u.cm/u.watt, # nonlinear refraction 2π * n_2 / λ,
#     'μ': 25, # FCD/FCA ratio
#     'σ': 1.45e-17 * u.cm**2, # FCA cross section (electron-hole average)
#     'c_v': 1./(0.6 * u.degK / u.joule * u.cm**3) # silicon volumetric heat capacity near room temp
# }
#
# p_expt_def = {
#     'λ': 1.55 * u.um, # free space laser wavelength
#     'FSR': 601 * u.GHz, # measured ring FSR
#     'd_ring': 40 * u.um, # microring diameter
#     'FWHM': 340 * u.MHz, # measured Lorentzian linewidth of single resonance
#     'FWHM_i': 190 * u.MHz, # measured "intrinsic" Lorentzian linewidth of single resonance
#     'splitting': 1.1*u.GHz, # measured double-lorentzian splitting
#     'Δ_min': -60*u.GHz, # f_cavity,0 - f_laser tuning minimum
#     'Δ_max': 5*u.GHz, # f_cavity,0 - f_laser tuning maximum
#     'P_bus_max': 1 * u.mW, # max input power in bus waveguide
#     'τ_th': 1.5 * u.us, # thermal "time constant" to fit
#     'df_dT': -9.7 * u.GHz / u.degK, # measured thermal tuning rate
#     'τ_fc0': 250 * u.ps, # measured/modeled free carrier lifetime at Vrb=0
#     'τ_fc_sat': 3 * u.ps, # measured/modeled minimum free carrier lifetime at Vrb>~15V
#     'V_bi': 1.1 * u.volt, # measured/modeled diode built-in voltage
#     'α_dB': 0.7/u.cm, # fit waveguide loss inside ring in units of dB/cm
#     'A': 0.1 * u.um**2, # mode effective area, from mode solver
#     'β_2': 2 * u.ps**2/u.m, # GVD roughly measured, expected to be ~ 1 ps^2 / m
#     'n_sf': 2, # number of significant figures to leave in the normalized parameters passed to mathematica. the fewer, the faster
#     'δs': 0.05, # s step size (sqrt normalized input power)
#     'δΔ': 0.2,  # Δ step size (cold cavity detuning)
#     'τ_th_norm_ζ_product': 4.5,  # τ_th_norm * ζ, inferred from experiment data
#     'dΔdt': 1e-6, # tuning rate for direct integration,
# }


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
    y0_b2r = [Δ0_b2r,a_0,a_star_0,b_0,b_star_0,n0,T0]
    y0_r2b = [Δ0_r2b,a_0,a_star_0,b_0,b_star_0,n0,T0]
    t_span = (0.0, t_max)
    sol_b2r = solve_ivp(fun=lambda t, y: f_tuning(t, y, p, -p['dΔdt']),
                            t_span=t_span,
                            y0=y0_b2r,
                            method='RK45',
                            # min_step=0.1,
                            max_step=100,
                            # jac=lambda t, y: jac_tuning(t, y, p, -p['dΔdt']),
                           )
    with open(fpath_b2r, 'wb') as f:
        pickle.dump(sol_b2r, f,fix_imports=True,protocol=pickle.HIGHEST_PROTOCOL)
    sol_r2b = solve_ivp(fun=lambda t, y: f_tuning(t, y, p, p['dΔdt']),
                            t_span=t_span,
                            y0=y0_r2b,
                            method='RK45',
                            # min_step=0.1,
                            max_step=100,
                            # jac=lambda t, y: jac_tuning(t, y, p, p['dΔdt']),
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
    params_list = []
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
            params_list.append(V_params_ss_list[sind])
            # run_mm_script(params=V_params_ss,data_dir=V_dir,data_fname=ss_fname)
            # Pa_ss,Pb_ss = import_mm_data(path.join(V_dir,ss_fname))
            # a_ss,b_ss,n_ss,T_ss,eigvals_ss,det_j_ss,L_ss = jac_eigvals_sweep(Pa_ss,Pb_ss,**V_params_ss)
        # map function onto pool of mathematica processes
        # with Pool(processes=n_proc) as pool:
        #     res = pool.map(ODEInt_Dsweep_trace,V_params_ss_list)
    with Pool(processes=n_proc) as pool:
        res = pool.map(ODEInt_Dsweep_trace,params_list)
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
