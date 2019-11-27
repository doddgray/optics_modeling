# -*- coding: utf-8 -*-
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

# os.environ['WolframKernel'] = '/usr/local/Wolfram/Mathematica/11.3/Executables/WolframKernel'
mm_script_fname = "FixedPointSweep_mm_script.wls"

hostname = socket.gethostname()
if hostname=='dodd-laptop':
    data_dir = "/home/dodd/google-drive/notebooks/IMEC V1 1550nm ring measurements/Thermal SOI ring cavity stability analysis"
    script_dir = "/home/dodd/google-drive/Documents/mathematica-scripts/"
    mm_script_fpath = path.normpath(path.join(script_dir,mm_script_fname))
    os.environ['WolframKernel'] = '/usr/local/Wolfram/Mathematica/12.0/Executables/WolframKernel'
    n_proc_def = 8
else: # assume I'm on a MTL server or something
    home = str( Path.home() )
    data_dir = home+'/data/'
    script_dir = home+'/Wolfram Mathematica/'
    os.environ['WolframKernel'] = '/opt/wolfram/Mathematica/12.0/Executables/WolframKernel'
    this_dir = os.path.dirname(os.path.realpath(__file__))
    local_wls_path = path.normpath(path.join(this_dir,mm_script_fname))
    mm_script_fpath = path.normpath(path.join(script_dir,mm_script_fname))
    # shutil.copyfile(local_wls_path,mm_script_fpath)
    # chmod(mm_script_fpath, 777)
    n_proc_def = 32


data_fname = 'test_data_fname.csv'
base_mm_script_fname = 'FixedPointSweepSkeletonScript.wls'

def n_sig_figs(x,n):
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))

params = {'Δ':np.arange(-130,20.25,0.25),
          's':np.arange(0,4,0.1),
          'γ':2.,
          'μ':30.,
          'r':0.2,
          'ζ':0.1,
          'τ_th':30.,
          'η':1.,
          'τ_fc':1,
          'χ':5.}

def run_mm_script(params=params,data_dir=data_dir,script_dir=script_dir,data_fname=data_fname):
    Δ_max = int(round(params['Δ'].max()))
    Δ_min = int(round(params['Δ'].min()))
    # d_Δ = n_sig_figs(params['Δ'][1]-params['Δ'][0],2)
    nΔ = len(params['Δ'])-1
    arg_list = [data_dir,data_fname,Δ_min,Δ_max,nΔ,params['s'],
        params['γ'],params['μ'],params['r'],params['ζ'],
        params['τ_th'],params['η'],params['τ_fc'],params['χ']]
    cmd = [mm_script_fpath]+[f'{arg}' for arg in arg_list]
    out = subp.run(cmd,check=True)
    return out.returncode

def run_mm_script_parallel(params=params):
    timestamp_str = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')
    Δ_max = int(round(params['Δ'].max()))
    Δ_min = int(round(params['Δ'].min()))
    # d_Δ = n_sig_figs(params['Δ'][1]-params['Δ'][0],2)
    nΔ = len(params['Δ'])-1
    data_dir = params['data_dir']
    # data_fname = params['name'] + '_' + timestamp_str + '.csv'
    data_fname = params['name'] + '.csv'
    arg_list = [data_dir,data_fname,Δ_min,Δ_max,nΔ,params['s'],
        params['γ'],params['μ'],params['r'],params['ζ'],
        params['τ_th'],params['η'],params['τ_fc'],params['χ']]
    cmd = [mm_script_fpath]+[f'{arg}' for arg in arg_list]
    # print('Δ_max:' + str(Δ_max))
    # print('Δ_min:' + str(Δ_min))
    # print('d_Δ:' + str(d_Δ))
    # print('cmd: ' + str(cmd))
    out = subp.run(cmd,check=True)
    return out.returncode
    # # try:
    # Pa_ss,Pb_ss = import_mm_data(path.join(data_dir,data_fname))
    # a_ss,b_ss,n_ss,T_ss,eigvals_ss,det_j_ss,L_ss = jac_eigvals_sweep(Pa_ss,Pb_ss,**params)
    # res = {'Pa':Pa_ss,
    #         'Pb':Pb_ss,
    #         'a':a_ss,
    #         'b':b_ss,
    #         'n':n_ss,
    #         'T':T_ss,
    #         'eigvals':eigvals_ss,
    #         'det_j':det_j_ss,
    #         'L':L_ss,
    #         'mm_out':out,
    # }
    # return res
    # # except:
    # #     nΔ = len(params['Δ'])
    # #     print('python processing fail, '+params['name'])
    # #     res = {'Pa':np.zeros((nΔ,6),dtype=np.float64),
    # #             'Pb':np.zeros((nΔ,6),dtype=np.float64),
    # #             'a':np.zeros((nΔ,6),dtype=np.complex128),
    # #             'b':np.zeros((nΔ,6),dtype=np.complex128),
    # #             'n':np.zeros((nΔ,6),dtype=np.float64),
    # #             'T':np.zeros((nΔ,6),dtype=np.float64),
    # #             'eigvals':np.zeros((nΔ,6,6),dtype=np.complex128),
    # #             'det_j':np.zeros((nΔ,6),dtype=np.float64),
    # #             'L':np.zeros((nΔ,6),dtype=np.float64),
    # #             'mm_out':out,
    # #     }
    # # return res

def process_mm_data_parallel(params=params):
    data_dir = params['data_dir']
    data_fname = params['name'] + '.csv'
    data_fpath = path.normpath(path.join(data_dir,data_fname))
    if path.exists(data_fpath):
        Pa_ss,Pb_ss = import_mm_data(path.join(data_dir,data_fname))
        a_ss,b_ss,n_ss,T_ss,eigvals_ss,det_j_ss,L_ss = jac_eigvals_sweep(Pa_ss,Pb_ss,**params)
        res = {'Pa':Pa_ss,
                'Pb':Pb_ss,
                'a':a_ss,
                'b':b_ss,
                'n':n_ss,
                'T':T_ss,
                'eigvals':eigvals_ss,
                'det_j':det_j_ss,
                'L':L_ss,
                # 'mm_out':out,
        }
    else:
        nΔ = len(params['Δ'])
        print('no file found, '+params['name'])
        res = {'Pa':np.zeros((nΔ,6),dtype=np.float64),
                'Pb':np.zeros((nΔ,6),dtype=np.float64),
                'a':np.zeros((nΔ,6),dtype=np.complex128),
                'b':np.zeros((nΔ,6),dtype=np.complex128),
                'n':np.zeros((nΔ,6),dtype=np.float64),
                'T':np.zeros((nΔ,6),dtype=np.float64),
                'eigvals':np.zeros((nΔ,6,6),dtype=np.complex128),
                'det_j':np.zeros((nΔ,6),dtype=np.float64),
                'L':np.zeros((nΔ,6),dtype=np.float64),
                # 'mm_out':out,
            }
    return res

def process_mm_data_line(line,ncol=12):
    line_proc = line.replace('{Pa -> ','').replace('Pb -> ','').replace('}','').replace('"','').replace('*^','e')
    for iter in range(ncol + 1 - len(line_proc.split(','))):
        line_proc = line_proc[:-1] + ',' + line_proc[-1]
    final_line = ','.join([x.strip() if x else '0.0' for x in  line_proc.split(',')[:-1]])+'\n'
    return final_line

def import_mm_data(data_fname,data_dir=data_dir,ncol=12):
    data_str = ''
    data_fpath = path.normpath(path.join(data_dir,data_fname))
    with open(data_fpath,'r') as f:
        for line in f:
            data_str += process_mm_data_line(line)
    proc_fpath = data_fpath.strip('.csv')+'_proc.csv'
    with open(proc_fpath,'w') as f:
        f.write(data_str)
    data = np.genfromtxt(proc_fpath,delimiter=',')
    Pa = data[:,::2]
    Pb = data[:,1::2]
    return Pa,Pb

def compute_2d_sweep(params=params,sweep_name='test',data_dir=data_dir,verbose=True,return_data=False):
    """
    Find steady state solutions and corresponding Jacobian eigenvalues
    for the specified normalized 2-mode+free-carrier+thermal microring model
    parameters over the range of cold-cavity detuning (Δ) and input power (s^2)
    values supplied.
    """
    timestamp_str = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')
    sweep_dir_name = 'MM_FPS_' + sweep_name + '_' + timestamp_str
    sweep_dir = path.normpath(path.join(data_dir,sweep_dir_name))
    makedirs(sweep_dir)
    sweep_data_fname = 'data.dat'
    sweep_data_fpath = path.join(sweep_dir,sweep_data_fname)
    mfname = 'metadata.dat'
    mfpath = path.join(sweep_dir,mfname)
    # add a couple other things to kwargs and save as a metadata dict
    metadata = params
    t_start = time()
    start_timestamp_str = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')
    metadata['t_start'] = start_timestamp_str
    with open(mfpath, 'wb') as f:
        pickle.dump(metadata,f)
    for sind, ss in enumerate(params['s']):
        timestamp_str = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')
        params_ss = params.copy()
        params_ss['s'] = ss
        ss_fname = f's{ss:3.3f}_' + timestamp_str + '.csv'
        run_mm_script(params=params_ss,data_dir=sweep_dir,data_fname=ss_fname)
        if verbose:
            print(f's: {ss:3.3f}, sind: {sind}, saving to ' + ss_fname)
        Pa_ss,Pb_ss = import_mm_data(path.join(sweep_dir,ss_fname))
        #return Pa_ss,Pb_ss,params_ss
        a_ss,b_ss,n_ss,T_ss,eigvals_ss,det_j_ss,L_ss = jac_eigvals_sweep(Pa_ss,Pb_ss,**params_ss)
        if sind==0:
            ns = len(params['s'])
            Pa = np.empty((ns,)+Pa_ss.shape,dtype=np.float64)
            Pb = np.empty((ns,)+Pb_ss.shape,dtype=np.float64)
            a = np.empty((ns,)+a_ss.shape,dtype=np.complex128)
            b = np.empty((ns,)+b_ss.shape,dtype=np.complex128)
            n = np.empty((ns,)+n_ss.shape,dtype=np.float64)
            T = np.empty((ns,)+T_ss.shape,dtype=np.float64)
            eigvals = np.empty((ns,)+eigvals_ss.shape,dtype=np.complex128)
            det_j = np.empty((ns,)+det_j_ss.shape,dtype=np.float64)
            L = np.empty((ns,)+L_ss.shape,dtype=np.float64)
        Pa[sind,:] = Pa_ss
        Pb[sind,:] = Pb_ss
        a[sind,:] = a_ss
        b[sind,:] = b_ss
        n[sind,:] = n_ss
        T[sind,:] = T_ss
        eigvals[sind,:] = eigvals_ss
        det_j[sind,:] = det_j_ss
        L[sind,:] = L_ss
        if verbose:
            print('saving data...')
        data = {'Pa':Pa,'Pb':Pb,'a':a,'b':b,'n':n,'T':T,
            'eigvals':eigvals,'det_j':det_j,'L':L,**params}
        with open(sweep_data_fpath, 'wb') as f:
            pickle.dump(data,f)
    stop_timestamp_str = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')
    t_elapsed_sec = time()-t_start
    if verbose:
        print('sweep finished at ' + stop_timestamp_str)
    metadata['t_stop'] = stop_timestamp_str
    metadata['t_elapsed_sec'] = t_elapsed_sec
    with open(mfpath, 'wb') as f:
        pickle.dump(metadata,f)
    if return_data:
        return data

def load_2d_sweep(sweep_name='test',data_dir=data_dir,verbose=True,fpath=None):
    if not(fpath):
        file_list =  glob(path.normpath(data_dir)+path.normpath('/MM_FPS_' + sweep_name + '*'))
        latest_file = max(file_list,key=path.getctime) + '/data.dat'
    else:
        latest_file = fpath + '/data.dat'
    if verbose:
        print('Loading ' + sweep_name +' sweep data from path: ' + path.basename(path.normpath(latest_file)))
    with open( latest_file, "rb" ) as f:
        data = pickle.load(f)
    return data

def ss_vals(Pa,Pb,Δ=-20.,s=np.sqrt(20.),γ=2.,μ=30.,r=0.2,ζ=0.1,τ_th=30.,η=8.,τ_fc=0.1,χ=5.,**kwargs):
    Δ = np.expand_dims(Δ,1) # have to do this so that broadcasting works right
    n_ss = 2 * χ * τ_fc * (Pa**2 + Pb**2)
    T_ss = ζ * τ_th * ( η * r * ( Pa**2 + Pb**2 ) + n_ss / μ * ( Pa + Pb ) )
    Ba = (-1./2. + 1j*Δ - 1j*γ/2) + (1j - r) * Pa + (-1j - 1/μ) * n_ss + 1j * T_ss
    Bb = (-1./2. + 1j*Δ + 1j*γ/2) + (1j - r) * Pb + (-1j - 1/μ) * n_ss + 1j * T_ss
    # ϕ_a = np.arccos(s/(np.abs(Ba) * np.sqrt(Pa))) - np.angle(Ba) ;
    # ϕ_b = np.arccos(s/(np.abs(Bb) * np.sqrt(Pb))) - np.angle(Bb) ;
    ϕ_a = -1j*np.log(-s/(Ba * np.sqrt(Pa)))
    ϕ_b = -1j*np.log(-s/(Bb * np.sqrt(Pb)))
    a_ss = np.sqrt(Pa) * np.exp(1j*ϕ_a)
    b_ss = np.sqrt(Pb) * np.exp(1j*ϕ_b)
    return a_ss, b_ss, n_ss, T_ss
    #return a_ss, b_ss, n_ss, T_ss,Ba,Bb,ϕ_a,ϕ_b

def jacobian(a,b,n,T,Δ=-20.,s=np.sqrt(20.),γ=2.,μ=30.,r=0.2,ζ=0.1,τ_th=30.,η=8.,τ_fc=0.1,χ=5.):
    a_star = np.conj(a)
    b_star = np.conj(b)
    j = np.array([
        [-1./2. - 2. * a * a_star * r + 1j * (2. * a * a_star - n + T - γ/2 + Δ) - n/μ,
            -a**2 * (-1j + r),
            0.,
            0.,
            a * (-1j - 1/μ),
            1j * a],
        [-a_star**2 * (1j + r),
            -2 * a * a_star * (1j + r) + 1./2. * 1j * (1j + 2. * n - 2. * T + γ - 2. * Δ) - n/μ,
            0.,
            0.,
            a_star * (1j - 1./μ),
            -1j * a_star],
        [0.,
            0.,
            -1./2. - 2. * b * b_star * r + 1j * (2 * b * b_star - n + T + γ/2 + Δ) - n/μ,
             -b**2 * (-1j + r),
             b * (-1j - 1./μ),
             1j * b],
        [0.,
            0.,
            -b_star**2 * (1j + r),
            -1./2. - 2. * b * b_star * (1j + r) - 1./2. * 1j * (-2 * n + 2. * T + γ + 2. * Δ) - n/μ,
            b_star * (1j - 1./μ),
            -1j * b_star],
        [4. * a * a_star**2 * χ,
            4. * a**2 * a_star * χ,
            4. * b * b_star**2 * χ,
            4. * b**2 * b_star * χ,
            -(2/τ_fc),
            0.],
        [(a_star * ζ * (n + 2. * a * a_star * r * η * μ))/μ,
            ( a * ζ * (n + 2. * a * a_star * r * η * μ))/μ,
            ( b_star * ζ * (n + 2. * b * b_star * r * η * μ))/μ,
            (b * ζ * ( n + 2. * b * b_star * r * η * μ))/μ,
            (a * a_star + b * b_star) * ζ / μ,
            -(1./τ_th)]])
    return j

def jac_eigvals_sweep(Pa,Pb,Δ,s=np.sqrt(20.),γ=2.,μ=30.,r=0.2,ζ=0.1,τ_th=30.,η=8.,τ_fc=0.1,χ=5.,n_eqs=6,verbose=True,**kwargs):
    m = n_eqs,
    eigvals = np.zeros(Pa.shape+m,dtype=np.complex128)
    det_j = np.zeros(Pa.shape,dtype=np.float64)
    L = np.zeros(Pa.shape,dtype=np.float64)
    a,b,n,T = ss_vals(Pa,Pb,Δ=Δ,s=s,γ=γ,
                μ=μ,r=r,ζ=ζ,τ_th=τ_th,η=η,τ_fc=τ_fc,χ=χ)
    for ind in range(Pa.size):
        if Pa.ravel()[ind]:
            inds = np.unravel_index(ind,Pa.shape)
            # if verbose:
                # print(f'ind: {ind}')
                # print(f'inds: {inds}')
                # print(f'a[inds]: {a[inds]}')
            j = jacobian(a[inds],b[inds],n[inds],T[inds],Δ=Δ[inds[0]],s=s,
                γ=γ,μ=μ,r=r,ζ=ζ,τ_th=τ_th,η=η,τ_fc=τ_fc,χ=χ)
            try:
                eigvals[inds,:] = np.linalg.eigvals(j)
                det_j[inds] = np.linalg.det(j)
                L[inds] = ( np.matrix.trace(j)**2 - np.matrix.trace(np.matmul(j,j)) ) * np.matrix.trace(j) - 2 * det_j[inds]
            except:
                eigvals[inds,:] = np.zeros(6)
                det_j[inds] = 0.
                L[inds] = 0.
    return a,b,n,T,eigvals,det_j,L

def analyze_eigvals(data,return_data=False):
    Pa = data['Pa']
    eigvals = data['eigvals']
    shape = (Pa.shape[0],Pa.shape[1])
    n_rhp = np.zeros(shape)
    n_rhp_cplx = np.zeros(shape)
    rhp_fmax = np.zeros(shape)
    rhp_fmin = np.zeros(shape)
    eig_class = np.zeros(shape)
    for ind in range(eig_class.size):
        inds = np.unravel_index(ind,eig_class.shape)
        if Pa[inds]:
            e = eigvals[inds]
            n_rhp[inds] = np.sum(e.real>0)
            n_rhp_cplx[inds] = np.sum((e.real>0)*(np.abs(e.imag)>1e-5))
            if n_rhp_cplx[inds]:
                rhp_fmax[inds] = np.abs(e.imag).max()
                rhp_fmin[inds] = np.abs(e.imag).min()
            if n_rhp[inds]==1:
                eig_class[inds] = 1
            elif (n_rhp[inds]==2)and(n_rhp_cplx[inds]==0):
                eig_class[inds] = 2
            elif (n_rhp[inds]==3)and(n_rhp_cplx[inds]==0):
                eig_class[inds] = 3
            elif (n_rhp[inds]==4)and(n_rhp_cplx[inds]==0):
                eig_class[inds] = 4
            elif (n_rhp[inds]==2)and(n_rhp_cplx[inds]==2):
                eig_class[inds] = 5
            elif (n_rhp[inds]==3)and(n_rhp_cplx[inds]==2):
                eig_class[inds] = 6
            elif (n_rhp[inds]==4)and(n_rhp_cplx[inds]==2):
                eig_class[inds] = 7
            elif (n_rhp[inds]==4)and(n_rhp_cplx[inds]==4):
                eig_class[inds] = 8
            else:
                eig_class[inds] = 9
    data['n_rhp'] = n_rhp
    data['n_rhp_cplx'] = n_rhp_cplx
    data['rhp_fmax'] = rhp_fmax
    data['rhp_fmin'] = rhp_fmin
    data['eig_class'] = eig_class
    if return_data:
        return n_rhp, n_rhp_cplx, rhp_fmax, rhp_fmin, eig_class


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


def compute_PVΔ_sweep(p_expt=p_expt_def,p_mat=p_si,sweep_name='test',nEq=6,n_proc=8,data_dir=data_dir,verbose=True,return_data=False):
    """
    Find steady state solutions and corresponding Jacobian eigenvalues
    for the specified normalized 2-mode+free-carrier+thermal microring model
    parameters over the range of cold-cavity detuning (Δ), input power (s^2) and Vrb
    values supplied.
    """
    timestamp_str = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')
    sweep_dir_name = 'MM_PVDsweep_' + sweep_name + '_' + timestamp_str
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
            # ss_fname = f's{sind}_' + timestamp_str + '.csv'
            V_params_ss_list[sind]['name'] = f's{sind}'
            V_params_ss_list[sind]['data_dir'] = V_dir
            # run_mm_script(params=V_params_ss,data_dir=V_dir,data_fname=ss_fname)
            # Pa_ss,Pb_ss = import_mm_data(path.join(V_dir,ss_fname))
            # a_ss,b_ss,n_ss,T_ss,eigvals_ss,det_j_ss,L_ss = jac_eigvals_sweep(Pa_ss,Pb_ss,**V_params_ss)
        # map function onto pool of mathematica processes
        with Pool(processes=n_proc) as pool:
            out = pool.map(run_mm_script_parallel,V_params_ss_list)
            res = pool.map(process_mm_data_parallel,V_params_ss_list)
        # fill in dataset arrays, creating them first if Vind==0
        if Vind==0:
            Pa = np.zeros((ns,nV,nΔ,nEq),dtype=np.float64)
            Pb = np.zeros((ns,nV,nΔ,nEq),dtype=np.float64)
            a = np.zeros((ns,nV,nΔ,nEq),dtype=np.complex128)
            b = np.zeros((ns,nV,nΔ,nEq),dtype=np.complex128)
            n = np.zeros((ns,nV,nΔ,nEq),dtype=np.float64)
            T = np.zeros((ns,nV,nΔ,nEq),dtype=np.float64)
            eigvals = np.zeros((ns,nV,nΔ,nEq,nEq),dtype=np.complex128)
            det_j = np.zeros((ns,nV,nΔ,nEq),dtype=np.float64)
            L = np.zeros((ns,nV,nΔ,nEq),dtype=np.float64)
        for sind in range(ns):
            Pa[sind,Vind,:] = res[sind]['Pa']
            Pb[sind,Vind,:] = res[sind]['Pb']
            a[sind,Vind,:] = res[sind]['a']
            b[sind,Vind,:] = res[sind]['b']
            n[sind,Vind,:] = res[sind]['n']
            T[sind,Vind,:] = res[sind]['T']
            eigvals[sind,Vind,:] = res[sind]['eigvals']
            det_j[sind,Vind,:] = res[sind]['det_j']
            L[sind,Vind,:] = res[sind]['L']
        # save data
        V_data = {'Pa':Pa[:,Vind],'Pb':Pb[:,Vind],'a':a[:,Vind],'b':b[:,Vind],'n':n[:,Vind],'T':T[:,Vind],
            'eigvals':eigvals[:,Vind],'det_j':det_j[:,Vind],'L':L[:,Vind],**V_params}
        with open(V_data_fpath, 'wb') as f:
            pickle.dump(V_data,f)
        data = {'Pa':Pa,'Pb':Pb,'a':a,'b':b,'n':n,'T':T,
            'eigvals':eigvals,'det_j':det_j,'L':L,'V_params':V_params_list,**metadata}
        with open(sweep_data_fpath, 'wb') as f:
            pickle.dump(data,f)
    stop_timestamp_str = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')
    t_elapsed_sec = time()-t_start
    if verbose:
        print('PVΔ sweep finished at ' + stop_timestamp_str)
    metadata['t_stop'] = stop_timestamp_str
    metadata['t_elapsed_sec'] = t_elapsed_sec
    with open(mfpath, 'wb') as f:
        pickle.dump(metadata,f)
    if return_data:
        return data

def load_PVΔ_sweep(sweep_name='test',data_dir=data_dir,verbose=True,fpath=None):
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
##
##                    plotting functions
##
################################################################################

def plot_ss_power(Pa,Pb,Δ):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lines_a = ax.plot(Δ.squeeze(),Pa,'r.')
    lines_b = ax.plot(Δ.squeeze(),Pb,'b.')
    lines_tot = ax.plot(Δ.squeeze(),Pa+Pb,'k.')
    line_a = lines_a[0]
    line_b = lines_b[0]
    line_tot = lines_tot[0]
    ax.legend([line_a,line_b,line_tot],['Pa','Pb','total'])
    ax.set_xlabel('Δ [1]')
    ax.set_ylabel('Power [1]')
    ax.grid()
    plt.show()
    return ax

def plot_eigvals(eigvals,Δ,ind1_max=None,cmap=cm.viridis):
    if not(ind1_max):
        ind1_max = eigvals.shape[1]
    norm = Normalize(Δ.min(),Δ.max())
    sm = cm.ScalarMappable(norm, cmap)
    sm.set_array([]) # You have to set a dummy-array for this to work...
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ind0 in range(eigvals.shape[0]):
        for ind1 in range(ind1_max):
            plt.plot(eigvals[ind0,ind1,:].real,
                eigvals[ind0,ind1,:].imag,
                '.-',
                markersize=2,
                color=cmap(norm(Δ.squeeze()[ind0])))
    ax.grid()
    cbar = plt.colorbar(sm)
    cbar.set_label('Δ')
    plt.show()
    return ax
