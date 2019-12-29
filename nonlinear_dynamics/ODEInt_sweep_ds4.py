# -*- coding: utf-8 -*-

# import sys
# mm_dir = "/home/dodd/google-drive/notebooks/IMEC V1 1550nm ring measurements/Thermal SOI ring cavity stability analysis"
# # import my MM data load library
# if mm_dir not in sys.path:
#     sys.path.append(mm_dir)
# import os
# os.environ['WolframKernel'] = '/usr/local/Wolfram/Mathematica/12.0/Executables/WolframKernel'
import ODEIntSweep as ois
from instrumental import u
import numpy as np

data_dir = ois.data_dir

p_si = {
    'r': 0.189, # nonlinear refraction 2π * n_2 / λ
    'γ': 3.1e-9 * u.cm/u.watt, # nonlinear refraction 2π * n_2 / λ,
    'μ': 30, # FCD/FCA ratio
    'σ': 1.45e-17 * u.cm**2, # FCA cross section (electron-hole average)
    'c_v': 1./(0.6 * u.degK / u.joule * u.cm**3), # silicon volumetric heat capacity near room temp
    'μ_e': 1400 * u.cm**2 / u.volt / u.second, # silicon electron mobility near room temp
    'μ_h': 450 * u.cm**2 / u.volt / u.second, # silicon hole mobility near room temp
    'c_v': 1./(0.6 * u.degK / u.joule * u.cm**3), # silicon volumetric heat capacity near room temp
}

# τ_th_list = np.array([30,100,300,1000])*u.ns
τ_th_list = np.array([30])*u.ns
V_rb = np.concatenate((np.arange(0,3,1),np.arange(4,24,6)))*u.volt


p_expt = {
    'λ': 1.55 * u.um, # free space laser wavelength
    'FSR': 601 * u.GHz, # measured ring FSR
    'd_ring': 40 * u.um, # microring diameter
    'FWHM': 350 * u.MHz, # measured Lorentzian linewidth of single resonance
    'FWHM_i': 190 * u.MHz, # measured "intrinsic" Lorentzian linewidth of single resonance
    'splitting': 1.15*u.GHz, # measured double-lorentzian splitting
    'Δ_min': -80*u.GHz, # f_cavity,0 - f_laser tuning minimum
    'Δ_max': 5*u.GHz, # f_cavity,0 - f_laser tuning maximum
    'P_bus_max': 2.0 * u.mW, # max input power in bus waveguide
    'V_rb': V_rb,
    'τ_th': 30 * u.ns, # thermal "time constant" to fit
    'df_dT': -9.7 * u.GHz / u.degK, # measured thermal tuning rate
    'τ_fc0': 150 * u.ps, # measured/modeled free carrier lifetime at Vrb=0
    'τ_fc_sat': 15 * u.ps, # measured/modeled minimum free carrier lifetime at Vrb>~15V
    'V_bi': 0.95 * u.volt, # measured/modeled diode built-in voltage
    'α_dB': 0.7/u.cm, # fit waveguide loss inside ring in units of dB/cm
    'α_abs_dB': 0.0825/u.cm, # based on Gil-Molina+Dainese papers, reporting 0.019/cm (not in dB/cm)
    'A': 0.1 * u.um**2, # mode effective area, from mode solver
    'n_eff': 2.6, # mode effective index from mode solver
    'β_2': 2 * u.ps**2/u.m, # GVD roughly measured, expected to be ~ 1 ps^2 / m
    'n_sf': 2, # number of significant figures to leave in the normalized parameters passed to mathematica. the fewer, the faster
    'δs': 0.4, # s step size (sqrt normalized input power)
    'δΔ': 0.2,  # Δ step size (cold cavity detuning)
    'τ_th_norm_ζ_product': 5,  # τ_th_norm * ζ, inferred from experiment data
    'χ3_sw_factor':1.5, # to reflect the effective χ(3) enhancement for standing waves vs. traveling waves = \int_0^\pi (E_{sw} * cos(z))^4 dz, with E_sw = sqrt(2)*E_tr
    'dΔdt': 3e-3,
}

tind_start=0 # in case you need to restart in the middle after a crash

if __name__ == '__main__':
    for tind,tt in enumerate(τ_th_list):
        if tind >= tind_start:
            p = p_expt.copy()
            p['τ_th'] = tt

            ois.collect_ODEInt_PVΔ_sweep(p_expt=p,
                                p_mat=p_si,
                                sweep_name=f'ds4_tau{tind}',
                                data_dir=data_dir,
                                verbose=True,
                                return_data=False,
                                )
