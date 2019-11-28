# -*- coding: utf-8 -*-

# import sys
# mm_dir = "/home/dodd/google-drive/notebooks/IMEC V1 1550nm ring measurements/Thermal SOI ring cavity stability analysis"
# # import my MM data load library
# if mm_dir not in sys.path:
#     sys.path.append(mm_dir)
# import os
# os.environ['WolframKernel'] = '/usr/local/Wolfram/Mathematica/12.0/Executables/WolframKernel'
import FixedPointSweep as fps
from instrumental import u
import numpy as np

data_dir = fps.data_dir

p_si = {
    'r': 0.189, # nonlinear refraction 2π * n_2 / λ
    'γ': 3.1e-9 * u.cm/u.watt, # nonlinear refraction 2π * n_2 / λ,
    'μ': 25, # FCD/FCA ratio
    'σ': 1.45e-17 * u.cm**2, # FCA cross section (electron-hole average)
    'c_v': 100./(0.6 * u.degK / u.joule * u.cm**3) # silicon volumetric heat capacity near room temp
}

# heat capacity increased 10x to check how this affects solutions.
# seems justifiable since I know the volume of the temperature distribution is ~100x greater than that of the optical mode volume

τ_th_list = np.array([30,100,300,1000])*u.ns
V_rb = np.concatenate((np.arange(0,3,0.5),np.arange(4,24,4)))*u.volt

p_expt = {
    'λ': 1.55 * u.um, # free space laser wavelength
    'FSR': 601 * u.GHz, # measured ring FSR
    'd_ring': 40 * u.um, # microring diameter
    'FWHM': 340 * u.MHz, # measured Lorentzian linewidth of single resonance
    'FWHM_i': 190 * u.MHz, # measured "intrinsic" Lorentzian linewidth of single resonance
    'splitting': 1.1*u.GHz, # measured double-lorentzian splitting
    'Δ_min': -35*u.GHz, # f_cavity,0 - f_laser tuning minimum
    'Δ_max': 10*u.GHz, # f_cavity,0 - f_laser tuning maximum
    'P_bus_max': 0.4 * u.mW, # max input power in bus waveguide
    'V_rb': np.arange(4,24,4)*u.volt,
    'τ_th': 50 * u.ns, # thermal "time constant" to fit
    'df_dT': -9.7 * u.GHz / u.degK, # measured thermal tuning rate
    'τ_fc0': 350 * u.ps, # measured/modeled free carrier lifetime at Vrb=0
    'τ_fc_sat': 3 * u.ps, # measured/modeled minimum free carrier lifetime at Vrb>~15V
    'V_bi': 1.1 * u.volt, # measured/modeled diode built-in voltage
    'α_dB': 0.7/u.cm, # fit waveguide loss inside ring in units of dB/cm
    'A': 0.1 * u.um**2, # mode effective area, from mode solver
    'β_2': 2 * u.ps**2/u.m, # GVD roughly measured, expected to be ~ 1 ps^2 / m
    'n_sf': 2, # number of significant figures to leave in the normalized parameters passed to mathematica. the fewer, the faster
    'δs': 0.2, # s step size (sqrt normalized input power)
    'δΔ': 0.2,  # Δ step size (cold cavity detuning)
}

tind_start=0 # in case you need to restart in the middle after a crash

if __name__ == '__main__':
    for tind,tt in enumerate(τ_th_list):
        if tind >= tind_start:
            p = p_expt.copy()
            p['τ_th'] = tt

            fps.compute_PVΔ_sweep(p_expt=p,
                                p_mat=p_si,
                                sweep_name=f'ds3_tau{tind}',
                                data_dir=data_dir,
                                verbose=True,
                                return_data=False,
                                )
