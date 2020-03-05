# -*- coding: utf-8 -*-

"""
a-Al2O3.py

script for wg_disp, my adaptation of MPB waveguide dispersion code written
by Ryan Hamerly.

models amorphous Hafnia waveguide dispersion as a function of wavelength and
several geometric parameters

@author: dodd
"""


import numpy as np
from wg_disp import collect_wgparams_sweep, data_dir, n_proc_def
from instrumental import u


params = {'w_top_list': np.linspace(300,1500,9) * u.nm,
         'λ_list': np.linspace(0.32,1.8,40)*u.um,
         'λ_factor_list': np.array([1,0.95,1.05]),
         'θ_list': np.array([0,20]), # sidewall internal angle at top of core, degrees
         't_core_list': np.array([100,200,300,400,500,600,700,800,900,1000]) * u.nm,  # core thickness
         't_etch': 200 * u.nm,  # partial (or complete) etch depth
         'mat_core': 'Hafnia',
         'mat_clad': 'SiO2',
         'Xgrid': 4, # x lattice vector
         'Ygrid': 4, # y lattice vector
         'n_points': 32, # number of k-points simulated
         'n_bands': 4, # number of bands simulated
         'res': 128, # real-space resolution
 }

# collect_wgparams_sweep(params,
#                         sweep_name='HfO2_SiO2',
#                         n_proc=n_proc_def,
#                         data_dir=data_dir,
#                         verbose=True,
#                         return_data=False,
#                         )

params_air = params.copy()
params_air['mat_clad'] = 'Air'

restart_sweep_dir = '/homes/dodd/data/wgparams_sweep_HfO2_Air_2020_03_03_21_14_30'

collect_wgparams_sweep(params_air,
                        sweep_name='HfO2_Air',
                        n_proc=n_proc_def,
                        data_dir=data_dir,
                        sweep_dir=restart_sweep_dir,
                        verbose=True,
                        return_data=False,
                        )
