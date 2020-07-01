# -*- coding: utf-8 -*-

"""
a-Al2O3.py

script for wg_disp, my adaptation of MPB waveguide dispersion code written
by Ryan Hamerly.

models silicon waveguide dispersion as a function of wavelength and
several geometric parameters

@author: dodd
"""


import numpy as np
from wg_disp import collect_wgparams_sweep, data_dir, n_proc_def, u
# from instrumental import u



params_f = {'w_top_list': np.linspace(1000,1500,5) * u.nm,
         'λ_list': np.linspace(1.2,1.8,11)*u.um,
         'λ_factor_list': 2*np.array([1,0.95,1.05]),
         'θ_list': np.array([0]), # sidewall internal angle at top of core, degrees
         't_core_list':  np.array([1800]) * u.nm,  # core thickness
         't_etch': 1600 * u.nm,  # partial (or complete) etch depth
         'mat_core': 'Si',
         'mat_clad': 'SiO2',
         'mat_subs': None,
         'Xgrid': 6, # x lattice vector
         'Ygrid': 6, # y lattice vector
         'n_points': 32, # number of k-points simulated
         'n_bands': 4, # number of bands simulated
         'res': 64, # real-space resolution
         'edge_gap': 1.0, # μm, gap between non-background objects that would normally extend to infinity in x or y and unit-cell edge, to avoid finding modes in substrates, slabs, etc.
 }

collect_wgparams_sweep(params_f,
                        sweep_name='VTT_thin_Si_f',
                        n_proc=n_proc_def,
                        data_dir=data_dir,
                        verbose=True,
                        return_data=False,
                        )


params_2f = {'w_top_list': np.linspace(1000,1500,5) * u.nm,
         'λ_list': np.linspace(1.2,1.8,11)*u.um,
         'λ_factor_list': np.array([1,0.95,1.05]),
         'θ_list': np.array([0]), # sidewall internal angle at top of core, degrees
         't_core_list':  np.array([1800]) * u.nm,  # core thickness
         't_etch': 1600 * u.nm,  # partial (or complete) etch depth
         'mat_core': 'Si',
         'mat_clad': 'SiO2',
         'mat_subs': None,
         'Xgrid': 6, # x lattice vector
         'Ygrid': 6, # y lattice vector
         'n_points': 32, # number of k-points simulated
         'n_bands': 4, # number of bands simulated
         'res': 64, # real-space resolution
         'edge_gap': 1.0, # μm, gap between non-background objects that would normally extend to infinity in x or y and unit-cell edge, to avoid finding modes in substrates, slabs, etc.
 }

collect_wgparams_sweep(params_2f,
                        sweep_name='VTT_thin_Si_2f',
                        n_proc=n_proc_def,
                        data_dir=data_dir,
                        verbose=True,
                        return_data=False,
                        )
