# -*- coding: utf-8 -*-

"""
SiNx_check.py

test script for wg_disp, my adaptation of MPB waveguide dispersion code written
by Ryan Hamerly.

Models dispersion of a-Si3N4 waveguides described in this paper:
 Okawachi,...,Lipson,Gaeta,
"Octave-spanning frequency comb generation in a silicon nitride chip"
 Optics Letters Vol. 36, Issue 17, pp. 3398-3400 (2011)
 https://doi.org/10.1364/OL.36.003398

@author: dodd
"""

import numpy as np
from wg_disp import collect_wgparams_sweep, data_dir, n_proc_def
from instrumental import u


params = {'w_top_list': np.array([1200,1650,2000]) * u.nm,
         'λ_list': np.linspace(0.78,2.6,30)*u.um,
         'λ_factor_list': np.array([1,0.95,1.05]),
         'θ': 0, # sidewall internal angle at top of core, degrees
         't_core': 730 * u.nm,  # core thickness
         't_etch': 730 * u.nm,  # partial (or complete) etch depth
         'mat_core': 'Si3N4',
         'mat_clad': 'SiO2',
         'Xgrid': 4, # x lattice vector
         'Ygrid': 4, # y lattice vector
         'n_points': 64, # number of k-points simulated
         'n_bands': 4, # number of bands simulated
         'res': 64, # real-space resolution
 }

collect_wgparams_sweep(params,
                        sweep_name='SiNx_check',
                        n_proc=n_proc_def,
                        data_dir=data_dir,
                        verbose=True,
                        return_data=False,
                        )
