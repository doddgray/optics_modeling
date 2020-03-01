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

from pathlib import Path
from multiprocessing import Pool
import socket
import pickle
from datetime import datetime
from time import time
from os import path, makedirs, chmod
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import meep as mp
import meep.materials as mats
from meep import mpb
import meeputils as mu
from wurlitzer import pipes, STDOUT
from io import StringIO
from instrumental import u


###
hostname = socket.gethostname()
if hostname=='dodd-laptop':
    data_dir = "/home/dodd/data/ODEInt_PVDsweep"
    n_proc_def = 6
else: # assume I'm on a MTL server or something
    home = str( Path.home() )
    data_dir = home+'/data/'
    n_proc_def = 30

###

########### Parameters ###############
### swept parameters
w_top_list = np.array([1200,1650,2000]) * u.nm
λ_list = np.linspace(0.78,2.6,3)*u.um
### fixed parameters
t_core = 730*u.nm # core thickness
t_etch = 730*u.nm # partial (or complete) etch depth



########## Simulation set up ##########
(Xgrid, Ygrid) = (4, 4) # simulated region in μm

n_points = 32
n_bands  = 4
res      = 32 #64
k_points = mp.interpolate(n_points, [mp.Vector3(0.05, 0, 0), mp.Vector3(0.05*n_points, 0, 0)])

lat = mp.Lattice(size=mp.Vector3(Xgrid, Ygrid,0))



def get_wgparams(w_top,θ,t_core,t_etch,lam,mat_core,mat_clad,do_func=None,):
    """
    Solves for mode, n_eff and ng_eff at some wavelength lam for a set of
    geometry and material parameters.
    """
    # convert all input parameters with length units to μm
    w_top = w_top.to(u.um).m
    t_core = t_core.to(u.um).m
    t_etch = t_etch.to(u.um).m
    lam = lam.to(u.um).m

    # get phase and group indices for these materials at this wavelength
    n_core = mu.get_index(mat_core, lam); ng_core = mu.get_ng(mat_core, lam)
    n_clad = mu.get_index(mat_clad, lam); ng_clad = mu.get_ng(mat_clad, lam)
    med_core = mp.Medium(index=n_core); med_clad = mp.Medium(index=n_clad)

    # Set up geometry.
    dx_base = np.tan(np.deg2rad(θ)) * t_etch
    verts_core = [mp.Vector3(-w_top/2.,t_core),
            mp.Vector3(w_top/2.,t_core),
            mp.Vector3(w_top/2+dx_base,t_core-t_etch),
            mp.Vector3(-w_top/2-dx_base,t_core-t_etch),
           ]
    core = mp.Prism(verts_core, height=mp.inf, material=med_core)
    if t_etch<t_core:   # partial etch
        slab = mp.Block(size=mp.Vector3(mp.inf, t_core-t_etch , mp.inf), center=mp.Vector3(0, (t_core-t_etch), 0),material=med_core)
        geom = [core,
                slab,
                ]
        # print('adding slab block')
    else:
        geom = [core,]

    ms = mpb.ModeSolver(geometry_lattice=lat,
                        geometry=[],
                        k_points=k_points,
                        resolution=res,
                        num_bands=n_bands,
                        default_material=med_clad)

    ms.geometry = geom
    ms.default_material = med_clad

    blackhole = StringIO()
    with pipes(stdout=blackhole, stderr=STDOUT):
        ms.init_params(mp.NO_PARITY, False)
        eps = np.array(ms.get_epsilon())
    mat_core_mask = (eps - n_clad ** 2) / (n_core ** 2 - n_clad ** 2)

    out = {}

    def get_fieldprops(ms, band):
        if (out != {}):
            return
        e = np.array(ms.get_efield(band))
        eps_ei2 = eps.reshape(eps.shape+(1,)) * np.abs(e[:, :, 0, :]**2)
        ei_pwr = eps_ei2.sum(axis=(0, 1))
        if (ei_pwr[0] > ei_pwr[1]):
            print ("TE mode!\n")
            # Calculate eps |E|^2.  Then get power in the core material, and distribution along x.
            # Get n_g.  Adjust by material disperision factor (n_{g,mat}/n_{mat}), weighted by eps |E|^2 above.
            epwr = eps_ei2.sum(-1) / eps_ei2.sum()
            p_mat_core_x = (epwr * mat_core_mask).sum(-1)
            p_mat_core = p_mat_core_x.sum()
            ng_nodisp = 1 / ms.compute_one_group_velocity_component(mp.Vector3(0, 0, 1), band)
            ng = ng_nodisp * (p_mat_core * (ng_core / n_core) + (1 - p_mat_core) * (ng_clad / n_clad))

            # Output various stuff.
            out['band'] = band
            out['ng'] = ng
            out['p_mat_core'] = p_mat_core
            out['p_mat_core_x'] = p_mat_core_x

            if (do_func != None):
                do_func(out, ms, band)

    # p = plt.pcolormesh(np.sqrt(eps))
    # plt.colorbar(p)
    # plt.show()

    with pipes(stdout=blackhole, stderr=STDOUT):
        k = ms.find_k(mp.NO_PARITY,
                      1/lam,
                      1,
                      1,
                      mp.Vector3(0, 0, 1),
                      1e-4,
                      (n_clad+n_core)/(2*lam),
                      n_clad/lam,
                      n_core/lam,
                      get_fieldprops,
                      )

    if (out == {}):
        print ("TE mode not found.")
        return out

    out['neff'] = k[out['band']-1] / (1 / lam)

    print ("lam = {:.2f}, w_top = {:.2f}, θ = {:.2f}, t_core = {:.2f}, t_etch = {:.2f}, neff = {:.3f}, ng = {:.3f}, "
           "p_mat_core = {:.2f}".format(lam, w_top, θ, t_core, t_etch, out['neff'], out['ng'], out['p_mat_core'],))
    return out



def get_wgparams_parallel(params):
    out = get_wgparams(params['w_top'],
                        params['θ'],
                        params['t_core'],
                        params['t_etch'],
                        params['λ'],
                        params['mat_core'],
                        params['mat_clad'],
                        )
    out.update(params)
    sweep_dir = params['sweep_dir']
    fpath = path.normpath(path.join(params['sweep_dir'],params['fname']))
    with open(fpath,'wb') as f:
            pickle.dump(out,f)
    return out



def collect_wgparams_sweep(params,sweep_name='test',n_proc=n_proc_def,data_dir=data_dir,verbose=True,return_data=False):
    """
    Find waveguide dispersion using mpb
    """
    timestamp_str = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')
    sweep_dir_name = 'wgparams_sweep_' + sweep_name + '_' + timestamp_str
    sweep_dir = path.normpath(path.join(data_dir,sweep_dir_name))
    makedirs(sweep_dir)
    sweep_data_fname = 'data.npy'
    sweep_data_fpath = path.join(sweep_dir,sweep_data_fname)
    mfname = 'metadata.dat'
    mfpath = path.join(sweep_dir,mfname)
    # add a couple other things to kwargs and save as a metadata dict
    metadata = params.copy()
    nλ = len(params['λ_list'])
    metadata['nλ'] = nλ
    nw = len(params['w_top_list'])
    metadata['nw'] = nw
    nfact = len(params['λ_factor_list'])
    metadata['nfact'] = nfact
    metadata['sweep_dir'] = sweep_dir
    t_start = time()
    start_timestamp_str = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')
    metadata['t_start'] = start_timestamp_str
    print(metadata)
    with open(mfpath, 'wb') as f:
        pickle.dump(metadata,f)
    params_list = []
    for (m, λ_factor) in enumerate(params['λ_factor_list']):
        for (i, λ0) in enumerate(params['λ_list']):
            for (j, ww) in enumerate(params['w_top_list']):
                p = metadata.copy()
                p['fname'] = f'{m}_{i}_{j}.dat'
                p['m'] = m
                p['j'] = j
                p['i'] = i
                p['λ'] = λ0 * λ_factor
                p['w_top'] = ww
                params_list += [p,]
    with Pool(processes=n_proc) as pool:
        out_list = pool.map(get_wgparams_parallel,params_list)
    # record stop time
    stop_timestamp_str = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')
    t_elapsed_sec = time()-t_start
    if verbose:
        print('wgparams sweep finished at ' + stop_timestamp_str)
    metadata['t_stop'] = stop_timestamp_str
    metadata['t_elapsed_sec'] = t_elapsed_sec
    with open(mfpath, 'wb') as f:
        pickle.dump(metadata,f)

    # compile output dataset
    px_len = len(out_list[0]['p_mat_core_x'])
    neff_list = np.zeros([nfact, nλ, nw], dtype=np.float)
    ng_list   = np.zeros([nfact, nλ, nw], dtype=np.float)
    p_mat_core_list  = np.zeros([nfact, nλ, nw], dtype=np.float)
    p_mat_core_x_list  = np.zeros([nfact, nλ, nw, px_len], dtype=np.float)

    for out_ind in range(len(out_list)):
        out = out_list[out_ind]
        m = out['m']
        i = out['i']
        j = out['j']
        inds = m,i,j
        if 'neff' in out.keys():
            neff_list[inds] = out['neff']
            ng_list[inds] = out['ng']
            p_mat_core_list[inds] = out['p_mat_core']
            p_mat_core_x_list[inds] = out['p_mat_core_x']
        else:
            neffList[inds] = np.nan
            ngList[inds] = np.nan
            p_mat_core_list[inds] = np.nan
            p_mat_core_x_list[inds] = np.nan

    np.save(sweep_data_fpath,
            {"neff": neff_list,
             "ng": ng_list,
             "p_mat_core": p_mat_core_list,
             "p_mat_core_x": p_mat_core_x_list})


###############################################################################
###############################################################################
##                                                                           ##
##                code to generate SiNx_check_data.npy                       ##
##                                                                           ##
###############################################################################
###############################################################################

# w_top_list = np.array([1200,1650,2000]) * u.nm
# λ_list = np.linspace(0.78,2.6,20)*u.um
# ### fixed parameters
# t_core = 730*u.nm # core thickness
# t_etch = 730*u.nm # partial (or complete) etch depth



params = {'w_top_list': np.array([1200,1650,2000]) * u.nm,
         'λ_list': np.linspace(0.78,2.6,30)*u.um,
         'λ_factor_list': np.array([1,0.95,1.05]),
         'θ': 0, # sidewall internal angle at top of core, degrees
         't_core': 730 * u.nm,  # core thickness
         't_etch': 730 * u.nm,  # partial (or complete) etch depth
         'mat_core': 'Si3N4',
         'mat_clad': 'SiO2',
 }

collect_wgparams_sweep(params,
                        sweep_name='SiNx_check',
                        n_proc=n_proc_def,
                        data_dir=data_dir,
                        verbose=True,
                        return_data=False,
                        )

# out_list = {}
# for (m, lamFact) in enumerate([1,0.95, 1.05]):
#     for (i, lam0) in enumerate(λ_list):
#         for (j, ww) in enumerate(w_top_list):
#             lam = lam0 * lamFact
#             # get_wgparams(w_top,θ,t_core,t_etch,lam,mat_core,mat_clad,do_func=None,)
#             out_list[(m, i, j)] = get_wgparams(ww,
#                                                 0, # θ, degrees
#                                                 t_core,
#                                                 t_core,
#                                                 lam,
#                                                 'Si3N4',
#                                                 'SiO2',
#                                                 do_func=None,
#                                                 )
#
# px_len = len(out_list[0,0,0]['p_mat_core_x'])
# # print(f'ind0: {ind0}')
# # print(f'len px: {len(out_list[ind0]))}')
#
# neff_list = np.zeros([3, len(λ_list), len(w_top_list)], dtype=np.float)
# ng_list   = np.zeros([3, len(λ_list), len(w_top_list)], dtype=np.float)
# p_mat_core_list  = np.zeros([3, len(λ_list), len(w_top_list)], dtype=np.float)
# p_mat_core_x_list  = np.zeros([3, len(λ_list), len(w_top_list), px_len], dtype=np.float)
#
# for ind in out_list.keys():
#     out = out_list[ind]
#     if out != {}:
#         neff_list[ind] = out['neff']
#         ng_list[ind] = out['ng']
#         p_mat_core_list[ind] = out['p_mat_core']
#         p_mat_core_x_list[ind] = out['p_mat_core_x']
#     else:
#         neffList[ind] = np.nan
#         ngList[ind] = np.nan
#         p_mat_core_list[ind] = np.nan
#         p_mat_core_x_list[ind] = np.nan
#
# np.save("SiNx_check_data.npy",
#         {"neff": neff_list,
#          "ng": ng_list,
#          "p_mat_core": p_mat_core_list,
#          "p_mat_core_x": p_mat_core_x_list})
