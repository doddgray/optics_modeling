# -*- coding: utf-8 -*-
# Copyright 2019-2020 Ryan Hamerly

# Note 19 Simulation 1
#
# Silicon EFISH.  Goal is to study several things including:
# - Quasi-Phase Matching
# - Field Confinement
# - GVD and group-velocity matching
#
# Geometry: strip of size W*H, surrounded by slab of width h.
#           Silicon-on-insulator (SOI).  Top can be either silicon or air.
#
# Variable ranges:
#   W   = [0.6λ, 0.7λ, ..., 1.2λ]
#   H   = [0.2, 0.3, ..., 0.7]
#   h/H = [0, 0.1, ..., 0.7]
#   λ   = [1.2, 1.3, 1.5, 1.8, 2.0] (and half-harmonics)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import meep as mp
import meep.materials as mats
from meep import mpb
import meeputils as mu
from wurlitzer import pipes, STDOUT
from io import StringIO

Wlist = np.linspace(0.4, 1.0, 25) #np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
Hlist = np.linspace(0.2, 0.7, 21) #np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
hlist = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
lamList = np.array([1.2, 1.3, 1.4, 1.55, 1.8, 2.0])

# ms = mpb.ModeSolver()

(Wgrid, Hgrid) = (2, 2)

Si = mp.Medium(index=3.5)
SiO2 = mp.Medium(index=1.45)
Air = mp.Medium(index=1)

n_points = 32
n_bands  = 4
res      = 32
k_points = mp.interpolate(n_points, [mp.Vector3(0.05, 0, 0), mp.Vector3(0.05*n_points, 0, 0)])

lat = mp.Lattice(size=mp.Vector3(0, Wgrid, Hgrid))

# ms = mpb.ModeSolver(geometry_lattice=lat,
#                     geometry=[],
#                     k_points=k_points,
#                     resolution=res,
#                     num_bands=n_bands,
#                     default_material=SiO2)

#%%  Functions get_wgparams and interp_grid


from scipy.interpolate import RectBivariateSpline

# get_wgparams:  Solves at some wavelength lam with size W, H, h, etc.
def get_wgparams(W, H, h, lam, do_func=None, solver=None):
    # if not (solver is None):
    #     ms = solver
    # Set up geometry.
    nSi = mu.get_index('Si', lam); ngSi = mu.get_ng('Si', lam)
    nSiO2 = mu.get_index('SiO2', lam); ngSiO2 = mu.get_ng('SiO2', lam)
    Si = mp.Medium(index=nSi); SiO2 = mp.Medium(index=nSiO2)

    ms = mpb.ModeSolver(geometry_lattice=lat,
                        geometry=[],
                        k_points=k_points,
                        resolution=res,
                        num_bands=n_bands,
                        default_material=SiO2)

    geom = [#mp.Block(size=mp.Vector3(mp.inf, mp.inf, -Wgrid/2), center=mp.Vector3(0, 0, -Wgrid/4), material=SiO2),
            mp.Block(size=mp.Vector3(mp.inf, W, H), material=Si),
            mp.Block(size=mp.Vector3(mp.inf, mp.inf, h), center=mp.Vector3(0, 0, -(H-h)/2), material=Si)]
    ms.geometry = geom
    ms.default_material = SiO2

    blackhole = StringIO()
    with pipes(stdout=blackhole, stderr=STDOUT):
        ms.init_params(mp.NO_PARITY, False)
        eps = np.array(ms.get_epsilon())
    isSi = (eps - nSiO2 ** 2) / (nSi ** 2 - nSiO2 ** 2)

    out = {}

    def get_fieldprops(ms, band):
        if (out != {}):
            return
        e = np.array(ms.get_efield(band))
        eps_ei2 = eps.reshape(eps.shape+(1,)) * np.abs(e[:, :, 0, :]**2)
        ei_pwr = eps_ei2.sum(axis=(0, 1))
        if (ei_pwr[1] > ei_pwr[2]):
            print ("TE mode!\n")
            # Calculate eps |E|^2.  Then get power in the silicon, and distribution along x.
            # Get n_g.  Adjust by material disperision factor (n_{g,mat}/n_{mat}), weighted by eps |E|^2 above.
            epwr = eps_ei2.sum(-1) / eps_ei2.sum()
            pSi_x = (epwr * isSi).sum(-1)
            pSi = pSi_x.sum()
            pSi_strip = (pSi_x * (np.abs(np.linspace(-Wgrid/2, Wgrid/2, len(pSi_x))) <= W/2)).sum()
            ng_nodisp = 1 / ms.compute_one_group_velocity_component(mp.Vector3(1, 0, 0), band)
            ng = ng_nodisp * (pSi * (ngSi / nSi) + (1 - pSi) * (ngSiO2 / nSiO2))

            # Output various stuff.
            out['band'] = band
            out['ng'] = ng
            out['pSi'] = pSi
            out['pSi_x'] = pSi_x #np.array(pSi_x[len(pSi_x)//2:] * 2)
            out['pSi_strip'] = pSi_strip

            if (do_func != None):
                do_func(out, ms, band)

    with pipes(stdout=blackhole, stderr=STDOUT):
        k = ms.find_k(mp.NO_PARITY,
                      1/lam,
                      1, 1,
                      mp.Vector3(1, 0, 0),
                      1e-4,
                      (nSiO2+nSi)/(2*lam), nSiO2/lam, nSi/lam,
                      get_fieldprops)

    if (out == {}):
        print ("TE mode not found.")
        return out

    out['neff'] = k[out['band']-1] / (1 / lam)

    print ("lam = {:.2f}, W = {:.2f}, H = {:.2f}, h = {:.2f}, neff = {:.3f}, ng = {:.3f}, "
           "pSi = {:.2f}, pStrip = {:.2f}".format(lam, W, H, h, out['neff'], out['ng'], out['pSi'], out['pSi_strip']))
    return out

# interp_grid:   Interpolates a coarse grid to a fine one (useful for nice contour plotting)
def interp_grid(z, fact=2):
    (nx, ny) = z.shape
    (x1, y1) = (np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    (x2, y2) = (np.linspace(0, 1, (nx - 1) * fact + 1), np.linspace(0, 1, (ny - 1) * fact + 1))
    sp = RectBivariateSpline(x1, y1, z, kx=1, ky=1)
    return sp(x2, y2)


###############################################################################
###############################################################################
##                                                                           ##
##                Uncomment to generate Note19-data.py                       ##
##                                                                           ##
###############################################################################
###############################################################################

outList = {}
for (m, lamFact) in enumerate([1, 2]):
    for (i, lam0) in enumerate(lamList):
        for (j, W0) in enumerate(Wlist):
            for (k, H) in enumerate(Hlist):
                for (l, h_over_H) in enumerate(hlist):
                    lam = lam0 * lamFact
                    W = W0 * lam0
                    h = H * h_over_H
                    outList[(m, i, j, k, l)] = get_wgparams(W, H, h, lam,)

neffList = np.zeros([2, len(lamList), len(Wlist), len(Hlist), len(hlist)], dtype=np.float)
ngList = np.zeros([2, len(lamList), len(Wlist), len(Hlist), len(hlist)], dtype=np.float)
pSiList = np.zeros([2, len(lamList), len(Wlist), len(Hlist), len(hlist)], dtype=np.float)
pStripList = np.zeros([2, len(lamList), len(Wlist), len(Hlist), len(hlist)], dtype=np.float)
pSi_xList = np.zeros([2, len(lamList), len(Wlist), len(Hlist), len(hlist), 64], dtype=np.float)

for ind in outList.keys():
    out = outList[ind]
    if out != {}:
        neffList[ind] = out['neff']
        ngList[ind] = out['ng']
        pSiList[ind] = out['pSi']
        pStripList[ind] = out['pSi_strip']
        pSi_xList[ind] = out['pSi_x']
    else:
        neffList[ind] = np.nan
        ngList[ind] = np.nan
        pSiList[ind] = np.nan
        pStripList[ind] = np.nan
        pSi_xList[ind] = np.nan

np.save("Note19-data.npy",
        {"neff": neffList,
         "ng": ngList,
         "pSi": pSiList,
         "pStrip": pStripList,
         "pSi_x": pSi_xList})

#%%  Reference code for generating Note19-data2.py


###############################################################################
###############################################################################
##                                                                           ##
##                Uncomment to generate Note19-data2.py                      ##
##                                                                           ##
###############################################################################
###############################################################################

# # assert False

outList = {}
for (m, lamFact) in enumerate([0.95, 1.05, 2*0.95, 2*1.05]):
    for (i, lam0) in enumerate(lamList):
        for (j, W0) in enumerate(Wlist):
            for (k, H) in enumerate(Hlist):
                for (l, h_over_H) in enumerate(hlist):
                    lam = lam0 * lamFact
                    W = W0 * lam0
                    h = H * h_over_H
                    outList[(m, i, j, k, l)] = get_wgparams(W, H, h, lam,)

neffList = np.zeros([4, len(lamList), len(Wlist), len(Hlist), len(hlist)], dtype=np.float)
ngList   = np.zeros([4, len(lamList), len(Wlist), len(Hlist), len(hlist)], dtype=np.float)
pSiList  = np.zeros([4, len(lamList), len(Wlist), len(Hlist), len(hlist)], dtype=np.float)
pStripList = np.zeros([4, len(lamList), len(Wlist), len(Hlist), len(hlist)], dtype=np.float)
pSi_xList  = np.zeros([4, len(lamList), len(Wlist), len(Hlist), len(hlist), 64], dtype=np.float)

for ind in outList.keys():
    out = outList[ind]
    if out != {}:
        neffList[ind] = out['neff']
        ngList[ind] = out['ng']
        pSiList[ind] = out['pSi']
        pStripList[ind] = out['pSi_strip']
        pSi_xList[ind] = out['pSi_x']
    else:
        neffList[ind] = np.nan
        ngList[ind] = np.nan
        pSiList[ind] = np.nan
        pStripList[ind] = np.nan
        pSi_xList[ind] = np.nan

np.save("Note19-data2.npy",
        {"neff": neffList,
         "ng": ngList,
         "pSi": pSiList,
         "pStrip": pStripList,
         "pSi_x": pSi_xList})

###############################################################################
###############################################################################
##                                                                           ##
##                Uncomment to generate Note19-data2.py                      ##
##                                                                           ##
###############################################################################
###############################################################################

ms2 = mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(0, Wgrid*1.5, Hgrid)),
                    geometry=[],
                    k_points=k_points,
                    resolution=64,
                    num_bands=n_bands,
                    default_material=mp.Medium(index=1))

ms2.init_params(mp.NO_PARITY, False)
ms2.solve_kpoint(mp.Vector3(1, 0, 0))


outList = {}
for (m, lamFact) in enumerate([1, 2, 0.95, 2*0.95, 1.05, 2*1.05]):
    for (i, lam0) in enumerate(lamList):
        for (j, W0) in enumerate(Wlist):
            for (k, H) in enumerate(Hlist):
                for (l, h_over_H) in enumerate(hlist):
                    lam = lam0 * lamFact
                    W = W0 * lam0
                    h = H * h_over_H
                    outList[(m, i, j, k, l)] = get_wgparams(W, H, h, lam, solver=ms)

neffList = np.zeros([6, len(lamList), len(Wlist), len(Hlist), len(hlist)], dtype=np.float)
ngList   = np.zeros([6, len(lamList), len(Wlist), len(Hlist), len(hlist)], dtype=np.float)
pSiList  = np.zeros([6, len(lamList), len(Wlist), len(Hlist), len(hlist)], dtype=np.float)
pStripList = np.zeros([6, len(lamList), len(Wlist), len(Hlist), len(hlist)], dtype=np.float)
pSi_xList  = np.zeros([6, len(lamList), len(Wlist), len(Hlist), len(hlist), 64], dtype=np.float)

for ind in outList.keys():
    out = outList[ind]
    if out != {}:
        neffList[ind] = out['neff']
        ngList[ind] = out['ng']
        pSiList[ind] = out['pSi']
        pStripList[ind] = out['pSi_strip']
        pSi_xList[ind] = out['pSi_x']
    else:
        neffList[ind] = np.nan
        ngList[ind] = np.nan
        pSiList[ind] = np.nan
        pStripList[ind] = np.nan
        pSi_xList[ind] = np.nan

np.save("Note19-data3.npy",
        {"neff": neffList,
         "ng": ngList,
         "pSi": pSiList,
         "pStrip": pStripList,
         "pSi_x": pSi_xList})
