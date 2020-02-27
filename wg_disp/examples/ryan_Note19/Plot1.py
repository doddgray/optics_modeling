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

data1 = np.load("Note19-data.npy", allow_pickle=True)[()]
data2 = np.load("Note19-data2.npy", allow_pickle=True)[()]

def combine_data(lbl):
    d1 = data1[lbl]; d2 = data2[lbl]; s1 = list(d1.shape); s1[0] = d1.shape[0] + d2.shape[0]
    d3 = np.zeros(s1, dtype=np.float); d3[:2] = d1; d3[2:4] = d2[::2]; d3[4:] = d2[1::2]; return d3
    # Note I needed to flip these indices around to get the desired lambda-factor order 1, 2, 0.95, 2*0.95, 1.05, 2*1.05
(neffList, ngList, pSiList, pStripList, pSi_xList) = map(combine_data, ['neff', 'ng', 'pSi', 'pStrip', 'pSi_x'])
del data1, data2



###############################################################################
###############################################################################
##                                                                           ##
##        Fig. 1: EFISH QPM wavelength, GVM group-index difference,          ##
##                and field confinement, for 1.3->2.6µm                      ##
##                                                                           ##
###############################################################################
###############################################################################


m = 3
lam = 1.3; lam_i = lamList.tolist().index(lam)

plt.clf()
(W, H) = (10., 6.)
f = plt.figure(figsize=(W, H))
(wax, hax, xax, yax, dxax, dyax) = np.array([1.65, 1.6, 0.6, 0.5, 0.1, 0.1])/np.array([W, H, W, H, W, H])
wbar = 0.1/W
axList = np.array([[f.add_axes([xax+(wax+dxax)*j, yax+(hax+dyax)*i, wax, hax]) for j in range(5)]
                   for i in range(3)])[::-1]
cbarList = np.array([f.add_axes([xax+(wax+dxax)*5, yax+(hax+dyax)*i, wbar, hax]) for i in range(3)])[::-1]
ext = [0.4*lam, 1.0*lam, 0.2, 0.7]
for ax in axList.flatten():
    ax.set_xlim(*ext[:2]); ax.set_ylim(*ext[2:]); ax.set_xticks([0.6, 0.8, 1.0, 1.2])
for ax in axList[:2].flatten(): ax.set_xticklabels([])
for ax in axList[:, 1:].flatten(): ax.set_yticklabels([])
axList[2, 2].set_xlabel(r"Width $W$ ($\mu$m)"); axList[1, 0].set_ylabel(r"Height $H$ ($\mu$m)")

qpmLambda = lam/(neffList[0, lam_i, :, :, :5] - neffList[1, lam_i, :, :, :5]).transpose(1, 0, 2)
gvm = (ngList[0, lam_i, :, :, :5] - ngList[1, lam_i, :, :, :5]).transpose(1, 0, 2)
pStrip = pStripList[1, lam_i, :, :, :5].transpose(1, 0, 2)

for (i, axRow, cbar, figdata, lvl, lbl) in zip([0, 1, 2], axList, cbarList, [qpmLambda, gvm, pStrip],
        [[1.5, 2, 2.5, 3, 3.5], [0, 0.5, 1, 1.5], [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
        [r"QPM $\Lambda$ ($\mu$m)", r"GVM $\Delta n_g$", r"Confinement $P(2\lambda)$"]):
    (vmin, vmax) = (np.nan_to_num(figdata, nan=1e10).min(), np.nan_to_num(figdata, nan=-1e10).max())
    for (j, ax) in enumerate(axRow):
        cax = ax.imshow(figdata[:, :, j], extent=ext, cmap='jet', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
        cax2 = ax.contour(figdata[:, :, j], extent=ext, levels=lvl, colors=['k'], linewidths=1)
    cbar = f.colorbar(cax, cax=cbar, orientation='vertical'); cbar.add_lines(cax2); cbar.set_label(lbl)
axList[0, 0].set_title("$h/H = 0$")
for (ax, h) in zip(axList[0, 1:], [0.1, 0.2, 0.3, 0.4]):
    ax.set_title("{:.1f}".format(h))
for i in range(3):
    for j in range(5):
        axList[i, j].contour(gvm[:, :, j], extent=ext, levels=[0], colors=['w'], linewidths=[1.5])

f.text(0.5, 0.97, r"SOI EFISH: {:.2f}$\mu$m $\rightarrow$ {:.2f}$\mu$m".format(lam, 2*lam),
       fontsize=12, horizontalalignment='center')
axList[1, 3].annotate("GVM = 0", (0.65, 0.45), color='w')

f.savefig("./Fig1.pdf", format="pdf")
plt.show()

###############################################################################
###############################################################################
##                                                                           ##
##            Fig. 2: Same as Fig. 1, but for more wavelengths               ##
##                                                                           ##
###############################################################################
###############################################################################

plt.clf()
(W, H) = (10., 10.)
f = plt.figure(figsize=(W, H))
(wax, hax, xax, yax, dxax, dyax) = np.array([1.4, 1.68, 0.6, 0.5, 0.1, 0.1])/np.array([W, H, W, H, W, H])
wbar = 0.1/W
axList = np.array([[f.add_axes([xax+(wax+dxax)*j, yax+(hax+dyax)*i, wax, hax]) for j in range(6)]
                   for i in range(5)])[::-1]
bbox = mpl.transforms.Bbox([[0.1, 0.06], [0.9, 0.36]])
bbox2 = f.transFigure.inverted().transform_bbox(axList[-1, -1].transAxes.transform_bbox(
    mpl.transforms.Bbox([[0.15, 0.28], [0.85, 0.33]])))
cbar = f.add_axes([bbox2.x0, bbox2.y0, bbox2.width, bbox2.height])
ext = [0.4*lam, 1.0*lam, 0.2, 0.7]
for ax in axList[:-1].flatten(): ax.set_xticklabels([])
for ax in axList[:, 1:].flatten(): ax.set_yticklabels([])
axList[-1, 2].set_xlabel(r"Width $W$ ($\mu$m)").set_position([1, 0]); axList[2, 0].set_ylabel(r"Height $H$ ($\mu$m)")
for (lam, ax) in zip(lamList, axList[0]): ax.set_title(r"{:.2f} $\rightarrow$ {:.2f}$\mu$m".format(lam, 2 * lam))
for (h, ax) in zip(hlist, axList[::-1, -1]): ax2 = ax.twinx(); ax2.set_yticks([]); ax2.set_ylabel("$h/H = {:.1f}$".format(h))
f.text(0.5, 0.97, r"SOI EFISH: QPM $\Lambda$, GVM $\Delta n_g$, and confinement $P(2\lambda)$",
       fontsize=12, horizontalalignment='center')

qpmLambda = lamList.reshape([6, 1, 1, 1])/(neffList[0, :, :, :, :5] - neffList[1, :, :, :, :5])
gvm = (ngList[0, :, :, :, :5] - ngList[1, :, :, :, :5])
pStrip = pStripList[1, :, :, :, :5]


for (lam, axCol) in zip(lamList, axList.T):
    for (h, ax) in zip(hlist, axCol[::-1]):
        lam_i = lamList.tolist().index(lam); m = hlist.tolist().index(h)
        ext = [0.4 * lam, 1.0 * lam, 0.2, 0.7]
        xlist = np.linspace(ext[0], ext[1], qpmLambda.shape[1])
        ax.set_xlim(*ext[:2]); ax.set_ylim(*ext[2:])
        qpm_i = interp_grid(np.nan_to_num(qpmLambda[lam_i, :, :, m].T / np.outer(1, xlist), nan=1e10), 4)
        pStrip_i = interp_grid(np.nan_to_num(pStrip[lam_i, :, :, m].T, nan=0), 4)
        gvm_i = interp_grid(np.nan_to_num(gvm[lam_i, :, :, m].T, nan=0), 4)
        isgood = (qpm_i < 10); qpm_i /= isgood; pStrip_i /= isgood; gvm_i /= isgood
        cax = ax.contourf(qpm_i, extent=ext, levels=np.arange(1, 5.001, 0.25), cmap='viridis_r')
        cax2 = ax.contour(qpm_i, extent=ext, levels=np.arange(1, 4.001, 0.25),
                          colors='k', linewidths=1, linestyles=['-', ':', ':', ':'])
        ax.contour(pStrip_i, extent=ext, levels=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9], colors='#444444',
                   linewidths=1, linestyles=['--', '--', '--', '--', '-', '--'])
        ax.contour(gvm_i, extent=ext, levels=[0], colors='w', linewidths=1.5)
        ax.contourf(qpm_i / (pStrip_i < 0.8), extent=ext, levels=np.arange(1, 5.001, 0.25), cmap='Greys')
cbar = f.colorbar(cax, cax=cbar, orientation='horizontal')
cbar.add_lines(cax2); cbar.set_ticks([1, 2, 3, 4, 5]); cbar.set_label(r"$\Lambda/W$", labelpad=0)
bbox = axList[-1, -1].transData.inverted().transform_bbox(axList[-1, -1].transAxes.transform_bbox(bbox))
axList[-1, -1].add_patch(mpl.patches.Rectangle(bbox.p0, bbox.width, bbox.height, linewidth=0, facecolor='w',
                                               alpha=0.8, zorder=3))
axList[-1, 4].text(0.8, 0.24, "Poor con-\nfinement", fontsize=8, bbox=dict(lw=0, facecolor='w', alpha=0.8))
axList[-1, 4].annotate("P = 0.8", (1.75, 0.33), color='k', horizontalalignment='right')
axList[-1, 4].annotate("0.9", (1.75, 0.46), color='k', horizontalalignment='right')
axList[-1, 4].annotate("0.7", (1.75, 0.24), color='k', horizontalalignment='right')
axList[-1, 4].annotate("GVM = 0", (1.2, 0.4), color='w', xytext=(1.25, 0.62),
                       arrowprops=dict(arrowstyle='->', color='w'))
axList[2, 0].plot([0.67], [0.5], '*', ms=10, mec='k', mfc='w')
axList[1, 1].plot([0.9], [0.6], '*', ms=10, mec='k', mfc='w')
axList[1, 2].plot([1.12], [0.55], '*', ms=10, mec='k', mfc='w')
axList[1, 3].plot([1.2], [0.5], '*', ms=10, mec='k', mfc='w')
axList[1, 4].plot([1.4], [0.6], '*', ms=10, mec='k', mfc='w')
axList[2, 5].plot([1.5], [0.6], '*', ms=10, mec='k', mfc='w')
f.savefig("./Fig2.pdf", format="pdf")

plt.show()

###############################################################################
###############################################################################
##                                                                           ##
##           Fig. 3: Modes and confinement for GVM-matched EFISH             ##
##                                                                           ##
###############################################################################
###############################################################################

ms2 = mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(0, Wgrid*1.5, Hgrid)),
                    geometry=[],
                    k_points=k_points,
                    resolution=64,
                    num_bands=n_bands,
                    default_material=SiO2)

ms2.init_params(mp.NO_PARITY, False)
ms2.solve_kpoint(mp.Vector3(1, 0, 0))

W0list = [0.68, 0.9, 1.12, 1.2, 1.4, 1.5]
H0list = [0.5, 0.6, 0.55, 0.5, 0.6, 0.6]
h0_Hlist = [0.2, 0.3, 0.3, 0.3, 0.3, 0.2]
pars1list = [0] * 6
pars2list = [0] * 6
P1list = [0] * 6
P2list = [0] * 6
epsList = [0] * 6

for (i, W, H, hH, lam) in zip(range(6), W0list, H0list, h0_Hlist, lamList):
    pars1list[i] = get_wgparams(W, H, hH*H, lam, solver=ms2); P1list[i] = ms2.get_dpwr(1)
    pars2list[i] = get_wgparams(W, H, hH*H, 2*lam, solver=ms2); P2list[i] = ms2.get_dpwr(1)
    epsList[i] = ms2.get_epsilon()

for (lam, pars1, pars2) in zip(lamList, pars1list, pars2list):
    print("{:.2f}um: neff = [{:.3f}, {:.3f}], ng = [{:.3f}, {:.3f}], Lambda = {:.2f}um".format(
        lam, pars1['neff'], pars2['neff'], pars1['ng'], pars2['ng'], lam/(pars1['neff'] - pars2['neff'])))

#%%  (fig. 3 cont.)

wh = [10., 5.25]
(x1, w1) = np.array([0.7, 1.5])/wh[0]
(y1, h1, s1, h2, s2) = np.array([0.5, 1.4, 0.6, 1.0, 0.2])/wh[1]
f = plt.figure(figsize=(10, 6))
plt.clf()
axList = np.array([[f.add_axes([x1+w1*j, y1, w1, h1]) for j in range(6)]] +
                  [[f.add_axes([x1+w1*j, y1+h1+s1+(h2+s2)*i, w1, h2]) for j in range(6)] for i in [0, 1]])[::-1]
ext = [-1.5*Wgrid/2, 1.5*Wgrid/2, -Hgrid/2, Hgrid/2]

for ((ax1, ax2, ax3), lam, pars1, pars2, P1, P2, eps, W, H, hH) in zip(
        axList.T, lamList, pars1list, pars2list, P1list, P2list, epsList, W0list, H0list, h0_Hlist):
    ax1.imshow(interp_grid(P1.T, 4), origin='lower', extent=ext, cmap='Blues')
    ax1.contour(eps.T, origin='lower', extent=ext, levels=[3], colors='k', linewidths=1)
    ax2.imshow(interp_grid(P2.T, 4), origin='lower', extent=ext, cmap='Reds')
    ax2.contour(eps.T, origin='lower', extent=ext, levels=[3], colors='k', linewidths=1)

    px1 = np.cumsum(pars1['pSi_x']) + np.cumsum(pars1['pSi_x'][::-1]); px1 = px1[:len(px1)//2][::-1]
    px2 = np.cumsum(pars2['pSi_x']) + np.cumsum(pars2['pSi_x'][::-1]); px2 = px2[:len(px2)//2][::-1]

    x = np.linspace(0, 1.5*Wgrid/2, len(px1))-W/2
    ax3.plot(x, px1/(x>=0), '-', x, px1/(x<0), '--', c=mpl.cm.Blues(0.7))
    ax3.plot(x, px2/(x>=0), '-', x, px2/(x<0), '--', c=mpl.cm.Reds(0.7))
    ax3.set_yscale("log"); ax3.set_ylim(1e-4, 1)
    ax3.grid(ls=':')
    ax3.fill_betweenx([1e-5, 10], [0.0, 0.0], [-0.6, -0.6], facecolor='#eeeeee', edgecolor='#888888', lw=1)
    plt.grid()
    ax1.set_title(r"{:.2f}$\mu$m".format(lam))

for ax in axList[0]: ax.set_xticklabels([])
for ax in axList[:-1, 1:].flatten(): ax.set_yticklabels([])
for ax in axList[-1]: ax.set_xlim(-0.5, 0.5); ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
for ax in axList[-1, 1:]: ax.yaxis.set_major_formatter(mpl.ticker.NullFormatter())
axList[2, 0].set_ylabel("Power in Doped Region")
axList[2, 2].set_xlabel(r"Dopant distance from Slab ($\mu$m)").set_position([1.0, 0.0])
axList[1, 2].set_xlabel(r"Dimensions ($\mu$m)").set_position([1.0, 0.0])
axList[0, 0].set_ylabel(r"Pump $\lambda$"); axList[1, 0].set_ylabel(r"Signal $2\lambda$")
f.text(0.5, 0.96, r"Group-velocity matched SOI EFISH: Modes and Confinement",
       fontsize=12, horizontalalignment='center')

f.savefig("Note19-Fig3.pdf", format="pdf")
plt.show()


#%% MOO COW!!!!!

factList = np.array([1.0, 2.0, 0.95, 2*0.95, 1.05, 2*1.05])
betaList = (neffList * 2*np.pi / (np.outer(factList, lamList).reshape([6, 6, 1, 1, 1])))
betaList = betaList.reshape((3, 2) + betaList.shape[1:])
ngList2 = ngList.reshape(betaList.shape)
# Compute β'(λ) and β''(λ)
lam = np.outer([1, 2], lamList).reshape(2, 6, 1, 1, 1); dlam = lam * 0.05
beta1_lam = (betaList[2] - betaList[1]) / (2 * dlam)
beta2_lam = (betaList[2] + betaList[1] - 2*betaList[0]) / (dlam**2)
# Compute β'(k) and β''(k) (in SI units)  Since k = 2πc/λ and β'(k) = ng/c, these are given by:
#   β'(k)  = -(λ²/2πc) β'(λ)
#   β''(k) = -(λ²/2πc²) ng'(λ)
c = 3e8 / 1e-6
beta1 = -lam**2/(2*np.pi*c) * beta1_lam
#beta2 = (lam**2/(2*np.pi*c))**2 * (beta2_lam + (2/lam) * beta1_lam)
beta2 = -lam**2/(2*np.pi*c**2) * (ngList2[2] - ngList2[1]) / (2*1e-6*dlam)
ng = c * beta1


###############################################################################
###############################################################################
##                                                                           ##
##                Fig. 4: Plot of Δn_g, β_2(λ), and β_2(2λ)                  ##
##                                                                           ##
###############################################################################
###############################################################################

lam_i = 3; lam = lamList[lam_i]

plt.clf(); (W, H) = (10., 6.); f = plt.figure(figsize=(W, H))
(wax, hax, xax, yax, dxax, dyax, wbar) = np.array([1.65, 1.6, 0.6, 0.5, 0.1, 0.1, 0.1])/np.array([W, H, W, H, W, H, W])
axList = np.array([[f.add_axes([xax+(wax+dxax)*j, yax+(hax+dyax)*i, wax, hax]) for j in range(5)]
                   for i in range(3)])[::-1]
cbarList = np.array([f.add_axes([xax+(wax+dxax)*5, yax+(hax+dyax)*i, wbar, hax]) for i in range(3)])[::-1]
ext = [0.4*lam, 1.0*lam, 0.2, 0.7]
for ax in axList.flatten():
    ax.set_xlim(*ext[:2]); ax.set_ylim(*ext[2:]); #ax.set_xticks([0.6, 0.8, 1.0, 1.2])
for ax in axList[:2].flatten(): ax.set_xticklabels([])
for ax in axList[:, 1:].flatten(): ax.set_yticklabels([])
axList[2, 2].set_xlabel(r"Width $W$ ($\mu$m)"); axList[1, 0].set_ylabel(r"Height $H$ ($\mu$m)")

qpmLambda = lam/(neffList[0, lam_i, :, :, :5] - neffList[1, lam_i, :, :, :5])
pStrip = pStripList[1, lam_i, :, :, :5]
gvm = (ngList[0, lam_i, :, :, :5] - ngList[1, lam_i, :, :, :5])
beta2_1 = beta2[0, lam_i, :, :, :5] * 1e24; beta2_2 = beta2[1, lam_i, :, :, :5] * 1e24

for (i, axRow, cbar, figdata, lvl, lbl) in zip([0, 1, 2], axList, cbarList, [gvm, beta2_1, beta2_2],
        [[0, 0.5, 1, 1.5], [-0.5, 0, 0.5, 1, 1.5], [0, 10, 20, 30, 40]],
        [r"GVM $\Delta n_g$", r"Pump GVD $\beta_2(\lambda)$", r"Signal GVD $\beta_2(2\lambda)$"]):
    (vmin, vmax) = (np.nan_to_num(figdata, nan=1e10).min(), np.nan_to_num(figdata, nan=-1e10).max())
    for (j, ax) in enumerate(axRow):
        cax = ax.imshow(figdata[:, :, j].T, extent=ext, cmap='jet', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
        cax2 = ax.contour(figdata[:, :, j].T, extent=ext, levels=lvl, colors=['k'], linewidths=1)
        cax3 = ax.contour(figdata[:, :, j].T, extent=ext, levels=[0], colors=['w'], linewidths=[1.5])
    cbar = f.colorbar(cax, cax=cbar, orientation='vertical'); cbar.add_lines(cax2)
    cbar.set_label(lbl)
axList[0, 0].set_title("$h/H = 0$")
for (ax, h) in zip(axList[0, 1:], [0.1, 0.2, 0.3, 0.4]):
    ax.set_title("{:.1f}".format(h))

f.text(0.5, 0.97, r"SOI EFISH GVM and GVD: {:.2f}$\mu$m $\rightarrow$ {:.2f}$\mu$m".format(lam, 2*lam),
       fontsize=12, horizontalalignment='center')
axList[2, 3].annotate("Zero contour", (1, 0.42), color='w')

f.savefig("./Fig4.pdf", format="pdf")
plt.show()


###############################################################################
###############################################################################
##                                                                           ##
##               Fig. 5: Plot of GVD & GVM for all wavelengths               ##
##                                                                           ##
###############################################################################
###############################################################################

plt.clf()
(W, H) = (10., 10.)
f = plt.figure(figsize=(W, H))
(wax, hax, xax, yax, dxax, dyax) = np.array([1.4, 1.68, 0.6, 0.5, 0.1, 0.1])/np.array([W, H, W, H, W, H])
wbar = 0.1/W
axList = np.array([[f.add_axes([xax+(wax+dxax)*j, yax+(hax+dyax)*i, wax, hax]) for j in range(6)]
                   for i in range(5)])[::-1]
bbox = mpl.transforms.Bbox([[0.1, 0.06], [0.9, 0.36]])
bbox2 = f.transFigure.inverted().transform_bbox(axList[-1, -1].transAxes.transform_bbox(
    mpl.transforms.Bbox([[0.15, 0.28], [0.85, 0.33]])))
cbar = f.add_axes([bbox2.x0, bbox2.y0, bbox2.width, bbox2.height])
ext = [0.4*lam, 1.0*lam, 0.2, 0.7]
for ax in axList[:-1].flatten(): ax.set_xticklabels([])
for ax in axList[:, 1:].flatten(): ax.set_yticklabels([])
axList[-1, 2].set_xlabel(r"Width $W$ ($\mu$m)").set_position([1, 0]); axList[2, 0].set_ylabel(r"Height $H$ ($\mu$m)")
for (lam, ax) in zip(lamList, axList[0]): ax.set_title(r"{:.2f} $\rightarrow$ {:.2f}$\mu$m".format(lam, 2 * lam))
for (h, ax) in zip(hlist, axList[::-1, -1]): ax2 = ax.twinx(); ax2.set_yticks([]); ax2.set_ylabel("$h/H = {:.1f}$".format(h))
f.text(0.5, 0.97, r"SOI EFISH: QPM $\Lambda$, GVM $\Delta n_g$, GVD $\beta_2$, and confinement $P(2\lambda)$",
       fontsize=12, horizontalalignment='center')

qpmLambda = lamList.reshape([6, 1, 1, 1])/(neffList[0, :, :, :, :5] - neffList[1, :, :, :, :5])
gvm = (ngList[0, :, :, :, :5] - ngList[1, :, :, :, :5])
pStrip = pStripList[1, :, :, :, :5]
beta2_1 = beta2[0, :, :, :, :5] * 1e24; beta2_2 = beta2[1, :, :, :, :5] * 1e24

for (lam, axCol) in zip(lamList, axList.T):
    for (h, ax) in zip(hlist, axCol[::-1]):
        lam_i = lamList.tolist().index(lam); m = hlist.tolist().index(h)
        ext = [0.4 * lam, 1.0 * lam, 0.2, 0.7]
        xlist = np.linspace(ext[0], ext[1], qpmLambda.shape[1])
        ax.set_xlim(*ext[:2]); ax.set_ylim(*ext[2:])
        qpm_i = interp_grid(np.nan_to_num(qpmLambda[lam_i, :, :, m].T / np.outer(1, xlist), nan=1e10), 4)
        pStrip_i = interp_grid(np.nan_to_num(pStrip[lam_i, :, :, m].T, nan=0), 4)
        gvm_i = interp_grid(np.nan_to_num(gvm[lam_i, :, :, m].T, nan=0), 4)
        beta2_1i = interp_grid(np.nan_to_num(beta2_1[lam_i, :, :, m].T, nan=0), 4)
        beta2_2i = interp_grid(np.nan_to_num(beta2_2[lam_i, :, :, m].T, nan=0), 4)
        isgood = (qpm_i < 10); qpm_i /= isgood; pStrip_i /= isgood; gvm_i /= isgood
        beta2_1i /= isgood; beta2_2i /= isgood
        cax = ax.contourf(qpm_i, extent=ext, levels=np.arange(1, 5.001, 0.25), cmap='viridis_r')
        cax2 = ax.contour(qpm_i, extent=ext, levels=np.arange(1, 4.001, 0.25),
                          colors='k', linewidths=1, linestyles=['-', ':', ':', ':'])
        ax.contour(pStrip_i, extent=ext, levels=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9], colors='#444444',
                   linewidths=1, linestyles=['--', '--', '--', '--', '-', '--'])
        ax.contour(gvm_i, extent=ext, levels=[0], colors='w', linewidths=1.5)
        ax.contour(beta2_1i, extent=ext, levels=[0], colors='b', linewidths=1.5)
        ax.contour(beta2_2i, extent=ext, levels=[0], colors='r', linewidths=1.5)
        ax.contour(beta2_1i - 0.5*beta2_2i, extent=ext, levels=[0], colors='r', linestyles='--', linewidths=1.5)
        ax.contourf(qpm_i / (pStrip_i < 0.8), extent=ext, levels=np.arange(1, 5.001, 0.25), cmap='Greys')
cbar = f.colorbar(cax, cax=cbar, orientation='horizontal')
cbar.add_lines(cax2); cbar.set_ticks([1, 2, 3, 4, 5]); cbar.set_label(r"$\Lambda/W$", labelpad=0)
bbox = axList[-1, -1].transData.inverted().transform_bbox(axList[-1, -1].transAxes.transform_bbox(bbox))
axList[-1, -1].add_patch(mpl.patches.Rectangle(bbox.p0, bbox.width, bbox.height, linewidth=0, facecolor='w',
                                               alpha=0.8, zorder=3))
axList[-1, 4].text(0.8, 0.24, "Poor con-\nfinement", fontsize=8, bbox=dict(lw=0, facecolor='w', alpha=0.8))
axList[-1, 4].annotate("P = 0.8", (1.75, 0.33), color='k', horizontalalignment='right')
axList[-1, 4].annotate("0.9", (1.75, 0.46), color='k', horizontalalignment='right')
axList[-1, 4].annotate("0.7", (1.75, 0.24), color='k', horizontalalignment='right')
axList[-1, 4].annotate("GVM = 0", (1.2, 0.4), color='w', xytext=(1.25, 0.62),
                       arrowprops=dict(arrowstyle='->', color='w'))
axList[3, 4].annotate(r"GVD($\lambda$)" + "\n = 0", (1.2, 0.56), color='b')
axList[3, 4].annotate(r"GVD($2\lambda$)" + "\n = 0", (1.6, 0.50), xytext=(1.3, 0.33), color='r',
                      arrowprops=dict(arrowstyle='->', color='r'))
axList[3, 3].annotate(r"GVD($2\lambda$) = " + "\n" + " 2 GVD($\lambda$)", (1.3, 0.35), xytext=(1.0, 0.54), color='r',
                      arrowprops=dict(arrowstyle='->', color='r'))
axList[2, 0].plot([0.67], [0.5], '*', ms=10, mec='k', mfc='w')
axList[1, 1].plot([0.9], [0.6], '*', ms=10, mec='k', mfc='w')
axList[1, 2].plot([1.12], [0.55], '*', ms=10, mec='k', mfc='w')
axList[1, 3].plot([1.2], [0.5], '*', ms=10, mec='k', mfc='w')
axList[1, 4].plot([1.4], [0.6], '*', ms=10, mec='k', mfc='w')
axList[2, 5].plot([1.5], [0.6], '*', ms=10, mec='k', mfc='w')
f.savefig("./Fig5.pdf", format="pdf")

plt.show()
