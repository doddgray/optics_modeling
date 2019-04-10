# -*- coding: utf-8 -*-
"""
OPO Gain Figures

Created on Thu Nov  6 16:17:39 2014

@author: dodd
"""
import matplotlib
import matplotlib.pyplot as plt
import NLO_tools as NLO
import numpy as np
from instrumental import u, Q_
import time, datetime, os
save_dir = '/Users/dodd/Dropbox/MabuchiLab/software/Python/OPO_Gain_Calcs'





#LM = np.arange(27.5,32,0.5)*u.um
#T = np.linspace(25,200,200)*u.degC
#lm_s_pm, lm_i_pm = NLO.QPM_PPLN(1.064*u.um, LM, T)

#matplotlib.rcParams.update({'font.size': 40})
#plt.style.use('ggplot')
#fig, ax = plt.subplots(1,1)
#
#for LM_idx, LM_curr in enumerate(LM):
#    ax.plot(T,lm_i_pm[:,LM_idx],label=r"$\Lambda$={}$\mu$m".format(LM_curr.to(u.um).magnitude))
#    
#ax.set_xlabel(r'PPLN Temperature [$\circ$C]')
#ax.set_ylabel(r'Phase Matched Idler Wavelength [$\mu$m]')
##ax.set_title('Phase Matched Idler Wavelength vs. PPLN Temperature')
#
## Shrink current axis by 20%
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
## Put a legend to the right of the current axis
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#
#plt.savefig(os.path.join(save_dir,'Fig1.png'))
##plt.show()
#
#ax.plot(T_exp1,lm_exp1,'ro')
#ax.plot(T_exp2,lm_exp2,'ro')
#ax.plot(T_exp3,lm_exp3,'ro')
#
#plt.savefig(os.path.join(save_dir,'Fig3.png'))

### Calculate phase matching conditions for 1.05um pump

##lm_s_pm2, lm_i_pm2 = NLO.QPM_PPLN(1.05*u.um, LM, T)
#
## Reset color sequence and plot on same axes with dashed lines
#ax.set_color_cycle(None)
#for LM_idx, LM_curr in enumerate(LM):
#    ax.plot(T,lm_i_pm2[:,LM_idx],'--',label=r"$\Lambda$={}$\mu$m".format(LM_curr.to(u.um).magnitude))
#plt.savefig(os.path.join(save_dir,'Fig2.png'))
#
#
#### GaAs phase matching calculation and plotting
#
#### Single Poling period vs. temperature
#
#LM_gaas1 = np.array([86]).flatten()*u.um
#T_gaas1 = np.linspace(25,200,200)*u.degC
#lm_s_pm_gaas1, lm_i_pm_gaas1 = NLO.QPM_PPLN(2.1*u.um, LM_gaas1, T_gaas1)
#
#fig, ax = plt.subplots(1,1)
#
#for LM_idx, LM_curr in enumerate(LM_gaas1):
#    ax.plot(T,lm_i_pm_gaas1[:,LM_idx],label=r"$\Lambda$={}$\mu$m".format(LM_curr.to(u.um).magnitude))
#    
#ax.set_xlabel(r'OP-GaAs Temperature [$\circ$C]')
#ax.set_ylabel(r'Phase Matched Idler Wavelength [$\mu$m]')
##ax.set_title('Phase Matched Idler Wavelength vs. OP-GaAs Temperature')
#
#
#### Continuously Varied Poling Regions
#
#LM_gaas2 = np.linspace(62,86,200)*u.um
#T_gaas2 = 25*u.degC
#lm_s_pm_gaas2, lm_i_pm_gaas2 = NLO.QPM_PPLN(2.1*u.um, LM_gaas2, T_gaas2)
#
#fig, ax = plt.subplots(1,1)
#
#for LM_idx, LM_curr in enumerate(LM_gaas2):
#    ax.plot(LM_curr,lm_i_pm_gaas2[:,LM_idx])
#    
#ax.set_xlabel(r'OP-GaAs Poling Period [$\mu$m]')
#ax.set_ylabel(r'Phase Matched Idler Wavelength [$\mu$m]')


#### Gain Calculations

n_lm = 2e3
n_T = 2e2
lm_p = 1.064*u.um
w_s = 66*u.um
l_c = 5*u.cm
#LM = np.arange(27,32,0.5)*u.um
#lm_i_max = np.array([4.7,4.5,4.3,4.18,4,3.8,3.65,3.45,3.21,2.9])
#lm_i_min = np.array([4.4,4.25,4,3.78,3.6,3.49,3.2,2.85,2.55,2.128])
#lm_s_min = 1/(1/(1.064*u.um) - 1/lm_i_max)
#lm_s_max = 1/(1/(1.064*u.um) - 1/lm_i_min)

LM = np.arange(27.5,32,0.5)*u.um
T = np.linspace(25,200,n_T)*u.degC
lm_s = Q_(np.linspace(lm_p.to(u.um).magnitude*1.3,2*lm_p.to(u.um).magnitude,n_lm),'um')

lm_i = 1/(1/lm_p - 1/lm_s)
gain = np.zeros((n_T,n_lm))
start = time.time()
for LM_idx, LM_curr in enumerate(LM):
    print('LM is {}'.format(LM_curr))
    #lm_s = np.linspace(lm_s_min[LM_idx],lm_s_max[LM_idx],n_lm)*u.um    
    gain += NLO.ParametricGain_BK_PPLN(lm_p,lm_s,T,w_s,l_c,LM_curr)
    print('time elapsed is {} min'.format((time.time()-start)/60))
calc_dir = os.path.join(save_dir,time.strftime('%Y-%m-%d_%H-%M-%S'))
os.mkdir(calc_dir)
np.savetxt(os.path.join(calc_dir,'T.csv'),T,delimiter=',')
np.savetxt(os.path.join(calc_dir,'lm_s.csv'),lm_s,delimiter=',')
np.savetxt(os.path.join(calc_dir,'lm_i.csv'),lm_i,delimiter=',')
np.savetxt(os.path.join(calc_dir,'gain.csv'),gain,delimiter=',')
fig, ax = plt.subplots()
p = ax.pcolor(T, lm_i, gain.T, cmap=matplotlib.cm.gist_heat, vmin=abs(gain).min(), vmax=abs(gain).max())
cb = fig.colorbar(p, ax=ax)
plt.savefig(os.path.join(calc_dir,'gain.png'))



