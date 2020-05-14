# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 09:04:14 2014

@author: dodd
"""
import time
import NLO_tools
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


#################################################################
# Reproduction of Parametric Gain calculation (Fig 12) from
# 'Parametric Interaction of Focused Gaussian Light Beams'
# Boyd and Kleinman, JAP, 39, 3597-3639 (1968)
# DOI:http://dx.doi.org/10.1063/1.1656831
#################################################################
start = time.time()
xi = np.logspace(-2,3.2,200,10)
sigma = np.linspace(0,1,200)
B = np.array([0,0.5,1,2,4])
hb = np.zeros((xi.shape[0],sigma.shape[0],B.shape[0]))

for ib, b in enumerate(B):
    for isig, sig in enumerate(sigma): 
        for ix, x in enumerate(xi):  
            print('b is {}, sig is {}, x is {}'.format(b,sig,x))            
            hb_cur = NLO_tools.hbar_BoydKleinman(x,sig,b)
            hb[ix-1,isig-1,ib-1] = hb_cur
            
hbm = np.amax(hb,axis=1)
i_sig_max = np.argmax(hb,axis=1)
i_xi_max = np.argmax(hbm,axis=0)
sigma_max = sigma[i_sig_max[i_xi_max,np.array(np.arange(B.shape[0]))]]

fig, ax = plt.subplots()
for i in np.arange(B.shape[0]):
    ax.loglog(xi,hbm[:,i],label = r"$B = {},  \sigma_m = {:01.3g}$".format(B[i], sigma_max[i]))
ax.legend()
ax.set_xlim((.01,1000))
ax.set_ylim((0.9*min(np.amin(hbm,axis=0)),1.1*max(np.amax(hbm,axis=0))))
ax.set_xlabel(r'Focusing Parameter $\xi$')
ax.set_ylabel(r'$\hbar(B,\xi)$')
ax.set_title(r'Parametric Gain $\hbar$ vs. Focusing Parameter $\xi$ for Various Walkoff Parameter $B$ Values')
plt.show()
end = time.time()
print('time elapsed: {}s'.format(end-start))


test = np.vectorize(NLO_tools.hbar_BoydKleinman)
start = time.time()
hb_test = test(xi,sigma,B)
hbm_test = np.amax(hb_test,axis=1)
end= time.time()
print('time elapsed: {}s'.format(end-start))

#################################################################
# Reproduction of Parametric Gain calculation (Fig 12) from
# 'Parametric Interaction of Focused Gaussian Light Beams'
# Boyd and Kleinman, JAP, 39, 3597-3639 (1968)
# DOI:http://dx.doi.org/10.1063/1.1656831
#################################################################

