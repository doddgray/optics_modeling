# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 16:23:49 2014

@author: dodd
"""

import numpy as np
import matplotlib.pyplot as plt
import RII_db_tools as RII
from instrumental import Q_

n_caf2 = RII.n_model('CaF2','Malitson.yml')
n_YAG = RII.n_model('Y3Al5O12','Zelmon.yml')
n_sap = RII.n_model('Al2O3','Malitson.yml')

gvd_caf2 = RII.gvd_model('CaF2','Malitson.yml')
gvd_YAG = RII.gvd_model('Y3Al5O12','Zelmon.yml')
gvd_sap = RII.gvd_model('Al2O3','Malitson.yml')

lm = Q_(np.linspace(1,2,100),'um')

plt.figure(1)
plt.subplot(211)
plt.plot(lm,n_caf2(lm),'r--',lm,n_sap(lm),'g--',lm,n_YAG(lm),'k-')
plt.xlabel('Wavelength [um]')
plt.ylabel('Refractive Index [1]')

plt.subplot(212)
plt.plot(lm,gvd_caf2(lm),'r--',lm,gvd_sap(lm),'g--',lm,gvd_YAG(lm),'k-')
plt.xlabel('Wavelength [um]')
plt.ylabel('GVD [fs^2 / mm]')
plt.show()
