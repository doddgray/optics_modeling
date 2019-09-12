
import numpy as np
import scipy as sp
from os import path
#u = rii.return_u()
from instrumental import u
import sys

from ..RII_db_tools import RII_db_tools as rii
data_dir = path.dirname(path.realpath(__file__))

#  Let's make a Frankenstein-style model for Si index and absortion (complex index)
#  as a function of wavelength, temperature and doping

# We'll start with wavelength dependence at 300K

# green08 data from
# Green, M.A., "Self-consistent optical parameters of intrinsic silicon at
# 300K including temperature coefficients."
# Solar Energy Materials and Solar Cells 92.11 (2008): 1305-1310.


green08_data = np.genfromtxt(path.join(data_dir,'green08_si_n_k_T.csv'),skip_header=2,delimiter=',')
green08_lm = green08_data[:,0] * 1000 * u.nm
green08_alpha = green08_data[:,1] / u.cm
green08_n = green08_data[:,2]
#green95_k = green95_data[:,3]  # this data stops at 1000nm but the alpha data continues to 1400, so convert from that
green08_k = ( green08_alpha * green08_lm / ( 4 * np.pi ) ).to(u.dimensionless)
green08_Cn = green08_data[:,4] * 1e-4 / u.degK
green08_Ck = green08_data[:,5] * 1e-4 / u.degK
green08_bn = ( green08_Cn * 300 * u.degK ).to(u.dimensionless).magnitude
green08_bk = ( green08_Ck * 300 * u.degK ).to(u.dimensionless).magnitude


# Import 'Li' T-dependent IR index dataset from RefractiveIndex.INFO

T_Li = np.array([100,150,200,250,293,350,400]).astype(int)
lm_Li = np.linspace(1.2,7,500)
n_T_lm_Li = np.empty((len(T_Li),len(lm_Li)))

for Tind, TT in enumerate(T_Li):
    n_model = rii.n_model('Si', 'Li-{}K.yml'.format(TT))
    n_T_lm_Li[Tind,:] = np.real(n_model(lm_Li*u.um))

# Import Macfarlane T-dependent alpha dataset captured from Ioffe image

T_mac = np.array([20,77,170,249,291,363,415]).astype(int)

data20 = np.genfromtxt(path.join(data_dir,'Macfarlane59_alpha_T_cryo/20K.csv'),delimiter=',')
hnu20 = data20[:,0] * u.eV
alpha20 = data20[:,1]**2 / hnu20.magnitude / u.cm
lm20 = ( u.speed_of_light / ( hnu20 / u.planck_constant ) ).to(u.um)
k20 = ( alpha20 * lm20 / ( 4 * np.pi ) ).to(u.dimensionless)

data77 = np.genfromtxt(path.join(data_dir,'Macfarlane59_alpha_T_cryo/77K.csv'),delimiter=',')
hnu77 = data77[:,0] * u.eV
alpha77 = data77[:,1]**2 / hnu77.magnitude / u.cm
lm77 = ( u.speed_of_light / ( hnu77 / u.planck_constant ) ).to(u.um)
k77 = ( alpha77 * lm77 / ( 4 * np.pi ) ).to(u.dimensionless)

data170 = np.genfromtxt(path.join(data_dir,'Macfarlane59_alpha_T_cryo/170K.csv'),delimiter=',')
hnu170 = data170[:,0] * u.eV
alpha170 = data170[:,1]**2 / hnu170.magnitude / u.cm
lm170 = ( u.speed_of_light / ( hnu170 / u.planck_constant ) ).to(u.um)
k170 = ( alpha170 * lm170 / ( 4 * np.pi ) ).to(u.dimensionless)

data249 = np.genfromtxt(path.join(data_dir,'Macfarlane59_alpha_T_cryo/249K.csv'),delimiter=',')
hnu249 = data249[:,0] * u.eV
alpha249 = data249[:,1]**2 / hnu249.magnitude / u.cm
lm249 = ( u.speed_of_light / ( hnu249 / u.planck_constant ) ).to(u.um)
k249 = ( alpha249 * lm249 / ( 4 * np.pi ) ).to(u.dimensionless)

data291 = np.genfromtxt(path.join(data_dir,'Macfarlane59_alpha_T_cryo/291K.csv'),delimiter=',')
hnu291 = data291[:,0] * u.eV
alpha291 = data291[:,1]**2 / hnu291.magnitude / u.cm
lm291 = ( u.speed_of_light / ( hnu291 / u.planck_constant ) ).to(u.um)
k291 = ( alpha291 * lm291 / ( 4 * np.pi ) ).to(u.dimensionless)

data363 = np.genfromtxt(path.join(data_dir,'Macfarlane59_alpha_T_cryo/363K.csv'),delimiter=',')
hnu363 = data363[:,0] * u.eV
alpha363 = data363[:,1]**2 / hnu363.magnitude / u.cm
lm363 = ( u.speed_of_light / ( hnu363 / u.planck_constant ) ).to(u.um)
k363 = ( alpha363 * lm363 / ( 4 * np.pi ) ).to(u.dimensionless)

data415 = np.genfromtxt(path.join(data_dir,'Macfarlane59_alpha_T_cryo/415K.csv'),delimiter=',')
hnu415 = data415[:,0] * u.eV
alpha415 = data415[:,1]**2 / hnu415.magnitude / u.cm
lm415 = ( u.speed_of_light / ( hnu415 / u.planck_constant ) ).to(u.um)
k415 = ( alpha415 * lm415 / ( 4 * np.pi ) ).to(u.dimensionless)

k20_itp = sp.interpolate.interp1d(lm20.magnitude,k20.magnitude,bounds_error=False,fill_value=1e-11)
k77_itp = sp.interpolate.interp1d(lm77.magnitude,k77.magnitude,bounds_error=False,fill_value=1e-11)
k170_itp = sp.interpolate.interp1d(lm170.magnitude,k170.magnitude,bounds_error=False,fill_value=1e-11)
k249_itp = sp.interpolate.interp1d(lm249.magnitude,k249.magnitude,bounds_error=False,fill_value=1e-11)


##### Gross, hacky stitching about to happen! avert your eyes!

green_n_itp_300K = sp.interpolate.interp1d(green08_lm.to(u.um).magnitude, green08_n)
green_cn_itp_300K = sp.interpolate.interp1d(green08_lm.to(u.um).magnitude, green08_Cn.magnitude)
green_k_itp_300K = sp.interpolate.interp1d(green08_lm.to(u.um).magnitude, green08_k,bounds_error=False,fill_value=1e-11)
green_bk_itp_300K = sp.interpolate.interp1d(green08_lm.to(u.um).magnitude, green08_bk,bounds_error=False,fill_value=1)
def green_n_itp(lm,T):
    return green_n_itp_300K(lm) * ( 1 + green_cn_itp_300K(lm) * ( T - 300 ) )

def green_alpha_itp(lm,T):
    return sp.interpolate.interp1d(green08_lm.to(u.um).magnitude, green08_alpha) * ( T / 300.0 )**green_bk_itp_300K(lm)

def green_k_itp(lm,T):
    return  green_k_itp_300K(lm) * ( T / 300.0 )**green_bk_itp_300K(lm)

Li_n_itp = sp.interpolate.interp2d(lm_Li, T_Li, n_T_lm_Li)

T = 100
# print('n_green: {}'.format(green_n_itp(1.2,T)))
# print('n_Li: {}'.format(Li_n_itp(1.2,T)[0]))

r_200 = Li_n_itp(1.2,200)[0] / green_n_itp(1.2,200)
r_150 = Li_n_itp(1.2,150)[0] / green_n_itp(1.2,150)
r_100 = Li_n_itp(1.2,100)[0] / green_n_itp(1.2,100)

r_itp = sp.interpolate.interp1d(np.array([100,150,200]),np.array([r_100,r_150,r_200]),bounds_error=False,fill_value=r_100)



def r_fu_python(T):
    if T >= 250:
        return 1
    if T < 250:
        return float(r_itp(T))

def n_r_si_lm_T(lm,T):
    if lm <= 1.2:
        if T >= 100:
            return r_fu_python(T) * green_n_itp(lm,T)
        if T < 100:
            return r_fu_python(20) * green_n_itp(lm,100)
    if lm > 1.2:
        return Li_n_itp(lm,T)[0]


p_20 = k20_itp(lm20.magnitude.min()) / green_k_itp(lm20.magnitude.min(),20)
p_77 = k77_itp(lm77.magnitude.min()) / green_k_itp(lm77.magnitude.min(),77)
p_170 = k170_itp(lm170.magnitude.min()) / green_k_itp(lm170.magnitude.min(),170)
p_249 = k249_itp(lm249.magnitude.min()) / green_k_itp(lm249.magnitude.min(),249)

p_itp = sp.interpolate.interp1d(np.array([20,77,170,249]),np.array([p_20,p_77,p_170,p_249]))

def k20_itp_ext(lm):
    if lm >= lm20.magnitude.min():
        return k20_itp(lm)
    if lm < lm20.magnitude.min():
        return p_20 * green_k_itp(lm,20)

def k77_itp_ext(lm):
    if lm >= lm77.magnitude.min():
        return k77_itp(lm)
    if lm < lm77.magnitude.min():
        return p_77 * green_k_itp(lm,77)

def k170_itp_ext(lm):
    if lm >= lm170.magnitude.min():
        return k170_itp(lm)
    if lm < lm170.magnitude.min():
        return p_170 * green_k_itp(lm,170)

def k249_itp_ext(lm):
    if lm >= lm249.magnitude.min():
        return k249_itp(lm)
    if lm < lm249.magnitude.min():
        return p_249 * green_k_itp(lm,249)

lm_itp = np.linspace(0.5,1.2)
logk_itp = np.empty((4,len(lm_itp)))
for lind, ll in enumerate(lm_itp):
    logk_itp[0,lind] = np.log10(k20_itp_ext(ll))
    logk_itp[1,lind] = np.log10(k77_itp_ext(ll))
    logk_itp[2,lind] = np.log10(k170_itp_ext(ll))
    logk_itp[3,lind] = np.log10(k249_itp_ext(ll))

logk_itp2d = sp.interpolate.interp2d(lm_itp,np.array([20,77,170,249]),logk_itp)



def n_i_si_lm_T(lm,T):
    if T >= 250:
        return green_k_itp(lm,T)
    if T < 250:
        return 10**logk_itp2d(lm,T)[0]


#### Plasma Dispersion model for doped Si
## model taken from
## Nedeljkovic, Soref & Mashanovich,
## Free-Carrier Electrorefraction and Electroabsorption Modulation Predictions for Silicon Over the 1-14um Infrared Wavelength Range
## IEEE Photonics Journal, Volume 3, Number 6, December 2011

lm = np.array([1.3, 1.55, 2.0, 2.5, 3.0, 3.5, 4.0])

# alpha model parameters
a = np.array([3.48e-22, 8.88e-21, 3.22e-20,1.67e-20,6.29e-21,3.10e-21, 7.45e-22])
b = np.array([1.229,1.167,1.149,1.169,1.193,1.210,1.245])
c = np.array([1.02e-19, 5.84e-20, 6.21e-20, 8.08e-20, 3.40e-20,6.05e-20,5.43e-20])
d = np.array([1.089,1.109, 1.119,1.123,1.151,1.145,1.153])

#a_int = interp1d(lm,a) #this value varies wildly, maybe intpolate loga?
loga_int = sp.interpolate.interp1d(lm,np.log(a),bounds_error=False,fill_value=np.log(a[0]))
def a_int(lm): return np.exp(loga_int(lm))
b_int = sp.interpolate.interp1d(lm,b,bounds_error=False,fill_value=b[0])
logc_int = sp.interpolate.interp1d(lm,np.log(c),bounds_error=False,fill_value=np.log(c[0]))
def c_int(lm): return np.exp(logc_int(lm))
d_int = sp.interpolate.interp1d(lm,d,bounds_error=False,fill_value=d[0])

# dn model parameters
p = np.array([2.98e-22, 5.40e-22,1.91e-21,5.70e-21,6.57e-21,6.95e-21,7.25e-21])
q = np.array([1.016, 1.011, 0.992, 0.976, 0.981, 0.986, 0.991])
r = np.array([1.25e-18, 1.53e-18, 2.28e-18, 5.19e-18, 3.62e-18, 9.28e-18, 9.99e-18])
s = np.array([0.835, 0.838, 0.841, 0.832, 0.849, 0.834, 0.839])
logp_int = sp.interpolate.interp1d(lm,np.log(p),bounds_error=False,fill_value=np.log(p[0]))
def p_int(lm): return np.exp(logp_int(lm))
q_int = sp.interpolate.interp1d(lm,q,bounds_error=False,fill_value=q[0])
logr_int = sp.interpolate.interp1d(lm,np.log(r),bounds_error=False,fill_value=np.log(r[0]))
def r_int(lm): return np.exp(logr_int(lm))
s_int = sp.interpolate.interp1d(lm,s,bounds_error=False,fill_value=s[0])

# complex index change
def si_plasma_alpha(lm,Ne,Nh):
    return a_int(lm) * Ne**b_int(lm) + c_int(lm) * Nh**d_int(lm)

def si_plasma_dn(lm,Ne,Nh):
    return p_int(lm) * Ne**q_int(lm) + r_int(lm) * Nh**s_int(lm)

def si_plasma_refraction(lm,Ne,Nh):
    dni = si_plasma_alpha(lm,Ne,Nh) * ( lm * 1e-4 ) / ( 4 * np.pi )
    dnr = si_plasma_dn(lm,Ne,Nh)
    return dnr - 1j * dni

def si_complex_index_model(λ=1.55*u.um, T=300*u.degK, Ne=0./u.cm**3, Nh=0./u.cm**3):
    λ_um = λ.to(u.um).m
    T_K = T.to(u.degK).m
    Ne_cm3 = Ne.to(1/u.cm**3).m
    Nh_cm3 = Nh.to(1/u.cm**3).m
    if type(λ_um)!=np.ndarray and type(T_K)!=np.ndarray and type(Ne_cm3)!=np.ndarray and type(Nh_cm3)!=np.ndarray:
        nr = n_r_si_lm_T(λ_um,T_K)
        ni = n_i_si_lm_T(λ_um,T_K)
        return nr - 1j*ni + si_plasma_refraction(λ_um,Ne_cm3,Nh_cm3)
    else:
        dims = [λ_um,T_K,Ne_cm3,Nh_cm3]
        for dind, dim in enumerate(dims):
            if type(dim)!=np.ndarray:
                dims[dind] = np.array([dim,])
        λ_um,T_K,Ne_cm3,Nh_cm3 = tuple(dims)
        output = np.empty((len(λ_um),len(T_K),len(Ne_cm3),len(Nh_cm3)),dtype=np.complex128)
        for lind,ll in enumerate(λ_um):
            for Tind,TT in enumerate(T_K):
                for Neind,NNe in enumerate(Ne_cm3):
                    for Nhind,NNh in enumerate(Nh_cm3):
                        nr = n_r_si_lm_T(ll,TT)
                        ni = n_i_si_lm_T(ll,TT)
                        output[lind,Tind,Neind,Nhind] = nr - 1j*ni + si_plasma_refraction(ll,NNe,NNh)
        return output.squeeze()



def si_real_index_model(λ=1.55*u.um, T=300*u.degK, Ne=0./u.cm**3, Nh=0./u.cm**3):
    λ_um = λ.to(u.um).m
    T_K = T.to(u.degK).m
    Ne_cm3 = Ne.to(1/u.cm**3).m
    Nh_cm3 = Nh.to(1/u.cm**3).m
    if type(λ_um)!=np.ndarray and type(T_K)!=np.ndarray and type(Ne_cm3)!=np.ndarray and type(Nh_cm3)!=np.ndarray:
        nr = n_r_si_lm_T(λ_um,T_K)
        return nr + np.real(si_plasma_refraction(λ_um,Ne_cm3,Nh_cm3))
    else:
        dims = [λ_um,T_K,Ne_cm3,Nh_cm3]
        for dind, dim in enumerate(dims):
            if type(dim)!=np.ndarray:
                dims[dind] = np.array([dim,])
        λ_um,T_K,Ne_cm3,Nh_cm3 = tuple(dims)
        output = np.empty((len(λ_um),len(T_K),len(Ne_cm3),len(Nh_cm3)),dtype=np.float64)
        for lind,ll in enumerate(λ_um):
            for Tind,TT in enumerate(T_K):
                for Neind,NNe in enumerate(Ne_cm3):
                    for Nhind,NNh in enumerate(Nh_cm3):
                        nr = n_r_si_lm_T(ll,TT)
                        output[lind,Tind,Neind,Nhind] = nr + np.real(si_plasma_refraction(ll,NNe,NNh))
        return output.squeeze()
