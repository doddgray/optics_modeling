# -*- coding: utf-8 -*-
"""
NLO_tools.py

A toolbox of functions written for nonlinear optics calculations

Created on Fri Oct 31 11:13:16 2014 (Ooooo spoooooky)

@author: dodd
"""
import sympy as sp
import numpy as np
from instrumental import Q_, u

##########################################################################
###### Temperature Dependent Index, Group Index and GVD models
###### for phase-matching calculations
##########################################################################
def n_MgO_LN_sym(axis='e'):
    """This function creates a symbolic representation (using SymPy) of the
    Sellmeier Equation model for the temperature and wavelength dependence
    of 5% MgO:LiNbO3's (congruent LiNbO3 (CLN) not stoichiometric LiNbO3 (SLN))
    ordinary and extraordinary indices of refraction.
    Equation form is based on "Temperature and
    wavelength dependent refractive index equations for MgO-doped congruent
    and stoichiometric LiNbO3" by Gayer et al., Applied Physics B 91,
    343?348 (2008)

    This model is then exported to other functions that use it and its
    derivatives to return index, group index and GVD values as a function
    of temperature and wavelength.

    Variable units are lm in [um] and T in [deg C]

    """
    if axis is 'e':
        a1 = 5.756
        a2 = 0.0983
        a3 = 0.202
        a4 = 189.32
        a5 = 12.52
        a6 = 1.32e-2
        b1 = 2.86e-6
        b2 = 4.7e-8
        b3 = 6.113e-8
        b4 = 1.516e-4
    elif axis is 'o':
        a1 = 5.653
        a2 = 0.1185
        a3 = 0.2091
        a4 = 89.61
        a5 = 10.85
        a6 = 1.97e-2
        b1 = 7.941e-7
        b2 = 3.134e-8
        b3 = -4.641e-9
        b4 = -2.188e-6
    else:
        raise Exception('unrecognized axis! must be "e" or "o"')
    lm, T = sp.symbols('lm T')
    T0 = 24.5 # reference temperature in [Deg C]
    f = (T - T0) * (T + T0 + 2*273.16) # so-called 'temperature dependent parameter'
    n_sym = sp.sqrt(    a1 + b1*f + (a2 + b2*f) / (lm**2 - (a3 + b3*f)**2) \
                        + (a4 + b4*f) / (lm**2 - a5**2) - a6*lm**2)
    return lm, T, n_sym

def n_MgO_LN(lm_in,T_in,axis='e'):
    """Sellmeier Equation model for the temperature and wavelength dependence
    of the index of refraction of 5% MgO:LiNbO3. Equation form is based on
    "Temperature and wavelength dependent refractive index equations
    for MgO-doped congruent and stoichiometric LiNbO3"
    by Gayer et al., Applied Physics B 91, p.343-348 (2008)

    Variable units are lm in [um] and T in [deg C]

    """
    lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
    T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
    lm, T, n_sym = n_MgO_LN_sym(axis=axis)
    n = sp.lambdify([lm,T],n_sym,'numpy')
    output = np.zeros((T_C.size,lm_um.size))
    for T_idx, TT in enumerate(T_C):
        output[T_idx,:] = n(lm_um, T_C[T_idx])
    return output

def n_g_MgO_LN(lm_in,T_in,axis='e'):
    """Sellmeier Equation model for the temperature and wavelength dependence
    of the group index of refraction of 5% MgO:LiNbO3. Equation form is based on
    "Temperature and wavelength dependent refractive index equations
    for MgO-doped congruent and stoichiometric LiNbO3"
    by Gayer et al., Applied Physics B 91, p.343-348 (2008)

    Variable units are lm in [um] and T in [deg C]

    """
    lm_um = Q_(lm_in).to(u.um).magnitude
    T_C = Q_(T_in).to(u.degC).magnitude
    lm, T, n_sym = n_MgO_LN_sym(axis=axis)
    n_sym_prime = sp.diff(n_sym,lm)
    n_g_sym = n_sym - lm*n_sym_prime
    n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
    return n_g(lm_um, T_C)


def gvd_MgO_LN(lm_in,T_in,axis='e'):
    """Sellmeier Equation model for the temperature and wavelength dependence
    of the group velocity dispersion of 5% MgO:LiNbO3. Equation form is based on
    "Temperature and wavelength dependent refractive index equations
    for MgO-doped congruent and stoichiometric LiNbO3"
    by Gayer et al., Applied Physics B 91, p.343-348 (2008)

    Variable units are lm in [um] and T in [deg C]

    """
    lm_um = Q_(lm_in).to(u.um).magnitude
    T_C = Q_(T_in).to(u.degC).magnitude
    lm, T, n_sym = n_MgO_LN_sym(axis=axis)
    n_sym_double_prime = sp.diff(n_sym,lm,lm)
    c = Q_(3e8,'m/s') # unitful definition of speed of light
    gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
    gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
    gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
    return gvd.to('fs**2 / mm')


def n_GaAs_sym():
    """This function creates a symbolic representation (using SymPy) of the
    a model for the temperature and wavelength dependence of GaAs's refractive
    index. The equation form and fit parameters are based on "Improved dispersion
    relations for GaAs and applications to nonlinear optics" by Skauli et al.,
    JAP 94, 6447 (2003); doi: 10.1063/1.1621740

    This model is then exported to other functions that use it and its
    derivatives to return index, group index and GVD values as a function
    of temperature and wavelength.

    Variable units are lm in [um] and T in [deg C]

    """

    lm, T = sp.symbols('lm T')
    T0 = 22 # reference temperature in [Deg C]
    deltaT = T-T0
    A = 0.689578
    eps2 = 12.99386
    G3 = 2.18176e-3
    E0 = 1.425 - 3.7164e-4 * deltaT - 7.497e-7* deltaT**2
    E1 = 2.400356 - 5.1458e-4 * deltaT
    E2 = 7.691979 - 4.6545e-4 * deltaT
    E3 = 3.4303e-2 + 1.136e-5 * deltaT
    E_phot = (u.h * u.c).to(u.eV*u.um).magnitude / lm  #
    n_sym = sp.sqrt( 1  + (A/np.pi)*sp.log((E1**2 - E_phot**2) / (E0**2-E_phot**2)) \
                        + (eps2/np.pi)*sp.log((E2**2 - E_phot**2) / (E1**2-E_phot**2)) \
                        + G3 / (E3**2-E_phot**2) )

    return lm, T, n_sym

###### Temperature Dependent Sellmeier Equation for phase-matching calculations
def n_GaAs(lm_in,T_in):
    """Sellmeier Equation model for the temperature and wavelength dependence
    of the index of refraction of 5% MgO:LiNbO3. Equation form is based on
    "Temperature and wavelength dependent refractive index equations
    for MgO-doped congruent and stoichiometric LiNbO3"
    by Gayer et al., Applied Physics B 91, p.343-348 (2008)

    Variable units are lm in [um] and T in [deg C]

    """
#    lm_um = Q_(lm_in).to(u.um).magnitude
#    T_C = Q_(T_in).to(u.degC).magnitude
#    lm, T, n_sym = n_GaAs_sym()
#    n = sp.lambdify([lm,T],n_sym,'numpy')
#    return n(lm_um, T_C)

    lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
    T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
    lm, T, n_sym = n_GaAs_sym()
    n = sp.lambdify([lm,T],n_sym,'numpy')
    output = np.zeros((T_C.size,lm_um.size))
    for T_idx, TT in enumerate(T_C):
        output[T_idx,:] = n(lm_um, T_C[T_idx])
    return output


def n_g_GaAs(lm_in,T_in):
    """Sellmeier Equation model for the temperature and wavelength dependence
    of the group index of refraction of 5% MgO:LiNbO3. Equation form is based on
    "Temperature and wavelength dependent refractive index equations
    for MgO-doped congruent and stoichiometric LiNbO3"
    by Gayer et al., Applied Physics B 91, p.343-348 (2008)

    Variable units are lm in [um] and T in [deg C]

    """
    lm_um = Q_(lm_in).to(u.um).magnitude
    T_C = Q_(T_in).to(u.degC).magnitude
    lm, T, n_sym = n_GaAs_sym()
    n_sym_prime = sp.diff(n_sym,lm)
    n_g_sym = n_sym - lm*n_sym_prime
    n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
    return n_g(lm_um, T_C)


def gvd_GaAs(lm_in,T_in):
    """Sellmeier Equation model for the temperature and wavelength dependence
    of the group velocity dispersion of 5% MgO:LiNbO3. Equation form is based on
    "Temperature and wavelength dependent refractive index equations
    for MgO-doped congruent and stoichiometric LiNbO3"
    by Gayer et al., Applied Physics B 91, p.343-348 (2008)

    Variable units are lm in [um] and T in [deg C]

    """
    lm_um = Q_(lm_in).to(u.um).magnitude
    T_C = Q_(T_in).to(u.degC).magnitude
    lm, T, n_sym = n_GaAs_sym()
    n_sym_double_prime = sp.diff(n_sym,lm,lm)
    c = Q_(3e8,'m/s') # unitful definition of speed of light
    gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
    gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
    gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
    return gvd.to('fs**2 / mm')


########### Si3N4 Sellmeier model for waveguide phase matching calculations

def n_Si3N4_sym():
    """This function creates a symbolic representation (using SymPy) of the
    Sellmeier Equation model for the index of refraction.
    Equation form is based on Luke, Okawachi, Lamont, Gaeta and Lipson,
    "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator"
    Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)
    https://doi.org/10.1364/OL.40.004823
    valid from 0.31–5.504 um

    Thermo-optic coefficients from
    Xue, et al.
    "Thermal tuning of Kerr frequency combs in silicon nitride microring resonators"
    Opt. Express 24.1 (2016) http://doi.org/10.1364/OE.24.000687

    This model is then exported to other functions that use it and its
    derivatives to return index, group index and GVD values as a function
    of temperature and wavelength.

    Variable units are lm in [um] and T in [deg C]

    """
    A0 = 1
    B1 = 3.0249
    C1 = (0.1353406)**2 # um^2
    B2 = 40314
    C2 = (1239.842)**2 # um^2
    dn_dT = 2.96e-5 # 1/degK
    lm, T = sp.symbols('lm T')
    T0 = 24.5 # reference temperature in [Deg C]
    n_sym = sp.sqrt(   A0  + ( B1 * lm**2 ) / ( lm**2 - C1 ) + ( B2 * lm**2 ) / ( lm**2 - C2 ) ) + dn_dT * ( T - T0 )
    return lm, T, n_sym

def n_Si3N4(lm_in,T_in):
    """Sellmeier Equation model for the temperature and wavelength dependence
    of the index of refraction of Si3N4. Equation form is based on
    "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator"
    by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)

    Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]

    """
    lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
    T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
    lm, T, n_sym = n_Si3N4_sym()
    n = sp.lambdify([lm,T],n_sym,'numpy')
    output = np.zeros((T_C.size,lm_um.size))
    for T_idx, TT in enumerate(T_C):
        output[T_idx,:] = n(lm_um, T_C[T_idx])
    return output

def n_g_Si3N4(lm_in,T_in):
    """Sellmeier Equation model for the temperature and wavelength dependence
    of the group index of refraction of Si3N4. Equation form is based on
    "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator"
    by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)

    Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]

    """
    lm_um = Q_(lm_in).to(u.um).magnitude
    T_C = Q_(T_in).to(u.degC).magnitude
    lm, T, n_sym = n_Si3N4_sym()
    n_sym_prime = sp.diff(n_sym,lm)
    n_g_sym = n_sym - lm*n_sym_prime
    n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
    return n_g(lm_um, T_C)


def gvd_Si3N4(lm_in,T_in):
    """Sellmeier Equation model for the temperature and wavelength dependence
    of the group velocity dispersion of Si3N4. Equation form is based on
    "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator"
    by Luke et al., Optics Letters Vol. 40, Issue 21, pp. 4823-4826 (2015)

    Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]

    """
    lm_um = Q_(lm_in).to(u.um).magnitude
    T_C = Q_(T_in).to(u.degC).magnitude
    lm, T, n_sym = n_Si3N4_sym()
    n_sym_double_prime = sp.diff(n_sym,lm,lm)
    c = Q_(3e8,'m/s') # unitful definition of speed of light
    gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
    gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
    gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
    return gvd.to('fs**2 / mm')


########### SiO2 Sellmeier model for waveguide phase matching calculations

def n_SiO2_sym():
    """This function creates a symbolic representation (using SymPy) of the
    Sellmeier Equation model for the index of refraction.
    Equation form is based on Kitamura, et al.
    "Optical constants of silica glass from extreme ultraviolet to far
    infrared at near room temperature." Applied optics 46.33 (2007): 8118-8133.
    which references Malitson, “Interspecimen comparison of the refractive
    index of fused silica,” J. Opt. Soc. Am.55,1205–1209 (1965)
    and has been validated from 0.21-6.7 μm (free space wavelength)

    Thermo-optic coefficients from the literature, forgot source.

    This model is then exported to other functions that use it and its
    derivatives to return index, group index and GVD values as a function
    of temperature and wavelength.

    Variable units are lm in [um] and T in [deg C]

    """
    A0 = 1
    B1 = 0.6961663
    C1 = (0.0684043)**2 # um^2
    B2 = 0.4079426
    C2 = (0.1162414)**2 # um^2
    B3 = 0.8974794
    C3 = (9.896161)**2 # um^2
    dn_dT = 6.1e-6 # 1/degK
    lm, T = sp.symbols('lm T')
    T0 = 20 # reference temperature in [Deg C]
    n_sym = sp.sqrt(   A0  + ( B1 * lm**2 ) / ( lm**2 - C1 ) + ( B2 * lm**2 ) / ( lm**2 - C2 ) + ( B3 * lm**2 ) / ( lm**2 - C3 ) ) + dn_dT * ( T - T0 )
    return lm, T, n_sym

def n_SiO2(lm_in,T_in):
    """Sellmeier Equation model for the temperature and wavelength dependence
    of the index of refraction of SiO2. Equation form is based on
    Kitamura, et al.
    "Optical constants of silica glass from extreme ultraviolet to far
    infrared at near room temperature." Applied optics 46.33 (2007): 8118-8133.

    Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]

    """
    lm_um = np.array([Q_(lm_in).to(u.um).magnitude]).flatten()
    T_C = np.array([Q_(T_in).to(u.degC).magnitude]).flatten()
    lm, T, n_sym = n_SiO2_sym()
    n = sp.lambdify([lm,T],n_sym,'numpy')
    output = np.zeros((T_C.size,lm_um.size))
    for T_idx, TT in enumerate(T_C):
        output[T_idx,:] = n(lm_um, T_C[T_idx])
    return output

def n_g_SiO2(lm_in,T_in):
    """Sellmeier Equation model for the temperature and wavelength dependence
    of the group index of refraction of SiO2. Equation form is based on
    Kitamura, et al.
    "Optical constants of silica glass from extreme ultraviolet to far
    infrared at near room temperature." Applied optics 46.33 (2007): 8118-8133.

    Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]

    """
    lm_um = Q_(lm_in).to(u.um).magnitude
    T_C = Q_(T_in).to(u.degC).magnitude
    lm, T, n_sym = n_SiO2_sym()
    n_sym_prime = sp.diff(n_sym,lm)
    n_g_sym = n_sym - lm*n_sym_prime
    n_g = sp.lambdify([lm,T],n_g_sym,'numpy')
    return n_g(lm_um, T_C)


def gvd_SiO2(lm_in,T_in):
    """Sellmeier Equation model for the temperature and wavelength dependence
    of the group velocity dispersion of SiO2. Equation form is based on
    Kitamura, et al.
    "Optical constants of silica glass from extreme ultraviolet to far
    infrared at near room temperature." Applied optics 46.33 (2007): 8118-8133.

    Variable [units] passed to symbolic equation are lm in [um] and T in [deg C]

    """
    lm_um = Q_(lm_in).to(u.um).magnitude
    T_C = Q_(T_in).to(u.degC).magnitude
    lm, T, n_sym = n_SiO2_sym()
    n_sym_double_prime = sp.diff(n_sym,lm,lm)
    c = Q_(3e8,'m/s') # unitful definition of speed of light
    gvd_sym_no_prefactors = (lm**3)*n_sym_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
    gvd_no_prefactors = sp.lambdify([lm,T],gvd_sym_no_prefactors,'numpy') # numerical gvd without unitful prefactors
    gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_um,T_C),'um')
    return gvd.to('fs**2 / mm')


###### "Miller Delta" scaling function for nonlinear polarizabilities

def d_MillerDelta_LN(lm1,lm2,lm3,T):
    """This function calculates the nonlinear coefficient d33 [pm/V]
    of LiNb03 as a funtion of wavelength in [um] and temperature in [degC].

    d33 value is from Shoji, et al., JOSA B 14, 2268 (1997).
    This is the value used by Jason Pelc and Marty Fejer in
    "Supercontinuum generation in quasi-phase-matched LiNbO3
    waveguide pumped by a Tm-doped fiber laser system" Optics Letters 36
    (2011). I'm assuming it's good since they trust it.

    """
#   Values more accurate for 1.064um pumped processes
    d_start = 20.3*2/np.pi * (u.pm / u.V)
    lm1_start = 1.313*u.um
    lm2_start = 1.313*u.um
    lm3_start = 1.313/2 *u.um
#    Values to use for green-pumped processes
#    d_start = 27*2/np.pi
#    lm1_start = 1.064 *u.um
#    lm2_start = 1.064 *u.um
#    lm3_start = 1.064/2 *u.um
    T_start = 25 * u.degC;

    chi1_start = n_MgO_LN( lm1_start, T_start)**2 - 1
    chi2_start = n_MgO_LN( lm2_start, T_start)**2 - 1
    chi3_start = n_MgO_LN( lm3_start, T_start)**2 - 1

    chi1 = n_MgO_LN( lm1, T)**2 - 1
    chi2 = n_MgO_LN( lm2, T)**2 - 1
    chi3 = n_MgO_LN( lm3, T)**2 - 1

    d = d_start*chi1*chi2*chi3/(chi1_start*chi2_start*chi3_start)

    return d


###### Three-wave mixing of gaussian beams based on Boyd & Kleinman

def hbar_BoydKleinman(xi,sigma,B):
    beta = B/np.sqrt(xi)
    tau = np.linspace(-1*xi,xi,300)     # position in crystal relative to focus,
                                        # normalized to focal paramter b
    d_tau = np.mean(np.diff(tau))
    Hbar_integrand = np.exp( 1j*(sigma*tau) - beta**2 * tau**2) / (1 + 1j*tau)
    Hbar = (1 / (2*np.pi)) * np.trapz(Hbar_integrand,tau)
    hbar = (np.pi**2 / xi) * np.abs(Hbar)**2
    return hbar


###### Three-wave mixing of gaussian beams based on Boyd & Kleinman

def Hbar_BoydKleinman(xi,sigma,B):
    n_pts = 300    # number of points used in integration
#    xi = xi.to_base_units().flatten()
#    sigma = sigma.to_base_units().flatten()
    xi = xi.to_base_units()
    sigma = sigma.to_base_units()
    xi_tile = np.tile(xi.T,(n_pts,1,1)).T
    sigma_tile = np.tile(sigma.T,(n_pts,1,1)).T
    beta = B/np.sqrt(xi_tile)
    tau = np.array([[np.linspace(-1*x,x,n_pts) for x in xi[y,:]] for y in range(xi.shape[0])])


    #tau = np.array([np.linspace(-1*x,x,n_pts) for x in xi])
    #tau = np.linspace(-1*xi,xi,300)     # position in crystal relative to focus,
                                        # normalized to focal paramter b
    #d_tau = np.mean(np.diff(tau))
    Hbar_integrand = np.exp( 1j*(sigma_tile*tau) - beta**2 * tau**2) / (1 + 1j*tau)
    Hbar = (1 / (2*np.pi)) * np.trapz(Hbar_integrand,tau,axis=2)
    return Hbar


###### OPG gain based on Boyd and Kleinman
def ParametricGain_BK(lm_p,lm_s,n_p,n_s,n_i,w_s,rho,deff,l_c,LM):
    """ Calculation of Optical Parametric gain (returned as a unitless
    quantity, eg. .01 is 1%) from a single pass through a periodically
    poled Chi^(2) crystal of length l_c [m] with effective nonlinear
    coefficient deff [pm/V] and walkoff angle rho [radians].

    Based on Boyd and Kleinman "Parametric Interaction
    of Focused Gaussian Light Beams"  JAP 39 p.3597, 1968

    Variables: pump and signal wavelengths lm_p and lm_s  [um], confocal
    parameter b=2*z_R  [m] (this is assumed to be the same
    for all three beams as in the cited paper), walkoff angle rho [rad],
    effective nonlinearity deff [pm/V], crystal length l_c [m],
    periodic poling/patterning/etc spatial frequency LM [m]. """

    # Constants
    c=u.speed_of_light # speed of light

    lm_i = 1/(1/lm_p-1/lm_s) # [um] idler free-space wavelength

    #n_p = n_MgO_LN(lm_p, T) # [1] pump index
    #n_s = n_MgO_LN(lm_s, T) # [1] signal index
    #n_i = n_MgO_LN(lm_i, T) # [1] idler index

    om_p = (2*np.pi*u.rad)*c/lm_p # [rad/s] pump radial freq
    om_s = (2*np.pi*u.rad)*c/lm_s # [rad/s] signal radial freq
    om_i = (2*np.pi*u.rad)*c/lm_i # [rad/s] idler radial freq

    k_p = n_p*om_p/c # [rad/m] pump wave vector
    k_s = n_s*om_s/c # [rad/m] signal wave vector
    k_i = n_i*om_i/c # [rad/m] idler wave vector

    Delta_k = k_s + k_i - k_p + (2*np.pi*u.rad)/LM # [rad/m] momentum mismatch

    b = w_s**2 * k_s / u.rad

    w_p = np.sqrt(b/k_p*u.rad) # [m] pump beam waist
    #    w_s = sqrt(b/k_s) # [m] signal beam waist
    #    w_i = sqrt((1/w_s**2 + 1/w_p^2)**(-1/2)) # [m] idler beam waist (assumes near field conditions, see EE346 lecture 9, slide 20)
    #
    delta_p = 2*w_p/b * u.rad # [rad] pump far field diffraction angle
    #    delta_s = 2*w_s/b # [rad] signal far field diffraction angle
    #    delta_i = 2*w_i/b # [rad] idler far field diffraction angle

    # Paramaters in terms of average and differences

    om_0 = .5*(om_s+om_i) # [rad/s] average of signal and idler angular frequencies
    n_0 = .5*(n_s+n_i) # % [1] average of signal and idler indices
    k_0 = om_0*n_0 / c # [rad/m] average of signal and idler wavenumbers
    w_0 = b/k_0 # [m] average waist
    gamma = 1-om_i/om_0 # [1] signal-idler frequency difference parameter
    zeta = 1-n_i/n_0 # [1] signal-idler index difference paramter
    xi = l_c/b # [1] length of crystal in units of matched focusing parameter
    sigma = .5*b*Delta_k / u.radian# what are the units of this object? looks like it should be [radians] and mean phase mismatch over focusing region? but apparently it is unitless based on context in equations
    beta = rho/delta_p*np.sqrt(n_p/n_0) # walkoff angle in units of pump diffraction

    z = Q_(np.linspace(0,l_c.magnitude,1000),l_c.units) # [m] position in crystal
    f = l_c/2 # [m] position of focus in crystal, assumed to be centered
    tau = 2*(z-f)/b # [1] position in crystal relative to focus, normalized to focal paramter b
    H_bar_integrand = np.exp(1j*(sigma*tau)- (beta**2)*(tau**2))/(1+1j*tau)
    H_bar = np.trapz(H_bar_integrand,tau)
    h_bar = (np.pi**2/xi)*(np.abs(H_bar))**2

    chi = deff / (4 * np.pi**(3/2) * np.sqrt(u.epsilon_0) * u.rad**(3/2))


    K_bar = (128 * np.pi**2 * om_0**2 * chi**2) / (n_p * n_s * n_i * c**3)
    # K_bar is defined according to Equation 3.27 of Boyd+Kleinman

    gain =  ( (1-gamma**2)**2 * (1-zeta**2) / (1+gamma*zeta) ) * (1/4) * K_bar * k_0 * l_c
    # gain is in units of [1/W] and gives the gain per pass per Watt pump power
    # this matches Equation 3.34 of Boyd+Kleinman

    return gain


def ThermExp_PPLN(LM,T):
    """ Function to return the thermally expanded poling period at a given
    temperature given the poling period at room temperature using ordinary
    axis thermal expansion data from the literatre. """

    #LiNbO3 Ordinary Axis Thermal Expansion Coefficients
    #from 'Measurement of the thermal expansion coefficients of
    # ferroelectric crystals by a moire interferometer',
    # Pignatiello et al., Opt. Comm. 277 (2007) p. 14-18

    alpha0= 13.4e-6 * 1/u.degK
    alpha1= 9.2e-9 * 1/u.degK**2

    LM_T = LM*(1+alpha0*(T-25*u.degC)+alpha1*(T-25*u.degC)**2) # Poling period after thermal expansion

    return LM_T


def QPM_PPLN(lm_p,LM,T):
    """ Function to calculate the quasi-phase-matched wavelengths in
    periodically poled (MgO-doped) lithium niobate as a function of
    pump wavelength lm_p [um], room temperature poling period LM [um]
    and crystal temperature T [degC]. """
    lm_s_size = 3e4 #size of wavelength space searched for phase matching condition
    LM_array = Q_(np.array(LM.magnitude),LM.units).flatten()
    T_array = Q_(np.array(T.magnitude),T.units).flatten()
    lm_s_output = np.zeros((T_array.size,LM_array.size))
    lm_i_output = np.zeros((T_array.size,LM_array.size))
    n_p = np.tile(n_MgO_LN(lm_p,T_array),(1,lm_s_size))
    lm_s = Q_(np.linspace(lm_p.to(u.um).magnitude*1.2,2*lm_p.to(u.um).magnitude,lm_s_size),'um')
    n_s = n_MgO_LN(lm_s,T_array)
    lm_i = 1/(1/lm_p - 1/lm_s)
    n_i = n_MgO_LN(lm_i,T_array)
    for LM_idx, LM_curr in enumerate(LM_array):
        LM_T = Q_(np.tile(ThermExp_PPLN(LM_curr,T_array).magnitude,(lm_s_size,1)).T, LM_array.units)
        rhs = n_s/lm_s + n_i/lm_i - n_p/lm_p + 1/LM_T
        lm_s_output[:,LM_idx] = lm_s[np.argmin(np.abs(rhs),axis=1)]
        lm_i_output[:,LM_idx] = lm_i[np.argmin(np.abs(rhs),axis=1)]
    return lm_s_output, lm_i_output


def ThermExp_GaAs(LM,T):
    """ Function to return the thermally expanded poling period at a given
    temperature given the poling period at room temperature using ordinary
    axis thermal expansion data from the literatre. """

    alpha = 0/u.degC
    LM_T = LM * (1 + alpha*T) # Poling period after thermal expansion

    return LM_T

def QPM_OPGaAs(lm_p,LM,T):
    """ Function to calculate the quasi-phase-matched wavelengths in
    periodically poled (orientation patterned) Gallium Arsenide as a function of
    pump wavelength lm_p [um], room temperature poling period LM [um]
    and crystal temperature T [degC]. """
    lm_s_size = 3e4 #size of wavelength space searched for phase matching condition
    LM_array = Q_(np.array(LM.magnitude),LM.units).flatten()
    T_array = Q_(np.array(T.magnitude),T.units).flatten()
    lm_s_output = np.zeros((T_array.size,LM_array.size))
    lm_i_output = np.zeros((T_array.size,LM_array.size))
    n_p = np.tile(n_GaAs(lm_p,T_array),(1,lm_s_size))
    lm_s = Q_(np.linspace(lm_p.to(u.um).magnitude*1.4,2*lm_p.to(u.um).magnitude,lm_s_size),'um')
    n_s = n_GaAs(lm_s,T_array)
    lm_i = 1/(1/lm_p - 1/lm_s)
    n_i = n_GaAs(lm_i,T_array)
    for LM_idx, LM_curr in enumerate(LM_array):
        LM_T = Q_(np.tile(ThermExp_GaAs(LM_curr,T_array).magnitude,(lm_s_size,1)).T, LM_array.units)
        rhs = n_s/lm_s + n_i/lm_i - n_p/lm_p + 1/LM_T
        lm_s_output[:,LM_idx] = lm_s[np.argmin(np.abs(rhs),axis=1)]
        lm_i_output[:,LM_idx] = lm_i[np.argmin(np.abs(rhs),axis=1)]
    return lm_s_output, lm_i_output


def ParametricGain_BK_PPLN(lm_p,lm_s,T,w_s,l_c,LM):
    """Wrapper for Boyd-Kleinman Parametric Gain Calculation function
    ParametricGain_BK (above) specific for MgO:PPLN."""
    c = u.speed_of_light
    eps0 = u.epsilon_0

    LM_array = Q_(np.array(LM.magnitude),LM.units).flatten()
    T_array = Q_(np.array(T.magnitude),T.units).flatten()
    lm_s_size = np.size(lm_s)

    for LM_idx, LM_curr in enumerate(LM_array):

        LM_T = Q_(np.tile(ThermExp_PPLN(LM_curr,T_array).magnitude,(lm_s_size,1)).T, LM_array.units)

        n_p = np.tile(n_MgO_LN(lm_p,T_array),(1,lm_s_size))
        #lm_s = Q_(np.linspace(lm_p.to(u.um).magnitude*1.2,2*lm_p.to(u.um).magnitude,lm_s_size),'um')
        n_s = n_MgO_LN(lm_s,T_array)
        lm_i = 1/(1/lm_p - 1/lm_s)
        n_i = n_MgO_LN(lm_i,T_array)
        om_s = lm_s.to(u.rad/u.s)
        om_i = lm_i.to(u.rad/u.s)

        w_p = np.sqrt( (lm_p / lm_s) * (n_s / n_p) ) * w_s

        z_R = ( np.pi * (w_s**2) * n_s ) / (lm_s)
        b = 2 * z_R
        xi = (l_c / b).to_base_units()

        delta_k = (2*np.pi*u.rad)* (n_s/lm_s + n_i/lm_i - n_p/lm_p + 1/LM_T)
        sigma = b * delta_k / 2  / u.rad

        deff = d_MillerDelta_LN(lm_p,lm_s,lm_i,T_array)

        bare_gain = (4 * om_s * om_i * deff**2 * l_c**2) / \
                    ( (n_p * n_s * n_i) * eps0 * (c**3) * np.pi * (w_p**2 + w_s**2) ) \
                    / (u.rad)**2 # this is necessary to make the units [W**-1]
        ## following Eq. 13 of "Quasi-phase-matched optical parametric oscillators
        ## in bulk periodically poled LiNbO3", Myers, Fejer, Byer et al. JOSA B,
        ## vol 12, p.2102 (1995)

        Hbar = Hbar_BoydKleinman(xi,sigma,0)
        ## following Eq. 3.30 in "Parametric Interaction of Focused Gaussian Light
        ## Beams", Boyd and Kleinman, JAP 39 p.3597 (1968)


        reduced_gain = bare_gain.to(1/u.W) * np.abs(np.pi/xi * Hbar)**2

    return reduced_gain
