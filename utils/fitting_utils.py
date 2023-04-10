# even more lorentzian fitting stuff

# define data processing functions

##### Lorentzian fitting ######
import numpy as np
import matplotlib.pyplot as plt

from units_utils import u, Q_
from scipy.optimize import curve_fit
from scipy.signal import lombscargle, find_peaks
from typing import Union, Optional

from plot_utils import handle_axes, add_textbox, save_figure

π = np.pi

################################################################################
##                                                                            ##
##                            Lineshape Functions                             ##
##                                                                            ##
################################################################################

def lorentzian(x, A, x0, FWHM):
    """
    Lorentzian curve. Takes an array ``x`` and returns an array
    :math:`\\frac{A}{1 + (\\frac{2(x-x0)}{FWHM})^2}`
    """
    return A / (1 + ( 2*(x-x0)/FWHM )**2)

def lorentzian_reflection(nu, A0, nu0, FWHM):
    """
    Lorentzian curve. Takes an array ``x`` and returns an array
    :math:`\\frac{A}{1 + (\\frac{2(x-x0)}{FWHM})^2}`
    """
    return 1 - A0 / (1 + ( 2*(nu-nu0)/FWHM )**2)

def offset_lorentzian(x, A, x0, FWHM,y0):
    """
    Lorentzian curve. Takes an array ``x`` and returns an array
    :math:`\\frac{A}{1 + (\\frac{2(x-x0)}{FWHM})^2}`
    """
    return A / (1 + ( 2*(x-x0)/FWHM )**2) + y0

def offset_lorentzian_reflection(nu, A0, nu0, FWHM,y0):
    """
    Lorentzian curve. Takes an array ``x`` and returns an array
    :math:`\\frac{A}{1 + (\\frac{2(x-x0)}{FWHM})^2}`
    """
    return (1 - A0 / (1 + ( 2*(nu-nu0)/FWHM )**2)) + y0

def double_lorentzian(t, A0, FWHM, t0, dt):
    """
    Triple lorentzian curve. Takes an array ``t`` and returns an array
    that is the sum of three lorentzians
    ``lorentzian(t, A0, t0, FWHM) + lorentzian(nu, B0, nu0-dt_sb, FWHM)
    + lorentzian(t, B0, t0+dt_sb, FWHM)``.
    """
    l1 = lorentzian(t, A0, t0-dt/2.0, FWHM)
    l2 = lorentzian(t, A0, t0+dt/2.0, FWHM)
    return l1 + l2 

def triple_lorentzian(t, A0, B0, FWHM, t0, dt_sb):
    """
    Triple lorentzian curve. Takes an array ``t`` and returns an array
    that is the sum of three lorentzians
    ``lorentzian(t, A0, t0, FWHM) + lorentzian(nu, B0, nu0-dt_sb, FWHM)
    + lorentzian(t, B0, t0+dt_sb, FWHM)``.
    """
    l1 = lorentzian(t, A0, t0, FWHM)
    l2 = lorentzian(t, B0, t0-dt_sb, FWHM)
    l3 = lorentzian(t, B0, t0+dt_sb, FWHM)
    return l1 + l2 + l3 

def triple_double_lorentzian(t, A0, B0, FWHM, t0, dt, dt_sb):
    """
    Triple lorentzian curve. Takes an array ``t`` and returns an array
    that is the sum of three lorentzians
    ``lorentzian(t, A0, t0, FWHM) + lorentzian(nu, B0, nu0-dt_sb, FWHM)
    + lorentzian(t, B0, t0+dt_sb, FWHM)``.
    """
    l1 = lorentzian(t, A0, t0-dt/2.0, FWHM)
    l2 = lorentzian(t, A0, t0+dt/2.0, FWHM)
    l3 = lorentzian(t, B0, t0-dt_sb-dt/2.0, FWHM)
    l4 = lorentzian(t, B0, t0-dt_sb+dt/2.0, FWHM)
    l5 = lorentzian(t, B0, t0+dt_sb-dt/2.0, FWHM)
    l6 = lorentzian(t, B0, t0+dt_sb+dt/2.0, FWHM)
    return l1 + l2 + l3 + l4 + l5 + l6

def triple_double_lorentzian(t, A0, B0, FWHM, f0, df, df_sb):
    """
    Triple lorentzian curve. Takes an array ``t`` and returns an array
    that is the sum of three lorentzians
    ``lorentzian(t, A0, t0, FWHM) + lorentzian(nu, B0, nu0-dt_sb, FWHM)
    + lorentzian(t, B0, t0+dt_sb, FWHM)``.
    """
    l1 = lorentzian(t, A0, f0-df/2.0, FWHM)
    l2 = lorentzian(t, A0, f0+df/2.0, FWHM)
    l3 = lorentzian(t, B0, f0-df_sb-df/2.0, FWHM)
    l4 = lorentzian(t, B0, f0-df_sb+df/2.0, FWHM)
    l5 = lorentzian(t, B0, f0+df_sb-df/2.0, FWHM)
    l6 = lorentzian(t, B0, f0+df_sb+df/2.0, FWHM)
    return l1 + l2 + l3 + l4 + l5 + l6

def triple_lorentzian_freq(f, A0, B0, FWHM, f0, df_sb):
    """
    Triple lorentzian curve. Takes an array ``t`` and returns an array
    that is the sum of three lorentzians
    ``lorentzian(t, A0, t0, FWHM) + lorentzian(nu, B0, nu0-dt_sb, FWHM)
    + lorentzian(t, B0, t0+dt_sb, FWHM)``.
    """
    l1 = lorentzian(f, A0, f0, FWHM)
    l2 = lorentzian(f, B0, f0-df_sb, FWHM)
    l3 = lorentzian(f, B0, f0+df_sb, FWHM)
    return l1 + l2 + l3 

def ring_resonator_reflection(f, a, r, FSR, f0):
    """
    Reflection spectrum from a ring resonator as a function
    of the resonator coupling, internal round trip (field) loss
    coefficient `a`, free spectral range `FSR` and resonance frequency
    `f0`. 

    This describes transmission through a bus waveguide transmission "past" a 
    microring resonator.
     
    Takes an array `f` and returns an array
    
    :math:`\\frac{a^2 - 2ar\\cos(\\Delta\\phi) + r^2}{1 - 2ar\\cos(\\Delta\\phi) + a^2 r^2}`

    where the relative round-trip phase `Δϕ` is computed from the FSR
    and resonance detuning `f-f0` as
    
    :math:`\\Delta\\phi = \frac{ 2\\pi }{FSR} (f - f\\_0)`

    `a` is the round trip field loss coefficient for the resonator, such that 
                Eᵣₜ = a * exp(-i*Δϕ) * E₀
    
    `r` is the field reflection coefficient from the mirror or directional coupler
    between the external continuum (bus waveguide) and the cavity mode. 1-r² is the 
    fractional power coupled out from the resonant mode on each round trip.
    """
    Δϕ   =  ( 2*π / FSR ) * (f - f0)
    return (a**2 - 2*a*r*np.cos(Δϕ) + r**2) / (1 - 2*a*r*np.cos(Δϕ) + (a**2 * r**2))

################################################################################
##                                                                            ##
##                   Utility Functions for Lineshape Fitting                  ##
##                                                                            ##
################################################################################

is_sorted = lambda a: np.all(a[:-1] <= a[1:])

def _estimate_FWHM_pint(t, amp, half_max, left_limit, center, right_limit,verbose=False):
    """ Pint-friendly way to estimate FWHM. Doesn't use 'bad' numpy stuff. Edited by Dodd to work on a time
    x-axis (ms) rather than frequency (MHz) as Nate originally wrote.  
    """
    if verbose:
        print('entering estimate_FWHM_pint')
        print('half_max: {}'.format(half_max))
        print('t[0]: {}'.format(t[0]))
        print('left_limit: {}'.format(left_limit))
        print('center: {}'.format(center))
        print('right_limit: {}'.format(right_limit))
    # left_sum, right_sum = 0 * u.ms, 0 * u.ms
    left_sum, right_sum = 0 * t[0], 0 * t[0]
    divisor = 0
    i = 0
    while t[i] < left_limit:
        #print('loop 1, i={}'.format(i))
        i += 1

    while t[i] < center:
        if amp[i-1] <= half_max and amp[i] > half_max:
            if verbose:
                print('entered loop2 conditional!')
                print('i={}'.format(i))
                print('amp[i-1]={}'.format(amp[i-1]))
            left_sum += t[i]
            divisor += 1
        i += 1

    if verbose:
        print('after loops 1 and 2, i={}'.format(i))
    left_mean = left_sum / divisor
    divisor = 0

    while t[i] < right_limit:
        if amp[i-1] >= half_max and amp[i] < half_max:
            right_sum += t[i]
            divisor += 1
        i += 1
    right_mean = right_sum / divisor
    return right_mean - left_mean

def _estimate_FWHM_freq_pint(freq, amp, half_max, left_limit, center, right_limit,verbose=False):
    """ Pint-friendly way to estimate FWHM. Doesn't use 'bad' numpy stuff.  
    """
    if verbose:
        print('entering estimate_FWHM_pint')
        print('half_max: {}'.format(half_max))
        print('freq[0]: {}'.format(freq[0]))
        print('left_limit: {}'.format(left_limit))
        print('center: {}'.format(center))
        print('right_limit: {}'.format(right_limit))
    # left_sum, right_sum = 0 * freq.units, 0 * freq.units
    left_sum, right_sum = 0 * freq[0], 0 * freq[0]
    divisor = 0
    i = 0
    while freq[i] < left_limit:
        #print('loop 1, i={}'.format(i))
        i += 1

    while freq[i] < center:
        if amp[i-1] <= half_max and amp[i] > half_max:
            if verbose:
                print('entered loop2 conditional!')
                print('i={}'.format(i))
                print('amp[i-1]={}'.format(amp[i-1]))
            left_sum += freq[i]
            divisor += 1
        i += 1

    if verbose:
        print('after loops 1 and 2, i={}'.format(i))
    left_mean = left_sum / divisor
    divisor = 0

    while freq[i] < right_limit:
        if amp[i-1] >= half_max and amp[i] < half_max:
            right_sum += freq[i]
            divisor += 1
        i += 1
    right_mean = right_sum / divisor
    return right_mean - left_mean

def _estimate_FWHM(nu, amp, half_max, left_limit, center, right_limit):
    # Get x values of points where the data crosses half_max
    all_points = extract(diff(sign(amp-half_max)), nu)

    # Filter out the crossings from the sidebands
    middle_points = extract(logical_and(left_limit < all_points, all_points < right_limit),
                            all_points)
    center_index = searchsorted(middle_points, center)
    FWHM = middle_points[center_index:].mean() - middle_points[:center_index].mean()
    return FWHM

################################################################################
##                                                                            ##
##                        Lineshape Fitting Functions                         ##
##                                                                            ##
################################################################################

@u.with_context('sp')
def fit_single_lorentzian_dip(
    x,
    y,
    fit_data = {
        'x_range': (1572.9, 1573.4),  # x-value range in x for fitting
        'x_norm_frac': 0.1,          # fraction of x-value range to use for normalizing y data
        'x_units': 'THz',
        'y_units': 'dimensionless',
        'A0_guess': None,
        'x0_guess': None,
        'FWHM_guess': None,
    },
    verbose=True,
):
    """
    Fit single Lorentzian dip to unitful input data `x` and `y`. Fitting is controlled
    with parameters in `fit_data` and fit results are stored there as well.
    """
    xx              =   Q_(x,fit_data['x_units']).m   # convert to the desired x-axis units
    yy              =   Q_(y,fit_data['y_units']).m   # convert to the desired y-axis units
    if not is_sorted(xx):
        sort_inds   =   np.argsort(xx)
        xx          =   xx[sort_inds]
        yy          =   yy[sort_inds]
    
    x_min,x_max     = fit_data['x_range']
    mask            = (xx>x_min) * (xx<x_max)
    idx_mask        = np.where(mask)
    x_fit           = xx[idx_mask]
    y_fit           = yy[idx_mask]

    Dx              =  x_fit.max() - x_fit.min()   # xuency range of input data for fit
    mask_norm_0     =  x_fit < (x_fit.min() + fit_data['x_norm_frac']/2 * Dx)
    mask_norm_1     =  x_fit > (x_fit.max() - fit_data['x_norm_frac']/2 * Dx)

    
    # Guess the center x value based on y data minimum and create x offset data (`dx_fit`)
    idx_ymin        =   np.argmin(y_fit)
    x_ymin          =   x_fit[idx_ymin]
    # Measure background "1" levels in outermost `x_norm_frac` portion of x value range.
    # Measure background slope between background values at high and low x values.
    # Create "normalized" y data with background slope removed and background level set to 1. 
    dx_fit          =   x_fit - x_ymin
    dx_norm_0       =   dx_fit[mask_norm_0].mean()
    dx_norm_1       =   dx_fit[mask_norm_1].mean()
    y_norm_0        =   y_fit[mask_norm_0].mean()
    y_norm_1        =   y_fit[mask_norm_1].mean()
    bg_slope        =    (y_norm_1 - y_norm_0) / (dx_norm_1 - dx_norm_0)
    y_norm          =    y_fit - bg_slope * dx_fit
    y_norm          /=   (y_norm[mask_norm_0].mean() + y_norm[mask_norm_1].mean())/2 

    ### fit ###
    if fit_data['A0_guess'] is None:
        fit_data['A0_guess']    =   y_norm.min()
    if fit_data['FWHM_guess'] is None:
        fit_data['FWHM_guess']    =   _estimate_FWHM_pint(dx_fit, 1-y_norm, np.max(1-y_norm)/2, dx_fit.min(), 0.0, dx_fit.max())
    if fit_data['x0_guess'] is None:
        fit_data['x0_guess'] = x_ymin
    p0 =  (fit_data['A0_guess'],fit_data['x0_guess']-x_ymin,fit_data['FWHM_guess'])
    popt, pcov = curve_fit(lorentzian_reflection, dx_fit, y_norm, p0=p0)

    A0_fit      =   popt[0]
    x0_fit      =   popt[1] + x_ymin
    FWHM_fit    =   popt[2]
    Q_fit       =   x0_fit / FWHM_fit

    A0_fit_err,x0_fit_err,FWHM_fit_err   =   np.sqrt(np.diag(pcov))
    # Q_fit_err       =   (f0_fit_err / FWHM_fit_err).m_as(u.dimensionless)

    fit_data.update({
        'A0_fit':   A0_fit,
        'x0_fit':   x0_fit,
        'FWHM_fit': FWHM_fit,
        'A0_fit_err':   A0_fit_err,
        'x0_fit_err':   x0_fit_err,
        'FWHM_fit_err': FWHM_fit_err,
        'Q_fit':   Q_fit,
        'x_fit':   x_fit,
        'dx_fit':   dx_fit,
        'y_norm_fit': y_norm,
    })

    if verbose:
        print('fit parameters:')
        print(f'\tA0:\t\t{A0_fit:3.2f} +- {A0_fit_err:3.2f}')
        print(f'\tx0:\t\t{x0_fit:3.2f} +- {x0_fit_err:3.2f}')
        print(f'\tFWHM:\t{FWHM_fit:3.2f} +- {FWHM_fit_err:3.2f}')
        print(f'\tQ:\t\t{Q_fit:4.3g}')

    return fit_data

@u.with_context('sp')
def fit_single_lorentzian_dip2(
    x,
    y,
    fit_data = {
        'start_idx': 0,  # x-value range in x for fitting
        'stop_idx': -1,  # x-value range in x for fitting
        'x_units': 'THz',
        'y_units': 'dimensionless',
        'A0_guess': None,
        'x0_guess': None,
        'FWHM_guess': None,
    },
    verbose=True,
):
    """
    Fit single Lorentzian dip to unitful input data `x` and `y`. Fitting is controlled
    with parameters in `fit_data` and fit results are stored there as well.
    """
    xx              =   Q_(x,fit_data['x_units']).m   # convert to the desired x-axis units
    yy              =   Q_(y,fit_data['y_units']).m   # convert to the desired y-axis units
    x_0             =   Q_(x[fit_data['start_idx']],fit_data['x_units']).m
    x_1             =   Q_(x[fit_data['stop_idx']],fit_data['x_units']).m
    x_min,x_max     =   min([x_0,x_1]), max([x_0,x_1])
    if verbose:
        print(f'x_min: {x_min}')
        print(f'x_max: {x_max}')
    # y_xmin           =   Q_(y[fit_data['start_idx']],fit_data['y_units']).m
    # y_xmax           =   Q_(y[fit_data['stop_idx']],fit_data['y_units']).m

    if not is_sorted(xx):
        sort_inds   =   np.argsort(xx)
        xx          =   xx[sort_inds]
        yy          =   yy[sort_inds]
    
    # x_min,x_max     = fit_data['x_range']
    
    mask            = (xx>x_min) * (xx<x_max)
    idx_mask        = np.where(mask)
    x_fit           = xx[idx_mask]
    y_fit           = yy[idx_mask]

    Dx              =  x_fit.max() - x_fit.min()   # xuency range of input data for fit
    # mask_norm_0     =  x_fit < (x_fit.min() + fit_data['x_norm_frac']/2 * Dx)
    # mask_norm_1     =  x_fit > (x_fit.max() - fit_data['x_norm_frac']/2 * Dx)

    
    # Guess the center x value based on y data minimum and create x offset data (`dx_fit`)
    idx_ymin        =   np.argmin(y_fit)
    x_ymin          =   x_fit[idx_ymin]
    # Measure background "1" levels in outermost `x_norm_frac` portion of x value range.
    # Measure background slope between background values at high and low x values.
    # Create "normalized" y data with background slope removed and background level set to 1. 
    dx_fit          =   x_fit - x_ymin
    # dx_norm_0       =   dx_fit[mask_norm_0].mean()
    # dx_norm_1       =   dx_fit[mask_norm_1].mean()
    # y_norm_0        =   y_fit[mask_norm_0].mean()
    # y_norm_1        =   y_fit[mask_norm_1].mean()
    dx_norm_0       =   dx_fit[0]
    dx_norm_1       =   dx_fit[-1]
    y_norm_0        =   y_fit[0]
    y_norm_1        =   y_fit[-1]
    bg_slope        =    (y_norm_1 - y_norm_0) / (dx_norm_1 - dx_norm_0)
    y_norm          =    y_fit - bg_slope * dx_fit
    # y_norm          /=   (y_norm[mask_norm_0].mean() + y_norm[mask_norm_1].mean())/2 
    y_norm          /=  (y_norm_0 + y_norm_1)/2.0

    ### fit ###
    if fit_data['A0_guess'] is None:
        fit_data['A0_guess']    =   y_norm.min()
    if fit_data['FWHM_guess'] is None:
        fit_data['FWHM_guess']    =   _estimate_FWHM_pint(dx_fit, 1-y_norm, np.max(1-y_norm)/2, dx_fit.min(), 0.0, dx_fit.max())
    if fit_data['x0_guess'] is None:
        fit_data['x0_guess'] = x_ymin
    p0 =  (fit_data['A0_guess'],fit_data['x0_guess']-x_ymin,fit_data['FWHM_guess'])
    popt, pcov = curve_fit(lorentzian_reflection, dx_fit, y_norm, p0=p0)

    A0_fit      =   popt[0]
    x0_fit      =   popt[1] + x_ymin
    FWHM_fit    =   popt[2]
    Q_fit       =   x0_fit / FWHM_fit

    A0_fit_err,x0_fit_err,FWHM_fit_err   =   np.sqrt(np.diag(pcov))
    # Q_fit_err       =   (f0_fit_err / FWHM_fit_err).m_as(u.dimensionless)

    fit_data.update({
        'A0_fit':   A0_fit,
        'x0_fit':   x0_fit,
        'FWHM_fit': FWHM_fit,
        'A0_fit_err':   A0_fit_err,
        'x0_fit_err':   x0_fit_err,
        'FWHM_fit_err': FWHM_fit_err,
        'Q_fit':   Q_fit,
        'x_fit':   x_fit,
        'dx_fit':   dx_fit,
        'y_norm_fit': y_norm,
        'dx_norm_0': dx_norm_0,
        'dx_norm_1': dx_norm_1,
        'y_norm_0': y_norm_0,
        'y_norm_1': y_norm_1,
    })

    if verbose:
        print('fit parameters:')
        print(f'\tA0:\t\t{A0_fit:3.2f} +- {A0_fit_err:3.2f}')
        print(f'\tx0:\t\t{x0_fit:3.2f} +- {x0_fit_err:3.2f}')
        print(f'\tFWHM:\t{FWHM_fit:3.2f} +- {FWHM_fit_err:3.2f}')
        print(f'\tQ:\t\t{Q_fit:4.3g}')

    return fit_data

@u.with_context('sp')
def fit_ring_resonator_reflection(
    x,
    y,
    fit_data = {
        'start_idx': 0,  # x-value range in x for fitting
        'stop_idx': -1,  # x-value range in x for fitting
        'x_units': 'GHz',
        'y_units': 'dimensionless',
        'a_guess': None,
        'r_guess': None,
        'FSR_guess': None,
        'f0_guess':None,
    },
    verbose=True,
):
    """
    Fit single Lorentzian dip to unitful input data `x` and `y`. Fitting is controlled
    with parameters in `fit_data` and fit results are stored there as well.
    """
    xx              =   Q_(x,fit_data['x_units']).m   # convert to the desired x-axis units
    yy              =   Q_(y,fit_data['y_units']).m   # convert to the desired y-axis units
    start_idx       =   fit_data['start_idx']
    stop_idx        =   fit_data['stop_idx']
    x_fit = xx[start_idx:stop_idx]
    y_fit = yy[start_idx:stop_idx]
    if not is_sorted(x_fit):
        sort_inds   =   np.argsort(x_fit)
        x_fit       =   x_fit[sort_inds]
        y_fit       =   y_fit[sort_inds]
    y_scale = y_fit.max()
    y_norm = y_fit/y_scale
    if fit_data['a_guess'] is None:
        fit_data['a_guess'] = 0.7
    if fit_data['r_guess'] is None:
        fit_data['r_guess'] = 0.7
    if fit_data['f0_guess'] is None:
        fit_data['f0_guess'] = x_fit[np.argmin(y_norm)]
    f0 = fit_data['f0_guess']
    dx_fit = x_fit - f0
    # fit `ring_resonator_reflection(f, a, r, FSR, f0)`
    p0 =  (fit_data['a_guess'],fit_data['r_guess'],fit_data['FSR_guess'],0.0)
    popt, pcov = curve_fit(ring_resonator_reflection, dx_fit, y_norm, p0=p0)
    
    a_fit,r_fit,FSR_fit,df0_fit  =   popt
    a_fit_err,r_fit_err,FSR_fit_err,f0_fit_err  =  np.sqrt(np.diag(pcov))
    f0_fit      =   df0_fit + f0
    
    fit_data.update({
        'a_fit':   a_fit,
        'r_fit':   r_fit,
        'FSR_fit': FSR_fit,
        'f0_fit': f0_fit,
        'a_fit_err':   a_fit_err,
        'r_fit_err':   r_fit_err,
        'FSR_fit_err': FSR_fit_err,
        'f0_fit_err': f0_fit_err,
        'y_scale':  y_scale,
        'x_fit':   x_fit,
        'y_fit':    y_fit,
    })

    if verbose:
        print('fit parameters:')
        print(f'\ta:\t\t{a_fit:3.2f} +- {a_fit_err:3.2f}')
        print(f'\tr:\t\t{r_fit:3.2f} +- {r_fit_err:3.2f}')
        print(f'\tFSR:\t\t{FSR_fit:3.2f} +- {FSR_fit_err:3.2f}')
        print(f'\tf0:\t\t{f0_fit:3.2f} +- {f0_fit_err:3.2f}')

    return fit_data



@u.with_context('sp')
def fit_fsr_lombscargle(
    x_data,
    y_data,
    frequency_units = 'GHz',
    FSR_guess_min = 30,  # in `frequency_units`
    FSR_guess_max = 200, # in `frequency_units`
    nperiods = 10000,       # number of frequencies/periods of computed periodogram
):
    """
    Find the dominant period in a resonance spectrum using the `lombscargle`
    periodogram provided by scipy.signal. Unlike FFTs and other periodogram
    algorithms this allows for non-uniform sampling. This is handy for spectra
    with irregularly spaced frequency points (eg. if the data had regularly
    spaced wavelength points.)
    """
    x_freq = x_data.to(frequency_units).m
    y = Q_(y_data,'dimensionless').m
    
    sort_inds = np.argsort(x_freq)
    x_freq = x_freq[sort_inds]
    y = y[sort_inds]

    # periodogram angular frequencies (radian/GHz if frequency_units is GHz) 
    w_freq  = 2*np.pi*np.linspace(
        1/FSR_guess_max,
        1/FSR_guess_min,
        nperiods,
    )
    pgram   = lombscargle(x_freq-x_freq[0], y, w_freq, normalize=True,precenter=True)
    fsr_lomb = 2*np.pi/w_freq[np.argmax(pgram)]     # peak period in periodogram, assumed to be FSR
    return Q_(fsr_lomb,frequency_units)

def find_spectrum_peaks(x_data,y_data,FSR_guess_min = 50*u.GHz):
    dx_min = np.abs(np.diff(x_data).min())
    Didx_guess_min = FSR_guess_min / dx_min
    peak_inds, peak_props = find_peaks(y_data, distance=Didx_guess_min)
    return peak_inds, peak_props

# @u.with_context('sp')

def text_single_lorentzian_freq_fit(fit_data,FSR=None,**other_params):    
    FWHM_fit = Q_(fit_data['FWHM_fit'],fit_data['x_units'])
    if FSR is not None:
        FSR = Q_(FSR,fit_data['x_units'])
        FSR_rad = 2*np.pi * Q_(FSR,u.Hz)
        tau_ph = (1 / (2 * np.pi * FWHM_fit)).to(u.ps)
        finesse = (tau_ph * FSR_rad).to(u.dimensionless).m
    fit_str = 'Fit parameters:\n'
    fit_str += 'λ₀ $= %3.2f \;\mathrm{nm}$\n'%(Q_(fit_data['x0_fit'],fit_data['x_units']).to('nm','sp').m)
    fit_str += '$T_\mathrm{min} = %3.2f$\n'%(1-fit_data['A0_fit'])
    fit_str += 'FWHM $= %3.2f \;\mathrm{GHz}$\n'%(FWHM_fit.m_as(u.GHz))
    fit_str += 'Q $= %6.0f$\n'%(fit_data['Q_fit'])
    if FSR is not None:
        fit_str = 'FSR $= %4.1f \;\mathrm{GHz}$\n'%(FSR.to(u.GHz).m) + fit_str
        fit_str += f'finesse: {finesse:6.1f}'
    for k,v in other_params.items():
        try:
            param_str = f'{k}: {v:3.3g}\n'
        except:
            param_str = f'{k}: {v}\n'
        fit_str = param_str + fit_str 

    return fit_str

def plot_single_lorentzian_dip_fit(
    fit_data,
    ax:Optional[plt.Axes]=None,
    FSR=None,
    # gap=None,
    xlabel='frequency offset [GHz]',
    ylabel='flattened/normalized transmission',
    other_params:dict={},
    data_color='C3',
    fit_color='k',
    fit_lw=2,
    data_alpha=0.8,
    figsize=(4,3),
    ylims=(0.0,1.05),
    textbox_params:dict = {
        'x_textbox': 0.02,         # X-position of text box in axes coords
        'y_textbox': 0.77,         # Y-position of text box in axes coords
        'text_props': {
            'fontsize':7,
            'verticalalignment':'top',
            'bbox': {
                'facecolor':'0.8',
                'alpha':0.4,
                'boxstyle':'round',
            },
        },
    },
):
    ax = handle_axes(ax,figsize=figsize)    # create figure & axes if needed
    npts_plot       =   300
    dx_fit  = fit_data['dx_fit']
    dx_fit_offset = fit_data['x_fit'][0] - dx_fit[0] 
    y_norm_fit  = fit_data['y_norm_fit']
    dx_fit_plot  =   np.linspace(dx_fit.min(),dx_fit.max(),npts_plot)
    popt = (fit_data['A0_fit'], fit_data['x0_fit']-dx_fit_offset, fit_data['FWHM_fit'])
    y_norm_fit_plot      =   lorentzian_reflection(dx_fit_plot,*popt)

    ax.plot(Q_(dx_fit,fit_data['x_units']),y_norm_fit,'.',color=data_color,alpha=data_alpha,label='meas.')
    ax.plot(Q_(dx_fit_plot,fit_data['x_units']),y_norm_fit_plot,ls='--',color=fit_color,lw=fit_lw,label='fit')

    ax.set_ylim(ylims)
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fit_str = text_single_lorentzian_freq_fit(fit_data,FSR=FSR,**other_params) # FSR=FSR,gap=gap)
    add_textbox(ax,fit_str,**textbox_params)
    return ax

def plot_single_lorentzian_dip_fit2(
    fit_data,
    ax:Optional[plt.Axes]=None,
    FSR=None,
    # gap=None,
    xlabel='frequency offset [GHz]',
    ylabel='flattened/normalized transmission',
    other_params:dict={},
    data_color='C3',
    fit_color='k',
    fit_lw=2,
    data_alpha=0.8,
    figsize=(4,3),
    ylims=(0.0,1.05),
    textbox_params:dict = {
        'x_textbox': 0.02,         # X-position of text box in axes coords
        'y_textbox': 0.77,         # Y-position of text box in axes coords
        'text_props': {
            'fontsize':7,
            'verticalalignment':'top',
            'bbox': {
                'facecolor':'0.8',
                'alpha':0.4,
                'boxstyle':'round',
            },
        },
    },
):
    ax = handle_axes(ax,figsize=figsize)    # create figure & axes if needed
    npts_plot       =   300
    dx_fit  = fit_data['dx_fit']
    dx_fit_offset = fit_data['x_fit'][0] - dx_fit[0] 
    y_norm_fit  = fit_data['y_norm_fit']
    
    dx_norm_0       =   fit_data['dx_norm_0']
    dx_norm_1       =   fit_data['dx_norm_1']
    y_norm_0        =   fit_data['y_norm_0']
    y_norm_1        =   fit_data['y_norm_1']
    bg_slope        =    (y_norm_1 - y_norm_0) / (dx_norm_1 - dx_norm_0)
    # y_norm          =    y_fit - bg_slope * dx_fit
    # y_norm          /=  (y_norm_0 + y_norm_1)/2.0
    
    y_rescale_fit   =   y_norm_fit * ((y_norm_0 + y_norm_1)/2.0)
    y_fit           =   y_rescale_fit + bg_slope * dx_fit

    dx_fit_plot  =   np.linspace(dx_fit.min(),dx_fit.max(),npts_plot)
    popt = (fit_data['A0_fit'], fit_data['x0_fit']-dx_fit_offset, fit_data['FWHM_fit'])
    y_norm_fit_plot      =   lorentzian_reflection(dx_fit_plot,*popt)

    y_rescale_fit_plot = y_norm_fit_plot * ((y_norm_0 + y_norm_1)/2.0)
    y_fit_plot      =   y_rescale_fit_plot + bg_slope * dx_fit_plot

    # ax.plot(Q_(dx_fit,fit_data['x_units']),y_norm_fit,'.',color=data_color,alpha=data_alpha,label='meas.')
    # ax.plot(Q_(dx_fit_plot,fit_data['x_units']),y_norm_fit_plot,ls='--',color=fit_color,lw=fit_lw,label='fit')

    ax.plot(Q_(dx_fit,fit_data['x_units']),y_fit,'.',color=data_color,alpha=data_alpha,label='meas.')
    ax.plot(Q_(dx_fit_plot,fit_data['x_units']),y_fit_plot,ls='--',color=fit_color,lw=fit_lw,label='fit')

    ax.set_ylim(ylims)
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fit_str = text_single_lorentzian_freq_fit(fit_data,FSR=FSR,**other_params) # FSR=FSR,gap=gap)
    add_textbox(ax,fit_str,**textbox_params)
    return ax
