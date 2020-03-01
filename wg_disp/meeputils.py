# Utilities for interfacing with Meep.
# Index code is from my wgms3d Python package.

import numpy as np
import meep as mp
import scipy.optimize as opt
from wurlitzer import pipes, STDOUT
from io import StringIO
import warnings
import h5py
import sys
from pathlib import Path

### import index models ###
home = str( Path.home() )
nlo_dir = home+'/github/optics_modeling/NLO_tools'
if nlo_dir not in sys.path:
    sys.path.append(nlo_dir)
import NLO_tools as nlo

u = nlo.u

def _n_linbo3(λ,T=300*u.degK,axis='e'):
    return nlo.n_MgO_LN(λ*u.um,T,axis=axis)[0,0]

def _ng_linbo3(λ,T=300*u.degK,axis='e'):
    return nlo.n_g_MgO_LN(λ*u.um,T,axis=axis)

def _n_si3n4(λ,T=300*u.degK):
    return nlo.n_Si3N4(λ*u.um,T)[0,0]

def _ng_si3n4(λ,T=300*u.degK):
    return nlo.n_g_Si3N4(λ*u.um,T)

def _n_sio2(λ,T=300*u.degK):
    return nlo.n_SiO2(λ*u.um,T)[0,0]

def _ng_sio2(λ,T=300*u.degK):
    return nlo.n_g_SiO2(λ*u.um,T)


# n_Si3N4(lm_in,T_in)

### Cauchy Equation fit coefficients for Gavin's ALD alumina films ###
# Eqn. form:
# n = A + B / lam**2 + C / lam**4
# ng = A + 3 * B / lam**2 + 5 * C / lam**4
A_alumina = 1.602
B_alumina = 0.01193
C_alumina = -0.00036


### Cauchy Equation fit coefficients for 100nm Hafnia (HfO2) films ###
## taken from
## Khoshman and Kordesch. "Optical properties of a-HfO2 thin films." Surface and Coatings Technology 201.6 (2006)
## great fit to measured data from sputtered amorphous Hafnia, haven't found one for ALD Hafnia specifically
## they also report loss, with a sharp absorption edge near 5.68 ± 0.09 eV (~218 nm)
# Eqn. form:
# n = A + B / lam**2 + C / lam**4
# ng = A + 3 * B / lam**2 + 5 * C / lam**4

# # fit for spectroscopic ellipsometer measurement for a 250nm thick film, good 300-1400nm
# A_hafnia = 1.85
# B_hafnia = 1.17e-8
# C_hafnia = 0.0

# fit for spectroscopic ellipsometer measurement for a 112nm thick film, good 300-1400nm
A_hafnia = 1.86
B_hafnia = 7.16e-9
C_hafnia = 0.0

# In_{1-x} Ga_x As_y P_{1-y}
# Modified single-oscillator model
# F. Fiedler and A. Schlachetzki. Optical parameters of InP-based waveguides.
#   Solid-State Electronics, 30(1):73–83, 1987.
def _n_ingaasp(y, lam_mu):
    Ed = 28.91 - 9.278 * y + 5.626 * y ** 2
    E0 = 3.391 - 1.652 * y + 0.863 * y ** 2 - 0.123 * y ** 3
    Eg = 1.35 - 0.72 * y + 0.12 * y ** 2
    Eph = 1.24 / lam_mu
    return np.sqrt(np.abs(1 + Ed / E0 + Ed * Eph ** 2 / E0 ** 3 + Ed * Eph ** 4 / (2 * E0 ** 3 * (E0 ** 2 - Eg ** 2)) *
                          np.log((2 * E0 ** 2 - Eg ** 2 - Eph ** 2) / (Eg ** 2 - Eph ** 2))))


# Just take the derivative: n_g = n + (omega/2n) d(n^2)/d(omega)
def _ng_ingaasp(y, lam_mu):
    n = _n_ingaasp(y, lam_mu);
    hw = 1.24 / lam_mu;
    Ed = 28.91 - 9.278 * y + 5.626 * y ** 2
    E0 = 3.391 - 1.652 * y + 0.863 * y ** 2 - 0.123 * y ** 3
    Eg = 1.35 - 0.72 * y + 0.12 * y ** 2
    return (n + 1 / (2 * n) * ((2 * Ed * hw ** 2) / E0 ** 3 + (2 * Ed * (hw ** 4)) / E0 ** 3 * (
                hw ** 2 / ((2 * E0 ** 2 - Eg ** 2 - hw ** 2) * (Eg ** 2 - hw ** 2)) +
                1 / (E0 ** 2 - Eg ** 2) * np.log((2 * E0 ** 2 - Eg ** 2 - hw ** 2) / (Eg ** 2 - hw ** 2)))))


# Lambda is in microns here.
def get_index(mat, lam):
    lam = lam
    if (mat == 'Si3N4'):
        # 1) H. R. Philipp. Optical properties of silicon nitride,
        #    J. Electrochim. Soc. 120, 295-300 (1973)
        # 2) T. Baak. Silicon oxynitride; a material for GRIN optics,
        #    Appl. Optics 21, 1069-1072 (1982)
        # return np.sqrt(1 + (2.8939 * lam ** 2) / (lam ** 2 - 0.1396 ** 2))
        return _n_si3n4(lam)
    elif (mat == 'SiO2'):
        # I. H. Malitson. Interspecimen Comparison of the Refractive Index of Fused Silica,
        # J. Opt. Soc. Am. 55, 1205-1208 (1965)
        # return np.sqrt(
        #     np.maximum(1 + (0.6961 * lam ** 2) / (lam ** 2 - 0.06840 ** 2) + (0.4079 * lam ** 2) / (lam ** 2 - 0.1162 ** 2)
        #     + (0.8974 * lam ** 2) / (lam ** 2 - 9.8961 ** 2), 1))
        return _n_sio2(lam)
    elif (mat == 'Si'):
        # 1) C. D. Salzberg and J. J. Villa. Infrared Refractive Indexes of Silicon,
        #    Germanium and Modified Selenium Glass, J. Opt. Soc. Am., 47, 244-246 (1957)
        # 2) B. Tatian. Fitting refractive-index data with the Sellmeier dispersion formula,
        #    Appl. Opt. 23, 4477-4485 (1984)
        return np.sqrt(
            1 + (10.6684 * lam ** 2) / (lam ** 2 - 0.3015 ** 2) + (0.003043 * lam ** 2) / (lam ** 2 - 1.1347 ** 2)
            + (1.5413 * lam ** 2) / (lam ** 2 - 1104 ** 2))
    elif (mat == 'Alumina'):
        return A_alumina + B_alumina / lam**2 + C_alumina / lam**4  # Cauchy Eqn. fit from Gavin's ALD alumina
    elif (mat == 'Hafnia'):
        return A_hafnia + B_hafnia / lam**2 + C_hafnia / lam**4  # Cauchy Eqn. fit from cited paper above for sputtered Hafnia
    elif (mat == 'Air'):
        return 1.0
    elif (mat == 'InP'):
        return _n_ingaasp(0, lam)
    elif (mat == 'LiNbO3'):
        return _n_linbo3(lam)
    elif (type(mat) == str and mat.startswith('InGaAsP_Q')):
        Eg = 1.24 / float(mat[9:])
        assert (0.75 <= Eg <= 1.35)
        y = (0.72 - np.sqrt(0.72 ** 2 - 4 * (0.12) * (1.35 - Eg))) / (2 * 0.12)
        return _n_ingaasp(y, lam)
    elif (type(mat) in [int, float, np.float64]):
        return mat
    else:
        raise ValueError("Material " + mat + " not supported.")


def get_ng(mat, lam):
    if (mat == 'InP'):
        return _ng_ingaasp(0, lam * 1e6)
    elif (mat == 'Si3N4'):
        return _ng_si3n4(lam)
    elif (mat == 'SiO2'):
        return _ng_sio2(lam)
    elif (mat == 'LiNbO3'):
        return _ng_linbo3(lam)
    elif (mat == 'Alumina'):
        return A_alumina + 3 * B_alumina / lam**2 + 5 * C_alumina / lam**4  # Cauchy Eqn. fit from Gavin's ALD alumina, analytic derivative for ng
    elif (mat == 'Hafnia'):
        return A_hafnia + 3 * B_hafnia / lam**2 + 5* C_hafnia / lam**4  # Cauchy Eqn. fit from cited paper above for sputtered Hafnia
    elif (mat.startswith('InGaAsP_Q')):
        Eg = 1.24 / float(mat[9:])
        assert (0.75 <= Eg <= 1.35)
        y = (0.72 - np.sqrt(0.72 ** 2 - 4 * (0.12) * (1.35 - Eg))) / (2 * 0.12)
        return _ng_ingaasp(y, lam * 1e6)
    else:
        # A lazy hack...
        [n100, n101, n099] = [get_index(mat, lam / x) for x in [1.00, 1.01, 0.99]]
        w0 = 2 * np.pi * 3e8 / lam
        dndw = (n101 - n099) / (0.02 * w0)
        return (dndw * w0 + n100)


# Taken from wgms3d/analytic.py

class ModeXYZ:
    beta = 0; k = 0; x = 0; y = 0;
    Ex = 0; Ey = 0; Ez = 0; Hx = 0; Hy = 0; Hz = 0;
    eps = 0;
    @property
    def omega(self): return self.k * 3e8
    @property
    def n_eff(self): return self.beta/self.k

class SlabWaveguide:
    w = 0; n0 = 0; n1 = 0; ns = 0;
    @property
    def a(self): return self.w/2.

    def __init__(self, w, n0, n1, ns):
        self.w = w; self.n0 = n0; self.n1 = n1; self.ns = ns;
    def beta(self, pol, m, k):
        r"""
        Wavenumber as function of k = omega/c, beta/k = n_eff
        """
        pars = self.modeParams(pol, m, k)
        if (pars.ndim == 1): return pars[0]
        elif (pars.ndim == 2): return pars[:, 0]
        elif (pars.ndim == 3): return pars[:, :, 0]
        else: raise NotImplementedError()

    def modeParams(self, pol, m, k):
        r"""
        Returns mode parameters [beta, kappa, sigma, xi, phi]

        pol: polarization ['te' | 'tm']
        m: mode index
        k: wavenumber = omega/c

        Ex (TE) | Hx (TM) = ...
          (y > a):   cos(kappa a - phi) exp(-sigma(y-a))
          (|y| < a): cos(kappa y - phi)
          (y < -a):  cos(-kappa a - phi) exp(xi(y+a))
        """

        if (np.iterable(k)):
            return np.array([self.modeParams(pol, m, ki) for ki in k])
        a = self.w/2;
        (n0, n1, ns) = [get_index(mat, 2*np.pi/k) for mat in [self.n0, self.n1, self.ns]]
        pol = pol.lower(); assert (pol in ['te', 'tm'])
        def fn(psi):
            v = (k*a)*np.sqrt(n1**2 - ns**2)
            u = v*np.cos(psi); w = v*np.sin(psi); wp = np.sqrt(v**2 * (ns**2-n0**2)/(n1**2-ns**2) + w**2)
            return (-u + (m*np.pi/2) + 0.5*np.arctan2(w, u) + 0.5*np.arctan2(wp, u) if (pol == 'te') else
                    -u + (m*np.pi/2) + 0.5*np.arctan2(n1**2*w, ns**2*u) + 0.5*np.arctan2(n1**2*wp, n0**2*u))
        psi = np.nan if (fn(0) * fn(np.pi/2) > 0) else opt.bisect(fn, 0, np.pi/2); v = (k*a)*np.sqrt(n1**2 - ns**2);
        u = v*np.cos(psi); w = v*np.sin(psi); wp = np.sqrt(v**2 * (ns**2-n0**2)/(n1**2-ns**2) + w**2);
        phi = ((m*np.pi/2) + 0.5*np.arctan2(w, u) - 0.5*np.arctan2(wp, u) if (pol == 'te') else
               (m*np.pi/2) + 0.5*np.arctan2(n1**2*w, ns**2*u) - 0.5*np.arctan2(n1**2*wp, n0**2*u))
        kappa = u/a; xi = w/a; sigma = wp/a;
        beta = np.sqrt(sigma**2 + (k*n0)**2)
        return np.array([beta, kappa, sigma, xi, phi])

    # Returns [beta, n, F, dy_F, dydy_F] for F = [Ex (TE) | Hx (TM)]
    def _fieldDerivs(self, pol, m, k, y):
        assert (not np.iterable(k))
        (n0, n1, ns) = [get_index(mat, 2*np.pi/k) for mat in [self.n0, self.n1, self.ns]]
        [beta, kappa, sigma, xi, phi] = self.modeParams(pol, m, k)
        a = self.w/2; reg1 = (y < -a); reg3 = (y > a); reg2 = 1 - reg1 - reg3
        F = [beta,
             reg1*ns + reg2*n1 + reg3*n0,
             reg1 * np.cos(-kappa*a - phi)*np.exp(-xi*np.abs(y+a))    +
             reg2 * np.cos( kappa*y - phi)                      +
             reg3 * np.cos( kappa*a - phi)*np.exp(-sigma*np.abs(y-a)),
             xi     * reg1 * np.cos(-kappa*a - phi)*np.exp(-xi*np.abs(y+a))    +
             -kappa * reg2 * np.sin( kappa*y - phi)                      +
             -sigma * reg3 * np.cos( kappa*a - phi)*np.exp(-sigma*np.abs(y-a)),
             xi**2     * reg1 * np.cos(-kappa*a - phi)*np.exp(-xi*np.abs(y+a))    +
             -kappa**2 * reg2 * np.cos( kappa*y - phi)                      +
             sigma**2  * reg3 * np.cos( kappa*a - phi)*np.exp(-sigma*np.abs(y-a))]
        return F

    def field(self, pol, m, k, y):
        out = ModeXYZ()
        [beta, n, F, dyF, dydyF] = self._fieldDerivs(pol, m, k, y)
        out.beta = beta; out.k = k; out.x = []; out.y = y;
        omega = k * 3e8; mu_0 = 4*np.pi*1e-7; eps_0 = 8.854e-12
        if (pol.lower() == 'te'):
            out.Ex = F;
            out.Ey = F * 0
            out.Ez = F * 0
            out.Hx = beta/(omega*mu_0) * F
            out.Hy = F * 0
            out.Hz = 1j/(omega*mu_0) * dyF
        else:
            out.Hx = F;
            out.Hy = F * 0;
            out.Hz = F * 0;
            out.Ex = F * 0;
            out.Ey = -beta/(omega*eps_0*n**2) * F
            out.Ez = -1j/(omega*eps_0*n**2) * dyF
        out.eps = self.n1**2 * (abs(y) <= self.w/2) + self.n0**2 * (y > self.w/2) + self.ns**2 * (y < -self.w/2)
        return out

class BandStructure(object):
    def _arrays(self):
        keys_arr = {'freqs', 'neff', 'ng', 'gmat', 'gobj', 'vg', 'ue', 'uh'}.intersection(self.__dict__.keys())
        return {key: self.__dict__[key] for key in keys_arr}
    def reorder(self, bands, notfound='warn'):
        """
        Reorders the bands according to some order.
        :param bands: List of bands.  Each element can be: 'ex', 'ey', 'ez', 'hx', 'hy', 'hz'
        :param notfound: What to do if a band is not found: 'none', 'warn', 'error'
        :return:
        """
        if len(bands) > len(self.freqs): raise ValueError("Too many bands")
        assert (self.freqs.ndim == 2)
        (ue, uh) = (False, False)
        for band in bands:
            if not (band[0] in ['e', 'h'] or len(set.difference(band[1:], 'xyz'))):
                raise ValueError("Unsupported band type " + str(band))
            elif (band[0] == 'e'): ue = True
            elif (band[0] == 'h'): uh = True
        out = BandStructure(); inp = self.__dict__; inp_arr = self._arrays(); outp = out.__dict__
        for key in inp_arr.keys(): outp[key] = inp[key][:len(bands)]*0 + np.nan
        out.k_points = self.k_points
        n_points = self.freqs.shape[1]
        (notfound_b, notfound_k) = (set(), set())
        if ue: indmax_e_bands = np.argmax(self.ue, axis=-1)
        if uh: indmax_h_bands = np.argmax(self.uh, axis=-1)
        # Loop over k-points.  For each k-point, loop through bands to find which matches
        # the desired output properties (ex, ey, etc.).
        for ik in range(n_points):
            inds = list(range(len(self.freqs)))
            for (ib, band) in enumerate(bands):
                indmax = [dict(x=0,y=1,z=2)[b] for b in band[1:]]
                indmax_bands = indmax_e_bands if (band[0] == 'e') else indmax_h_bands
                for ind in inds + [-1]:
                    if (ind == -1):
                        notfound_k.add(ik); notfound_b.add(ib)
                    elif (indmax_bands[ind, ik] in indmax):
                        for key in inp_arr.keys(): outp[key][ib, ik] = inp[key][ind, ik]
                        inds.remove(ind)
                        break
        if (len(notfound_b) > 0):
            msg = "Bands not found for {:d} bands on {:d} k-points".format(len(notfound_b), len(notfound_k))
            if (notfound == 'error'):
                raise KeyError(msg)
            elif (notfound == 'warn'):
                warnings.warn(msg)
        return out
    def load(self, filename):
        r"""
        Loads data from an HDF5 file.
        :param filename: Name of the file.
        :return: This object.  So that you can do things like: data = BandStructure().load("file.hdf5")
        """
        with h5py.File(filename, "r") as f:
            self.__dict__ = {}
            for key in f:
                self.__dict__[key] = f[key][:]
            return self
    def save(self, filename):
        r"""
        Saves data to an HDF5 file.
        :param filename: Name of the file.
        """
        with h5py.File(filename, "w") as f:
            for (key, val) in self.__dict__.items():
                f.create_dataset(key, data=val)
    def merge(self, *other, axis=0):
        r"""
        Merges multiple BandStructure instances.  Must have the same k_points.
        :param other: One or more additional band structure instances.
            Can also call as BandStructure.merge(*bsList, axis)
        :param axis: Axis to merge along, if axis >= 0.
            If axis = -1, create a new initial axis for merging.
        :return:
        """
        for other_i in other:
            assert ((other_i.k_points == self.k_points).all())
            assert (other_i.freqs.shape[:axis] == self.freqs.shape[:axis]
                    and other_i.freqs.shape[axis+1:] == self.freqs.shape[axis+1:])
        bsList = (self,) + other; keys = self._arrays().keys()
        out = BandStructure(); out.k_points = np.array(self.k_points)
        if axis == -1:
            for key in keys: out.__dict__[key] = np.array([bs.__dict__[key] for bs in bsList])
        elif axis >= 0:
            for key in keys: out.__dict__[key] = np.concatenate([bs.__dict__[key] for bs in bsList], axis=axis)
        else: raise ValueError("axis: " + str(axis))
        return out


def run_solver(ms, out=(), p=mp.NO_PARITY, objList=(), ind=(), verbose=False):
    r"""
    Run a solver and get some auxiliary data.
    :param ms: Mode solver object.
    :param out: things you can extract as outputs (choices: neff, ng, vg, gmat, gobj, ue, uh)
        neff, ng: effective & group index (along x).  vg: group velocity vector.
        gmat, gobj: E-field energy fraction in each material / in object list objList.
        uex, uey, uez: fraction of energy in Ex, Ey, Ez.
    :param p: Parity [mp.NO_PARITY, mp.EVEN_Z, etc.]
    :param objList: List of objects to calculate power inside
    :param ind: substitution rules for wavelength-dependent indices of refraction
    :param verbose: Whether to issue wordy MPB console output
    :return: An object containing all the output parameters.
    """
    if (out == 'all'): out = ('neff', 'ng', 'gmat', 'gobj', 'vg', 'ue', 'uh')
    if (type(out) == str): out = (out,)
    if (ind != ()): raise NotImplementedError()
    objList = [(x if np.iterable(x) else [x]) for x in objList]; out = ['freqs'] + list(out)
    getE = len(set.intersection({'gmat', 'gobj', 'ue'}, set(out))) > 0
    nList = list(set([obj.material.epsilon_diag[0] for obj in ms.geometry] +
                     [ms.default_material.epsilon_diag[0]]))
    epsTh = [0] + [(x1+x2)/2. for (x1, x2) in zip(nList[1:], nList[:-1])] + [1000]

    # Initialize output object to include desired properties.
    ans = BandStructure()
    ans.k_points = np.array(list(map(np.array, ms.k_points)))
    for lbl in out:
        if lbl in ['freqs', 'neff', 'ng']: size = (ms.num_bands, len(ms.k_points))
        elif lbl in ['gobj']: size = (ms.num_bands, len(ms.k_points), len(objList))
        elif lbl in ['gmat']: size = (ms.num_bands, len(ms.k_points), len(nList))
        elif lbl in ['vg', 'ue', 'uh']: size = (ms.num_bands, len(ms.k_points), 3)
        ans.__setattr__(lbl, np.zeros(size, dtype=np.float))

    # Callback function to set desired properties of output for each band.
    def callback(ms, band):
        ib = band-1; ik = ms.k_points.index(ms.current_k)
        # Group velocity and index
        if ('ng' in out): ans.ng[ib, ik] = 1/ms.compute_one_group_velocity_component(mp.Vector3(1,0,0), band)
        if ('vg' in out): ans.vg[ib, ik] = np.array(ms.compute_one_group_velocity(band))
        # Things that depend on the electric field
        if getE:
            ms.get_dfield(band); (U, xr, xi, yr, yi, zr, zi) = ms.compute_field_energy()
            if ('gmat' in out):
                for (imat, eps1, eps2) in zip(range(len(nList)), epsTh[:-1], epsTh[1:]):
                    ans.gmat[ib, ik, imat] = ms.compute_energy_in_dielectric(eps1, eps2)
            if ('gobj' in out):
                for (iobj, obj) in enumerate(objList):
                    ans.gobj[ib, ik, iobj] = ms.compute_energy_in_objects(obj)
            if ('ue' in out): ans.ue[ib, ik] = [xr+xi, yr+yi, zr+zi]
        # Things that depend on the magnetic field
        if ('uh' in out):
            ms.get_hfield(band); (U, xr, xi, yr, yi, zr, zi) = ms.compute_field_energy()
            ans.uh[ib, ik] = [xr+xi, yr+yi, zr+zi]

    # The function that runs MPB.
    def runfn():
        ms.run_parity(p, True, callback)
        freqs = ms.all_freqs.T; ans.freqs = freqs
        if ('neff' in out): ans.neff = np.outer(1, np.array([v.norm() for v in ms.k_points]))/np.array(freqs)

    # Run with / without wordy MPB console output.
    if verbose:
        runfn()
    else:
        blackhole = StringIO()
        with pipes(stdout=blackhole, stderr=STDOUT): runfn()

    return ans
