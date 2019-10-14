# -*- coding: utf-8 -*-
"""
RII_db_tools

Functions to serve as tools for parsing/accessing the RefractiveIndex.info database.
These tools should be able to calculate index, group velocity and other dispersion
related quantities from data in the RII database as well as tell the user what data
is available in the current database and the source of the data. If data from multiple
sources is available, the user should be able to choose the source.


Created on Sat Oct 25 14:47:55 2014

@author: dodd
"""
import os
import yaml
import numpy as np
import sympy as sp
from scipy import interpolate
from instrumental import Q_, u
from bs4 import BeautifulSoup as bs
from pathlib import Path
home = str( Path.home() )
RII_db_dir = home + '/github/refractiveindex.info-database/database/data'
if not os.path.isdir(RII_db_dir):
    print('Warning: refractiveindex.info database directory:')
    print(RII_db_dir)
    print('does not exist. For RII_db_tools to work please download RII database to this location or change RII_db_dir value in RII_db_tools.py')


def check_db(material,book='main'):
    """ Function to return available sources, data types and wavelength ranges of index data
    for a given material. The material is input as a string (ex 'GaAs', 'Al2O3'). The material
    catageory must also be put in as a string ('glass', 'main', 'organic', 'other')."""

    if book not in ['glass', 'main', 'organic', 'other']:
        raise Exception('invalid material category. choose glass, main, organic or other')

    material_dir = os.path.join(RII_db_dir,book,material)

    if not os.path.isdir(material_dir):
        raise Exception('material directory not found')

    for filename in os.listdir(material_dir):
        print('################################################## \n')
        file_path = os.path.join(material_dir,filename)
        file_contents = yaml.load(open(file_path,'r'), Loader=yaml.SafeLoader)
        print('filename: {0}\n'.format(filename))
        file_refs_md = file_contents['REFERENCES']
        file_refs = ''.join(bs(file_refs_md).findAll(text=True))
        print(u'\t reference: {0}\n'.format(file_refs))
        file_data_type = file_contents['DATA'][0]['type']
        print('\t data type: {0}\n'.format(file_data_type))
        if 'wavelength_range' in file_contents['DATA'][0]:
            #file_data_range = map(float,file_contents['DATA'][0]['wavelength_range'].split())
            file_data_range = [float(j) for j in file_contents['DATA'][0]['wavelength_range'].split()]
            print('\t data range [um]: {0}\n'.format(file_data_range))
        elif file_data_type == 'tabulated nk':
            lm, n, k = n_k_tabulated(material,filename,book)
            file_data_range = [min(lm),max(lm)]
            print('\t data range [um]: {0}\n'.format(file_data_range))
        if 'COMMENTS' in file_contents:
            file_comments = file_contents['COMMENTS']
            print(u'\t comments: {0}\n'.format(file_comments))

    print('################################################## \n')

def material_datasets(material,book='main'):
    """ This function returns a list of dataset names available for a requested
    material."""

    if book not in ['glass', 'main', 'organic', 'other']:
        raise Exception('invalid material category. choose glass, main, organic or other')

    material_dir = os.path.join(RII_db_dir,book,material)

    if not os.path.isdir(material_dir):
        raise Exception('material directory not found')

    filenames = []

    for filename in os.listdir(material_dir):
        filenames.append(filename)

    return filenames


def dataset_type(material, dataset, book='main'):
    if book not in ['glass', 'main', 'organic', 'other']:
        raise Exception('invalid material category. choose glass, main, organic or other')

    material_dir = os.path.join(RII_db_dir,book,material)

    if not os.path.isdir(material_dir):
        raise Exception('material directory not found')

    dataset_path = os.path.join(material_dir,dataset)

    if not os.path.exists(dataset_path):
        raise Exception('material found, but requested dataset not found')

    dataset_contents = yaml.load(open(dataset_path,'r'), Loader=yaml.SafeLoader)
    dataset_type = dataset_contents['DATA'][0]['type']
    return dataset_type

def dataset_range(material, dataset, book='main'):
    if book not in ['glass', 'main', 'organic', 'other']:
        raise Exception('invalid material category. choose glass, main, organic or other')

    material_dir = os.path.join(RII_db_dir,book,material)

    if not os.path.isdir(material_dir):
        raise Exception('material directory not found')

    dataset_path = os.path.join(material_dir,dataset)

    if not os.path.exists(dataset_path):
        raise Exception('material found, but requested dataset not found')

    dataset_contents = yaml.load(open(dataset_path,'r'), Loader=yaml.SafeLoader)
    dataset_type = dataset_contents['DATA'][0]['type']
    if 'wavelength_range' in dataset_contents['DATA'][0]:
        #data_range = map(float,dataset_contents['DATA'][0]['wavelength_range'].split())
        data_range = [float(j) for j in dataset_contents['DATA'][0]['wavelength_range'].split()]
        return data_range
    elif (dataset_type == 'tabulated nk') or (dataset_type == 'tabulated n'):
        lm, n, k = n_k_tabulated(material,dataset,book)
        data_range = [min(lm),max(lm)]
        return data_range
    else:
        raise Exception('could not find data range in file')


def dataset_comments(material, dataset, book='main'):
    if book not in ['glass', 'main', 'organic', 'other']:
        raise Exception('invalid material category. choose glass, main, organic or other')

    material_dir = os.path.join(RII_db_dir,book,material)

    if not os.path.isdir(material_dir):
        raise Exception('material directory not found')

    dataset_path = os.path.join(material_dir,dataset)

    if not os.path.exists(dataset_path):
        raise Exception('material found, but requested dataset not found')

    dataset_contents = yaml.load(open(dataset_path,'r'), Loader=yaml.SafeLoader)
    try:
        dataset_comments = dataset_contents['COMMENTS']
    except:
        dataset_comments = ''
    return dataset_comments


def index_coeffs(material, dataset, book='main'):
    """This function returns index formula (Sellmeier Eq. or other) coefficients
    and type for a specificed material if the specified material and dataset are present
    in the RII database and if the specified dataset is of the correct type."""

    if book not in ['glass', 'main', 'organic', 'other']:
        raise Exception('invalid material category. choose glass, main, organic or other')

    material_dir = os.path.join(RII_db_dir,book,material)

    if not os.path.isdir(material_dir):
        raise Exception('material directory not found')

    dataset_path = os.path.join(material_dir,dataset)

    if not os.path.exists(dataset_path):
        raise Exception('material found, but requested dataset not found')

    dataset_contents = yaml.load(open(dataset_path,'r'), Loader=yaml.SafeLoader)
    dataset_type = dataset_contents['DATA'][0]['type']
    dataset_type_split = dataset_contents['DATA'][0]['type'].split()

    if not dataset_type_split[0]=='formula':
        raise Exception('dataset type is {0}. datatype should be "formula n" with 1<n<9'.format(dataset_type))

    formula = int(dataset_type_split[1])

    #coeffs = np.array(map(float,dataset_contents['DATA'][0]['coefficients'].split()))
    coeffs = np.array(dataset_contents['DATA'][0]['coefficients'].split(),dtype='float')

    coeffs_padded = np.lib.pad(coeffs,(0,17-np.size(coeffs)),'constant',constant_values=(0,0))

    return formula, coeffs_padded



def n_k_tabulated(material, dataset, book='main'):
    """This function returns tabulated index (n) and extinction coefficient (k)
    vs. wavelength (in um) for a specificed material if the specified material
    and dataset are present in the RII database and if the specified dataset is
    of the correct type."""

    if book not in ['glass', 'main', 'organic', 'other']:
        raise Exception('invalid material category. choose glass, main, organic or other')

    material_dir = os.path.join(RII_db_dir,book,material)

    if not os.path.isdir(material_dir):
        raise Exception('material directory not found')

    dataset_path = os.path.join(material_dir,dataset)

    if not os.path.exists(dataset_path):
        raise Exception('material found, but requested dataset not found')

    dataset_contents = yaml.load(open(dataset_path,'r'), Loader=yaml.SafeLoader)
    dataset_type = dataset_contents['DATA'][0]['type']

    if not ((dataset_type == 'tabulated nk') or (dataset_type == 'tabulated n')):
        raise Exception('dataset type is {0}. datatype should be "tabulated nk" or "tabulated n"'.format(dataset_type))

    if dataset_type == 'tabulated nk':
        nk_data = [float(j) for j in dataset_contents['DATA'][0]['data'].split()]
        lm = nk_data[0::3]
        n = nk_data[1::3]
        k = nk_data[2::3]
    else:
        n_data = [float(j) for j in dataset_contents['DATA'][0]['data'].split()]
        lm = n_data[0::2]
        n = n_data[1::2]
        k = [0 for n_val in n]
    return lm, n, k



lm, c1, c2, c3, c4, c5, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17 = sp.symbols('lm c1 c2 c3 c4 c5 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17')

def n_symb(formula,coeffs):
    """Define and return sympy representaton of index model based on
    the specified formula and coefficients."""

    if formula==1:    # Smellmeier Ver. 1
        n = sp.sqrt(1 + c1 + (c2*lm**2)/(lm**2-c3**2) + \
                    (c4*lm**2)/(lm**2-c5**2) + \
                    (c6*lm**2)/(lm**2-c7**2) + \
                    (c8*lm**2)/(lm**2-c9**2) + \
                    (c10*lm**2)/(lm**2-c11**2) + \
                    (c12*lm**2)/(lm**2-c13**2) + \
                    (c14*lm**2)/(lm**2-c15**2) + \
                    (c16*lm**2)/(lm**2-c17**2) )
    elif formula==2:    # Smellmeier Ver. 2
        n = sp.sqrt(1 + c1 + (c2*lm**2)/(lm**2-c3) + \
                    (c4*lm**2)/(lm**2-c5) + \
                    (c6*lm**2)/(lm**2-c7) + \
                    (c8*lm**2)/(lm**2-c9) + \
                    (c10*lm**2)/(lm**2-c11) + \
                    (c12*lm**2)/(lm**2-c13) + \
                    (c14*lm**2)/(lm**2-c15) + \
                    (c16*lm**2)/(lm**2-c17) )
    elif formula==3:    # Polynomial
        n = sp.sqrt(    c1 + c2*(lm**c3) + \
                        c4*(lm**c5) + \
                        c6*(lm**c7) + \
                        c10*(lm**c9) + \
                        c12*(lm**c13) + \
                        c14*(lm**c15) + \
                        c16*(lm**c17) )
    elif formula==4:    # RII formula
                        # Is this wrong? why are coefficients raised to other coefficients?
        n = sp.sqrt(    c1 + (c2*lm**c3)/(lm**2-c4**c5) + \
                        (c6*lm**c7)/(lm**2-c8**c9) + \
                        c10*(lm**c9) + \
                        c12*(lm**c13) + \
                        c14*(lm**c15) + \
                        c16*(lm**c17) )
    elif formula==5:    # Cushy Cauchy
        n = c1 + c2*(lm**c3) + \
            c4*(lm**c5) + \
            c6*(lm**c7) + \
            c10*(lm**c9) + \
            c12*(lm**c13) + \
            c14*(lm**c15) + \
            c16*(lm**c17)
    elif formula==6:    # Gassy Gasses
        n = 1 + c1 + c2/(c3-(lm**-2)) + \
                c4/(c5-(lm**-2)) + \
                c6/(c7-(lm**-2)) + \
                c8/(c9-(lm**-2)) + \
                c10/(c11-(lm**-2)) + \
                c12/(c13-(lm**-2)) + \
                c14/(c15-(lm**-2)) + \
                c16/(c17-(lm**-2))
    elif formula==7:    # Double Bacon Herzberger
        n = c1 + c2*(1/(lm**2-0.028)) + \
            c3*(1/(lm**2-0.028))**2 + \
            c4*lm**2 + \
            c5*lm**4 + \
            c6*lm**6
    elif formula==8:    # Retro (so whack)
        rhs =   c1 + c2*lm**2/(lm**2 - c3) + c4*lm**2
        n = np.sqrt( (1+2*rhs) / (1-rhs) )
    elif formula==9:    # Exotic!
        n = sp.sqrt(    c1 + c2/(lm**2-c3) + \
                        (c4*(lm-c5)/((lm-c5)**2+c6) ) )
    else:
        raise Exception('formula type not found, must be 1-9')

    n_subs = n.subs( [  (c1,coeffs[0]),(c2,coeffs[1]),(c3,coeffs[2]),(c4,coeffs[3]), \
                            (c5,coeffs[4]),(c6,coeffs[5]),(c7,coeffs[6]),(c8,coeffs[7]), \
                            (c9,coeffs[8]),(c10,coeffs[9]),(c11,coeffs[10]),(c12,coeffs[11]), \
                            (c13,coeffs[12]),(c14,coeffs[13]),(c15,coeffs[14]),(c16,coeffs[15]), \
                            (c17,coeffs[16]) ] )
    return n_subs

def from_url(url):
    """
    Return tuple of strings (book, material, dataset) extracted from
    RII url.
    """
    book, material, page = tuple([x.split('=')[1] for x in url.split('?')[1].split('&')])
    dataset = page + '.yml'
    return material, dataset, book

def n_model(material="",dataset="",lm_range = 0, book="main",formula=0,coeffs=0,verbose=False):
    """ Return a lambdified model of a material's index based on
        a symbolic representation created using coefficients and a
        formula from RII."""

    dataset_lm_range = 0 #indicating dataset range is unknown, reset later if known

    if formula==0 and coeffs==0:
        if material[0:5]=='https':
            material,dataset,book = from_url(material)
        elif dataset=="":
            datasets = material_datasets(material,book)
            if lm_range==0: usable_datasets = datasets
            else:
                lm_range_micron = Q_(lm_range).to('um').magnitude
                lm_max = np.max(lm_range_micron)
                lm_min = np.min(lm_range_micron)
                usable_datasets = filter((lambda x: np.max(dataset_range(material,x))>=lm_max and np.min(dataset_range(material,x))<=lm_min ),datasets)
                if len(usable_datasets)==0:
                    raise Exception('no suitable datasets found for desired wavelength range')
            def f_range_mag(x):
                return max(dataset_range(material,x)) - min(dataset_range(material,x))
            #range_magnitudes = map((lambda x:  max(dataset_range(material,x)) - min(dataset_range(material,x))), usable_datasets)
            range_magnitudes = [f_range_mag(j) for j in usable_datasets]
            dataset = usable_datasets[range_magnitudes.index(max(range_magnitudes))]

        dataset_lm_range = dataset_range(material,dataset,book)
        dataset_comms = dataset_comments(material,dataset,book)
        if ((dataset_type(material,dataset,book) =='tabulated nk') or (dataset_type(material,dataset,book) =='tabulated n')):
            lm_tab, n_tab, k_tab = n_k_tabulated(material,dataset,book)
            tck_n = interpolate.splrep(lm_tab,n_tab,s=0)
            tck_k = interpolate.splrep(lm_tab,k_tab,s=0)
            def n_interp_unitful(lm_unitful):
                lm_in = Q_(lm_unitful).to('um').magnitude
                n_r = interpolate.splev(lm_in,tck_n,der=0)
                n_im = interpolate.splev(lm_in,tck_k,der=0)
                n = n_r + n_im*1j
                if dataset_lm_range and not(max(lm_in)<=max(dataset_lm_range) and min(lm_in)>=min(dataset_lm_range)):
                    raise Exception('input wavelength outside of dataset validity range')
                return n
            if verbose:
                print('Using dataset {}'.format(dataset))
                print('wavelength range: {}'.format(dataset_lm_range))
                print(u'comments: {}'.format(dataset_comms))
            return n_interp_unitful
        formula, coeffs = index_coeffs(material,dataset,book)
    n_s = n_symb(formula,coeffs)
    n = sp.lambdify([lm],n_s,'numpy')

    def n_unitful(lm_unitful):
        lm_in = Q_(lm_unitful).to('um').magnitude
        if dataset_lm_range and not(max(lm_in)<=max(dataset_lm_range) and min(lm_in)>=min(dataset_lm_range)):
                    raise Exception('input wavelength outside of dataset validity range')
        return n(lm_in)
    if (dataset and verbose):
        print('Using dataset {}'.format(dataset))
    if (dataset_lm_range and verbose):
        print('wavelength range: {}'.format(dataset_lm_range))
    if (dataset_comms and verbose):
        print(u'comments: {}'.format(dataset_comms))

    return n_unitful

def n_g_model(material="",dataset="",book="main",formula=0,coeffs=0,verbose=False,smoothing=0):
    """ Return a lambdified model of a material's group index based on
        a symbolic representation created using coefficients and a
        formula from RII."""
    # case 0: tabulated data rather than Sellmeier form
    if ((dataset_type(material,dataset,book) =='tabulated nk') or (dataset_type(material,dataset,book) =='tabulated n')):
        dataset_lm_range = dataset_range(material,dataset,book)
        lm_tab, n_tab, k_tab = n_k_tabulated(material,dataset,book)
        ω_tab = (lm_tab*u.um).to(1/u.s,'sp') # tabulated RII lambda values are in microns
        tck_n = interpolate.splrep(ω_tab[::-1].m,n_tab[::-1],s=smoothing) # spline routines require input x-values sorted largest to smallest
        def n_interp_unitful(lm_unitful):
            ω_in_Hz = lm_unitful.to(1/u.s,'sp')[::-1].m # convert input wavelengths to frequencies in Hz, order largest to smallest
            lm_in = Q_(lm_unitful).to('um').magnitude
            n_interp = interpolate.splev(ω_in_Hz,tck_n,der=0)
            dn_dω_interp = interpolate.splev(ω_in_Hz,tck_n,der=1)
            ng_interp = n_interp + ω_in_Hz * dn_dω_interp
            if dataset_lm_range and not(max(lm_in)<=max(dataset_lm_range) and min(lm_in)>=min(dataset_lm_range)):
                print('Warning: input wavelength outside or at boundary of dataset validity range')
            return ng_interp[::-1]
        if verbose:
            print('Using dataset {}'.format(dataset))
            print('wavelength range: {}'.format(dataset_lm_range))
            print(u'comments: {}'.format(dataset_comms))
        return n_interp_unitful
    # case 1: Sellmeier form
    if formula==0 and coeffs==0:
        formula, coeffs = index_coeffs(material,dataset,book)
    n_s = n_symb(formula,coeffs)
    n_s_prime = sp.diff(n_s,lm)
    n_g_s = n_s - lm*n_s_prime
    n_g = sp.lambdify([lm],n_g_s,'numpy')

    def n_g_unitful(lm_unitful):
        lm_in = Q_(lm_unitful).to('um').magnitude
        return n_g(lm_in)

    return n_g_unitful

def gvd_model(material="",dataset="",book="main",formula=0,coeffs=0):
    """ Return a lambdified model of a material's group velocity dispersion
        based on a symbolic representation created using coefficients and a
        formula from RII."""
    if formula==0 and coeffs==0:
        formula, coeffs = index_coeffs(material,dataset,book)
    n_s = n_symb(formula,coeffs)
    n_s_double_prime = sp.diff(n_s,lm,lm)
    #c = 3e-4 # speed of light in [mm/fs]
    #gvd_s = 1e-3*(lm**3)/(2*np.pi*(c**2))*n_s_double_prime # should give gvd in [fs**2/cm] for lm in [um], still need to make unitful with pint
    c = Q_(3e8,'m/s') # unitful definition of speed of light
    gvd_s_no_prefactors = (lm**3)*n_s_double_prime # symbolic gvd without unitful prefactors, to be made unitful below
    gvd_no_prefactors = sp.lambdify([lm],gvd_s_no_prefactors,'numpy') # numerical gvd without unitful prefactors

    def gvd_unitful(lm_unitful):
        lm_in = Q_(lm_unitful).to('um').magnitude
        gvd = (1 / (2 * np.pi * (c**2))) * Q_(gvd_no_prefactors(lm_in),'um')
        return gvd.to('fs**2 / mm')

    return gvd_unitful
