# -*- coding: utf-8 -*-
# copyright Dodd Gray 2020

import emopt
from emopt.misc import NOT_PARALLEL

import numpy as np
from math import pi

from petsc4py import PETSc
from mpi4py import MPI

####################################################################################
# Simulation parameters
####################################################################################
X = 5.0
Y = 3.0
Z = 2.5
dx = 0.04
dy = 0.04
dz = 0.04

wavelength = 1.55

#####################################################################################
# Setup simulation
#####################################################################################
sim = emopt.fdtd.FDTD(X,Y,Z,dx,dy,dz,wavelength, rtol=1e-5)
w_pml = dx * 15
sim.w_pml = [w_pml, w_pml, w_pml, w_pml, w_pml, w_pml]

X = sim.X
Y = sim.Y
Z = sim.Z

Nx = sim.Nx
Ny = sim.Ny
Nz = sim.Nz

#####################################################################################
# Define the geometry/materials
#####################################################################################
r1 = emopt.grid.Rectangle(X/2, Y/2, 2*X, 0.5); r1.layer = 1
r2 = emopt.grid.Rectangle(X/2, Y/2, 2*X, 2*Y); r2.layer = 2

r1.material_value = 3.45**2
r2.material_value = 1.444**2

eps = emopt.grid.StructuredMaterial3D(X, Y, Z, dx, dy, dz)
eps.add_primitive(r2, -Z, Z)
eps.add_primitive(r1, Z/2-0.11, Z/2+0.11)

mu = emopt.grid.ConstantMaterial3D(1.0)

sim.set_materials(eps, mu)
sim.build()

#####################################################################################
# Setup the sources
#####################################################################################
mode_slice = emopt.misc.DomainCoordinates(0.8, 0.8, w_pml, Y-w_pml, w_pml, Z-w_pml, dx, dy, dz)

mode = emopt.modes.ModeFullVector(wavelength, eps, mu, mode_slice, n0=3.45,
                                   neigs=4)
mode.build()
mode.solve()

# get the result
Ez = mode.get_field_interp('Ez')
