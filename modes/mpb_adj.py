# -*- coding: utf-8 -*-


import meep as mp
from meep import mpb
import numpy as np
import autograd.numpy as npa
from autograd import grad, jacobian
from collections import namedtuple
from wurlitzer import pipes, STDOUT
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec

Grid = namedtuple('Grid', ['x', 'y', 'z', 'w'])

"""
Autodiff-compatible functions for use inside 2D MPB solver
"""
def curl_2D(V,x,y,kz,a=1):
    """
    Calculate and return the curl of the 2D 3-vector field V(x,y) = [Vx(x,y),Vy(x,y),Vz(x,y)]
    for MPB scaled spatial coordiantes x and y with scaling factor a (= 1/(2*kz) ?)
    and an assumed spatial period along z of kz
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dVx_dx = npa.gradient(V[:,:,0],axis=0) / ( dx / a )
    dVy_dx = npa.gradient(V[:,:,1],axis=0) / ( dx / a )
    dVz_dx = npa.gradient(V[:,:,2],axis=0) / ( dx / a )
    dVx_dy = npa.gradient(V[:,:,0],axis=1) / ( dy / a )
    dVy_dy = npa.gradient(V[:,:,1],axis=1) / ( dy / a )
    dVz_dy = npa.gradient(V[:,:,2],axis=1) / ( dy / a )

    curl_V = npa.stack([dVz_dy - 1j * kz * V[:,:,1],
                       1j * kz * V[:,:,0] - dVz_dx,
                       dVy_dx - dVx_dy,
                      ],
                      axis=-1)
    return curl_V

def D_2D(H,ω,x,y,kz,a=1):
    return (1j / ω) *  curl_2D(H,x,y,kz,a=a)

def E_2D(H,ω,eps,x,y,kz,a=1):
    return (1j / ω) *  curl_2D(H,x,y,kz,a=a) / npa.reshape(eps,npa.shape(eps)+(1,))

def S_2D(H,ω,eps,x,y,kz,a=1):
    return npa.cross(npa.conjugate(E_2D(H,ω,eps,x,y,kz,a=a)),H)

def U_2D(H,ω,eps,x,y,kz,a=1):
    eps_rs = npa.reshape(eps,npa.shape(eps)+(1,))
    return npa.sum(eps_rs*npa.abs(E_2D(H,ω,eps,x,y,kz,a=a))**2+npa.abs(H)**2,axis=2)

def ng_2D(H,ω,eps,x,y,kz,a=1):
    return 0.5 / ( npa.sum(npa.real(S_2D(H,ω,eps,x,y,kz,a=a)[:,:,2]),axis=(0,1)) / npa.sum(U_2D(H,ω,eps,x,y,kz,a=a),axis=(0,1)) )

grad_ng_2D = grad(ng_2D,(0,1,2))

# def ng_2D(H,ω,eps,x,y,kz,a=1):
#     #V = npa.zeros_like(H)
#     Vx = -1j* H[:,:,1]
#     Vy = 1j* H[:,:,0]
#     Vz = npa.zeros_like(H[:,:,0])
#     V = npa.stack([Vx,Vy,Vz,],axis=2)
#     eps_rs = npa.reshape(eps,npa.shape(eps)+(1,))
#     X = curl_2D(V/eps_rs,x,y,kz,a=a)
#     Z = npa.conjugate(H)*X
#     dx = x[1] - x[0]
#     dy = y[1] - y[0]
#     Zint = npa.sum(npa.real(Z),axis=(0,1,2)) * (dx * dy)
# #     Zint = integrate.simps(integrate.simps(np.sum(Z,axis=2).real,x=y,axis=1),x=x,axis=0)
#     return ((1 / (Zint )) * ω)



class OptimizationProblem(object):
    """Top-level class in the MPB adjoint module, adapted from Alec Hammond's
    adjoint module for MEEP. WIP

    Intended to be instantiated from user scripts with mandatory constructor
    input arguments specifying the data required to define an adjoint-based
    optimization.

    The class knows how to do one basic thing: Given an input vector
    of design variables, compute the objective function value (forward
    calculation) and optionally its gradient (adjoint calculation).
    This is done by the __call__ method.

    """

    def __init__(self,
                modesolver,
                objective_functions,
                objective_arguments,
                design_variables,
                tracked_mode_ind=0,
                parity=None,
                frequencies=None,
                fcen=None,
                df=None,
                nf=None,
                x_grid_offset=1,
                y_grid_offset=1,
                verbose=False,
                 ):

        self.ms = modesolver
        if not parity:
            self.parity = mp.NO_PARITY
        else:
            self.parity = parity
        self.verbose = verbose
        self.n_modes = self.ms.num_bands
        self.band_funcs = [mpb.fix_efield_phase,
                           mpb.output_efield,
                           mpb.output_hfield,
                           mpb.output_dpwr,
                           mpb.output_poynting,
                           mpb.output_tot_pwr,
                           #mpb.output_dpwr_in_objects,
                          ]
        self.k_points = np.array([kk[2] for kk in self.ms.k_points])
        self.epsilon = np.array(self.ms.get_epsilon())
        self.res = self.ms.resolution
        self.Xgrid = self.ms.geometry_lattice.size[0]
        self.Ygrid = self.ms.geometry_lattice.size[1]
        self.Zgrid = self.ms.geometry_lattice.size[2]
        self.shape = self.epsilon.shape
        self.Nx = self.shape[0]
        self.Ny = self.shape[1]
        try:
            self.Nz = self.shape[2]
        except:
            self.Nz = 1
        self.N = self.Nx * self.Ny * self.Nz

        self.x = self.Xgrid * np.linspace(-1/2., 1/2., self.Nx)
        self.y = self.Ygrid * np.linspace(-1/2., 1/2., self.Ny)
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]
        # self.z =
        self.x_grid_offset = x_grid_offset
        self.y_grid_offset = y_grid_offset
        self.tracked_mode_ind = tracked_mode_ind
        if isinstance(objective_functions, list):
            self.objective_functions = objective_functions
        else:
            self.objective_functions = [objective_functions]
        self.objective_arguments = objective_arguments
        self.f_bank = [] # objective function evaluation history

        if isinstance(design_variables, list):
            self.design_variables = design_variables
        else:
            self.design_variables = [design_variables]

        self.num_design_params = [ni.num_design_params for ni in self.design_variables]
        self.design_regions = [dr.volume for dr in self.design_variables]
        self.num_design_regions = len(self.design_regions)

        self.design_region_inds = [self.dv_inds(self.design_variables[i]) for i in range(self.num_design_regions)]
        self.design_region_inds_vec = [self.dv_inds_vec(self.design_variables[i]) for i in range(self.num_design_regions)]
        # # store sources for finite difference estimations
        # self.forward_sources = self.ms.sources

        # The optimizer has three allowable states : "INIT", "SOLVE", and "ADJ".
        #    INIT - The optimizer is initialized and ready to run a mode solve
        #    SOLVE  - The optimizer has already run a mode solve
        #    ADJ  - The optimizer has already run an adjoint simulation (but not yet calculated the gradient)
        self.current_state = "INIT"




    def dv_inds(self,dv):
        xinds_dv = np.zeros(dv.Nx,dtype='int')
        yinds_dv = np.zeros(dv.Ny,dtype='int')
        # zinds_dv = np.zeros(dv.Nz,dtype='int')
        for xind_dr,xx_dr in enumerate(dv.rho_x):
            xx_dr_eff =  xx_dr
            if xind_dr + 1 == len(dv.rho_x):
                xinds_dv[xind_dr] = int(np.argmin(np.abs(self.x-xx_dr_eff)))
            elif xind_dr == 0:
                xinds_dv[xind_dr] = int(np.argmin(np.abs(self.x-xx_dr_eff))) + 2
            else:
                xinds_dv[xind_dr] = int(np.argmin(np.abs(self.x-xx_dr_eff))) + 1

        for yind_dr,yy_dr in enumerate(dv.rho_y):
            yy_dr_eff = yy_dr
            if yind_dr + 1 == len(dv.rho_y):
                yinds_dv[yind_dr] = int(np.argmin(np.abs(self.y-yy_dr_eff)))
            elif yind_dr == 0:
                yinds_dv[yind_dr] = int(np.argmin(np.abs(self.y-yy_dr_eff))) + 2
            else:
                yinds_dv[yind_dr] = int(np.argmin(np.abs(self.y-yy_dr_eff))) + 1
        # for zind_dr,z_dr in enumerate(dv.rho_z):
        #     zinds_dv[zind_dr] = int(np.argmin(np.abs(self.z-z_dr)))
        return xinds_dv,yinds_dv # ,zinds_dv

    def dv_inds_vec(self,dv):
        # xinds_dv,yinds_dv,zinds_dv = self.dv_inds(dv)
        xinds_dv,yinds_dv = self.dv_inds(dv)
        xinds_dv_vec,yinds_dv_vec = np.meshgrid(xinds_dv,yinds_dv,indexing='ij')
        xinds_dv_vec = xinds_dv_vec.flatten()
        yinds_dv_vec = yinds_dv_vec.flatten()
        # zinds_dv_vec = zinds_dv_vec.T.flatten() + 1
        return xinds_dv_vec,yinds_dv_vec # ,zinds_dv_vec

    def __call__(self, rho_vector=None, need_value=True, need_gradient=True):
        """Evaluate value and/or gradient of objective function.
        """
        if rho_vector:
            self.update_design(rho_vector=rho_vector)

        # Run solve if requested
        if need_value and self.current_state == "INIT":
            print("Starting solve...")
            self.solve()

        # Run adjoint simulation and calculate gradient if requested
        if need_gradient:
            if self.current_state == "INIT":
                # we need to run a solve before an adjoint run
                print("Starting solve...")
                self.solve()
                # print("Starting adjoint run...")
                # self.a_E = []
                # for ar in range(len(self.objective_functions)):
                #     self.adjoint_run(ar)
                print("Calculating gradient...")
                self.calculate_gradient()
            elif self.current_state == "SOLVE":
                # print("Starting adjoint run...")
                # self.a_E = []
                # for ar in range(len(self.objective_functions)):
                #     self.adjoint_run(ar)
                print("Calculating gradient...")
                self.calculate_gradient()
            else:
                raise ValueError("Incorrect solver state detected: {}".format(self.current_state))

        # clean up design grid list object
        dg = self.design_grids[0] if len(self.design_grids) == 1 else self.design_grids

        return self.f0, self.gradient, dg

    def get_fdf_funcs(self):
        """construct callable functions for objective function value and gradient

        Returns
        -------
        2-tuple (f_func, df_func) of standalone (non-class-method) callables, where
            f_func(beta) = objective function value for design variables beta
           df_func(beta) = objective function gradient for design variables beta
        """

        def _f(x=None):
            (fq, _) = self.__call__(rho_vector = x, need_gradient = False)
            return fq

        def _df(x=None):
            (_, df) = self.__call__(need_value = False)
            return df

        return _f, _df

    def prepare_solve(self):
        # prepare solve
        if self.verbose:
            self.ms.init_params(self.parity, False)
        else:
            blackhole = StringIO()
            with pipes(stdout=blackhole, stderr=STDOUT):
                self.ms.init_params(self.parity, False)


        # store design region voxel parameters
        #self.design_grids = [Grid(*self.ms.get_array_metadata(dft_cell=drm)) for drm in self.design_region_monitors]

    def solve(self):
        # set up monitors
        self.prepare_solve()

        # self.results = {"E":np.zeros((self.nx,self.ny,self.nz,3,self.n_modes),dtype='complex'),
        #                "H":np.zeros((self.nx,self.ny,self.nz,3,self.n_modes),dtype='complex'),
        #                "D":np.zeros((self.nx,self.ny,self.nz,3,self.n_modes),dtype='complex'),
        #                "Dpwr":np.zeros((self.nx,self.ny,self.nz,3,self.n_modes),dtype='float'),
        #                "ng":np.zeros((self.n_modes),dtype='float'),
        #                # "ng_nodisp":np.zeros((self.n_modes),dtype='float'),
        #                # "band":np.zeros((self.n_modes),dtype='int'),
        #                # "pol":["unknown" for mind in range(self.n_modes)],
        #             }
        self.results = {}


        # solve
        if self.verbose:
            self.ms.run(*self.band_funcs)
        else:
            blackhole = StringIO()
            with pipes(stdout=blackhole, stderr=STDOUT):
                self.ms.run(*self.band_funcs)

        # record objective quantities

        # for m in self.objective_arguments:
        #     self.results_list.append(m())
        self.E = np.stack([np.array(self.ms.get_efield(mind+1)) for mind in range(self.n_modes)],axis=-1)
        self.D = np.stack([np.array(self.ms.get_dfield(mind+1)) for mind in range(self.n_modes)],axis=-1)
        self.H = np.stack([np.array(self.ms.get_hfield(mind+1)) for mind in range(self.n_modes)],axis=-1)
        self.S = np.stack([np.array(self.ms.get_poynting(mind+1)) for mind in range(self.n_modes)],axis=-1)
        self.U = np.stack([np.array(self.ms.get_tot_pwr(mind+1)) for mind in range(self.n_modes)],axis=-1)
        self.ω = np.array(self.ms.freqs)
        self.neff = self.k_points/self.ω
        # compute geometric dispersion contribution to group velocity
        self.ng_mpb = np.array([1 / self.ms.compute_one_group_velocity_component(mp.Vector3(0, 0, 1), band_ind+1) for band_ind in range(self.n_modes)])
        self.ng_ag = np.array([ng_2D(self.H[:,:,0,:,band_ind],
                                    self.ω[band_ind],
                                    self.epsilon,
                                    self.x,
                                    self.y,
                                    self.k_points[0],
                                    a=( 1 / self.k_points[0] / 2 ), # TODO: figure out what/why this value of "a" works
                                    ) for band_ind in range(self.n_modes)])
        # compute material dispersion contribution to group velocity
        # ng = ng_nodisp * (p_mat_core * (ng_core / n_core) + (1 - p_mat_core) * (ng_clad / n_clad))
        self.vecs = self.ms.get_eigenvectors(1,self.n_modes)

        # results_list = [self.ω[1]**2,]
        results_list = [ self.ω**2, self.ng_ag ]

        # evaluate objectives
        self.f0 = [fi(*results_list) for fi in self.objective_functions]
        if len(self.f0) == 1:
            self.f0 = self.f0[0]
        # # Store forward fields for each set of design variables in array (x,y,z,field_components,frequencies)
        # self.d_E = [np.zeros((len(dg.x),len(dg.y),len(dg.z),3,self.nf),dtype=np.complex128) for dg in self.design_grids]
        # for nb, dgm in enumerate(self.design_region_monitors):
        #     for f in range(self.nf):
        #         for ic, c in enumerate([mp.Ex,mp.Ey,mp.Ez]):
        #             self.d_E[nb][:,:,:,ic,f] = atleast_3d(self.ms.get_dft_array(dgm,c,f))

        # store objective function evaluation in memory
        self.f_bank.append(self.f0)

        # update solver's current state
        self.current_state = "SOLVE"


    def calculate_gradient(self):
        # Iterate through all design region bases and store gradient w.r.t. permittivity
        #self.gradient = [[2*np.sum(np.real(self.a_E[ar][nb]*self.d_E[nb]),axis=(3)) for nb in range(self.num_design_regions)] for ar in range(len(self.objective_functions))]


        self.grad_H = np.squeeze(np.zeros_like(self.H))
        self.grad_ω = np.zeros_like(self.ω)
        self.grad_eps_ng = np.zeros((self.Nx,self.Ny,self.n_modes))
        for band_ind in range(self.n_modes):
            grad_H_temp, grad_ω_temp, grad_eps_temp = grad_ng_2D(self.H[:,:,0,:,band_ind],
                                                                        self.ω[band_ind],
                                                                        self.epsilon,
                                                                        self.x,
                                                                        self.y,
                                                                        self.k_points[0],
                                                                        a=( 1 / self.k_points[0] / 2 ), # TODO: figure out what/why this value of "a" works
                                                                        )
            self.grad_H[:,:,:,band_ind] = grad_H_temp
            self.grad_ω[band_ind] = grad_ω_temp
            self.grad_eps_ng[:,:,band_ind] = grad_eps_temp  #np.sum(grad_eps_temp,axis=-1)

        self.grad_eps_ng_eigs = np.zeros(self.epsilon.shape + (3,),dtype='complex128')
        for j in range(self.n_modes):
            if j==self.tracked_mode_ind:
                self.grad_eps_ng_eigs += np.abs(self.E[:,:,0,:,j])**2 * ( 2 * self.grad_ω[j] )
            else:
                scale = np.sum(np.conjugate(self.H[:,:,0,:,j])*self.grad_H[...,j]) / ( self.ω[j]**2 - self.ω[self.tracked_mode_ind]**2 ) * self.dx * self.dy
                self.grad_eps_ng_eigs += (np.conjugate(self.E[:,:,0,:,j]) * self.E[:,:,0,:,self.tracked_mode_ind] ) * scale

        E_dr = F[self.design_region_inds_vec[0][0],self.design_region_inds_vec[0][1],0,:,:]
        grad_eps_ω_sq_dr = np.sum(np.abs(E_dr)**2,axis=-2) # sum contributions of independent E-field components (ax=-1 is modes
        grad_eps_ng_dr = self.grad_eps_ng[self.design_region_inds_vec[0][0],self.design_region_inds_vec[0][1]]
        grad_eps_ng_eigs_dr = self.grad_eps_ng_eigs[self.design_region_inds_vec[0][0],self.design_region_inds_vec[0][1]]
        self.gradient = [grad_eps_ω_sq_dr,grad_eps_ng_dr,grad_eps_ng_eigs_dr]

        # # Cleanup list of lists
        # if len(self.gradient) == 1:
        #     self.gradient = self.gradient[0] # only one objective function
        #     if len(self.gradient) == 1:
        #         self.gradient = self.gradient[0] # only one objective function and one design region
        # else:
        #     if len(self.gradient[0]) == 1:
        #         self.gradient = [g[0] for g in self.gradient] # multiple objective functions bu one design region
        # # Return optimizer's state to initialization
        self.current_state = "INIT"

    def calculate_fd_gradient(self,num_gradients=1,db=1e-4,design_variables_idx=0,filter=None):
        '''
        Estimate central difference gradients.

        Parameters
        ----------
        num_gradients ... : scalar
            number of gradients to estimate. Randomly sampled from parameters.
        db ... : scalar
            finite difference step size
        design_variables_idx ... : scalar
            which design region to pull design variables from

        Returns
        -----------
        fd_gradient ... : lists
            [number of objective functions][number of gradients]

        '''
        if filter is None:
            filter = lambda x: x
        if num_gradients > self.num_design_params[design_variables_idx]:
            raise ValueError("The requested number of gradients must be less than or equal to the total number of design parameters.")

        # cleanup simulation object
        # with pipes(stdout=blackhole, stderr=STDOUT):
        #     self.ms.init_params()

        # preallocate result vector
        fd_gradient = []

        # randomly choose indices to loop estimate
        fd_gradient_idx = np.random.choice(self.num_design_params[design_variables_idx],num_gradients,replace=False)

        for k in fd_gradient_idx:

            b0 = np.ones((self.num_design_params[design_variables_idx],))
            b0[:] = (self.design_variables[design_variables_idx].rho_vector)
            # -------------------------------------------- #
            # left function evaluation
            # -------------------------------------------- #

            # assign new design vector
            b0[k] -= db
            self.design_variables[design_variables_idx].set_rho_vector(b0)

            blackhole = StringIO()
            with pipes(stdout=blackhole, stderr=STDOUT):
                self.ms.init_params(self.parity, False)
                self.ms.run(*self.band_funcs)
            ω = np.array(self.ms.freqs)
            ng = np.array([1 / self.ms.compute_one_group_velocity_component(mp.Vector3(0, 0, 1), band_ind+1) for band_ind in range(self.n_modes)])
            neff = self.k_points/ω
            results_list = [ω**2,ng,]

            # record final objective function value
            # results_list = []
            # for m in self.objective_arguments:
            #     results_list.append(m())

            fm = [fi(*results_list) for fi in self.objective_functions]

            # -------------------------------------------- #
            # right function evaluation
            # -------------------------------------------- #

            # assign new design vector
            b0[k] += 2*db # central difference rule...
            self.design_variables[design_variables_idx].set_rho_vector(b0)

            # run
            blackhole = StringIO()
            with pipes(stdout=blackhole, stderr=STDOUT):
                self.ms.init_params(self.parity, False)
                self.ms.run(*self.band_funcs)
            ω = np.array(self.ms.freqs)
            neff = self.k_points/ω
            ng = np.array([1 / self.ms.compute_one_group_velocity_component(mp.Vector3(0, 0, 1), band_ind+1) for band_ind in range(self.n_modes)])
            results_list = [ω**2,ng,]
            # record final objective function value
            # results_list = []
            # for m in self.objective_arguments:
            #     results_list.append(m())
            fp = [fi(*results_list) for fi in self.objective_functions]

            # -------------------------------------------- #
            # estimate derivative
            # -------------------------------------------- #
            fd_gradient.append( [np.squeeze((fp[fi] - fm[fi]) / (2*db)) for fi in range(len(self.objective_functions))] )

        # Cleanup singleton dimensions
        if len(fd_gradient) == 1:
            fd_gradient = fd_gradient[0]

        return fd_gradient, fd_gradient_idx

    def update_design(self, rho_vector):
        """Update the design permittivity function.

        rho_vector ....... a list of numpy arrays that maps to each design region
        """
        for bi, b in enumerate(self.design_variables):
            if np.array(rho_vector[bi]).ndim > 1:
                raise ValueError("Each vector of design variables must contain only one dimension.")
            b.set_rho_vector(rho_vector[bi])
        self.epsilon = self.ms.get_epsilon()
        self.current_state = "INIT"

    def get_objective_arguments(self):
        '''Return list of evaluated objective arguments.
        '''
        objective_args_evaluation = [m.get_evaluation() for m in self.objective_arguments]
        return objective_args_evaluation

    def plot2D(self,F='E',zind=0,xlim=None,ylim=None,**kwargs):
        """Produce a graphical visualization of the geometry and/or fields,
           as appropriately autodetermined based on the current state of
           progress.
        """


        figsize=(14, 5 + 3*self.n_modes)
        #### plot various computed fields for this instance ###
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(self.n_modes+1,
                      3,
                      wspace=0.6,
                      hspace=0.2,
                     )

        ax00 = fig.add_subplot(gs[0,0])
        ax01 = fig.add_subplot(gs[0,1])
        #ax02 = fig.add_subplot(gs[0,2])

        eps = self.epsilon
        E = self.E


        # # # prepare spatial coordinate vectors
        # nx = eps.shape[0]
        # ny = eps.shape[1]
        # x = Xgrid * np.linspace(-1/2., 1/2., nx)
        # y = Ygrid * np.linspace(-1/2., 1/2., ny)

        # plot index as a function of space
        p00 = ax00.pcolormesh(x,y,np.sqrt(eps.T),label='index')

        # plot binary variable detecting core material
        # p01 = ax01.pcolormesh(x,y,mat_core_mask.T,cmap=cm.Greys,label='core_mat')

        ax = [ax00,ax01,]
        p = [p00,p01,]
        labels = ['index',
                  'core_mat',
                  'ϵ|E$_x$|$^2$',
                  'E$_{x}$',
                  'E$_{y}$',
                  'E$_{z}$',
                  'H$_{x}$',
                  'H$_{y}$',
                  'H$_{z}$',
                 ]


        for aind,a in enumerate(ax):
            if xlim:
                a.set_xlim(xlim)
            if ylim:
                a.set_ylim(ylim)
            a.set_aspect('equal')
            a.set_xlabel('x [μm]')
            a.set_ylabel('y [μm]')
            divider = make_axes_locatable(a)
            cax = divider.append_axes("top", size="5%", pad=0.05)
            cb = plt.colorbar(p[aind],
                         cax=cax,
                         orientation="horizontal",
                        )
            cb.set_label(labels[aind])#,labelpad=-1)
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.xaxis.set_label_position('top')

        ####################################################################


        for mind in range(n_modes):
            ax =  [fig.add_subplot(gs[1+mind,0]),
                    fig.add_subplot(gs[1+mind,1]),
                    fig.add_subplot(gs[1+mind,2]),
                  ]


        #     F_norm = E[...,zind,:,mind] / E[...,zind,:,mind].flatten()[np.argmax(np.abs(E[...,zind,:,mind]))]
            F_norm = E[...,zind,:,mind] / E[...,zind,:,mind].flatten()[np.argmax(np.abs(E[...,zind,:,mind]))]
            eps_ei2 = eps * ( np.abs(F_norm[:,:,0])**2 + np.abs(F_norm[:,:,1])**2 + np.abs(F_norm[:,:,2])**2 )
            labels = [
                      'E$_{x}$',
                      'E$_{y}$',
        #               'E$_{z}$',
                      'ϵ|E|$^2$',
            ]

            if F == "H":
                H = self.H
                F_norm = H[...,zind,:,mind] / H[...,zind,:,mind].flatten()[np.argmax(np.abs(H[...,zind,:,mind]))]
                labels = [
                          'H$_{x}$',
                          'H$_{y}$',
            #               'H$_{z}$',
                           'ϵ|E|$^2$',
                         ]

            # plot Fx, Fy, Fz and ϵ|E|^2
            vmax = np.abs(E_norm.real).max()
            vmin = -1 * np.abs(E_norm.real).max() #E_norm.real.min()




            axind = 0
            p10 = ax[axind].pcolormesh(x,
                                  y,
                                  F_norm[:,:,axind].T.real,
                                  cmap=cm.RdBu,
                                  vmin=vmin,
                                  vmax=vmax,
                                 )

            axind = 1
            p11 = ax[axind].pcolormesh(x,
                                  y,
                                  F_norm[:,:,axind].T.real,
                                  cmap=cm.RdBu,
                                  vmin=vmin,
                                  vmax=vmax,
                                 )

        #     axind = 2
        #     p12 = ax[axind].pcolormesh(x,
        #                           y,
        #                           F_norm[:,:,axind].T.real,
        #                           cmap=cm.RdBu,
        #                           vmin=vmin,
        #                           vmax=vmax,
        #                          )

            axind = 2
            p12 = ax[axind].pcolormesh(x,y,eps_ei2.T,cmap=cm.magma)








            ## format and label plots

        #     ax = [ax10,ax11,ax12,ax13,ax20,ax21,ax22]
            p = [p10,p11,p12]





            for aind,a in enumerate(ax):
                if xlim:
                    a.set_xlim(xlim)
                if ylim:
                    a.set_ylim(ylim)
                a.set_aspect('equal')
                a.set_xlabel('x [μm]')
                a.set_ylabel('y [μm]')
                divider = make_axes_locatable(a)
                cax = divider.append_axes("top", size="5%", pad=0.05)
                cb = plt.colorbar(p[aind],
                             cax=cax,
                             orientation="horizontal",
                            )
                cb.set_label(labels[aind])#,labelpad=-1)
                cb.ax.xaxis.set_ticks_position('top')
                cb.ax.xaxis.set_label_position('top')

        # print field energy integrals
        res = ms.compute_field_energy()

def atleast_3d(*arys):
    from numpy import array, asanyarray, newaxis
    '''
    Modified version of numpy's `atleast_3d`

    Keeps one dimensional array data in first dimension, as
    opposed to moving it to the second dimension as numpy's
    version does. Keeps the meep dimensionality convention.

    View inputs as arrays with at least three dimensions.
    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences.  Non-array inputs are converted to
        arrays.  Arrays that already have three or more dimensions are
        preserved.
    Returns
    -------
    res1, res2, ... : ndarray
        An array, or list of arrays, each with ``a.ndim >= 3``.  Copies are
        avoided where possible, and views with three or more dimensions are
        returned.  For example, a 1-D array of shape ``(N,)`` becomes a view
        of shape ``(N, 1, 1)``, and a 2-D array of shape ``(M, N)`` becomes a
        view of shape ``(M, N, 1)``.
    '''
    res = []
    for ary in arys:
        ary = asanyarray(ary)
        if ary.ndim == 0:
            result = ary.reshape(1, 1, 1)
        elif ary.ndim == 1:
            result = ary[:, newaxis, newaxis]
        elif ary.ndim == 2:
            result = ary[:, :, newaxis]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res
