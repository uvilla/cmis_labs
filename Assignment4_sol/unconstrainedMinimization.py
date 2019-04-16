# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.

from __future__ import print_function, division, absolute_import

import dolfin as dl
import math

class InexactNewtonCG:
    """
    Inexact Newton-CG method to solve unconstrained optimization problems.
    The Newton system is solved inexactly by early termination of CG iterations via Eisenstat-Walker
    (to prevent oversolving).
    Globalization is performed using the armijo sufficient reduction condition (backtracking).
    The stopping criterion is based on a control on the norm of the gradient and a control of the
    inner product between the gradient and the Newton direction.
       
    The user must provide the variational forms for the energy functional. 
    The gradient and the Hessian of the energy functional can be either provided by the user
    or computed by FEniCS using automatic differentiation.
    
    NOTE: Essential Boundary Conditions are not supported
    """
    termination_reasons = [
                           "Maximum number of Iteration reached",      #0
                           "Norm of the gradient less than tolerance", #1
                           "Maximum number of backtracking reached",   #2
                           "Norm of (g, dm) less than tolerance"       #3
                           ]
    
    def __init__(self):
        """
        Initialize the InexactNewtonCG with the following parameters.
        rel_tolerance         --> we converge when ||g||_2/||g_0||_2 <= rel_tolerance
        abs_tolerance         --> we converge when ||g||_2 <= abs_tolerance
        gdm_tolerance         --> we converge when (g,dm) <= gdm_tolerance
        max_iter              --> maximum number of iterations
        c_armijo              --> Armijo constant for sufficient reduction
        max_backtracking_iter --> Maximum number of backtracking iterations
        print_level           --> Print info on screen
        cg_coarse_tolerance   --> Coarsest tolerance for the CG method (Eisenstat-Walker)
        """        
        self.parameters = {}
        self.parameters["rel_tolerance"]         = 1e-6
        self.parameters["abs_tolerance"]         = 1e-12
        self.parameters["gdm_tolerance"]         = 1e-18
        self.parameters["max_iter"]              = 20
        self.parameters["c_armijo"]              = 1e-4
        self.parameters["max_backtracking_iter"] = 10
        self.parameters["print_level"]           = 0
        self.parameters["cg_coarse_tolerance"]   = .5
        
        self.it = 0
        self.converged = False
        self.total_cg_iter = 0
        self.reason = 0
        self.final_grad_norm = 0

    def solve(self, J, grad, H, m):

            
            
        
        
        rtol          = self.parameters["rel_tolerance"]
        atol          = self.parameters["abs_tolerance"]
        gdm_tol       = self.parameters["gdm_tolerance"]
        max_iter      = self.parameters["max_iter"]
        c_armijo      = self.parameters["c_armijo"] 
        max_backtrack = self.parameters["max_backtracking_iter"]
        prt_level     = self.parameters["print_level"]
        cg_coarse_tol = self.parameters["cg_coarse_tolerance"]
        
        Jn = dl.assemble( J(m)   )
        gn = dl.assemble( grad(m) )
        g0_norm = gn.norm("l2")
        gn_norm = g0_norm
        tol = max(g0_norm*rtol, atol)
        dm = dl.Vector()
        
        self.converged = False
        self.reason = 0
        
        if prt_level > 0:
            print( "{0:>3} {1:>15} {2:>15} {3:>15} {4:>15} {5:>15} {6:>7}".format(
                "It", "Energy", "||g||", "(g,du)", "alpha", "tol_cg", "cg_it") )
        
        for self.it in range(max_iter):
            Hn = dl.assemble( H(m) )
            
            Hn.init_vector(dm,1)
            solver = dl.PETScKrylovSolver("cg", "petsc_amg")
            solver.set_operator(Hn)
            solver.parameters["nonzero_initial_guess"] = False
            cg_tol = min(cg_coarse_tol, math.sqrt( gn_norm/g0_norm) )
            solver.parameters["relative_tolerance"] = cg_tol
            lin_it = solver.solve(dm,-gn)
            
            self.total_cg_iter += lin_it
            

            dm_gn = dm.inner(gn)
            
            if(-dm_gn < gdm_tol):
                self.converged=True
                self.reason = 3
                break
             
            m_backtrack = m.copy(deepcopy=True)
            alpha = 1.   
            bk_converged = False
            
            #Backtrack
            for j in range(max_backtrack):
                m.assign(m_backtrack)
                m.vector().axpy(alpha, dm)
                Jnext = dl.assemble( J(m) )
                if Jnext < Jn + alpha*c_armijo*dm_gn:
                    Jn = Jnext
                    bk_converged = True
                    break
                
                alpha = alpha/2.
                
            if not bk_converged:
                self.reason = 2
                break
                   
            gn = dl.assemble( grad(m) )
            gn_norm = gn.norm("l2")
            
            if prt_level > 0:
                print( "{0:3d} {1:15e} {2:15e} {3:15e} {4:15e} {5:15e} {6:7d}".format(
                        self.it, Jn, gn_norm, dm_gn, alpha, cg_tol, lin_it) )
                
            if gn_norm < tol:
                self.converged = True
                self.reason = 1
                break
            
        self.final_grad_norm = gn_norm
        
        if prt_level > -1:
            print( self.termination_reasons[self.reason] )
            if self.converged:
                print( "Inexact Newton CG converged in ", self.it, \
                "nonlinear iterations and ", self.total_cg_iter, "linear iterations." )
            else:
                print( "Inexact Newton CG did NOT converge after ", self.it, \
                "nonlinear iterations and ", self.total_cg_iter, "linear iterations.")
            print ("Final norm of the gradient", self.final_grad_norm)
            print ("Value of the cost functional", Jn)
                
                
