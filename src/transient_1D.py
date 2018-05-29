import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import copy
import math # for exp-function
import os
import sys
import time
import pandas as pd # only used for dubugging by writing csv file
from ad import adnumber, jacobian

import FD_sensitivities
import mesh_1D 
import visualization
import optimizer

#---------------------------------------------------------------------------------------#
"""This codes solves the 1D heat eq: 
PDE: dT/dt - dD(x)/dx*dT/dx  - D(x)*d^2T/dx^2 = 0, 
BC: T(x_left,.)=T0, T(x_right,.)=T1, 
IC: T(.,t=0)=T_init

And the respective Adjoint equation.
"""
#---------------------------------------------------------------------------------------#
class Numerics:
  """docstring TODO"""

  @staticmethod
  def get_FD_Operator_cental_1st(mat_size, dx):
    """docstring TODO"""
    A = np.zeros((mat_size, mat_size))
    factor = 1/(2*dx) 
    for i in range(mat_size):
      if i>0 and i<mat_size-1:
        A[i][i-1] = -1 * factor
        A[i][i+1] = 1 * factor

    return sp.csr_matrix(A)

  @staticmethod
  def get_FD_Operator_cental_2nd(mat_size, dx):
    """docstring TODO"""
    A = np.zeros((mat_size, mat_size))
    factor = 1/dx**2 
    for i in range(mat_size):
      if i>0 and i<mat_size-1:
        A[i][i] = -2 * factor
        A[i][i-1] = 1 * factor
        A[i][i+1] = 1 * factor

    return sp.csr_matrix(A)
  
  @staticmethod
  def perform_dualTimeStepping(num_timesteps, dt, dtau, num_pseudo_timesteps, 
                                u_initial, u_left, u_right, getSpatialResidual, reverse_timestepping_factor):
    """docstring.

    Args:
      reverse_timestepping_factor (float): 1=forward in time for primal, -1=reverse in time for adjoint
    """
    flowRes = np.zeros(num_timesteps*num_pseudo_timesteps)
    primal_solution = []
    primal_solution.append(u_initial)

    u = copy.copy(u_initial)
    u[0] = u_left
    u[-1] = u_right

    u_next = copy.copy(u)
    
    for timestep in range(num_timesteps):
      for pseudo_timestep in range(num_pseudo_timesteps):
        R = getSpatialResidual(u_next)
        R_star = reverse_timestepping_factor * (u_next - u)/dt + R
        flowRes[timestep*num_pseudo_timesteps+pseudo_timestep] = np.sqrt(np.mean(R_star**2))
        u_next = u_next  - reverse_timestepping_factor * dtau * R_star

      u = copy.copy(u_next)
      primal_solution.append(copy.copy(u))

    return primal_solution, flowRes    

#---------------------------------------------------------------------------------------#
class Transient_1D_Diffusion:
  """docstring."""
  def __init__(self, mesh, D, init, u_right, u_left, Nt, alpha):
    """docstring TODO"""
    self.name = "Transient_1D_Diffusion"
    self.mesh = mesh
    self.D = D
    self.u_init = init
    self.u_right = u_right
    self.u_left = u_left
    self.Nt = Nt
    self.alpha = alpha

    self.dt = 0.0004
    self.dtau = self.dt
    self.NPt = 10 # number Pseudo time steps
    self.FD1 = Numerics.get_FD_Operator_cental_1st(self.mesh.numnodes, self.mesh.dx)
    self.FD2 = Numerics.get_FD_Operator_cental_2nd(self.mesh.numnodes, self.mesh.dx)
    self.FirstTime_AdjointResidual = True
    self.FirstTime_PrimalResidual = True

  def calculateTimestep(self):
    """docstring."""
    pass

  def calculatePrimal(self):
    """docstring."""
    self.full_solution, self.Res = Numerics.perform_dualTimeStepping(self.Nt, self.dt, self.dt, self.NPt, self.u_init, self.u_left, self.u_right, self.get_PrimalResidual, reverse_timestepping_factor=1)

  def get_PrimalResidual(self, u):
    """docstring TODO"""
    #  create Residual operator self.A at first call
    if self.FirstTime_PrimalResidual == True:
      TMP = self.FD1@self.D
      A1 = TMP[:, np.newaxis]*self.FD1.toarray()
      A2 = self.D[:, np.newaxis]*self.FD2.toarray()
      self.A = A1 + A2
      self.FirstTime_PrimalResidual = False

    #R = - (self.FD1@self.D) * (self.FD1@u) - self.D * (self.FD2@u)# this is kinda the long version
    R = - self.A@u
    return R

  def calculateObjectiveFunction(self):
    """docstring TODO"""
    obj = 0.
    for i in range(len(self.alpha)):
      obj += - self.alpha[i] * self.full_solution[self.Nt-1][i] 

    self.obj = obj
    return obj

#---------------------------------------------------------------------------------------#
  def calculateAdjoint(self):
    """docstring TODO"""
    self.djdu = self.alpha
    self.u_init = np.zeros(self.mesh.numnodes)
    self.u_init = -1.0 * self.alpha
    adjoint_solution, self.adjointRes = Numerics.perform_dualTimeStepping(self.Nt, self.dt, self.dt, self.NPt, self.u_init, self.u_left, self.u_right, self.get_AdjointResidual2, reverse_timestepping_factor=1.)# here reverse time stepping factor is set to zero
    self.adjoint_solution = list(reversed(adjoint_solution))

  def get_AdjointResidual(self, l):
    """docstring TODO"""
    if self.FirstTime_AdjointResidual == True:
      TMP = self.FD1@self.D
      self.A1 = TMP[:, np.newaxis]*self.FD1.toarray()
      self.A2 = self.D[:, np.newaxis]*self.FD2.toarray()
      self.FirstTime_AdjointResidual = False
    R =  (- self.A1.T - self.A2.T)@l #- self.djdu.T HERE
    return R

  def get_AdjointResidual2(self, l):
    """docstring TODO"""
    if self.FirstTime_AdjointResidual == True:
      self.B = -self.A.T
    R = self.B@l
    return R

  def calculateSensitivities(self):
    """docstring TODO"""
    dJda = np.zeros(self.mesh.numnodes)
    for timestep in range(self.Nt):
      dRstarda = self.calculate_dRstarda(self.full_solution[timestep])
      dJda += dRstarda.T@self.adjoint_solution[timestep] # self.dt is in fact dtau
    
    self.Dderivative = dJda

  def calculate_dRstarda(self, primal_solution):
    """Computes derivative of flow Res wrt design variables.
    
    dR*/da = d/da [d/dt u - d/dx a*d/dx u - a*d/dx u]
           = - diag(A1*u)*A1 - diag(A2*u)
    Where A1/A2 are FD-Operators for 1st/2nd order space discretization
    Result works as an operator on the adjoint solution.     

    Args:
      None
    Returns:
      np.ndarray((n,n)): derivative of flow Res wrt design variables
    """
    #dRstarda = -np.diag(A1@primal_solution)@A1 - np.diag(A2@primal_solution)
    TMP = self.FD1@primal_solution
    dRstarda = - TMP[:, np.newaxis]*self.FD1.toarray() - np.diag(self.FD2@primal_solution)
    
    return -self.dtau * dRstarda

#---------------------------------------------------------------------------------------#
if __name__ == '__main__':
  pass
    

