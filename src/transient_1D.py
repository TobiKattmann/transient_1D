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
"""This codes solves the 1D heat eq: 
PDE: dT/dt - dD(x)/dx*dT/dx  - D(x)*d^2T/dx^2 = 0, 
BC: T(x_left,.)=T0, T(x_right,.)=T1, 
IC: T(.,t=0)=T_init

And the respective Adjoint equation.
"""
#========================================================================================#
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
    self.NPt = 10 # number Pseudo time steps
    self.FD1 = Numerics.get_FD_Operator_cental_1st(self.mesh.numnodes, self.mesh.dx)
    self.FD2 = Numerics.get_FD_Operator_cental_2nd(self.mesh.numnodes, self.mesh.dx)
    self.FirstTime_AdjointResidual = True
    self.FirstTime_PrimalResidual = True

  def calculatePrimal(self):
    """docstring."""
    self.full_solution, self.Res = Numerics.perform_dualTimeStepping(self.Nt, self.dt, self.dt, self.NPt, self.u_init, self.u_left, self.u_right, self.get_PrimalResidual, reverse_timestepping_factor=1)

  def get_PrimalResidual(self, u):
    """docstring TODO"""
    # TODO create Residual operator
    if self.FirstTime_PrimalResidual == True:
      TMP = self.FD1@self.D
      A1 = TMP[:, np.newaxis]*self.FD1.toarray()
      A2 = self.D[:, np.newaxis]*self.FD2.toarray()
      self.A = A1 + A2
      self.FirstTime_PrimalResidual = False

    R = - self.A@u
    #R = - (self.FD1@self.D) * (self.FD1@u) - self.D * (self.FD2@u)
    return R

  def calculateObjectiveFunction(self):
    """docstring TODO"""
    obj = 0.
    for n in range(self.Nt):
      if n == self.Nt-1:#HERE
        for i in range(len(self.alpha)):
          obj += self.mesh.dx * self.alpha[i] * self.full_solution[n][i] 
        obj *= self.dt
    
    #Quick Fix
    obj = 0.
    for i in range(len(self.alpha)):
      obj += self.alpha[i] * self.full_solution[self.Nt-1][i] 

    self.obj = obj
    return obj

  def calculateAdjoint(self):
    """docstring TODO"""
    self.djdu = self.alpha
    self.u_init = np.zeros(self.mesh.numnodes)
    self.u_init = -1* -1 *self.alpha#HERE HERE changed additional -1
    adjoint_solution, self.adjointRes = Numerics.perform_dualTimeStepping(self.Nt, self.dt, self.dt, self.NPt, self.u_init, self.u_left, self.u_right, self.get_AdjointResidual, reverse_timestepping_factor=1.)# here reverse time stepping factor is set to zero
    self.adjoint_solution = list(reversed(adjoint_solution))

  def get_AdjointResidual(self, l):
    """docstring TODO"""
    # TODO: what about temporal derivative, Make code Faster
    if self.FirstTime_AdjointResidual == True:
      TMP = self.FD1@self.D
      self.A1 = TMP[:, np.newaxis]*self.FD1.toarray()
      self.A2 = self.D[:, np.newaxis]*self.FD2.toarray()
      self.FirstTime_AdjointResidual = False
    #R =  -(- self.A1.T - self.A2.T)@l #- self.djdu.T HERE
    R =  (- self.A1.T - self.A2.T)@l #- self.djdu.T HERE
    return R

  def calculateSensitivities(self):
    """docstring TODO"""
    dJda = np.zeros(self.mesh.numnodes)
    for timestep in range(self.Nt):
      dRstarda = self.calculate_dRstarda2(self.full_solution[timestep])
      dJda += self.dt * (dRstarda.T@self.adjoint_solution[timestep]).T
    
    self.Dderivative = dJda

  def calculate_dRstarda2(self, primal_solution):
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
    
    return dRstarda

  def calculateTimestep(self):
    """docstring."""
    pass
  
#############################################################################################
if __name__ == '__main__':
  if False:
    #cleanup old files
    os.system('rm -v ../NEWimages-tobi-code/*')

    # Parameters for Optimization
    cycles = 1
    #eps = 1.0e1 # Relaxation for Design change
    eps = 10  
    Objective_Function = np.zeros(cycles)
    Sensitivities = []
    DiffusionCoefficients = []

    Thermal_diffuisvity_over_Time = []
    # Design Optimization loop
    for i in range(cycles):
      print("#--------------------------------------------#\nDesign Iteration: ",i)
      # Pre-processing
      filename = 'Transient_1D_heat_' + str(i)
      num_nodes = 101
      mesh = mesh1D(begin=0, end=35, numnodes=num_nodes)
      init = np.array([math.exp(-(x-15)**2) for x in getattr(mesh,'X')])
      alpha = np.array([math.exp(-(x-20)**2) for x in getattr(mesh,'X')]) # for objective function
      if i == 0:
        D = np.ones(num_nodes)*2.
        #D = np.linspace(0, 1, num_nodes)*2.
      sim = TransHeatEq1D(filename, mesh, init, alpha, D, T0=0, T1=0, Nt=4000)
      Thermal_diffuisvity_over_Time.append(copy.copy(D))    
      DiffusionCoefficients.append(copy.copy(D))

      # Solver
      t0=time.time()
      sim.calculatePrimal()
      t1=time.time()
      sim.calculateAdjoint()
      t2=time.time()
      sim.calculateDiffusionDerivative()
      t3=time.time()
      print(t1-t0,t2-t1,t3-t2)
      
      # Post-processsing
      print("Obj. Func.: ", sim.ObjectiveFunction())
      Objective_Function[i] = sim.ObjectiveFunction()
      Sensitivities.append(sim.Dderivative)
      #D[1:-1] += eps* sim.Dderivative
      D += eps* sim.Dderivative
      # visualizationn
      VisuNew = visualization.Visualization(sim)
      VisuNew.PrimalAdjointSensitivitiesDiffusivity(show=False, save=True)
      if i==2e4:
        plt.plot(sim.mesh.X,D)
        plt.show()


    VisuNew = visualization.Visualization(sim)
    VisuNew.Residuals(save=False,show=False)
    VisuNew.Animate_Primal(show=False, save=False)
    VisuNew.Animate_Adjoint(show=False, save=False)
    VisuNew.Animate_Primal_and_Adjoint(show=False, save=False)
    VisuNew.PrimalAdjointSensitivitiesDiffusivity(show=False, save=True)
    VisuNew.ObjectiveFunction(False, True, cycles, Objective_Function)
    VisuNew.Sensitivities(Sensitivities, show=False, save=True)
    VisuNew.DiffusionCoefficient(DiffusionCoefficients, show=False, save=True)

  if False:
    #cleanup old files
    os.system('rm -v ../NEWimages-tobi-code/*')

    print("Optimzer called")
    num_nodes = 101
    def obj(D):
      # Preprocessing
      filename = "Trans_1D_Diffusion"
      mesh = mesh_1D.mesh1D(begin=0, end=35, numnodes=num_nodes)
      init = np.array([math.exp(-(x-15)**2) for x in getattr(mesh,'X')])
      alpha = np.array([math.exp(-(x-20)**2) for x in getattr(mesh,'X')]) # for objective function
      #sim = TransHeatEq1D(filename, mesh, init, alpha, D, T0=0, T1=0, Nt=4000)
      sim = Transient_1D_Diffusion(mesh, D, init, 0, 0 , Nt=4000, alpha=alpha)
      # Solver
      sim.calculatePrimal()
      J = sim.calculateObjectiveFunction()
      return J, sim

    def derivative(sim):
      sim.calculateAdjoint()
      sim.calculateSensitivities()
      return sim.Dderivative
    
    obj_handle = lambda alpha: obj(alpha)
    derivative_handle = lambda sim: derivative(sim)
    initial_DesignVars = np.ones(num_nodes)*2.

    opt = optimizer.Optimizer(obj_handle, derivative_handle, initial_DesignVars, num_DesignCycles=15, factorDesignVarUpdate=10)
    opt.optimize()

  if False:
    print("Computing FD Sensitivities.")
    num_nodes = 101
    def primal(D):
      # Preprocessing
      filename = "Trans_1D_Diffusion"
      mesh = mesh_1D.mesh1D(begin=0, end=35, numnodes=num_nodes)
      init = np.array([math.exp(-(x-15)**2) for x in getattr(mesh,'X')])
      alpha = np.array([math.exp(-(x-20)**2) for x in getattr(mesh,'X')]) # for objective function
      #sim = TransHeatEq1D(filename, mesh, init, alpha, D, T0=0, T1=0, Nt=4000)
      sim = Transient_1D_Diffusion(mesh, D, init, 0, 0 , Nt=4000, alpha=alpha)
      # Solver
      sim.calculatePrimal()
      J = sim.calculateObjectiveFunction()
      return J
    primal_handle = lambda alpha: primal(alpha)

    D = np.ones(num_nodes)*2.
    print("obj: ", primal_handle(D))

    FD_sens = FD_sensitivities.FD_sensitivities(primal_handle, D, gradientMethod='forward')
    dJda = FD_sens.calculateSensitivities()    
    mesh = mesh_1D.mesh1D(begin=0, end=35, numnodes=num_nodes)
    plt.plot(mesh.X, dJda)
    plt.grid(True, axis='y')
    plt.savefig("FD_sens", bbox_inches='tight')
    plt.show()

    FD_sens.Plot_Sensitivities(mesh.X)
    FDderivative = copy.copy(dJda)

  if True: # with Numerics2
    num_nodes = 101
    Nt = 4000
    mesh = mesh_1D.mesh1D(begin=0, end=35, numnodes=num_nodes)
    D = np.ones(num_nodes)*2.
    init = np.array([math.exp(-(x-15)**2) for x in getattr(mesh,'X')])
    alpha = np.array([math.exp(-(x-20)**2) for x in getattr(mesh,'X')]) # for objective function
    sim = Transient_1D_Diffusion(mesh, D, init, 0, 0 , Nt=4000, alpha=alpha)
    t0=time.time()
    sim.calculatePrimal()
    t1=time.time()
    print(sim.calculateObjectiveFunction())
    sim.calculateAdjoint()
    t2=time.time()
    sim.calculateSensitivities()
    AdjointDerivative = copy.copy(sim.Dderivative)
    t3=time.time()
    print(t1-t0,t2-t1,t3-t2)

    VisuNew = visualization.Visualization(sim)
    VisuNew.Residuals(save=True,show=False)
    VisuNew.Animate_Primal(show=True, save=False)
    VisuNew.Animate_Adjoint(show=True, save=False)
    VisuNew.Animate_Primal_and_Adjoint(show=True, save=False)
    VisuNew.PrimalAdjointSensitivitiesDiffusivity(show=False, save=True)
    #VisuNew.ObjectiveFunction(False, True, cycles, Objective_Function)
    VisuNew.Sensitivities([sim.Dderivative], show=False, save=True)
    #VisuNew.DiffusionCoefficient(DiffusionCoefficients, show=False, save=True)
    a =1

  compareFDandAdjoint = True
  if compareFDandAdjoint == True:
    visualization.compareFDandAdjointSensitivities(sim.mesh.X, AdjointDerivative, FDderivative)
    
  adjoint_check = False
  if adjoint_check == True:
    print(sim.dt, sim.Nt, D)
    eta = np.sqrt((1. + 4*D*sim.dt*sim.Nt)/1.)
    analytical_solution = -np.exp(-((np.array(sim.mesh.X)-20.)**2)/eta**2)/eta

    plt.plot(sim.mesh.X, analytical_solution, label="Analytical")
    plt.plot(sim.mesh.X, sim.adjoint_solution[0], marker=".",label="Numerical")
  
    plt.title("Adjoint solution")
    plt.xlabel("x")
    plt.ylabel("i")
    plt.grid(True, axis='both')
    plt.legend()
    plt.savefig('adjoint_sol', bbox_inches='tight')
    plt.show()

