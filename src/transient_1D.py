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
import Visualization
"""This codes solves the 1D heat eq: 
PDE: dT/dt - dD(x)/dx*dT/dx  - D(x)*d^2T/dx^2 = 0, 
BC: T(x_left,.)=T0, T(x_right,.)=T1, 
IC: T(.,t=0)=T_init

And the respective Adjoint equation.
"""
#========================================================================================#
class Numerics2:
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
    self.FD1 = Numerics2.get_FD_Operator_cental_1st(self.mesh.numnodes, self.mesh.dx)
    self.FD2 = Numerics2.get_FD_Operator_cental_2nd(self.mesh.numnodes, self.mesh.dx)
    self.FirstTime_AdjointResidual = True
    self.FirstTime_PrimalResidual = True

  def calculatePrimal(self):
    """docstring."""
    self.full_solution, self.Res = Numerics2.perform_dualTimeStepping(self.Nt, self.dt, self.dt, self.NPt, self.u_init, self.u_left, self.u_right, self.get_PrimalResidual, reverse_timestepping_factor=1)

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
      for i in range(len(self.alpha)):
        obj += self.mesh.dx * self.alpha[i] * self.full_solution[n][i] 
      obj *= self.dt

    self.obj = obj
    return obj

  def calculateAdjoint(self):
    """docstring TODO"""
    self.djdu = self.alpha
    self.u_init = np.zeros(self.mesh.numnodes)
    adjoint_solution, self.adjointRes = Numerics2.perform_dualTimeStepping(self.Nt, self.dt, self.dt, self.NPt, self.u_init, self.u_left, self.u_right, self.get_AdjointResidual, reverse_timestepping_factor=-1.)
    self.adjoint_solution = list(reversed(adjoint_solution))

  def get_AdjointResidual(self, l):
    """docstring TODO"""
    # TODO: what about temporal derivative, Make code Faster
    if self.FirstTime_AdjointResidual == True:
      TMP = self.FD1@self.D
      self.A1 = TMP[:, np.newaxis]*self.FD1.toarray()
      self.A2 = self.D[:, np.newaxis]*self.FD2.toarray()
      self.FirstTime_AdjointResidual = False
    R =  -(- self.A1.T - self.A2.T)@l - self.djdu.T
    return R

  def calculateSensitivities(self):
    """docstring TODO"""
    dJda = np.zeros(self.mesh.numnodes)
    for timestep in range(self.Nt):
      dRstarda  = self.calculate_dRstarda2(self.full_solution[timestep])
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
  
#========================================================================================#
#############################################################################################
class mesh1D:
  def __init__(self, begin, end, numnodes):
    """Provides equidistant mesh.

    Args:
      begin (int): left boundary
      end (int): right boundary
      numnodes (int): number of nodes in the grid
    Returns:
      None 
    Sets:
      all Args ...
      self.dx (float): mesh spacing (equidistant)
      self.X (np.ndarray(n,)): 1D coordinates of grid points
    """
    self.begin = begin
    self.end = end
    self.numnodes = numnodes
  
    # Derived quantities
    self.dx = (end - begin)/(numnodes-1)
    self.X = np.array([self.begin + i*self.dx for i in range(self.numnodes)]) 

#############################################################################################
class Numerics:
  def __init__(self, mesh, D):
    """Initializes and computes reusable parts."""
    # Inputs
    self.mesh = mesh
    self.D = D
    
    # Initialization of object variables
    self.AD_mode = False # TODO for testing purposes
    self.FD_central_FirstOrder_Operator()
    self.FD_central_SecondOrder_Operator()

  def FD_central_FirstOrder_Operator(self):
    """Sets Operator for first order central Finite Difference.
      
    FD1*u => - d/dx D(x) * d/dx u_i = -(-D_i-1 + D_i+1)/(2*dx)*(-u_i-1 + u_i+1)/(2*dx)
    Works on full solution vector including boundary values. Rows at Boundary values return 0 residual.    

    Args:
      None
    Returns:
      None
    Sets:
      self.FD1 (np.ndarray((n,n))): central 1st order FD Operator
    """
    A = np.zeros((self.mesh.numnodes,self.mesh.numnodes))
    # set side-diagonals with ones
    for i in range(self.mesh.numnodes):
      if i>0 and i<self.mesh.numnodes-1:
        factor = -1 * (-self.D[i-1] + self.D[i+1])/(4*self.mesh.dx**2) 
        A[i][i-1] = -1 * factor
        A[i][i+1] = 1 * factor
      elif i == 0:
        pass
      elif i==self.mesh.numnodes-1:
        pass
      else: 
        raise Exception("Matrix out of bounds")

    if self.AD_mode == True: return A
    self.FD1 = A

  def FD_central_SecondOrder_Operator(self):
    """Sets Operator for second order central Finite Difference.
    
    FD2*u => -D(x) * d2/dx2 u_i = -D_i * (u_i-1 - 2*u_i + u_i+1)/(dx**2)
    Works on full solution vector including boundary values. Rows at Boundary values return 0 residual.    

    Args:
      None
    Returns:
      None
    Sets:
      self.FD2 (np.ndarray((n,n))): central 2nd order FD Operator
    """
    A = np.zeros((self.mesh.numnodes,self.mesh.numnodes))

    # set diagonal and side-diagonals 
    for i in range(self.mesh.numnodes):
      factor = -1 * self.D[i]/self.mesh.dx**2 
      if i>0 and i<self.mesh.numnodes-1:
        A[i][i] = -2 * factor
        A[i][i-1] = 1 * factor
        A[i][i+1] = 1 * factor
      elif i == 0:
        pass
      elif i==self.mesh.numnodes-1:
        pass
      else:
        raise Exception("Matrix out of bounds")

    if self.AD_mode == True: return A
    self.FD2 = A

  def getPrimalSpatialResidual(self, u):
    """Computes the spatial residual for the primal.

    R = - dD(x)/dx*du/dx  - D(x)*d^2u/dx^2 
    such that: du/dt + R(u,D) = 0 at convergence

    Args:
      u (np.ndarray(n,): current state vector to evaluate the residual for
    Returns:
      np.ndarray(n,): spatial resuidual of primal
    """
    R = self.FD1@u + self.FD2@u
    return R

  def dualTimeSteppingPrimal(self, num_timesteps, dt, num_pseudo_timesteps, dtau, u_initial, u_left=0, u_right=1):
    """Computes solution for each time-step.
    
    R = spatial residual
    U^n_p+1 = U^n_p - dtau * [(U^n_p - U^n-1)/dt + R(U^n_p)]
    In the case of dtau=dt the above formula becomes:
    U^n_p+1 = U^n-1 - dt * R(U^n_p)

    Args: 
      num_timesteps (int): number of physical timesteps
      dt (float): pyhsical timestep size
      num_pseudo_timesteps (int): number of pseudo timesteps for each physical
      dtau (float): pseudo timestep size
      u_initial (np.ndarray(n,)): initial solution
      u_left (float)= boundary condition on the left 
      u_right (float)= boundary condition on the right
    Returns: 
      list(num_timesteps * np.ndarray(n,)): solution array for each timestep 
      np.ndarray(num_t*num_pseudo_t,): vector containing mean-square residual at each iter
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
        R = self.getPrimalSpatialResidual(u_next)
        R_star = (u_next - u)/dt + R
        flowRes[timestep*num_pseudo_timesteps+pseudo_timestep] = np.sqrt(np.mean(R_star**2))
        u_next = u_next - dtau * R_star

      u = copy.copy(u_next)
      primal_solution.append(copy.copy(u))

    return primal_solution, flowRes    
  
  def getAdjointSpatialResidual(self, l, djdu):
    """Computes the spatial residual for the primal.

    R = -[dR*/du].T*l -[dj/du].T
    R = -[- dD(x)/dx*d/dx  - D(x)*d^2/dx^2].T*l -[dj/du].T
    such that: dl/dt + R(l,D) = 0 at convergence

    Args:
      u (np.ndarray(n,): current state vector to evaluate the residual for
    Returns:
      np.ndarray(n,): spatial resuidual of primal
    """
    #TODO: what about the term: d/du d/dt u => matrix with 1/dt on diagonal? -A contribution
    # TODO: signs  mixed up, compare primal residual
    R = -self.FD1.T@l - self.FD2.T@l - djdu.T
    #TODO: to enforce the boundary conditions necessary??
    #R[0] = 0
    #R[-1] = 0
    return R

  def dualTimeSteppingAdjoint(self, num_timesteps, dt, num_pseudo_timesteps, dtau, l_initial, l_left, l_right, djdu):
    """Computes the solution to all timesteps of the adjoint equation.

    d/dt l - (dR*/du).T*l - (dj/du).T = 0
    s.t. l(t=T) = 0 
    Note that therefore the timestepping walks backwards in time.
    The terms containing "u" correspond the the primal.
    
    Args: 
      num_timesteps (int): number of physical timesteps
      dt (float): pyhsical timestep size
      num_pseudo_timesteps (int): number of pseudo timesteps for each physical
      dtau (float): pseudo timestep size
      l_initial (np.ndarray(n,)): initial solution
      l_left (float)= boundary condition on the left 
      l_right (float)= boundary condition on the right
      djdu (np.ndarray(n,) of floats): objective function derivative wrt primal DOF's
    Returns: 
      list(num_timesteps * np.ndarray(n,)): solution array for each timestep, in same pyhical order as primal 
      np.ndarray(num_t*num_pseudo_t,): vector containing mean-square residual at each iter
    """
    flowRes = np.zeros(num_timesteps*num_pseudo_timesteps)
    adjoint_solution = []
    adjoint_solution.append(l_initial)

    l = copy.copy(l_initial)
    l[0] = l_left
    l[-1] = l_right

    l_next = copy.copy(l)
    
    for timestep in range(num_timesteps):
      for pseudo_timestep in range(num_pseudo_timesteps):
        R = self.getAdjointSpatialResidual(l_next, djdu)
        R_star = (l - l_next)/dt + R # reverse timestepping
        flowRes[timestep*num_pseudo_timesteps+pseudo_timestep] = np.sqrt(np.mean(R_star**2))
        l_next = l_next + dtau * R_star # time stepping backwards!! Therfore the +

      l = copy.copy(l_next)
      adjoint_solution.append(copy.copy(l))

    return list(reversed(adjoint_solution)), flowRes    

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
    # 1st order spatial derivative Operator L*u = d/dx u
    A1 = np.zeros((self.mesh.numnodes,self.mesh.numnodes))
    for i in range(self.mesh.numnodes):
      if i>0 and i<self.mesh.numnodes-1:
        factor = 1/(2*self.mesh.dx) 
        A1[i][i-1] = -1 * factor
        A1[i][i+1] = 1 * factor
    
    # 2nd order spatial derivative Operator L*u = d2/dx2 u
    A2 = np.zeros((self.mesh.numnodes,self.mesh.numnodes))
    for i in range(self.mesh.numnodes):
      factor = 1/(self.mesh.dx**2)#/2 # TODO factor 1/2 artificial why?
      if i>0 and i<self.mesh.numnodes-1:
        A2[i][i] = -2 * factor
        A2[i][i-1] = 1 * factor
        A2[i][i+1] = 1 * factor

    #dRstarda = -np.diag(A1@primal_solution)@A1 - np.diag(A2@primal_solution)
    TMP = A1@primal_solution
    dRstarda = - TMP[:, np.newaxis]*A1 - np.diag(A2@primal_solution)
    
    return dRstarda

  def calculate_dJda(self, num_timesteps, dt, primal_solution, adjoint_solution):
    """Calculates derivative of obj-func wrt to design-vars.
    
    (dJ/da).T = (dj/da).T - (dR*/da).T*l
    or to be more precice in discrete time
    (dJ/da).T = sum_t (dj_t/da).T - (dR*_t/da).T*l_t
    Note that in our current case dj/da = 0.

    Args:
      None
    Returns:
      np.ndarray(n,): derivative of obj-func wrt to design-vars
    """
    dJda = np.zeros(self.mesh.numnodes)
    for timestep in range(num_timesteps):
      dRstarda  = self.calculate_dRstarda(primal_solution[timestep])
      #ad_dRstarda = self.TEMP_compute_dRstarda_by_ad(primal_solution)

      if False: # TODO This is complete bullshit
        for i in range(self.mesh.numnodes):
          dRstarda[i][i] = 0

      dJda += dt * (dRstarda.T@adjoint_solution[timestep]).T

      if timestep == num_timesteps-1 or timestep == 0 or timestep == 2000 and True:
        self.DEBUGGING_plot_adjoint(adjoint_solution[timestep], 'adjoint_'+str(timestep))
        self.DEBUGGING_plot_dRstarda(dRstarda, timestep, 'dRstarda_'+str(timestep))
        self.DEBUGGING_plot_djda(dJda, 'dJda_'+str(timestep))
        self.DEBUGGING_plot_sums_of_dRstarda(-dRstarda, 'sum_'+str(timestep))
        #raise Exception

    return dt*dJda 

  def DEBUGGING_plot_adjoint(self, adj, save_name):
    """docstring."""
    #plt.close("all")
    plt.figure(figsize=(16,5))
    plt.grid(True, axis='both')  
    plt.plot(self.mesh.X, adj)
    plt.savefig(save_name, bbox_inches='tight')
    #plt.show()
    
  def DEBUGGING_plot_djda(self, djda, save_name):
    """docstring."""
    #plt.close("all")
    plt.figure(figsize=(16,5))
    plt.grid(True, axis='both')  
    plt.plot(self.mesh.X, djda)
    plt.savefig(save_name, bbox_inches='tight')
    #plt.show()
    
  def DEBUGGING_plot_dRstarda(self, dRstarda, timestep, save_name):
    """docstring."""
    #plt.close("all")
    #plt.spy(dRstarda)
    #plt.show()
    plt.figure(figsize=(16,5))
    plt.grid(True, axis='both')  
    #plt.ylim([-0.1,0.1])
    #plt.ylim([-2,2])
    plt.plot(self.mesh.X,[dRstarda[i][i] for i in range(self.mesh.numnodes)], c='red')      
    plt.plot(self.mesh.X[:-1],[dRstarda[i][i+1] for i in range(self.mesh.numnodes-1)], c='black')      
    plt.plot(self.mesh.X[1:],[dRstarda[i+1][i] for i in range(self.mesh.numnodes-1)], c='green')      
    plt.savefig(save_name, bbox_inches='tight')
    plt.close('all')
    plt.matshow(dRstarda)
    plt.savefig('matshow'+save_name, bbox_inches='tight')
    #plt.show()
    print(dRstarda.shape)
    df = pd.DataFrame(dRstarda)
    df.to_csv('dRstarda_'+str(timestep)+'.csv')
  
  def DEBUGGING_plot_sums_of_dRstarda(self, A, save_name):
    """Plot row and column sums of a Matrix A and plot the result."""
    # rows
    plt.close('all')
    row_sums = [sum(A[i,:]) for i in range(A.shape[0])]
    plt.figure(figsize=(16,5))
    plt.plot(self.mesh.X, row_sums)  
    plt.grid(True, axis='both')  
    plt.savefig('row_'+save_name, bbox_inches='tight')
    plt.close('all')
    # columns
    column_sums = [sum(A[:,i]) for i in range(A.shape[1])]
    plt.figure(figsize=(16,5))
    plt.plot(self.mesh.X, column_sums)  
    plt.grid(True, axis='both')  
    plt.savefig('column_'+save_name, bbox_inches='tight')
    plt.close('all')

  def TEMP_compute_dRstarda_by_ad(self, primal_solution):
    """docstring."""
    self.AD_mode == True
    self.D = adnumber(self.D) # set 
    # compute Rstar
    FD1 = self.FD_central_FirstOrder_Operator()
    FD2 = self.FD_central_SecondOrder_Operator()
    print(type(FD1), FD1)
    R_star = FD1.dot(primal_solution) + FD2.dot(primal_solution)

    dRstarda = jacobian(R_star, self.D)
    
    plt.spy(dRstarda)
    plt.show()

    df = pd.DataFrame(dRstarda)
    df.to_csv('ad_dRstarda.csv')
      
#############################################################################################
class TransHeatEq1D:
  def __init__(self, name, mesh, init, alpha, D, T0, T1, Nt):
    """Initialization of input and derived instance variables."""
    # Inputs at initialization
    self.mesh = mesh
    self.init = init
    self.D = D
    self.T0 = T0
    self.T1 = T1
    self.Nt = Nt
    self.alpha = alpha
    self.name = name
    
    self.intIter = 10 # internal iterations of the dual time stepping method
    self.dt = self.computeTimestep()
    #print(self.dt)
    self.dt = 0.0004
    self.full_solution = []
    self.full_solution.append(copy.copy(self.init))
    self.Res = np.zeros(self.Nt*self.intIter)    

  def calculatePrimal(self):
    """Computes primal flow solution."""  
    self.numerics = Numerics(self.mesh, self.D)
    self.full_solution, self.Res = self.numerics.dualTimeSteppingPrimal(self.Nt, self.dt, self.intIter, self.dt, self.init, self.T0, self. T1)
    #print("Primal, ", end="")
  
  def calculateAdjoint(self):
    """Computes Adjoint solution."""
    l_initial = np.zeros(self.mesh.numnodes)
    # TODO: check if that assumption is correct, or l_l=l_r=0 !
    l_left = self.T0
    l_right = self.T1
    djdu = self.alpha
    #TODO: Dow we have to set that? Evokes non-zero Residual of boundary dofs
    #djdu[0] = 0
    #djdu[-1] = 0

    self.adjoint_solution, self.adjointRes = self.numerics.dualTimeSteppingAdjoint(self.Nt, self.dt, self.intIter, self.dt, l_initial, l_left, l_right, djdu)
    #print("Adjoint, ", end="")

  def calculateDiffusionDerivative(self):
    """Computes dJ/da."""
    self.Dderivative = self.numerics.calculate_dJda(self.Nt, self.dt, self.full_solution, self.adjoint_solution)
    #print("Derivative. ", end="")

  def calculateDiffusionDerivative_ad(self):
    """Compute dR/dD via ad and the dJ/d alpha.  
    Args:
      None
    Returns:
      np.ndarray(n-2,n): dR/dD
      np.ndarray(n,): dJ/d alpha
    """ 
    T = self.full_solution[0]
    Dderivative = np.zeros(self.mesh.numnodes-2)
    for timestep in range(self.Nt-1): # -1 as 0th timestep is initial solution
      timestep = 10
      # compute flow residual
      self.D = adnumber(self.D)
      #B = self.compute1stOrderSpaceDiscretization()
      A = self.compute2ndOrderSpaceDiscretization()
      Res = A@self.full_solution[timestep+1][1:len(T)-1]
      dAdD = jacobian(Res, self.D)
      print("dadd",dAdD,"end",Res)
      rhs = self.rhs

      R = rhs - A@self.full_solution[timestep+1][1:len(T)-1] - B@self.full_solution[timestep+1][1:len(T)-1] # calculate spatial residuum 
      T_t = (self.full_solution[timestep+1]- self.full_solution[timestep])/self.dt # residuum of time dependent PDE term dT/dt

      Residual = T_t[1:-1]-R
      # retrieve the jacobain
      dRdD = jacobian(Residual,self.D)
      dRdD = np.asarray(dRdD)
      #.................................................................#
      print(dRdD)
      raise Exception('Stop')
      #.................................................................#
      Dderivative += self.adjoint_solution[timestep+1].T@dRdD[:,1:-1]

    Dderivative *= self.dt

    return Dderivative

  def computeTimestep(self):
    CFL = 0.01
    return CFL * (self.mesh.dx**2/2/np.amax(self.D)) # timestep size, after CFL   

  def ObjectiveFunction(self):
    """Calculate objective function.
    
    integrate(time) integrate(space) [alpha*temp] dx dt

    Args:
      None
    Returns:
      float: value objective function
    """
    obj = 0.
    for n in range(self.Nt):
      for i in range(len(self.alpha)):
        obj += self.mesh.dx * self.alpha[i] * self.full_solution[n][i] 
      obj *= self.dt

    self.obj = obj
    return obj

#############################################################################################
class Optimizer:
  """Performs multiple Design Iterations and stores given Visualizations."""
  def __init__(self, evaluateObjectiveFunction, computeDerivative, initial_DesignVars, num_DesignCycles, factorDesignVarUpdate):
    """
    Args:
      evaluateObjectiveFunction (function handle): returns J(alpha)
      computeDerivative (function handle): returns dJ/dalpha
      num_DesignCycles (int): number of Design cycles 
      factorDesignVarUpdate (float): multiplier for the DesignVar Update 
    """
    self.evaluateObjectiveFunction = evaluateObjectiveFunction
    self.computeDerivative = computeDerivative
    self.alpha = initial_DesignVars
    self.num_DesignCycles = num_DesignCycles
    self.factorDesignVarUpdate = factorDesignVarUpdate
    
    self.ObjectiveFunction = np.zeros(self.num_DesignCycles)
    self.Sensitivities = [] # list of np.ndarrays
    self.DiffusionCoefficients = [] # list of np.ndarrays
    
  def optimize(self):
    """docstring."""
    for i in range(self.num_DesignCycles):
      print("#--------------------------------------------#\nDesign Iteration: ", i)
      self.DiffusionCoefficients.append(copy.copy(self.alpha))
      obj, sim = self.evaluateObjectiveFunction(self.alpha)
      self.ObjectiveFunction[i] = obj
      print("Obj.Func: ", obj)

      dJdalpha = self.computeDerivative(sim)
      self.Sensitivities.append(dJdalpha)      

      self.alpha = self.updateDesignVars(copy.copy(self.alpha), dJdalpha)

      VisuEachIteration = True
      if VisuEachIteration == True:
        filename = 'Transient_1D_heat_' + str(i)
        self.VisuSingleOptIteration(sim, filename)
    
    self.VisualizeOptimizerVars(sim)

  def updateDesignVars(self, alpha, dJdalpha):
    """docstring.

    Args:
      dJdalpha (array of floats): Derivative of obj.func. wrt to Design variables
    Returns:
      None
    """
    alpha += self.factorDesignVarUpdate * dJdalpha
    return alpha

  def VisualizeOptimizerVars(self, sim):
    """docstring."""
    VisuNew = Visualization.Visualization(sim)
    VisuNew.ObjectiveFunction(False, True, self.num_DesignCycles, self.ObjectiveFunction)
    VisuNew.Sensitivities(self.Sensitivities, show=False, save=True)
    VisuNew.DiffusionCoefficient(self.DiffusionCoefficients, show=False, save=True)
    
  def VisuSingleOptIteration(self, sim, name):
    """docstring."""
    sim.name = name
    VisuNew = Visualization.Visualization(sim)
    VisuNew.Residuals(save=False,show=False)
    VisuNew.Animate_Primal(show=False, save=False)
    VisuNew.Animate_Adjoint(show=False, save=False)
    VisuNew.Animate_Primal_and_Adjoint(show=False, save=False)
    VisuNew.PrimalAdjointSensitivitiesDiffusivity(show=False, save=True)

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
      # Visualization
      VisuNew = Visualization.Visualization(sim)
      VisuNew.PrimalAdjointSensitivitiesDiffusivity(show=False, save=True)
      if i==2e4:
        plt.plot(sim.mesh.X,D)
        plt.show()


    VisuNew = Visualization.Visualization(sim)
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
      mesh = mesh1D(begin=0, end=35, numnodes=num_nodes)
      init = np.array([math.exp(-(x-15)**2) for x in getattr(mesh,'X')])
      alpha = np.array([math.exp(-(x-20)**2) for x in getattr(mesh,'X')]) # for objective function
      sim = TransHeatEq1D(filename, mesh, init, alpha, D, T0=0, T1=0, Nt=4000)
      # Solver
      sim.calculatePrimal()
      J = sim.ObjectiveFunction()
      return J, sim

    def derivative(sim):
      sim.calculateAdjoint()
      sim.calculateDiffusionDerivative()
      return sim.Dderivative
    
    obj_handle = lambda alpha: obj(alpha)
    derivative_handle = lambda sim: derivative(sim)
    initial_DesignVars = np.ones(num_nodes)*2.

    opt = Optimizer(obj_handle, derivative_handle, initial_DesignVars, num_DesignCycles=1, factorDesignVarUpdate=10)
    opt.optimize()

  if False:
    print("Computing FD Sensitivities.")
    num_nodes = 101
    def primal(D):
      # Preprocessing
      filename = "Trans_1D_Diffusion"
      mesh = mesh1D(begin=0, end=35, numnodes=num_nodes)
      init = np.array([math.exp(-(x-15)**2) for x in getattr(mesh,'X')])
      alpha = np.array([math.exp(-(x-20)**2) for x in getattr(mesh,'X')]) # for objective function
      sim = TransHeatEq1D(filename, mesh, init, alpha, D, T0=0, T1=0, Nt=4000)
      # Solver
      sim.calculatePrimal()
      J = sim.ObjectiveFunction()
      return J
    primal_handle = lambda alpha: primal(alpha)

    D = np.ones(num_nodes)*2.
    print("obj: ", primal_handle(D))

    FD_sens = FD_sensitivities.FD_sensitivities(primal_handle, D, gradientMethod='forward')
    dJda = FD_sens.calculateSensitivities()    
    mesh = mesh1D(begin=0, end=35, numnodes=num_nodes)
    plt.plot(mesh.X, dJda)
    plt.grid(True, axis='y')
    plt.savefig("FD_sens", bbox_inches='tight')
    plt.show()

    FD_sens.Plot_Sensitivities(mesh.X)

  if True: # with Numerics2
    num_nodes = 101
    Nt = 4000
    mesh = mesh1D(begin=0, end=35, numnodes=num_nodes)
    D = np.ones(num_nodes)*2.
    init = np.array([math.exp(-(x-15)**2) for x in getattr(mesh,'X')])
    alpha = np.array([math.exp(-(x-20)**2) for x in getattr(mesh,'X')]) # for objective function
    sim = Transient_1D_Diffusion(mesh, D, init, 0, 0 , Nt, alpha)
    t0=time.time()
    sim.calculatePrimal()
    t1=time.time()
    print(sim.calculateObjectiveFunction())
    sim.calculateAdjoint()
    t2=time.time()
    sim.calculateSensitivities()
    t3=time.time()
    print(t1-t0,t2-t1,t3-t2)

    VisuNew = Visualization.Visualization(sim)
    VisuNew.Residuals(save=True,show=False)
    VisuNew.Animate_Primal(show=False, save=False)
    VisuNew.Animate_Adjoint(show=False, save=False)
    VisuNew.Animate_Primal_and_Adjoint(show=False, save=False)
    VisuNew.PrimalAdjointSensitivitiesDiffusivity(show=False, save=True)
    #VisuNew.ObjectiveFunction(False, True, cycles, Objective_Function)
    #VisuNew.Sensitivities(Sensitivities, show=False, save=True)
    #VisuNew.DiffusionCoefficient(DiffusionCoefficients, show=False, save=True)
    a =1

