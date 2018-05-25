import numpy as np
import math

import mesh_1D
import transient_1D
import optimizer

#cleanup old files
#os.system('rm -v ../NEWimages-tobi-code/*')

#---------------------------------------------------------------------------------------#
print("Optimzer called")
num_nodes = 101
def obj(D):
  # Preprocessing
  filename = "Trans_1D_Diffusion"
  mesh = mesh_1D.mesh1D(begin=0, end=35, numnodes=num_nodes)
  init = np.array([math.exp(-(x-15)**2) for x in getattr(mesh,'X')])
  alpha = np.array([math.exp(-(x-20)**2) for x in getattr(mesh,'X')]) # for objective function
  #sim = TransHeatEq1D(filename, mesh, init, alpha, D, T0=0, T1=0, Nt=4000)
  sim = transient_1D.Transient_1D_Diffusion(mesh, D, init, 0, 0 , Nt=4000, alpha=alpha)
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

#---------------------------------------------------------------------------------------#
opt = optimizer.Optimizer(obj_handle, derivative_handle, initial_DesignVars, num_DesignCycles=3, factorDesignVarUpdate=10)
opt.optimize()
