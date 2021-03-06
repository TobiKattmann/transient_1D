import copy
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

import transient_1D
import mesh_1D
import FD_sensitivities

#---------------------------------------------------------------------------------------#
print("Computing FD Sensitivities.")
num_nodes = 101
def primal(D):
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
  return J
primal_handle = lambda alpha: primal(alpha)

D = np.ones(num_nodes)*2.
print("obj: ", primal_handle(D))

FD_sens = FD_sensitivities.FD_sensitivities(primal_handle, D, gradientMethod='forward')
FD_sens.print_derivatives = True
dJda = FD_sens.calculateSensitivities()    

mesh = mesh_1D.mesh1D(begin=0, end=35, numnodes=num_nodes)
FD_sens.writeSensToFile(mesh.X)

# Read .csv data back into np.array
#df = pd.read_csv('FD_sens.csv')
#x = df.iloc[:,1:2].values
#djda = df.iloc[:,2:3].values
#print(x,djda)

#---------------------------------------------------------------------------------------#
plt.plot(mesh.X, dJda)
plt.grid(True, axis='y')
plt.savefig("FD_sens", bbox_inches='tight')
#plt.show()

FD_sens.Plot_Sensitivities(mesh.X)
FDderivative = copy.copy(dJda)
