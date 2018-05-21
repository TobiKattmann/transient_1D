import numpy as np
import math
import time
import copy 
import matplotlib.pyplot as plt

import mesh_1D
import transient_1D
import visualization

num_nodes = 101
Nt = 4000
mesh = mesh_1D.mesh1D(begin=0, end=35, numnodes=num_nodes)
D = np.ones(num_nodes)*2.
init = np.array([math.exp(-(x-15)**2) for x in getattr(mesh,'X')])
alpha = np.array([math.exp(-(x-20)**2) for x in getattr(mesh,'X')]) # for objective function
sim = transient_1D.Transient_1D_Diffusion(mesh, D, init, 0, 0 , Nt=4000, alpha=alpha)
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
VisuNew.Animate_Primal(show=False, save=False)
VisuNew.Animate_Adjoint(show=False, save=False)
VisuNew.Animate_Primal_and_Adjoint(show=False, save=False)
VisuNew.PrimalAdjointSensitivitiesDiffusivity(show=False, save=True)
#VisuNew.ObjectiveFunction(False, True, cycles, Objective_Function)
VisuNew.Sensitivities([sim.Dderivative], show=True, save=True)
#VisuNew.DiffusionCoefficient(DiffusionCoefficients, show=False, save=True)

adjoint_check = True
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
