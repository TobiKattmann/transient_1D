import unittest
import numpy as np
import math
#import matplotlib.pyplot as plt

import NEW_transient_1D_heat as solver

#############################################################################################
class Tests_Primal(unittest.TestCase):
  """Unittests for the primal 1D solver. Contains:

  -> ZeroBCandINIT
  -> LinearSolution
  -> LinearSolution2
  -> FundamentalSolution
  """
  def test_ZeroBCandINIT(self): 
    """All zero Test. BC, INIT is zero, solution is all zero at all times."""
    filename = 'UNITTEST_ZeroBCandINIT'
    num_nodes = 11
    mesh = solver.mesh1D(begin=0, end=1, numnodes=num_nodes)
    init = np.zeros(num_nodes)
    alpha = None
    D = np.ones(num_nodes)
    sim = solver.TransHeatEq1D(filename, mesh, init, alpha, D, T0=0, T1=0, Nt=2)
    sim.calculatePrimal()
    self.assertTrue(np.allclose(sim.full_solution[0], np.zeros(num_nodes)))
    self.assertTrue(np.allclose(sim.full_solution[1], np.zeros(num_nodes)))

  def test_LinearSolution(self):
    """Linear Solution. INIT is zero. BC are u(0)=0 and u(0)=1."""
    filename = 'UNITTEST_LinearSolution'
    num_nodes = 10
    mesh = solver.mesh1D(begin=0, end=1, numnodes=num_nodes)
    init = np.zeros(num_nodes)
    alpha = None
    D = np.ones(num_nodes)
    sim = solver.TransHeatEq1D(filename, mesh, init, alpha, D, T0=0, T1=1, Nt=2000)
    sim.calculatePrimal()
    self.assertTrue(np.allclose(sim.full_solution[1999], np.linspace(0,1,num_nodes), rtol=1e-3, atol=1e-3))

  def test_LinearSolution2(self):
    """Linear Solution. INIT is zero. BC are u(0)=1 and u(0)=0."""
    filename = 'UNITTEST_LinearSolution2'
    num_nodes = 10
    mesh = solver.mesh1D(begin=0, end=1, numnodes=num_nodes)
    init = np.zeros(num_nodes)
    alpha = None
    D = np.ones(num_nodes)
    sim = solver.TransHeatEq1D(filename, mesh, init, alpha, D, T0=1, T1=0, Nt=2000)
    sim.calculatePrimal()
    self.assertTrue(np.allclose(sim.full_solution[1999], np.linspace(1,0,num_nodes), rtol=1e-3, atol=1e-3))

  def test_FundamentalSolution(self):
    """Transient fundamental solution."""
    filename = 'UNITTEST_FundamentalSolution'
    num_nodes = 101
    mesh = solver.mesh1D(begin=0, end=35, numnodes=num_nodes)
    init = np.array([math.exp(-(x-15)**2) for x in getattr(mesh,'X')])
    alpha = None
    D = np.ones(num_nodes)
    sim = solver.TransHeatEq1D(filename, mesh, init, alpha, D, T0=0, T1=0, Nt=2000)
    sim.calculatePrimal()

    eta = np.sqrt((1. + 4*D*sim.dt*sim.Nt)/1.)
    analytical_solution = np.exp(-((np.array(sim.mesh.X)-15.)**2)/eta**2)/eta
    #plt.plot([sim.mesh.X for i in range(2)], [sim.full_solution[1999], analytical_solution]) pretty solution again
    self.assertTrue(np.allclose(sim.full_solution[1999], analytical_solution, rtol=1e-3, atol=1e-2))

class Tests_Adjoint(unittest.TestCase):
  """Unittests for adjoint 1D solver. Contains:

  -> FundamentalSolution
  """
  def test_FundamentalSolution(self): 
    """All zero Test. BC, INIT is zero, solution is all zero at all times."""
    filename = 'UNITTEST_ZeroBCandINIT'
    num_nodes = 11
    mesh = solver.mesh1D(begin=0, end=1, numnodes=num_nodes)
    init = np.array([math.exp(-(x-15)**2) for x in getattr(mesh,'X')])
    alpha = np.array([math.exp(-(x-20)**2) for x in getattr(mesh,'X')]) # for objective function
    D = np.ones(num_nodes)
    sim = solver.TransHeatEq1D(filename, mesh, init, alpha, D, T0=0, T1=0, Nt=2)
    sim.calculatePrimal()
    sim.calculateAdjoint()

    eta = np.sqrt((1. + 4*D*sim.dt*sim.Nt)/1.)
    analytical_solution = -np.exp(-((np.array(sim.mesh.X)-20.)**2)/eta**2)/eta
    self.assertTrue(np.allclose(sim.adjoint_solution[0], analytical_solution, rtol=1e-3, atol=1e-2))

class Tests_Sensitivities(unittest.TestCase):
  """Unittest for Sensitivities. Contains:

  -> ...
  """
  pass

#############################################################################################
if __name__ == '__main__':
  unittest.main()