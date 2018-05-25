import numpy as np
import unittest

import sys
sys.path.append('../src')
import FD_sensitivities as FD
#---------------------------------------------------------------------------------------#
"""
Unittests for FD_sensitivities module in ../src
1st order forward FD are exact for polynomials of degree 1 (linear func)
2nd order central FD are exact for polynomials of degree 2 (quadratic func)
"""
#---------------------------------------------------------------------------------------#
class Tests_FD_sensitivities(unittest.TestCase):
  """Unit Tests for FD_sensitivities."""
#---------------------------------------------------------------------------------------#
  def test_linear_function(self):
    def linearFunction(alpha):
      """Simple linear func J = 0*x0+1*x1+2*x2+3*x3, alpha=x, dJ/dalpha = (0 1 2 3) for every x."""
      J = 0
      factors = np.array([0,1,2,3])
      for alpha_i, factor in zip(alpha, factors):
        J += factor * alpha_i
      return J
    linearFunction_handle = lambda alpha: linearFunction(alpha)  
    # input zeros
    design_vars = np.zeros(4)
    FD_sens = FD.FD_sensitivities(linearFunction_handle, design_vars, 'forward')
    numerical_result_1 = FD_sens.calculateSensitivities()
    analytical_result_1 = np.array([0., 1., 2., 3.])
    self.assertTrue(np.allclose(numerical_result_1, analytical_result_1))
    # input ones
    design_vars = np.ones(4)
    FD_sens = FD.FD_sensitivities(linearFunction_handle, design_vars, 'forward')
    numerical_result_2 = FD_sens.calculateSensitivities()
    analytical_result_2 = np.array([0., 1., 2., 3.])
    self.assertTrue(np.allclose(numerical_result_2, analytical_result_2))
  
#---------------------------------------------------------------------------------------#
  def test_quadratic_function(self):
    def quadraticFunction(alpha):
      """Simple quadratic func J = 0*x0+1*x1**2+2*x2**2+3*x3**2, alpha=x, dJ/dalpha = (0 2 4 6) for x_i=1."""
      J = 0
      factors = np.array([0,1,2,3])
      for alpha_i, factor in zip(alpha, factors):
        J += factor * alpha_i**2
      return J
    quadraticFunction_handle = lambda alpha: quadraticFunction(alpha)  
    # input zeros
    design_vars = np.zeros(4)
    FD_sens = FD.FD_sensitivities(quadraticFunction_handle, design_vars, 'central')
    numerical_result_1 = FD_sens.calculateSensitivities()
    analytical_result_1 = np.array([0., 0., 0., 0.])
    self.assertTrue(np.allclose(numerical_result_1, analytical_result_1))
    # input ones
    design_vars = np.ones(4)
    FD_sens = FD.FD_sensitivities(quadraticFunction_handle, design_vars, 'central')
    numerical_result_2 = FD_sens.calculateSensitivities()
    analytical_result_2 = np.array([0., 2., 4., 6.])
    self.assertTrue(np.allclose(numerical_result_2, analytical_result_2))

#---------------------------------------------------------------------------------------#
if __name__ == '__main__':
  unittest.main()
