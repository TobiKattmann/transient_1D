import numpy as np
from copy import copy
import matplotlib.pyplot as plt

import unittest

"""
This class computes 1st derivatives (gradients) of a scalar function
wrt to an input vector of size n.

Translated for CFD-optimization: computes the derivative of a scalar objective 
function wrt to the design variables which can be a vector of size n.
"""
#########################################################################################

class Tests_FD_sensitivities(unittest.TestCase):
  """Unit Tests for FD_sensitivities."""
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
    FD_sens = FD_sensitivities(linearFunction_handle, design_vars, 'forward')
    numerical_result_1 = FD_sens.calculateSensitivities()
    analytical_result_1 = np.array([0., 1., 2., 3.])
    self.assertTrue(np.allclose(numerical_result_1, analytical_result_1))
    # input ones
    design_vars = np.ones(4)
    FD_sens = FD_sensitivities(linearFunction_handle, design_vars, 'forward')
    numerical_result_2 = FD_sens.calculateSensitivities()
    analytical_result_2 = np.array([0., 1., 2., 3.])
    self.assertTrue(np.allclose(numerical_result_2, analytical_result_2))
  
  def test_quadratic_function(self):
    def linearFunction(alpha):
      """Simple linear func J = 0*x0+1*x1+2*x2+3*x3, alpha=x, dJ/dalpha = (0 1 2 3) for every x."""
      J = 0
      factors = np.array([0,1,2,3])
      for alpha_i, factor in zip(alpha, factors):
        J += factor * alpha_i**2
      return J
    linearFunction_handle = lambda alpha: linearFunction(alpha)  
    # input zeros
    design_vars = np.zeros(4)
    FD_sens = FD_sensitivities(linearFunction_handle, design_vars, 'central')
    numerical_result_1 = FD_sens.calculateSensitivities()
    analytical_result_1 = np.array([0., 0., 0., 0.])
    self.assertTrue(np.allclose(numerical_result_1, analytical_result_1))
    # input ones
    design_vars = np.ones(4)
    FD_sens = FD_sensitivities(linearFunction_handle, design_vars, 'central')
    numerical_result_2 = FD_sens.calculateSensitivities()
    analytical_result_2 = np.array([0., 2., 4., 6.])
    self.assertTrue(np.allclose(numerical_result_2, analytical_result_2))

#########################################################################################
class FD_sensitivities:
  def __init__(self, evaluateObjFunc,  alpha, gradientMethod='forward'):
    """Necessary imputs are method for obj.func. evaluation and initial design vars.
    
    Args:
      evaluateObjFunc (lambda function handle): method to evaluate obj.func. for aritrary alpha
      alpha (np.ndarray(n,) of floats): state vector at which the derivative is evaluated
      gradientMethod (string): method for derivative calculation (1st: forward, central, 2nd: None)
    """
    self.evaluateObjFunc = evaluateObjFunc # lambda function handle
    self.alpha = alpha
    self.delta_alpha = np.max([0.001, 0.001*np.mean(np.absolute(alpha))] ) # TODO: more sophisticated method for dalpha computation necessary
    self.delta_alpha = 0.0001
    print("delta: ",self.delta_alpha)
    self.gradientMethod = gradientMethod # forward, central
    if gradientMethod == 'forward':
      self.J_init = self.evaluateObjFunc(self.alpha)
    
  def calculateSensitivities(self):
    """Main driver of the class, calculates dJdalpha.

    Returns:
      np.ndarray(n,) = dJ/dalpha
    """
    dJdalpha = np.zeros(len(self.alpha))
    
    for i in range(len(self.alpha)):
      dJdalpha[i] = self.computeSingleSensitivity(i)
      if True:
        print("Design Variable: ", i, "| dJ/da_i= ", dJdalpha[i])

    self.dJdalpha = copy(dJdalpha) 

    return dJdalpha
    
  def computeSingleSensitivity(self, i):
    """Returns derivative for single design var.
    
    Args:
      i (int): indicates ith entry of design vars. alpha
    Returns:
      float: dJ/dalpha_i
    """
    if self.gradientMethod =='forward':
      dJdalpha_i = self.forward_1st_order_derivative(i, self.delta_alpha)
      
    elif self.gradientMethod == 'central':
      dJdalpha_i = self.central_2nd_order_derivative(i, self.delta_alpha)
    
    return dJdalpha_i
    
  def forward_1st_order_derivative(self, i, delta_alpha_i):
    """Compute 1st derivative dJ/dalpha_i with forward FD.
    
    dJ/dalpha_i = [J(alpha + d alpha_i) - J(alpha)] / [d alpha_i] 
    
    Args:
      i (int): indicates ith entry of design vars. alpha
      delta_alpha_i (float): incremental increase of single design var. alpha_i
    Returns:
      float: derivative of obj.func. wrt to single design var. alpha_i
    """
    # Create new design variable vector
    plus_alpha = copy(self.alpha)
    plus_alpha[i] += delta_alpha_i
    # Create value for objective function with new design vars
    J_plus_alpha_i = self.evaluateObjFunc(plus_alpha)
    dJdalpha_i = (J_plus_alpha_i - self.J_init) / (delta_alpha_i)
    return dJdalpha_i

  def central_2nd_order_derivative(self, i, delta_alpha_i):
    """Compute 1st derivative dJ/dalpha_i with central FD.
    
    dJ/dalpha_i = [J(alpha + d alpha_i) - J(alpha - d alpha_i)] / [2*d alpha_i] 
    
    Args:
      i (int): indicates ith entry of design vars. alpha
      delta_alpha_i (float): incremental increase/decrease of single design var. alpha_i
    Returns:
      float: derivative of obj.func. wrt to single design var. alpha_i
    """
    # Create new design variable vectors
    plus_alpha = copy(self.alpha)
    plus_alpha[i] += delta_alpha_i

    minus_alpha = copy(self.alpha)
    minus_alpha[i] -= delta_alpha_i

    # Create values for objective function with new design vars
    J_plus_alpha_i = self.evaluateObjFunc(plus_alpha)
    J_minus_alpha_i = self.evaluateObjFunc(minus_alpha)
    
    dJdalpha_i = (J_plus_alpha_i - J_minus_alpha_i) / (2*delta_alpha_i)
    return dJdalpha_i
    
  def Plot_Sensitivities(self, X_axis=None, show=False, save=True):
    """Plots and/or saves Sensitivitiy vector dJ/dalpha."""
    plt.plot(X_axis ,self.dJdalpha, label="FD_sensitivities")

    plt.title('Sensitivities')
    plt.xlabel('Design Variable')
    plt.grid(True, axis='both')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    fullImagePath = ''.join(["./", "FD_Sensitivities.png"])
    if save: plt.savefig(fullImagePath, bbox_inches='tight')
    if show: plt.show()
    plt.close("all")
    plt.plot()
#########################################################################################
if __name__ == '__main__':
  unittest.main() 
