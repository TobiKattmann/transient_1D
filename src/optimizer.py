import numpy as np
import copy

import visualization

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
    VisuNew = visualization.Visualization(sim)
    VisuNew.ObjectiveFunction(False, True, self.num_DesignCycles, self.ObjectiveFunction)
    VisuNew.Sensitivities(self.Sensitivities, show=False, save=True)
    VisuNew.DiffusionCoefficient(self.DiffusionCoefficients, show=False, save=True)
    
  def VisuSingleOptIteration(self, sim, name):
    """docstring."""
    sim.name = name
    VisuNew = visualization.Visualization(sim)
    VisuNew.Residuals(save=False,show=False)
    VisuNew.Animate_Primal(show=False, save=False)
    VisuNew.Animate_Adjoint(show=False, save=False)
    VisuNew.Animate_Primal_and_Adjoint(show=False, save=False)
    VisuNew.PrimalAdjointSensitivitiesDiffusivity(show=False, save=True)

if __name__ == '__main__':
  print("Nothing to excecute here. But there is no error either, that's fine.")
