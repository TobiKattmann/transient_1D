import numpy as np

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

if __name__ == '__main__':
  print("Nothing to excecute here. But there is no error either, that's fine.")
