import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Visualization:
  def __init__(self, Sim):
    """Hand over solver instance which contains all solution data."""
    self.sim = Sim
    self.folderpath = "../NEWimages-tobi-code/"
    
  def Residuals(self, show=False, save=True): 
    """Saves/Shows Primal and Adjoint Residuals."""
    plt.close("all")

    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.semilogy(self.sim.Res, label="Primal Res")
    ax1.set_title('Primal Residuals')
    ax1.set_xlabel('Iter (including internal Iteration)')
    ax1.set_ylabel('RMS')  
    ax1.grid(True, axis='y')
  
    ax2.semilogy(self.sim.adjointRes, label="Adjoint Res")
    ax2.set_title('Adjoint Residuals')
    ax2.set_xlabel('Iter (including internal Iteration)')
    ax2.set_ylabel('RMS')  
    ax2.grid(True, axis='both')
  
    fig.tight_layout()

    fullImagePath = ''.join([self.folderpath, "Residuals.png"])
    if save: plt.savefig(fullImagePath, bbox_inches='tight')
    if show: plt.show()

  def Animate_Primal(self, show=False, save=True):
    """Docstring."""
    plt.close("all")
    
    def update_line_primal(i):
      line.set_ydata(self.sim.full_solution[i])
      return line,
    
    fig = plt.figure(figsize=(12,5))
    ax = plt.axes()

    line, = ax.plot(self.sim.mesh.X,self.sim.full_solution[1])
    sol_ani = animation.FuncAnimation(fig, update_line_primal, self.sim.Nt, blit=True, interval=1)
    plt.plot(self.sim.mesh.X,self.sim.full_solution[1], label='initial')
    plt.plot(self.sim.mesh.X,self.sim.full_solution[-1], label='final')
    
    ax.set_title('Primal solution')
    ax.set_xlabel('u')
    ax.set_ylabel('x')
    ax.grid(True, axis='y')
      
    fig.tight_layout()
  
    fullImagePath = ''.join([self.folderpath, "Ani_primal.mp4"])
    if save: sol_ani.save(fullImagePath) # not available for movies :(
    if show: plt.show()

  def Animate_Adjoint(self, show=False, save=True):
    """Docstring."""
    plt.close("all")
    
    def update_line_adjoint(i):
      line.set_ydata(self.sim.adjoint_solution[i])
      return line,
    
    fig = plt.figure(figsize=(12,5))
    ax = plt.axes()

    line, = ax.plot(self.sim.mesh.X,self.sim.adjoint_solution[1])
    sol_ani = animation.FuncAnimation(fig, update_line_adjoint, self.sim.Nt, blit=True, interval=1)
    plt.plot(self.sim.mesh.X,self.sim.adjoint_solution[1], label='initial')
    plt.plot(self.sim.mesh.X,self.sim.adjoint_solution[-1], label='final')
    
    ax.set_title('Adjoint solution')
    ax.set_xlabel('l')
    ax.set_ylabel('x')
    ax.grid(True, axis='y')
      
    fig.tight_layout()
  
    fullImagePath = ''.join([self.folderpath, "Ani_Adjoint.mp4"])
    if save: sol_ani.save(fullImagePath) # not available for movies :(
    if show: plt.show()

  def Animate_Primal_and_Adjoint(self, show=False, save=True):
    """Docstring."""
    plt.close("all")
    
    fig = plt.figure(figsize=(12,5))
    ax = plt.axes(xlim=(self.sim.mesh.begin, self.sim.mesh.end), ylim=(-1,1))
    line, = ax.plot([], [])

    lines =[]
    lobj_primal = ax.plot([], [], c='red')[0]
    lobj_adjoint = ax.plot([], [], c='green')[0]
    lines.append(lobj_primal)
    lines.append(lobj_adjoint)
    
    def init():
      for line in lines:
        line.set_data([], [])
      return lines

    def update_line_primal(i):
      xlist = [self.sim.mesh.X, self.sim.mesh.X]
      ylist = [self.sim.full_solution[i], self.sim.adjoint_solution[i]]

      for lnum, line in enumerate(lines):
        line.set_data(xlist[lnum], ylist[lnum])
      return lines

    sol_ani = animation.FuncAnimation(fig, update_line_primal, init_func=init, frames=self.sim.Nt, blit=True, interval=1)
    plt.plot(self.sim.mesh.X,self.sim.full_solution[1], label='initial primal', c='black')
    plt.plot(self.sim.mesh.X,self.sim.full_solution[-1], label='final primal', c='grey')
    plt.plot(self.sim.mesh.X,self.sim.adjoint_solution[1], label='initial adjoint', c='black')
    plt.plot(self.sim.mesh.X,self.sim.adjoint_solution[-1], label='final adjoint', c='grey')
    
    ax.set_title('Primal and Adjoint solution')
    ax.set_xlabel('solution')
    ax.set_ylabel('x')
    ax.grid(True, axis='y')
    ax.legend()
      
    fig.tight_layout()
  
    fullImagePath = ''.join([self.folderpath, "Ani_Primal_and_Adjoint.mp4"])
    if save: sol_ani.save(fullImagePath) # not available for movies :(
    if show: plt.show()

  def PrimalAdjointSensitivitiesDiffusivity(self, show=False, save=True):
    """Docstring."""
    plt.close("all")
    
    fig = plt.figure(figsize=(12,5))
    ax = plt.axes()
    
    #ax.plot([self.sim.mesh.X for i in [0, -1]], [self.sim.full_solution[i] for i in [0,-1]])
    for i in [0, -1]:
      ax.plot(self.sim.mesh.X, self.sim.full_solution[i])
      ax.plot(self.sim.mesh.X, self.sim.adjoint_solution[i])
    plt.plot(self.sim.mesh.X, self.sim.alpha, label='Kernel')
    plt.plot(self.sim.mesh.X, self.sim.Dderivative/np.amax(np.absolute(self.sim.Dderivative)), label='scaled Sensitivities', c='red', linewidth=1)
  
    plt.grid(True, axis='y')
    plt.title('All in one')
    plt.xlabel('x')
    plt.legend()
    fig.tight_layout()

    fullImagePath = ''.join([self.folderpath, self.sim.name, ".png"])
    if save: plt.savefig(fullImagePath, bbox_inches='tight')
    if show: plt.show()
    plt.close("all")

  def ObjectiveFunction(self, show, save, num_cycles, objFunc):
    """Plots the obj.func. over multiple design cycles."""
    plt.plot(np.linspace(0, num_cycles, num_cycles), objFunc, marker='x')
    plt.title('Objective_Function')
    plt.xlabel('Design Iterations')
    plt.grid(True, axis='y')

    fullImagePath = ''.join([self.folderpath, "Objective_Function.png"])
    if save: plt.savefig(fullImagePath, bbox_inches='tight')
    if show: plt.show()
    plt.close("all")

  def Sensitivities(self, Sensitivities, show=False, save=True):
    """Plots the sensitivities (all, if multiple design cycles)."""
    for idx, Sensitivity in enumerate(Sensitivities):
      plt.plot(self.sim.mesh.X, Sensitivity, label=''.join(["Sensitivities ", str(idx)]))

    plt.title('Sensitivities')
    plt.xlabel('x')
    plt.grid(True, axis='both')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    fullImagePath = ''.join([self.folderpath, "Sensitivities.png"])
    if save: plt.savefig(fullImagePath, bbox_inches='tight')
    if show: plt.show()
    plt.close("all")

  def DiffusionCoefficient(self, DiffusionCoefficients, show=False, save=True):
    """Plots the diff.coeff. (all, if multiple Design cycles)."""
    for idx, DiffusionCoeff in enumerate(DiffusionCoefficients):
      plt.plot(self.sim.mesh.X, DiffusionCoeff, label=''.join(["Diffusion Coeff ", str(idx)]))

    plt.title('Diffusion Coefficients')
    plt.xlabel('x')
    plt.grid(True, axis='y')
  
    fullImagePath = ''.join([self.folderpath, "DiffusionCoefficient.png"])
    if save: plt.savefig(fullImagePath, bbox_inches='tight')
    if show: plt.show()
    plt.close("all")

  def template(self, show=False, save=False):
    """Docstring."""
    ...  
  
    fullImagePath = ''.join([self.folderpath, "image_name.png"])
    if save: plt.savefig(fullImagePath, bbox_inches='tight')
    if show: plt.show()
    plt.close("all")

if __name__ == '__main__':
  print("Nothing to excecute here. But there is no error either, that's fine.")

