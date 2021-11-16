import numpy as np
import sys
sys.path.append("../../..")

from optimization_algorithms.interface.mathematical_program import MathematicalProgram

class Feature(MathematicalProgram):

    def __init__(self,a,c):
        self.a = a
        self.c = c

    def evaluate(self, x):
        f = np.array([np.sin(self.a*x[0]),
                      np.sin(self.a*self.c*x[1]),
                      2*x[0],
                      2*self.c*x[1]])
        y = f.T @ f        
        J = np.array([[self.a*np.cos(self.a*x[0]), 0.0                                     ],
                      [0.0,                        self.a*self.c*np.cos(self.a*self.c*x[1])], 
                      [1.0,                        0.0                                     ],
                      [0.0,                        self.c                                  ]])
        G = 2 * J.T @ f[:,np.newaxis]
        return np.array([y]), G.reshape(1,-1) # f_x, ff_x    

    def getFHessian(self, x):
        J = np.array([[self.a*np.cos(self.a*x[0]), 0.0                                     ],
                      [0.0,                        self.a*self.c*np.cos(self.a*self.c*x[1])], 
                      [1.0,                        0.0                                     ],
                      [0.0,                        self.c                                  ]])
        return 2 * J.T @ J # Gauß-Newton approximation for Hessian
