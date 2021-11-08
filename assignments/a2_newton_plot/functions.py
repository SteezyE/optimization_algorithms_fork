import numpy as np
import sys
sys.path.append("../../..")

from optimization_algorithms.interface.mathematical_program import MathematicalProgram

class Square(MathematicalProgram):

    def __init__(self,C):
        self.C = C

    def evaluate(self, x) :
        y = x @ self.C @ x[:,np.newaxis]       
        J = x @ self.C + x @ self.C.T 
        return np.array([y]), J.reshape(1,-1)

class Hole(MathematicalProgram):

    def __init__(self, C, a):
        self.C = C
        self.aa = a*a

    def evaluate(self, x) :
        y = (x @ self.C @ x[:,np.newaxis]) / (x @ self.C @ x[:,np.newaxis] + self.aa) 
        J = ((x @ self.C + x @ self.C.T) * ((x @ self.C @ x[:,np.newaxis] + self.aa) - (x @ self.C @ x[:,np.newaxis]))) / (x @ self.C @ x[:,np.newaxis] + self.aa)**2
        return np.array([y]), J.reshape(1,-1)
