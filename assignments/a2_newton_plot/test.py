import numpy as np
import unittest
import sys
sys.path.append("../../..")

from optimization_algorithms.utils.finite_diff import *
from optimization_algorithms.interface.nlp_solver import  NLPSolver
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.mathematical_program_traced import  MathematicalProgramTraced

from functions import Square, Hole 
from solver import Gradient 

class testGradientDescent(unittest.TestCase):

    dim = 100 
    a = 0.1
    C = np.eye(dim) 
    Cinv = np.eye(dim) 
    for i in range(dim):
        C[i,i] *= 10**(i/~-dim)
        Cinv[i,i] /= 10**(i/~-dim)

    # print(C)
    # print(Cinv)

    def testSquareFunction(self): 
        print("\n---[testing square function (dim=%d) w/o C^-1]------------------------" % self.dim)
        square = MathematicalProgramTraced(Square(self.C)) 
        gradient = Gradient()
        gradient.setProblem((square))
        x_0 = np.array([1.0 for x in range(self.dim)])
        output = gradient.line_search(x_0)
        self.assertTrue(True)

    def testHoleFunction(self):
        print("\n---[testing hole function (dim=%d) w/o C^-1]--------------------------" % self.dim)
        hole = MathematicalProgramTraced(Hole(self.C,self.a))
        gradient = Gradient()
        gradient.setProblem((hole))
        x_0 = np.array([1.0 for x in range(self.dim)])
        output = gradient.line_search(x_0)
        self.assertTrue(True)

    def testSquareFunctionCinv(self): 
        print("\n---[testing square function (dim=%d) w/ C^-1]-------------------------" % self.dim)
        square = MathematicalProgramTraced(Square(self.C)) 
        gradient = Gradient()
        gradient.setProblem((square))
        x_0 = np.array([1.0 for x in range(self.dim)])
        output = gradient.line_search(x_0,self.Cinv)
        self.assertTrue(True)

    def testHoleFunctionCinv(self):
        print("\n---[testing hole function (dim=%d) w/ C^-1]---------------------------" % self.dim)
        hole = MathematicalProgramTraced(Hole(self.C,self.a))
        gradient = Gradient()
        gradient.setProblem((hole))
        x_0 = np.array([1.0 for x in range(self.dim)])
        output = gradient.line_search(x_0,self.Cinv)
        self.assertTrue(True)

    def testSquareFunctionNewton(self): 
        print("\n---[testing square function (dim=%d) w/o C^-1]------------------------" % self.dim)
        square = MathematicalProgramTraced(Square(self.C)) 
        gradient = Gradient()
        gradient.setProblem((square))
        x_0 = np.array([1.0 for x in range(self.dim)])
        output = gradient.newton(x_0)
        self.assertTrue(True)

    def testHoleFunctionNewton(self):
        print("\n---[testing hole function (dim=%d) w/o C^-1]--------------------------" % self.dim)
        hole = MathematicalProgramTraced(Hole(self.C,self.a))
        gradient = Gradient()
        gradient.setProblem((hole))
        x_0 = np.array([1.0 for x in range(self.dim)])
        output = gradient.newton(x_0)
        self.assertTrue(True)

if __name__ == "__main__":
   unittest.main()
