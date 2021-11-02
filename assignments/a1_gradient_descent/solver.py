import numpy as np
import sys
sys.path.append("../../..")

from optimization_algorithms.interface.nlp_solver import NLPSolver

class Gradient(NLPSolver):
    def __init__(self):
        self.tolerance = 1e-3 # theta
        self.verbose = True

    def evaluate(self, x):
        a, b = self.problem.evaluate(x)
        return a[0], b[0]

    def fixed_step_search(self, x):
        iteration = 0
        max_iteration = 40
        step_sz = np.random.uniform(0.01, 0.02) # alpha
        # x_0 = x.copy() # needs fix

        if(self.verbose): 
            print("commencing gradient descent with fixed stepsize (alpha = %.5f)" % step_sz)

        while True:
            if(iteration == max_iteration):
                iteration = 0 
                step_sz = np.random.uniform(0.01, 0.02)
                # x = x_0.copy() # needs fix
                if(self.verbose):
                    print("continuing gradient descent with new fixed stepsize (alpha = %.5f)" % step_sz)
            f_x, ff_x = self.evaluate(x)

            if(self.verbose):
                print("iteration %d: x = %s, f(x) = %.5f" % (iteration, np.array2string(x,precision=2), f_x))
                iteration += 1
            
            if abs(f_x) < self.tolerance:
                return x

            x -= step_sz * ff_x 

    def line_search(self, x) :
        iteration = 0
        step_sz = 1.0 # alpha
        step_inc = 1.2 # phi_alpha^plus 
        step_dec = 0.5 # phi_alpha^minus
        step_dec_min = 0.01 # phi_ls 

        if(self.verbose):
            print("commencing gradient descent with line search")

        while True:
            f_x, ff_x = self.evaluate(x) 

            if(self.verbose):
                print("iteration %d: x = %s, f(x) = %.5f, alpha = %.5f" % (iteration, np.array2string(x,precision=2), f_x, step_sz))
                iteration += 1

            if abs(f_x) < self.tolerance:
                return x

            step_dir = -ff_x / np.linalg.norm(ff_x)             
            f_xs, ff_xs = self.evaluate(x + step_sz * step_dir)
            while np.all(np.greater(f_xs, f_x + step_dec_min * np.transpose(ff_x) @ (step_sz * step_dir))):
                f_x, ff_x = self.evaluate(x) 
                f_xs, ff_xs = self.evaluate(x + step_sz * step_dir)
                step_sz *= step_dec 
            x += step_sz * step_dir
            step_sz *= step_inc
