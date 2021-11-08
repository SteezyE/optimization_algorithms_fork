import numpy as np
import sys
sys.path.append("../../..")

from optimization_algorithms.interface.nlp_solver import NLPSolver

class Gradient(NLPSolver):
    def __init__(self):
        self.tolerance = 1e-4 # theta
        self.verbose = True

    def evaluate(self, x):
        a, b = self.problem.evaluate(x)
        return a[0], b[0]

    def line_search(self, x, Cinv=None) :
        if Cinv is None: Cinv = np.eye(x.shape[0])
        counter = 0
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
                print("iteration %d: f(x) = %.5f, alpha = %.5f" % (iteration, f_x, step_sz))
                iteration += 1

            step_dir = -(Cinv @ ff_x) / np.linalg.norm(ff_x)             

            if np.linalg.norm(step_sz * step_dir, ord=1) < self.tolerance:
                if counter == 10:
                    return x
                else:
                    counter += 1
            else: 
                counter = 0

            f_xs, ff_xs = self.evaluate(x + step_sz * step_dir)
            while np.all(np.greater(f_xs, f_x + step_dec_min * np.transpose(ff_x) @ (step_sz * step_dir))):
                f_x, ff_x = self.evaluate(x) 
                f_xs, ff_xs = self.evaluate(x + step_sz * step_dir)
                step_sz *= step_dec 
            x += step_sz * step_dir
            step_sz *= step_inc
