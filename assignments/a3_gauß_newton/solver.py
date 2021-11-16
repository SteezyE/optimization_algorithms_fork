import numpy as np
import sys
sys.path.append("../../..")

from optimization_algorithms.interface.nlp_solver import NLPSolver

class Solver(NLPSolver):
    def __init__(self, itercnt=100):
        self.tolerance = 1e-4 # theta
        self.max_iter = itercnt 
        self.verbose = False 
        self.testconv = False

    def evaluate(self, x):
        a, b = self.problem.evaluate(x)
        return a[0], b[0]

    def gauss_newton(self, x):
        counter = 0
        iteration = 0
        step_sz = 1.0 # alpha
        step_inc = 1.2 # phi_alpha^plus 
        step_dec = 0.5 # phi_alpha^minus
        step_dec_min = 0.01 # phi_ls 
        eigen_shift = 1.0 # lambda

        if(self.verbose):
            print("commencing gauß-newton method with line search and fallback")

        while self.testconv or iteration < self.max_iter:
            f_x, ff_x = self.evaluate(x) 
            fff_x = self.problem.getFHessian(x)
            shift = eigen_shift * np.eye(x.shape[0])
            
            if(self.verbose):
                print("iteration %d: f(x) = %.5f, alpha = %.5f" % (iteration, f_x, step_sz))
                
            iteration += 1
            step_dir = np.linalg.solve(fff_x + shift, -ff_x)

            try:
                step_dir = np.linalg.solve(fff_x + shift, -ff_x)
                if ff_x @ step_dir > 0.0: step_dir = -ff_x / np.linalg.norm(ff_x) 
            except:
                step_dir = -ff_x / np.linalg.norm(ff_x)             

            if self.testconv and np.linalg.norm(step_sz * step_dir, ord=1) < self.tolerance:
                if counter == 10:
                    break 
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
            step_sz = min(step_inc*step_sz,1)
        return x

    def gradient(self, x):
        counter = 0
        iteration = 0
        step_sz = 1.0 # alpha
        step_inc = 1.2 # phi_alpha^plus 
        step_dec = 0.5 # phi_alpha^minus
        step_dec_min = 0.01 # phi_ls 

        if(self.verbose):
            print("commencing gradient descent with line search")

        while self.testconv or iteration < self.max_iter:
            f_x, ff_x = self.evaluate(x) 

            if(self.verbose):
                print("iteration %d: x = %s, f(x) = %.5f, alpha = %.5f" % (iteration, np.array2string(x,precision=2), f_x, step_sz))
                
            iteration += 1

            step_dir = -ff_x / np.linalg.norm(ff_x)             

            if self.testconv and np.linalg.norm(step_sz * step_dir, ord=1) < self.tolerance:
                if counter == 10:
                    break 
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
        return x
