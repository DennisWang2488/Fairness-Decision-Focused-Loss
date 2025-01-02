import numpy as np
import cvxpy as cp


class optModel:
    def __init__(self,x,r,c,Q,alpha):
        self.alpha = alpha
        self.Q = Q
        self.r = r
        self.c = c
        self.x = x
        self.num_data, self.num_items, self.num_features = x.shape
    def setObj(self, r, c):
        if self.alpha == 1:
            self.objective = cp.sum(cp.log(cp.multiply(r, self.d)))
        
        self.objective = cp.sum(cp.power(cp.multiply(r, self.d), 1 - self.alpha)) / (1 - self.alpha)
        self.constraints = [
            self.d >= 0,
            cp.sum(cp.multiply(c, self.d)) <= self.Q
        ]
        self.problem = cp.Problem(cp.Maximize(self.objective), self.constraints)


    def solve(self,closed=False):
        """
        A method to solve the optimization problem

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        opt_sol = []
        opt_val = []
        if closed:
            return self.solveC()

        for i in range(self.num_data):
            self.d = cp.Variable(self.num_items)
            self.setObj(self.r[i], self.c[i])
            self.problem.solve()
            opt_sol.append(self.d.value.reshape(1,self.num_items))
            opt_val.append(self.problem.value)

        opt_sol = np.concatenate(opt_sol)


        return opt_sol, opt_val 
    
    def solveC(self):
        if self.alpha == 1:
            return print("Work in progress")

        S = np.sum(self.c ** (1 - 1 / self.alpha) * self.r ** (-1 + 1 / self.alpha))
        opt_sol_c = (self.c ** (-1 / self.alpha) * self.r ** (-1 + 1 / self.alpha) * self.Q) / S
        opt_val_c = np.sum((self.r * opt_sol_c) ** (1 - self.alpha)) / (1 - self.alpha)
    
        return opt_sol_c, opt_val_c
    