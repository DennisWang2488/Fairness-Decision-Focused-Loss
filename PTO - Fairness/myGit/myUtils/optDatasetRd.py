import numpy as np
import torch
from torch.utils.data import Dataset
from optModelRd import optModel

class optDatasetRd(Dataset):
    def __init__(self, x, r, c, Q, alpha, num_items, num_data):
        self.x = x
        self.r = r
        self.c = c
        self.Q = Q
        self.alpha = alpha
        self.num_items = num_items
        self.num_data = num_data

        self.opt_solutions_solver = []
        self.opt_objective_values_solver = []
        self.opt_solutions_closed = []
        self.opt_objective_values_closed = []

        self._solve_optimization_problems()
        self._solve_optimization_problems_closed()

    def _solve_optimization_problems(self):
        n_data = self.r.shape[0]
        for i in range(n_data):
            model = optModel(
                num_items=self.num_items,
                num_data=self.num_data,
                alpha=self.alpha,
                Q=self.Q,
                r=self.r[i].reshape(-1),  # Ensure r has compatible shape
                c=self.c[i].reshape(-1)   # Ensure c has compatible shape
            )
            print(f"r[{i}]:", self.r[i])
            print(f"c[{i}]:", self.c[i])
            d_opt, opt_value = model.solveP()
            print(f"d_opt[{i}]:", d_opt)
            print(f"opt_value[{i}]:", opt_value)

            self.opt_solutions_solver.append(d_opt.squeeze())
            self.opt_objective_values_solver.append(opt_value)
        self.opt_solutions_solver = np.array(self.opt_solutions_solver)
        self.opt_objective_values_solver = np.array(self.opt_objective_values_solver)

    def _solve_optimization_problems_closed(self):
        n_data = self.r.shape[0]
        for i in range(n_data):
            model = optModelRd(
                num_items=self.num_items,
                num_data=self.num_data,
                alpha=self.alpha,
                Q=self.Q,
                r=self.r[i].reshape(-1),  # Ensure r has compatible shape
                c=self.c[i].reshape(-1)   # Ensure c has compatible shape
            )
            print(f"r[{i}]:", self.r[i])
            print(f"c[{i}]:", self.c[i])
            
            d_opt_closed, opt_value_closed = model.solveC()
            print(f"d_opt_closed[{i}]:", d_opt_closed)
            print(f"opt_value_closed[{i}]:", opt_value_closed)

            self.opt_solutions_closed.append(d_opt_closed.squeeze())
            self.opt_objective_values_closed.append(opt_value_closed)
        self.opt_solutions_closed = np.array(self.opt_solutions_closed)
        self.opt_objective_values_closed = np.array(self.opt_objective_values_closed)

    def __len__(self):
        return len(self.r)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.x[idx]),
            torch.FloatTensor(self.r[idx]),
            torch.FloatTensor(self.c[idx]),
            torch.FloatTensor(self.opt_solutions_solver[idx]),
            torch.FloatTensor([self.opt_objective_values_solver[idx]]),
            torch.FloatTensor(self.opt_solutions_closed[idx]),
            torch.FloatTensor([self.opt_objective_values_closed[idx]])
        )
        

