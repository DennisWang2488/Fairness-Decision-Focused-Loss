import cvxpy as cp
import numpy as np
import torch

alpha = 1
Q = 10

def AlphaFairness(util,alpha):
    if alpha == 1:
        return np.sum(np.log(util))
    elif alpha == 0:
        return np.sum(util)
    elif alpha == 'inf':
        return np.min(util)
    else:
        return np.sum(util**(1-alpha)/(1-alpha))


def compute_d_star_closed_form(g, r, c, alpha=alpha, Q=Q):

    if not isinstance(c, np.ndarray) or not isinstance(r, np.ndarray) or not isinstance(g, np.ndarray):
        raise TypeError("c, r, and g must be numpy arrays.")
    if c.shape != r.shape or c.shape != g.shape:
        raise ValueError("c, r, and g must have the same shape.")
    if np.any(c <= 0):
        raise ValueError("All cost values must be positive.")
    if np.any(r <= 0):
        raise ValueError("All risk values must be positive.")
    if np.any(g <= 0):
        raise ValueError("All gain factors must be positive.")
    
    n = len(c)
    utility = r * g
    
    if alpha == 0:
        ratios = utility / c
        sorted_indices = np.argsort(-ratios)  # Descending order
        d_star_closed = np.zeros(n)
        d_star_closed[sorted_indices[0]] = Q / c[sorted_indices[0]]
        
    elif alpha == 1:
        d_star_closed = Q / (n * c)
    
    elif alpha == 'inf':
        d_star_closed = (Q * c) / (utility * np.sum(c * c / utility))
        
    else:
        if alpha <= 0:
            raise ValueError("Alpha must be positive for general case.")
        #
        # d_i* = (c_i^(-1/alpha) * (r_i*g_i)^(1/alpha - 1) * Q) / sum_j(c_j^(-1/alpha) * (r_j*g_j)^(1/alpha - 1))
        
        numerator = np.power(c, -1/alpha) * np.power(utility, 1/alpha - 1)
        denominator = np.sum(np.power(c, 1-1/alpha) * np.power(utility, 1/alpha - 1))
        
        if denominator == 0:
            raise ValueError("Denominator is zero in closed-form solution.")
            
        d_star_closed = (numerator / denominator) * Q
    
    # if not np.isclose(np.sum(c * d_star_closed), Q, rtol=1e-5):
    #     raise ValueError("Solution does not satisfy budget constraint.")
        
    return d_star_closed

def solve_optimization(gainF, risk, cost, alpha=alpha, Q=Q):
    # Flatten input arrays

    # if any of the inputs are tensor, convert to numpy array
    gainF = gainF.detach().cpu().numpy() if isinstance(gainF, torch.Tensor) else gainF
    risk = risk.detach().cpu().numpy() if isinstance(risk, torch.Tensor) else risk
    cost = cost.detach().cpu().numpy() if isinstance(cost, torch.Tensor) else cost

    risk = risk.clip(min=0.001)
    gainF, risk, cost = gainF.flatten(), risk.flatten(), cost.flatten()
    d = cp.Variable(risk.shape, nonneg=True)

    # raise error if dimensions do not match
    if gainF.shape != risk.shape or risk.shape != cost.shape:
        raise ValueError("Dimensions of gainF, risk, and cost do not match")
    
    utils = cp.multiply(cp.multiply(gainF, risk), d)
    
    if alpha == 'inf':
        # Maximin formulation
        t = cp.Variable()  # auxiliary variable for minimum utility
        objective = cp.Maximize(t)
        constraints = [
            d >= 0,
            # d <= 1,
            cp.sum(cost * d) <= Q,
            utils >= t  # t is the minimum utility
        ]
    elif alpha == 1:
        # Nash welfare (alpha = 1)
        objective = cp.Maximize(cp.sum(cp.log(utils)))
        constraints = [
            d >= 0,
            # d <= 1,
            cp.sum(cost * d) <= Q
        ]
    elif alpha == 0:
        # Utilitarian welfare (alpha = 0)
        objective = cp.Maximize(cp.sum(utils))
        constraints = [
            d >= 0,
            # d <= 1,
            cp.sum(cost * d) <= Q
        ]
    else:
        # General alpha-fairness
        objective = cp.Maximize(cp.sum(utils**(1-alpha))/(1-alpha) if alpha != 0 
                              else cp.sum(utils))
        constraints = [
            d >= 0,
            # d <= 1,
            cp.sum(cost * d) <= Q
        ]
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=False, warm_start=True, mosek_params={'MSK_IPAR_LOG': 1})
    
    if problem.status != 'optimal':
        print(f"Warning: Problem status is {problem.status}")
    
    optimal_decision = d.value
    optimal_value = AlphaFairness(optimal_decision * gainF * risk, alpha)
    
    return optimal_decision, optimal_value

# Problem size
n = 5  # Example dimension

# Define variables
x = cp.Variable(n,nonneg=True)

# Define parameters
p = cp.Parameter(n, nonneg=True)  # p_i = r_i * g_i
a = 1.5  # Example: a > 1 to ensure convexity
c = np.array([1, 2, 3, 4, 5])  # Constraint coefficients
Q = 10  # Constraint bound
g = np.array([1, 2, 3, 4, 5])
r = np.array([1, 2, 3, 4, 5])

# Define objective function in parameterized form
objective = cp.Minimize(- (1 / (1 - a)) * cp.sum(cp.power(cp.multiply(p, x),1 - a)))

# Define constraints
constraints = [cp.sum(cp.multiply(c,x)) <= Q, x >= 0]

# Define the problem
problem = cp.Problem(objective, constraints)

# Assign values to parameters and solve
p.value = g*r  # Assign example values to parameter
problem.solve(verbose=False)

x1 = x.value
x2 = compute_d_star_closed_form(g, r, c, alpha=a, Q=Q)

print("Optimal x from cvxpy:", x1)
print("alpha fairness value from cvxpy: ", AlphaFairness(x1*r*g, a))

print("Optimal x from closed form solution:", x2)
print("alpha fairness value from closed form: ", AlphaFairness(x2*r*g, a))