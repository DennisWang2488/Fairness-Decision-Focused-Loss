import cvxpy as cp
import numpy as np
import torch

def AlphaFairnesstorch(util: torch.Tensor, alpha):
    """
    util: 1D tensor of (benefit_i * decision_i)
    """
    if alpha == 1:
        return torch.sum(torch.log(util))
    elif alpha == 0:
        return torch.sum(util)
    elif alpha == 'inf':
        return torch.min(util)
    else:
        return torch.sum(util**(1-alpha) / (1-alpha))
    
def AlphaFairness(util, alpha):
    if isinstance(util, torch.Tensor):
        util = util.detach().cpu().numpy() if isinstance(util, torch.Tensor) else util
    if alpha == 1:
        return np.sum(np.log(util))
    elif alpha == 0:
        return np.sum(util)
    elif alpha == 'inf':
        return np.min(util)
    else:
        return np.sum(util**(1-alpha) / (1-alpha))

def objValue(d, b, alpha):
    """
    Calculate the objective value based on the data and utility predictions.
    """
    objval = None
    return objval

def solveIndProblem(benefit, cost, alpha, Q):

    d = cp.Variable(benefit.shape, nonneg=True)


    utils = cp.multiply(benefit, d)
    constraints = [d >= 0, cp.sum(cost * d) <= Q]

    if alpha == 'inf':
        t = cp.Variable()
        objective = cp.Maximize(t)
        constraints.append(utils >= t)
    elif alpha == 1:
        objective = cp.Maximize(cp.sum(cp.log(utils)))
    elif alpha == 0:
        objective = cp.Maximize(cp.sum(utils))
    else:
        objective = cp.Maximize(cp.sum(utils**(1-alpha)) / (1-alpha))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=False, warm_start=True, mosek_params={'MSK_IPAR_LOG': 1})

    if problem.status != 'optimal':
        print(f"Warning: Problem status is {problem.status}")

    optimal_decision = d.value
    optimal_value = objValue(benefit, optimal_decision, alpha)

    return optimal_decision, optimal_value
def solve_closed_form(g, r, c, alpha, Q):
    g = g.detach().cpu().numpy() if isinstance(g, torch.Tensor) else g
    r = r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r
    c = c.detach().cpu().numpy() if isinstance(c, torch.Tensor) else c

    if np.any(c <= 0) or np.any(r <= 0) or np.any(g <= 0):
        raise ValueError("Inputs must be strictly positive.")

    n = len(c)
    utility = np.maximum(r * g, 1e-6)

    if alpha == 0:
        ratios = utility / c
        sorted_indices = np.argsort(-ratios)
        d_star_closed = np.zeros(n)
        i = sorted_indices[0]
        d_star_closed[i] = Q / c[i]

    elif alpha == 1:
        weight = c / utility
        denom = np.sum(weight)
        d_star_closed = (Q / denom) * (1 / utility)

    elif alpha == 'inf':
        denom = np.sum(c * c / utility)
        d_star_closed = (Q * c) / (utility * denom)

    else:
        if alpha <= 0:
            raise ValueError("Alpha must be positive.")

        numerator = np.power(c, -1/alpha) * np.power(utility, 1/alpha - 1)
        d_unscaled = numerator
        cost_total = np.sum(c * d_unscaled)
        if cost_total == 0:
            raise ValueError("Degenerate solution: cost_total is zero")
        d_star_closed = (Q / cost_total) * d_unscaled

    obj = AlphaFairness(d_star_closed * utility, alpha)
    return d_star_closed, obj

def alpha_fairness_group_utilities(benefit, allocation, group, alpha):
    """
    Compute group-wise alpha-fairness utilities.
    """
    groups = np.unique(group)
    utils = []
    for k in groups:
        mask = (group == k)
        Gk = float(mask.sum())
        # Compute average utility in group k
        util_k = (benefit[mask] * allocation[mask]).sum(axis=0).mean()  # mean total utility per individual in group
        if alpha == 1:
            val = np.log(util_k) if util_k > 0 else -np.inf
        elif alpha == 0:
            val = util_k
        elif alpha == float('inf'):
            # Min utility as min total utility)
            val = (benefit[mask] * allocation[mask]).sum(axis=0).min()
        else:
            val = util_k**(1 - alpha) / (1 - alpha)
        utils.append(val)
    return np.array(utils).sum()


def solveGroupProblem(benefit,
                      cost,
                      group,
                      alpha,
                      Q):
    """
    Solve the group-based alpha-fair allocation problem:
       max_d  W_alpha(d)
       s.t.   sum(cost * d) <= Q,  d >= 0

    Parameters
    ----------
    benefit : np.ndarray, shape (n, T)
        Predicted utilities \\hat b_{i,t} > 0.
    cost : np.ndarray, shape (n, T)
        Costs c_{i,t} > 0.
    group : np.ndarray, shape (n,)
        Integer group labels in {0,...,K-1}.
    alpha : float, or 0, 1, or 'inf'
        Fairness parameter.
    Q : float
        Total available budget.
    """
    n, T = benefit.shape
    groups = np.unique(group)
    K = groups.size

    # decision variable
    d = cp.Variable((n, T), nonneg=True)

    # build per-group utilities
    utils = []
    for k in groups:
        mask = (group == k)
        Gk = float(mask.sum())
        # sum over i in Gk and all t
        util_k = cp.sum(cp.multiply(benefit[mask], d[mask])) / Gk
        utils.append(util_k)
    utils = cp.hstack(utils)

    # choose α-fair objective
    constraints = [cp.sum(cp.multiply(cost, d)) <= Q, d >= 0]
    if alpha == 'inf':
        t = cp.Variable()
        constraints.append(utils >= t)
        objective = cp.Maximize(t)
    elif alpha == 1:
        objective = cp.Maximize(cp.sum(cp.log(utils)))
    elif alpha == 0:
        objective = cp.Maximize(cp.sum(utils))
    else:
        # generic 0<α<∞, α≠1
        objective = cp.Maximize(cp.sum(utils**(1 - alpha)) / (1 - alpha))

    # solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, warm_start=True)

    if prob.status != 'optimal':
        print(f"[solveGroupProblem] Warning: status = {prob.status}")

    opt_val = alpha_fairness_group_utilities(benefit, d.value, group, alpha)
    if opt_val is None:
        print("[solveGroupProblem] Warning: Objective value is None, check the problem formulation.")
        return None, None
    
    return np.array(d.value), opt_val


def closed_form_group_alpha(b_hat, cost, group, Q, alpha):
    """
    b_hat : (N,)  or (N,1) or (N,T)    strictly positive
    cost  : same shape as b_hat (broadcast OK)
    group : (N,)  or (N,1)  integer labels 0 … K-1
    Q     : scalar > 0
    alpha : 0, 1, np.inf, or positive float
    """
    # ------------ 1. normalise shapes ---------------------------------
    b_hat = np.asarray(b_hat, dtype=float)
    cost  = np.asarray(cost,  dtype=float)

    if b_hat.ndim == 1:                        # promote to 2-D (N,1)
        b_hat = b_hat[:, None]
    if cost.ndim == 1:
        cost  = cost[:, None]
    if cost.shape[1] == 1 and b_hat.shape[1] > 1:
        cost = np.repeat(cost, b_hat.shape[1], axis=1)
    if b_hat.shape[1] == 1 and cost.shape[1] > 1:
        b_hat = np.repeat(b_hat, cost.shape[1], axis=1)
    assert b_hat.shape == cost.shape, "benefit & cost must broadcast"

    # ------------ 2. squeeze group to 1-D int array -------------------
    group = np.asarray(group).astype(int).reshape(-1)
    if group.ndim != 1:
        raise ValueError("`group` must be 1-D after reshape")
    N, T = b_hat.shape
    if group.size != N:
        raise ValueError("length of `group` must equal #rows of b_hat")

    K  = group.max() + 1
    G  = np.bincount(group, minlength=K)       # each G_k > 0 ?

    # ------------ 3. best ratio per group ----------------------------
    rho   = np.empty(K)
    idx_k = np.empty((K, 2), dtype=int)

    for k in range(K):
        rows = np.flatnonzero(group == k)
        ratio_sub = b_hat[rows] / cost[rows]   # shape (|rows|, T)
        flat_idx  = ratio_sub.argmax()         # 0 … |rows|·T−1
        r_loc, t_star = divmod(flat_idx, T)
        i_star = rows[r_loc] # type: ignore
        rho[k]  = ratio_sub.flat[flat_idx]
        idx_k[k] = (i_star, t_star)

    p = rho / G                                # p_k = ρ_k / G_k

    # ------------ 4. allocate budgets x_k ----------------------------
    if alpha == 0:                             # utilitarian
        winners = np.flatnonzero(p == p.max())
        x = np.zeros(K)
        x[winners] = Q / len(winners)
    elif alpha == 1:                           # log utility
        x = np.full(K, Q / K)
    elif alpha == np.inf:                      # max–min
        inv = 1 / p
        x = Q * inv / inv.sum()
    else:                                      # generic α
        beta   = 1.0 / alpha
        w      = p ** (beta - 1)
        x = Q * w / w.sum()

    # ------------ 5. build decision matrix ---------------------------
    d_star = np.zeros_like(b_hat)
    for k, (i, t) in enumerate(idx_k):
        d_star[i, t] = x[k] / cost[i, t]

    return d_star, idx_k, x, rho