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
    optimal_value = AlphaFairness(benefit*optimal_decision, alpha)

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


# ---------------------------------------------------------------------
# gradient of objective  W(b)  w.r.t. each b_{it}
# ---------------------------------------------------------------------
def grad_W_wrt_b(b, c, g, Q, alpha):
    """
    Returns  ∇_b W  with the same shape as b
    """
    d_star, idx_k, p, _ = closed_form_group_alpha(b, c, g, Q, alpha)
    K = len(p)
    G = np.bincount(g, minlength=K)
    grad = np.zeros_like(b)

    if alpha in (0, np.inf):
        # objective is non-smooth here; return NaNs
        grad[:] = np.nan
        return grad

    if alpha == 1:                                  # W = Σ log u_k
        for k, (i, t) in enumerate(idx_k):
            grad[i, t] = 1 / (p[k] * G[k] * c[i, t])
        return grad

    # ---------- generic  0<α<∞, α≠1  -------------------------------
    beta = 1.0 / alpha
    D = np.sum(p ** (beta - 1))                     # denominator
    u = Q * p ** beta / D                          # group utilities

    # ∂W/∂u_k  and helper coeffs
    dW_du = u ** (-alpha)                          # u_k^{−α}
    coeff = Q ** (-alpha) * D ** (alpha - 2)       # common front factor

    for k, (i, t) in enumerate(idx_k):
        pk = p[k]
        term = (beta * D - (beta - 1) * pk ** (beta - 1))
        dW_dpk = coeff * pk ** (beta - 2) * term
        grad[i, t] = dW_dpk / (G[k] * c[i, t])     # dp/db = 1/(G_k c)
    return grad


# ---------------------------------------------------------------------
# full Jacobian  ∂d*/∂b   (sparse dictionary representation)
# ---------------------------------------------------------------------
def jacobian_d_wrt_b(b, c, g, Q, alpha):
    """
    Returns
    -------
    J : dict mapping (i*,t*)  ->  gradient row (N,T) as numpy array
        Only K rows are non-zero: one per group winner (i*,t*).
    """
    d_star, idx_k, p, x = closed_form_group_alpha(b, c, g, Q, alpha)
    K = len(p)
    G = np.bincount(g, minlength=K)
    N, T = b.shape
    J = {}

    # special / non-smooth cases --------------------------------------
    if alpha in (0, 1, np.inf):
        # Jacobian exists but is piecewise-constant & sparse:
        #  ∂d*(winner k)/∂b(winner k) via budget split; others zero.
        # Users typically rely on sub-gradients → return NaNs.
        for k, (i, t) in enumerate(idx_k):
            J[(i, t)] = np.full_like(b, np.nan)
        return J

    # ---------- generic  0<α<∞, α≠1 ----------------------------------
    beta = 1.0 / alpha
    D = np.sum(p ** (beta - 1))
    dD_dpk = (beta - 1) * p ** (beta - 2)          # derivative of D
    # pre-compute    ∂x_l / ∂p_k   for every pair (l,k)
    x_grad = np.zeros((K, K))
    for l in range(K):
        for k in range(K):
            if k == l:
                num = (beta - 1) * p[k] ** (beta - 2) * D
                num -= p[k] ** (beta - 1) * dD_dpk[k]
                x_grad[l, k] = Q * num / D ** 2
            else:
                x_grad[l, k] = -Q * p[l] ** (beta - 1) * dD_dpk[k] / D ** 2

    # build Jacobian rows (only one non-zero col per group)
    for k, (i_win, t_win) in enumerate(idx_k):
        row = np.zeros_like(b)
        # effect of p_k on every x_l  (thus on every winner l)
        for l, (i_l, t_l) in enumerate(idx_k):
            row[i_l, t_l] += (
                x_grad[l, k] / c[i_l, t_l] / (G[k] * c[i_win, t_win])
            )
        J[(i_win, t_win)] = row
    return J

import collections

def solve_coupled_group_alpha(b, c, group_idx, Q, alpha, beta):
    """
    Calculates the optimal allocation d using the closed-form solution.
    
    Args:
        b (np.ndarray): Vector of benefit coefficients.
        c (np.ndarray): Vector of cost coefficients.
        group_idx (np.ndarray): Array of group assignments for each individual.
        Q (float): Total budget.
        alpha (float): Outer fairness parameter.
        beta (float): Inner fairness parameter.

    Returns:
        np.ndarray: The optimal allocation vector d*.
    """
    if beta == 1:
        raise ValueError("The closed-form solution is not defined for beta = 1.")
    
    n = len(b)
    d_star = np.zeros(n)
    gamma = beta + alpha - alpha * beta
    if gamma == 0:
        raise ValueError("gamma (alpha + beta - alpha*beta) cannot be zero.")

    unique_groups = np.unique(group_idx)
    
    # Store group-level aggregates in dictionaries keyed by group label
    S, H, Psi = {}, {}, {}
    
    for k in unique_groups:
        members_mask = (group_idx == k)
        G_k = np.sum(members_mask)
        b_k, c_k = b[members_mask], c[members_mask]
        
        S[k] = np.sum((c_k ** (-(1 - beta) / beta)) * (b_k ** ((1 - beta) / beta)))
        H[k] = np.sum((c_k ** ((beta - 1) / beta)) * (b_k ** ((1 - beta) / beta)))
        
        if S[k] == 0:
             raise ValueError(f"S_k for group {k} is zero. Cannot proceed.")

        Psi[k] = (
            (G_k ** ((alpha - 1) / gamma)) *
            (S[k] ** (-alpha / gamma)) *
            ((1 - beta) ** (alpha / gamma))
        )

    # Calculate global normalization constant Xi
    Xi = np.sum([H[k] * Psi[k] for k in unique_groups])
    if Xi == 0:
        raise ValueError("Normalization constant Xi is zero. Cannot divide by zero.")

    # Assemble the final solution for each d_i using its group's prefactor
    for k in unique_groups:
        members_mask = (group_idx == k)
        phi = (c[members_mask] ** (-1 / beta)) * (b[members_mask] ** ((1 - beta) / beta))
        d_star[members_mask] = (Q / Xi) * Psi[k] * phi

    return d_star

def solve_coupled_group_grad(b, c, group_idx, Q, alpha, beta):
    """
    Computes the Jacobian matrix d(d*)/d(b) using the analytical formula.
    
    Args:
        b (np.ndarray): Vector of benefit coefficients.
        c (np.ndarray): Vector of cost coefficients.
        group_idx (np.ndarray): Array of group assignments for each individual.
        ... and other parameters
        
    Returns:
        np.ndarray: The n x n Jacobian matrix.
    """

    
    n = len(b)
    jacobian = np.zeros((n, n))

    # --- 1. Forward Pass: Pre-compute all terms from the d* calculation ---
    gamma = beta + alpha - alpha * beta
    unique_groups = np.unique(group_idx)
    S, H, Psi, phi = {}, {}, {}, {}

    for k in unique_groups:
        members_mask = (group_idx == k)
        G_k = np.sum(members_mask)
        b_k, c_k = b[members_mask], c[members_mask]
        S[k] = np.sum((c_k ** (-(1 - beta) / beta)) * (b_k ** ((1 - beta) / beta)))
        H[k] = np.sum((c_k ** ((beta - 1) / beta)) * (b_k ** ((1 - beta) / beta)))
        Psi[k] = ((G_k ** ((alpha - 1) / gamma)) * (S[k] ** (-alpha / gamma)) * ((1 - beta) ** (alpha / gamma)))
    
    Xi = np.sum([H[k] * Psi[k] for k in unique_groups])
    d_star = solve_coupled_group_alpha(b, c, group_idx, Q, alpha, beta)
    
    # Pre-compute individual phi_i terms
    phi_all = (c ** (-1 / beta)) * (b ** ((1 - beta) / beta))

    # --- 2. Backward Pass: Calculate Jacobian column by column ---
    for j in range(n):  # Differentiating with respect to b_j
        m = group_idx[j]  # Group of the variable b_j
        
        # --- Derivatives of intermediate terms w.r.t. b_j ---
        # These are non-zero only for group m
        
        dS_m_db_j = ((1 - beta) / beta) * (c[j] ** (-(1 - beta) / beta)) * (b[j] ** ((1 - 2 * beta) / beta))
        dH_m_db_j = ((1 - beta) / beta) * (c[j] ** ((beta - 1) / beta)) * (b[j] ** ((1 - 2 * beta) / beta))
        dPsi_m_db_j = (-alpha / (gamma * S[m])) * Psi[m] * dS_m_db_j
        dXi_db_j = dH_m_db_j * Psi[m] + H[m] * dPsi_m_db_j
        
        for i in range(n):  # Calculating derivative for d_i
            k = group_idx[i]  # Group of the component d_i
            
            # d(phi_i)/d(b_j) is non-zero only if i == j
            dphi_i_db_j = 0
            if i == j:
                dphi_i_db_j = ((1 - beta) / beta) * (c[i] ** (-1/beta)) * (b[i] ** ((1 - 2 * beta) / beta))

            # d(Psi_k)/d(b_j) is non-zero only if k == m
            dPsi_k_db_j = dPsi_m_db_j if k == m else 0
            
            # Derivative of the numerator term N_i = Q * Psi_k * phi_i
            dN_i_db_j = Q * (dPsi_k_db_j * phi_all[i] + Psi[k] * dphi_i_db_j)
            
            # Final assembly using the quotient rule derivative: (u/v)' = u'/v - u*v'/v^2
            jacobian[i, j] = (1 / Xi) * dN_i_db_j - (d_star[i] / Xi) * dXi_db_j


    return jacobian

def compute_coupled_group_obj(d, b, group_idx, alpha, beta):
    """
    Calculates the objective value for the coupled alpha-fairness problem.

    This function correctly handles the special cases for alpha and beta.

    Args:
        d (np.ndarray): The allocation vector d.
        b (np.ndarray): The vector of benefit coefficients.
        group_idx (np.ndarray): An array of group assignments for each individual.
        alpha (float or str): The outer fairness parameter. Can be 'inf'.
        beta (float): The inner fairness parameter.

    Returns:
        float: The final scalar objective value.
    """
    # --- Step 1: Calculate all group utilities (mu_k) ---
    
    # Calculate the argument of the inner utility function, y_i = b_i * d_i
    # Add a small epsilon for numerical stability with log operations
    y = b * d + 1e-12

    # Calculate individual utilities g_beta(y_i)
    if abs(beta - 1.0) < 1e-9:
        # Case beta = 1 (logarithmic utility)
        g_beta_values = np.log(y)
    else:
        # General case for beta
        g_beta_values = (y**(1 - beta)) / (1 - beta)

    # Aggregate to find the mean utility for each group
    unique_groups = np.unique(group_idx)
    mu_k_values = np.zeros(len(unique_groups))
    for i, k in enumerate(unique_groups):
        members_mask = (group_idx == k)
        mu_k_values[i] = np.mean(g_beta_values[members_mask])

    # --- Step 2: Apply the outer fairness function (f_alpha) and aggregate ---

    if alpha == float('inf') or str(alpha).lower() == 'inf':
        # Case alpha = inf (Max-Min Fairness)
        # The value of the Rawlsian objective is the utility of the worst-off group.
        objective_value = np.min(mu_k_values)

    elif abs(alpha - 1.0) < 1e-9:
        # Case alpha = 1 (Proportional Fairness / Logarithmic Utility)
        # Objective is sum(log(mu_k))
        objective_value = np.sum(np.log(mu_k_values + 1e-12))
    
    elif abs(alpha - 0.0) < 1e-9:
        # Case alpha = 0 (Utilitarian)
        # Objective is sum(mu_k)
        objective_value = np.sum(mu_k_values)
        
    else:
        # General case for alpha
        # Objective is sum(mu_k^(1-alpha) / (1-alpha))
        f_alpha_values = (mu_k_values**(1 - alpha)) / (1 - alpha)
        objective_value = np.sum(f_alpha_values)
        
    return objective_value