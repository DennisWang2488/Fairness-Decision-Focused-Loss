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

import numpy as np

def solve_coupled_group_alpha(b, c, group_idx, Q, alpha, beta):
    """
    Calculates the optimal allocation d using the closed-form solution.
    
    This version handles the standard formulation for beta < 1 and the modified,
    positive-definite formulation for beta > 1.
    """
    # Ensure inputs are NumPy arrays
    b = np.asarray(b).reshape(-1)
    c = np.asarray(c).reshape(-1)
    group_idx = np.asarray(group_idx).reshape(-1)

    if abs(beta - 1.0) < 1e-9:
        raise ValueError("The closed-form solution is not defined for beta = 1.")

    n = len(b)
    d_star = np.zeros(n)
    
    # --- Define parameters based on the beta regime ---
    if beta > 1:
        # New formulation for beta > 1
        gamma = beta - 2 + alpha - alpha * beta
        const_factor = beta - 1
        s_exp = (2 - alpha) / gamma
        const_exp = (alpha - 2) / gamma
    else: # beta < 1
        # Original formulation
        gamma = beta + alpha - alpha * beta
        const_factor = 1 - beta
        s_exp = -alpha / gamma
        const_exp = alpha / gamma
        
    if abs(gamma) < 1e-9:
        raise ValueError("Gamma exponent is zero, leading to instability.")

    unique_groups = np.unique(group_idx)
    S, H, Psi = {}, {}, {}
    
    for k in unique_groups:
        members_mask = (group_idx == k)
        G_k = np.sum(members_mask)
        b_k, c_k = b[members_mask], c[members_mask]
        
        # S_k and H_k formulas are identical in both regimes
        S[k] = np.sum((c_k ** (-(1 - beta) / beta)) * (b_k ** ((1 - beta) / beta)))
        H[k] = np.sum((c_k ** ((beta - 1) / beta)) * (b_k ** ((1 - beta) / beta)))
        
        if abs(S[k]) < 1e-12:
             raise ValueError(f"S_k for group {k} is near zero. Cannot proceed.")

        # Psi_k formula changes based on the regime's exponents
        if beta > 1:
             Psi[k] = (S[k] ** s_exp) * (const_factor ** const_exp)
        else: # beta < 1
             Psi[k] = (G_k ** ((alpha - 1) / gamma)) * (S[k] ** s_exp) * (const_factor ** const_exp)

    # Global normalization constant Xi
    Xi = np.sum([H[k] * Psi[k] for k in unique_groups])
    if abs(Xi) < 1e-12:
        raise ValueError("Normalization constant Xi is near zero.")

    # Assemble the final solution
    for k in unique_groups:
        members_mask = (group_idx == k)
        # phi_i formula is identical in both regimes
        phi = (c[members_mask] ** (-1 / beta)) * (b[members_mask] ** ((1 - beta) / beta))
        d_star[members_mask] = (Q / Xi) * Psi[k] * phi

    return d_star

def solve_coupled_group_grad(b, c, group_idx, Q, alpha, beta):
    """
    Computes the Jacobian matrix d(d*)/d(b) using the analytical formula,
    handling both regimes for beta.
    """
    b = np.asarray(b).reshape(-1)
    c = np.asarray(c).reshape(-1)
    group_idx = np.asarray(group_idx).reshape(-1)

    n = len(b)
    jacobian = np.zeros((n, n))

    # --- 1. Forward Pass: Pre-compute terms using the same logic as the solver ---
    # Define parameters based on the beta regime
    if beta > 1:
        gamma = beta - 2 + alpha - alpha * beta
        psi_s_exp_factor = (2 - alpha) / gamma
    else: # beta < 1
        gamma = beta + alpha - alpha * beta
        psi_s_exp_factor = -alpha / gamma

    # Get the allocation vector d* and all intermediate terms (S, H, Psi, Xi)
    # by calling the solver function itself.
    d_star = solve_coupled_group_alpha(b, c, group_idx, Q, alpha, beta)
    
    unique_groups = np.unique(group_idx)
    S, H, Psi = {}, {}, {}
    # Re-calculate intermediate terms to have them available
    for k in unique_groups:
        members_mask = (group_idx == k)
        G_k, b_k, c_k = np.sum(members_mask), b[members_mask], c[members_mask]
        S[k] = np.sum((c_k ** (-(1 - beta) / beta)) * (b_k ** ((1 - beta) / beta)))
        H[k] = np.sum((c_k ** ((beta - 1) / beta)) * (b_k ** ((1 - beta) / beta)))
        
        const_factor = (beta - 1) if beta > 1 else (1 - beta)
        if beta > 1:
            Psi[k] = (S[k] ** psi_s_exp_factor) * (const_factor ** ((alpha - 2) / gamma))
        else:
            Psi[k] = (G_k ** ((alpha - 1) / gamma)) * (S[k] ** psi_s_exp_factor) * (const_factor ** (alpha/gamma))
            
    Xi = np.sum([H[k] * Psi[k] for k in unique_groups])
    phi_all = (c ** (-1 / beta)) * (b ** ((1 - beta) / beta))

    # --- 2. Backward Pass: Calculate Jacobian column by column ---
    for j in range(n):  # Differentiating with respect to b_j
        m = group_idx[j]  # Group of the variable b_j
        
        # Derivatives of S and H are the same in both regimes
        dS_m_db_j = ((1 - beta) / beta) * (c[j] ** (-(1 - beta) / beta)) * (b[j] ** ((1 - 2 * beta) / beta))
        dH_m_db_j = ((1 - beta) / beta) * (c[j] ** ((beta - 1) / beta)) * (b[j] ** ((1 - 2 * beta) / beta))
        
        # d(Psi)/d(S) * d(S)/d(b_j)
        dPsi_m_db_j = (psi_s_exp_factor / S[m]) * Psi[m] * dS_m_db_j
        dXi_db_j = dH_m_db_j * Psi[m] + H[m] * dPsi_m_db_j
        
        for i in range(n):  # Calculating derivative for d_i
            k = group_idx[i]  # Group of the component d_i
            
            dphi_i_db_j = 0
            if i == j:
                dphi_i_db_j = ((1 - beta) / beta) * (c[i] ** (-1 / beta)) * (b[i] ** ((1 - 2 * beta) / beta))

            dPsi_k_db_j = dPsi_m_db_j if k == m else 0
            
            dN_i_db_j = Q * (dPsi_k_db_j * phi_all[i] + Psi[k] * dphi_i_db_j)
            
            jacobian[i, j] = (1 / Xi) * dN_i_db_j - (d_star[i] / Xi) * dXi_db_j

    return jacobian

def compute_coupled_group_obj(d, b, group_idx, alpha, beta):
    """
    Calculates the objective value, handling the modified formulation for beta > 1.
    """
    d = np.asarray(d).reshape(-1)
    b = np.asarray(b).reshape(-1)
    group_idx = np.asarray(group_idx).reshape(-1)
    
    # Add a small epsilon for numerical stability
    y = b * d + 1e-12
    unique_groups = np.unique(group_idx)
    mu_k_values = np.zeros(len(unique_groups))

    # --- Step 1: Calculate group utilities (mu_k) based on beta regime ---
    if beta > 1:
        # New formulation for beta > 1
        for i, k in enumerate(unique_groups):
            members_mask = (group_idx == k)
            y_k = y[members_mask]
            # mu_k = (beta - 1) / sum(y_i^(1-beta))
            denominator = np.sum(y_k**(1 - beta))
            mu_k_values[i] = (beta - 1) / (denominator + 1e-12)
    else:
        # Original formulation for beta < 1
        if abs(beta - 1.0) < 1e-9:
            g_beta_values = np.log(y)
        else:
            g_beta_values = (y**(1 - beta)) / (1 - beta)
        
        for i, k in enumerate(unique_groups):
            members_mask = (group_idx == k)
            mu_k_values[i] = np.mean(g_beta_values[members_mask])

    # --- Step 2: Apply the outer fairness function (f_alpha) ---
    # This part is now safe because mu_k is always positive.
    if alpha == float('inf') or str(alpha).lower() == 'inf':
        objective_value = np.min(mu_k_values)
    elif abs(alpha - 1.0) < 1e-9:
        objective_value = np.sum(np.log(mu_k_values + 1e-12))
    elif abs(alpha - 0.0) < 1e-9:
        objective_value = np.sum(mu_k_values)
    else:
        f_alpha_values = (mu_k_values**(1 - alpha)) / (1 - alpha)
        objective_value = np.sum(f_alpha_values)
        
    return objective_value



def compute_coupled_group_obj_torch(d, b, group_idx, alpha, beta):
    """
    Calculates the objective value using PyTorch, handling the modified formulation for beta > 1.
    """
    d = torch.as_tensor(d, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    group_idx = torch.as_tensor(group_idx)
    
    if not d.requires_grad:
        d.requires_grad_(True)
    
    epsilon = 1e-12
    y = b * d + epsilon
    unique_groups = torch.unique(group_idx)
    mu_k_values = torch.zeros(len(unique_groups), dtype=torch.float64, device=d.device)

    if beta > 1:
        for i, k in enumerate(unique_groups):
            members_mask = (group_idx == k)
            y_k = y[members_mask]
            denominator = torch.sum(y_k.pow(1 - beta))
            mu_k_values[i] = (beta - 1) / (denominator + epsilon)
    else:
        if abs(beta - 1.0) < epsilon:
            g_beta_values = torch.log(y)
        else:
            g_beta_values = (y.pow(1 - beta)) / (1 - beta)
        
        for i, k in enumerate(unique_groups):
            members_mask = (group_idx == k)
            mu_k_values[i] = torch.mean(g_beta_values[members_mask])

    if alpha == 'inf' or (isinstance(alpha, str) and alpha.lower() == 'inf'):
        objective_value = torch.min(mu_k_values)
    elif abs(alpha - 1.0) < epsilon:
        objective_value = torch.sum(torch.log(mu_k_values + epsilon))
    elif abs(alpha - 0.0) < epsilon:
        objective_value = torch.sum(mu_k_values)
    else:
        f_alpha_values = (mu_k_values.pow(1 - alpha)) / (1 - alpha)
        objective_value = torch.sum(f_alpha_values)
        
    return objective_value, d



def compute_gradient_closed_form(g, r, c, alpha, Q):
    """
    Compute the analytical gradient of the optimal solution with respect to r.

    This function computes the gradient matrix where each element (i, k) is the partial derivative
    of d_i* with respect to r_k.

    Parameters:
    - g (np.ndarray): Gain factors (g_i), shape (n,)
    - r (np.ndarray): Risk values (r_i), shape (n,)
    - c (np.ndarray): Cost values (c_i), shape (n,)
    - alpha (float or str): Fairness parameter. Can be 0, 1, 'inf', or a positive real number.
    - Q (float): Total budget.

    Returns:
    - gradient (np.ndarray): Gradient matrix of shape (n, n)
    """
    if alpha == 1:
        S = np.sum(c / (r * g))

    if alpha == 0:
        # Utilitarian case: Allocate everything to the individual with the highest ratio
        ratios = (r * g) / c
        i_star = np.argmax(ratios)
        # Gradient is Q * g_i / c_i at the allocated index, zero elsewhere
        gradient[i_star, i_star] = Q * g[i_star] / c[i_star]
        return gradient

    elif alpha == 'inf':
        # Maximin case
        n = len(c)
        utility = r * g  # Shape: (n,)
        S = np.sum(c**2 / utility)  # Scalar

        # Compute d_star
        d_star, _ = solve_closed_form(g,r,c, alpha='inf', Q=Q)  # Shape: (n,)

        # Initialize gradient matrix
        gradient = np.zeros((n, n))

        for i in range(n):
            for k in range(n):
                if i == k:
                    # ∂d_i*/∂r_i = -d_i*/r_i - (d_i* * c_i) / (r_i * g_i * S)
                    gradient[i, k] = -d_star[i] / r[i] - (d_star[i] * c[i]) / (r[i] * g[i] * S)
                else:
                    # ∂d_i*/∂r_k = (d_i* * c_k^2) / (c_i * r_k^2 * g_k * S)
                    gradient[i, k] = (d_star[i] * c[k]**2) / (c[i] * r[k]**2 * g[k] * S)
        return gradient

    else:
        # General alpha case
        if not isinstance(alpha, (int, float)):
            raise TypeError("Alpha must be a positive real number, 0, 1, or 'inf'.")
        if alpha <= 0:
            raise ValueError("Alpha must be positive for gradient computation.")

        # Compute the optimal decision variables
        d_star, _ = solve_closed_form(g, r, c, alpha, Q)  # Shape: (n,)

        # Compute the term (1/alpha - 1) * g / r
        term = (1.0 / alpha - 1.0) * g / r  # Shape: (n,)

        # Compute the outer product for off-diagonal elements
        # Each element (i, k) = -d_star[i] * d_star[k] * term[k] / Q
        gradient = -np.outer(d_star, d_star * term) / Q  # Shape: (n, n)

        # Compute the diagonal elements
        # Each diagonal element (i, i) = d_star[i] * term[i] * (1 - d_star[i]/Q)
        diag_elements = d_star * term * (1 - d_star / Q)  # Shape: (n,)

        # Set the diagonal elements
        np.fill_diagonal(gradient, diag_elements)

        return gradient
    

# ==============================================================================
# ===== 1. INDIVIDUAL-BASED ALPHA-FAIRNESS GRADIENT
# ==============================================================================

def compute_individual_gradient_analytical(d: torch.Tensor, b: torch.Tensor, alpha):
    """
    Computes the analytical gradient of individual-based alpha-fairness w.r.t. decisions 'd'.
    
    This function implements the derived mathematical formulas directly without using autograd.

    Args:
        d (torch.Tensor): The decision variables for each individual.
        b (torch.Tensor): The benefit for each individual.
        alpha (float or str): The fairness parameter.

    Returns:
        torch.Tensor: The gradient of the objective with respect to 'd'.
    """
    # Ensure inputs are tensors
    d = torch.as_tensor(d, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    u = b * d

    if alpha == 1:
        # Gradient formula: 1 / d_j
        grad = 1.0 / d
    elif alpha == 0:
        # Gradient formula: b_j
        grad = b
    elif alpha == 'inf' or (isinstance(alpha, str) and alpha.lower() == 'inf'):
        # Gradient formula: b_j if j is the argmin, 0 otherwise
        grad = torch.zeros_like(d)
        min_idx = torch.argmin(u)
        grad[min_idx] = b[min_idx]
    else:
        # General gradient formula: b_j^(1-alpha) * d_j^(-alpha)
        # We can also write this as u_j^(-alpha) * b_j
        grad = (u.pow(-alpha)) * b
        
    return grad

# ==============================================================================
# ===== 2. GROUP-BASED ALPHA-FAIRNESS GRADIENT
# ==============================================================================

def compute_group_gradient_analytical(d: torch.Tensor, b: torch.Tensor, group_idx: torch.Tensor, alpha, beta):
    """
    Computes the analytical gradient of group-based alpha-fairness w.r.t. decisions 'd'.

    This function implements the derived chain rule formula directly.
    Gradient_j = (dJ/d(mu_Gj)) * (d(mu_Gj)/du_j) * (du_j/d_j)

    Args:
        d (torch.Tensor): The decision variables for each individual.
        b (torch.Tensor): The benefit for each individual.
        group_idx (torch.Tensor): The group index for each individual.
        alpha (float or str): The outer fairness parameter.
        beta (float): The inner group utility fairness parameter.

    Returns:
        torch.Tensor: The gradient of the objective with respect to 'd'.
    """
    # Ensure inputs are tensors
    d = torch.as_tensor(d, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    group_idx = torch.as_tensor(group_idx)
    u = b * d
    epsilon = 1e-12

    unique_groups, group_inv_indices, group_counts = torch.unique(group_idx, return_inverse=True, return_counts=True)
    num_groups = len(unique_groups)
    mu_k_values = torch.zeros(num_groups, dtype=torch.float64)
    
    # --- Pre-computation: Calculate mu_k for each group ---
    if beta > 1:
        for i, k in enumerate(unique_groups):
            members_mask = (group_idx == k)
            y_k = u[members_mask]
            denominator = torch.sum(y_k.pow(1 - beta))
            mu_k_values[i] = (beta - 1) / (denominator + epsilon)
    else:
        if abs(beta - 1.0) < epsilon:
            g_beta_values = torch.log(u + epsilon)
        else:
            g_beta_values = (u.pow(1 - beta)) / (1 - beta)
        # Use scatter_add to sum g_beta_values for each group, then divide by count
        sum_g_beta_k = torch.zeros(num_groups, dtype=torch.float64).scatter_add_(0, group_inv_indices, g_beta_values)
        mu_k_values = sum_g_beta_k / group_counts

    # --- Term 1: dJ/d(mu_k) ---
    dJ_dmuk = torch.zeros_like(mu_k_values)
    if alpha == 1:
        dJ_dmuk = 1.0 / (mu_k_values + epsilon)
    elif alpha == 0:
        dJ_dmuk = torch.ones_like(mu_k_values)
    elif alpha == 'inf' or (isinstance(alpha, str) and alpha.lower() == 'inf'):
        min_group_idx = torch.argmin(mu_k_values)
        dJ_dmuk[min_group_idx] = 1.0
    else:
        dJ_dmuk = mu_k_values.pow(-alpha)

    # Map the group-level derivative back to each individual
    dJ_dmuk_mapped = dJ_dmuk[group_inv_indices]

    # --- Term 2: d(mu_k)/d(u_j) ---
    dmuk_duj = torch.zeros_like(d)
    if beta > 1:
        # Formula: mu_k^2 * u_j^(-beta)
        mu_k_mapped = mu_k_values[group_inv_indices]
        dmuk_duj = mu_k_mapped.pow(2) * u.pow(-beta)
    else:
        # Formula: (1/N_k) * u_j^(-beta)
        n_k_mapped = group_counts[group_inv_indices]
        dmuk_duj = (1.0 / n_k_mapped) * u.pow(-beta)
        
    # --- Term 3: du_j/d_j ---
    dud_dj = b

    # --- Combine using Chain Rule ---
    gradient = dJ_dmuk_mapped * dmuk_duj * dud_dj
    return gradient

