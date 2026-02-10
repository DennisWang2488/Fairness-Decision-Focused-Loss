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

# solve individual alpha-fairness problem using CVXPY
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
        
        # This vector is common to the numerator of d_i and the terms in the sum
        # It corresponds to c_i^(-1/alpha) * utility_i^(1/alpha - 1)
        common_terms = np.power(c, -1/alpha) * np.power(utility, 1/alpha - 1)
        
        # The correct denominator is Σ_j(c_j * common_term_j)
        denominator = np.sum(c * common_terms)
        
        if denominator == 0:
            raise ValueError("Denominator is zero in closed-form solution.")
            
        # The numerator of d_i is Q * common_term_i
        d_star_closed = (Q * common_terms) / denominator
    
    # if not np.isclose(np.sum(c * d_star_closed), Q, rtol=1e-5):
    #     raise ValueError("Solution does not satisfy budget constraint.")
    obj = AlphaFairness(d_star_closed * utility, alpha)
        
    return d_star_closed, obj

def solve_d_and_gradient_analytical(g, r, c, alpha, Q):
    """
    Computes the optimal decision d* and its analytical gradient w.r.t. r.

    Args:
        g (np.ndarray): Gain factors, shape (n,).
        r (np.ndarray): Predicted risk values, shape (n,).
        c (np.ndarray): Cost values, shape (n,).
        alpha (float or str): Fairness parameter.
        Q (float): Total budget.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - d_star (np.ndarray): The optimal decision vector, shape (n,).
            - grad_d_r (np.ndarray): The Jacobian matrix (gradient) of d* w.r.t. r,
                                     shape (n, n), where grad[i, k] = ∂d_i*/∂r_k.
    """
    n = len(r)
    g, r, c = map(np.atleast_1d, [g, r, c])
    utility = g * r
    grad_d_r = np.zeros((n, n))

    # --- 1. Solve for d* ---
    if alpha == 0:
        # Utilitarian: non-differentiable, handled below.
        i_star = np.argmax(utility / c)
        d_star = np.zeros(n)
        d_star[i_star] = Q / c[i_star]
    elif alpha == 1:
        d_star = Q / (n * c)
    elif alpha == 'inf':
        sum_cost_per_utility = np.sum(c / utility)
        d_star = (1 / utility) * (Q / sum_cost_per_utility)
    else: # General alpha
        common_terms = np.power(c, -1/alpha) * np.power(utility, (1-alpha)/alpha)
        denominator = np.sum(c * common_terms)
        d_star = (Q * common_terms) / denominator

    # --- 2. Compute Gradient ∂d*/∂r ---
    if alpha == 0:
        # The argmax operation is non-differentiable.
        # The gradient is 0 almost everywhere. Finite difference is the practical approach.
        pass # grad_d_r is already zeros
    elif alpha == 1:
        # d* does not depend on r, so the gradient is zero.
        pass # grad_d_r is already zeros
    elif alpha == 'inf':
        for i in range(n):
            for k in range(n):
                if i == k: # Diagonal: ∂d_i*/∂r_i
                    grad_d_r[i, i] = -d_star[i] / r[i] * (1 - c[i] * d_star[i] / Q)
                else: # Off-diagonal: ∂d_i*/∂r_k
                    grad_d_r[i, k] = d_star[i] * d_star[k] * c[k] / (r[k] * Q)
    else: # General alpha
        # Precompute the common term from the derivative
        term = (1 - alpha) / (alpha * r) # Shape (n,)
        for i in range(n):
            for k in range(n):
                if i == k: # Diagonal: ∂d_i*/∂r_i
                    grad_d_r[i, i] = d_star[i] * term[i] * (1 - c[i] * d_star[i] / Q)
                else: # Off-diagonal: ∂d_i*/∂r_k
                    grad_d_r[i, k] = -d_star[i] * d_star[k] * c[k] * term[k] / Q

    return d_star, grad_d_r

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
        gradient = -np.outer(d_star, d_star * term * c) / Q  # Shape: (n, n)

        # Compute the diagonal elements
        # Each diagonal element (i, i) = d_star[i] * term[i] * (1 - d_star[i]/Q)
        diag_elements = d_star * term * (1 - c * d_star / Q)  # Shape: (n,)

        # Set the diagonal elements
        np.fill_diagonal(gradient, diag_elements)

        return gradient

def solve_optimization(gainF, risk, cost, alpha, Q):
    gainF = gainF.detach().cpu().numpy() if isinstance(gainF, torch.Tensor) else gainF
    risk = risk.detach().cpu().numpy() if isinstance(risk, torch.Tensor) else risk
    cost = cost.detach().cpu().numpy() if isinstance(cost, torch.Tensor) else cost

    risk = risk.clip(0.001)
    gainF, risk, cost = gainF.flatten(), risk.flatten(), cost.flatten()
    d = cp.Variable(risk.shape, nonneg=True)

    if gainF.shape != risk.shape or risk.shape != cost.shape:
        raise ValueError("Dimensions of gainF, risk, and cost do not match")

    utils = cp.multiply(cp.multiply(gainF, risk), d)
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
    optimal_value = AlphaFairness(optimal_decision * gainF * risk, alpha)

    return optimal_decision, optimal_value



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


# Function to solve the coupled group-based alpha-fairness problem
def solve_coupled_group_alpha_slow(b, c, group_idx, Q, alpha, beta=None):
    """
    Calculates the optimal allocation d for the group-based alpha-fairness
    problem where alpha = beta, based on the provided proposition.

    This function implements three distinct closed-form solutions for the cases:
    1. 0 < alpha < 1
    2. alpha = 1
    3. alpha > 1

    Args:
        b (np.ndarray): Vector of benefit coefficients (denoted as r_i in the proposition).
        c (np.ndarray): Vector of cost coefficients.
        group_idx (np.ndarray): Array of group assignments for each individual.
        Q (float): Total budget.
        alpha (float): The fairness parameter.

    Returns:
        np.ndarray: The optimal allocation vector d*.
    """
    # Ensure inputs are NumPy arrays
    b = np.asarray(b).reshape(-1)
    c = np.asarray(c).reshape(-1)
    group_idx = np.asarray(group_idx).reshape(-1)
    
    if alpha <= 0:
        raise ValueError("This closed-form solution is defined for alpha > 0.")

    # --- Case 1: alpha = 1 (Proportional Fairness) ---
    if abs(alpha - 1.0) < 1e-9:
        unique_groups = np.unique(group_idx)
        K = len(unique_groups)
        d_star = np.zeros_like(b, dtype=float)
        
        for k in unique_groups:
            members_mask = (group_idx == k)
            G_k = np.sum(members_mask)
            if G_k == 0: continue
            
            # d_i = Q / (K * |G_k| * c_i)
            d_star[members_mask] = Q / (K * G_k * c[members_mask])
            
        return d_star

    # --- Cases for alpha != 1 ---
    
    # Pre-compute S_k and H_k, which are common to both remaining cases.
    unique_groups = np.unique(group_idx)
    S, H = {}, {}
    for k in unique_groups:
        members_mask = (group_idx == k)
        r_k, c_k = b[members_mask], c[members_mask]
        
        # S_k = sum( (c_i^(-1/a) * r_i^(1/a))^(1-a) )
        term_s = (c_k**(-1/alpha) * r_k**(1/alpha))**(1-alpha)
        S[k] = np.sum(term_s)

        # H_k = sum( c_i^((a-1)/a) * r_i^((1-a)/a) )
        term_h = (c_k**((alpha - 1) / alpha)) * (r_k**((1 - alpha) / alpha))
        H[k] = np.sum(term_h)

    # Calculate the group-specific prefactor Psi_k based on the alpha regime
    Psi = {}
    if 0 < alpha < 1:
        exponent = 1 / (-2 + alpha)
        for k in unique_groups:
            Psi[k] = (S[k] / (1 - alpha)) ** exponent
            
    elif alpha > 1:
        exponent = (-alpha + 2) / (-alpha**2 + 2 * alpha - 2)
        for k in unique_groups:
            Psi[k] = (S[k] / (alpha - 1)) ** exponent
            
    else: # Should not be reached due to the initial check
        raise ValueError("Invalid value for alpha.")

    # Calculate the global normalization constant Xi
    Xi = np.sum([H[k] * Psi[k] for k in unique_groups])
    if abs(Xi) < 1e-12:
        raise ValueError("Normalization constant Xi is near-zero, leading to instability.")

    # Assemble the final solution
    d_star = np.zeros_like(b, dtype=float)
    # The individual phi_i term is the same in both cases
    phi_all = c**(-1/alpha) * b**((1-alpha)/alpha)
    
    for k in unique_groups:
        members_mask = (group_idx == k)
        # d_i = (Q / Xi) * Psi_k * phi_i
        d_star[members_mask] = (Q / Xi) * Psi[k] * phi_all[members_mask]
        
    return d_star

# Function to compute the Jacobian of the coupled group-based alpha-fairness problem
def solve_coupled_group_grad(b, c, group_idx, Q, alpha, beta=None):
    """
    Computes the Jacobian matrix d(d*)/d(b) for the new alpha=beta formulation.
    Handles the three distinct regimes for alpha.
    """
    b = np.asarray(b).reshape(-1)
    c = np.asarray(c).reshape(-1)
    group_idx = np.asarray(group_idx).reshape(-1)
    n = len(b)
    jacobian = np.zeros((n, n))

    if alpha <= 0:
        raise ValueError("This solution is defined for alpha > 0.")

    # --- Case 1: alpha = 1 ---
    if abs(alpha - 1.0) < 1e-9:
        # For alpha=1, d_i* = Q / (K * |G_k| * c_i), which does not depend on b.
        # Therefore, the gradient is a zero matrix.
        return jacobian

    # --- Cases for alpha != 1 ---

    # --- 1. Forward Pass: Pre-compute all terms from the solver ---
    d_star = solve_coupled_group_alpha_slow(b, c, group_idx, Q, alpha)
    unique_groups = np.unique(group_idx)
    S, H, Psi = {}, {}, {}

    for k in unique_groups:
        members_mask = (group_idx == k)
        r_k, c_k = b[members_mask], c[members_mask]
        S[k] = np.sum((c_k**(-1/alpha) * r_k**(1/alpha))**(1-alpha))
        H[k] = np.sum((c_k**((alpha - 1) / alpha)) * (r_k**((1 - alpha) / alpha)))

    # Calculate Psi_k based on the alpha regime
    if 0 < alpha < 1:
        exponent = 1 / (-2 + alpha)
        psi_s_exp_factor = exponent
        for k in unique_groups:
            Psi[k] = (S[k] / (1 - alpha)) ** exponent
    else: # alpha > 1
        exponent = (-alpha + 2) / (-alpha**2 + 2 * alpha - 2)
        psi_s_exp_factor = exponent
        for k in unique_groups:
            Psi[k] = (S[k] / (alpha - 1)) ** exponent
            
    Xi = np.sum([H[k] * Psi[k] for k in unique_groups])
    phi_all = (c**(-1/alpha)) * (b**((1-alpha)/alpha))

    # --- 2. Backward Pass: Calculate Jacobian column by column ---
    for j in range(n):  # Differentiating with respect to b_j (r_j)
        m = group_idx[j]  # Group of the variable b_j
        
        # Derivatives of S and H w.r.t b_j
        # dS/db_j = d/db_j [ sum(...) ] = (1-a)/a * ... * (1/b_j)
        dS_m_db_j = ((1 - alpha) / alpha) * (c[j]**(-(1-alpha)/alpha)) * (b[j]**((1-2*alpha)/alpha))
        dH_m_db_j = ((1 - alpha) / alpha) * (c[j]**((alpha-1)/alpha)) * (b[j]**((1-2*alpha)/alpha))
        
        # d(Psi)/d(S) * d(S)/d(b_j)
        dPsi_m_db_j = (psi_s_exp_factor / S[m]) * Psi[m] * dS_m_db_j
        dXi_db_j = dH_m_db_j * Psi[m] + H[m] * dPsi_m_db_j
        
        for i in range(n):  # Calculating derivative for d_i
            k = group_idx[i]  # Group of the component d_i
            
            dphi_i_db_j = 0
            if i == j:
                dphi_i_db_j = ((1 - alpha) / alpha) * (c[i]**(-1/alpha)) * (b[i]**((1-2*alpha)/alpha))

            dPsi_k_db_j = dPsi_m_db_j if k == m else 0
            
            dN_i_db_j = Q * (dPsi_k_db_j * phi_all[i] + Psi[k] * dphi_i_db_j)
            
            jacobian[i, j] = (1 / Xi) * dN_i_db_j - (d_star[i] / Xi) * dXi_db_j

    return jacobian


def solve_group(b, c, group_idx, Q, alpha, return_intermediates=False):
    """
    Computes the optimal d* using a fast, vectorized approach.
    (Corrected version to fix the reduceat logic)
    """
    # --- Setup and Input Validation ---
    b, c, group_idx = map(np.asarray, [b, c, group_idx])
    if alpha <= 0:
        raise ValueError("This solution is defined for alpha > 0.")
    if abs(alpha - 1.0) < 1e-9:
        unique_groups, group_inv, group_counts = np.unique(group_idx, return_inverse=True, return_counts=True)
        K = len(unique_groups)
        G_k_mapped = group_counts[group_inv]
        d_star = Q / (K * G_k_mapped * c)
        # For the alpha=1 case, no intermediates are needed for the zero-gradient
        return (d_star, None) if return_intermediates else d_star

    # --- Vectorized Calculations for alpha != 1 ---
    unique_groups, group_inv = np.unique(group_idx, return_inverse=True)
    
    # <<<<<<<<<<<<<<<<<<<<<<<< FIX IS HERE <<<<<<<<<<<<<<<<<<<<<<<<
    # To use np.add.reduceat correctly, we must sort the data by group first.
    
    # 1. Get the indices that would sort the group_idx array
    sort_order = np.argsort(group_idx)
    
    # 2. Find the start index of each group in the *sorted* array
    sorted_groups = group_idx[sort_order]
    _, group_start_indices = np.unique(sorted_groups, return_index=True)
    
    # 3. Apply reduceat to the data, which must also be sorted in the same order
    term_s_all = (c**(-1/alpha) * b**(1/alpha))**(1-alpha)
    term_h_all = (c**((alpha - 1)/alpha)) * (b**((1 - alpha)/alpha))
    
    S_k = np.add.reduceat(term_s_all[sort_order], group_start_indices)
    H_k = np.add.reduceat(term_h_all[sort_order], group_start_indices)
    # >>>>>>>>>>>>>>>>>>>>>>> END OF FIX >>>>>>>>>>>>>>>>>>>>>>>>>

    # Calculate Psi_k vectorially
    if 0 < alpha < 1:
        exponent = 1 / (alpha - 2)
        Psi_k = (S_k / (1 - alpha))**exponent
    else:  # alpha > 1
        exponent = (2 - alpha) / (alpha**2 - 2*alpha + 2)
        Psi_k = (S_k / (alpha - 1))**exponent

    # Calculate global normalization constant Xi
    Xi = np.sum(H_k * Psi_k)
    
    # Map group-level values back to each individual to find d*
    Psi_mapped = Psi_k[group_inv]
    phi_all = c**(-1/alpha) * b**((1-alpha)/alpha)
    d_star = (Q / Xi) * Psi_mapped * phi_all

    if return_intermediates:
        intermediates = {
            'd_star': d_star, 'S_k': S_k, 'H_k': H_k, 'Psi_k': Psi_k,
            'Xi': Xi, 'phi_all': phi_all, 'exponent': exponent,
            'group_inv': group_inv, 'group_idx': group_idx
        }
        return d_star, intermediates
    
    return d_star

def solve_group_grad(b, c, group_idx, Q, alpha):
    """
    Computes the Jacobian d(d*)/d(b) by calling the fast solver and
    using its intermediate results to build the gradient vectorially.
    """
    b, c, group_idx = map(np.asarray, [b, c, group_idx])
    n = len(b)
    
    # Handle alpha=1 case: gradient is always zero
    if abs(alpha - 1.0) < 1e-9:
        return np.zeros((n, n))
        
    # --- 1. Call the solver to get d* and all pre-computed values ---
    d_star, intermediates = solve_group(
        b, c, group_idx, Q, alpha, return_intermediates=True
    )
    # Unpack the intermediates for use
    S_k = intermediates['S_k']
    H_k = intermediates['H_k']
    Psi_k = intermediates['Psi_k']
    Xi = intermediates['Xi']
    phi_all = intermediates['phi_all']
    exponent = intermediates['exponent']
    group_inv = intermediates['group_inv']
    
    # --- 2. Vectorized Backward Pass ---
    # Derivatives of S_k and H_k w.r.t b_j (these are diagonal)
    dS_db_diag = ((1 - alpha) / alpha) * (c**(-(1-alpha)/alpha)) * (b**((1-2*alpha)/alpha))
    dH_db_diag = ((1 - alpha) / alpha) * (c**((alpha-1)/alpha)) * (b**((1-2*alpha)/alpha))
    
    # Map group-level values needed for derivatives
    Psi_mapped = Psi_k[group_inv]
    S_mapped = S_k[group_inv]
    H_mapped = H_k[group_inv]

    # Vector of derivatives d(Psi_k)/db_j and d(Xi)/db_j
    dPsi_db_diag = exponent * (Psi_mapped / S_mapped) * dS_db_diag
    dXi_db = dH_db_diag * Psi_mapped + H_mapped * dPsi_db_diag

    # Build the Jacobian using matrix operations instead of loops
    # d(phi_i)/db_j is a diagonal matrix
    dphi_db_diag = ((1 - alpha) / alpha) * (phi_all / b)
    
    # Create a boolean mask where mask[i,j] is true if i and j are in the same group
    same_group_mask = intermediates['group_idx'][:, None] == intermediates['group_idx'][None, :]
    
    # Build the two parts of the numerator's derivative
    term1_dN = np.outer(phi_all, dPsi_db_diag) * same_group_mask
    term2_dN = np.diag(Psi_mapped * dphi_db_diag)
    grad_N = Q * (term1_dN + term2_dN)
    
    # Build the term from the derivative of the denominator (Xi)
    grad_Xi_term = np.outer(d_star, dXi_db)
    
    # Final Jacobian using the quotient rule in matrix form: (N'D - ND') / D^2
    jacobian = (grad_N - grad_Xi_term) / Xi

    return jacobian

# Function to compute the objective value for the coupled group-based alpha-fairness problem
def compute_coupled_group_obj(d, b, group_idx, alpha, beta=None):
    """
    Calculates the objective value for the new alpha=beta formulation.
    """
    d = np.asarray(d).reshape(-1)
    b = np.asarray(b).reshape(-1)
    group_idx = np.asarray(group_idx).reshape(-1)
    
    # Epsilon for numerical stability
    epsilon = 1e-12
    y = b * d + epsilon
    unique_groups = np.unique(group_idx)
    g_k_values = np.zeros(len(unique_groups), dtype=float)

    # --- Step 1: Calculate group utilities (g_k) based on the proposition ---
    for i, k in enumerate(unique_groups):
        members_mask = (group_idx == k)
        y_k = y[members_mask]
        
        if 0 < alpha < 1:
            g_k_values[i] = np.sum(y_k**(1 - alpha)) / (1 - alpha)
        elif alpha > 1:
            g_k_values[i] = (alpha - 1) / np.sum(y_k**(1 - alpha))
        elif abs(alpha - 1.0) < epsilon:
            # For alpha=1, the group utility is sum(log(y_i))
            g_k_values[i] = np.sum(np.log(y_k))
        else: # alpha <= 0
             # For alpha=0, it's sum(y_i). Follows g_k formula for alpha<1.
             g_k_values[i] = np.sum(y_k)

    # --- Step 2: Apply the outer fairness function F(g_k) and aggregate ---
    if alpha == float('inf') or str(alpha).lower() == 'inf':
        # Rawlsian objective is the minimum group utility
        objective_value = np.min(g_k_values)
    elif abs(alpha - 1.0) < epsilon:
        # Objective is sum(log(g_k))
        objective_value = np.sum(np.log(g_k_values + epsilon))
    elif abs(alpha - 0.0) < epsilon:
        # Utilitarian objective is sum(g_k)
        objective_value = np.sum(g_k_values)
    else:
        # General objective is sum(g_k^(1-alpha) / (1-alpha))
        f_alpha_values = (g_k_values**(1 - alpha)) / (1 - alpha)
        objective_value = np.sum(f_alpha_values)
        
    return objective_value

# Function to compute the objective value using PyTorch for the coupled group-based alpha-fairness problem
def compute_coupled_group_obj_torch(d, b, group_idx, alpha, beta=None):
    """
    🚀 Vectorized version of the objective function.
    """
    d, b, group_idx = map(torch.as_tensor, [d, b, group_idx])
    if d.dtype != torch.float64: d = d.to(torch.float64)
    if b.dtype != torch.float64: b = b.to(torch.float64)

    epsilon = 1e-12
    y = b * d
    unique_groups, group_inv_indices = torch.unique(group_idx, return_inverse=True)
    num_groups = len(unique_groups)
    g_k_values = torch.zeros(num_groups, dtype=torch.float64, device=d.device)

    # --- Step 1: Calculate group utilities (g_k) without loops ---
    if 0 < alpha < 1:
        # Calculate (y_k^(1-a))/(1-a) for all y, then sum by group
        per_element_g = y.pow(1 - alpha) / (1 - alpha)
        g_k_values.scatter_add_(0, group_inv_indices, per_element_g)
    elif alpha > 1:
        # Calculate y_k^(1-a) for all y, sum by group to get the denominator
        per_element_denom = y.pow(1 - alpha)
        sum_denom_k = torch.zeros(num_groups, dtype=torch.float64, device=d.device)
        sum_denom_k.scatter_add_(0, group_inv_indices, per_element_denom)
        g_k_values = (alpha - 1) / (sum_denom_k + epsilon)
    elif abs(alpha - 1.0) < epsilon:
        # Calculate log(y_k) for all y, then sum by group
        per_element_g = torch.log(y + epsilon)
        g_k_values.scatter_add_(0, group_inv_indices, per_element_g)
    else: # alpha <= 0
        # This is just a grouped sum of y
        g_k_values.scatter_add_(0, group_inv_indices, y)
    
    # --- Step 2: Apply the outer fairness function F(g_k) ---
    # This part of your code was already vectorized and is fine.
    if alpha == 'inf' or (isinstance(alpha, str) and alpha.lower() == 'inf'):
        objective_value = torch.min(g_k_values)
    elif abs(alpha - 1.0) < epsilon:
        objective_value = torch.sum(torch.log(g_k_values + epsilon))
    elif abs(alpha - 0.0) < epsilon:
        objective_value = torch.sum(g_k_values)
    else:
        f_alpha_values = (g_k_values.pow(1 - alpha)) / (1 - alpha)
        objective_value = torch.sum(f_alpha_values)
        
    return objective_value


# ==============================================================================
# ===== GROUP-BASED ALPHA-FAIRNESS GRADIENT
# ==============================================================================

def compute_group_gradient_analytical_slow(d: torch.Tensor, b: torch.Tensor, group_idx: torch.Tensor, alpha):
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
    d = torch.as_tensor(d, dtype=torch.float64, device=d.device)
    b = torch.as_tensor(b, dtype=torch.float64, device=d.device)
    group_idx = torch.as_tensor(group_idx, device=d.device)
    
    u = b * d
    epsilon = 1e-12
    beta = alpha

    unique_groups, group_inv_indices, group_counts = torch.unique(group_idx, return_inverse=True, return_counts=True)
    num_groups = len(unique_groups)
    mu_k_values = torch.zeros(num_groups, dtype=torch.float64, device=d.device)
    
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
        sum_g_beta_k = torch.zeros(num_groups, dtype=torch.float64, device=d.device).scatter_add_(0, group_inv_indices, g_beta_values)
        mu_k_values = sum_g_beta_k

    # --- Term 1: dJ/d(mu_k) ---
    dJ_dmuk = torch.zeros_like(mu_k_values, device=d.device)
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

def compute_group_gradient_analytical(d: torch.Tensor, b: torch.Tensor, group_idx: torch.Tensor, alpha):
    """
    🚀 Vectorized version of the analytical gradient function.
    """
    d, b, group_idx = map(lambda t: torch.as_tensor(t, device=d.device), [d, b, group_idx])
    if d.dtype != torch.float64: d = d.to(torch.float64)
    if b.dtype != torch.float64: b = b.to(torch.float64)
    
    u = b * d
    epsilon = 1e-12
    beta = alpha

    unique_groups, group_inv_indices, group_counts = torch.unique(group_idx, return_inverse=True, return_counts=True)
    num_groups = len(unique_groups)
    
    # --- Pre-computation: Calculate mu_k for each group (Vectorized) ---
    mu_k_values = torch.zeros(num_groups, dtype=torch.float64, device=d.device)
    if beta > 1:
        # <<< This is the main change: removing the for loop >>>
        # 1. Compute component of denominator for all elements
        denom_components = u.pow(1 - beta)
        # 2. Sum components by group to get full denominators
        sum_denom_k = torch.zeros(num_groups, dtype=torch.float64, device=d.device)
        sum_denom_k.scatter_add_(0, group_inv_indices, denom_components)
        # 3. Calculate all mu_k values at once
        mu_k_values = (beta - 1) / (sum_denom_k + epsilon)
    else:
        # This part of your code was already correctly vectorized
        if abs(beta - 1.0) < epsilon:
            g_beta_values = torch.log(u + epsilon)
        else:
            g_beta_values = (u.pow(1 - beta)) / (1 - beta)
        mu_k_values.scatter_add_(0, group_inv_indices, g_beta_values)

    # --- The rest of your code is already well-vectorized and requires no changes ---

    # --- Term 1: dJ/d(mu_k) ---
    dJ_dmuk = torch.zeros_like(mu_k_values, device=d.device)
    if alpha == 1:
        dJ_dmuk = 1.0 / (mu_k_values + epsilon)
    elif alpha == 0:
        dJ_dmuk = torch.ones_like(mu_k_values)
    elif alpha == 'inf' or (isinstance(alpha, str) and alpha.lower() == 'inf'):
        min_group_idx = torch.argmin(mu_k_values)
        dJ_dmuk[min_group_idx] = 1.0
    else:
        dJ_dmuk = mu_k_values.pow(-alpha)
    dJ_dmuk_mapped = dJ_dmuk[group_inv_indices]

    # --- Term 2: d(mu_k)/d(u_j) ---
    dmuk_duj = torch.zeros_like(d)
    if beta > 1:
        mu_k_mapped = mu_k_values[group_inv_indices]
        dmuk_duj = mu_k_mapped.pow(2) * u.pow(-beta)
    else:
        n_k_mapped = group_counts[group_inv_indices]
        dmuk_duj = (1.0 / n_k_mapped) * u.pow(-beta)
        
    # --- Term 3: du_j/d_j ---
    dud_dj = b

    # --- Combine using Chain Rule ---
    gradient = dJ_dmuk_mapped * dmuk_duj * dud_dj
    return gradient