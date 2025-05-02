import numpy as np
alpha = 0.5
Q = 1000

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
        denominator = np.sum(numerator)
        
        if denominator == 0:
            raise ValueError("Denominator is zero in closed-form solution.")
            
        d_star_closed = (numerator / denominator) * Q
    
    # if not np.isclose(np.sum(c * d_star_closed), Q, rtol=1e-5):
    #     raise ValueError("Solution does not satisfy budget constraint.")
        
    return d_star_closed

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
    if alpha == 0:
        # Utilitarian case: Gradient is undefined or zero
        # Since allocation is discrete, gradients are effectively zero
        return np.zeros((len(c), len(c)))

    elif alpha == 1:
        # Nash welfare case: Gradient is zero
        return np.zeros((len(c), len(c)))

    elif alpha == 'inf':
        # Maximin case
        n = len(c)
        utility = r * g  # Shape: (n,)
        S = np.sum(c**2 / utility)  # Scalar

        # Compute d_star
        d_star = compute_d_star_closed_form(g,r,c, alpha='inf', Q=Q)  # Shape: (n,)

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
        d_star = compute_d_star_closed_form(g, r, c, alpha, Q)  # Shape: (n,)

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

def regret(predModel, optModel, dataloader, alphas=[alpha], Q=Q):
    predModel.eval()
    features, risk, gainF, cost, race, true_sols, true_vals = next(iter(dataloader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, risk, gainF, cost, race, true_sols, true_vals = (
        features.to(device),
        risk.to(device),
        gainF.to(device),
        cost.to(device),
        race.to(device),
        true_sols.to(device),
        true_vals.to(device),
    )

    with torch.no_grad():
        pred_risk = predModel(features).clamp(min=0.001)
    risk = risk.clamp(min=0.001)

    pred_risk = pred_risk.cpu().numpy()
    risk = risk.cpu().numpy()
    gainF = gainF.cpu().numpy()
    cost = cost.cpu().numpy()

    regrets = []
    for alpha in alphas:
        true_sol, true_obj = optModel(gainF, risk, cost, alpha, Q)
        pred_sol, _ = optModel(gainF, pred_risk, cost, alpha, Q)
        pred_obj = AlphaFairness(gainF * risk * pred_sol, alpha)
        normalized_regret = (true_obj - pred_obj) / (abs(true_obj) + 1e-7)
        regrets.append(normalized_regret)

    predModel.train()
    return np.mean(regrets)