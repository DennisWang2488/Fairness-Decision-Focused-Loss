# profile_test.py
import numpy as np
from line_profiler import profile

# We need a dummy @profile decorator if not running with kernprof
# This allows the script to run normally as well.
try:
    # This will be injected by kernprof
    profile
except NameError:
    # If not, create a dummy decorator
    def profile(func):
        return func

# ==============================================================================
# SLOW VERSION (Your original code with the O(n^2) loop)
# ==============================================================================

def solve_coupled_group_alpha_slow(b, c, group_idx, Q, alpha):
    # This is the original, slow solver needed by the slow gradient function
    unique_groups = np.unique(group_idx)
    S, H, Psi = {}, {}, {}
    for k in unique_groups:
        members_mask = (group_idx == k)
        r_k, c_k = b[members_mask], c[members_mask]
        S[k] = np.sum((c_k**(-1/alpha) * r_k**(1/alpha))**(1-alpha))
        H[k] = np.sum((c_k**((alpha - 1) / alpha)) * (r_k**((1 - alpha) / alpha)))
    if 0 < alpha < 1:
        exponent = 1 / (-2 + alpha)
        for k in unique_groups: Psi[k] = (S[k] / (1 - alpha)) ** exponent
    else:
        exponent = (-alpha + 2) / (-alpha**2 + 2 * alpha - 2)
        for k in unique_groups: Psi[k] = (S[k] / (alpha - 1)) ** exponent
    Xi = np.sum([H[k] * Psi[k] for k in unique_groups])
    phi_all = c**(-1/alpha) * b**((1-alpha)/alpha)
    d_star = np.zeros_like(b, dtype=float)
    for k in unique_groups:
        members_mask = (group_idx == k)
        d_star[members_mask] = (Q / Xi) * Psi[k] * phi_all[members_mask]
    return d_star

@profile
def solve_coupled_group_grad_slow(b, c, group_idx, Q, alpha):
    n = len(b)
    jacobian = np.zeros((n, n))
    d_star = solve_coupled_group_alpha_slow(b, c, group_idx, Q, alpha) # Uses the slow solver
    unique_groups = np.unique(group_idx)
    S, H, Psi = {}, {}, {}
    for k in unique_groups:
        members_mask = (group_idx == k)
        r_k, c_k = b[members_mask], c[members_mask]
        S[k] = np.sum((c_k**(-1/alpha) * r_k**(1/alpha))**(1-alpha))
        H[k] = np.sum((c_k**((alpha - 1) / alpha)) * (r_k**((1 - alpha) / alpha)))
    if 0 < alpha < 1:
        exponent = 1 / (-2 + alpha); psi_s_exp_factor = exponent
        for k in unique_groups: Psi[k] = (S[k] / (1 - alpha)) ** exponent
    else:
        exponent = (-alpha + 2) / (-alpha**2 + 2 * alpha - 2); psi_s_exp_factor = exponent
        for k in unique_groups: Psi[k] = (S[k] / (alpha - 1)) ** exponent
    Xi = np.sum([H[k] * Psi[k] for k in unique_groups])
    phi_all = (c**(-1/alpha)) * (b**((1-alpha)/alpha))

    for j in range(n):
        m = group_idx[j]
        dS_m_db_j = ((1 - alpha) / alpha) * (c[j]**(-(1-alpha)/alpha)) * (b[j]**((1-2*alpha)/alpha))
        dH_m_db_j = ((1 - alpha) / alpha) * (c[j]**((alpha-1)/alpha)) * (b[j]**((1-2*alpha)/alpha))
        dPsi_m_db_j = (psi_s_exp_factor / S[m]) * Psi[m] * dS_m_db_j
        dXi_db_j = dH_m_db_j * Psi[m] + H[m] * dPsi_m_db_j
        for i in range(n):
            k = group_idx[i]
            dphi_i_db_j = 0
            if i == j:
                dphi_i_db_j = ((1 - alpha) / alpha) * (c[i]**(-1/alpha)) * (b[i]**((1-2*alpha)/alpha))
            dPsi_k_db_j = dPsi_m_db_j if k == m else 0
            dN_i_db_j = Q * (dPsi_k_db_j * phi_all[i] + Psi[k] * dphi_i_db_j)
            jacobian[i, j] = (1 / Xi) * dN_i_db_j - (d_star[i] / Xi) * dXi_db_j
    return jacobian


# ==============================================================================
# FAST VERSION (My vectorized code)
# ==============================================================================

def solve_coupled_group_alpha_fast(b, c, group_idx, Q, alpha, return_intermediates=False):
    b, c, group_idx = map(np.asarray, [b, c, group_idx])
    if abs(alpha - 1.0) < 1e-9:
        unique_groups, group_inv, group_counts = np.unique(group_idx, return_inverse=True, return_counts=True)
        K, G_k_mapped = len(unique_groups), group_counts[group_inv]
        d_star = Q / (K * G_k_mapped * c)
        return (d_star, None) if return_intermediates else d_star
    unique_groups, group_inv = np.unique(group_idx, return_inverse=True)
    sort_order = np.argsort(group_idx)
    sorted_groups = group_idx[sort_order]
    _, group_start_indices = np.unique(sorted_groups, return_index=True)
    term_s_all = (c**(-1/alpha) * b**(1/alpha))**(1-alpha)
    term_h_all = (c**((alpha-1)/alpha)) * (b**((1-alpha)/alpha))
    S_k = np.add.reduceat(term_s_all[sort_order], group_start_indices)
    H_k = np.add.reduceat(term_h_all[sort_order], group_start_indices)
    if 0 < alpha < 1: exponent = 1/(alpha-2); Psi_k = (S_k/(1-alpha))**exponent
    else: exponent = (2-alpha)/(alpha**2-2*alpha+2); Psi_k = (S_k/(alpha-1))**exponent
    Xi = np.sum(H_k * Psi_k)
    Psi_mapped = Psi_k[group_inv]
    phi_all = c**(-1/alpha) * b**((1-alpha)/alpha)
    d_star = (Q / Xi) * Psi_mapped * phi_all
    if return_intermediates:
        intermediates = { 'd_star': d_star, 'S_k': S_k, 'H_k': H_k, 'Psi_k': Psi_k, 'Xi': Xi, 'phi_all': phi_all, 'exponent': exponent, 'group_inv': group_inv, 'group_idx': group_idx }
        return d_star, intermediates
    return d_star

@profile
def solve_coupled_group_grad_fast(b, c, group_idx, Q, alpha):
    b, c, group_idx = map(np.asarray, [b, c, group_idx])
    n = len(b)
    if abs(alpha - 1.0) < 1e-9: return np.zeros((n, n))
    d_star, intermediates = solve_coupled_group_alpha_fast(b, c, group_idx, Q, alpha, return_intermediates=True)
    S_k, H_k, Psi_k, Xi, phi_all, exponent, group_inv = (intermediates['S_k'], intermediates['H_k'], intermediates['Psi_k'], intermediates['Xi'], intermediates['phi_all'], intermediates['exponent'], intermediates['group_inv'])
    dS_db_diag = ((1-alpha)/alpha) * (c**(-(1-alpha)/alpha)) * (b**((1-2*alpha)/alpha))
    dH_db_diag = ((1-alpha)/alpha) * (c**((alpha-1)/alpha)) * (b**((1-2*alpha)/alpha))
    Psi_mapped, S_mapped, H_mapped = Psi_k[group_inv], S_k[group_inv], H_k[group_inv]
    dPsi_db_diag = exponent * (Psi_mapped/S_mapped) * dS_db_diag
    dXi_db = dH_db_diag * Psi_mapped + H_mapped * dPsi_db_diag
    dphi_db_diag = ((1-alpha)/alpha) * (phi_all/b)
    same_group_mask = intermediates['group_idx'][:, None] == intermediates['group_idx'][None, :]
    term1_dN = np.outer(phi_all, dPsi_db_diag) * same_group_mask
    term2_dN = np.diag(Psi_mapped*dphi_db_diag)
    grad_N = Q * (term1_dN + term2_dN)
    grad_Xi_term = np.outer(d_star, dXi_db)
    jacobian = (grad_N - grad_Xi_term) / Xi
    return jacobian

# ==============================================================================
# Main execution block to run the profiler
# ==============================================================================
if __name__ == '__main__':
    # Setup sample data
    n_items = 500
    n_groups = 20
    alpha_test = 2.0
    Q_test = 1000.0
    
    np.random.seed(42)
    b_test = np.random.rand(n_items) + 0.5
    c_test = np.random.rand(n_items) + 0.5
    group_idx_test = np.random.randint(0, n_groups, size=n_items)
    
    print("Running SLOW version...")
    slow_result = solve_coupled_group_grad_slow(b_test, c_test, group_idx_test, Q_test, alpha_test)
    
    print("\nRunning FAST version...")
    fast_result = solve_coupled_group_grad_fast(b_test, c_test, group_idx_test, Q_test, alpha_test)
    
    # Verify results are the same
    assert np.allclose(slow_result, fast_result)
    print("\nResults from both versions match.")