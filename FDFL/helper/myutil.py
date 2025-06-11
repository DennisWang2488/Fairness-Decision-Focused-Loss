"""
Utility functions.
"""
import sys
import warnings
import time
import copy
import json
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import cvxpy as cp

from pyepo.model.opt import optModel

# Suppress warnings
warnings.filterwarnings("ignore")


sys.path.insert(0, 'E:\\User\\Stevens\\MyRepo\\FDFL\\helper')
sys.path.insert(0, 'E:\\User\\Stevens\\MyRepo\\fold-opt-package\\fold_opt')
from GMRES import *
from fold_opt import *


def proj_budget(x, cost, Q, max_iter=60):
    """
    x : (B,n)   or (n,)   –– internally promoted to (B,n)
    cost : (n,) positive
    Q : scalar or length‑B tensor
    """
    batched = x.dim() == 2
    if not batched:                       # (n,)  →  (1,n)
        x = x.unsqueeze(0)

    B, n = x.shape
    cost = cost.to(x)
    Q    = torch.as_tensor(Q, dtype=x.dtype, device=x.device).reshape(-1, 1)  # (B,1)

    d    = x.clamp(min=0.)                # enforce non‑neg
    viol = (d @ cost) > Q.squeeze(1)      # which rows violate the budget?

    if viol.any():
        dv, Qv = d[viol], Q[viol]
        lam_lo = torch.zeros_like(Qv.squeeze(1))
        lam_hi = (dv / cost).max(1).values   # upper bound for λ⋆

        for _ in range(max_iter):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            trial   = (dv - lam_mid[:, None] * cost).clamp(min=0.)
            too_big = (trial @ cost) > Qv.squeeze(1)
            lam_lo[too_big] = lam_mid[too_big]
            lam_hi[~too_big]= lam_mid[~too_big]

        d[viol] = (dv - lam_hi[:, None] * cost).clamp(min=0.)

    return d if batched else d.squeeze(0)   # restore original rank


def alpha_fair(u, alpha):
    if alpha == 1:
        return torch.log(u).sum(-1)
    elif alpha == 0:
        return u.sum(-1)
    elif alpha == 'inf':
        return u.min(-1).values
    return (u.pow(1-alpha)/(1-alpha)).sum(-1)

def pgd_step(r, d, g, cost, Q, alpha, lr):
    d = d.clone().requires_grad_(True)
    obj     = alpha_fair(d * r * g, alpha).sum()
    grad_d, = torch.autograd.grad(obj, d, create_graph=True)
    return proj_budget(d + lr * grad_d, cost, Q)

def closed_form_solver_torch(r, g, cost, alpha, Q):
    if r.dim() == 1:                      # (n,) → (1,n) before looping
        r = r.unsqueeze(0)
    out = []
    for r_i in r:
        d_np, _ = solve_closed_form(g.cpu().numpy(),
                                    r_i.detach().cpu().numpy(),
                                    cost.cpu().numpy(),
                                    alpha, Q)
        out.append(torch.as_tensor(d_np, dtype=r.dtype, device=r.device))
    return torch.stack(out)               # (B,n) even if B=1

def make_foldopt_layer(g, cost, alpha, Q,
                       lr=1e-2, n_fixedpt=200, rule='GMRES'):
    g    = g.detach() # gainF
    cost = cost.detach() # cost

    # -------- solver: no gradients flow ----------------------------
    def solver_fn(r):
        return closed_form_solver_torch(r, g, cost, alpha, Q)

    # -------- one differentiable PGD step --------------------------
    def update_fn(r, x_star, *_):
        # promote to (B,n) if needed
        if r.dim() == 1:       r = r.unsqueeze(0)
        if x_star.dim() == 1:  x_star = x_star.unsqueeze(0)

        g_b = g.expand_as(r) if g.dim() == 1 else g
        return pgd_step(r, x_star, g_b, cost, Q, alpha, lr)  # (B,n)

    return FoldOptLayer(solver_fn, update_fn,
                        n_iter=n_fixedpt, backprop_rule=rule)


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

def AlphaFairnesstorch(util, alpha):
    if alpha == 1:
        return torch.sum(torch.log(util))
    elif alpha == 0:
        return torch.sum(util)
    elif alpha == 'inf':
        return torch.min(util)
    else:
        return torch.sum(util**(1-alpha) / (1-alpha))
    

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

def visLearningCurve(loss_log, loss_log_regret, mse_loss_log, fairness_log=None):
    if fairness_log is not None:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 4))
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))
    # Plot original loss log
    ax1.plot(loss_log, color="c", lw=2)
    ax1.tick_params(axis="both", which="major", labelsize=12)
    ax1.set_xlabel("Iters", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=16)
    ax1.set_title("Training Loss Curve", fontsize=16)
    ax1.grid(True, alpha=0.3)

    # Plot regret log
    ax2.plot(loss_log_regret, color="royalblue", ls="--", alpha=0.7, lw=2)
    ax2.tick_params(axis="both", which="major", labelsize=12)
    ax2.set_xlabel("Epochs", fontsize=16)
    ax2.set_ylabel("Regret", fontsize=16)
    ax2.set_title("Learning Curve (Test Regret)", fontsize=16)
    ax2.grid(True, alpha=0.3)

    # Plot MSE loss log
    ax3.plot(mse_loss_log, color="orange", lw=2)
    ax3.tick_params(axis="both", which="major", labelsize=12)
    ax3.set_xlabel("Iters", fontsize=16)
    ax3.set_ylabel("MSE Loss", fontsize=16)
    ax3.set_title("Learning Curve (MSE Loss)", fontsize=16)
    ax3.grid(True, alpha=0.3)

    # Plot fairness log if provided
    if fairness_log is not None:
        ax4.plot(fairness_log, color="green", lw=2)
        ax4.tick_params(axis="both", which="major", labelsize=12)
        ax4.set_xlabel("Iters", fontsize=16)
        ax4.set_ylabel("Fairness", fontsize=16)
        ax4.set_title("Learning Curve (Fairness)", fontsize=16)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


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
        #
        # d_i* = (c_i^(-1/alpha) * (r_i*g_i)^(1/alpha - 1) * Q) / sum_j(c_j^(-1/alpha) * (r_j*g_j)^(1/alpha - 1))
        
        numerator = np.power(c, -1/alpha) * np.power(utility, 1/alpha - 1)
        denominator = np.sum(numerator)
        
        if denominator == 0:
            raise ValueError("Denominator is zero in closed-form solution.")
            
        d_star_closed = (numerator / denominator) * Q
    
    # if not np.isclose(np.sum(c * d_star_closed), Q, rtol=1e-5):
    #     raise ValueError("Solution does not satisfy budget constraint.")
    obj = AlphaFairness(d_star_closed * utility, alpha)
        
    return d_star_closed, obj


import torch

def solve_closed_form_torch(g, r, c, alpha, Q):
    # Convert inputs to PyTorch tensors if they aren't already
    if not isinstance(g, torch.Tensor):
        g = torch.tensor(g, dtype=torch.float32)
    if not isinstance(r, torch.Tensor):
        r = torch.tensor(r, dtype=torch.float32)
    if not isinstance(c, torch.Tensor):
        c = torch.tensor(c, dtype=torch.float32)
    
    # Input validation
    if c.shape != r.shape or c.shape != g.shape:
        raise ValueError("c, r, and g must have the same shape.")
    if torch.any(c <= 0):
        raise ValueError("All cost values must be positive.")
    if torch.any(r <= 0):
        raise ValueError("All risk values must be positive.")
    if torch.any(g <= 0):
        raise ValueError("All gain factors must be positive.")
    
    n = c.size(0)
    utility = r * g
    
    if alpha == 0:
        ratios = utility / c
        sorted_indices = torch.argsort(ratios, descending=True)
        d_star_closed = torch.zeros_like(c)
        d_star_closed[sorted_indices[0]] = Q / c[sorted_indices[0]]
        
    elif alpha == 1:
        d_star_closed = Q / (n * c)
    
    elif alpha == 'inf':
        d_star_closed = (Q * c) / (utility * torch.sum(c * c / utility))
        
    else:
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError("Alpha must be a positive number for general case.")
        numerator = torch.pow(c, -1/alpha) * torch.pow(utility, 1/alpha - 1)
        denominator = torch.sum(numerator)
        if denominator == 0:
            raise ValueError("Denominator is zero in closed-form solution.")
        d_star_closed = (numerator / denominator) * Q
    
    # Note: Original code returned obj = AlphaFairness(d_star_closed * utility, alpha)
    # Since AlphaFairness is undefined, only d_star_closed is returned
    return d_star_closed, AlphaFairnesstorch(d_star_closed * utility, alpha)


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
    r = r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r
    g = g.detach().cpu().numpy() if isinstance(g, torch.Tensor) else g
    c = c.detach().cpu().numpy() if isinstance(c, torch.Tensor) else c
        
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


def compute_gradient_closed_form_torch(g, r, c, alpha, Q):
    # Convert inputs to PyTorch tensors if they aren't already
    if not isinstance(g, torch.Tensor):
        g = torch.tensor(g, dtype=torch.float32)
    if not isinstance(r, torch.Tensor):
        r = torch.tensor(r, dtype=torch.float32)
    if not isinstance(c, torch.Tensor):
        c = torch.tensor(c, dtype=torch.float32)
    
    n = c.size(0)
    
    if alpha == 0:
        ratios = (r * g) / c
        i_star = torch.argmax(ratios)
        gradient = torch.zeros((n, n), device=c.device, dtype=torch.float32)
        gradient[i_star, i_star] = Q * g[i_star] / c[i_star]
        return gradient
    
    elif alpha == 'inf':
        utility = r * g
        S = torch.sum(c**2 / utility)
        d_star = solve_closed_form(g, r, c, alpha='inf', Q=Q)
        # Off-diagonal elements
        gradient = (d_star[:, None] * c[None, :]**2) / (c[:, None] * r[None, :]**2 * g[None, :] * S)
        # Diagonal elements
        gradient_diag = -d_star / r - (d_star * c) / (r * g * S)
        gradient.diagonal().copy_(gradient_diag)
        return gradient
    
    else:
        if not isinstance(alpha, (int, float)):
            raise TypeError("Alpha must be a positive real number, 0, 1, or 'inf'.")
        if alpha <= 0:
            raise ValueError("Alpha must be positive for gradient computation.")
        d_star = solve_closed_form(g, r, c, alpha, Q)
        term = (1.0 / alpha - 1.0) * g / r
        gradient = -(d_star[:, None] * (d_star * term)[None, :]) / Q
        diag_elements = d_star * term * (1 - d_star / Q)
        gradient.diagonal().copy_(diag_elements)
        return gradient
    
    