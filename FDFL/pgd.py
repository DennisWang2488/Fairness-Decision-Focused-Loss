
import sys
sys.path.insert(0, 'E:\\User\\Stevens\\Code\\Fold-opt\\fold_opt')
from GMRES import *
from fold_opt import *
import torch
import numpy as np
import cvxpy as cp
import math


def proj_knapsack_closed(v, c, Q, max_iter: int = 25):
    """
    Water-filling projection (Duchi et al., 2008) – vectorised & differentiable
    v : (B,n)   raw iterate
    c : (n,)    positive costs   (Torch tensor)
    Q : float   budget
    """
    B, n = v.shape
    c = c.to(v)

    # 1) clip to orthant
    y     = v.clamp_min_(0.)
    cost  = (y * c).sum(1)
    mask  = cost > Q
    if not mask.any():
        return y

    # 2) batched bisection for λ s.t. Σ_i max(0, y_i − λ c_i)c_i = Q
    lam_lo = torch.zeros_like(cost)
    lam_hi = (y / c).max(1).values          # tight upper bound
    for _ in range(max_iter):
        lam   = 0.5 * (lam_lo + lam_hi)
        d_tmp = (y - lam.unsqueeze(1)*c).clamp_min_(0.)
        excess = (d_tmp*c).sum(1) - Q
        lam_lo = torch.where(excess > 0, lam, lam_lo)
        lam_hi = torch.where(excess <= 0, lam, lam_hi)

    lam  = lam_hi.unsqueeze(1)
    d_pf = (y - lam*c).clamp_min_(0.)
    return torch.where(mask.unsqueeze(1), d_pf, y)


def proj_knapsack_solver(v, c, Q):
    out = []
    c_np = c.cpu().numpy()
    for row in v.cpu().numpy():
        n   = row.size
        d   = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(0.5*cp.sum_squares(d - row)),
                          [d >= 0, c_np @ d <= Q])
        prob.solve(solver=cp.OSQP, eps_abs=1e-8, verbose=False)
        out.append(torch.tensor(d.value, dtype=v.dtype))
    return torch.stack(out).to(v)

class PGDStep(torch.nn.Module):
    """
    update_step(c, d)   where   c == r   (risk/parameter vector).
    All constants are stored as buffers so autograd only tracks r.
    """
    def __init__(self, g, c_cost, Q: float, alpha: float,
                 lr: float = 5e-2, closed_proj: bool = True):
        super().__init__()
        self.register_buffer("g",      g)        # shape (n,)
        self.register_buffer("c_cost", c_cost)   # shape (n,)
        self.Q      = float(Q)
        self.alpha  = float(alpha)
        self.lr     = float(lr)
        self.closed = closed_proj

    def forward(self, r, d):
        """Projected-gradient step  d_{t+1} ← Π_Ω(d_t − lr ∇f)"""
        util = r * self.g
        if self.alpha == 1.0:
            grad = -(util / d.clamp_min(1e-12))
        else:
            grad = - (util**(1 - self.alpha)) * torch.pow(
                     d.clamp_min(1e-12), -self.alpha)

        d_new = d - self.lr * grad
        proj  = proj_knapsack_closed if self.closed else proj_knapsack_solver
        return proj(d_new, self.c_cost, self.Q)


# ----------------------------------------------------------
def my_solver(c):
    return torch.clamp(c, min=0.0)

def my_update_step(c, x):
    alpha = 0.1
    grad  = x - c
    x_new = torch.clamp(x - alpha*grad, min=0.0)
    return x_new

fold_layer = FoldOptLayer(
                solver      = my_solver,
                update_step = my_update_step,
                n_iter      = 20,
                backprop_rule='FPI')

c      = torch.tensor([[-1.0], [ 2.0]], requires_grad=True)   # (B=2, n=1)
target = torch.tensor([[ 3.0], [ 1.0]])

x_star = fold_layer(c)
print("x* from FoldOptLayer:", x_star.squeeze().tolist())     # → [0.0, 2.0]

loss = 0.5 * torch.sum((x_star - target) ** 2)
loss.backward()

print("Grad wrt c:", c.grad.squeeze().tolist())

