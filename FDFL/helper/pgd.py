import math
import sys
from pathlib import Path
import numpy as np
import cvxpy as cp
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
FOLD_OPT_DIR = REPO_ROOT / "fold-opt-package" / "fold_opt"
if str(FOLD_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(FOLD_OPT_DIR))

from fold_opt import FoldOptLayer

try:
    from .myutil import solve_closed_form
except ImportError:
    from myutil import solve_closed_form


# ---------------- little utilities ---------------------------------
def to_col(x):              # (…,n)  ->  (…,n,1)
    return x.unsqueeze(-1) if x.dim() == 1 else x
def from_col(x):            # (…,n,1) -> (…,n)
    return x.squeeze(-1)

def from_numpy_torch(x):  # (n,) -> (1,n) -> (1,n,1)
    return torch.from_numpy(x).float().unsqueeze(0).unsqueeze(-1)
def to_torch_numpy(x):    # (1,n,1) -> (1,n) -> (n,)
    return x.squeeze(0).squeeze(-1).numpy()


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


def alpha_fair_torch(u, alpha):
    if alpha == 1:
        return torch.log(u).sum(-1)
    elif alpha == 0:
        return u.sum(-1)
    elif alpha == 'inf':
        return u.min(-1).values
    return (u.pow(1-alpha)/(1-alpha)).sum(-1)

def pgd_step(r, d, g, cost, Q, alpha, lr):
    d = d.clone().requires_grad_(True)
    obj     = alpha_fair_torch(d * r * g, alpha).sum()
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
                       lr=1e-2, n_fixedpt=40, rule='GMRES'):
    g    = g.detach()
    cost = cost.detach()

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
                        n_iter=n_fixedpt, backprop_rule='FPI')

