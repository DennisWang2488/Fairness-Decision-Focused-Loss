"""Utilities to compute alpha-fair objective values and normalized regret."""

from __future__ import annotations

import numpy as np
import torch


def _is_inf_alpha(alpha) -> bool:
    return alpha == "inf" or alpha == float("inf")


def _alpha_fairness_torch(util: torch.Tensor, alpha):
    util = util.clamp_min(1e-12)
    if alpha == 1:
        return torch.log(util).sum()
    if alpha == 0:
        return util.sum()
    if _is_inf_alpha(alpha):
        return util.min()
    return torch.sum(util.pow(1 - alpha) / (1 - alpha))


def _alpha_fairness_numpy(util: np.ndarray, alpha):
    util = np.clip(util, 1e-12, None)
    if alpha == 1:
        return np.log(util).sum()
    if alpha == 0:
        return util.sum()
    if _is_inf_alpha(alpha):
        return util.min()
    return np.sum(util ** (1 - alpha) / (1 - alpha))


def objValue(d, b, alpha):
    """Evaluate alpha-fair objective for utility ``u = d * b``."""
    if isinstance(d, torch.Tensor) or isinstance(b, torch.Tensor):
        d_t = d if isinstance(d, torch.Tensor) else torch.as_tensor(d)
        b_t = b if isinstance(b, torch.Tensor) else torch.as_tensor(b, device=d_t.device)
        return _alpha_fairness_torch((d_t * b_t).reshape(-1), alpha)

    d_np = np.asarray(d)
    b_np = np.asarray(b)
    return _alpha_fairness_numpy((d_np * b_np).reshape(-1), alpha)


def calRegret(predictor, optModel, data, alpha):
    """Compute normalized regret using predicted decisions vs. true objective."""
    preds = predictor(data)
    true_obj = data["true_obj"]
    pred_sol, _ = optModel(preds)
    pred_obj = objValue(pred_sol, data["benefit"], alpha)

    if isinstance(true_obj, torch.Tensor):
        denom = torch.abs(true_obj) + 1e-7
        return (true_obj - pred_obj) / denom

    true_obj_np = np.asarray(true_obj)
    denom = np.abs(true_obj_np) + 1e-7
    return (true_obj_np - np.asarray(pred_obj)) / denom
