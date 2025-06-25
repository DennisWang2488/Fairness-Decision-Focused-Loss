import torch

def atkinson_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    race: torch.Tensor | None = None,
    beta: float = 0.5,
    mode: str = "individual"
) -> torch.Tensor:
    """
    Compute the Atkinson index on squared errors.

    Parameters
    ----------
    pred, true : 1-D float tensors of length n
        Model predictions and ground-truth values.
    race : 1-D tensor of same length (float or int, optional)
        Encodes group membership; required when mode == 'between'.
    beta : float in [0, 1)
        Inequality aversion (0 = utilitarian, →1 emphasises the tail).
    mode : 'individual' | 'between'
        'individual'  - overall inequality across all data points.
        'between'     - inequality across group means (no within-group dispersion).

    Returns
    -------
    scalar  (shape = [])
        Atkinson index A_beta.
    """
    b = (pred - true).pow(2).clamp_min(1e-12)      # shape (n,)
    mu = b.mean()                                  # overall mean μ

    if mode == "individual":
        # ----- overall / individual-level formulation -----
        if abs(beta - 1.0) > 1e-8:
            ede = b.pow(1.0 - beta).mean().pow(1.0 / (1.0 - beta))
        else:  # β → 1 limit → geometric mean
            ede = torch.exp(b.log().mean())
        return 1.0 - ede / mu

    # b1 = mean group 1 MSE and group 2 MSE average
    elif mode == "between":
        if race is None:
            raise ValueError("`race` tensor required for mode='between'.")

        g_id = race.to(dtype=torch.int64)           # (n,)
        G = int(g_id.max().item() + 1)              # number of groups

        # Compute group masks
        mu_g = []
        w_g = []
        n = b.numel()
        for g in range(G):
            mask = (g_id == g)
            n_g = mask.sum()
            if n_g == 0:
                mu_g.append(torch.tensor(0.0, device=pred.device, dtype=pred.dtype))
                w_g.append(torch.tensor(0.0, device=pred.device, dtype=pred.dtype))
            else:
                mu_g.append(b[mask].mean())
                w_g.append(n_g.float() / n)
        mu_g = torch.stack(mu_g)  # (G,)
        w_g = torch.stack(w_g)    # (G,)

        if abs(beta - 1.0) > 1e-8:
            ede = (w_g * mu_g.pow(1.0 - beta)).sum().pow(1.0 / (1.0 - beta))
        else:
            ede = torch.exp((w_g * mu_g.log()).sum())

        return 1.0 - ede / mu

    else:
        raise ValueError("mode must be 'individual' or 'between'.")

def mean_abs_dev(
    pred: torch.Tensor,
    true: torch.Tensor,
    race: torch.Tensor,
    mode: str = "individual"
) -> torch.Tensor:
    """
    Computes the mean absolute deviation (MAD) of squared errors.
    Args:
        pred: Predicted values, shape (n,)
        true: Ground truth values, shape (n,)
        race: Group membership labels (0 or 1), shape (n,)
        mode: "individual" (MAD over all squared errors) or "group" (MAD of group MSEs)
    Returns:
        mad: scalar tensor
    """

    pred = pred.view(-1)
    true = true.view(-1)
    race = race.view(-1)

    if mode == "individual":
        errors = (pred - true).pow(2)
        mean_error = errors.mean()
        mad = torch.abs(errors - mean_error).mean()
        return mad
    elif mode == "between":
        group_ids = torch.unique(race)
        mses = []
        for g in group_ids:
            mask = (race == g)
            if mask.sum() == 0:
                continue
            mse = ((pred[mask] - true[mask]).pow(2)).mean()
            mses.append(mse)
        mses = torch.stack(mses)
        mean_mse = mses.mean()
        mad = torch.abs(mses - mean_mse).mean()
        return mad
    else:
        raise ValueError("mode must be 'individual' or 'between'")



    
def compute_fairness_grad(pred, true, race, fairness_measure, epsilon=0.0, d_func=None, beta=0.5):
    """
    Compute d(fairness_penalty)/d(pred) for various measures.
    pred, true, race: 1D torch tensors of shape (n,)
    fairness_measure: "individual", "group", "atkinson", "stat_parity", "accuracy_parity"
    """
    if d_func is None:
        d_func = lambda y1, y2: torch.exp(-(y1 - y2).pow(2))
    n = pred.numel()
    grad = torch.zeros_like(pred)

    if fairness_measure == "individual":
        mask0 = (race == 0)
        mask1 = (race == 1)
        n0, n1 = mask0.sum().item(), mask1.sum().item()
        if n0 == 0 or n1 == 0:
            return grad
        pred0, pred1 = pred[mask0], pred[mask1]
        true0, true1 = true[mask0], true[mask1]
        diff = pred0.unsqueeze(1) - pred1.unsqueeze(0)
        D = d_func(true0.unsqueeze(1), true1.unsqueeze(0))
        grad0 = 2 * (D * diff).sum(dim=1) / (n0 * n1)
        grad1 = -2 * (D * diff).sum(dim=0) / (n0 * n1)
        grad[mask0] = grad0
        grad[mask1] = grad1
        return grad

    elif fairness_measure == "atkinson":
        b = (pred - true).pow(2).clamp_min(1e-12)
        mu = b.mean()
        U = b.pow(1.0 - beta).mean()
        if abs(beta - 1.0) > 1e-8:
            term = U.pow(1.0 / (1.0 - beta))
        else:
            term = torch.exp(torch.log(b).mean())
        # Derived gradient:
        grad = -2 * (pred - true) / n * (
            U.pow(beta/(1.0-beta)) / ((1.0-beta)*mu) * b.pow(-beta)
            - term / (mu**2)
        )
        return grad

    elif fairness_measure == "stat_parity":
        mask0 = (race == 0)
        mask1 = (race == 1)
        n0, n1 = mask0.sum().item(), mask1.sum().item()
        if n0 == 0 or n1 == 0:
            return grad
        mean0, mean1 = pred[mask0].mean(), pred[mask1].mean()
        true0, true1 = true[mask0].mean(), true[mask1].mean()
        pred_diff = torch.abs(mean0 - mean1)
        data_diff = torch.abs(true0 - true1)
        bound = data_diff + epsilon
        penalty = pred_diff - bound
        if penalty > 0:
            sign = torch.sign(mean0 - mean1)
            grad[mask0] = sign / n0
            grad[mask1] = -sign / n1
        return grad

    elif fairness_measure == "accuracy_parity":
        mask0 = (race == 0)
        mask1 = (race == 1)
        n0, n1 = mask0.sum().item(), mask1.sum().item()
        if n0 == 0 or n1 == 0:
            return grad
        grad[mask0] = 2 * (pred[mask0] - true[mask0]) / n0
        grad[mask1] = -2 * (pred[mask1] - true[mask1]) / n1
        return grad

    else:
        return grad


def compute_individual_fairness(pred: torch.Tensor,
                                 true: torch.Tensor,
                                 race: torch.Tensor,
                                 d_func=None) -> torch.Tensor:
    """
    Computes individual fairness: similar individuals across groups should receive similar predictions.
    """
    # Flatten everything to 1D
    pred = pred.view(-1)
    true = true.view(-1)
    race = race.view(-1)

    if d_func is None:
        d_func = lambda y1, y2: torch.exp(-(y1 - y2).pow(2))

    mask0 = (race == 0)
    mask1 = (race == 1)

    pred0, pred1 = pred[mask0], pred[mask1]
    true0, true1 = true[mask0], true[mask1]

    n0, n1 = pred0.shape[0], pred1.shape[0]
    if n0 == 0 or n1 == 0:
        return torch.tensor(0.0, device=pred.device)

    pred_diff = pred0.unsqueeze(1) - pred1.unsqueeze(0)       # (n0, n1)
    true_sim = d_func(true0.unsqueeze(1), true1.unsqueeze(0)) # (n0, n1)

    fairness_penalty = (true_sim * pred_diff.pow(2)).mean()
    return fairness_penalty

def compute_statistical_parity(pred: torch.Tensor,
                              race: torch.Tensor) -> torch.Tensor:
    """
    Computes statistical parity: absolute difference in mean predictions between groups.
    """
    pred = pred.view(-1)
    race = race.view(-1)

    mask0 = (race == 0)
    mask1 = (race == 1)

    n0 = mask0.sum().item()
    n1 = mask1.sum().item()
    if n0 == 0 or n1 == 0:
        return torch.tensor(0.0, device=pred.device)

    mean0 = pred[mask0].mean()
    mean1 = pred[mask1].mean()
    stat_parity_penalty = torch.abs(mean0 - mean1)
    return stat_parity_penalty

def compute_group_accuracy_parity(pred: torch.Tensor,
                                   true: torch.Tensor,
                                   race: torch.Tensor) -> torch.Tensor:
    """
    Computes group fairness as the absolute difference in MSE between two race groups.

    Args:
        pred: Predicted values, shape (n,)
        true: Ground truth values, shape (n,)
        race: Group membership labels (0 or 1), shape (n,)

    Returns:
        fairness_penalty: scalar tensor
    """
    pred = pred.view(-1)
    true = true.view(-1)
    race = race.view(-1)

    mask0 = (race == 0)
    mask1 = (race == 1)

    n0 = mask0.sum().item()
    n1 = mask1.sum().item()
    if n0 == 0 or n1 == 0:
        return torch.tensor(0.0, device=pred.device)

    mse0 = ((pred[mask0] - true[mask0]).pow(2)).mean()
    mse1 = ((pred[mask1] - true[mask1]).pow(2)).mean()
    f_acc = torch.abs(mse0 - mse1)

    return f_acc

def compute_atkinson_index(pred: torch.Tensor,
                           true: torch.Tensor,
                           beta: float = 0.5) -> torch.Tensor:
    """
    Computes the Atkinson index A on squared errors:
        e_i = (pred_i - true_i)^2  (clamped > 0)
        μ   = mean(e_i)
        U   = mean(e_i^(1-β))                  if β ≠ 1
              exp(mean(log(e_i)))             if β = 1
        A   = 1 – [U^(1/(1-β)) / μ]             if β ≠ 1
              1 – [exp(mean(log(e_i))) / μ]     if β = 1

    Args:
        pred: Predicted values, shape (n,)
        true:  Ground‐truth values, shape (n,)
        beta: Atkinson parameter (β ≥ 0)

    Returns:
        A: scalar Atkinson index in [0,1)
    """
    # flatten
    pred = pred.view(-1)
    true = true.view(-1)

    # squared errors, clamped to avoid zeros
    e = (pred - true).pow(2).clamp_min(1e-12)
    mu = e.mean()

    if abs(beta - 1.0) > 1e-8:
        U = e.pow(1.0 - beta).mean()
        term = U.pow(1.0 / (1.0 - beta))
    else:
        term = torch.exp(torch.log(e).mean())

    A = 1.0 - term / mu
    return A
