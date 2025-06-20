import torch




def compute_fairness_value(pred, true, race, fairness_measure, epsilon=1e-6, d_func=None, beta=0.5):
    """
    Compute the fairness penalty value based on the selected fairness_measure.
    Parameters:
      pred: Tensor of predictions (1D, shape: [n])
      true: Tensor of true values (1D, shape: [n])
      race: Tensor of group indicators (1D, same shape)
      fairness_measure: one of "individual", "group", "atkinson", "stat_parity", "accuracy_parity"
      epsilon: extra allowed margin for stat_parity
      d_func: weighting function for "individual" or "group" (if None, use default exponential)
      beta: parameter used in the Atkinson index
    Returns:
      fairness_value: a scalar torch tensor.
    """
    if d_func is None:
        # Default: for continuous (normalized) targets.
        d_func = lambda y1, y2: torch.exp(-(y1 - y2).pow(2))
    
    n = pred.numel()
    
    if fairness_measure == "individual":
        mask0 = (race == 0)
        mask1 = (race == 1)
        n0 = mask0.sum().item()
        n1 = mask1.sum().item()
        if n0 == 0 or n1 == 0:
            return torch.tensor(0.0, device=pred.device)
        pred0 = pred[mask0]
        pred1 = pred[mask1]
        true0 = true[mask0]
        true1 = true[mask1]
        diff = pred0.unsqueeze(1) - pred1.unsqueeze(0)
        D = d_func(true0.unsqueeze(1), true1.unsqueeze(0))
        f1 = (D * diff.pow(2)).sum() / (n0 * n1)
        return f1


    elif fairness_measure == "group":
        mask0 = (race == 0)
        mask1 = (race == 1)
        n0 = mask0.sum().item()
        n1 = mask1.sum().item()
        if n0 == 0 or n1 == 0:
            return torch.tensor(0.0, device=pred.device)
        pred0 = pred[mask0]
        pred1 = pred[mask1]
        true0 = true[mask0]
        true1 = true[mask1]
        diff = pred0.unsqueeze(1) - pred1.unsqueeze(0)
        D = d_func(true0.unsqueeze(1), true1.unsqueeze(0))
        sum_term = (D * diff).sum() / (n0 * n1)
        f2 = sum_term.pow(2)
        return f2

    elif fairness_measure == "atkinson":
        # Use b_i = (pred - true)^2 (must be > 0)
        b = (pred - true).pow(2).clamp_min(1e-12)
        mu = b.mean()
        U = b.pow(1.0 - beta).mean()
        if abs(beta - 1.0) > 1e-8:
            term = U.pow(1.0 / (1.0 - beta))
        else:
            term = torch.exp(torch.log(b).mean())
        A = 1.0 - term / mu
        return A

    elif fairness_measure == "stat_parity":
        mask0 = (race == 0)
        mask1 = (race == 1)
        n0 = mask0.sum().item()
        n1 = mask1.sum().item()
        if n0 == 0 or n1 == 0:
            return torch.tensor(0.0, device=pred.device)
        mean_pred0 = pred[mask0].mean()
        mean_pred1 = pred[mask1].mean()
        mean_true0 = true[mask0].mean()
        mean_true1 = true[mask1].mean()
        pred_diff = torch.abs(mean_pred0 - mean_pred1)
        data_diff = torch.abs(mean_true0 - mean_true1)
        bound = data_diff + epsilon
        penalty = torch.relu(pred_diff - bound)
        return penalty

    elif fairness_measure == "accuracy_parity":
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

    else:
        return torch.tensor(0.0, device=pred.device)

def compute_fairness_grad(pred, true, race, fairness_measure, epsilon=0.0, d_func=None, beta=0.5):
    """
    Compute the gradient of the fairness penalty (d(fairness_value)/d(pred)).
    Parameters:
      pred: Tensor of predictions (1D, shape: [n])
      true: Tensor of true values (1D, shape: [n])
      race: Tensor of group indicators (1D, shape: [n])
      fairness_measure: one of "individual", "group", "atkinson", "stat_parity", "accuracy_parity"
      epsilon: used for stat_parity
      d_func: weighting function (if None, use default exponential)
      beta: parameter used in the Atkinson index case.
    Returns:
      grad_fair: Tensor of same shape as pred representing the fairness gradient.
    """
    if d_func is None:
        d_func = lambda y1, y2: torch.exp(-(y1 - y2).pow(2))
    
    n = pred.numel()
    grad_fair = torch.zeros_like(pred)
    
    if fairness_measure == "individual":
        mask0 = (race == 0)
        mask1 = (race == 1)
        n0 = mask0.sum().item()
        n1 = mask1.sum().item()
        grad = torch.zeros_like(pred)
        if n0 == 0 or n1 == 0:
            return grad
        pred0 = pred[mask0]
        pred1 = pred[mask1]
        true0 = true[mask0]
        true1 = true[mask1]
        diff = pred0.unsqueeze(1) - pred1.unsqueeze(0)
        D = d_func(true0.unsqueeze(1), true1.unsqueeze(0))
        grad0 = 2 * (D * diff).sum(dim=1) / (n0 * n1)
        grad1 = -2 * (D * diff).sum(dim=0) / (n0 * n1)
        grad[mask0] = grad0
        grad[mask1] = grad1
        return grad

    elif fairness_measure == "group":
        mask0 = (race == 0)
        mask1 = (race == 1)
        n0 = mask0.sum().item()
        n1 = mask1.sum().item()
        grad = torch.zeros_like(pred)
        if n0 == 0 or n1 == 0:
            return grad
        pred0 = pred[mask0]
        pred1 = pred[mask1]
        true0 = true[mask0]
        true1 = true[mask1]
        diff = pred0.unsqueeze(1) - pred1.unsqueeze(0)
        D = d_func(true0.unsqueeze(1), true1.unsqueeze(0))
        sum_term = (D * diff).sum() / (n0 * n1)
        grad0 = 2 * sum_term * (D.sum(dim=1)) / (n0 * n1)
        grad1 = -2 * sum_term * (D.sum(dim=0)) / (n0 * n1)
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
        grad_fair = -2 * (pred - true) / n * ( U.pow(beta/(1.0-beta))/(1.0-beta)/ mu * b.pow(-beta) - term/(mu**2) )
        return grad_fair

    elif fairness_measure == "stat_parity":
        mask0 = (race == 0)
        mask1 = (race == 1)
        n0 = mask0.sum().item()
        n1 = mask1.sum().item()
        if n0 == 0 or n1 == 0:
            return grad_fair
        mean_pred0 = pred[mask0].mean()
        mean_pred1 = pred[mask1].mean()
        mean_true0 = true[mask0].mean()
        mean_true1 = true[mask1].mean()
        pred_diff = torch.abs(mean_pred0 - mean_pred1)
        data_diff = torch.abs(mean_true0 - mean_true1)
        bound = data_diff + epsilon
        penalty = pred_diff - bound
        if penalty > 0:
            sign_val = torch.sign(mean_pred0 - mean_pred1)
            grad_fair[mask0] = sign_val / n0
            grad_fair[mask1] = -sign_val / n1
        else:
            grad_fair.zero_()
        return grad_fair

    elif fairness_measure == "accuracy_parity":
        mask0 = (race == 0)
        mask1 = (race == 1)
        n0 = mask0.sum().item()
        n1 = mask1.sum().item()
        if n0 == 0 or n1 == 0:
            return grad_fair
        grad = torch.zeros_like(pred)
        grad[mask0] = 2 * (pred[mask0] - true[mask0]) / n0
        grad[mask1] = -2 * (pred[mask1] - true[mask1]) / n1
        return grad

    else:
        return grad_fair


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
