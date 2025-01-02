import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import cvxpy as cp
import torch
from torch.autograd import Function

import matplotlib.pyplot as plt

class OptimizationDataset(Dataset):
    """
    Dataset class for medical cost minimization with racial bias.
    Includes methods to calculate true objectives and prepare data for optimization.
    """

    def __init__(self, features, costs, risks, g_factors, propensity_score, budget, alpha, power=1):
        """
        Initialize the dataset.
        Args:
            features (np.ndarray): Patient feature matrix (x).
            costs (np.ndarray): Cost vector (c).
            risks (np.ndarray): Risk scores (r).
            g_factors (np.ndarray): Multiplicative factor (g).
            budget (float): Total budget (Q).
            alpha (float): Alpha parameter for fairness.
            p (float): Exponent for utility calculation (default 1).
        """
        self.features = features
        self.costs = costs
        self.risks = risks
        self.g_factors = g_factors
        self.budget = budget
        self.alpha = alpha
        self.power = power
        self.propensity_score = propensity_score
        self.utilities = self.calculate_utilities()

    def calculate_utilities(self):
        """
        Compute utilities for all individuals.
        Returns:
            np.ndarray: Utilities computed as (r * g * d)^p.
        """
        utilities = np.zeros_like(self.risks)

        for i in range(len(self.risks)):

            utilities[i] = (self.risks[i] * self.g_factors[i]* self.propensity_score[i])**self.power
        return utilities

    def __len__(self):
        return len(self.risks)

    def __getitem__(self, idx):
        """
        Get data for an individual.
        Args:
            idx (int): Index.
        Returns:
            Tuple: Features, costs, risks, g_factors, and utility for the individual.
        """
        return (
            torch.FloatTensor(self.features[idx]),
            torch.FloatTensor([self.costs[idx]]),
            torch.FloatTensor([self.risks[idx]]),
            torch.FloatTensor([self.g_factors[idx]]),
            torch.FloatTensor([self.utilities[idx]]),
        )


class OptimizationModel:
    """
    Optimization model for alpha fairness maximization under constraints.
    """

    def __init__(self, alpha, budget, costs):
        """
        Initialize the optimization model.
        Args:
            alpha (float): Alpha parameter for fairness.
            budget (float): Total budget (Q).
            costs (np.ndarray): Cost vector (c).
        """
        self.alpha = alpha
        self.budget = budget
        self.costs = costs

    def objValue(self, u, r):
        """
        A function to calculate objective value
        """
        alpha = self.alpha
        if alpha == 1:
            return np.sum(np.log(np.multiply(r, u)))
        else:
            return np.sum(np.power(np.multiply(r, u), 1 - alpha)) / (1 - alpha)

    def solve(self, risks, g_factors, p=1, constr='decision'):
        """
        Solve the optimization problem.
        Args:
            risks (np.ndarray): Risk scores (r).
            g_factors (np.ndarray): Multiplicative factors (g).
            p (float): Utility exponent (default 1).
        Returns:
            np.ndarray: Optimal decision variables (d).
            float: Objective value.
        """
        n = len(risks)
        d = cp.Variable(n, nonneg=True)
        utilities = cp.multiply((risks * g_factors)**p, d)

        if self.alpha == 1:
            objective = cp.sum(cp.log(utilities))
        else:
            objective = cp.sum(cp.power(utilities, 1 - self.alpha)) / (1 - self.alpha)

        if constr == 'utility':
            constraints = [
                cp.sum(cp.multiply(self.costs, utilities)) <= self.budget,
                d >= 0
            ]

        elif constr == 'decision':
            constraints = [
                cp.sum(cp.multiply(self.costs, d)) <= self.budget,
                d >= 0
            ]


        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve()

        return d.value, problem.value


def regret(predicted_risks, true_risks, g_factors, costs, opt_model, budget, alpha):
    """
    Compute regret for predicted vs. true risks.
    Args:
        predicted_risks (np.ndarray): Predicted risk scores.
        true_risks (np.ndarray): True risk scores.
        g_factors (np.ndarray): Multiplicative factors (g).
        costs (np.ndarray): Cost vector (c).
        opt_model (OptimizationModel): Optimization model instance.
        budget (float): Budget (Q).
        alpha (float): Alpha parameter for fairness.
    Returns:
        float: Regret loss.
    """
    # Optimal decision and objective with true risks
    true_decisions, true_objective = opt_model.solve(true_risks.sqeeze(), g_factors)
    # Optimal decision and objective with predicted risks
    predicted_decisions, predicted_objective = opt_model.solve(predicted_risks.sqeeze(), g_factors)

    # Compute regret
    regret_loss = true_objective - opt_model.objValue(predicted_decisions, true_risks)
    return regret_loss


class RegretLossFunction(Function):
    @staticmethod
    def forward(ctx, predicted_risks, true_risks, g_factors, costs, opt_model, budget, alpha):
        """
        Forward pass to calculate regret loss.
        """
        predicted_risks_np = predicted_risks.detach().cpu().numpy()
        true_risks_np = true_risks.detach().cpu().numpy()
        g_factors_np = g_factors.detach().cpu().numpy()
        costs_np = costs.detach().cpu().numpy()

        regret_loss = regret(predicted_risks_np, true_risks_np, g_factors_np, costs_np, opt_model, budget, alpha)
        ctx.save_for_backward(predicted_risks, true_risks, g_factors, costs)
        ctx.opt_model = opt_model
        ctx.budget = budget
        ctx.alpha = alpha

        return torch.tensor(regret_loss, device=predicted_risks.device)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for regret loss.
        """
        predicted_risks, true_risks, g_factors, costs = ctx.saved_tensors
        grad_predicted_risks = torch.zeros_like(predicted_risks)

        epsilon = 1e-5
        for i in range(predicted_risks.size(0)):
            perturbed = predicted_risks.clone()
            perturbed[i] += epsilon

            # Compute perturbed regret loss
            perturbed_loss = regret(
                perturbed.cpu().numpy(),
                true_risks.cpu().numpy(),
                g_factors.cpu().numpy(),
                costs.cpu().numpy(),
                ctx.opt_model,
                ctx.budget,
                ctx.alpha
            )
            grad_predicted_risks[i] = (perturbed_loss - ctx.saved_tensors[0][i]) / epsilon

        return grad_output.view(-1, 1) * grad_predicted_risks, None, None, None, None, None, None


def visualize_training_curves(mse_loss_log, regret_loss_log, regret_test_log, time_log):
    """
    Visualize training curves.
    Args:
        mse_loss_log (list): MSE loss values.
        regret_loss_log (list): Regret loss during training.
        regret_test_log (list): Regret loss on test set.
        time_log (list): Time taken for training.
    """
    epochs = len(mse_loss_log)

    plt.figure(figsize=(15, 5))

    # MSE Loss
    plt.subplot(1, 3, 1)
    plt.plot(range(epochs), mse_loss_log, label="MSE Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Training MSE Loss")

    # Regret Loss
    plt.subplot(1, 3, 2)
    plt.plot(range(epochs), regret_loss_log, label="Regret Loss (Train)")
    plt.plot(range(epochs), regret_test_log, label="Regret Loss (Test)", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Regret Loss")
    plt.title("Regret Loss")
    plt.legend()

    # Time per Epoch
    plt.subplot(1, 3, 3)
    plt.plot(range(epochs), time_log, label="Time Per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Time (s)")
    plt.title("Training Time")
    plt.legend()

    plt.tight_layout()
    plt.show()
