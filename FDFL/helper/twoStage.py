import warnings
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

try:
    from .myutil import *  # noqa: F401,F403
    from .features import *  # noqa: F401,F403
except ImportError:
    from myutil import *  # noqa: F401,F403
    from features import *  # noqa: F401,F403

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load and preprocess data
df = pd.read_csv('data/data.csv')

columns_to_keep = [
    'risk_score_t', 'program_enrolled_t', 'cost_t', 'cost_avoidable_t', 'race', 'dem_female',
    'gagne_sum_tm1', 'gagne_sum_t', 'risk_score_percentile', 'screening_eligible',
    'avoidable_cost_mapped', 'propensity_score', 'g_binary', 'g_continuous',
    'utility_binary', 'utility_continuous'
]

# Filter relevant columns
df_stat = df[columns_to_keep]
df_feature = df[[col for col in df.columns if col not in columns_to_keep]]

# Replace values less than 0.1 with 0.1 to avoid issues in calculations
df['risk_score_t'] = df['risk_score_t'].apply(lambda x: 0.1 if x < 0.1 else x)
df['g_continuous'] = df['g_continuous'].apply(lambda x: 0.1 if x < 0.1 else x)

# Define input variables for DFL
feats = df[get_all_features(df)].values
risk = df['risk_score_t'].values
gainF = df['g_continuous'].values
pScore = df['propensity_score'].values
cost = np.ones_like(risk)  # Assuming uniform cost; adjust as needed

# Define alpha and budget Q
alpha = 0.5
Q = 100

# Compute utility (if needed elsewhere)
utility = risk * gainF * pScore

# Feature scaling
scaler = StandardScaler()
feats_scaled = scaler.fit_transform(feats)

# Define the RiskDataset class
class RiskDataset(Dataset):
    def __init__(self, features, risks):
        self.features = torch.FloatTensor(features)
        self.risks = torch.FloatTensor(risks).reshape(-1, 1)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.risks[idx]

# Define the RiskPredictor model
class RiskPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Softplus()  # Ensures non-negative predictions
        )
    
    def forward(self, x):
        return self.model(x)

# Training function (if training is needed)
def train_model(features, risks, epochs=10, batch_size=32):
    dataset = RiskDataset(features, risks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = RiskPredictor(features.shape[1])
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        for batch_features, batch_risks in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_risks)
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

def AlphaFairness(util, alpha):
    """
    Compute the alpha-fairness objective.

    Parameters:
    - util (np.ndarray): Array of utilities (u_i), shape (n,)
    - alpha (float or 'inf'): Fairness parameter

    Returns:
    - obj (float): Alpha-fairness objective value
    """
    if alpha == 1:
        return np.sum(np.log(util + 1e-9))  # Add epsilon to avoid log(0)
    elif alpha == 0:
        return np.sum(util)
    elif alpha == 'inf':
        return np.min(util)
    else:
        return np.sum((util ** (1 - alpha)) / (1 - alpha))

def solve_optimization(gainF, risk, cost, alpha, Q):
    """
    Solve the alpha-fairness optimization problem using CVXPY.

    Parameters:
    - gainF (np.ndarray): Gain factors, shape (n,)
    - risk (np.ndarray): Risks, shape (n,)
    - cost (np.ndarray): Costs, shape (n,)
    - alpha (float or 'inf'): Fairness parameter
    - Q (float): Budget constraint

    Returns:
    - optimal_decision (np.ndarray): Optimal decision variables, shape (n,)
    - optimal_value (float): Alpha-fairness objective value
    """
    gainF, risk, cost = gainF.flatten(), risk.flatten(), cost.flatten()
    n = len(risk)
    d = cp.Variable(n, nonneg=True)
    
    utils = cp.multiply(gainF * risk, d)  # Element-wise multiplication
    
    constraints = [
        cp.sum(cp.multiply(cost, d)) <= Q
    ]
    
    if alpha == 'inf':
        # Maximin formulation
        t = cp.Variable()
        objective = cp.Maximize(t)
        constraints += [utils >= t]
    elif alpha == 1:
        # Nash welfare (log utility)
        objective = cp.Maximize(cp.sum(cp.log(utils + 1e-9)))  # Add epsilon to avoid log(0)
    elif alpha == 0:
        # Utilitarian welfare
        objective = cp.Maximize(cp.sum(utils))
    else:
        # General alpha-fairness
        objective = cp.Maximize(cp.sum(cp.power(utils, 1 - alpha)) / (1 - alpha))
    
    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.MOSEK, verbose=False, warm_start=True)
    except cp.error.SolverError:
        # Fallback to a different solver if MOSEK is not available
        problem.solve(solver=cp.SCS, verbose=False)
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Warning: Problem status is {problem.status}")
    
    optimal_decision = d.value
    if optimal_decision is None:
        optimal_decision = np.zeros(n)
    
    # Compute utilities for objective
    utilities = gainF * risk * optimal_decision
    optimal_value = AlphaFairness(utilities, alpha)
    
    return optimal_decision, optimal_value

def compute_d_star_closed_form(c, r, g, Q, alpha):
    """
    Compute the optimal decision variables d* using the closed-form solution.

    Parameters:
    - c (np.ndarray): Array of costs (c_i), shape (n,)
    - r (np.ndarray): Array of risks (r_i), shape (n,)
    - g (np.ndarray): Array of gain factors (g_i), shape (n,)
    - Q (float): Total budget constraint
    - alpha (float or 'inf'): Fairness parameter

    Returns:
    - d_star_closed (np.ndarray): Optimal decision variables (d_i*), shape (n,)
    """
    if np.any(c <= 0) or np.any(r <= 0) or np.any(g <= 0):
        raise ValueError("All cost, risk, and gain values must be positive.")
    if Q <= 0:
        raise ValueError("Total budget Q must be positive.")
    
    n = len(r)
    
    if alpha == 0:
        # Utilitarian Welfare: Allocate all budget to the most efficient decision
        efficiency = (r * g) / c
        max_idx = np.argmax(efficiency)
        d_star_closed = np.zeros_like(c)
        d_star_closed[max_idx] = Q / c[max_idx]
    
    elif alpha == 1:
        # Nash Welfare: Allocate uniformly inversely proportional to costs
        d_star_closed = Q / (n * c)
    
    elif alpha == 'inf':
        # Maximin Welfare: Allocate to equalize utilities as much as possible
        denominator = np.sum(c / (r * g))
        if denominator == 0:
            raise ValueError("Denominator in closed-form solution for alpha=inf is zero.")
        d_star_closed = (Q / denominator) / (r * g)
    
    else:
        # General alpha-fairness solution
        numerator = (c ** (-1 / alpha)) * ((r * g) ** (-1 + 1 / alpha))
        denominator = np.sum(numerator)
        if denominator == 0:
            raise ValueError("Denominator in closed-form solution is zero.")
        d_star_closed = (numerator * Q) / denominator
    
    return d_star_closed

def analyze_race_stats(alpha, race, true_sol, pred_sol, true_utility, pred_utility, race_labels={0: 'White', 1: 'Black'}):
    """
    Analyze and compute statistics based on race.

    Parameters:
    - alpha (float): Fairness parameter
    - race (np.ndarray): Array indicating race (e.g., 0 for White, 1 for Black)
    - true_sol (np.ndarray): True optimal solutions
    - pred_sol (np.ndarray): Predicted optimal solutions
    - true_utility (np.ndarray): True utilities
    - pred_utility (np.ndarray): Predicted utilities
    - race_labels (dict): Mapping from race codes to labels

    Returns:
    - stats (list of dict): List containing statistics per race
    """
    stats = []
    for r_code, r_label in race_labels.items():
        mask = race == r_code
        if np.sum(mask) == 0:
            continue  # Skip if no instances of this race
        race_stats = {
            'Alpha': alpha,
            'Race': r_label,
            'True Solution Mean': np.mean(true_sol[mask]),
            'True Solution Std': np.std(true_sol[mask]),
            'Predicted Solution Mean': np.mean(pred_sol[mask]),
            'Predicted Solution Std': np.std(pred_sol[mask]),
            'True Utility Mean': np.mean(true_utility[mask]),
            'True Utility Std': np.std(true_utility[mask]),
            'Predicted Utility Mean': np.mean(pred_utility[mask]),
            'Predicted Utility Std': np.std(pred_utility[mask])
        }
        stats.append(race_stats)
    return stats

def twoStagePTO_with_bias_analysis(
    model, fair_model, feats, gainF, risk, cost, race, 
    Q=1000, alphas=[0.5]
):
    """
    Perform two-stage PTO with bias analysis by computing both solver-based and closed-form solutions.

    Parameters:
    - model: PyTorch model for risk prediction
    - fair_model: PyTorch model for fair risk prediction
    - feats (np.ndarray): Feature matrix, shape (n_samples, n_features)
    - gainF (np.ndarray): Gain factors, shape (n_samples,)
    - risk (np.ndarray): True risks, shape (n_samples,)
    - cost (np.ndarray): Costs, shape (n_samples,)
    - race (np.ndarray): Race indicators (e.g., 0 for White, 1 for Black), shape (n_samples,)
    - Q (float): Total budget constraint
    - alphas (list of float or str): List of alpha fairness parameters

    Returns:
    - results_df (pd.DataFrame): Aggregated results comparing solver and closed-form
    - bias_analysis_df (pd.DataFrame): Bias analysis per race for both methods
    - solutions (dict): Contains optimal solutions from solver and closed-form
    """
    # Feature scaling
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    # Predict risks using the trained models
    model.eval()
    fair_model.eval()
    with torch.no_grad():
        pred_risk = model(torch.FloatTensor(feats_scaled)).numpy().flatten()
        fair_pred_risk = fair_model(torch.FloatTensor(feats_scaled)).numpy().flatten()

    # Initialize result storage
    results = []
    bias_analysis = []
    solutions = {
        'solver': {},
        'closed_form': {}
    }

    # Iterate over alphas
    for alpha in alphas:
        # Solver-based solutions
        true_sol_solver, true_obj_solver = solve_optimization(gainF, risk, cost, alpha, Q)
        pred_sol_solver, pred_obj_solver = solve_optimization(gainF, pred_risk, cost, alpha, Q)
        fair_pred_sol_solver, fair_pred_obj_solver = solve_optimization(gainF, fair_pred_risk, cost, alpha, Q)

        # Closed-form solutions
        try:
            true_sol_cf = compute_d_star_closed_form(cost, risk, gainF, Q, alpha)
            pred_sol_cf = compute_d_star_closed_form(cost, pred_risk, gainF, Q, alpha)
            fair_pred_sol_cf = compute_d_star_closed_form(cost, fair_pred_risk, gainF, Q, alpha)
        except ValueError as e:
            print(f"Closed-form computation skipped for alpha={alpha}: {e}")
            true_sol_cf = pred_sol_cf = fair_pred_sol_cf = np.zeros_like(risk)

        # Store solutions
        solutions['solver'][alpha] = {
            'true_sol': true_sol_solver,
            'pred_sol': pred_sol_solver,
            'fair_pred_sol': fair_pred_sol_solver
        }
        solutions['closed_form'][alpha] = {
            'true_sol': true_sol_cf,
            'pred_sol': pred_sol_cf,
            'fair_pred_sol': fair_pred_sol_cf
        }

        # Compute utilities
        utilities_solver = gainF * risk * true_sol_solver
        pred_util_solver = gainF * risk * pred_sol_solver
        fair_pred_util_solver = gainF * risk * fair_pred_sol_solver

        utilities_cf = gainF * risk * true_sol_cf
        pred_util_cf = gainF * risk * pred_sol_cf
        fair_pred_util_cf = gainF * risk * fair_pred_sol_cf

        # Compute objectives using helper function
        true_obj_cf = AlphaFairness(utilities_cf, alpha)
        pred_obj_cf = AlphaFairness(pred_util_cf, alpha)
        fair_pred_obj_cf = AlphaFairness(fair_pred_util_cf, alpha)

        # Compute regrets and normalized regrets
        # Solver-based
        regret_solver = true_obj_solver - AlphaFairness(gainF * risk * pred_sol_solver, alpha)
        normalized_regret_solver = regret_solver / (abs(true_obj_solver) + 1e-7)

        fair_regret_solver = true_obj_solver - AlphaFairness(gainF * risk * fair_pred_sol_solver, alpha)
        fair_normalized_regret_solver = fair_regret_solver / (abs(true_obj_solver) + 1e-7)

        # Closed-form
        regret_cf = true_obj_cf - pred_obj_cf
        normalized_regret_cf = regret_cf / (abs(true_obj_cf) + 1e-7)

        fair_regret_cf = true_obj_cf - fair_pred_obj_cf
        fair_normalized_regret_cf = fair_regret_cf / (abs(true_obj_cf) + 1e-7)

        # Append solver results
        results.append({
            'Alpha': alpha,
            'Method': 'Solver',
            'Predicted Risk Mean': pred_risk.mean(),
            'True Risk Mean': risk.mean(),
            'True Objective': true_obj_solver,
            'Predicted Objective': AlphaFairness(gainF * risk * pred_sol_solver, alpha),
            'Regret': regret_solver,
            'Normalized Regret': normalized_regret_solver
        })

        # Append closed-form results
        results.append({
            'Alpha': alpha,
            'Method': 'Closed-Form',
            'Predicted Risk Mean': pred_risk.mean(),
            'True Risk Mean': risk.mean(),
            'True Objective': true_obj_cf,
            'Predicted Objective': pred_obj_cf,
            'Regret': regret_cf,
            'Normalized Regret': normalized_regret_cf
        })

        # Append fair solver results
        results.append({
            'Alpha': alpha,
            'Method': 'Solver (Fair)',
            'Predicted Risk Mean': fair_pred_risk.mean(),
            'True Risk Mean': risk.mean(),
            'True Objective': true_obj_solver,
            'Predicted Objective': AlphaFairness(gainF * risk * fair_pred_sol_solver, alpha),
            'Regret': fair_regret_solver,
            'Normalized Regret': fair_normalized_regret_solver
        })

        # Append fair closed-form results
        results.append({
            'Alpha': alpha,
            'Method': 'Closed-Form (Fair)',
            'Predicted Risk Mean': fair_pred_risk.mean(),
            'True Risk Mean': risk.mean(),
            'True Objective': true_obj_cf,
            'Predicted Objective': fair_pred_obj_cf,
            'Regret': fair_regret_cf,
            'Normalized Regret': fair_normalized_regret_cf
        })

        # Bias analysis for solver
        bias_solver = analyze_race_stats(
            alpha, race, true_sol_solver, pred_sol_solver, 
            utilities_solver, pred_util_solver
        )
        bias_analysis.extend([{'Method': 'Solver'} | stat for stat in bias_solver])

        # Bias analysis for closed-form
        bias_cf = analyze_race_stats(
            alpha, race, true_sol_cf, pred_sol_cf, 
            utilities_cf, pred_util_cf
        )
        bias_analysis.extend([{'Method': 'Closed-Form'} | stat for stat in bias_cf])

        # Bias analysis for fair solver
        bias_fair_solver = analyze_race_stats(
            alpha, race, true_sol_solver, fair_pred_sol_solver, 
            utilities_solver, fair_pred_util_solver
        )
        bias_analysis.extend([{'Method': 'Solver (Fair)'} | stat for stat in bias_fair_solver])

        # Bias analysis for fair closed-form
        bias_fair_cf = analyze_race_stats(
            alpha, race, true_sol_cf, fair_pred_sol_cf, 
            utilities_cf, fair_pred_util_cf
        )
        bias_analysis.extend([{'Method': 'Closed-Form (Fair)'} | stat for stat in bias_fair_cf])

    # Create DataFrames for results and bias analysis
    results_df = pd.DataFrame(results)
    bias_analysis_df = pd.DataFrame(bias_analysis)
    bias_analysis_df['Race'] = bias_analysis_df['Race'].astype(str)

    return results_df, bias_analysis_df, solutions

# Add 'race' to the dataset
class FairRiskDataset(Dataset):
    def __init__(self, features, races, risks):
        self.features = torch.FloatTensor(features)
        self.races = torch.LongTensor(races)
        self.risks = torch.FloatTensor(risks).reshape(-1, 1)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.races[idx], self.risks[idx]

class FairRiskPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Softplus()
        )
        
    def forward(self, x):
        return self.model(x)
    
def train_fair_model(features, races, risks, epochs=10, batch_size=32, lambda_fairness=1.0):
    """
    Train a fair regression model with a fairness regularizer.
    
    Args:
        features (np.ndarray): Feature array.
        races (np.ndarray): Array indicating race (0: white, 1: black).
        risks (np.ndarray): True risk values.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lambda_fairness (float): Weight for the fairness regularizer.
        
    Returns:
        nn.Module: Trained fair regression model.
    """
    dataset = FairRiskDataset(features, races, risks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = FairRiskPredictor(features.shape[1])
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_features, batch_races, batch_risks in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            mse_loss = criterion(predictions, batch_risks)
            
            # Compute fairness loss
            group0 = predictions[batch_races == 0]
            group1 = predictions[batch_races == 1]
            if len(group0) > 0 and len(group1) > 0:
                fairness_loss = torch.abs(group0.mean() - group1.mean())
            else:
                fairness_loss = torch.tensor(0.0)
            
            # Total loss
            total_loss = mse_loss + lambda_fairness * fairness_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    return model

