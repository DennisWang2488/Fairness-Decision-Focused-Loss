"""
Next Step
Discuss general algorithm: need to approximate gradient for back propagation. Then present gradient approximation methods.
- Closed-Form Decisions
- Linear Decision Objective
- Quadratic Decision Objective
- Generic Decision Objective

Gradient Free Methods

Experiments

Methods to compare:
- Two-stage: prediction then decision, prediction then fair decision, fair prediction then decision, fair prediction then fair decision
- DFL: DFL version of each of the above two-stage settings


Performance measures to report:
- Prediction accuracy: mean square errors of $r$ and $\hat{r}$
- Decision accuracy: mean square errors of $d(r)$ and $d(\hat{r})$
- Prediction fairness: prediction fairness measure of $\hat{r}$
- Decision fairness: decision fairness measure of $d(\hat(r))$
- Runtime of algorithm
"""
import sys
sys.path.insert(0, 'E:\\User\\Stevens\\Code\\The Paper\\algorithm')

import warnings
warnings.filterwarnings("ignore")

import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler


from myutil import *
from features import *

df = pd.read_csv('data/data.csv')

columns_to_keep = [
    'risk_score_t', 'program_enrolled_t', 'cost_t', 'cost_avoidable_t', 'race', 'dem_female', 'gagne_sum_tm1', 'gagne_sum_t', 
    'risk_score_percentile', 'screening_eligible', 'avoidable_cost_mapped', 'propensity_score', 'g_binary', 
    'g_continuous', 'utility_binary', 'utility_continuous'
]

# for race 0 is white, 1 is black
df_stat = df[columns_to_keep]
df_feature = df[[col for col in df.columns if col not in columns_to_keep]]

# Replace all values less than 0.1 with 0.1
df['risk_score_t'] = df['risk_score_t'].apply(lambda x: 0.1 if x < 0.1 else x)
df['g_continuous'] = df['g_continuous'].apply(lambda x: 0.1 if x < 0.1 else x)

# df = df.sample(n=10000, random_state=1)

# Define input variables for DFL
feats = df[get_all_features(df)].values
risk = df['risk_score_t'].values
gainF = df['g_continuous'].values
pScore = df['propensity_score'].values
cost = np.ones(risk.shape)

alpha = 0.5
Q = 100

utility = risk * gainF * pScore

scaler = StandardScaler()
feats = scaler.fit_transform(feats)

# Train the Prediction Model
class RiskDataset(Dataset):
    def __init__(self, features, risks):
        self.features = torch.FloatTensor(features)
        self.risks = torch.FloatTensor(risks).reshape(-1, 1)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.risks[idx]
    
class RiskPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1),

            nn.Softplus()
        )
    
    def forward(self, x):
        return self.model(x)

# Training function
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



# model = train_model(feats, risk)
# torch.save(model.state_dict(), 'risk_predictor_model.pth')

# Load the model from local
model = RiskPredictor(feats.shape[1])
model.load_state_dict(torch.load('risk_predictor_model.pth'))
model.eval()

pred_risk = model(torch.FloatTensor(feats)).detach().numpy().flatten()

# Fair Regulated Prediction Stage

# TODO


# Optimization Stage

def AlphaFairness(util,alpha):
    if alpha == 1:
        return np.sum(np.log(util))
    elif alpha == 0:
        return np.sum(util)
    elif alpha == 'inf':
        return np.min(util)
    else:
        return np.sum(util**(1-alpha)/(1-alpha))
    
def solve_optimization(gainF, risk, cost, alpha, Q):
    # Flatten input arrays
    gainF, risk, cost = gainF.flatten(), risk.flatten(), cost.flatten()
    d = cp.Variable(risk.shape, nonneg=True)
    
    utils = cp.multiply(cp.multiply(gainF, risk), d)
    
    if alpha == 'inf':
        # Maximin formulation
        t = cp.Variable()  # auxiliary variable for minimum utility
        objective = cp.Maximize(t)
        constraints = [
            d >= 0,
            # d <= 1,
            cp.sum(cost * d) <= Q,
            utils >= t  # t is the minimum utility
        ]
    elif alpha == 1:
        # Nash welfare (alpha = 1)
        objective = cp.Maximize(cp.sum(cp.log(utils)))
        constraints = [
            d >= 0,
            # d <= 1,
            cp.sum(cost * d) <= Q
        ]
    elif alpha == 0:
        # Utilitarian welfare (alpha = 0)
        objective = cp.Maximize(cp.sum(utils))
        constraints = [
            d >= 0,
            # d <= 1,
            cp.sum(cost * d) <= Q
        ]
    else:
        # General alpha-fairness
        objective = cp.Maximize(cp.sum(utils**(1-alpha))/(1-alpha) if alpha != 0 
                              else cp.sum(utils))
        constraints = [
            d >= 0,
            # d <= 1,
            cp.sum(cost * d) <= Q
        ]
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=True, warm_start=True, mosek_params={'MSK_IPAR_LOG': 1})
    
    if problem.status != 'optimal':
        print(f"Warning: Problem status is {problem.status}")
    
    optimal_decision = d.value
    optimal_value = AlphaFairness(optimal_decision * gainF * risk, alpha)
    
    return optimal_decision, optimal_value

def twoStagePTO(model, feats, gainF, risk, cost, Q, alphas=[0.5]):
    """
    Perform a two-stage optimization analysis with predictions and calculate normalized regrets.

    Args:
        model (nn.Module): A regression neural network for risk prediction.
        feats (np.ndarray): Feature array for predictions.
        gainF (np.ndarray): Gain factors.
        risk (np.ndarray): True risk values.
        cost (np.ndarray): Cost constraints.
        Q (float): Budget constraint.
        alphas (list): List of alpha values for fairness.

    Returns:
        pd.DataFrame: A table of prediction risk means, true risk mean, objectives, and normalized regrets.
    """
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    # Feature scaling
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    # Predict risks
    model.eval()
    pred_risk = model(torch.FloatTensor(feats_scaled)).detach().numpy().flatten()

    # Initialize result storage
    results = []

    # Iterate over alphas
    for alpha in alphas:
        # Solve optimization problems
        true_sol, _ = solve_optimization(gainF, risk, cost, alpha, Q)
        pred_sol, _ = solve_optimization(gainF, pred_risk, cost, alpha, Q)

        # Calculate true and predicted utilities
        true_utility = risk * gainF * true_sol
        pred_utility = pred_risk * gainF * pred_sol

        # Calculate objectives
        true_obj = AlphaFairness(true_utility, alpha)
        pred_obj = AlphaFairness(pred_utility, alpha)

        # Calculate regret and normalized regret
        regret = true_obj - pred_obj
        normalized_regret = regret / (abs(true_obj) + 1e-7)

        # Collect results for this alpha
        results.append({
            'Alpha': alpha,
            'Predicted Risk Mean': pred_risk.mean(),
            'True Risk Mean': risk.mean(),
            'True Objective': true_obj,
            'Predicted Objective': pred_obj,
            'Normalized Regret': normalized_regret
        })

    # Create a DataFrame for results
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df
