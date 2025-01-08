import cvxpy as cp
import numpy as np
import warnings
import sys
from IPython.core.interactiveshell import InteractiveShell
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, 'E:\\User\\Stevens\\Code\\The Paper\\algorithm')
from torch.utils.data import Dataset, DataLoader


import warnings
warnings.filterwarnings("ignore")

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

# if needed subset a sample of 5000 rows of df
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

from sklearn.preprocessing import StandardScaler

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
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
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


# Train the model
# model = train_model(feats, risk)

# Save the model to local
# torch.save(model.state_dict(), 'risk_predictor_model.pth')

# Load the model from local
model = RiskPredictor(feats.shape[1])
model.load_state_dict(torch.load('risk_predictor_model.pth'))
model.eval()

pred_risk = model(torch.FloatTensor(feats)).detach().numpy().flatten()

pred_risk.mean(), risk.mean()

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
    problem.solve(solver=cp.MOSEK, verbose=True)
    
    if problem.status != 'optimal':
        print(f"Warning: Problem status is {problem.status}")
    
    optimal_decision = d.value
    optimal_value = AlphaFairness(optimal_decision * gainF * risk, alpha)
    
    return optimal_decision, optimal_value


data_sample = df.sample(n=5000, random_state=42)
feats_sample = data_sample[get_all_features(data_sample)].values
risk_sample = data_sample['risk_score_t'].values
gainF_sample = data_sample['g_continuous'].values
decision_sample = data_sample['propensity_score'].values
cost_sample = np.ones(risk_sample.shape)


predicted_risk = model(torch.FloatTensor(scaler.transform(feats_sample))).detach().numpy()

predicted_risk.mean()
# predicted_risk[predicted_risk< 0.01] = 0.01

def analyze_alpha_fairness(model, feats, gainF, risk, cost, Q):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Solution Distribution Across Different α Values')

    alphas = ['inf']
    regrets = []

    model.eval()
    predicted_risk = model(torch.FloatTensor(feats)).detach().numpy()

    opt_sol, opt_val = solve_optimization(gainF, risk, cost, alpha='inf', Q=Q)
    
    for idx, alpha in enumerate(alphas):
        row = idx // 2
        col = idx % 2
        
        pred_sol, _ = solve_optimization(gainF, predicted_risk, cost, alpha, Q)
        
        # Calculate objectives
        if alpha != 'inf':
            if alpha == 1:
                pred_obj = np.sum(np.log(risk * gainF * pred_sol))
                true_obj = np.sum(np.log(risk * gainF * opt_sol))
            elif alpha == 0:
                pred_obj = np.sum(risk * gainF * pred_sol)
                true_obj = np.sum(risk * gainF * opt_sol)
            else:
                pred_obj = np.sum((risk * gainF * pred_sol)**(1-alpha)/(1-alpha))
                true_obj = np.sum((opt_sol * gainF * risk)**(1-alpha)/(1-alpha))
        else:
            pred_obj = np.min(risk * gainF * pred_sol)
            true_obj = np.min(risk * gainF * opt_sol)
        
        regret = true_obj - pred_obj
        regrets.append((alpha, regret, true_obj, pred_obj))
        
        axes[row, col].hist(pred_sol, bins=50, edgecolor='k')
        axes[row, col].set_xlabel('Predicted Solution')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].set_title(f'alpha = {alpha}\nRegret = {regret:.4f}')
        
    plt.tight_layout()
    
    for alpha, regret, true_obj, pred_obj in regrets:
        print(f'Alpha: {alpha}, Regret: {regret:.4f}, True Objective: {true_obj:.4f}, Predicted Objective: {pred_obj:.4f}')
    
    return fig, regrets

fig, regrets = analyze_alpha_fairness(model, feats, gainF, risk, cost, Q)
plt.show()