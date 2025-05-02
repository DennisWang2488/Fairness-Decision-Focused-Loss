



import cvxpy as cp
import numpy as np
import warnings
import sys
from IPython.core.interactiveshell import InteractiveShell
from sklearn.preprocessing import StandardScaler
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, 'E:\\User\\Stevens\\Code\\The Paper\\algorithm')

from myutil import *
from features import get_all_features

# Suppress warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Import Data
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
#df['risk_score_t'] = df['risk_score_t'].apply(lambda x: 0.1 if x < 0.1 else x)
df['g_continuous'] = df['g_continuous'].apply(lambda x: 0.1 if x < 0.1 else x)

# subset a sample of 5000 rows of df
# df = df.sample(n=10000, random_state=1)

df.shape
# Define input variables for DFL
feats = df[get_all_features(df)].values
risk = df['risk_score_t'].values
gainF = df['g_continuous'].values
decision = df['propensity_score'].values
cost = np.ones(risk.shape)
race = df['race'].values
alpha = 0.5
Q = 1000

# transform the features
scaler = StandardScaler()
feats = scaler.fit_transform(feats)

from sklearn.model_selection import train_test_split

# Perform train-test split
feats_train, feats_test, gainF_train, gainF_test, risk_train, risk_test, cost_train, cost_test, race_train, race_test = train_test_split(
    feats, gainF, risk, cost, df['race'].values, test_size=0.6, random_state=42
)

print(f"Train size: {feats_train.shape[0]}")
print(f"Test size: {feats_test.shape[0]}")
# Processing Data
## Define the optimization and prediction model
def AlphaFairness(util,alpha):
    if alpha == 1:
        return np.sum(np.log(util))
    elif alpha == 0:
        return np.sum(util)
    elif alpha == 'inf':
        return np.min(util)
    else:
        return np.sum(util**(1-alpha)/(1-alpha))


def solve_optimization(gainF, risk, cost, alpha=alpha, Q=Q):
    # Flatten input arrays

    # if any of the inputs are tensor, convert to numpy array
    gainF = gainF.detach().cpu().numpy() if isinstance(gainF, torch.Tensor) else gainF
    risk = risk.detach().cpu().numpy() if isinstance(risk, torch.Tensor) else risk
    cost = cost.detach().cpu().numpy() if isinstance(cost, torch.Tensor) else cost


    risk = risk.clip(min=0.001)
    gainF, risk, cost = gainF.flatten(), risk.flatten(), cost.flatten()
    d = cp.Variable(risk.shape, nonneg=True)

    # raise error if dimensions do not match
    if gainF.shape != risk.shape or risk.shape != cost.shape:
        raise ValueError("Dimensions of gainF, risk, and cost do not match")
    
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
    problem.solve(solver=cp.MOSEK, verbose=False, warm_start=True, mosek_params={'MSK_IPAR_LOG': 1})
    
    if problem.status != 'optimal':
        print(f"Warning: Problem status is {problem.status}")
    
    optimal_decision = d.value
    optimal_value = AlphaFairness(optimal_decision * gainF * risk, alpha)
    
    return optimal_decision, optimal_value
# Define the prediction model
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
        
def train_fair_model(features, races, risks, epochs=20, batch_size=32, lambda_fairness=0):
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
# Dataset
class FairDFLDataset(Dataset):
    def __init__(self, features, risk, gainF, cost, race, alpha=alpha, Q=Q):
        self.features = features
        self.risk = risk
        self.gainF = gainF
        self.cost = cost
        self.race = race
        self.alpha = alpha
        self.Q = Q

        self.sols, self.vals = self._get_solutions()
        self._to_tensor()

    def __len__(self):
        return len(self.features)

    def _get_solutions(self):
        sols, vals = solve_optimization(self.gainF, self.risk, self.cost, self.alpha, self.Q)
        return sols, vals

    def _to_tensor(self):
        self.features = torch.FloatTensor(self.features)
        self.risk = torch.FloatTensor(self.risk)
        self.gainF = torch.FloatTensor(self.gainF)
        self.cost = torch.FloatTensor(self.cost)
        self.race = torch.LongTensor(self.race)
        self.sols = torch.FloatTensor(self.sols)
        self.vals = torch.FloatTensor([self.vals])

    def __getitem__(self, idx):
        return self.features[idx], self.risk[idx], self.gainF[idx], self.cost[idx], self.race[idx], self.sols[idx], self.vals



# test the dataset and dataloader
dataset_train = FairDFLDataset(feats_train, risk_train, gainF_train, cost_train, race_train)
dataset_test = FairDFLDataset(feats_test, risk_test, gainF_test, cost_test, race_test)
print('The current alpha and Q values are:', alpha, Q)
# Load the dataset into a DataLoader
loader_train = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)
def regret(predModel, optModel, dataloader, alphas=[0.5], Q=1000):
    """
    A function to evaluate model performance with normalized true regret.

    Args:
        predModel (nn.Module): Trained prediction model.
        optModel (nn.Module): Trained optimization model.
        dataloader (DataLoader): DataLoader for the dataset.
        alphas (list): List of alpha values for fairness.
        Q (int): Budget constraint.

    Returns:
        float: Normalized regret.
    """
    predModel.eval()

    features, risk, gainF, cost, race, true_sols, true_vals = next(iter(dataloader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, risk, gainF, cost, race, true_sols, true_vals = (
        features.to(device),
        risk.to(device),
        gainF.to(device),
        cost.to(device),
        race.to(device),
        true_sols.to(device),
        true_vals.to(device),
    )

    # Predict risk
    with torch.no_grad():
        pred_risk = predModel(features).clamp(min=0.001)  # Ensure no zero values

    risk = risk.clamp(min=0.001)

    # Convert tensors to numpy arrays
    pred_risk = pred_risk.cpu().numpy()
    risk = risk.cpu().numpy()
    gainF = gainF.cpu().numpy()
    cost = cost.cpu().numpy()

    regrets = []
    for alpha in alphas:
        # Calculate true utility
        true_sol, _ = solve_optimization(gainF, risk, cost, alpha, Q)
        true_utility = gainF * risk * true_sol
        true_val = AlphaFairness(true_utility, alpha)

        # Calculate predicted utility
        pred_sol, _ = solve_optimization(gainF, pred_risk, cost, alpha, Q)
        pred_utility_true_risk = gainF * risk * pred_sol
        pred_val = AlphaFairness(pred_utility_true_risk, alpha)

        # Calculate regret
        regret = (true_val - pred_val) / (true_val + 1e-6)
        regrets.append(regret)

    predModel.train()

    return np.mean(regrets)
# test the regret calculation
# regret(fair_model, model, loader_test, alphas=[0.5], Q=1000)
# DFL

