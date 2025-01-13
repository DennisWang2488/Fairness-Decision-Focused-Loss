import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cvxpy as cp

class BaseProblem:
    """Base problem class defining the optimization problem"""
    def __init__(self, alpha=0.5, Q=1000):
        self.alpha = alpha
        self.Q = Q
        self.num_feats = None
        self.lancer_out_activation = "relu"
    
    def solve_optimization(self, gainF, risk, cost):
        """Solves the optimization problem"""
        # Convert tensors to numpy if needed
        gainF = gainF.detach().cpu().numpy() if isinstance(gainF, torch.Tensor) else gainF
        risk = risk.detach().cpu().numpy() if isinstance(risk, torch.Tensor) else risk
        cost = cost.detach().cpu().numpy() if isinstance(cost, torch.Tensor) else cost
        
        risk = risk.clip(min=0.001)
        gainF, risk, cost = gainF.flatten(), risk.flatten(), cost.flatten()
        d = cp.Variable(risk.shape, nonneg=True)
        
        utils = cp.multiply(cp.multiply(gainF, risk), d)
        
        if self.alpha == 'inf':
            t = cp.Variable()
            objective = cp.Maximize(t)
            constraints = [
                d >= 0,
                cp.sum(cost * d) <= self.Q,
                utils >= t
            ]
        else:
            objective = cp.Maximize(cp.sum(utils**(1-self.alpha))/(1-self.alpha) if self.alpha != 0 
                                  else cp.sum(utils))
            constraints = [
                d >= 0,
                cp.sum(cost * d) <= self.Q
            ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=False)
        
        optimal_decision = d.value
        optimal_value = self.eval_utility(optimal_decision, gainF, risk)
        
        return optimal_decision, optimal_value

    def eval_utility(self, decision, gainF, risk):
        """Evaluates the utility function"""
        utils = decision * gainF * risk
        if self.alpha == 1:
            return np.sum(np.log(utils))
        elif self.alpha == 0:
            return np.sum(utils)
        elif self.alpha == 'inf':
            return np.min(utils)
        else:
            return np.sum(utils**(1-self.alpha)/(1-self.alpha))
    
    def eval(self, z_pred, z_true, **kwargs):
        """Evaluates the decision loss"""
        sols, vals = self.solve_optimization(z_pred, z_true, kwargs.get('cost', None))
        return vals
    
    def get_c_shapes(self):
        """Returns shapes for the C model"""
        return self.num_feats, 1
    
    def get_activations(self):
        """Returns activation functions for the models"""
        return "tanh", "relu"

class MLPCModel(nn.Module):
    """MLP-based C Model for predicting risk scores"""
    def __init__(self, input_dim, output_dim, hidden_dim=64, n_layers=2):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim]*n_layers + [output_dim]
        
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU() if i < len(dims)-2 else nn.Softplus()
            ])
            
        self.model = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        return self.model(x)
    
class MLPLancer(nn.Module):
    """MLP-based LANCER Model for estimating decision loss"""
    def __init__(self, input_dim, hidden_dim=64, n_layers=2):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim]*n_layers + [1]
        
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU() if i < len(dims)-2 else nn.Identity()
            ])
            
        self.model = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, z_pred, z_true):
        diff = torch.square(z_pred - z_true)
        return self.model(diff)

class LancerLearner:
    """Main LANCER learning framework"""
    def __init__(self, bb_problem, c_model, lancer_model, device='cuda'):
        self.bb_problem = bb_problem
        self.c_model = c_model.to(device)
        self.lancer_model = lancer_model.to(device)
        self.device = device
        
        self.c_optimizer = optim.Adam(self.c_model.parameters())
        self.lancer_optimizer = optim.Adam(self.lancer_model.parameters())
        
    def train_step(self, feats, risk, gainF, cost, race, sols, vals):
        # Convert to tensors
        feats = torch.FloatTensor(feats).to(self.device)
        risk = torch.FloatTensor(risk).to(self.device)
        gainF = torch.FloatTensor(gainF).to(self.device)
        cost = torch.FloatTensor(cost).to(self.device)
        sols = torch.FloatTensor(sols).to(self.device)
        
        # Train LANCER model
        self.lancer_optimizer.zero_grad()
        z_pred = self.c_model(feats)
        lancer_pred = self.lancer_model(z_pred, risk)
        lancer_loss = self.lancer_model.loss_fn(lancer_pred, sols)
        lancer_loss.backward()
        self.lancer_optimizer.step()
        
        # Train C model using LANCER loss
        self.c_optimizer.zero_grad()
        z_pred = self.c_model(feats)
        c_loss = torch.mean(self.lancer_model(z_pred, risk))
        c_loss.backward()
        self.c_optimizer.step()
        
        return lancer_loss.item(), c_loss.item()
    
    def train(self, train_loader, n_epochs=10):
        for epoch in range(n_epochs):
            epoch_lancer_loss = 0
            epoch_c_loss = 0
            
            for batch in train_loader:
                feats, risk, gainF, cost, race, sols, vals = batch
                lancer_loss, c_loss = self.train_step(
                    feats, risk, gainF, cost, race, sols, vals
                )
                epoch_lancer_loss += lancer_loss
                epoch_c_loss += c_loss
            
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"LANCER Loss: {epoch_lancer_loss/len(train_loader):.4f}")
            print(f"C Model Loss: {epoch_c_loss/len(train_loader):.4f}")
    
    def predict(self, feats):
        self.c_model.eval()
        with torch.no_grad():
            feats = torch.FloatTensor(feats).to(self.device)
            return self.c_model(feats).cpu().numpy()

# Training pipeline setup
def setup_training(feats_train, risk_train, gainF_train, cost_train, race_train, 
                  alpha=0.5, Q=1000, batch_size=32):
    # Initialize problem
    bb_problem = BaseProblem(alpha=alpha, Q=Q)
    bb_problem.num_feats = feats_train.shape[1]
    
    # Initialize models
    c_model = MLPCModel(
        input_dim=feats_train.shape[1],
        output_dim=1
    )
    
    lancer_model = MLPLancer(
        input_dim=1  # For squared difference between pred and true
    )
    
    # Create dataset
    dataset = FairDFLDataset(
        feats_train, risk_train, gainF_train, cost_train, race_train,
        bb_problem
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Initialize learner
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learner = LancerLearner(bb_problem, c_model, lancer_model, device)
    
    return learner, dataloader

# Example usage:
# learner, dataloader = setup_training(feats_train, risk_train, gainF_train, cost_train, race_train)
# learner.train(dataloader, n_epochs=10)
# predictions = learner.predict(feats_test)