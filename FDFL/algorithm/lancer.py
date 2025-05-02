import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cvxpy as cp

class FairDFLDataset(Dataset):
    """Dataset class for Fair DFL"""
    def __init__(self, features, risk, gainF, cost, race, bb_problem):
        self.features = features
        self.risk = risk
        self.gainF = gainF
        self.cost = cost
        self.race = race
        self.bb_problem = bb_problem
        
        self.sols, self.vals = self._get_solutions()
        self._to_tensor()

    def __len__(self):
        return len(self.features)

    def _get_solutions(self):
        sols, vals = self.bb_problem.solve_optimization(self.gainF, self.risk, self.cost)
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
        return (
            self.features[idx],
            self.risk[idx],
            self.gainF[idx],
            self.cost[idx],
            self.race[idx],
            self.sols[idx],
            self.vals
        )

class BaseProblem:
    """Base problem class defining the optimization problem"""
    def __init__(self, alpha=alpha, Q=Q):
        self.alpha = float(alpha)  # Ensure alpha is float
        self.Q = float(Q)  # Ensure Q is float
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
        
        if self.alpha == float('inf'):
            t = cp.Variable()
            objective = cp.Maximize(t)
            constraints = [
                d >= 0,
                cp.sum(cost * d) <= self.Q,
                utils >= t
            ]
        elif self.alpha == 1:
            objective = cp.Maximize(cp.sum(cp.log(utils)))
            constraints = [
                d >= 0,
                cp.sum(cost * d) <= self.Q
            ]
        elif self.alpha == 0:
            objective = cp.Maximize(cp.sum(utils))
            constraints = [
                d >= 0,
                cp.sum(cost * d) <= self.Q
            ]
        else:
            objective = cp.Maximize(cp.sum(cp.power(utils, 1 - self.alpha)) / (1 - self.alpha))
            constraints = [
                d >= 0,
                cp.sum(cost * d) <= self.Q
            ]
        
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
            
            if problem.status != 'optimal':
                print(f"Warning: Problem status is {problem.status}")
                
            optimal_decision = d.value
            optimal_value = self.eval_utility(optimal_decision, gainF, risk)
            
        except Exception as e:
            print(f"Optimization error: {e}")
            optimal_decision = np.zeros(risk.shape)
            optimal_value = 0.0
            
        return optimal_decision, optimal_value

    def eval_utility(self, decision, gainF, risk):
        """Evaluates the utility function"""
        utils = decision * gainF * risk
        if self.alpha == 1:
            return np.sum(np.log(utils + 1e-10))
        elif self.alpha == 0:
            return np.sum(utils)
        elif self.alpha == float('inf'):
            return np.min(utils)
        else:
            return np.sum(np.power(utils, 1 - self.alpha)) / (1 - self.alpha)
    
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

import time
class MLPLancer(nn.Module):
    """MLP-based LANCER Model for estimating decision loss"""
    def __init__(self, input_dim=1, hidden_dim=64, n_layers=2):
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
        # Ensure correct dimensions
        z_pred = z_pred.view(-1, 1)
        z_true = z_true.view(-1, 1)
        diff = torch.square(z_pred - z_true)
        return self.model(diff)

class LancerLearner:
    """Main LANCER learning framework"""
    def __init__(self, bb_problem, c_model, lancer_model, device='cuda', 
                 c_lr=1e-4, lancer_lr=1e-4, weight_decay=1e-5):
        self.bb_problem = bb_problem
        self.c_model = c_model.to(device)
        self.lancer_model = lancer_model.to(device)
        self.device = device
        
        # Use lower learning rates and add weight decay
        self.c_optimizer = optim.AdamW(
            self.c_model.parameters(),
            lr=c_lr,
            weight_decay=weight_decay
        )
        self.lancer_optimizer = optim.AdamW(
            self.lancer_model.parameters(),
            lr=lancer_lr,
            weight_decay=weight_decay
        )
        
        # Add learning rate schedulers
        self.c_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.c_optimizer, mode='min', factor=0.5, patience=2
        )
        self.lancer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.lancer_optimizer, mode='min', factor=0.5, patience=2
        )
        
    def train_step(self, feats, risk, gainF, cost, race, sols, vals):
        # Convert to tensors if they aren't already
        if not isinstance(feats, torch.Tensor):
            feats = torch.FloatTensor(feats)
        if not isinstance(risk, torch.Tensor):
            risk = torch.FloatTensor(risk)
        if not isinstance(gainF, torch.Tensor):
            gainF = torch.FloatTensor(gainF)
        if not isinstance(cost, torch.Tensor):
            cost = torch.FloatTensor(cost)
        if not isinstance(sols, torch.Tensor):
            sols = torch.FloatTensor(sols)
            
        feats = feats.to(self.device)
        risk = risk.to(self.device)
        gainF = gainF.to(self.device)
        cost = cost.to(self.device)
        sols = sols.to(self.device)
        
        # Train LANCER model
        self.lancer_optimizer.zero_grad()
        z_pred = self.c_model(feats)
        lancer_pred = self.lancer_model(z_pred, risk)
        lancer_loss = self.lancer_model.loss_fn(lancer_pred, sols.view(-1, 1))
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.lancer_model.parameters(), max_norm=1.0)
        lancer_loss.backward()
        self.lancer_optimizer.step()
        
        # Train C model using LANCER loss
        self.c_optimizer.zero_grad()
        z_pred = self.c_model(feats)
        c_loss = torch.mean(self.lancer_model(z_pred, risk))
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.c_model.parameters(), max_norm=1.0)
        c_loss.backward()
        self.c_optimizer.step()
        
        return lancer_loss.item(), c_loss.item()
    
    def train(self, train_loader, test_loader, n_epochs=10, early_stop_patience=5):
        best_loss = float('inf')
        patience_counter = 0
        
        # Logs to return
        train_loss_log = []
        test_regret_log = []
        mse_loss_log = []
        elapsed = 0
        
        print("Training model...")
        
        for epoch in range(n_epochs):
            self.c_model.train()
            self.lancer_model.train()
            
            epoch_lancer_loss = 0
            epoch_c_loss = 0
            mse_epoch_loss = 0
            num_batches = 0
            
            tick = time.time()
            
            for batch in train_loader:
                feats, risk, gainF, cost, race, sols, vals = batch
                if self.device.type == "cuda":
                    feats, risk, gainF, cost, sols, vals = (
                        feats.cuda(), risk.cuda(), gainF.cuda(), cost.cuda(), sols.cuda(), vals.cuda()
                    )
                
                # Perform training step
                lancer_loss, c_loss = self.train_step(feats, risk, gainF, cost, race, sols, vals)
                mse_loss = nn.MSELoss()(self.c_model(feats), risk)
                
                epoch_lancer_loss += lancer_loss
                epoch_c_loss += c_loss
                mse_epoch_loss += mse_loss.item()
                num_batches += 1
            
            # Calculate average losses
            avg_lancer_loss = epoch_lancer_loss / num_batches
            avg_c_loss = epoch_c_loss / num_batches
            avg_mse_loss = mse_epoch_loss / num_batches
            
            train_loss_log.append(avg_lancer_loss + avg_c_loss)
            mse_loss_log.append(avg_mse_loss)
            
            # Calculate regret on test set
            self.c_model.eval()
            test_regret = regret(self.c_model, solve_optimization, test_loader, alphas=[alpha], Q=Q)
            test_regret_log.append(test_regret)
            
            # Update learning rate schedulers
            self.lancer_scheduler.step(avg_lancer_loss)
            self.c_scheduler.step(avg_c_loss)
            
            # Early stopping check
            current_loss = avg_lancer_loss + avg_c_loss
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            tock = time.time()
            elapsed += tock - tick
            
            # Print progress
            print(f"Epoch {epoch + 1}/{n_epochs}")
            print(f"LANCER Loss: {avg_lancer_loss:.4f}")
            print(f"C Model Loss: {avg_c_loss:.4f}")
            print(f"MSE Loss: {avg_mse_loss:.4f}")
            print(f"Test Regret: {test_regret:.4f}")
            print(f"Elapsed: {elapsed:.2f}s")
            print(f"Learning rates - LANCER: {self.lancer_optimizer.param_groups[0]['lr']:.6f}, "
                f"C: {self.c_optimizer.param_groups[0]['lr']:.6f}")
        
        # Return logs
        return test_regret_log, train_loss_log, mse_loss_log

    
    def predict(self, feats):
        self.c_model.eval()
        with torch.no_grad():
            feats = torch.FloatTensor(feats).to(self.device)
            return self.c_model(feats).cpu().numpy()

def setup_training(feats_train, risk_train, gainF_train, cost_train, race_train, 
                  alpha=alpha, Q=Q, batch_size=32):
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


if __name__ == "__main__":
    #     # Create train and test dataloaders
    # learner, train_dataloader = setup_training(
    #     feats_train, risk_train, gainF_train, cost_train, race_train,
    #     alpha=alpha, Q=Q, batch_size=32)

    # # Create test dataloader
    # test_dataset = FairDFLDataset(
    #     feats_test, risk_test, gainF_test, cost_test, race_test,
    #     learner.bb_problem
    # )
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=len(test_dataset),
    #     shuffle=False
    # )

    # # Train with regret tracking
    # learner.train(train_dataloader, test_dataloader, n_epochs=10, early_stop_patience=5)
    pass