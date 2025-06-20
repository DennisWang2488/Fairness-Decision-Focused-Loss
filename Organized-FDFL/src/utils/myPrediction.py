# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

def generate_random_features(n_samples, n_features, n_time_periods, n_groups, budget, seed=42):
    """
    Generate random features, costs, and group assignments for the allocation problem.
    
    Args:
        n_samples (int): Number of individuals.
        n_features (int): Number of features per individual.
        n_time_periods (int): Number of time periods.
        n_groups (int): Number of groups.
        budget (float): Total budget for allocation.
        seed (int): Random seed for reproducibility.
        
    Returns:
        tuple: (features, costs, group_labels, budget)
    """
    np.random.seed(seed)
    # Generate random features
    features = np.random.uniform(0.1, 1.0, (n_samples, n_features, n_time_periods))
    # Generate random costs
    costs = np.random.uniform(0.1, 0.5, (n_samples, n_time_periods))
    # Assign individuals to groups
    group_labels = np.random.randint(0, n_groups, n_samples)
    return features, costs, group_labels, budget

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class NeuralNets(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(NeuralNets, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def customPredictionModel(data, model_class, input_dim, output_dim=1, hidden_layers=None, activations=None, **kwargs):
    """
    Create and run a customizable prediction model.
    
    Args:
        data (torch.Tensor): Input data.
        model_class (nn.Module): Model class to instantiate.
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        hidden_layers (list, optional): List of hidden layer sizes.
        activations (list, optional): List of activation functions (nn.Module).
        **kwargs: Additional arguments for the model class.
        
    Returns:
        torch.Tensor: Model predictions.
    """
    if hidden_layers is not None and activations is not None:
        # Build a sequential model dynamically
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(activations):
                layers.append(activations[i])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        model = nn.Sequential(*layers)
    else:
        # Use the provided model class
        model = model_class(input_dim, output_dim=output_dim, **kwargs)
    preds = model(data)
    return preds

