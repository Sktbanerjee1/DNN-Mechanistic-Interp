import torch
import torch.nn as nn

class ModuloNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1, num_hidden_layers=3, dropout_rate=0.1):
        """
        A dynamic MLP that supports variable depth and Bayesian approximation via Dropout.
        """
        super(ModuloNet, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropout_rate = dropout_rate
        
        # Input Layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden Layers
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, return_activations=False):
        """
        Forward pass. 
        If return_activations is True, returns a list of tensors representing 
        the manifold geometry at each layer (The 'Chain of Thought').
        """
        activations = []
        
        out = x
        
        # Pass through hidden layers
        for layer in self.layers:
            out = layer(out)
            out = self.activation(out)
            # Apply dropout even during inference for Bayesian Analysis (MC Dropout)
            out = self.dropout(out) 
            if return_activations:
                activations.append(out)
        
        # Final prediction
        prediction = self.output_layer(out)
        
        if return_activations:
            return prediction, activations
        return prediction