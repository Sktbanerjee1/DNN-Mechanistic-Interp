import torch
import numpy as np
from sklearn.decomposition import PCA
import warnings

# Try importing UMAP, fall back to t-SNE if not installed
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.manifold import TSNE

class NonLinearAnalyzer:
    """
    Uses Non-Linear Dimensionality Reduction (UMAP/t-SNE) 
    to see structures PCA misses.
    """
    def __init__(self, model):
        self.model = model

    def get_manifold_embedding(self, inputs, layer_idx=-1):
        self.model.eval()
        with torch.no_grad():
            _, activations = self.model(inputs, return_activations=True)
        
        # Get specific layer (default: last hidden layer)
        data = activations[layer_idx].cpu().numpy()
        
        # Deduplicate to prevent UMAP/t-SNE crashing on identical points
        unique_data, indices = np.unique(data, axis=0, return_index=True)
        
        if HAS_UMAP:
            # n_neighbors=15 preserves local structure, min_dist=0.1 controls clump tightness
            reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1, random_state=42)
            embedding = reducer.fit_transform(unique_data)
            method = "UMAP"
        else:
            warnings.warn("UMAP not found. Installing 'umap-learn' is recommended. Using t-SNE instead.")
            reducer = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
            embedding = reducer.fit_transform(unique_data)
            method = "t-SNE"
            
        return embedding, indices, method

class FourierDetective:
    """
    Analyzes if neurons are acting as Fourier components (Sine/Triangle waves).
    """
    def __init__(self, model):
        self.model = model

    def scan_periodicity(self, range_vals=(0, 100), fixed_y=10.0, num_steps=1000):
        """
        Sweeps X from 0 to 100 while keeping Y fixed.
        Checks model input dimension to handle Single-Task vs Multi-Task automatically.
        """
        self.model.eval()
        
        # Create sweep data
        x = np.linspace(range_vals[0], range_vals[1], num_steps)
        y = np.full_like(x, fixed_y)
        
        # --- FIXED LOGIC ---
        # Explicitly check what dimension the model expects
        first_layer = self.model.layers[0]
        required_dim = first_layer.in_features
        
        if required_dim == 3:
            # Multi-Task Mode: We need [x, y, task_id]
            # We assume Task ID 0 (Modulo) for this scan
            z = np.zeros_like(x)
            inputs = np.vstack((x, y, z)).T
        else:
            # Single-Task Mode: We need [x, y]
            inputs = np.vstack((x, y)).T
            
        t_inputs = torch.tensor(inputs, dtype=torch.float32)

        # Execute with NO_GRAD to prevent RuntimeError during numpy conversion
        with torch.no_grad():
            _, activations = self.model(t_inputs, return_activations=True)
            last_hidden = activations[-1].detach().cpu().numpy()
        
        # Perform FFT on every neuron
        neuron_freqs = []
        for i in range(last_hidden.shape[1]):
            signal = last_hidden[:, i]
            # Normalize signal
            signal = signal - np.mean(signal)
            
            # FFT
            fft_vals = np.fft.rfft(signal)
            fft_freq = np.fft.rfftfreq(len(signal), d=(x[1]-x[0]))
            
            # Find dominant frequency
            magnitude = np.abs(fft_vals)
            if len(magnitude) > 1:
                peak_idx = np.argmax(magnitude[1:]) + 1 # Ignore DC component
            else:
                peak_idx = 0
                
            neuron_freqs.append((magnitude, fft_freq, signal))
            
        return x, neuron_freqs, last_hidden
    
    
class ManifoldAnalyzer:
    """
    Analyzes the geometric structure of the learned representations.
    """
    def __init__(self, model):
        self.model = model

    def get_layer_manifolds(self, inputs):
        """
        Projects high-dimensional neuron activations into 3D space using PCA.
        This visualizes how the network 'unfolds' the data to solve the modulo problem.
        """
        self.model.eval()
        with torch.no_grad():
            _, activations = self.model(inputs, return_activations=True)
        
        projected_layers = []
        pca = PCA(n_components=3)
        
        for act in activations:
            # Convert to numpy
            act_np = act.cpu().numpy()
            # If layer has < 3 neurons, pad it, otherwise project down
            if act_np.shape[1] >= 3:
                projected = pca.fit_transform(act_np)
            else:
                # Pad with zeros for visualization if layer is too small
                padded = np.zeros((act_np.shape[0], 3))
                padded[:, :act_np.shape[1]] = act_np
                projected = padded
            projected_layers.append(projected)
            
        return projected_layers

class BayesianObserver:
    """
    Performs Bayesian Analysis using Monte Carlo Dropout.
    Evaluates the 'Chain of Thoughts' by measuring uncertainty.
    """
    def __init__(self, model):
        self.model = model

    def compute_uncertainty(self, inputs, num_samples=50):
        """
        Runs the model multiple times on the same input with dropout enabled.
        Mean = Prediction
        Variance = Epistemic Uncertainty (Confusion)
        """
        self.model.train() # Enable Dropout
        outputs = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.model(inputs)
                outputs.append(pred.numpy())
        
        outputs = np.array(outputs)
        mean_pred = outputs.mean(axis=0)
        uncertainty = outputs.var(axis=0) # Variance represents uncertainty
        
        return mean_pred, uncertainty