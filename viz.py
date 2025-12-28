import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Visualizer:
    @staticmethod
    def _calculate_rolling_stats(data, window_size=50):
        """Helper for smoothing loss curves."""
        n = len(data)
        if n < window_size:
            return np.array(data), np.array(data), np.array(data)
        
        medians, p25s, p75s = [], [], []
        for i in range(n):
            start = max(0, i - window_size // 2)
            end = min(n, i + window_size // 2)
            window = data[start:end]
            medians.append(np.median(window))
            p25s.append(np.percentile(window, 25))
            p75s.append(np.percentile(window, 75))
            
        return np.array(medians), np.array(p25s), np.array(p75s)

    @staticmethod
    def plot_performance(train_losses, test_losses, window_size=100):
        """Plots RMSE Learning Curve with Error Bands."""
        train_rmse = np.sqrt(np.array(train_losses))
        test_rmse = np.sqrt(np.array(test_losses))
        
        t_med, t_low, t_high = Visualizer._calculate_rolling_stats(train_rmse, window_size)
        v_med, v_low, v_high = Visualizer._calculate_rolling_stats(test_rmse, window_size)
        epochs = np.arange(len(train_losses))

        plt.figure(figsize=(12, 6))
        plt.plot(epochs, t_med, label='Training RMSE (Median)', color='#1f77b4', linewidth=2)
        plt.fill_between(epochs, t_low, t_high, color='#1f77b4', alpha=0.2)
        
        plt.plot(epochs, v_med, label='Test RMSE (Median)', color='#ff7f0e', linewidth=2)
        plt.fill_between(epochs, v_low, v_high, color='#ff7f0e', alpha=0.2)

        plt.title('Model Performance: RMSE with Uncertainty Bands')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.yscale('log')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_lr_schedule(lr_history):
        plt.figure(figsize=(10, 3))
        plt.plot(lr_history, color='purple')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Steps')
        plt.ylabel('LR')
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def plot_evolution_grid(evolution_history, target_values, epochs_captured):
        """Grid plot: Rows=Epochs, Cols=Layers."""
        num_epochs = len(evolution_history)
        num_layers = len(evolution_history[0])
        
        fig = plt.figure(figsize=(3.5 * num_layers, 3 * num_epochs))
        plot_idx = 1
        
        for i, epoch_data in enumerate(evolution_history):
            epoch_label = epochs_captured[i]
            for j, manifold in enumerate(epoch_data):
                ax = fig.add_subplot(num_epochs, num_layers, plot_idx, projection='3d')
                ax.scatter(manifold[:, 0], manifold[:, 1], manifold[:, 2], 
                           c=target_values.flatten(), cmap='viridis', s=2, alpha=0.5)
                
                if i == 0: ax.set_title(f"Layer {j+1}")
                if j == 0: 
                    ax.set_zlabel(f"Epoch {epoch_label}", fontweight='bold')
                    ax.zaxis.set_rotate_label(False)
                    ax.set_zticks([])
                
                ax.set_xticklabels([]); ax.set_yticklabels([])
                plot_idx += 1
                
        plt.suptitle("Geometric Evolution of Thoughts", fontsize=16)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_multitask_separation(manifold, task_ids):
        """Visualizes task separation in the final layer."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(manifold[:, 0], manifold[:, 1], manifold[:, 2], 
                             c=task_ids.flatten(), cmap='coolwarm', s=10, alpha=0.7)
        ax.set_title("Multi-Task Separation (Blue=Mod, Red=Add)")
        cbar = plt.colorbar(scatter, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Modulo', 'Addition']) 
        plt.show()

    @staticmethod
    def plot_bayesian_analysis(inputs, mean_pred, uncertainty, task_mode="single"):
        """Plots Prediction surface and Uncertainty map."""
        fig = plt.figure(figsize=(14, 6))
        
        # In multi-task, inputs has 3 cols. We only plot X/Y (cols 0,1)
        x_vals = inputs[:, 0]
        y_vals = inputs[:, 1]
        
        # Plot 1: Mean Prediction
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        sc1 = ax1.scatter(x_vals, y_vals, mean_pred.flatten(), 
                   c=mean_pred.flatten(), cmap='coolwarm', s=5)
        ax1.set_title(f"Bayesian Prediction ({task_mode})")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Output")

        # Plot 2: Uncertainty
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        sc2 = ax2.scatter(x_vals, y_vals, uncertainty.flatten(), 
                        c=uncertainty.flatten(), cmap='magma', s=5)
        ax2.set_title("Epistemic Uncertainty (Confusion)")
        
        plt.colorbar(sc2, ax=ax2, label="Variance")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_nonlinear_manifold(embedding, values, method_name="UMAP"):
        """
        Visualizes UMAP/t-SNE embedding.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        p = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], 
                       c=values.flatten(), cmap='turbo', s=5, alpha=0.8)
        
        ax.set_title(f"Non-Linear Manifold Projection ({method_name})\nUnfolding the 'Swiss Roll'", fontsize=14)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        plt.colorbar(p, label="Modulo Value", shrink=0.7)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_neuron_frequencies(x_domain, neuron_data, fixed_y):
        """
        Plots the raw activation waves of the top 3 most periodic neurons.
        """
        # Sort neurons by max FFT magnitude (finding the 'loudest' periodic neurons)
        # neuron_data is list of (magnitude, freq, signal)
        scored_neurons = []
        for i, (mag, freq, sig) in enumerate(neuron_data):
            score = np.max(mag[1:]) # Max AC energy
            scored_neurons.append((score, i, sig, mag, freq))
            
        scored_neurons.sort(key=lambda x: x[0], reverse=True)
        top_neurons = scored_neurons[:3] # Top 3
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle(f"Fourier Analysis of Top 3 Neurons (Fixed Y={fixed_y})", fontsize=16)
        
        for idx, (score, neuron_id, signal, mag, freq) in enumerate(top_neurons):
            # Waveform Plot
            ax_wave = axes[idx, 0]
            ax_wave.plot(x_domain, signal, color='cyan', linewidth=1.5)
            ax_wave.set_title(f"Neuron {neuron_id} Activation Wave (Spatial)", fontsize=10)
            ax_wave.set_ylabel("Activation")
            ax_wave.set_facecolor('black')
            ax_wave.grid(color='gray', alpha=0.3)
            
            # Frequency Spectrum Plot
            ax_freq = axes[idx, 1]
            ax_freq.plot(freq[:50], mag[:50], color='magenta') # Zoom in on low freqs
            ax_freq.set_title(f"Neuron {neuron_id} Frequency Spectrum (Spectral)", fontsize=10)
            ax_freq.set_ylabel("Magnitude")
            ax_freq.set_xlabel("Frequency")
            ax_freq.grid(True, alpha=0.3)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()