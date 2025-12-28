import umap
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings

from data import TaskDataGenerator
from analyzer import NonLinearAnalyzer, FourierDetective, BayesianObserver, ManifoldAnalyzer
from model import ModuloNet
from viz import Visualizer

# --- CONFIGURATION ---
TASK_MODE = "multi_task" 
HIDDEN_LAYERS = 3
HIDDEN_DIM = 64
EPOCHS = 3000
LR_MAX = 0.01

def main():
    print(f"==================================================")
    print(f"   FULL SPECTRUM ANALYSIS: {TASK_MODE.upper()} TASK")
    print(f"==================================================")

    # ---------------------------------------------------------
    # 1. SETUP & TRAINING
    # ---------------------------------------------------------
    input_dim = 3 if TASK_MODE == "multi_task" else 2
    
    # Generate Data
    print("[1/6] Generating Data...")
    X_train, X_test, y_train, y_test = TaskDataGenerator.get_task_data(TASK_MODE, num_samples=2500)
    
    # Initialize Model
    model = ModuloNet(input_dim=input_dim, 
                      hidden_dim=HIDDEN_DIM, 
                      output_dim=1, 
                      num_hidden_layers=HIDDEN_LAYERS,
                      dropout_rate=0.1) # Needed for Bayesian

    optimizer = optim.Adam(model.parameters(), lr=LR_MAX)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, 
                                              steps_per_epoch=1, epochs=EPOCHS)
    criterion = nn.MSELoss()

    # Storage for analysis
    raw_snapshots = [] # For UMAP Evolution
    capture_epochs = [0, 100, 500, EPOCHS-1] 
    train_losses = []
    test_losses = []
    lr_history = []

    print(f"[2/6] Training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        # --- Train Step ---
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # --- Validation Step ---
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            t_loss = criterion(test_preds, y_test)
        
        # --- Recording ---
        train_losses.append(loss.item())
        test_losses.append(t_loss.item())
        lr_history.append(scheduler.get_last_lr()[0])
        
        # --- Snapshots for UMAP Evolution ---
        if epoch in capture_epochs:
            print(f"      > Snapshot Epoch {epoch} | Test RMSE: {np.sqrt(t_loss.item()):.4f}")
            with torch.no_grad():
                # Capture all layer activations
                _, activations = model(X_test, return_activations=True)
                # Store as numpy arrays
                snapshot_layers = [act.detach().cpu().numpy() for act in activations]
                raw_snapshots.append(snapshot_layers)

    print("Training Complete.")

    # ---------------------------------------------------------
    # 2. STANDARD PERFORMANCE PLOTS
    # ---------------------------------------------------------
    print("\n[3/6] Visualizing Performance Metrics...")
    Visualizer.plot_performance(train_losses, test_losses, window_size=100)
    Visualizer.plot_lr_schedule(lr_history)

    # ---------------------------------------------------------
    # 3. GEOMETRIC EVOLUTION (UMAP GRID)
    # ---------------------------------------------------------
    print("\n[4/6] Processing Evolutionary UMAP Grid...")
    # This takes the snapshots and runs UMAP on each one
    evolution_history = []
    
    # Suppress UMAP warnings for cleanliness
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, epoch_layers in enumerate(raw_snapshots):
            epoch_num = capture_epochs[i]
            print(f"      > Processing Epoch {epoch_num}...")
            
            umap_epoch_manifolds = []
            for layer_data in epoch_layers:
                # Run UMAP on the full layer data
                reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1, random_state=42)
                embedding = reducer.fit_transform(layer_data)
                umap_epoch_manifolds.append(embedding)
            
            evolution_history.append(umap_epoch_manifolds)

    Visualizer.plot_evolution_grid(evolution_history, y_test.numpy(), capture_epochs)

    # ---------------------------------------------------------
    # 4. FINAL FORENSICS (Detailed UMAP + Fourier)
    # ---------------------------------------------------------
    print("\n[5/6] Running Advanced Forensics...")
    
    # A. Final High-Res UMAP
    print("      > Generating High-Res Final UMAP...")
    nl_analyzer = NonLinearAnalyzer(model)
    embedding, indices, method = nl_analyzer.get_manifold_embedding(X_test)
    # Filter targets to match deduplicated indices
    filtered_targets = y_test.numpy()[indices]
    Visualizer.plot_nonlinear_manifold(embedding, filtered_targets, method_name=method)

    # B. Fourier Analysis
    print("      > Running Fourier Detective (Periodicity Scan)...")
    detective = FourierDetective(model)
    # Scan X from 0-100 while holding Y=12.0
    x_domain, neuron_data, _ = detective.scan_periodicity(range_vals=(0, 100), fixed_y=12.0)
    Visualizer.plot_neuron_frequencies(x_domain, neuron_data, fixed_y=12.0)

    # ---------------------------------------------------------
    # 5. BAYESIAN ANALYSIS
    # ---------------------------------------------------------
    print("\n[6/6] Performing Bayesian Uncertainty Analysis...")
    observer = BayesianObserver(model)
    mean_pred, uncertainty = observer.compute_uncertainty(X_test, num_samples=100)
    Visualizer.plot_bayesian_analysis(X_test, mean_pred, uncertainty, task_mode=TASK_MODE)

    # ---------------------------------------------------------
    # 6. MULTI-TASK SEPARATION CHECK
    # ---------------------------------------------------------
    if TASK_MODE == "multi_task":
        print("\n[Bonus] Checking Task Separation...")
        # Use PCA for the simple separation check as it's cleaner for global clusters
        pca_analyzer = ManifoldAnalyzer(model)
        manifolds = pca_analyzer.get_layer_manifolds(X_test)
        last_layer = manifolds[-1]
        task_ids = X_test[:, 2].numpy()
        Visualizer.plot_multitask_separation(last_layer, task_ids)

    print("\n==================================================")
    print("   ANALYSIS COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    main()