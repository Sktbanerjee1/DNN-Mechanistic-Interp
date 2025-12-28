import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data import TaskDataGenerator
from analyzer import ManifoldAnalyzer, BayesianObserver
from model import ModuloNet
from viz import Visualizer


# --- CONFIGURATION ---
# Options: 'modulo', 'multiply', 'add', 'multi_task'
# Try 'multi_task' to see the model learn two math rules at once!
TASK_MODE = "multi_task" 

HIDDEN_LAYERS = 3
HIDDEN_DIM = 64
EPOCHS = 10000
LR_MAX = 0.01

def main():
    print(f"--- Starting Experiment: {TASK_MODE.upper()} ---")

    # 1. Data Generation
    # Multi-task input: (x, y, task_id). Single-task: (x, y)
    input_dim = 3 if TASK_MODE == "multi_task" else 2
    
    X_train, X_test, y_train, y_test = TaskDataGenerator.get_task_data(TASK_MODE, num_samples=2500)
    
    print(f"Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")

    # 2. Model Initialization
    model = ModuloNet(input_dim=input_dim, 
                      hidden_dim=HIDDEN_DIM, 
                      output_dim=1, 
                      num_hidden_layers=HIDDEN_LAYERS,
                      dropout_rate=0.1) # 0.1 needed for Bayesian Analysis

    # 3. Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=LR_MAX)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, 
                                              steps_per_epoch=1, epochs=EPOCHS)
    criterion = nn.MSELoss()

    # 4. Storage
    analyzer = ManifoldAnalyzer(model)
    evolution_history = []
    lr_history = []
    train_losses = []
    test_losses = []
    
    capture_epochs = [0, 200, 1000, EPOCHS-1] 

    print("Training started...")
    
    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # --- Evaluate (Loss Tracking) ---
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            t_loss = criterion(test_preds, y_test)
        
        # Store metrics
        train_losses.append(loss.item())
        test_losses.append(t_loss.item())
        lr_history.append(scheduler.get_last_lr()[0])
        
        # --- Capture Evolution ---
        if epoch in capture_epochs:
            print(f"  > Snapshot Epoch {epoch} | Test RMSE: {np.sqrt(t_loss.item()):.4f}")
            manifolds = analyzer.get_layer_manifolds(X_test)
            evolution_history.append(manifolds)

    print("Training Complete.")

    # ================= VISUALIZATION =================
    
    # 1. Performance (RMSE + Bands)
    print("\n[1/5] Visualizing Performance (RMSE)...")
    Visualizer.plot_performance(train_losses, test_losses, window_size=100)

    # 2. Learning Rate
    print("[2/5] Visualizing Learning Rate...")
    Visualizer.plot_lr_schedule(lr_history)

    # 3. Geometric Evolution
    print("[3/5] Visualizing Geometric Evolution...")
    Visualizer.plot_evolution_grid(evolution_history, y_test.numpy(), capture_epochs)

    # 4. Bayesian Analysis (Uncertainty)
    print("[4/5] Performing Bayesian Analysis (This takes a moment)...")
    observer = BayesianObserver(model)
    # Note: We use X_test to see where the model is confused on unseen data
    mean_pred, uncertainty = observer.compute_uncertainty(X_test, num_samples=100)
    Visualizer.plot_bayesian_analysis(X_test, mean_pred, uncertainty, task_mode=TASK_MODE)

    # 5. Multi-Task Separation (Specific to multi_task mode)
    if TASK_MODE == "multi_task":
        print("[5/5] Visualizing Task Separation...")
        manifolds = analyzer.get_layer_manifolds(X_test)
        last_hidden_layer = manifolds[-1]
        task_ids = X_test[:, 2].numpy() # 3rd column is task ID
        Visualizer.plot_multitask_separation(last_hidden_layer, task_ids)

if __name__ == "__main__":
    main()