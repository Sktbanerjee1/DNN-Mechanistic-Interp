import torch
import torch.nn as nn
import torch.optim as optim
from analyzer import ManifoldAnalyzer, BayesianObserver
from model import ModuloNet
from data import generate_modulo_data
from viz import Visualizer

def main():
    # --- Configuration ---
    HIDDEN_LAYERS = 4
    HIDDEN_DIM = 64
    EPOCHS = 10000  
    LR = 0.005

    # 1. Prepare Data
    print("Generating Data...")
    X_train, X_test, y_train, y_test, _, _ = generate_modulo_data(num_samples=2000, test_size=0.2)
    
    # 2. Initialize Model
    model = ModuloNet(input_dim=2, hidden_dim=HIDDEN_DIM, output_dim=1, 
                      num_hidden_layers=HIDDEN_LAYERS, dropout_rate=0.1)

    # 3. Training Loop
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    print(f"Training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()
        
        # Test (Evaluation)
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            t_loss = criterion(test_preds, y_test)
        
        # Store raw MSE (Visualizer will convert to RMSE)
        train_losses.append(loss.item())
        test_losses.append(t_loss.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Train MSE {loss.item():.5f} | Test MSE {t_loss.item():.5f}")

    print("Training Complete.")

    # 4. Improved Loss Visualization
    print("Visualizing Performance (RMSE with Error Bands)...")
    # Window size determines how 'smooth' the line is. 
    # For 10,000 epochs, a window of 100-200 is good.
    Visualizer.plot_performance(train_losses, test_losses, window_size=150)

    # 5. Manifolds
    print("Visualizing Manifolds...")
    analyzer = ManifoldAnalyzer(model)
    manifolds = analyzer.get_layer_manifolds(X_test)
    Visualizer.plot_manifolds(manifolds, color_by_value=y_test.numpy())

    # 6. Bayesian
    print("Visualizing Uncertainty...")
    observer = BayesianObserver(model)
    mean_pred, uncertainty = observer.compute_uncertainty(X_test, num_samples=50)
    Visualizer.plot_bayesian_analysis(X_test, mean_pred, uncertainty)

if __name__ == "__main__":
    main()