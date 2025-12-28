import torch
import numpy as np
from sklearn.model_selection import train_test_split

class TaskDataGenerator:
    @staticmethod
    def get_task_data(task_name="modulo", num_samples=2000, test_size=0.2):
        """
        Generates data for specific mathematical tasks.
        Supported: 'modulo', 'multiply', 'add', 'mix'
        """
        range_vals = (1, 100)
        X_vals = np.random.uniform(range_vals[0], range_vals[1], num_samples)
        Y_vals = np.random.uniform(range_vals[0], range_vals[1], num_samples)
        Y_vals = np.maximum(Y_vals, 1.0) # Safety
        
        # --- Task Logic ---
        if task_name == "modulo":
            Z_vals = np.mod(X_vals, Y_vals)
            # Normalize Target (0 to ~1)
            Z_norm = Z_vals / range_vals[1]
            inputs = np.vstack((X_vals, Y_vals)).T
            
        elif task_name == "multiply":
            Z_vals = X_vals * Y_vals
            # Normalize Target (1 to 10000 -> 0 to 1)
            Z_norm = (Z_vals - 1) / (10000 - 1)
            inputs = np.vstack((X_vals, Y_vals)).T
            
        elif task_name == "add":
            Z_vals = X_vals + Y_vals
            # Normalize Target (2 to 200 -> 0 to 1)
            Z_norm = (Z_vals - 2) / (200 - 2)
            inputs = np.vstack((X_vals, Y_vals)).T

        elif task_name == "multi_task":
            # Half data is Modulo, Half is Addition
            # Input format: [x, y, task_id] where task_id is 0 for mod, 1 for add
            half = num_samples // 2
            
            # Task 0: Modulo
            x1 = X_vals[:half]
            y1 = Y_vals[:half]
            z1 = np.mod(x1, y1) / 100.0
            t1 = np.zeros(half) # Task ID 0
            
            # Task 1: Addition
            x2 = X_vals[half:]
            y2 = Y_vals[half:]
            z2 = (x2 + y2) / 200.0
            t2 = np.ones(half) # Task ID 1
            
            # Combine
            inputs = np.vstack((
                np.concatenate([x1, x2]),
                np.concatenate([y1, y2]),
                np.concatenate([t1, t2])
            )).T
            
            Z_norm = np.concatenate([z1, z2])
            
            # Shuffle manually since we concatenated
            idx = np.random.permutation(len(inputs))
            inputs = inputs[idx]
            Z_norm = Z_norm[idx]

        # Normalize Inputs (X, Y)
        # Note: If multi-task, we don't normalize the 3rd column (Task ID)
        inputs[:, 0] = (inputs[:, 0] - range_vals[0]) / (range_vals[1] - range_vals[0])
        inputs[:, 1] = (inputs[:, 1] - range_vals[0]) / (range_vals[1] - range_vals[0])
        
        targets = Z_norm.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            inputs, targets, test_size=test_size, random_state=42
        )

        return (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )