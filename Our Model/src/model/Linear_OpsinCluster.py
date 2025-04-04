import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -----------------------------
# 1) Define the single-cluster opsin model (with full A_o)
# -----------------------------
class SingleClusterOpsinModel(nn.Module):
    def __init__(self, No):
        super().__init__()
        self.A_o = nn.Parameter(torch.randn(No, No) * 0.1) 
        self.B_o = nn.Parameter(torch.randn(No, 1) * 0.1)
        self.C_o = nn.Parameter(torch.randn(1, No) * 0.1)
        self.beta = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, u):
        """
        u: Tensor of shape [T] representing the scalar light input over T time steps.
        Returns the photocurrent y of shape [T].
        """
        T = u.shape[0]
        x = torch.zeros((self.A_o.size(0),), dtype=u.dtype, device=u.device)
        y = torch.zeros((T,), dtype=u.dtype, device=u.device)

        for t in range(T):
            x = self.A_o @ x + self.B_o.view(-1) * (self.beta * u[t])
            y[t] = self.C_o @ x
        return y

# -----------------------------
# 2) Define the RelativeErrorLoss class
# -----------------------------
class RelativeErrorLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        return torch.mean(((y_pred - y_true) / (y_true + self.epsilon)) ** 2)

# -----------------------------
# 3) Load real pulse data from CSV
# -----------------------------
# Load pulse_data.csv
try:
    df = pd.read_csv('pulse_data.csv')
    print(f"Successfully loaded data with {len(df)} rows")
    
    t_ms = torch.tensor(df['t_ms'].values, dtype=torch.float32)
    u = torch.tensor(df['u_mWmm2'].values, dtype=torch.float32)
    y_target = torch.tensor(df['I_nA'].values, dtype=torch.float32)
    
    max_u = u.max()
    if max_u > 0:
        u = u / max_u
        
    max_points = min(500, len(t_ms))
    step = len(t_ms) // max_points if len(t_ms) > max_points else 1
    
    t_ms = t_ms[::step]
    u = u[::step]
    y_target = y_target[::step]
    
    print(f"Using {len(u)} data points with time range: {t_ms[0]:.1f} to {t_ms[-1]:.1f} ms")
    
except Exception as e:
    print(f"Error loading data: {e}")
    print("Falling back to synthetic data...")
    T = 500
    t_ms = torch.linspace(0, 500, T)
    u = torch.zeros(T)
    u[100:300] = 1.0  # Light on from t=100 to t=300
    
    y_target = torch.zeros(T)
    for i in range(T):
        if i < 100:
            y_target[i] = 0
        elif i < 300:
            y_target[i] = 2 * (1 - torch.exp(-(i-100)/50))
        else:
            y_target[i] = y_target[299] * torch.exp(-(i-300)/100)

# -----------------------------
# 4) Experiment with different numbers of opsin states
# -----------------------------
No_options = [10]
best_model = None
best_loss = float('inf')
best_No = None

for No in No_options:
    print(f"\nTraining model with {No} opsin states...")
    
    model = SingleClusterOpsinModel(No)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)
    loss_fn = RelativeErrorLoss(epsilon=1e-6)
    
    # Training loop
    epochs = 4000
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(u)
        loss = loss_fn(y_pred, y_target)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        losses.append(loss.item())
        
        # Print learning rate and loss every 200 epochs
        if epoch % 200 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}, Loss = {loss.item():.6f}, Learning Rate = {current_lr:.6f}")
    
    final_loss = losses[-1]
    print(f"Final loss for No={No}: {final_loss:.6f}")
    
    if final_loss < best_loss:
        best_loss = final_loss
        best_model = model
        best_No = No

print(f"\nBest model has {best_No} opsin states with loss: {best_loss:.6f}")

# -----------------------------
# 5) Visualize results with the best model
# -----------------------------
with torch.no_grad():
    y_pred = best_model(u)

plt.figure(figsize=(12, 10))

# Plot the training loss
plt.subplot(3, 1, 1)
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training Loss (No={best_No})")
plt.grid(True)

# Plot the input light
plt.subplot(3, 1, 2)
plt.plot(t_ms.numpy(), (u * max_u).numpy() if 'max_u' in locals() else u.numpy())
plt.xlabel("Time (ms)")
plt.ylabel("Light Intensity (mW/mm²)")
plt.title("Light Input")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t_ms.numpy(), y_target.numpy(), 'b-', label="Measured Photocurrent")
plt.plot(t_ms.numpy(), y_pred.numpy(), 'r--', label="Model Prediction")
plt.xlabel("Time (ms)")
plt.ylabel("Photocurrent (nA)")
plt.title("Photocurrent Fitting")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("photocurrent_fitting_results.png", dpi=300)
plt.show()


print("Learned parameters for the best model:")
print("A_o:\n", best_model.A_o.data)
print("B_o:\n", best_model.B_o.data)
print("C_o:\n", best_model.C_o.data)
print("beta:\n", best_model.beta.data)

mse = nn.MSELoss()(y_pred, y_target).item()
mae = nn.L1Loss()(y_pred, y_target).item()
r2 = 1 - torch.sum((y_target - y_pred)**2) / torch.sum((y_target - y_target.mean())**2)

print("\nPerformance Metrics:")
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R-squared: {r2:.6f}")
