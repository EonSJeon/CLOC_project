import torch
import torch.nn as nn
from DPAD import DPADModel
from data_utils import *
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class UnnaturalStates(nn.Module):
    def __init__(self, n1, n2, p, true_Cy1 = None) -> None:
        super(UnnaturalStates, self).__init__()
        """
        Model Formulation:
        x2[t+1] = A22 * x2[t] - (K2 Cy1) x1[t] + K2 * y[t]
        y2[t] = Cy2 * x2[t]
        y[t] = y2[t] + y1[t]

        Shapes: 
        x2 -> (timesteps, num_unnatural_states)
        x1 -> (timesteps, num_natural_states)

        
        Parameters:
        A22 -> (num_unnatural_states, num_unnatural states)
        Cy1 -> (num_neural_states, num_natural_states)
        K2 -> (num_unnatural_states, num_neural_states)
        Cy2 -> (num_neural_states, num_unnatural_states)

        """
        self.n2 = n2
        self.ny = p
        self.n1 = n1

        # Init A22, K2, Cy2
        self.A22 = nn.Parameter(torch.randn((n2, n2)) * 0.1)  # Recursion parameter for x2[t]
        self.K2 = nn.Parameter(torch.randn(n2, self.ny) * 0.1)  # Mapping from y[t] to x2[t]
        self.Cy2 = nn.Parameter(torch.randn(self.ny, n2) * 0.1)  # x2 -> y2

    def forward(self, y, y1):

        
        # return x2s, y2s
        assert (y.shape[0] == y1.shape[0], """
               Natural States and Neural Signals must match time dimension, instead got:
                
                y: {y.shape[0]} 
                x1: {x1.shape[0]}
                """)
        
        T = y.shape[0]

        x2 = torch.zeros(size=(T, self.n2))
        y_out = torch.zeros(size=(T, self.ny))

        for t in range(T-1):
            x2_next = self.A22 @ x2[t] + self.K2 @ y[t] - self.K2 @ y1[t]
            x2 = x2.clone()
            x2[t+1] = x2_next
            y_out[t] = self.Cy2 @ x2[t]


        return x2, y_out

            
            
# --------------
# Training Loop
# --------------
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    # Get Stanley Data
    y_data, _, _ = process_stanley_data("../../data/AB020_3_Proc.nwb") 
    train_y, test_y = y_data[:500], y_data[500:1000]

    print(y_data[0]) # (, 32)

    losses = []

    # Get natural states by running DPAD
    natural_states_model = DPADModel()
    methodCode = DPADModel.prepare_args("DPAD_Linear")
    natural_states_model.fit(train_y.T, nx=5, n1=5) # 5 natural states
    _, y1_pred, x1_pred = natural_states_model.predict(train_y)

    y1_pred = torch.tensor(y1_pred, dtype=torch.float32)

    # Training X2 parameters
    model = UnnaturalStates(
        n1=5,
        n2=5,
        p=32
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # -------------
    # Training Loop
    # -------------
    epochs = 500
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Ensure y_data is properly formatted
        if isinstance(train_y, np.ndarray):
            train_y = torch.tensor(train_y, dtype=torch.float32)
        
        # Make sure dimensions match what your model expects
        _, y2_pred = model(train_y, y1_pred)
        
        # Make sure pred and train_y have compatible shapes for the loss function
        loss = criterion(y1_pred + y2_pred, train_y)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

        print(f"Eppch {epoch+1} ----------------------------- Loss: {loss.item()}")


    fig, axs = plt.subplots(1, 2, figsize=(15, 6))  # 1 row, 2 columns
    axs[0].plot(losses, label="Loss")
    axs[0].set_title("Training Loss of Unnatural States Model")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("MSE Loss")
    axs[0].legend()
    plt.show()


    # -----------
    # Testing Run
    # -----------
    _, y1_preds, x1_preds = natural_states_model.predict(test_y)
    y1_preds = torch.tensor(y1_preds)
    test_y = torch.tensor(test_y)
    model.eval()
    with torch.no_grad():
        _, y2_preds = model.forward(test_y, y1_preds)
        y_preds = y1_preds + y2_preds
        loss = criterion(y_preds, test_y)
        print(f"Testing Run (all 32 channels of y_data) ------------------ Testing Loss: {loss.item()}")

