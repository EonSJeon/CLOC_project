import torch
import torch.nn as nn
import torch.optim as optim
from OpsinModel import OpsinModel

class CLOCModel(nn.Module):
    """
    A PyTorch module that builds an internal OpsinModel and defines:
      - Natural and Unnatural states' matrices as learnable parameters.
      - A forward() method that simulates the entire system over time.
    """

    def __init__(self, N_nat, N_unnat, N_opsin, N_cluster, N_elec):
        super(CLOCModel, self).__init__()
        self.N_nat = N_nat
        self.N_unnat = N_unnat

        # Build the opsin sub-module
        self.opsin_model = OpsinModel(N_opsin, N_cluster, N_elec)

        # ------ Natural dynamics parameters ------
        # A_natnat: shape (N_nat x N_nat)
        self.A_natnat = nn.Parameter(
            0.01*torch.randn(N_nat, N_nat)
        )
        # K_nat: shape (N_nat x 1)
        self.K_nat = nn.Parameter(
            0.01*torch.randn(N_nat, 1)
        )
        # C_y_nat: shape (1 x N_nat)
        self.C_y_nat = nn.Parameter(
            0.01*torch.randn(1, N_nat)
        )

        # ------ Unnatural dynamics parameters ------
        self.A_unnatunnat = nn.Parameter(
            0.01*torch.randn(N_unnat, N_unnat)
        )
        self.K_unnat = nn.Parameter(
            0.01*torch.randn(N_unnat, 1)
        )
        self.C_y_unnat = nn.Parameter(
            0.01*torch.randn(1, N_unnat)
        )

        # ------ Photocurrent mapping to x_nat, x_unnat ------
        # shape (N_nat, b) and (N_unnat, b)
        self.Bp_nat = nn.Parameter(
            0.01*torch.randn(N_nat, N_cluster)
        )
        self.Bp_unnat = nn.Parameter(
            0.01*torch.randn(N_unnat, N_cluster)
        )

        # We'll store the dimension info needed for forward pass
        self.N_opsin = N_opsin
        self.b = N_cluster
        self.p = N_elec

    def predict_output(self, x_nat, x_unnat):
        """
        Single-step output:
          y[t] = C_y_nat x_nat[t] + C_y_unnat x_unnat[t].
        """
        y_nat_part = torch.matmul(self.C_y_nat, x_nat)   # shape (1,)
        y_unnat_part = torch.matmul(self.C_y_unnat, x_unnat) # shape (1,)
        return y_nat_part + y_unnat_part

    def state_update(self, x_nat, x_unnat, x_opsin, u, y):
        """
        Single-step update:
          x_nat[t+1]   = A_natnat x_nat[t] + K_nat y[t] + Bp_nat * photocurrents
          x_unnat[t+1] = A_unnatunnat x_unnat[t] - K_unnat*(C_y_nat x_nat[t]) + K_unnat y[t] + Bp_unnat * photocurrents
          x_opsin[t+1] = opsin_model(...)
        """
        # 1) natural
        x_nat_next = torch.matmul(self.A_natnat, x_nat) + self.K_nat.view(-1)*y

        # 2) unnatural
        c_yx = torch.matmul(self.C_y_nat, x_nat)  # scalar
        x_unnat_next = (
            torch.matmul(self.A_unnatunnat, x_unnat)
            - self.K_unnat.view(-1)*c_yx
            + self.K_unnat.view(-1)*y
        )

        # 3) opsin
        x_opsin_next = self.opsin_model.opsin_state_update(x_opsin, u)

        # 4) photocurrents from x_opsin[t]
        photocurrents = self.opsin_model.compute_photocurrents(x_opsin)  # shape (b,)
        # Add them in
        # shape(Bp_nat) = (N_nat, b), shape(photocurrents) = (b,)
        x_nat_next   += torch.matmul(self.Bp_nat, photocurrents)
        x_unnat_next += torch.matmul(self.Bp_unnat, photocurrents)

        return x_nat_next, x_unnat_next, x_opsin_next

    def forward(self, x_nat_0, x_unnat_0, x_opsin_0, U):
        """
        Run a forward simulation over the entire time sequence:
          - x_nat_0, x_unnat_0, x_opsin_0 are (N_nat,), (N_unnat,), (N_opsin*b,)
          - U: shape (T, p) for T time steps
        Returns:
          y_preds: shape (T,) of predicted outputs
          x_nat_hist, x_unnat_hist, x_opsin_hist for reference
        """
        T = U.shape[0]
        N_opsin_total = self.N_opsin * self.b

        # Make sure everything is float tensors
        x_nat = x_nat_0.clone()
        x_unnat = x_unnat_0.clone()
        x_opsin = x_opsin_0.clone()

        y_preds = []
        x_nat_hist = [x_nat]
        x_unnat_hist = [x_unnat]
        x_opsin_hist = [x_opsin]

        for t in range(T):
            # predict y[t] from current states
            y_t = self.predict_output(x_nat, x_unnat)  # shape (1,)
            y_preds.append(y_t)

            # update states
            x_nat, x_unnat, x_opsin = self.state_update(
                x_nat, x_unnat, x_opsin, U[t], y_t
            )
            x_nat_hist.append(x_nat)
            x_unnat_hist.append(x_unnat)
            x_opsin_hist.append(x_opsin)

        y_preds = torch.stack(y_preds).view(-1)  # shape (T,)
        # If you want the states stacked: shape (T+1, dim)
        x_nat_hist = torch.stack(x_nat_hist)
        x_unnat_hist = torch.stack(x_unnat_hist)
        x_opsin_hist = torch.stack(x_opsin_hist)

        return y_preds, x_nat_hist, x_unnat_hist, x_opsin_hist
