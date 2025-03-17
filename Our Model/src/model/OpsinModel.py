import torch
import torch.nn as nn

class OpsinModel(nn.Module):
    """
    A simple PyTorch module for opsin state update:
      x_opsin[t+1] = A_opsinopsin * x_opsin[t] + B_u * u[t]
    plus a method to compute photocurrents: photocurrent_i = C_opsin[i,:] x_opsin_i[t].
    """

    def __init__(self, N_opsin, b, p):
        super(OpsinModel, self).__init__()
        self.N_opsin = N_opsin
        self.b = b
        self.p = p

        # We store these as learnable Parameters:
        # For a real application, you might initialize them more carefully.
        # shape: (b*N_opsin, b*N_opsin)
        self.A_opsinopsin = nn.Parameter(
            0.01*torch.randn(N_opsin*b, N_opsin*b)
        )
        # shape: (b*N_opsin, p)
        self.B_u = nn.Parameter(
            0.01*torch.randn(N_opsin*b, p)
        )
        # shape: (b, N_opsin)
        self.C_opsin = nn.Parameter(
            0.01*torch.randn(b, N_opsin)
        )

    def opsin_state_update(self, x_opsin, u):
        """
        Given x_opsin[t] and u[t], return x_opsin[t+1].
        x_opsin: (b*N_opsin,) or (batch_size, b*N_opsin)
        u:       (p,) or (batch_size, p)
        """
        # For simplicity, assume x_opsin has shape (b*N_opsin,) in a single-sample scenario.
        return torch.matmul(self.A_opsinopsin, x_opsin) + torch.matmul(self.B_u, u)

    def compute_photocurrents(self, x_opsin):
        """
        Returns a vector of length b, each entry is C_opsin[i,:] x_opsin_i.
        x_opsin: shape (b*N_opsin,)
        """
        # Reshape x_opsin to (b, N_opsin)
        x_opsin_blocked = x_opsin.view(self.b, self.N_opsin)
        # Multiply each row by self.C_opsin[i,:]
        # shape of self.C_opsin: (b, N_opsin)
        photocurrents = torch.sum(self.C_opsin * x_opsin_blocked, dim=1)
        return photocurrents
