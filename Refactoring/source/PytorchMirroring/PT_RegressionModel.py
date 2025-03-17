import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##############################################################################
# Optional: Utility masked losses
##############################################################################
def masked_mse(pred, target, mask):
    """Compute MSE only over entries where mask == True."""
    diff = pred - target
    diff_sq = diff * diff
    diff_sq_masked = diff_sq[mask]
    return diff_sq_masked.mean() if diff_sq_masked.numel() > 0 else torch.tensor(0.0)

def masked_poisson_loss(pred, target, mask):
    """
    pred: predicted Poisson mean (>=0)
    target: observed counts
    mask: boolean mask
    """
    # PyTorch PoissonNLLLoss can handle either log_input=True or False
    # Here we assume pred is already in 'mean space' => log_input=False
    # We'll pick out only the valid entries from pred & target
    if mask.sum() == 0:
        return torch.tensor(0.0)
    pred_masked = pred[mask]
    tgt_masked  = target[mask]
    loss_fn = nn.PoissonNLLLoss(log_input=False, reduction='mean')
    return loss_fn(pred_masked, tgt_masked)

def masked_categorical_loss(logits, target, mask):
    """
    logits: shape (batch_size, n_out, num_classes).
    target: shape (batch_size, n_out) with integer class indices.
    mask:   boolean shape (batch_size, n_out).
    We flatten so we can index properly.
    """
    bsz, n_out, n_cls = logits.shape
    # Flatten
    logits_2d = logits.reshape(bsz*n_out, n_cls)
    target_1d = target.reshape(bsz*n_out)
    mask_1d   = mask.reshape(bsz*n_out)

    if mask_1d.sum() == 0:
        return torch.tensor(0.0)
    logits_masked = logits_2d[mask_1d]
    target_masked = target_1d[mask_1d]
    return F.cross_entropy(logits_masked, target_masked, reduction='mean')

def build_mask(tensor, missing_marker):
    """Return a boolean mask of shape = tensor.shape. True => valid."""
    if missing_marker is None:
        return torch.ones_like(tensor, dtype=torch.bool)
    return (tensor != missing_marker)


##############################################################################
# PyTorch version of the RegressionModel
##############################################################################
class PyTorchRegressionModel(nn.Module):
    """
    A multi-layer perceptron with optional:
    - dropout
    - classification or Poisson output
    - prior prediction injection (add or multiply)
    - masked training
    """
    def __init__(
        self,
        n_in,
        n_out,
        units=None,
        use_bias=False,
        dropout_rate=0.0,
        activation="linear",
        output_activation=None,
        num_classes=None,
        out_dist=None,           # e.g. "poisson"
        has_prior_pred=False,
        prior_pred_op=None,      # "add" or "multiply"
        missing_marker=None,
    ):
        """
        Args:
            n_in         (int):   Input dimension.
            n_out        (int):   Output dimension.
            units        (list):  E.g. [64, 64] for two hidden layers each with 64 units.
            use_bias     (bool or list): Whether to use bias in each Dense layer.
            dropout_rate (float or list): Dropout rate(s). 
            activation   (str):   Activation for hidden layers, e.g. "relu".
            output_activation (str or None): e.g. "exponential" if out_dist="poisson".
            num_classes  (int or None): If set, final output = (batch_size, n_out, num_classes).
            out_dist     (str or None): e.g. "poisson" => final is forced >= 0.
            has_prior_pred (bool): if True, expects a prior to add or multiply with final output.
            prior_pred_op (str): "add" or "multiply" if has_prior_pred=True. 
            missing_marker: if not None, indicates a special value for masked training.
        """
        super().__init__()
        self.n_in          = n_in
        self.n_out         = n_out
        self.units         = units if units is not None else []
        self.use_bias      = use_bias if isinstance(use_bias, list) else [use_bias]*len(self.units)
        self.dropout_rate  = dropout_rate if isinstance(dropout_rate, list) else [dropout_rate]*len(self.units)
        self.activation    = activation
        self.output_activation = output_activation
        self.num_classes   = num_classes
        self.out_dist      = out_dist
        self.has_prior_pred = has_prior_pred
        self.prior_pred_op  = prior_pred_op
        self.missing_marker = missing_marker

        if self.prior_pred_op is None:
            # Default
            if self.out_dist == "poisson":
                self.prior_pred_op = "multiply"
            else:
                self.prior_pred_op = "add"

        if self.out_dist == "poisson" and self.output_activation is None:
            self.output_activation = "exponential"

        # Build layer dims
        # final_out_dim = n_out*(num_classes) if classification, else n_out
        final_out_dim = n_out*(num_classes if num_classes else 1)
        layer_dims = [n_in] + self.units + [final_out_dim]

        # Construct layers
        modules = []
        for i in range(len(layer_dims) - 1):
            in_dim  = layer_dims[i]
            out_dim = layer_dims[i+1]
            # A linear layer
            linear = nn.Linear(in_dim, out_dim, bias=(self.use_bias[i] if i < len(self.use_bias) else True))
            modules.append(linear)
            # Dropout if not last layer and dropout_rate>0
            if i < len(layer_dims) - 2:
                dr = self.dropout_rate[i] if i < len(self.dropout_rate) else 0.0
                if dr > 0:
                    modules.append(nn.Dropout(dr))

        self.layers = nn.ModuleList(modules)

    def forward(self, x, prior_pred=None):
        """
        x:          shape (batch_size, n_in)
        prior_pred: shape (batch_size, n_out) or (batch_size, n_out, num_classes)
                    if self.has_prior_pred=True. 
        """
        out = x
        linear_count = sum(isinstance(m, nn.Linear) for m in self.layers)
        linear_idx = 0

        for module in self.layers:
            if isinstance(module, nn.Linear):
                # Is it the final linear?
                linear_idx += 1
                is_final = (linear_idx == linear_count)
                out = module(out)
                # Activation
                if is_final:
                    # final activation
                    if self.num_classes is not None:
                        # reshape => (batch_size, n_out, num_classes)
                        bsz, total_dim = out.shape
                        out = out.view(bsz, self.n_out, self.num_classes)
                    if self.output_activation is not None:
                        out = self._activate(out, self.output_activation)
                else:
                    # hidden activation
                    out = self._activate(out, self.activation)
            else:
                # dropout
                out = module(out)

        # If prior is requested
        if self.has_prior_pred and (prior_pred is not None):
            if self.prior_pred_op == "add":
                out = out + prior_pred
            elif self.prior_pred_op == "multiply":
                out = out * prior_pred
            else:
                raise ValueError(f"Unsupported prior_pred_op: {self.prior_pred_op}")

        return out

    def _activate(self, tensor, activation_str):
        """Applies a string-based activation."""
        act_str = activation_str.lower()
        if act_str == "linear":
            return tensor
        elif act_str == "relu":
            return F.relu(tensor)
        elif act_str == "tanh":
            return torch.tanh(tensor)
        elif act_str == "sigmoid":
            return torch.sigmoid(tensor)
        elif act_str == "exponential":
            return torch.exp(tensor)
        else:
            raise ValueError(f"Unsupported activation: {activation_str}")

    def compute_loss(self, pred, target):
        """
        Chooses the appropriate masked loss:
         - MSE if out_dist not "poisson" and no classification
         - Poisson if out_dist="poisson"
         - Cross entropy if classification
        """
        # Build mask
        device = pred.device
        target_t = target.to(device)
        if self.num_classes is not None:
            # classification => target shape (batch, n_out) of integer class indices
            mask = build_mask(target_t, self.missing_marker)
            return masked_categorical_loss(pred, target_t.long(), mask)
        elif self.out_dist == "poisson":
            # Poisson => pred shape (batch, n_out)
            mask = build_mask(target_t, self.missing_marker)
            return masked_poisson_loss(pred, target_t, mask)
        else:
            # MSE
            mask = build_mask(target_t, self.missing_marker)
            return masked_mse(pred, target_t, mask)

    ########################################################################
    # Optional: fit(...) and predict(...) for a quick demonstration
    ########################################################################
    def fit(
        self,
        X_in,
        X_out,
        prior_pred=None,
        epochs=50,
        batch_size=32,
        lr=1e-3,
        verbose=True,
    ):
        """
        A simple training loop. This doesn't replicate all Keras features
        (no multiple init attempts, no LR schedules, etc.). You can expand as needed.
        
        X_in shape: (n_in, n_samples)
        X_out shape: (n_out, n_samples) or (n_out, n_samples, num_classes) if classification
        prior_pred shape:   (n_out, n_samples) or (n_out, n_samples, num_classes), optional
        """
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Transpose inputs: we want (n_samples, n_in)
        X_in_t  = torch.from_numpy(X_in.T).float()
        X_out_t = torch.from_numpy(X_out.T).float()

        # If we have prior
        if self.has_prior_pred and prior_pred is not None:
            prior_pred_t = torch.from_numpy(
                prior_pred.transpose(1,0,2) if self.num_classes is not None else prior_pred.T
            ).float()
        else:
            prior_pred_t = None

        # Setup DataLoader
        dataset = torch.utils.data.TensorDataset(X_in_t, X_out_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            for batch_idx, (xb, yb) in enumerate(dataloader):
                xb = xb.to(device)
                yb = yb.to(device)
                # If prior_pred is given, we need to gather the same subset from prior_pred_t
                # A quick hack is to rely on no shuffling or we store the indices. For simplicity:
                # let's do no shuffle => or you can store index in a custom dataset.
                
                if prior_pred_t is not None:
                    # same indexing
                    start = batch_idx * batch_size
                    end   = start + xb.size(0)
                    prior_batch = prior_pred_t[start:end].to(device)
                    pred = self.forward(xb, prior_pred=prior_batch)
                else:
                    pred = self.forward(xb)
                
                loss = self.compute_loss(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_loss = total_loss / len(dataloader)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    def predict(self, X_in, prior_pred=None):
        """
        Inference. 
        X_in shape: (n_in, n_samples)
        prior_pred shape: (n_out, n_samples) or (n_out, n_samples, num_classes)
        Returns: 
         - if classification: (n_out, n_samples, num_classes)
         - else: (n_out, n_samples)
        """
        device = next(self.parameters()).device
        self.eval()

        X_in_t = torch.from_numpy(X_in.T).float().to(device)
        if self.has_prior_pred and prior_pred is not None:
            if self.num_classes is not None:
                prior_pred_t = torch.from_numpy(prior_pred.transpose(1,0,2)).float().to(device)
            else:
                prior_pred_t = torch.from_numpy(prior_pred.T).float().to(device)
        else:
            prior_pred_t = None

        with torch.no_grad():
            if prior_pred_t is not None:
                preds = self.forward(X_in_t, prior_pred=prior_pred_t)
            else:
                preds = self.forward(X_in_t)
        preds_np = preds.cpu().numpy()

        if self.num_classes is not None:
            # shape => (batch_size, n_out, num_classes) => (n_out, n_samples, num_classes)
            preds_np = preds_np.transpose([1,0,2])
        else:
            # shape => (batch_size, n_out) => (n_out, n_samples)
            preds_np = preds_np.transpose([1,0])
        return preds_np
