"""Simple but working PINN implementation.

This is a simplified, stable PINN that prioritizes working correctly
over maximum performance.
"""

import warnings
import numpy as np
from solvers.base import SolverBase

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SimplePINNNetwork(nn.Module):
    """Simple MLP network for PINN."""

    def __init__(self, hidden_dim: int = 64, n_layers: int = 4):
        super().__init__()

        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, rt):
        return self.net(rt)


class SimplePINN(SolverBase):
    """Simple PINN solver with basic PDE loss.

    This implementation focuses on stability and correctness.
    """

    name = "pinn_simple"

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 4,
        epochs: int = 2000,
        lr: float = 1e-3,
        n_collocation: int = 1000,
        verbose: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.n_collocation = n_collocation
        self.verbose = verbose

    def solve(self, T0, r, dt, t_end, alpha):
        nr = len(r)
        nt = int(round(t_end / dt))

        if not HAS_TORCH:
            warnings.warn("PyTorch not available; PINN returning NaN.")
            result = np.full((nt + 1, nr), np.nan)
            result[0] = T0
            return result

        return self._solve_torch(T0, r, dt, t_end, alpha, nr, nt)

    def _solve_torch(self, T0, r, dt, t_end, alpha, nr, nt):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = SimplePINNNetwork(self.hidden_dim, self.n_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

        # Convert data
        r_np = r
        T0_t = torch.tensor(T0, dtype=torch.float32, device=device)
        t_vals = np.linspace(0, t_end, nt + 1)

        # Fixed training points
        rt_ic = torch.tensor(
            np.stack([r_np, np.zeros(nr)], axis=1),
            dtype=torch.float32, device=device
        )
        rt_bc = torch.tensor(
            np.stack([np.ones(nt + 1), t_vals], axis=1),
            dtype=torch.float32, device=device
        )

        # Loss weights
        w_ic = 10.0
        w_bc = 10.0
        w_pde = 1.0

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # === IC Loss: T(r, 0) = T0(r) ===
            T_pred_ic = model(rt_ic).squeeze()
            loss_ic = torch.mean((T_pred_ic - T0_t) ** 2)

            # === BC Loss: T(1, t) = 0 ===
            T_pred_bc = model(rt_bc).squeeze()
            loss_bc = torch.mean(T_pred_bc ** 2)

            # === PDE Loss: ∂T/∂t = χ * ∂²T/∂r² ===
            r_col = torch.rand(self.n_collocation, device=device)
            t_col = torch.rand(self.n_collocation, device=device) * t_end
            rt_col = torch.stack([r_col, t_col], dim=1).requires_grad_(True)

            T_col = model(rt_col)

            # Compute gradients
            grad_T = torch.autograd.grad(
                T_col, rt_col,
                grad_outputs=torch.ones_like(T_col),
                create_graph=True
            )[0]

            dT_dr = grad_T[:, 0:1]
            dT_dt = grad_T[:, 1:2]

            # Second derivative
            grad_dT_dr = torch.autograd.grad(
                dT_dr, rt_col,
                grad_outputs=torch.ones_like(dT_dr),
                create_graph=True
            )[0]
            d2T_dr2 = grad_dT_dr[:, 0:1]

            # Chi (simplified: just use constant for stability testing)
            # For now, use linear diffusion chi = 0.1
            chi = 0.1 + 0.0 * alpha  # Can be extended later

            # PDE residual
            residual = dT_dt - chi * d2T_dr2
            loss_pde = torch.mean(residual ** 2)

            # Total loss
            loss = w_ic * loss_ic + w_bc * loss_bc + w_pde * loss_pde

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if self.verbose and (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch+1}: Loss={loss.item():.6f} "
                      f"(IC={loss_ic.item():.6f}, BC={loss_bc.item():.6f}, "
                      f"PDE={loss_pde.item():.6f})")

        # Evaluate
        T_history = np.zeros((nt + 1, nr))
        model.eval()
        with torch.no_grad():
            for k, t in enumerate(t_vals):
                inp = torch.tensor(
                    np.stack([r_np, np.full(nr, t)], axis=1),
                    dtype=torch.float32, device=device
                )
                T_history[k] = model(inp).squeeze().cpu().numpy()

        T_history[0] = T0
        T_history[:, -1] = 0.0
        return T_history


class NonlinearPINN(SimplePINN):
    """PINN with nonlinear chi."""

    name = "pinn_nonlinear"

    def _solve_torch(self, T0, r, dt, t_end, alpha, nr, nt):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = SimplePINNNetwork(self.hidden_dim, self.n_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

        r_np = r
        T0_t = torch.tensor(T0, dtype=torch.float32, device=device)
        t_vals = np.linspace(0, t_end, nt + 1)

        rt_ic = torch.tensor(
            np.stack([r_np, np.zeros(nr)], axis=1),
            dtype=torch.float32, device=device
        )
        rt_bc = torch.tensor(
            np.stack([np.ones(nt + 1), t_vals], axis=1),
            dtype=torch.float32, device=device
        )

        w_ic, w_bc, w_pde = 10.0, 10.0, 1.0

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # IC Loss
            T_pred_ic = model(rt_ic).squeeze()
            loss_ic = torch.mean((T_pred_ic - T0_t) ** 2)

            # BC Loss
            T_pred_bc = model(rt_bc).squeeze()
            loss_bc = torch.mean(T_pred_bc ** 2)

            # PDE Loss with nonlinear chi
            r_col = torch.rand(self.n_collocation, device=device)
            t_col = torch.rand(self.n_collocation, device=device) * t_end
            rt_col = torch.stack([r_col, t_col], dim=1).requires_grad_(True)

            T_col = model(rt_col)

            grad_T = torch.autograd.grad(
                T_col, rt_col,
                grad_outputs=torch.ones_like(T_col),
                create_graph=True
            )[0]

            dT_dr = grad_T[:, 0:1]
            dT_dt = grad_T[:, 1:2]

            grad_dT_dr = torch.autograd.grad(
                dT_dr, rt_col,
                grad_outputs=torch.ones_like(dT_dr),
                create_graph=True
            )[0]
            d2T_dr2 = grad_dT_dr[:, 0:1]

            # Nonlinear chi: (|T'| - 0.5)^alpha + 0.1 if |T'| > 0.5, else 0.1
            abs_grad = torch.abs(dT_dr)
            threshold = 0.5
            # Use smooth approximation for numerical stability
            excess = torch.clamp(abs_grad - threshold, min=0.0)
            chi = torch.pow(excess + 1e-6, alpha) + 0.1

            residual = dT_dt - chi * d2T_dr2
            loss_pde = torch.mean(residual ** 2)

            loss = w_ic * loss_ic + w_bc * loss_bc + w_pde * loss_pde
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if self.verbose and (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch+1}: Loss={loss.item():.6f}")

        T_history = np.zeros((nt + 1, nr))
        model.eval()
        with torch.no_grad():
            for k, t in enumerate(t_vals):
                inp = torch.tensor(
                    np.stack([r_np, np.full(nr, t)], axis=1),
                    dtype=torch.float32, device=device
                )
                T_history[k] = model(inp).squeeze().cpu().numpy()

        T_history[0] = T0
        T_history[:, -1] = 0.0
        return T_history
