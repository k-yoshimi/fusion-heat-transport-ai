"""PINN stub solver — uses PyTorch if available, otherwise returns NaN."""

import warnings
import numpy as np
from solvers.base import SolverBase

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PINNStub(SolverBase):
    """Physics-Informed Neural Network stub solver.

    If PyTorch is available, trains a tiny MLP for a few iterations.
    Otherwise returns NaN arrays with a warning.
    """

    name = "pinn_stub"

    def __init__(self, hidden: int = 32, epochs: int = 200, lr: float = 1e-3):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr

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
        # Tiny MLP: (r, t) -> T
        model = torch.nn.Sequential(
            torch.nn.Linear(2, self.hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, self.hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, 1),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        r_t = torch.tensor(r, dtype=torch.float32)
        T0_t = torch.tensor(T0, dtype=torch.float32)

        # Collocation points
        t_vals = np.linspace(0, t_end, nt + 1)
        rr, tt = np.meshgrid(r, t_vals)
        rt_col = torch.tensor(
            np.stack([rr.ravel(), tt.ravel()], axis=1), dtype=torch.float32
        )
        rt_col.requires_grad_(True)

        # Initial condition points
        rt_ic = torch.tensor(
            np.stack([r, np.zeros(nr)], axis=1), dtype=torch.float32
        )

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # IC loss
            T_pred_ic = model(rt_ic).squeeze()
            loss_ic = torch.mean((T_pred_ic - T0_t) ** 2)

            # BC loss: T(1, t) = 0
            rt_bc = torch.tensor(
                np.stack([np.ones(nt + 1), t_vals], axis=1), dtype=torch.float32
            )
            T_pred_bc = model(rt_bc).squeeze()
            loss_bc = torch.mean(T_pred_bc**2)

            loss = loss_ic + loss_bc
            loss.backward()
            optimizer.step()

        # Evaluate on grid
        T_history = np.zeros((nt + 1, nr))
        with torch.no_grad():
            for k, t in enumerate(t_vals):
                inp = torch.tensor(
                    np.stack([r, np.full(nr, t)], axis=1), dtype=torch.float32
                )
                T_history[k] = model(inp).squeeze().numpy()

        T_history[0] = T0
        T_history[:, -1] = 0.0  # enforce BC
        return T_history
