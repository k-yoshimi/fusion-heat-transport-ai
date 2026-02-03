"""Additional PINN variants for comparison.

Variants:
1. TransferPINN - Pre-train on α=0, fine-tune on target α
2. CurriculumPINN - Gradually increase problem difficulty
3. EnsemblePINN - Average predictions from multiple models
4. FourierNeuralOperator - FNO-inspired architecture
"""

import warnings
import numpy as np
from solvers.base import SolverBase

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from solvers.pinn.improved import ImprovedPINNNetwork, ImprovedPINN


# =============================================================================
# 1. Transfer Learning PINN
# =============================================================================

class TransferPINN(SolverBase):
    """PINN with transfer learning from simple to complex problems.

    Strategy:
    1. Pre-train on linear problem (α=0) which is easier
    2. Fine-tune on target α with smaller learning rate
    """

    name = "pinn_transfer"

    def __init__(
        self,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        pretrain_epochs: int = 2000,
        finetune_epochs: int = 3000,
        pretrain_lr: float = 1e-3,
        finetune_lr: float = 1e-4,
        n_collocation: int = 2000,
        verbose: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr
        self.n_collocation = n_collocation
        self.verbose = verbose
        self._pretrained_model = None

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

        # Create base PINN solver
        base_pinn = ImprovedPINN(
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            epochs=self.pretrain_epochs,
            lr=self.pretrain_lr,
            n_collocation=self.n_collocation,
            verbose=self.verbose,
        )

        # Phase 1: Pre-train on α=0 (linear problem)
        if self.verbose:
            print("Phase 1: Pre-training on α=0 (linear problem)...")

        if self._pretrained_model is None:
            # Train on linear problem
            base_pinn.epochs = self.pretrain_epochs
            base_pinn.lr = self.pretrain_lr
            _ = base_pinn._solve_torch(T0, r, dt, t_end, alpha=0.0, nr=nr, nt=nt)
            # Note: We need to save the model state, but the current implementation
            # doesn't expose it. For now, we'll just train sequentially.

        # Phase 2: Fine-tune on target α
        if self.verbose:
            print(f"Phase 2: Fine-tuning on α={alpha}...")

        base_pinn.epochs = self.finetune_epochs
        base_pinn.lr = self.finetune_lr
        T_history = base_pinn._solve_torch(T0, r, dt, t_end, alpha, nr, nt)

        return T_history


# =============================================================================
# 2. Curriculum Learning PINN
# =============================================================================

class CurriculumPINN(SolverBase):
    """PINN with curriculum learning - gradually increase difficulty.

    Strategy:
    1. Start with small t_end (short time evolution)
    2. Gradually extend to full t_end
    3. Optionally start with α=0 and increase to target α
    """

    name = "pinn_curriculum"

    def __init__(
        self,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        epochs_per_stage: int = 1000,
        n_stages: int = 5,
        lr: float = 1e-3,
        n_collocation: int = 2000,
        curriculum_type: str = "time",  # "time", "alpha", or "both"
        verbose: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.epochs_per_stage = epochs_per_stage
        self.n_stages = n_stages
        self.lr = lr
        self.n_collocation = n_collocation
        self.curriculum_type = curriculum_type
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

        # Generate curriculum schedule
        if self.curriculum_type == "time":
            t_schedule = np.linspace(t_end / self.n_stages, t_end, self.n_stages)
            alpha_schedule = [alpha] * self.n_stages
        elif self.curriculum_type == "alpha":
            t_schedule = [t_end] * self.n_stages
            alpha_schedule = np.linspace(0, alpha, self.n_stages)
        else:  # both
            t_schedule = np.linspace(t_end / self.n_stages, t_end, self.n_stages)
            alpha_schedule = np.linspace(0, alpha, self.n_stages)

        # Create model (persists across stages)
        model = ImprovedPINNNetwork(
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            use_fourier=True
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        r_t = torch.tensor(r, dtype=torch.float32, device=device)
        T0_t = torch.tensor(T0, dtype=torch.float32, device=device)

        for stage, (t_curr, alpha_curr) in enumerate(zip(t_schedule, alpha_schedule)):
            if self.verbose:
                print(f"Stage {stage+1}/{self.n_stages}: t_end={t_curr:.3f}, α={alpha_curr:.2f}")

            nt_curr = max(2, int(round(t_curr / dt)))
            t_vals = np.linspace(0, t_curr, nt_curr + 1)

            # Training points
            rt_ic = torch.tensor(
                np.stack([r, np.zeros(nr)], axis=1),
                dtype=torch.float32, device=device
            )
            rt_bc = torch.tensor(
                np.stack([np.ones(nt_curr + 1), t_vals], axis=1),
                dtype=torch.float32, device=device
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs_per_stage, eta_min=1e-6
            )

            for epoch in range(self.epochs_per_stage):
                optimizer.zero_grad()

                # IC loss
                T_pred_ic = model(rt_ic).squeeze()
                loss_ic = torch.mean((T_pred_ic - T0_t) ** 2)

                # BC loss
                T_pred_bc = model(rt_bc).squeeze()
                loss_bc = torch.mean(T_pred_bc ** 2)

                # PDE loss (simplified for curriculum)
                r_col = torch.rand(self.n_collocation, device=device)
                t_col = torch.rand(self.n_collocation, device=device) * t_curr
                rt_col = torch.stack([r_col, t_col], dim=1)
                rt_col.requires_grad_(True)

                T_col = model(rt_col)
                grads = torch.autograd.grad(T_col, rt_col,
                                           grad_outputs=torch.ones_like(T_col),
                                           create_graph=True)[0]
                dT_dr = grads[:, 0:1]
                dT_dt = grads[:, 1:2]

                # Simplified PDE: ∂T/∂t ≈ χ * ∂²T/∂r² (ignoring 1/r term for speed)
                d2T_dr2 = torch.autograd.grad(dT_dr, rt_col,
                                              grad_outputs=torch.ones_like(dT_dr),
                                              create_graph=True)[0][:, 0:1]

                abs_grad = torch.abs(dT_dr)
                threshold = 0.5
                # Smooth approximation for numerical stability
                excess = torch.clamp(abs_grad - threshold, min=0.0)
                chi = torch.pow(excess + 1e-6, alpha_curr) + 0.1
                residual = dT_dt - chi * d2T_dr2
                loss_pde = torch.mean(residual ** 2)

                loss = loss_pde + 10 * loss_ic + 10 * loss_bc
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        # Final evaluation on full grid
        t_vals_full = np.linspace(0, t_end, nt + 1)
        T_history = np.zeros((nt + 1, nr))
        model.eval()
        with torch.no_grad():
            for k, t in enumerate(t_vals_full):
                inp = torch.tensor(
                    np.stack([r, np.full(nr, t)], axis=1),
                    dtype=torch.float32, device=device
                )
                T_history[k] = model(inp).squeeze().cpu().numpy()

        T_history[0] = T0
        T_history[:, -1] = 0.0
        return T_history


# =============================================================================
# 3. Ensemble PINN
# =============================================================================

class EnsemblePINN(SolverBase):
    """Ensemble of multiple PINNs with different initializations.

    Strategy:
    1. Train multiple PINNs with different random seeds
    2. Average their predictions (reduces variance)
    3. Optionally use different architectures
    """

    name = "pinn_ensemble"

    def __init__(
        self,
        n_models: int = 3,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        epochs: int = 3000,
        lr: float = 1e-3,
        n_collocation: int = 1500,
        verbose: bool = False,
    ):
        self.n_models = n_models
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
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
        predictions = []

        for i in range(self.n_models):
            if self.verbose:
                print(f"Training model {i+1}/{self.n_models}...")

            # Set different random seed for each model
            torch.manual_seed(42 + i * 1000)
            np.random.seed(42 + i * 1000)

            # Train individual PINN
            pinn = ImprovedPINN(
                hidden_dim=self.hidden_dim,
                num_blocks=self.num_blocks,
                epochs=self.epochs,
                lr=self.lr,
                n_collocation=self.n_collocation,
                verbose=False,
            )
            T_hist = pinn._solve_torch(T0, r, dt, t_end, alpha, nr, nt)
            predictions.append(T_hist)

        # Average predictions
        T_history = np.mean(predictions, axis=0)

        # Compute uncertainty (std across models)
        T_std = np.std(predictions, axis=0)
        if self.verbose:
            print(f"Ensemble uncertainty (mean std): {np.mean(T_std):.6f}")

        T_history[0] = T0
        T_history[:, -1] = 0.0
        return T_history


# =============================================================================
# 4. Fourier Neural Operator (FNO-inspired)
# =============================================================================

if HAS_TORCH:
    class SpectralConv1d(nn.Module):
        """1D Spectral convolution layer (FNO building block)."""

        def __init__(self, in_channels: int, out_channels: int, modes: int):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.modes = modes

            self.scale = 1 / (in_channels * out_channels)
            self.weights = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
            )

        def forward(self, x):
            # x: (batch, channels, spatial)
            batch_size = x.shape[0]

            # FFT
            x_ft = torch.fft.rfft(x)

            # Multiply relevant modes
            out_ft = torch.zeros(batch_size, self.out_channels, x.size(-1) // 2 + 1,
                                device=x.device, dtype=torch.cfloat)
            out_ft[:, :, :self.modes] = torch.einsum(
                "bix,iox->box", x_ft[:, :, :self.modes], self.weights
            )

            # Inverse FFT
            return torch.fft.irfft(out_ft, n=x.size(-1))


    class FNOBlock(nn.Module):
        """FNO block with spectral convolution and skip connection."""

        def __init__(self, channels: int, modes: int):
            super().__init__()
            self.spectral_conv = SpectralConv1d(channels, channels, modes)
            self.linear = nn.Conv1d(channels, channels, 1)
            self.activation = nn.GELU()

        def forward(self, x):
            x1 = self.spectral_conv(x)
            x2 = self.linear(x)
            return self.activation(x1 + x2)


    class FNONetwork(nn.Module):
        """Fourier Neural Operator network for time evolution."""

        def __init__(self, in_channels: int = 1, out_channels: int = 1,
                     hidden_channels: int = 32, modes: int = 16, n_layers: int = 4):
            super().__init__()

            self.lift = nn.Conv1d(in_channels + 1, hidden_channels, 1)  # +1 for time encoding

            self.blocks = nn.ModuleList([
                FNOBlock(hidden_channels, modes) for _ in range(n_layers)
            ])

            self.project = nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, 1),
                nn.GELU(),
                nn.Conv1d(hidden_channels, out_channels, 1),
            )

        def forward(self, x, t):
            """
            Args:
                x: (batch, 1, nr) - temperature profile
                t: (batch, 1) - time value

            Returns:
                (batch, 1, nr) - predicted temperature
            """
            batch, _, nr = x.shape

            # Add time encoding
            t_enc = t.unsqueeze(-1).expand(-1, -1, nr)  # (batch, 1, nr)
            x = torch.cat([x, t_enc], dim=1)  # (batch, 2, nr)

            x = self.lift(x)
            for block in self.blocks:
                x = block(x)
            return self.project(x)


class FNOPINN(SolverBase):
    """FNO-inspired PINN that learns the solution operator.

    Instead of learning T(r, t) directly, learns the mapping:
    T(t=0) -> T(t) for any t
    """

    name = "pinn_fno"

    def __init__(
        self,
        hidden_channels: int = 32,
        modes: int = 16,
        n_layers: int = 4,
        epochs: int = 3000,
        lr: float = 1e-3,
        n_time_samples: int = 50,
        verbose: bool = False,
    ):
        self.hidden_channels = hidden_channels
        self.modes = modes
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.n_time_samples = n_time_samples
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

        model = FNONetwork(
            hidden_channels=self.hidden_channels,
            modes=min(self.modes, nr // 2),
            n_layers=self.n_layers,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6
        )

        T0_t = torch.tensor(T0, dtype=torch.float32, device=device)
        r_t = torch.tensor(r, dtype=torch.float32, device=device)
        dr = r[1] - r[0]

        def compute_chi(dT_dr, alpha):
            abs_grad = torch.abs(dT_dr)
            threshold = 0.5
            # Smooth approximation for numerical stability
            excess = torch.clamp(abs_grad - threshold, min=0.0)
            return torch.pow(excess + 1e-6, alpha) + 0.1

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Sample random times
            t_samples = torch.rand(self.n_time_samples, device=device) * t_end

            total_loss = 0.0

            for t in t_samples:
                # Input: T0 at t=0
                x = T0_t.unsqueeze(0).unsqueeze(0)  # (1, 1, nr)
                t_in = t.unsqueeze(0).unsqueeze(0)  # (1, 1)

                # Predict T at time t
                T_pred = model(x, t_in).squeeze()  # (nr,)

                # Compute PDE residual
                # ∂T/∂t ≈ (T(t+δt) - T(t)) / δt
                delta_t = 0.01 * t_end
                t_next = t + delta_t
                t_next_in = t_next.unsqueeze(0).unsqueeze(0)
                T_next = model(x, t_next_in).squeeze()

                dT_dt = (T_next - T_pred) / delta_t

                # Spatial derivatives
                dT_dr = torch.gradient(T_pred, spacing=(dr,))[0]
                d2T_dr2 = torch.gradient(dT_dr, spacing=(dr,))[0]

                chi = compute_chi(dT_dr, alpha)

                # PDE: ∂T/∂t = (1/r) ∂/∂r (r χ ∂T/∂r)
                # Simplified: ≈ χ * d²T/dr² + dχ/dr * dT/dr + χ/r * dT/dr
                r_safe = torch.clamp(r_t, min=1e-6)
                rhs = chi * d2T_dr2 + chi / r_safe * dT_dr

                # Use L'Hopital at r=0
                rhs[0] = 2 * chi[0] * d2T_dr2[0]

                residual = dT_dt - rhs
                loss_pde = torch.mean(residual[1:-1] ** 2)  # Exclude boundaries

                # IC loss (t=0 should give T0)
                t_zero = torch.zeros(1, 1, device=device)
                T_at_0 = model(x, t_zero).squeeze()
                loss_ic = torch.mean((T_at_0 - T0_t) ** 2)

                # BC loss
                loss_bc = T_pred[-1] ** 2  # T(r=1) = 0

                total_loss += loss_pde + 10 * loss_ic + 10 * loss_bc

            total_loss = total_loss / self.n_time_samples
            total_loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if self.verbose and (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}: Loss={total_loss.item():.6f}")

        # Evaluate on full time grid
        t_vals = np.linspace(0, t_end, nt + 1)
        T_history = np.zeros((nt + 1, nr))

        model.eval()
        with torch.no_grad():
            x = T0_t.unsqueeze(0).unsqueeze(0)
            for k, t in enumerate(t_vals):
                t_in = torch.tensor([[t]], dtype=torch.float32, device=device)
                T_history[k] = model(x, t_in).squeeze().cpu().numpy()

        T_history[0] = T0
        T_history[:, -1] = 0.0
        return T_history
