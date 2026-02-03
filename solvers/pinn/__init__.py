"""PINN solvers for heat transport equation.

Available solvers:
- PINNStub: Basic stub (returns NaN without PyTorch)
- ImprovedPINN: Full PINN with PDE residual loss
- AdaptivePINN: PINN with adaptive collocation sampling
- TransferPINN: Pre-train on α=0, fine-tune on target
- CurriculumPINN: Gradually increase difficulty
- EnsemblePINN: Average multiple PINN predictions
- FNOPINN: Fourier Neural Operator inspired architecture
"""

from solvers.pinn.stub import PINNStub

__all__ = ["PINNStub"]

try:
    from solvers.pinn.simple import SimplePINN, NonlinearPINN
    __all__.extend(["SimplePINN", "NonlinearPINN"])
except ImportError:
    pass

try:
    from solvers.pinn.improved import ImprovedPINN, AdaptivePINN
    __all__.extend(["ImprovedPINN", "AdaptivePINN"])
except ImportError:
    pass

try:
    from solvers.pinn.variants import (
        TransferPINN, CurriculumPINN, EnsemblePINN, FNOPINN
    )
    __all__.extend(["TransferPINN", "CurriculumPINN", "EnsemblePINN", "FNOPINN"])
except ImportError:
    pass
