"""VanillaNet package: organized modules extracted from VanillaNetMatrix notebook.

Modules:
- device: device selection helpers
- utils: FFT operations and utility functions
- data: synthetic dataset generation
- metrics: loss and evaluation metrics
- model: VanillaModel definition
- neural_net: neural network training, evaluation, and visualization
- gs: Gerchberg–Saxton implementation
- grad_des: gradient descent optimization
- compare: multi-method comparison utilities
- debug: debugging utilities
"""

# # Import all modules
# from . import device
# from . import utils
# from . import data
# from . import metrics
# from . import model
# from . import neural_net
# from . import gs
# from . import grad_des
# from . import compare
# from . import debug

# # Import key functions and classes for easy access
# from .device import get_device
# from .data import DiscretePointsDataset
# from .model import VanillaModel
# from .metrics import (
#     calculate_inefficiency,
#     calculate_non_uniformity,
#     calculate_holography_metrics,
#     get_intensity_normalised_from_amplitude
# )
# from .utils import (
#     fftshift,
#     ifftshift,
#     w_theta_grid,
#     w_theta_grid_2d,
#     calculate_complex_ft,
#     inverse_complex_ft
# )
# from .gs import (
#     GerchbergSaxtonSolver,
#     gerchberg_saxton,
#     gerchberg_saxton_with_metrics,
#     test_gerchberg_saxton
# )
# from .grad_des import (
#     train_gd_optimization,
#     visualize_gd_results_with_metrics
# )
# from .neural_net import (
#     evaluate_model,
#     train_model,
#     visualize_results_with_metrics,
#     load_model
# )
# from .compare import (
#     compare_all_methods,
#     visualize_comparison,
#     print_comparison_summary
# )
# from .debug import debug_real_data_detailed

__all__ = [
    "device",
    "utils",
    "data",
    "metrics",
    "model",
    "gs",
    "grad_des",
    "compare",
    "debug",
    "neural_net",
]


### Where symbols are defined and used in your `vanillanet` package

# - device.get_device
#   - Defined in: `vanillanet/device.py`
#   - Used by: `vanillanet/neural_net.py` (and can be used by `gs.py`)

# - data.DiscretePointsDataset
#   - Defined in: `vanillanet/data.py`
#   - Used by: `vanillanet/grad_des.py` (imported inside `train_gd_optimization`), `vanillanet/gs.py` test helpers

# - model.VanillaModel
#   - Defined in: `vanillanet/model.py`
#   - Used by: `vanillanet/neural_net.py` (train/eval/load)

# - metrics
#   - `_normalize_intensity`
#     - Defined in: `vanillanet/metrics.py`
#     - Used by: `vanillanet/grad_des.py`
#   - `get_intensity_normalised_from_amplitude`
#     - Defined in: `vanillanet/metrics.py`
#     - Used by: `vanillanet/neural_net.py`, `vanillanet/compare.py`, `vanillanet/debug.py`, `vanillanet/gs.py`
#   - `calculate_inefficiency`, `calculate_non_uniformity`, `calculate_intensity_error`, `calculate_holography_metrics`
#     - Defined in: `vanillanet/metrics.py`
#     - Used by: `vanillanet/neural_net.py`, `vanillanet/grad_des.py`, `vanillanet/compare.py`, `vanillanet/gs.py`

# - utils
#   - `fftshift`, `ifftshift`
#     - Defined in: `vanillanet/utils.py`
#     - Used by: `vanillanet/grad_des.py`
#   - `calculate_complex_ft`, `inverse_complex_ft`, `inverse_complex_ft_grid`
#     - Defined in: `vanillanet/utils.py`
#     - Used by: `vanillanet/compare.py`, `vanillanet/gs.py`, `vanillanet/neural_net.py`
#   - `w_theta_grid`, `w_theta_grid_2d`
#     - Defined in: `vanillanet/utils.py`
#     - Used by: `vanillanet/neural_net.py` (`w_theta_grid`), `vanillanet/grad_des.py` (`w_theta_grid_2d`), `vanillanet/compare.py` (`w_theta_grid`)

# - gs
#   - `GerchbergSaxtonSolver`, `gerchberg_saxton`, `gerchberg_saxton_with_metrics`, `visualize_gs_results_multirow`, `test_gs_multiple_samples`, `test_gerchberg_saxton`
#     - Defined in: `vanillanet/gs.py`
#     - Used by: `vanillanet/compare.py` (`gerchberg_saxton_with_metrics`), notebooks (`test_gerchberg_saxton`)

# - grad_des
#   - `train_gd_optimization`, `visualize_gd_results_with_metrics`
#     - Defined in: `vanillanet/grad_des.py`
#     - Used by: `vanillanet/compare.py` (imported lazily inside `compare_all_methods` recommended)

# - neural_net
#   - `evaluate_model`, `train_model`, `visualize_results_with_metrics`, `load_model`
#     - Defined in: `vanillanet/neural_net.py`
#     - Used by: notebooks and scripts (e.g., `train_vanillanet.py`)

# - debug.debug_real_data_detailed
#   - Defined in: `vanillanet/debug.py`
#   - Used by: `vanillanet/neural_net.py` (optional), `vanillanet/gs.py` (optional)

# Quick way to check usages yourself:
# - Terminal grep:
#   - find usages of a symbol: `rg -n "symbol_name" vanillanet/`
#   - find imports of a module: `rg -n "^from \.module_name|^import module_name" vanillanet/`

# Summary:
# - I mapped each exported symbol to where it’s defined and which modules consume it.
# - Keep `__init__.py` minimal to avoid eager-import cycles; import cross-module symbols where used (lazy).
