"""Per-connection session state management."""

import torch
import torch.nn as nn

import backend.config as cfg


class Session:
    def __init__(self):
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.gradient_ascent: bool = False
        self.loss_fn: callable | None = None

        self.train_loader = None
        self.test_loader = None
        self.dataset_info: dict | None = None

        self.training_task = None
        self._stop_training_flag = False

        self.loss_history: list[float] = []
        self.accuracy_history: list[float] = []
        self.param_snapshots: list[dict] = []

        self._cached_hessian: dict | None = None  # {"type": ..., "data": ..., "param_count": ..., "memory_mb": ...}
        self._cached_gradient: torch.Tensor | None = None
        self._cached_eigenvalues: dict | None = None
        self._cached_ntk: dict | None = None
        self._cached_ntk_eigenvalues: dict | None = None

        self.sandbox_globals = {}
        self.sandbox_locals = {}

        self.input_size: int | None = None
        self.output_size: int | None = None
        self.task_type: str = "classification"

        self._param_count = 0

        self._model_code: str = ""

    @property
    def device(self) -> torch.device:
        return torch.device(cfg.DEVICE)

    @property
    def model_code(self) -> str:
        return self._model_code

    @model_code.setter
    def model_code(self, code: str):
        self._model_code = code

    def invalidate_cache(self):
        self._cached_hessian = None
        self._cached_gradient = None
        self._cached_eigenvalues = None
        self._cached_ntk = None
        self._cached_ntk_eigenvalues = None

    @property
    def param_count(self):
        return self._param_count

    def count_params(self):
        if self.model is None:
            self._param_count = 0
        else:
            self._param_count = sum(p.numel() for p in self.model.parameters())
        return self._param_count

    def get_flattened_params(self) -> torch.Tensor:
        """Return all model parameters as a single flat vector."""
        return torch.cat([p.data.view(-1).float() for p in self.model.parameters()])

    def get_flattened_gradients(self) -> torch.Tensor | None:
        """Return all parameter gradients as a single flat vector."""
        grads = []
        for p in self.model.parameters():
            if p.grad is None:
                return None
            grads.append(p.grad.data.view(-1).float())
        return torch.cat(grads)

    def set_flat_params(self, flat_params: torch.Tensor):
        """Set model parameters from a single flat vector."""
        from backend.utils import set_flat_params as _set_flat
        _set_flat(self.model, flat_params)

    def save_snapshot(self):
        """Deep-copy current model parameters."""
        self.param_snapshots.append(
            {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        )

    def get_snapshot_flat(self, idx: int) -> torch.Tensor | None:
        """Return a snapshot as a flat parameter vector."""
        if idx >= len(self.param_snapshots):
            return None
        sd = self.param_snapshots[idx]
        return torch.cat([v.float().view(-1) for v in sd.values()])

    def set_snapshot_params(self, idx: int):
        """Restore model parameters from a snapshot."""
        if idx >= len(self.param_snapshots):
            return
        self.model.load_state_dict(self.param_snapshots[idx])
