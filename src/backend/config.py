"""Configuration for Hessian Playground — values are mutable at runtime."""

# Parameter limits
MAX_PARAM_COUNT_WARN = 10_000
MAX_PARAM_COUNT_DIAGONAL = 10_000
HARD_PARAM_LIMIT = 1_000_000

# Hessian display
HESSIAN_DISPLAY_MAX_SIZE = 200

# Landscape
MAX_GRID_RESOLUTION = 50
MIN_SNAPSHOTS_FOR_PCA = 3

# Sandbox
SANDBOX_TIMEOUT = 5

# Training
DEFAULT_BATCH_SIZE = 64
TRAINING_STATUS_INTERVAL = 0.5

# Server (require restart to take effect)
DEFAULT_PORT = 8080
DEFAULT_HOST = "127.0.0.1"
MAX_CONNECTIONS = 10

# Data
DATASET_CACHE_DIR = "./data"

# Keys that can be changed at runtime via update_config
_RUNTIME_CONFIG_KEYS = {
    "MAX_PARAM_COUNT_WARN",
    "MAX_PARAM_COUNT_DIAGONAL",
    "HARD_PARAM_LIMIT",
    "HESSIAN_DISPLAY_MAX_SIZE",
    "MAX_GRID_RESOLUTION",
    "MIN_SNAPSHOTS_FOR_PCA",
    "SANDBOX_TIMEOUT",
    "DEFAULT_BATCH_SIZE",
    "TRAINING_STATUS_INTERVAL",
}

_DEFAULTS = {k: None for k in _RUNTIME_CONFIG_KEYS}
for _k in _RUNTIME_CONFIG_KEYS:
    _DEFAULTS[_k] = globals()[_k]


def get_runtime_config():
    """Return all runtime-changeable config values."""
    return {k: globals()[k] for k in _RUNTIME_CONFIG_KEYS}


def update_runtime_config(updates):
    """Update runtime-changeable config values. Returns the new full config."""
    for k, v in updates.items():
        if k not in _RUNTIME_CONFIG_KEYS:
            continue
        default_val = _DEFAULTS[k]
        if isinstance(default_val, bool):
            globals()[k] = bool(v)
        elif isinstance(default_val, int):
            globals()[k] = int(v)
        elif isinstance(default_val, float):
            globals()[k] = float(v)
        else:
            globals()[k] = v
    return get_runtime_config()


def reset_runtime_config():
    """Reset all runtime config values to defaults."""
    for k, v in _DEFAULTS.items():
        globals()[k] = v
    return get_runtime_config()
