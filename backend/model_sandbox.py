"""Code execution environment for user-provided Python code."""

import collections
import math
import typing

import numpy
import torch
import torch.nn as nn
import torch.optim


def _get_globals() -> dict:
    return {
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "optim": torch.optim,
        "numpy": numpy,
        "np": numpy,
        "math": math,
        "OrderedDict": collections.OrderedDict,
        "List": typing.List,
        "Optional": typing.Optional,
        "Tuple": typing.Tuple,
    }


def exec_user_code(code: str, extra_globals: dict | None = None) -> dict:
    """Execute user code and return locals dict."""
    globs = _get_globals()
    if extra_globals:
        globs.update(extra_globals)

    locals_ = {}
    exec(compile(code, "<user_code>", "exec"), globs, locals_)
    return locals_


def _find_module_sublcass(locals_dict: dict):
    """Find the first nn.Module subclass in locals that isn't nn.Module itself."""
    for v in locals_dict.values():
        if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module:
            return v
    return None


def instantiate_model(code: str, model_name: str, input_size: int,
                      hidden_sizes: list[int], output_size: int) -> tuple:
    """Execute user code and instantiate the model.

    Returns (model, architecture_summary, param_count).
    """
    locals_ = exec_user_code(code)

    # First, check if 'model' was created directly in the code
    model = locals_.get("model")
    if model is not None and isinstance(model, nn.Module):
        param_count = sum(p.numel() for p in model.parameters())
        return model, str(model), param_count

    # Otherwise find the Module subclass
    model_cls = None
    if model_name:
        model_cls = locals_.get(model_name)
    if model_cls is None or not isinstance(model_cls, type):
        model_cls = _find_module_sublcass(locals_)

    if model_cls is None:
        raise ValueError(
            "No torch.nn.Module subclass found in code. "
            "Define a class inheriting from torch.nn.Module "
            "or create a 'model' variable."
        )

    # Try to instantiate with size args, fallback to no-args
    try:
        model = model_cls(input_size, hidden_sizes, output_size)
    except TypeError:
        try:
            model = model_cls()
        except TypeError as e:
            raise ValueError(
                f"Failed to instantiate model '{model_cls.__name__}'. "
                f"Tried constructor signatures: (input_size, hidden_sizes, output_size) and (). "
                f"Error: {e}"
            ) from e

    if not isinstance(model, nn.Module):
        raise ValueError(f"'{model_cls.__name__}' is not a torch.nn.Module")

    param_count = sum(p.numel() for p in model.parameters())
    return model, str(model), param_count
