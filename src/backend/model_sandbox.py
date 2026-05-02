"""Code execution environment for user-provided Python code."""

import collections
import inspect
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
        "List": list,
        "Optional": typing.Optional,
        "Tuple": tuple,
    }


def exec_user_code(code: str, extra_globals: dict | None = None) -> dict:
    """Execute user code and return the resulting namespace dict.

    Uses a single dict for both globals and locals so that classes /
    functions defined in the code can reference each other at
    instantiation time (the class's __globals__ is the namespace).
    """
    namespace = _get_globals()
    if extra_globals:
        namespace.update(extra_globals)

    exec(compile(code, "<user_code>", "exec"), namespace)
    return namespace


def _find_model_class(locals_dict: dict) -> tuple:
    """Find the best nn.Module subclass to instantiate.

    Prefers a class whose __init__ accepts (input_size, hidden_sizes,
    output_size).  Falls back to the *last* nn.Module subclass defined
    (the one appearing last in the user's code), accompanied by a
    warning message.

    Returns (class_or_None, warning_or_None).
    """
    candidates = []
    for v in locals_dict.values():
        if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module:
            candidates.append(v)

    if not candidates:
        return None, None

    # Prefer classes whose __init__ explicitly accepts the three sizing
    # parameters we pass to the constructor.
    target = {"input_size", "hidden_sizes", "output_size"}
    for cls in candidates:
        try:
            sig = inspect.signature(cls.__init__)
            params = {
                name
                for name, p in sig.parameters.items()
                if name != "self"
                and p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
            if target.issubset(params):
                return cls, None
        except (ValueError, TypeError):
            pass

    # Fallback — last defined class (presumed to be the "main" model)
    fallback = candidates[-1]
    names = [c.__name__ for c in candidates]
    warning = (
        f"No model class with (input_size, hidden_sizes, output_size) "
        f"constructor found. Using '{fallback.__name__}' (last "
        f"nn.Module in code). Available classes: {names}"
    )
    return fallback, warning


def instantiate_model(code: str, model_name: str, input_size: int,
                      hidden_sizes: list[int], output_size: int) -> tuple:
    """Execute user code and instantiate the model.

    Returns (model, architecture_summary, param_count, warning_or_None).
    """
    warning = None
    locals_ = exec_user_code(code)

    # First, check if 'model' was created directly in the code
    model = locals_.get("model")
    if model is not None and isinstance(model, nn.Module):
        param_count = sum(p.numel() for p in model.parameters())
        return model, str(model), param_count, warning

    # Otherwise find the most suitable Module subclass
    model_cls = None
    if model_name:
        model_cls = locals_.get(model_name)
        if model_cls is not None and not isinstance(model_cls, type):
            model_cls = None

    if model_cls is None:
        model_cls, warning = _find_model_class(locals_)

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
            if warning is None:
                warning = (
                    f"'{model_cls.__name__}' does not accept "
                    f"(input_size, hidden_sizes, output_size); "
                    f"instantiated with no arguments."
                )
        except TypeError as e:
            raise ValueError(
                f"Failed to instantiate model '{model_cls.__name__}'. "
                f"Tried constructor signatures: (input_size, hidden_sizes, output_size) and (). "
                f"Error: {e}"
            ) from e

    if not isinstance(model, nn.Module):
        raise ValueError(f"'{model_cls.__name__}' is not a torch.nn.Module")

    param_count = sum(p.numel() for p in model.parameters())
    return model, str(model), param_count, warning
