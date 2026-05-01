"""WebSocket connection handler and message dispatcher."""

import asyncio
import json
import traceback

import torch
import torch.nn as nn
from aiohttp import web, WSMsgType

import backend.config as cfg
from backend.protocol import (
    VALID_REQUEST_TYPES,
    PUSH_TYPES,
    make_push,
    make_response,
    make_error,
    make_status,
    validate_message,
)
from backend.session import Session


active_sessions: dict[web.WebSocketResponse, Session] = {}
_remote_executor = None  # singleton RemoteExecutor, created on first connect_remote request


def _get_remote():
    global _remote_executor
    if _remote_executor is None:
        from backend.remote import RemoteExecutor
        _remote_executor = RemoteExecutor()
    return _remote_executor


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=50 * 1024 * 1024)  # 50MB max
    await ws.prepare(request)

    session = Session()
    active_sessions[ws] = session

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                await _handle_message(ws, session, msg.data)
            elif msg.type == WSMsgType.ERROR:
                print(f"WebSocket error: {ws.exception()}")
                break
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"WebSocket handler error: {e}")
    finally:
        if session.training_task and not session.training_task.done():
            session._stop_training_flag = True
        active_sessions.pop(ws, None)

    return ws


ROUTER = {
    "create_model": "_handle_create_model",
    "set_optimizer": "_handle_set_optimizer",
    "set_custom_optimizer": "_handle_set_custom_optimizer",
    "set_dataset": "_handle_set_dataset",
    "set_custom_dataset": "_handle_set_custom_dataset",
    "start_training": "_handle_start_training",
    "stop_training": "_handle_stop_training",
    "reset_model": "_handle_reset_model",
    "compute_hessian": "_handle_compute_hessian",
    "compute_hessian_eigenvalues": "_handle_compute_eigenvalues",
    "compute_pca_landscape": "_handle_compute_pca_landscape",
    "compute_random_landscape": "_handle_compute_random_landscape",
    "solve_newton_step": "_handle_solve_newton_step",
    "solve_linear_system": "_handle_solve_linear_system",
    "get_model_summary": "_handle_get_model_summary",
    "adapt_model": "_handle_adapt_model",
    "get_config": "_handle_get_config",
    "update_config": "_handle_update_config",
    "connect_remote": "_handle_connect_remote",
    "disconnect_remote": "_handle_disconnect_remote",
    "get_remote_status": "_handle_get_remote_status",
}


async def _handle_message(ws, session, raw_text):
    try:
        msg = json.loads(raw_text)
        validate_message(msg)
    except (json.JSONDecodeError, ValueError) as e:
        await ws.send_json(make_error(None, "INVALID_MESSAGE", str(e)))
        return

    msg_type = msg["type"]
    msg_id = msg.get("msg_id", "")
    payload = msg.get("payload", {})

    handler_name = ROUTER.get(msg_type)
    if handler_name is None:
        await ws.send_json(make_error(msg_id, "UNKNOWN_TYPE", f"Unknown message type: {msg_type}"))
        return

    try:
        handler = getattr(_Dispatcher, handler_name)
        response_payload = await handler(session, payload, ws)
        if response_payload is not None:
            if msg_type in VALID_REQUEST_TYPES:
                response_type = _get_response_type(msg_type)
                await ws.send_json(make_response(msg_id, response_type, response_payload))
    except ValueError as e:
        await ws.send_json(make_error(msg_id, "VALIDATION_ERROR", str(e)))
    except Exception as e:
        traceback.print_exc()
        await ws.send_json(make_error(msg_id, "INTERNAL_ERROR", str(e)))


def _get_response_type(request_type):
    mapping = {
        "create_model": "model_created",
        "set_dataset": "dataset_ready",
        "set_custom_dataset": "dataset_ready",
        "start_training": "response",
        "stop_training": "response",
        "reset_model": "response",
        "compute_hessian": "hessian_computed",
        "compute_hessian_eigenvalues": "hessian_eigenvalues",
        "compute_pca_landscape": "landscape_computed",
        "compute_random_landscape": "landscape_computed",
        "solve_newton_step": "equation_solved",
        "solve_linear_system": "equation_solved",
        "get_model_summary": "model_summary",
        "adapt_model": "model_adapted",
    }
    return mapping.get(request_type, "response")


def _replace_module(model, name, new_module):
    """Replace a submodule given its dotted attr name (e.g. 'net.fc0')."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _ensure_loss_fn(session):
    if session.loss_fn is None:
        session.loss_fn = nn.CrossEntropyLoss() if session.task_type == "classification" else nn.MSELoss()


class _Dispatcher:
    """Stub dispatcher methods — filled in as modules are built."""

    @staticmethod
    async def _handle_create_model(session, payload, ws):
        from backend.model_sandbox import instantiate_model
        code = payload.get("code", "")
        model_name = payload.get("model_name", "")
        input_size = payload.get("input_size", cfg.DEFAULT_INPUT_SIZE)
        hidden_sizes = payload.get("hidden_sizes", cfg.DEFAULT_HIDDEN_SIZES)
        output_size = payload.get("output_size", cfg.DEFAULT_OUTPUT_SIZE)

        model, arch_summary, param_count, warning = instantiate_model(code, model_name, input_size, hidden_sizes, output_size)

        if param_count > cfg.HARD_PARAM_LIMIT:
            raise ValueError(f"Model has {param_count} params, exceeds limit of {cfg.HARD_PARAM_LIMIT}")

        # Move model to configured device
        device = torch.device(cfg.DEVICE)
        model = model.to(device)

        session.model = model
        session.model_code = code
        session._param_count = param_count
        session.invalidate_cache()

        if warning:
            await ws.send_json(make_status("warning", warning))

        param_shapes = {name: list(p.shape) for name, p in model.named_parameters()}

        return {
            "model_name": model.__class__.__name__,
            "num_parameters": param_count,
            "num_trainable": param_count,
            "parameter_shapes": param_shapes,
            "architecture": arch_summary,
            "device": cfg.DEVICE,
        }

    @staticmethod
    async def _handle_set_optimizer(session, payload, ws):
        if session.model is None:
            raise ValueError("Create a model first")

        opt_name = payload.get("optimizer", "Adam")
        params = payload.get("params", {})

        optimizer_cls = getattr(torch.optim, opt_name, None)
        if optimizer_cls is None:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        session.optimizer = optimizer_cls(session.model.parameters(), **params)

        return {
            "status": "ok",
            "optimizer": opt_name,
            "param_count": session.param_count,
            "config": params,
        }

    @staticmethod
    async def _handle_set_custom_optimizer(session, payload, ws):
        if session.model is None:
            raise ValueError("Create a model first")

        from backend.model_sandbox import exec_user_code
        code = payload.get("code", "")
        extra_globals = {"model": session.model, "torch": torch}

        locals_ = exec_user_code(code, extra_globals)
        optimizer = locals_.get("optimizer")
        if optimizer is None:
            raise ValueError("Custom optimizer code must define an 'optimizer' variable")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("'optimizer' must be a torch.optim.Optimizer instance")

        session.optimizer = optimizer
        session.invalidate_cache()

        return {
            "status": "ok",
            "optimizer": type(optimizer).__name__,
            "custom": True,
        }

    @staticmethod
    async def _handle_set_dataset(session, payload, ws):
        from backend.datasets import load_dataset
        ds_name = payload.get("dataset", "mnist")
        params = payload.get("params", {})
        params.setdefault("batch_size", cfg.DEFAULT_BATCH_SIZE)

        result = load_dataset(ds_name, params)
        session.train_loader = result["train_loader"]
        session.test_loader = result["test_loader"]
        session.dataset_info = result
        session.task_type = result.get("task", "classification")
        session.input_size = result.get("input_size", cfg.DEFAULT_INPUT_SIZE)
        session.output_size = result.get("num_classes", cfg.DEFAULT_OUTPUT_SIZE)
        session.invalidate_cache()

        return {k: v for k, v in result.items() if k not in ("train_loader", "test_loader")}

    @staticmethod
    async def _handle_set_custom_dataset(session, payload, ws):
        from backend.model_sandbox import exec_user_code
        code = payload.get("code", "")
        batch_size = payload.get("batch_size", cfg.DEFAULT_BATCH_SIZE)
        task = payload.get("task", "classification")

        extra_globals = {"torch": torch, "np": __import__("numpy")}
        locals_ = exec_user_code(code, extra_globals)

        dataset_cls = None
        for v in locals_.values():
            if isinstance(v, type) and issubclass(v, torch.utils.data.Dataset) and v is not torch.utils.data.Dataset:
                dataset_cls = v
                break

        if dataset_cls is None:
            ds = locals_.get("dataset")
            if ds is not None and isinstance(ds, torch.utils.data.Dataset):
                dataset_cls = ds
            else:
                raise ValueError("Custom dataset code must define a torch.utils.data.Dataset subclass or 'dataset' variable")

        if isinstance(dataset_cls, type):
            ds = dataset_cls()
        else:
            ds = dataset_cls

        total = len(ds)
        split = int(cfg.DEFAULT_TRAIN_SPLIT * total)
        train_ds = torch.utils.data.Subset(ds, range(split))
        test_ds = torch.utils.data.Subset(ds, range(split, total))

        session.train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        session.test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        session.task_type = task
        session.invalidate_cache()

        # Infer sizes from a sample
        sample_x, sample_y = next(iter(session.train_loader))
        input_size = sample_x.view(sample_x.size(0), -1).size(1)
        if task == "classification":
            num_classes = len(torch.unique(sample_y))
        else:
            num_classes = sample_y.size(1) if sample_y.ndim > 1 else 1

        session.input_size = input_size
        session.output_size = num_classes

        return {
            "dataset_name": "custom",
            "train_samples": len(train_ds),
            "test_samples": len(test_ds),
            "input_shape": list(sample_x.shape[1:]),
            "input_size": input_size,
            "num_classes": num_classes,
            "batch_size": batch_size,
            "task": task,
        }

    @staticmethod
    async def _handle_start_training(session, payload, ws):
        if session.model is None:
            raise ValueError("Create a model first")
        if session.optimizer is None:
            raise ValueError("Configure an optimizer first")
        if session.train_loader is None:
            raise ValueError("Set a dataset first")

        if cfg.REMOTE_ENABLED and _get_remote().connected:
            session._stop_training_flag = False
            session.training_task = asyncio.ensure_future(_run_remote_training(session, payload, ws))
        else:
            from backend.training import run_training
            session._stop_training_flag = False
            session.training_task = asyncio.ensure_future(run_training(session, payload, ws))
        return {"status": "started"}

    @staticmethod
    async def _handle_stop_training(session, payload, ws):
        session._stop_training_flag = True
        return {"status": "stopping"}

    @staticmethod
    async def _handle_reset_model(session, payload, ws):
        if session.training_task and not session.training_task.done():
            session._stop_training_flag = True
        session.model = None
        session.optimizer = None
        session.loss_fn = None
        session.train_loader = None
        session.test_loader = None
        session.dataset_info = None
        session.invalidate_cache()
        session.loss_history.clear()
        session.accuracy_history.clear()
        session.param_snapshots.clear()
        return {"status": "reset"}

    @staticmethod
    async def _handle_compute_hessian(session, payload, ws):
        if session.model is None:
            raise ValueError("Create a model first")
        _ensure_loss_fn(session)

        from backend.hessian import compute_full_hessian, compute_diagonal_hessian, hessian_to_display_matrix

        use_diag = payload.get("use_diagonal_approx", False)
        sample_batches = payload.get("sample_batches", 1)
        force = payload.get("force_compute", False)

        if session._cached_hessian is not None and not force:
            H, is_diag = session._cached_hessian
        else:
            n = session.param_count
            if n > cfg.MAX_PARAM_COUNT_DIAGONAL and not use_diag:
                await ws.send_json(make_status("warning",
                    f"Model has {n} parameters. Full Hessian would require significant memory. Using diagonal approximation."))
                use_diag = True

            if cfg.REMOTE_ENABLED and _get_remote().connected:
                loop = asyncio.get_event_loop()
                remote = _get_remote()
                H, is_diag = await loop.run_in_executor(
                    None, remote.compute_hessian, session, use_diag, sample_batches)
                H = H.to(torch.device(cfg.DEVICE))
            elif use_diag:
                H = compute_diagonal_hessian(session)
                is_diag = True
            else:
                H = compute_full_hessian(session, max_batches=sample_batches)
                is_diag = False
            session._cached_hessian = (H, is_diag)

        # Move Hessian to CPU for display / JSON serialization
        H_cpu = H.cpu() if H.ndim > 1 else H.cpu()
        display_matrix, dim_labels = hessian_to_display_matrix(H_cpu, is_diag, session.model)

        return {
            "num_parameters": session.param_count,
            "is_diagonal": is_diag,
            "hessian_matrix": display_matrix.tolist() if hasattr(display_matrix, 'tolist') else display_matrix,
            "hessian_shape": list(display_matrix.shape),
            "display_type": "block_averaged" if display_matrix.shape[0] < session.param_count else "full",
            "dim_labels": dim_labels,
        }

    @staticmethod
    async def _handle_compute_eigenvalues(session, payload, ws):
        if session._cached_hessian is None:
            raise ValueError("Compute Hessian first")

        H, is_diag = session._cached_hessian
        method = payload.get("method", "exact" if not is_diag else "diagonal")

        if cfg.REMOTE_ENABLED and _get_remote().connected:
            loop = asyncio.get_event_loop()
            remote = _get_remote()
            result = await loop.run_in_executor(None, remote.compute_eigenvalues, session, method)
        else:
            from backend.hessian import compute_eigenvalues as compute_ev
            result = compute_ev(H.cpu() if H is not None else H, method, is_diag)

        session._cached_eigenvalues = result
        return result

    @staticmethod
    async def _handle_compute_pca_landscape(session, payload, ws):
        _ensure_loss_fn(session)
        resolution = min(payload.get("grid_resolution", 30), 50)
        range_factor = payload.get("range_factor", 2.0)

        if cfg.REMOTE_ENABLED and _get_remote().connected:
            loop = asyncio.get_event_loop()
            remote = _get_remote()
            return await loop.run_in_executor(
                None, lambda: remote.compute_landscape_sync(session, resolution, range_factor, "pca", None))

        from backend.landscape import compute_pca_landscape
        return await compute_pca_landscape(session, resolution, range_factor, ws)

    @staticmethod
    async def _handle_compute_random_landscape(session, payload, ws):
        _ensure_loss_fn(session)
        resolution = min(payload.get("grid_resolution", 30), 50)
        range_factor = payload.get("range_factor", 2.0)
        seed = payload.get("seed", None)

        if cfg.REMOTE_ENABLED and _get_remote().connected:
            loop = asyncio.get_event_loop()
            remote = _get_remote()
            return await loop.run_in_executor(
                None, lambda: remote.compute_landscape_sync(session, resolution, range_factor, "random", seed))

        from backend.landscape import compute_random_landscape
        return await compute_random_landscape(session, resolution, range_factor, seed, ws)

    @staticmethod
    async def _handle_solve_newton_step(session, payload, ws):
        _ensure_loss_fn(session)
        reg = payload.get("regularization", cfg.DEFAULT_REGULARIZATION)
        apply_step = payload.get("apply_step", True)
        step_scale = payload.get("step_scale", cfg.DEFAULT_STEP_SCALE)

        if cfg.REMOTE_ENABLED and _get_remote().connected:
            loop = asyncio.get_event_loop()
            remote = _get_remote()
            return await loop.run_in_executor(
                None, remote.solve_newton, session, reg, apply_step, step_scale)

        from backend.equations import solve_newton
        return solve_newton(session, reg, apply_step, step_scale, ws)

    @staticmethod
    async def _handle_solve_linear_system(session, payload, ws):
        _ensure_loss_fn(session)
        if cfg.REMOTE_ENABLED and _get_remote().connected:
            raise ValueError("Linear system solving not yet supported in remote mode")

        from backend.equations import solve_linear
        rhs = payload.get("right_hand_side", None)
        reg = payload.get("regularization", 0.0)
        return solve_linear(session, rhs, reg, ws)

    @staticmethod
    async def _handle_get_model_summary(session, payload, ws):
        if session.model is None:
            raise ValueError("No model created")
        return {
            "model_name": session.model.__class__.__name__,
            "num_parameters": session.param_count,
            "architecture": str(session.model),
            "task_type": session.task_type,
            "has_optimizer": session.optimizer is not None,
            "has_dataset": session.dataset_info is not None,
            "training_progress": {
                "epochs_completed": len(session.loss_history),
                "snapshots_saved": len(session.param_snapshots),
            },
        }

    @staticmethod
    async def _handle_adapt_model(session, payload, ws):
        if session.model is None:
            raise ValueError("Create a model first")

        mode = payload.get("mode", "expand_matrices")
        new_input_size = int(payload.get("input_size", 0))
        new_output_size = int(payload.get("output_size", 0))

        if new_input_size <= 0 or new_output_size <= 0:
            raise ValueError("input_size and output_size must be positive")

        # Collect Linear layers in traversal order
        linear_layers = []
        for mname, mod in session.model.named_modules():
            if isinstance(mod, nn.Linear):
                linear_layers.append((mname, mod))

        if not linear_layers:
            raise ValueError("Model has no Linear layers to adapt")

        first_name, first_layer = linear_layers[0]
        last_name, last_layer = linear_layers[-1]
        same_layer = (first_name == last_name)

        old_input_size = first_layer.in_features
        old_output_size = last_layer.out_features

        if mode == "reset_first_last":
            if same_layer:
                new_mod = nn.Linear(new_input_size, new_output_size,
                                    bias=first_layer.bias is not None)
                new_mod = new_mod.to(first_layer.weight.device)
                _replace_module(session.model, first_name, new_mod)
            else:
                new_first = nn.Linear(new_input_size, first_layer.out_features,
                                      bias=first_layer.bias is not None)
                new_first = new_first.to(first_layer.weight.device)
                _replace_module(session.model, first_name, new_first)

                new_last = nn.Linear(last_layer.in_features, new_output_size,
                                     bias=last_layer.bias is not None)
                new_last = new_last.to(last_layer.weight.device)
                _replace_module(session.model, last_name, new_last)

        elif mode == "expand_matrices":
            if same_layer:
                # Single Linear layer: grow both columns and rows at once
                new_mod = nn.Linear(new_input_size, new_output_size,
                                    bias=first_layer.bias is not None)
                new_mod = new_mod.to(first_layer.weight.device)
                copy_cols = min(old_input_size, new_input_size)
                copy_rows = min(old_output_size, new_output_size)
                new_mod.weight.data[:copy_rows, :copy_cols] = first_layer.weight.data[:copy_rows, :copy_cols]
                if first_layer.bias is not None:
                    new_mod.bias.data[:copy_rows] = first_layer.bias.data[:copy_rows]
                _replace_module(session.model, first_name, new_mod)
            else:
                # --- first layer: grow columns to match new_input_size ---
                if new_input_size != old_input_size:
                    new_first = nn.Linear(new_input_size, first_layer.out_features,
                                          bias=first_layer.bias is not None)
                    new_first = new_first.to(first_layer.weight.device)
                    copy_cols = min(old_input_size, new_input_size)
                    new_first.weight.data[:, :copy_cols] = first_layer.weight.data[:, :copy_cols]
                    if first_layer.bias is not None:
                        new_first.bias.data = first_layer.bias.data
                    _replace_module(session.model, first_name, new_first)

                # --- last layer: grow rows to match new_output_size ---
                if new_output_size != old_output_size:
                    new_last = nn.Linear(last_layer.in_features, new_output_size,
                                         bias=last_layer.bias is not None)
                    new_last = new_last.to(last_layer.weight.device)
                    copy_rows = min(old_output_size, new_output_size)
                    new_last.weight.data[:copy_rows, :] = last_layer.weight.data[:copy_rows, :]
                    if last_layer.bias is not None:
                        new_last.bias.data[:copy_rows] = last_layer.bias.data[:copy_rows]
                    _replace_module(session.model, last_name, new_last)

        else:
            raise ValueError(f"Unknown adapt mode: {mode}")

        session.count_params()
        session.invalidate_cache()
        session.input_size = new_input_size
        session.output_size = new_output_size
        session.optimizer = None  # stale param refs after module replacement

        return {
            "status": "adapted",
            "mode": mode,
            "num_parameters": session.param_count,
            "old_input_size": old_input_size,
            "new_input_size": new_input_size,
            "old_output_size": old_output_size,
            "new_output_size": new_output_size,
        }

    @staticmethod
    async def _handle_get_config(session, payload, ws):
        return cfg.get_runtime_config()

    @staticmethod
    async def _handle_update_config(session, payload, ws):
        updates = payload.get("updates", {})
        if not updates:
            raise ValueError("No config updates provided")
        return cfg.update_runtime_config(updates)

    @staticmethod
    async def _handle_connect_remote(session, payload, ws):
        loop = asyncio.get_event_loop()
        remote = _get_remote()
        if remote.connected:
            await loop.run_in_executor(None, remote.disconnect)
        msg = await loop.run_in_executor(None, remote.connect)
        return {"status": "connected", "message": msg}

    @staticmethod
    async def _handle_disconnect_remote(session, payload, ws):
        loop = asyncio.get_event_loop()
        remote = _get_remote()
        if remote.connected:
            await loop.run_in_executor(None, remote.disconnect)
        return {"status": "disconnected"}

    @staticmethod
    async def _handle_get_remote_status(session, payload, ws):
        remote = _get_remote()
        return {
            "connected": remote.connected,
            "remote_device": remote.remote_device,
        }


async def _run_remote_training(session, payload, ws):
    """Run training remotely with progress push messages."""
    remote = _get_remote()
    try:
        result = remote.run_training(session, payload)

        # Build a simplified training_complete response
        r = result.get("result", result)
        loss_history = r.get("loss_history", [])
        final_loss = r.get("final_loss", 0.0)

        await ws.send_json(make_push("training_complete", {
            "final_loss": final_loss,
            "final_train_accuracy": None,
            "final_test_accuracy": None,
            "loss_history": loss_history,
            "accuracy_history": [],
            "param_snapshots_saved": len(session.param_snapshots),
            "total_epochs_completed": payload.get("epochs", 5),
            "total_batches_completed": 0,
            "elapsed_seconds": 0,
        }))
    except Exception as e:
        from backend.protocol import make_error
        await ws.send_json(make_error(None, "TRAINING_FAILED", str(e)))
    finally:
        session._stop_training_flag = False
