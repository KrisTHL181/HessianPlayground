"""WebSocket connection handler and message dispatcher."""

import asyncio
import json
import traceback

import torch
import torch.nn as nn
from aiohttp import WSMsgType, web

import backend.config as cfg
from backend.protocol import (
    VALID_REQUEST_TYPES,
    make_error,
    make_push,
    make_response,
    make_status,
    validate_message,
)
from backend.session import Session
from backend.utils import make_loss_fn

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
    "compute_ntk": "_handle_compute_ntk",
    "compute_ntk_eigenvalues": "_handle_compute_ntk_eigenvalues",
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
    "compute_weight_histogram": "_handle_compute_weight_histogram",
    "compute_gradient_stats": "_handle_compute_gradient_stats",
    "compute_activation_stats": "_handle_compute_activation_stats",
    "compute_layer_stats": "_handle_compute_layer_stats",
    "compute_fisher": "_handle_compute_fisher",
    "compute_fisher_eigenvalues": "_handle_compute_fisher_eigenvalues",
    "compute_interpolation": "_handle_compute_interpolation",
    "start_lr_test": "_handle_start_lr_test",
    "compute_gradient_noise_scale": "_handle_compute_gradient_noise_scale",
    "compute_sharpness_landscape": "_handle_compute_sharpness_landscape",
    "compute_spectral_density": "_handle_compute_spectral_density",
    "solve_natural_gradient": "_handle_solve_natural_gradient",
    "export_session": "_handle_export_session",
    "import_session": "_handle_import_session",
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
        "compute_ntk": "ntk_computed",
        "compute_ntk_eigenvalues": "ntk_eigenvalues",
        "compute_pca_landscape": "landscape_computed",
        "compute_random_landscape": "landscape_computed",
        "solve_newton_step": "equation_solved",
        "solve_linear_system": "equation_solved",
        "get_model_summary": "model_summary",
        "adapt_model": "model_adapted",
        "get_config": "response",
        "update_config": "response",
        "connect_remote": "response",
        "disconnect_remote": "response",
        "get_remote_status": "response",
        "compute_weight_histogram": "weight_histogram",
        "compute_gradient_stats": "gradient_stats",
        "compute_activation_stats": "activation_stats",
        "compute_layer_stats": "layer_stats",
        "compute_fisher": "fisher_computed",
        "compute_fisher_eigenvalues": "fisher_eigenvalues",
        "compute_interpolation": "interpolation_computed",
        "start_lr_test": "response",
        "compute_gradient_noise_scale": "gradient_noise_scale",
        "compute_sharpness_landscape": "landscape_computed",
        "compute_spectral_density": "spectral_density",
        "solve_natural_gradient": "natural_gradient_solved",
        "export_session": "session_exported",
        "import_session": "session_imported",
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
        session.loss_fn = make_loss_fn(session.task_type)


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
        gradient_ascent = payload.get("gradient_ascent", False)

        optimizer_cls = getattr(torch.optim, opt_name, None)
        if optimizer_cls is None:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        session.optimizer = optimizer_cls(session.model.parameters(), **params)
        session.gradient_ascent = gradient_ascent
        session.invalidate_cache()

        return {
            "status": "ok",
            "optimizer": opt_name,
            "param_count": session.param_count,
            "config": params,
            "gradient_ascent": gradient_ascent,
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
        session.gradient_ascent = payload.get("gradient_ascent", False)
        session.invalidate_cache()

        return {
            "status": "ok",
            "optimizer": type(optimizer).__name__,
            "custom": True,
            "gradient_ascent": session.gradient_ascent,
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
    async def _handle_export_session(session, payload, ws):
        import json
        data = {
            "model_code": session.model_code if session.model else "",
            "dataset_info": session.dataset_info,
            "task_type": session.task_type,
            "input_size": session.input_size,
            "output_size": session.output_size,
            "loss_history": [float(x) for x in session.loss_history],
            "accuracy_history": [float(x) for x in session.accuracy_history],
            "num_snapshots": len(session.param_snapshots),
            "has_hessian": session._cached_hessian is not None,
            "has_ntk": session._cached_ntk is not None,
            "has_fisher": session._cached_fisher is not None,
        }
        if session.dataset_info:
            data["dataset_info"] = {k: v for k, v in session.dataset_info.items()
                                    if k not in ("train_loader", "test_loader")}

        # Serialize snapshots (weight tensors -> lists)
        if payload.get("include_snapshots", False) and session.param_snapshots:
            snap_list = []
            for sd in session.param_snapshots:
                snap_list.append({k: v.tolist() for k, v in sd.items()})
            data["snapshots"] = snap_list

        return {"data": json.dumps(data)}

    @staticmethod
    async def _handle_import_session(session, payload, ws):
        import json
        data = json.loads(payload.get("data", "{}"))

        if data.get("model_code"):
            # Execute model code
            from backend.model_sandbox import instantiate_model
            in_size = data.get("input_size", 784)
            out_size = data.get("output_size", 10)
            hidden_sizes = [128, 64]

            model, arch, pcount, warn = instantiate_model(
                data["model_code"], "", in_size, hidden_sizes, out_size
            )
            model = model.to(session.device)
            session.model = model
            session.model_code = data["model_code"]
            session._param_count = pcount
            session.input_size = in_size
            session.output_size = out_size
            session.task_type = data.get("task_type", "classification")

        session.loss_history = [float(x) for x in data.get("loss_history", [])]
        session.accuracy_history = [float(x) for x in data.get("accuracy_history", [])]

        return {"status": "imported", "model_restored": data.get("model_code", "") != ""}

    @staticmethod
    async def _handle_compute_spectral_density(session, payload, ws):
        if session.model is None:
            raise ValueError("Create a model first")
        _ensure_loss_fn(session)

        from backend.hessian import compute_spectral_density_kpm
        num_moments = min(payload.get("num_moments", 50), 100)
        num_vectors = min(payload.get("num_vectors", 1), 3)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, compute_spectral_density_kpm, session, num_moments, num_vectors)

    @staticmethod
    async def _handle_compute_sharpness_landscape(session, payload, ws):
        _ensure_loss_fn(session)
        resolution = min(payload.get("grid_resolution", 30), 50)
        range_factor = payload.get("range_factor", 2.0)
        from backend.landscape import compute_sharpness_landscape
        return await compute_sharpness_landscape(session, resolution, range_factor, ws)

    @staticmethod
    async def _handle_compute_gradient_noise_scale(session, payload, ws):
        if session.model is None or session.train_loader is None:
            raise ValueError("Model and dataset required")
        _ensure_loss_fn(session)
        from backend.training import compute_gradient_noise_scale
        batch_sizes = payload.get("batch_sizes", [16, 32, 64, 128, 256])
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, compute_gradient_noise_scale, session, batch_sizes)

    @staticmethod
    async def _handle_start_lr_test(session, payload, ws):
        if session.model is None:
            raise ValueError("Create a model first")
        if session.optimizer is None:
            raise ValueError("Configure an optimizer first")
        if session.train_loader is None:
            raise ValueError("Set a dataset first")

        from backend.training import run_lr_range_test
        session._stop_training_flag = False
        session.training_task = asyncio.ensure_future(run_lr_range_test(session, payload, ws))
        return {"status": "started"}

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

        from backend.hessian import (
            compute_block_diag_hessian,
            compute_diagonal_hessian,
            compute_full_hessian,
            compute_kfac,
            hessian_to_display_matrix,
            quantize_hessian,
        )

        method = payload.get("method", "auto")
        use_diag = payload.get("use_diagonal_approx", False)
        sample_batches = payload.get("sample_batches", 1)
        force = payload.get("force_compute", False)
        dtype_str = payload.get("dtype", "float32")
        dtype = cfg.HESSIAN_DTYPES.get(dtype_str, torch.float32)

        if method == "auto" and use_diag:
            method = "diagonal"

        n = session.param_count

        if method == "auto":
            if n <= 2000:
                method = "full"
            elif n <= 50000:
                method = "block_diag"
            else:
                method = "kfac"

        # Check cached result — invalidate if method changed
        if session._cached_hessian is not None and not force:
            if session._cached_hessian.get("type") != method:
                session.invalidate_cache()

        if session._cached_hessian is not None and not force:
            cached = session._cached_hessian
        elif method in ("full", "diagonal") and cfg.REMOTE_ENABLED and _get_remote().connected:
            loop = asyncio.get_event_loop()
            remote = _get_remote()
            H, is_diag = await loop.run_in_executor(
                None, remote.compute_hessian, session, method == "diagonal", sample_batches)
            H = H.to(torch.device(cfg.DEVICE))
            cached = {
                "type": "diagonal" if is_diag else "full",
                "data": H,
                "param_count": n,
                "memory_mb": H.numel() * H.element_size() / 1024 / 1024,
            }
        else:
            if method == "full":
                if n > cfg.MAX_PARAM_COUNT_DIAGONAL:
                    await ws.send_json(make_status("warning",
                        f"Model has {n} parameters. Full Hessian will use significant memory."))
                H = compute_full_hessian(session, max_batches=sample_batches)
                if dtype != torch.float32:
                    H = quantize_hessian(H, dtype)
                cached = {
                    "type": "full", "data": H,
                    "param_count": n,
                    "memory_mb": H.numel() * H.element_size() / 1024 / 1024,
                }
            elif method == "diagonal":
                H = compute_diagonal_hessian(session)
                cached = {
                    "type": "diagonal", "data": H,
                    "param_count": n,
                    "memory_mb": H.numel() * H.element_size() / 1024 / 1024,
                }
            elif method == "kfac":
                await ws.send_json(make_status("info", "Computing K-FAC Hessian approximation..."))
                kfac_data = compute_kfac(session, sample_batches=sample_batches, dtype=dtype)
                cached = {
                    "type": "kfac", "data": kfac_data,
                    "param_count": n,
                    "memory_mb": kfac_data.get("memory_mb", 0),
                }
            elif method == "block_diag":
                await ws.send_json(make_status("info", "Computing block-diagonal Hessian..."))
                bd_data = compute_block_diag_hessian(session, sample_batches=sample_batches, dtype=dtype)
                cached = {
                    "type": "block_diag", "data": bd_data,
                    "param_count": n,
                    "memory_mb": bd_data.get("memory_mb", 0),
                }
            else:
                raise ValueError(f"Unknown Hessian method: {method}")
            session._cached_hessian = cached

        display = hessian_to_display_matrix(cached, session.model)

        result = {
            "num_parameters": cached["param_count"],
            "is_diagonal": cached["type"] == "diagonal",
            "method": cached["type"],
            "hessian_matrix": display.get("hessian_matrix"),
            "hessian_shape": display.get("hessian_shape", []),
            "display_type": display.get("display_type", "full"),
            "dim_labels": display.get("dim_labels", []),
            "memory_mb": cached["memory_mb"],
        }
        if cached["type"] == "kfac":
            result["kfac_factors"] = display.get("kfac_factors", [])
        elif cached["type"] == "block_diag":
            result["block_matrices"] = display.get("block_matrices", [])

        return result

    @staticmethod
    async def _handle_compute_eigenvalues(session, payload, ws):
        if session._cached_hessian is None:
            raise ValueError("Compute Hessian first")

        cached = session._cached_hessian
        method = payload.get("method", "exact" if cached["type"] != "diagonal" else "diagonal")

        if cfg.REMOTE_ENABLED and _get_remote().connected and cached["type"] in ("full", "diagonal"):
            loop = asyncio.get_event_loop()
            remote = _get_remote()
            result = await loop.run_in_executor(None, remote.compute_eigenvalues, session, method)
        else:
            from backend.hessian import compute_eigenvalues as compute_ev
            result = compute_ev(cached, method)

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
        solver = payload.get("solver", "auto")

        if cfg.REMOTE_ENABLED and _get_remote().connected:
            loop = asyncio.get_event_loop()
            remote = _get_remote()
            return await loop.run_in_executor(
                None, remote.solve_newton, session, reg, apply_step, step_scale)

        from backend.equations import solve_newton
        return solve_newton(session, reg, apply_step, step_scale, ws, solver=solver)

    @staticmethod
    async def _handle_solve_natural_gradient(session, payload, ws):
        _ensure_loss_fn(session)
        reg = payload.get("regularization", cfg.DEFAULT_REGULARIZATION)
        apply_step = payload.get("apply_step", True)
        step_scale = payload.get("step_scale", cfg.DEFAULT_STEP_SCALE)
        solver = payload.get("solver", "auto")

        if cfg.REMOTE_ENABLED and _get_remote().connected:
            loop = asyncio.get_event_loop()
            remote = _get_remote()
            return await loop.run_in_executor(
                None, remote.solve_natural_gradient, session, reg, apply_step, step_scale)

        from backend.equations import solve_natural_gradient
        return solve_natural_gradient(session, reg, apply_step, step_scale, ws, solver=solver)

    @staticmethod
    async def _handle_solve_linear_system(session, payload, ws):
        _ensure_loss_fn(session)
        if cfg.REMOTE_ENABLED and _get_remote().connected:
            raise ValueError("Linear system solving not yet supported in remote mode")

        from backend.equations import solve_linear
        rhs = payload.get("right_hand_side", None)
        reg = payload.get("regularization", 0.0)
        solver = payload.get("solver", "auto")
        return solve_linear(session, rhs, reg, ws, solver=solver)

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
    async def _handle_compute_ntk(session, payload, ws):
        if session.model is None:
            raise ValueError("Create a model first")

        from backend.ntk import compute_ntk, ntk_to_display_matrix

        ntk_mode = payload.get('ntk_mode', 'sample')
        max_samples = payload.get('max_samples', cfg.NTK_MAX_SAMPLES)
        force = payload.get('force_compute', False)

        if session._cached_ntk is not None and not force:
            if session._cached_ntk.get('mode') != ntk_mode:
                session._cached_ntk = None

        if session._cached_ntk is not None and not force:
            cached = session._cached_ntk
        else:
            if cfg.REMOTE_ENABLED and _get_remote().connected:
                loop = asyncio.get_event_loop()
                remote = _get_remote()
                cached = await loop.run_in_executor(
                    None, remote.compute_ntk, session, ntk_mode, max_samples)
            else:
                cached = compute_ntk(session, max_samples=max_samples, ntk_mode=ntk_mode)

            session._cached_ntk = cached

        display = ntk_to_display_matrix(cached)

        return {
            'mode': cached['mode'],
            'N': cached['N'],
            'K': cached['K'],
            'P': cached['P'],
            'ntk_matrix': display['ntk_matrix'],
            'ntk_shape': display['ntk_shape'],
            'display_type': display['display_type'],
            'dim_labels': display['dim_labels'],
            'memory_mb': cached['memory_mb'],
        }

    @staticmethod
    async def _handle_compute_ntk_eigenvalues(session, payload, ws):
        if session._cached_ntk is None:
            raise ValueError("Compute NTK first")

        if session._cached_ntk_eigenvalues is not None:
            return session._cached_ntk_eigenvalues

        from backend.ntk import compute_ntk_eigenvalues

        if cfg.REMOTE_ENABLED and _get_remote().connected:
            loop = asyncio.get_event_loop()
            remote = _get_remote()
            result = await loop.run_in_executor(
                None, remote.compute_ntk_eigenvalues, session._cached_ntk)
        else:
            result = compute_ntk_eigenvalues(session._cached_ntk)

        session._cached_ntk_eigenvalues = result
        return result

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
    @staticmethod
    def _safe_round(val, ndigits=6):
        import math
        if math.isfinite(val):
            return round(val, ndigits)
        return None

    @staticmethod
    async def _handle_compute_interpolation(session, payload, ws):
        if session.model is None:
            raise ValueError("Create a model first")
        _ensure_loss_fn(session)

        from backend.landscape import compute_interpolation
        num_steps = min(payload.get("num_steps", 20), 100)
        snapshot_a = payload.get("snapshot_a", -1)
        snapshot_b = payload.get("snapshot_b", 0)
        return await compute_interpolation(session, num_steps, snapshot_a, snapshot_b, ws)

    @staticmethod
    async def _handle_compute_fisher(session, payload, ws):
        if session.model is None:
            raise ValueError("Create a model first")
        _ensure_loss_fn(session)

        from backend.fisher import compute_fisher, fisher_to_display_matrix

        mode = payload.get("mode", "auto")
        max_samples = min(payload.get("max_samples", cfg.NTK_MAX_SAMPLES), 256)
        force = payload.get("force_compute", False)

        if session._cached_fisher is not None and not force:
            if session._cached_fisher.get("mode") != mode:
                session._cached_fisher = None

        if session._cached_fisher is not None and not force:
            cached = session._cached_fisher
        else:
            loop = asyncio.get_event_loop()
            cached = await loop.run_in_executor(
                None, lambda: compute_fisher(session, max_samples=max_samples, mode=mode))
            session._cached_fisher = cached

        display = fisher_to_display_matrix(cached, session.model)

        return {
            "num_parameters": cached["param_count"],
            "is_diagonal": cached["type"] == "diagonal",
            "method": cached["type"],
            "fisher_matrix": display.get("fisher_matrix"),
            "fisher_shape": display.get("fisher_shape", []),
            "display_type": display.get("display_type", "full"),
            "dim_labels": display.get("dim_labels", []),
            "memory_mb": cached["memory_mb"],
            "num_samples": cached["num_samples"],
        }

    @staticmethod
    async def _handle_compute_fisher_eigenvalues(session, payload, ws):
        if session._cached_fisher is None:
            raise ValueError("Compute Fisher matrix first")

        if session._cached_fisher_eigenvalues is not None:
            return session._cached_fisher_eigenvalues

        from backend.fisher import compute_fisher_eigenvalues
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, compute_fisher_eigenvalues, session._cached_fisher)
        session._cached_fisher_eigenvalues = result
        return result

    @staticmethod
    async def _handle_compute_layer_stats(session, payload, ws):
        if session.model is None:
            raise ValueError("Create a model first")

        layers = []
        # Hessian diagonal if cached
        hessian_diag = None
        if session._cached_hessian is not None:
            hc = session._cached_hessian
            if hc["type"] == "diagonal":
                hessian_diag = hc["data"]
            elif hc["type"] == "full" and hc["data"].ndim == 2:
                hessian_diag = hc["data"].diagonal()

        diag_idx = 0
        for name, p in session.model.named_parameters():
            info = {
                "name": name,
                "shape": list(p.shape),
                "numel": p.numel(),
                "weight_norm": _Dispatcher._safe_round(p.data.norm(2).item()),
                "weight_mean": _Dispatcher._safe_round(p.data.mean().item()),
                "weight_std": _Dispatcher._safe_round(p.data.std(unbiased=(p.numel() > 1)).item()),
            }
            if p.grad is not None:
                info["grad_norm"] = _Dispatcher._safe_round(p.grad.data.norm(2).item())
                info["grad_mean"] = _Dispatcher._safe_round(p.grad.data.mean().item())
            else:
                info["grad_norm"] = None
                info["grad_mean"] = None

            if hessian_diag is not None and diag_idx < hessian_diag.numel():
                block = hessian_diag.flatten()[diag_idx:diag_idx + p.numel()]
                info["hessian_diag_mean"] = _Dispatcher._safe_round(block.mean().item())
                info["hessian_diag_max"] = _Dispatcher._safe_round(block.max().item())
                diag_idx += p.numel()

            layers.append(info)

        return {
            "layers": layers,
            "total_params": session.param_count,
            "has_hessian_diag": hessian_diag is not None,
        }

    @staticmethod
    async def _handle_compute_activation_stats(session, payload, ws):
        from backend.activation_stats import compute_activation_stats
        if session.model is None:
            raise ValueError("Create a model first")
        if session.train_loader is None:
            raise ValueError("Set a dataset first")
        num_batches = min(payload.get("num_batches", 2), 16)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, compute_activation_stats, session, num_batches)

    @staticmethod
    async def _handle_compute_gradient_stats(session, payload, ws):
        from backend.gradient_stats import compute_gradient_stats
        if session.model is None:
            raise ValueError("Create a model first")
        if session.train_loader is None:
            raise ValueError("Set a dataset first")
        _ensure_loss_fn(session)
        num_batches = min(payload.get("num_batches", 4), 32)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, compute_gradient_stats, session, num_batches)

    @staticmethod
    async def _handle_get_remote_status(session, payload, ws):
        remote = _get_remote()
        return {
            "connected": remote.connected,
            "remote_device": remote.remote_device,
        }

    @staticmethod
    async def _handle_compute_weight_histogram(session, payload, ws):
        if not session.param_snapshots:
            raise ValueError("No training snapshots available. Run training first.")


        num_bins = min(payload.get("num_bins", 50), 100)
        max_layers = min(payload.get("max_layers", 8), 20)

        # Collect weight layers (exclude biases and batch-norm running stats)
        sample_sd = session.param_snapshots[0]
        weight_layers = []
        for key, val in sample_sd.items():
            if "weight" in key and val.ndim >= 2:
                weight_layers.append(key)

        if not weight_layers:
            # Fallback: include all non-bias layers
            weight_layers = [k for k, v in sample_sd.items() if "bias" not in k and "running" not in k and "tracked" not in k][:max_layers]

        weight_layers = weight_layers[:max_layers]

        snapshots_data = []
        for snap_idx, sd in enumerate(session.param_snapshots):
            layer_hists = {}
            for layer_name in weight_layers:
                w = sd.get(layer_name)
                if w is None:
                    continue
                vals = w.float().flatten().tolist()
                mn = min(vals)
                mx = max(vals)
                rng = mx - mn
                if rng == 0:
                    rng = 1.0
                # Compute histogram
                bins = []
                counts = [0] * num_bins
                bin_width = rng / num_bins
                for i in range(num_bins):
                    bins.append(round(mn + i * bin_width, 6))
                for v in vals:
                    idx = min(int((v - mn) / bin_width), num_bins - 1)
                    counts[idx] += 1
                layer_hists[layer_name] = {
                    "bins": bins,
                    "counts": counts,
                    "mean": round(sum(vals) / len(vals), 6),
                    "std": round((sum((x - sum(vals)/len(vals))**2 for x in vals) / len(vals))**0.5, 6),
                    "min": round(mn, 6),
                    "max": round(mx, 6),
                }
            snapshots_data.append({
                "index": snap_idx,
                "histograms": layer_hists,
            })

        return {
            "layers": weight_layers,
            "num_snapshots": len(session.param_snapshots),
            "snapshots": snapshots_data,
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
