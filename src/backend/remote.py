"""Remote execution via SSH for offloading computation to a remote server."""

import os
import pickle
import tempfile
import uuid

import torch

import backend.config as cfg
from backend.utils import deserialize_tensor, serialize_tensor


class RemoteExecutor:
    """Manages SSH connection and remote computation execution."""

    def __init__(self):
        self._client = None
        self._sftp = None
        self._remote_dir = None
        self._worker_uploaded = False
        self._remote_device = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._client is not None

    @property
    def remote_device(self) -> str | None:
        return self._remote_device

    def connect(self) -> str:
        host = cfg.REMOTE_HOST
        port = cfg.REMOTE_PORT
        user = cfg.REMOTE_USER
        password = cfg.REMOTE_PASSWORD

        if not host or not user:
            raise ValueError("Remote host and user must be configured")

        import paramiko

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, port=port, username=user, password=password, timeout=cfg.REMOTE_CONNECT_TIMEOUT)

        self._client = client
        self._sftp = client.open_sftp()

        # Create working directory on remote
        uid = uuid.uuid4().hex[:8]
        self._remote_dir = f"{cfg.REMOTE_TEMP_DIR}/hp_{uid}"
        self._sftp.mkdir(self._remote_dir)

        # Upload worker script
        worker_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote_worker.py")
        self._sftp.put(worker_src, f"{self._remote_dir}/worker.py")

        # Upload backend package so the worker can import computation kernels
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        self._sftp.mkdir(f"{self._remote_dir}/backend")
        for fname in ["__init__.py", "config.py", "protocol.py", "hessian.py",
                      "landscape.py", "equations.py", "training.py",
                      "session.py", "model_sandbox.py"]:
            src = os.path.join(backend_dir, fname)
            if os.path.exists(src):
                self._sftp.put(src, f"{self._remote_dir}/backend/{fname}")

        self._worker_uploaded = True

        # Detect remote device
        try:
            _, stdout, _ = client.exec_command(
                f"{cfg.REMOTE_PYTHON} -c 'import torch; print(\"cuda\" if torch.cuda.is_available() else \"cpu\")'",
                timeout=cfg.REMOTE_DETECT_TIMEOUT
            )
            stdout.channel.recv_exit_status()
            self._remote_device = stdout.read().decode().strip()
        except Exception:
            self._remote_device = "cpu"

        return f"Connected to {host}:{port}"

    def disconnect(self):
        if self._sftp:
            try:
                # Clean up remote directory
                if self._remote_dir:
                    self._run_remote(f"rm -rf {self._remote_dir}")
            except Exception:
                pass
            self._sftp.close()
            self._sftp = None
        if self._client:
            self._client.close()
            self._client = None
        self._remote_dir = None
        self._worker_uploaded = False
        self._remote_device = None

    # ------------------------------------------------------------------
    # Public API — mirrors local computation functions
    # ------------------------------------------------------------------

    def compute_hessian(self, session, use_diagonal=False, sample_batches=1):
        data = self._serialize_session(session)
        data["type"] = "compute_hessian"
        data["params"] = {
            "use_diagonal_approx": use_diagonal,
            "sample_batches": sample_batches,
        }
        if session.loss_fn is None:
            data["loss_fn"] = "cross_entropy" if session.task_type == "classification" else "mse"
        else:
            data["loss_fn"] = "cross_entropy"  # default

        result = self._execute_remote(data, cfg.REMOTE_COMPUTE_TIMEOUT)
        r = result["result"]

        hessian = r.get("hessian_matrix")
        if hessian is None:
            hessian = r.get("hessian_diag")
        if hessian is None:
            hessian = r.get("hessian")
        is_diag = r["is_diagonal"]

        # Update model if remote training modified it
        if r.get("model_state"):
            session.model.load_state_dict(deserialize_tensor(r["model_state"], cfg.DEVICE))

        return hessian, is_diag

    def compute_eigenvalues(self, session, method="exact"):
        if session._cached_hessian is None:
            raise ValueError("Compute Hessian first")

        H, is_diag = session._cached_hessian

        data = {
            "type": "compute_eigenvalues",
            "hessian": serialize_tensor(H),
            "is_diagonal": is_diag,
            "params": {"method": method},
        }
        result = self._execute_remote(data, cfg.REMOTE_COMPUTE_TIMEOUT)
        return result["result"]

    async def compute_landscape(self, session, resolution, range_factor, mode, seed, ws):
        data = self._serialize_landscape(session, resolution, range_factor, mode, seed)
        result = self._execute_remote(data, cfg.REMOTE_COMPUTE_TIMEOUT)
        return result["result"]

    def compute_landscape_sync(self, session, resolution, range_factor, mode, seed):
        """Synchronous version for run_in_executor."""
        data = self._serialize_landscape(session, resolution, range_factor, mode, seed)
        result = self._execute_remote(data, cfg.REMOTE_COMPUTE_TIMEOUT)
        return result["result"]

    def _serialize_landscape(self, session, resolution, range_factor, mode, seed):
        data = self._serialize_session(session)
        data["type"] = "compute_landscape"
        data["params"] = {
            "mode": mode,
            "grid_resolution": resolution,
            "range_factor": range_factor,
            "seed": seed,
        }
        if mode == "pca" and session.param_snapshots:
            S = torch.zeros(len(session.param_snapshots), session.param_count)
            for i, sd in enumerate(session.param_snapshots):
                flat = torch.cat([v.float().view(-1) for v in sd.values()])
                S[i] = flat
            data["snapshots"] = serialize_tensor(S)
        return data

    def solve_newton(self, session, regularization, apply_step, step_scale):
        data = self._serialize_session(session)
        data["type"] = "solve_newton"
        data["params"] = {
            "regularization": regularization,
            "apply_step": apply_step,
            "step_scale": step_scale,
        }
        if session.loss_fn is None:
            data["loss_fn"] = "cross_entropy" if session.task_type == "classification" else "mse"
        else:
            data["loss_fn"] = "cross_entropy"

        result = self._execute_remote(data, cfg.REMOTE_COMPUTE_TIMEOUT)
        r = result["result"]

        # Update model if step was applied
        if r.get("step_applied") and r.get("model_state"):
            session.model.load_state_dict(deserialize_tensor(r["model_state"], cfg.DEVICE))
            session.invalidate_cache()
        r.pop("model_state", None)  # bytes, not JSON-serializable

        return r

    def run_training(self, session, payload):
        data = self._serialize_session(session)
        data["type"] = "run_training"
        data["params"] = {
            "epochs": payload.get("epochs", cfg.DEFAULT_EPOCHS),
            "batch_size": payload.get("batch_size", cfg.DEFAULT_BATCH_SIZE) if hasattr(session, 'train_loader') else cfg.DEFAULT_BATCH_SIZE,
            "lr": cfg.DEFAULT_LEARNING_RATE,
        }
        if session.loss_fn is None:
            data["loss_fn"] = "cross_entropy" if session.task_type == "classification" else "mse"
        else:
            data["loss_fn"] = "cross_entropy"

        result = self._execute_remote(data, cfg.REMOTE_TRAINING_TIMEOUT, capture_progress=True)
        r = result["result"]

        if r.get("model_state"):
            session.model.load_state_dict(deserialize_tensor(r["model_state"], cfg.DEVICE))
            session.invalidate_cache()
        r.pop("model_state", None)  # bytes, not JSON-serializable

        return r

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _serialize_session(self, session) -> dict:
        """Extract serializable data from a session for remote execution."""
        model_code = session.model_code or ""
        model_state = serialize_tensor(session.model.state_dict())

        # Get a batch of data
        x, y = next(iter(session.train_loader))
        data_x = serialize_tensor(x)
        data_y = serialize_tensor(y)

        return {
            "model_code": model_code,
            "model_state": model_state,
            "data_x": data_x,
            "data_y": data_y,
            "loss_fn": "cross_entropy" if session.task_type == "classification" else "mse",
            "input_size": session.input_size,
            "output_size": session.output_size,
        }

    def _execute_remote(self, request: dict, timeout: int, capture_progress: bool = False) -> dict:
        """Execute a computation on the remote and return the response dict."""
        if not self.connected:
            raise RuntimeError("Remote executor is not connected")

        input_path = f"{self._remote_dir}/input.pkl"
        output_path = f"{self._remote_dir}/output.pkl"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(request, f)
            local_input = f.name

        response = None
        progress_lines = []
        try:
            self._sftp.put(local_input, input_path)

            cmd = f"{cfg.REMOTE_PYTHON} {self._remote_dir}/worker.py --input {input_path} --output {output_path}"
            _, stdout, stderr = self._client.exec_command(cmd, timeout=timeout)

            if capture_progress:
                for line in iter(stdout.readline, ""):
                    line = line.strip()
                    if line:
                        progress_lines.append(line)

            exit_code = stdout.channel.recv_exit_status()
            stderr_text = stderr.read().decode("utf-8", errors="replace")

            if exit_code != 0:
                raise RuntimeError(f"Remote worker exited with code {exit_code}: {stderr_text}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
                local_output = f.name
            self._sftp.get(output_path, local_output)

            with open(local_output, "rb") as f:
                response = pickle.load(f)

            os.unlink(local_output)

        finally:
            os.unlink(local_input)
            try:
                self._run_remote(f"rm -f {input_path} {output_path}")
            except Exception:
                pass

        if response is None or not response.get("success"):
            raise RuntimeError(f"Remote computation failed: {response.get('error', 'unknown error') if response else 'no response'}")

        if capture_progress:
            response["progress"] = progress_lines
        return response

    def _run_remote(self, cmd: str) -> tuple:
        """Run a shell command on the remote and return (exit_code, stdout, stderr)."""
        _, stdout, stderr = self._client.exec_command(cmd, timeout=cfg.REMOTE_DETECT_TIMEOUT)
        exit_code = stdout.channel.recv_exit_status()
        return exit_code, stdout.read().decode(), stderr.read().decode()


