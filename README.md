# Hessian Playground

Interactive web-based tool for exploring neural network models, Hessian matrices, NTK (Neural Tangent Kernel), and loss landscapes using PyTorch.

## Features

- **Model Editor**: Write PyTorch models in Python with a built-in CodeMirror editor. Presets include MLP, SwiGLU, ResNet, and Transformer.
- **Datasets**: MNIST, CIFAR-10, CIFAR-100, XOR, polynomial regression, or custom `torch.utils.data.Dataset` code.
- **Optimizers**: All `torch.optim` optimizers (SGD, Adam, AdamW, RMSprop, etc.) with configurable parameters, gradient ascent mode, or custom optimizer code.
- **Live Training**: Real-time loss and accuracy plots via WebSocket push messages.
- **Hessian Computation**: Auto-selects from four methods based on parameter count:
  - Full Hessian (≤2K params), Block-diagonal (2K–50K), K-FAC (>50K), or Diagonal (Hutchinson's estimator).
- **NTK Computation**: Sample-wise (N×N) or output-wise (K×K) Neural Tangent Kernel with eigenvalue analysis.
- **Eigenvalue Analysis**: Spectral decomposition via exact `eigvalsh`, QR-orthogonalized power iteration, or K-FAC Kronecker eigenvalues. Displays histogram, condition number, effective rank, and trace.
- **Loss Landscape**: 2D visualization via PCA of training trajectory or random orthonormal parameter directions.
- **Equation Solving**: Newton step (H·Δθ = −g with rollback on loss increase) and linear system solving (H·x = b) with CG, Cholesky, or direct solvers.
- **Model Adaptation**: Reshape model input/output layers when switching datasets without losing trained weights.
- **Remote Execution**: Optional SSH-based offloading of heavy computation to a remote server.
- **i18n & Theming**: Multi-language support (EN/ZH) and light/dark/auto theme toggle.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
# Or editable install (provides the `hessian-playground` console script)
pip install -e .

# Start server
python src/backend/main.py --port 8080

# Open in browser
# http://localhost:8080
```

## Usage

1. Select a preset model (MLP, SwiGLU, ResNet, Transformer) or write your own `nn.Module` in the left panel
2. Click **Create Model** to instantiate it
3. Choose a dataset and optimizer in the center panel (or write custom code)
4. Click **Train** to start training — loss updates in real time
5. Explore via the visualization tabs:
   - **Loss**: Training loss curve
   - **Hessian**: Compute and visualize the Hessian matrix heatmap
   - **Landscape**: PCA or random-direction loss landscape
   - **Eigenvalues**: Spectral analysis of the Hessian
   - **Equation**: Newton step or linear system solve
   - **NTK**: Neural Tangent Kernel matrix and eigenvalues

### Custom Code

All code editors allow arbitrary Python:

- **Custom Model**: Define any `torch.nn.Module` subclass.
- **Custom Optimizer**: Write code that creates an `optimizer` variable (e.g., `optimizer = torch.optim.SGD(model.parameters(), lr=0.01)`).
- **Custom Dataset**: Define a `torch.utils.data.Dataset` subclass or create a `dataset` variable.

### Remote Execution

Enable SSH offloading in Settings → Remote. When connected, Hessian, eigenvalues, landscape, Newton step, NTK, and training run on the remote machine. Configure via environment variables or the Settings UI.

## Architecture

- **Backend**: Python aiohttp server with a single WebSocket endpoint (`/ws`). JSON request/response protocol with `msg_id` correlation.
- **Frontend**: Single HTML page with CodeMirror 5 editor and Plotly.js visualizations. Plain ES6 JavaScript, no bundler.
- **Compute**: PyTorch on CPU or CUDA (auto-detected). Hessian methods scale from exact (small models) to K-FAC (large models).

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision
- aiohttp
- numpy
- scipy
- paramiko (for remote execution)

## Limitations

- Hard parameter limit: 1M (configurable)
- Full Hessian: ≤2K params (auto); warns above 10K
- Hessian/NTK display matrices capped at 200×200 (block-averaged)
- Landscape grid resolution capped at 50×50
- Sandbox `exec()` timeout: 5s
- Remote execution requires SSH access to the target machine
