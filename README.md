# Hessian Playground

Interactive web-based tool for exploring MLP models, optimizers, Hessian matrices, and loss landscapes using PyTorch.

## Features

- **Model Editor**: Write PyTorch MLP models in Python (browser code editor)
- **Dataset Selection**: MNIST, CIFAR-10, CIFAR-100, XOR, polynomial regression, or custom datasets
- **Optimizer Configuration**: Choose from torch.optim (SGD, Adam, AdamW, RMSprop, etc.) or write custom optimizer code
- **Live Training**: Watch loss and accuracy update in real-time via WebSocket
- **Hessian Computation**: Full or diagonal Hessian matrix with heatmap visualization
- **Eigenvalue Analysis**: Spectral decomposition of the Hessian (exact or power iteration)
- **Loss Landscape**: 2D visualization via PCA of training trajectory or random parameter directions
- **Newton Step Solver**: Solve H·Δθ = −g to improve model parameters directly

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
python backend/main.py --port 8080

# Open in browser
# http://localhost:8080
```

## Architecture

- **Backend**: Python aiohttp server with WebSocket
- **Frontend**: Single HTML page with CodeMirror editor and Plotly.js visualizations
- **Protocol**: JSON messages over WebSocket, request/response with msg_id correlation

## Usage

1. Write or modify the model code in the left panel
2. Click **Create Model** to instantiate it
3. Select a dataset and optimizer from the center panel
4. Click **Train** to start training (live loss plot updates)
5. Explore:
   - **Hessian**: Compute and visualize the Hessian matrix
   - **PCA Landscape**: View loss landscape from PCA of training trajectory
   - **Random Landscape**: View loss landscape from random parameter directions
   - **Newton Step**: Apply Newton update to improve model parameters

### Custom Code

All code editors allow arbitrary Python. Examples:

**Custom Model**: Define any `torch.nn.Module` subclass.

**Custom Optimizer**: Write code that creates an `optimizer` variable (e.g., `optimizer = torch.optim.SGD(model.parameters(), lr=0.01)`).

**Custom Dataset**: Define a `torch.utils.data.Dataset` subclass or create a `dataset` variable.

## Limitations

- CPU-only (no CUDA in this environment)
- Full Hessian limited to models with ≤10K parameters
- Larger models use diagonal approximation (Hutchinson's estimator)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision
- aiohttp
- numpy
- scipy
