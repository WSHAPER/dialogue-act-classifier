# Prerequisites

## Hardware

- **GPU**: NVIDIA GPU with CUDA 12.8+ support (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: ~5GB free (PyTorch CUDA + model downloads + training artifacts)

Training on CPU works but is ~10-20x slower. Expect ~2 hours on CPU vs ~15-30 minutes on GPU.

## Software

### NVIDIA Driver + CUDA

```bash
nvidia-smi  # verify driver loaded, check CUDA version
```

Requires CUDA 12.8+ (driver 570+). Install from [NVIDIA](https://developer.nvidia.com/cuda-downloads) if missing.

### Python 3.10+

```bash
python3 --version
```

### PyTorch with CUDA

**Do not** install from PyPI (that gives you the CPU-only build). Use the CUDA index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verify:
```bash
python3 -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
# Expected: 2.x.x+cu128 CUDA: True
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

### datasets Library Version

The `silicone` dataset on HuggingFace uses a legacy loading script. You need `datasets >= 3.0` with `trust_remote_code=True` support. Version `4.x+` drops script support entirely.

```bash
pip install 'datasets>=3.0,<4.0'
```

We tested with `3.2.0`. If you see:

```
RuntimeError: Dataset scripts are no longer supported
```

then downgrade: `pip install 'datasets==3.2.0'`

## Quick Start (Full Setup)

```bash
# 1. PyTorch CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. Dependencies (includes datasets<4.0)
pip install -r requirements.txt

# 3. Verify GPU
python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'"

# 4. Train
python train.py

# 5. Evaluate
python evaluate.py

# 6. Export for Rust
python export.py --onnx
```

## Troubleshooting

### `torch.cuda.is_available()` returns False

You have the CPU-only PyTorch. Reinstall from the CUDA index:

```bash
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu128
```

### `RuntimeError: Dataset scripts are no longer supported`

The `datasets` library is too new. Downgrade:

```bash
pip install 'datasets==3.2.0'
```

### OOM during training

Reduce batch size:

```bash
python train.py --batch-size 16
```

### CUDA out of memory

DistilBERT is small (67M params), but if you're sharing GPU memory with other processes (e.g. Whisper transcription):

```bash
python train.py --batch-size 8
```
