# dafne-monai-inference

Lightweight inference and incremental learning package for [Dafne](https://github.com/dafne-imaging/dafne) models, built on [MONAI](https://monai.io/) and [PyTorch](https://pytorch.org/).

## Overview

This package provides:
- **Inference** — sliding-window (3D) and slice-by-slice (2D) segmentation inference using trained Dafne models
- **Incremental learning** — online fine-tuning of a deployed model on new annotated data, with optional EWC (Elastic Weight Consolidation) regularization to prevent catastrophic forgetting
- **Network architectures** — `DafneUnetModel` and `DafneDynUnetModel`, wrapping MONAI's `UNet` and `DynUNet` with Dafne-compatible interfaces
- **Transforms** — preprocessing pipeline with anisotropy handling, spacing normalization, and bounding-box cropping

## Installation

```bash
pip install dafne-monai-inference
```

## Requirements

- Python >= 3.8
- torch >= 2.0
- monai >= 1.3
- numpy >= 1.24
- scikit-image >= 0.19

## Usage

### Inference

```python
from dafne_inference.inference import run_inference

# model_obj: a DynamicTorchModel loaded from a .model file
# data_dict must contain 'image' (numpy array) and either 'affine' or 'resolution'
result = run_inference(model_obj, data_dict)
# result: dict { label_name: binary_mask (np.int8) }
```

### Incremental learning

```python
from dafne_inference.incremental import run_incremental_learning

# trainingData: dict with key 'image_list' (list of numpy arrays) and 'resolution'
# trainingOutputs: dict of masks corresponding to each image
run_incremental_learning(model_obj, trainingData, trainingOutputs, bs=1, minTrainImages=2)
```

### Network architectures

```python
from dafne_inference.networks import DafneUnetModel, DafneDynUnetModel

# Standard UNet
model = DafneUnetModel(
    spatial_dims=3,
    in_channels=1,
    out_channels=5,
    start_channel=32,
    n_levels=5
)

# Dynamic UNet (nnU-Net style)
model = DafneDynUnetModel(
    spatial_dims=3,
    in_channels=1,
    out_channels=5,
    kernels=[[3,3,3], [3,3,3], [3,3,3]],
    strides=[[1,1,1], [2,2,2], [2,2,2]]
)
```

## License

GNU General Public License v3 or later (GPLv3+)
