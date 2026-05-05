## Overview

This codebase supports the following experimental pipeline:

1. Self-supervised pretraining on paired DFS/CIR ultrasound tensors.
2. Downstream frozen linear probing.
3. Downstream full finetuning.
4. Ablation studies through YAML configuration files.

The main components are:

```text
.
├── README.md
├── requirements.txt
├── pretrain.py
├── finetune.py
├── configs/
│   ├── pretrain_default.yaml
│   └── finetune_default.yaml
├── datasets/
├── losses/
├── models/
├── utils/
└── scripts/
```

---
## Environment

The code is intended to run with Python and PyTorch.

Recommended environment:

```text
Python >= 3.10
PyTorch >= 2.1.0
CUDA-enabled GPU recommended
Linux recommended
```

Install the required packages with:

```bash
pip install -r requirements.txt
```

The expected `requirements.txt` is:

```text
torch>=2.1.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
PyYAML>=6.0
scikit-learn>=1.3.0
tqdm>=4.66.0
```

---

## Data Format

The code expects paired ultrasound sensing data stored in `.mat` files.

Each `.mat` file should contain the following keys:

```text
dfs2d
cir
```

where:

- `dfs2d` is a 2D DFS matrix.
- `cir` is a 2D CIR matrix. It may be real-valued or complex-valued depending on preprocessing.

The training and evaluation scripts read data through CSV manifest files.

A typical pretraining manifest should contain paths to the corresponding `.mat` files:

```csv
path
sample_0001.mat
sample_0002.mat
sample_0003.mat
```

A typical downstream manifest should contain paths and labels:

```csv
path,label
sample_0001.mat,0
sample_0002.mat,1
sample_0003.mat,2
```

Relative paths are resolved with respect to the location from which the scripts are launched, unless otherwise specified in the dataset implementation.

---

## Configuration Files

Default configuration files are provided in:

```text
configs/pretrain_default.yaml
configs/finetune_default.yaml
```

The pretraining configuration controls:

- Model architecture.
- Projection dimension.
- Batch size.
- Number of epochs.
- Optimizer settings.
- Contrastive loss settings.
- Physics-guided consistency loss settings.
- Checkpoint and output paths.

The finetuning configuration controls:

- Downstream task setup.
- Number of classes.
- Frozen probing or full finetuning mode.
- Optimizer settings.
- Training schedule.
- Checkpoint loading.

Before running experiments, please update the manifest paths in the commands or configuration files to match the local data location.

---

## Pretraining

Run self-supervised pretraining with:

```bash
python pretrain.py \
  --config configs/pretrain_default.yaml \
  --train_csv path/to/pretrain_manifest.csv \
  --save_dir outputs/pretrain_run
```

---

## Downstream Finetuning

Run full downstream finetuning with:

```bash
python finetune.py \
  --config configs/finetune_default.yaml \
  --train_csv path/to/train_manifest.csv \
  --val_csv path/to/val_manifest.csv \
  --pretrained_ckpt outputs/pretrain_run/best.pt \
  --save_dir outputs/finetune_run \
  --num_classes <NUM_CLASSES> \
  --mode finetune
```

---

## Frozen Linear Probing

Run frozen linear probing with:

```bash
python finetune.py \
  --config configs/finetune_default.yaml \
  --train_csv path/to/train_manifest.csv \
  --val_csv path/to/val_manifest.csv \
  --pretrained_ckpt outputs/pretrain_run/best.pt \
  --save_dir outputs/frozen_probe_run \
  --num_classes <NUM_CLASSES> \
  --mode frozen
```

In frozen probing mode, the pretrained encoder is frozen and only the downstream classifier head is trained.

---

## Evaluation

Validation is performed during downstream training using the validation manifest specified by `--val_csv`.

The script reports standard classification metrics such as loss and accuracy, depending on the task configuration and implementation.

A typical workflow is:

```bash
# 1. Pretrain
python pretrain.py \
  --config configs/pretrain_default.yaml \
  --train_csv path/to/pretrain_manifest.csv \
  --save_dir outputs/pretrain_run

# 2. Frozen linear probing
python finetune.py \
  --config configs/finetune_default.yaml \
  --train_csv path/to/train_manifest.csv \
  --val_csv path/to/val_manifest.csv \
  --pretrained_ckpt outputs/pretrain_run/best.pt \
  --save_dir outputs/frozen_probe_run \
  --num_classes <NUM_CLASSES> \
  --mode frozen

# 3. Full finetuning
python finetune.py \
  --config configs/finetune_default.yaml \
  --train_csv path/to/train_manifest.csv \
  --val_csv path/to/val_manifest.csv \
  --pretrained_ckpt outputs/pretrain_run/best.pt \
  --save_dir outputs/finetune_run \
  --num_classes <NUM_CLASSES> \
  --mode finetune
```

---

## Ablation Studies

Ablations can be performed by modifying the YAML configuration files.

Relevant options include:

- Model depth and hidden dimensions.
- Projection dimension.
- Temperature for contrastive learning.
- Weight of the physics-guided consistency loss.
- Downstream training mode: frozen probing or full finetuning.
- Batch size and learning rate.
- Number of epochs.

Example:

```bash
python pretrain.py \
  --config configs/pretrain_default.yaml \
  --train_csv path/to/pretrain_manifest.csv \
  --save_dir outputs/ablation_run
```

Then modify the corresponding configuration values in `configs/pretrain_default.yaml`.

---

## Hardware Requirements

A CUDA-enabled GPU is recommended for pretraining and full finetuning.

The code may run on CPU for small-scale debugging, but full experiments are expected to require GPU acceleration.

The actual memory requirement depends on:

- Input tensor size.
- Batch size.
- Model dimension.
- Number of encoder layers.
- Whether full finetuning or frozen probing is used.

If GPU memory is limited, reduce the batch size in the YAML configuration file.

---
