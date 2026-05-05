# Anonymous Code for NeurIPS 2026 Submission

This repository contains the anonymous code release for a NeurIPS 2026 submission.

The code implements a physics-guided pretraining and downstream finetuning pipeline for paired dual-view ultrasound sensing data. The repository has been anonymized for double-blind review: all author names, affiliations, acknowledgments, institutional identifiers, and personal links have been removed.

This repository is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0) for anonymous peer review. Attribution should be made to "Anonymous Authors" during the review period.

---

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

## Anonymity Statement

This repository is prepared for double-blind peer review.

The repository intentionally does not include:

- Author names.
- Institutional affiliations.
- Personal email addresses.
- Personal GitHub usernames or project URLs.
- Acknowledgments.
- Funding information.
- Internal directory paths.
- Non-anonymous commit history.

Please do not attempt to identify the authors during the review period.

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

If CUDA is used, please install the PyTorch build matching the local CUDA version according to the official PyTorch installation instructions.

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

## Data Availability During Review

The raw ultrasound dataset is not included in this anonymous repository.

The repository provides the complete model, loss functions, data loading interface, pretraining script, finetuning script, and configuration files required to reproduce the experimental pipeline using data formatted according to the manifest specification above.

The raw data are omitted during anonymous review because of one or more of the following constraints:

- Dataset size.
- Data sharing restrictions.
- Privacy or institutional constraints.
- Review-time anonymization requirements.

A de-anonymized version of the dataset access instructions may be provided after the review process, if permitted.

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

Expected outputs:

```text
outputs/pretrain_run/
├── best.pt
├── last.pt
└── training logs
```

The exact checkpoint filenames may depend on the saving logic in the script. The best checkpoint is selected according to the validation or training criterion defined in the code.

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

Replace `<NUM_CLASSES>` with the number of downstream classes.

Expected outputs:

```text
outputs/finetune_run/
├── best.pt
├── last.pt
└── training logs
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

## Reproducibility Notes

To improve reproducibility, we recommend:

1. Using the same Python and PyTorch versions across runs.
2. Fixing random seeds if supported by the scripts.
3. Keeping train/validation/test manifest files unchanged across experiments.
4. Saving the exact configuration file used for each run.
5. Reporting the mean and standard deviation across multiple random seeds when possible.

For each experiment, the following information should be recorded:

```text
config file
training manifest
validation manifest
random seed
checkpoint path
hardware type
software environment
```

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

## Repository Hygiene for Anonymous Review

This repository should not contain the following files:

```text
data/
outputs/
checkpoints/
runs/
wandb/
*.mat
*.pt
*.pth
*.ckpt
*.log
.env
```

These files should be excluded through `.gitignore`.

Recommended `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.DS_Store
.env

*.log
outputs/
checkpoints/
runs/
wandb/

data/
*.mat
*.pt
*.pth
*.ckpt
```

Before submitting the repository link or supplementary ZIP, we recommend checking for accidental identity leakage:

```bash
grep -RInE "author|affiliation|university|institute|laboratory|lab|gmail|edu|github.com|acknowledg|funding|grant|/Users/|/home/" .
```

Also check Git metadata if using a Git repository:

```bash
git log --format='%an <%ae>'
git remote -v
```

The repository should not expose real author names, personal email addresses, institutional paths, or links to non-anonymous repositories.

---

## Troubleshooting

### Import errors

Please ensure that all dependencies are installed:

```bash
pip install -r requirements.txt
```

### CUDA out-of-memory error

Reduce the batch size in the corresponding YAML configuration file.

### Missing `.mat` keys

Each `.mat` file should contain:

```text
dfs2d
cir
```

If different key names are used, update the dataset loading code accordingly.

### Manifest path errors

Check that the paths in the CSV manifest are correct relative to the working directory.

---

## License

This repository is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

During the double-blind review period, attribution should be made to:

```text
Anonymous Authors
```

After the review process, attribution information may be updated to include the de-anonymized author and project details, if appropriate.

For more information about CC BY 4.0, see the license file included with this repository or the official Creative Commons license text.

---

## Citation

Citation information is omitted during double-blind review.

A complete citation entry will be added after the review process, if appropriate.