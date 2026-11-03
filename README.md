# CDPL

> **Status: Under Review**
>
> This repository contains the code and configurations for a paper currently under review. The materials are shared for reference only. Please do not redistribute or use them to develop, evaluate, or benchmark competing methods before the paper is published.
>
> We will update the repository after the review process is complete.

Official implementation of **CDPL** for multi-view long-tail semi-supervised
classification.

## Repository layout

```
CDPL/
├── main.py                     # entry point (argument parsing, dataset root resolution)
├── configs/                    # one yaml per dataset / labeled ratio / imbalance setting
├── datasets/
│   ├── MultiviewDataset.py     # multi-view .mat loader (train/val/test splits)
│   └── BasicDataset.py         # torch Dataset returning {idx, x, y} dicts
├── models/
│   ├── extract.py              # FeatureExtract: per-view encoders/decoders/classifiers + fusion
│   ├── model_vote_right.py     # training model (pseudo-label selection, EMA teacher, losses)
│   ├── model_category.py       # evaluation model (same backbone, metrics + checkpoint loading)
│   └── ema.py                  # exponential moving average of the student weights
├── trainers/
│   └── train1.py               # builds datasets/loaders and dispatches train vs. test
└── utils/
    └── utils.py                # per-run CSV log path
```

## Installation

```bash
conda create -n cdpl python=3.10 -y
conda activate cdpl
pip install -r requirements.txt
```

## Data

Each dataset is stored in a MATLAB file with two variables:

- `X`  : a 1×V cell array, view `v` is an `[N, d_v]` feature matrix
- `gnd` (or `label`): an `N`-vector of class labels

Splits are expected under:

```
data/<DATASET>/imbalance/<labeled_ratio>/<imb_type>/gamma_<imb_ratio>/
├── train_label.mat
├── train_unlabel.mat
├── val.mat
└── test.mat
```

`data/` is resolved as `<repo>/data` by default (config key `data_base_local`);
override it with `--data_base /path/to/data`.

## Training

```bash
python main.py \
  --train True --test False \
  --config_file CIFAR100/0.09_exp_0.1_CIFAR100.yaml \
  --ratio 0.09 --seed 2023 --lr 1e-4 \
  --device cuda:0
```

Checkpoints and the per-run CSV log are written to
`checkpoint_<ratio>_<seed>/` and `result_<ratio>_<seed>/` next to `main.py`.

## Evaluation

```bash
python main.py \
  --train False --test True \
  --config_file CIFAR100/0.09_exp_0.1_CIFAR100.yaml \
  --ratio 0.09 --seed 2023 --lr 1e-4 \
  --device cuda:0
```

The evaluation model loads `<checkpoint>/<DATASET>/<run_name>_model_best.pth`
and reports accuracy, macro F1 / precision / recall, AUC (ovr and ovo),
G-mean, specificity and average precision.

## Configs

`configs/<DATASET>/` holds one file per `(labeled ratio, imbalance type,
imbalance ratio)` setting. `configs/CIFAR100/` contains the settings used in
the paper: a 0.09 labeled ratio with four unlabeled imbalance types (`exp`,
`exp_random`, `pxe`, `no_imbalance`) and three imbalance ratios (0.1, 0.2, 0.3).
