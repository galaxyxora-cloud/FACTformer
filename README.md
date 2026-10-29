# FACTformer: Frequency-Aware Calibration Transformer for Image Dehazing

This repository accompanies the manuscript **FACTformer: Frequency-Aware Calibration Transformer for Image Dehazing**. FACTformer uses frequency information to guide local feature aggregation, spatial relationships along image axes to refine features, and cross-level feature selection in an encoder–decoder architecture.

## Release contents and availability

This repository contains the data loading and augmentation code, training and evaluation scripts, and selected model components. It is a **partial code release**: the model implementation depends on modules that are not included here. Consequently, the training and evaluation commands below cannot run from this repository alone; the missing components must first be made available separately.

Datasets and pretrained weights are not included in the repository. The original project materials contain the following weights link, whose availability and compatibility have not been verified for this release:

- Baidu Netdisk: https://pan.baidu.com/s/1lfVckkSEck1Jqutit9LBaA
- Extraction code: `uybi`

## Repository structure

```text
.
├── main.py                   # Command-line entry point
├── train.py                  # Training and checkpoint handling
├── valid.py                  # Validation
├── eval.py                   # Evaluation and image output
├── utils.py                  # Metrics and utilities
├── data/
│   ├── __init__.py
│   ├── data_load.py          # Paired-image dataset and data loaders
│   └── data_augment.py       # Paired-image augmentation
└── models/
    ├── former.py             # Encoder–decoder model
    ├── transformer_block.py  # Transformer block
    ├── feed_forward.py       # Axial feature transformation
    └── layer.py              # Patch embedding layers
```

## Environment

The available scripts use Python, PyTorch, torchvision, NumPy, Pillow, scikit-image and TensorBoard. The model uses `torch.nn.functional.scaled_dot_product_attention`, so use a PyTorch version that provides this function (PyTorch 2.0 or later) with a compatible torchvision version. A CUDA-capable GPU can be used for training and evaluation.

Install PyTorch and torchvision for your platform following the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
pip install numpy pillow scikit-image tensorboard
```

An exact environment lockfile is not supplied. The commands have not been tested as a standalone release because the model components described above are absent.

## Dataset preparation

Download the datasets separately and arrange paired images with identical filenames as follows:

```text
dataset-root/
├── train/
│   ├── hazy/
│   └── gt/
└── test/
    ├── hazy/
    └── gt/
```

The loader matches filenames present in both directories and supports PNG, JPG, and JPEG images. Training applies paired random cropping, rotation and flips.

The supplied loader uses the `test` directory for both validation during training and final evaluation. The dataset split and model-selection procedure should therefore be reported accordingly when reproducing experiments. The manuscript describes the benchmark datasets and the real-world split used in the study.

## Training and evaluation

The following examples describe the supplied entry point and require the missing model components and datasets mentioned above:

```bash
python main.py --mode train --data_dir /path/to/dataset-root \
  --batch_size 4 --learning_rate 1e-4 --num_epoch 30000
```

To resume from a checkpoint, pass `--resume /path/to/checkpoint`. Training uses AdamW, learning-rate warmup followed by cosine annealing, and spatial and frequency-domain reconstruction losses. Checkpoints and training logs are written under the paths configured in the scripts.

For evaluation, provide a compatible checkpoint:

```bash
python main.py --mode test --data_dir /path/to/dataset-root \
  --test_model /path/to/checkpoint
```

The evaluation script computes PSNR and SSIM for paired images and writes results under `results/`. See `python main.py --help` and the scripts for the full set of arguments and defaults, once the missing model components are available.

## Citation

If you use this work, please cite the manuscript **FACTformer: Frequency-Aware Calibration Transformer for Image Dehazing**. Publication details can be added once available.
