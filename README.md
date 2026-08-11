# Analysis of Semantic Segmentation under Adverse Weather Conditions

This repository contains the implementation for the dissertation project:

**Analysis of Semantic Segmentation under Adverse Weather Conditions**

The project evaluates semantic segmentation models for road-scene understanding under clean and adverse weather conditions. The main focus is on adverse-weather robustness, safety-critical class performance, mitigation experiments, and real-time inference efficiency.

## 1. Project Objective

Semantic segmentation assigns a semantic class label to every pixel in an image. In ADAS and autonomous driving systems, segmentation must remain reliable under difficult visual conditions such as fog, rain, snow, and low light.

This project studies the following question:

**Which segmentation architecture provides the best balance of adverse-weather robustness, safety-critical class performance, and real-time inference efficiency?**

## 2. Models Used

The following models are implemented and evaluated:

- U-Net with ResNet-50 encoder
- DeepLabV3+ with ResNet-50 backbone
- SegFormer-B2

## 3. Datasets Used

The project uses:

- IDD: clean-weather Indian road-scene dataset {IDD Segmentation (IDD 20k Part I) & IDD Segmentation (IDD 20k Part II)}, Download Link: https://idd.insaan.iiit.ac.in/dataset/download/
- IDD-AW: adverse-weather Indian road-scene dataset, Download Link: https://idd.insaan.iiit.ac.in/dataset/download/

The adverse-weather categories are:

- Fog
- Rain
- Snow
- Lowlight

The repository does not include datasets because of size and licensing constraints, please download the datasets from the links provided above and follow the below provided structure.

Expected local dataset structure:

```text
dissertation/data/
├── idd/
│   └── IDD_Segmentation/
│       ├── leftImg8bit/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── gtFine/
│           ├── train/
│           └── val/
└── idd_aw/
    └── IDDAW/
        ├── train/
        │   ├── FOG/
        │   ├── LOWLIGHT/
        │   ├── RAIN/
        │   └── SNOW/
        ├── val/
        │   ├── FOG/
        │   ├── LOWLIGHT/
        │   ├── RAIN/
        │   └── SNOW/
        └── test/
```

## 4. Repository Structure

```text
dissertation/
├── configs/              # YAML configuration files
├── datasets/             # Dataset loaders
├── dataloaders/          # Additional data loading utilities
├── models/               # Model definitions
├── training/             # Training scripts
├── evaluation/           # Evaluation, plotting, and benchmarking scripts
├── utils/                # Utility functions and transforms
├── scripts/              # Data preparation scripts
├── inference/            # Inference utilities
├── explainability/       # Explainability-related scripts
├── unit_tests/           # Basic test scripts
└── results/
    └── summary/          # Final summarized CSV files and plots
```

Large files are intentionally excluded from GitHub:

- Raw datasets
- Processed masks
- Pretrained weights
- Model checkpoints
- Virtual environment
- Large experiment folders

## 5. Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch version compatible with the local CUDA driver.

## 6. Data Preparation

Generate segmentation masks before training.

Example:

```bash
python -m dissertation.scripts.generate_masks
```

The generated masks are expected under:

```text
dissertation/data/processed/
```

Note: The exact dataset paths are controlled through the YAML config files under `dissertation/configs/`.

## 7. Training Commands

### 7.1 Train U-Net on IDD-AW

```bash
python -m dissertation.training.train_unet \
    --config dissertation/configs/unet.yaml
```

### 7.2 Train U-Net on clean IDD

```bash
python -m dissertation.training.train_unet_idd_clean \
    --config dissertation/configs/unet_idd_clean.yaml
```

### 7.3 Train DeepLabV3+ on IDD-AW

```bash
python -m dissertation.training.train_deeplabv3plus \
    --config dissertation/configs/deeplabv3plus.yaml
```

### 7.4 Train DeepLabV3+ on clean IDD

```bash
python -m dissertation.training.train_deeplabv3plus_idd_clean \
    --config dissertation/configs/deeplabv3plus_idd_clean.yaml
```

### 7.5 Train SegFormer-B2 on IDD-AW

```bash
python -m dissertation.training.train_segformer \
    --config dissertation/configs/segformer.yaml
```

### 7.6 Train SegFormer-B2 on clean IDD

```bash
python -m dissertation.training.train_segformer_idd_clean \
    --config dissertation/configs/segformer_idd_clean.yaml
```

## 8. Basic Evaluation Commands

### 8.1 Evaluate U-Net on IDD-AW

```bash
python -m dissertation.evaluation.eval_unet \
    --config dissertation/configs/unet.yaml \
    --checkpoint dissertation/results/unet/best_checkpoint.pth
```

### 8.2 Evaluate U-Net on clean IDD

```bash
python -m dissertation.evaluation.eval_unet_idd_clean \
    --config dissertation/configs/unet_idd_clean.yaml \
    --checkpoint dissertation/results/unet_idd_clean/best_checkpoint.pth
```

### 8.3 Evaluate DeepLabV3+ on IDD-AW

```bash
python -m dissertation.evaluation.eval_deeplabv3plus \
    --config dissertation/configs/deeplabv3plus.yaml \
    --checkpoint dissertation/results/deeplabv3plus/best_checkpoint.pth
```

### 8.4 Evaluate DeepLabV3+ on clean IDD

```bash
python -m dissertation.evaluation.eval_deeplabv3plus_idd_clean \
    --config dissertation/configs/deeplabv3plus_idd_clean.yaml \
    --checkpoint dissertation/results/deeplabv3plus_idd_clean/best_checkpoint.pth
```

### 8.5 Evaluate SegFormer-B2 on IDD-AW

```bash
python -m dissertation.evaluation.eval_segformer \
    --config dissertation/configs/segformer.yaml \
    --checkpoint dissertation/results/segformer/best_checkpoint.pth
```

### 8.6 Evaluate SegFormer-B2 on clean IDD

```bash
python -m dissertation.evaluation.eval_segformer_idd_clean \
    --config dissertation/configs/segformer_idd_clean.yaml \
    --checkpoint dissertation/results/segformer_idd_clean/best_checkpoint.pth
```

## 9. Weather-wise and Safety-critical Evaluation

### 9.1 U-Net Detailed Evaluation

```bash
python -m dissertation.evaluation.eval_unet_iddaw_detailed \
    --config dissertation/configs/unet.yaml \
    --checkpoint dissertation/results/unet/best_checkpoint.pth
```

### 9.2 DeepLabV3+ Detailed Evaluation

```bash
python -m dissertation.evaluation.eval_deeplabv3plus_iddaw_detailed \
    --config dissertation/configs/deeplabv3plus.yaml \
    --checkpoint dissertation/results/deeplabv3plus/best_checkpoint.pth
```

### 9.3 SegFormer-B2 Detailed Evaluation

```bash
python -m dissertation.evaluation.eval_segformer_iddaw_detailed \
    --config dissertation/configs/segformer.yaml \
    --checkpoint dissertation/results/segformer/best_checkpoint.pth
```

These scripts generate weather-wise mIoU and safety-critical class IoU summaries.

## 10. Mitigation Experiments

Two mitigation approaches were evaluated:

1. Class-weighted cross entropy
2. Weak safety-critical oversampling fine-tuning

The weak safety-critical classes considered include:

- Sidewalk
- Traffic light
- Traffic sign
- Person
- Rider
- Truck
- Motorcycle
- Bicycle

### 10.1 U-Net Weak Safety-critical Fine-tuning

```bash
python -m dissertation.training.train_weak_safety_finetune \
    --model unet \
    --epochs 10 \
    --oversample_factor 3 \
    --lr 5e-6 \
    --checkpoint dissertation/results/unet/best_checkpoint.pth \
    --output_dir dissertation/results/unet_weak_safety_finetune
```

### 10.2 DeepLabV3+ Weak Safety-critical Fine-tuning

```bash
python -m dissertation.training.train_weak_safety_finetune \
    --model deeplabv3plus \
    --epochs 10 \
    --oversample_factor 3 \
    --lr 5e-6 \
    --checkpoint dissertation/results/deeplabv3plus/best_checkpoint.pth \
    --output_dir dissertation/results/deeplabv3plus_weak_safety_finetune
```

### 10.3 SegFormer-B2 Weak Safety-critical Fine-tuning

```bash
python -m dissertation.training.train_weak_safety_finetune \
    --model segformer \
    --epochs 10 \
    --oversample_factor 3 \
    --lr 5e-6 \
    --checkpoint dissertation/results/segformer/best_checkpoint.pth \
    --output_dir dissertation/results/segformer_weak_safety_finetune
```

## 11. Plotting Commands

### 11.1 Plot DeepLabV3+ Detailed Evaluation

```bash
python -m dissertation.evaluation.plot_detailed_deeplabv3plus_eval
```

### 11.2 Plot SegFormer Detailed Evaluation

```bash
python -m dissertation.evaluation.plot_detailed_segformer_eval
```

### 11.3 Plot Mitigation Comparison

```bash
python -m dissertation.evaluation.plot_mitigation_comparison
```

## 12. Real-time Inference Efficiency

The models were benchmarked using two inference modes:

- FP32: reference mode
- AMP/FP16: faster real-time inference mode

Run the benchmark:

```bash
python -m dissertation.evaluation.benchmark_model_efficiency
```

This generates:

```text
dissertation/results/summary/efficiency/pytorch_precision_efficiency_summary.csv
dissertation/results/summary/efficiency/precision_improvement_summary.csv
```

Generate real-time efficiency plots:

```bash
python -m dissertation.evaluation.plot_realtime_efficiency
```

Final plots are saved under:

```text
dissertation/results/summary/efficiency/plots/
```

## 13. Main Results

### 13.1 Robustness Comparison

| Model | Clean to Clean mIoU | Clean to IDD-AW mIoU | Clean-to-Adverse Drop | IDD-AW to IDD-AW mIoU |
|---|---:|---:|---:|---:|
| U-Net ResNet50 | 0.4866 | 0.3444 | 0.1422 | 0.4036 |
| DeepLabV3+ ResNet50 | 0.5096 | 0.3715 | 0.1381 | 0.4495 |
| SegFormer-B2 | 0.5288 | 0.4091 | 0.1197 | 0.4840 |

### 13.2 Real-time AMP/FP16 Results

| Model | IDD-AW mIoU | AMP/FP16 Latency | AMP/FP16 FPS |
|---|---:|---:|---:|
| U-Net ResNet50 | 0.4036 | 10.32 ms | 96.92 |
| DeepLabV3+ ResNet50 | 0.4495 | 11.81 ms | 84.68 |
| SegFormer-B2 | 0.4840 | 20.34 ms | 49.18 |

### 13.3 Mitigation Summary

Class-weighted loss reduced overall mIoU for all three models. Weak safety-critical oversampling fine-tuning was more effective, as it preserved or slightly improved overall performance while improving selected weak safety-critical classes.

## 14. Final Conclusion

SegFormer-B2 achieved the highest adverse-weather segmentation accuracy and the lowest clean-to-adverse robustness drop. U-Net achieved the highest inference speed but had the lowest segmentation accuracy. DeepLabV3+ provided a strong practical balance between accuracy and real-time performance.

All three models achieved more than 30 FPS under AMP/FP16 inference at 512 x 1024 resolution.
