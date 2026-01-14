# Expert Model Contributions - Implementation Guide

This document describes the implementation of 3 research contributions for the RADAR Expert Model, designed to run on a **GTX 1650 Ti (4GB VRAM)**.

## Overview of Contributions

| Contribution | File | Description |
|-------------|------|-------------|
| **1.1** | `modeling_expert_model_uncertainty.py` | Uncertainty-Aware Expert Model with MC Dropout |
| **1.2** | `modeling_expert_model_gnn.py` | Hierarchical Multi-Label Classification with GNN |
| **1.3** | `modeling_expert_model_contrastive.py` | Contrastive Learning Pre-training |

---

## Quick Start

### 1. Install Dependencies

```bash
conda activate radar
pip install torch-geometric  # For GNN (optional, basic GNN included)
```

### 2. Prepare Data

You need the following files in `./data/mimic_cxr/`:
- `annotation.json` - Image annotations
- `clinical_context.json` - Clinical context
- `observation.json` - Observation labels
- `images/` - Image folder

### 3. Train Models

```bash
# Train Uncertainty-Aware Model (Contribution 1.1)
python train_expert_models.py --model_type uncertainty --batch_size 8 --epochs 10

# Train GNN Model (Contribution 1.2)
python train_expert_models.py --model_type gnn --batch_size 8 --epochs 10

# Train Contrastive Model (Contribution 1.3)
python train_expert_models.py --model_type contrastive --batch_size 4 --epochs 10 --pretrain
```

### 4. Evaluate and Compare

```bash
# Evaluate all models
python evaluate_expert_models.py --test_all

# Evaluate single model
python evaluate_expert_models.py --model_type uncertainty --checkpoint ./checkpoints/expert_models/best_uncertainty_model.safetensors
```

---

## Contribution 1.1: Uncertainty-Aware Expert Model

### Key Features
- **Monte Carlo Dropout**: Multiple forward passes with dropout enabled during inference
- **Temperature Scaling**: Learnable temperature parameter for probability calibration
- **Confidence Thresholds**: Observation-specific learnable thresholds

### Usage

```python
from annotate_retrieve.modeling_expert_model_uncertainty import UncertaintyAwareExpertModel

model = UncertaintyAwareExpertModel(
    config=config,
    text_model=text_model,
    num_mc_samples=10,  # Number of MC samples
    dropout_rate=0.1
)

# Standard forward pass
logits = model(images, input_ids, attention_mask)

# Forward with uncertainty estimation
mean_pred, uncertainty, calibrated_pred = model.forward_with_uncertainty(
    images, input_ids, attention_mask
)

# Get high-confidence observations
confident_mask, confidence_scores = model.get_confident_observations(
    mean_pred, uncertainty
)
```

### New Metrics
- **ECE (Expected Calibration Error)**: Measures probability calibration
- **Uncertainty-Error Correlation**: Higher = better uncertainty estimates

---

## Contribution 1.2: Hierarchical Multi-Label Classification with GNN

### Key Features
- **Graph Attention Network**: Models dependencies between observations
- **Clinical Knowledge Graph**: Encodes known medical relationships
- **Consistency Loss**: Penalizes impossible observation combinations
- **Correlation Loss**: Encourages correlated observations to have similar predictions

### Clinical Relationships Encoded
```
Cardiomegaly ←→ Edema ←→ Pleural Effusion
Consolidation ←→ Pneumonia ←→ Lung Opacity
Atelectasis ←→ Lung Opacity
No Finding ⊥ All other observations (mutually exclusive)
```

### Usage

```python
from annotate_retrieve.modeling_expert_model_gnn import HierarchicalExpertModel, HierarchicalLoss

model = HierarchicalExpertModel(
    config=config,
    text_model=text_model,
    num_gnn_layers=2
)

criterion = HierarchicalLoss(
    bce_weight=1.0,
    consistency_weight=0.1,
    correlation_weight=0.05
)

logits = model(images, input_ids, attention_mask)
loss, loss_dict = criterion(logits, labels, model)
```

---

## Contribution 1.3: Contrastive Learning Pre-training

### Key Features
- **CLIP-style Contrastive Loss**: Image-text alignment
- **Hard Negative Mining**: Focus on similar but different observations
- **Multi-view Augmentation**: Data augmentation for robustness
- **Two-phase Training**: Pre-training + Fine-tuning

### Training Phases
1. **Pre-training**: Learn image-text alignment with contrastive loss
2. **Fine-tuning**: Standard classification with BCE loss

### Usage

```python
from annotate_retrieve.modeling_expert_model_contrastive import ContrastiveExpertModel, ContrastiveLoss

model = ContrastiveExpertModel(
    config=config,
    text_model=text_model,
    projection_dim=256
)

# Contrastive forward (for pre-training)
image_embeds, text_embeds, logit_scale = model.contrastive_forward(
    images, input_ids, attention_mask
)

criterion = ContrastiveLoss(use_hard_negatives=True)
loss, loss_dict = criterion(image_embeds, text_embeds, logit_scale, labels)

# Standard forward (for fine-tuning)
logits = model(images, input_ids, attention_mask)
```

---

## Memory Requirements (GTX 1650 Ti - 4GB)

| Model | Batch Size | Est. Memory | Status |
|-------|-----------|-------------|--------|
| Baseline | 8 | ~3.5 GB | ✅ Works |
| Uncertainty | 8 | ~3.8 GB | ✅ Works |
| GNN | 8 | ~3.9 GB | ✅ Works |
| Contrastive | 4-6 | ~3.5 GB | ✅ Works |

**Tips for low memory:**
- Reduce batch size to 4
- Use gradient accumulation
- Disable `num_workers` in DataLoader

---

## Expected Results

Based on the RADAR paper and typical improvements:

| Model | Expected Macro-F1 | Expected Improvement |
|-------|------------------|---------------------|
| Baseline | ~0.45 | - |
| Uncertainty | ~0.47 | +2-4% |
| GNN | ~0.48 | +3-5% |
| Contrastive | ~0.49 | +4-7% |

---

## File Structure

```
Radar-main/
├── annotate_retrieve/
│   ├── modeling_expert_model.py           # Original baseline
│   ├── modeling_expert_model_uncertainty.py  # Contribution 1.1
│   ├── modeling_expert_model_gnn.py          # Contribution 1.2
│   └── modeling_expert_model_contrastive.py  # Contribution 1.3
├── train_expert_models.py                 # Training script
├── evaluate_expert_models.py              # Evaluation script
├── demo_expert_models.py                  # Demo script
└── CONTRIBUTIONS_README.md                # This file
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train_expert_models.py --model_type uncertainty --batch_size 4

# Or use CPU (slow but works)
python train_expert_models.py --model_type uncertainty --device cpu
```

### Missing Data Files
Make sure you have:
1. Downloaded MIMIC-CXR dataset
2. Run the data preparation scripts from the original README
3. Generated `observation.json` using `annotate_reference.py`

### Import Errors
```bash
# Make sure you're in the project root
cd Radar-main
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Citation

If you use these contributions, please cite the original RADAR paper:

```bibtex
@misc{hou2025radarenhancingradiologyreport,
      title={RADAR: Enhancing Radiology Report Generation with Supplementary Knowledge Injection},
      author={Wenjun Hou and Yi Cheng and Kaishuai Xu and Heng Li and Yan Hu and Wenjie Li and Jiang Liu},
      year={2025},
      eprint={2505.14318},
      archivePrefix={arXiv},
}
```
