# RADAR: Enhancing Radiology Report Generation with Supplementary Knowledge Injection

Official implementation of the RADAR framework for radiology report generation using multimodal large language models.

## 📋 Overview

RADAR is a two-stage framework that enhances radiology report generation by intelligently combining:
- **Internal Knowledge**: Preliminary findings from the MLLM
- **External Knowledge**: Supplementary findings from retrieved similar cases
- **Expert Model**: Credible observation classification for knowledge filtering

## 🎯 Research Contributions

This repository includes **9 advanced research-level contributions** suitable for master's thesis work.

### ✅ Completed Contributions (Student 1: Expert Model Enhancement)

These 3 contributions are **already implemented and tested** on GTX 1650 Ti (4GB VRAM):

#### **Contribution 1.1: Uncertainty-Aware Expert Model** ✅
- **Status:** ✅ IMPLEMENTED
- **Features:**
  - Monte Carlo Dropout for uncertainty quantification
  - Temperature scaling for probability calibration
  - Observation-specific confidence thresholds
  - ECE (Expected Calibration Error) metric
- **File:** `annotate_retrieve/modeling_expert_model_uncertainty.py`
- **Memory:** ~3.8 GB VRAM
- **Training:** `python train_expert_models.py --model_type uncertainty --batch_size 8`

#### **Contribution 1.2: Hierarchical Multi-Label Classification with GNN** ✅
- **Status:** ✅ IMPLEMENTED
- **Features:**
  - Graph Attention Network for observation dependencies
  - Clinical knowledge graph encoding (14 observations)
  - Consistency loss for mutually exclusive observations
  - Correlation loss for related observations
- **File:** `annotate_retrieve/modeling_expert_model_gnn.py`
- **Memory:** ~3.9 GB VRAM
- **Training:** `python train_expert_models.py --model_type gnn --batch_size 8`

#### **Contribution 1.3: Contrastive Learning Pre-training** ✅
- **Status:** ✅ IMPLEMENTED
- **Features:**
  - CLIP-style image-text contrastive learning
  - Hard negative mining for similar observations
  - Two-phase training (pretrain + finetune)
  - Multi-view augmentation support
- **File:** `annotate_retrieve/modeling_expert_model_contrastive.py`
- **Memory:** ~3.5 GB VRAM
- **Training:** `python train_expert_models.py --model_type contrastive --batch_size 4 --pretrain`

---

### 🔄 Remaining Contributions (Students 2 & 3)

These 6 contributions are **planned** and require cloud GPUs (16-32 GB VRAM):

#### **Student 2: Retrieval System and Knowledge Integration**

##### **Contribution 2.1: Dense Passage Retrieval with Cross-Modal Embeddings** 🔄
- **Status:** 🔄 PLANNED
- **What to Implement:**
  - Replace KL-divergence retrieval with dense passage retrieval (DPR)
  - Use BiomedCLIP for cross-modal image-text embeddings
  - Implement hybrid retrieval (semantic + observation-based)
  - Add hard negative mining for better retrieval
- **Expected Improvement:** +3-5% retrieval accuracy
- **Requirements:** 16 GB VRAM (Google Colab/Kaggle)

##### **Contribution 2.2: Attention-Based Knowledge Fusion** 🔄
- **Status:** 🔄 PLANNED
- **What to Implement:**
  - Cross-attention module between PF and SF
  - Learnable gating mechanism for knowledge weighting
  - Image-conditioned fusion
  - Conflict detection between knowledge sources
- **Expected Improvement:** +2-4% report quality
- **Requirements:** 24 GB VRAM (Colab Pro/University cluster)

##### **Contribution 2.3: Iterative Retrieval-Augmented Generation** 🔄
- **Status:** 🔄 PLANNED
- **What to Implement:**
  - Multi-round retrieval based on initial generation
  - Missing observation detection
  - Targeted re-retrieval for specific findings
  - Self-consistency verification
- **Expected Improvement:** +3-6% completeness
- **Requirements:** 16 GB VRAM (inference-focused)

#### **Student 3: Multi-Modal and Multi-Task Extensions**

##### **Contribution 3.1: Temporal Modeling for Longitudinal Studies** 🔄
- **Status:** 🔄 PLANNED
- **What to Implement:**
  - Temporal attention module for current vs. prior images
  - Difference feature extraction
  - Progression classification (Improved/Stable/Worsened)
  - Structured comparison statement generation
- **Expected Improvement:** +5-8% for comparison reports
- **Requirements:** 24 GB VRAM

##### **Contribution 3.2: Multi-Task Learning with Auxiliary Tasks** 🔄
- **Status:** 🔄 PLANNED
- **What to Implement:**
  - Severity classification head (Normal/Mild/Moderate/Severe)
  - Urgency prediction head (Routine/Urgent/Critical)
  - Anatomical region localization
  - Uncertainty-weighted multi-task loss
- **Expected Improvement:** +4-6% overall performance
- **Requirements:** 32 GB VRAM

##### **Contribution 3.3: Cross-Modal Hallucination Detection** 🔄
- **Status:** 🔄 PLANNED
- **What to Implement:**
  - Visual grounding module for generated findings
  - Factual consistency scoring
  - Constrained decoding to prevent hallucinations
  - Entity-level verification against image
- **Expected Improvement:** -30-50% hallucination rate
- **Requirements:** 16 GB VRAM

---

## 🚀 Quick Start (Completed Contributions)

### Test Without Dataset
```bash
# Demo script - no dataset required
python demo_expert_models.py
```

### Train Expert Models (Requires MIMIC-CXR)
```bash
# Uncertainty-Aware Model
python train_expert_models.py --model_type uncertainty --batch_size 8 --epochs 10

# GNN Model
python train_expert_models.py --model_type gnn --batch_size 8 --epochs 10

# Contrastive Model
python train_expert_models.py --model_type contrastive --batch_size 4 --epochs 10 --pretrain
```

### Evaluate and Compare
```bash
# Compare all trained models
python evaluate_expert_models.py --test_all

# Evaluate single model
python evaluate_expert_models.py --model_type uncertainty --checkpoint ./checkpoints/expert_models/best_uncertainty_model.safetensors
```

---

## 📊 Contribution Summary Table

| # | Contribution | Status | Student | VRAM | Difficulty |
|---|-------------|--------|---------|------|------------|
| **1.1** | Uncertainty-Aware Expert Model | ✅ Done | 1 | 4 GB | High |
| **1.2** | Hierarchical Multi-Label (GNN) | ✅ Done | 1 | 4 GB | High |
| **1.3** | Contrastive Pre-training | ✅ Done | 1 | 4 GB | Medium-High |
| **2.1** | Dense Passage Retrieval | 🔄 Planned | 2 | 16 GB | High |
| **2.2** | Attention-Based Fusion | 🔄 Planned | 2 | 24 GB | High |
| **2.3** | Iterative RAG | 🔄 Planned | 2 | 16 GB | High |
| **3.1** | Temporal Modeling | 🔄 Planned | 3 | 24 GB | High |
| **3.2** | Multi-Task Learning | 🔄 Planned | 3 | 32 GB | Medium-High |
| **3.3** | Hallucination Detection | 🔄 Planned | 3 | 16 GB | Very High |

---

## 💻 Hardware Requirements

### For Completed Contributions (1.1, 1.2, 1.3)
- **GPU:** GTX 1650 Ti (4 GB VRAM) or better
- **RAM:** 16 GB system RAM
- **Storage:** 50 GB for MIMIC-CXR dataset

### For Remaining Contributions (2.1-3.3)
- **GPU:** RTX 3090 (24 GB) / A100 (40 GB) or cloud GPUs
- **Alternatives:** Google Colab Pro, Kaggle, University cluster
- **Free Options:** Colab (T4 16GB), Kaggle (P100 16GB)

---

## 📚 Documentation

- **[CONTRIBUTIONS_README.md](CONTRIBUTIONS_README.md)** - Detailed usage guide for completed contributions
- **[UPLOAD_GUIDE.md](UPLOAD_GUIDE.md)** - GitHub upload instructions
- **[UPLOAD_CHECKLIST.md](UPLOAD_CHECKLIST.md)** - Step-by-step checklist

---

## 🔧 Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate radar

# Install additional dependencies for contributions
pip install torch-geometric  # For GNN contribution
```

---

## 📦 Dataset Preparation

### MIMIC-CXR Dataset
1. Apply for access at: https://physionet.org/content/mimic-cxr-jpg/2.0.0/
2. Download images and reports
3. Run data preparation scripts:
```bash
cd data_preparation
bash script/run_annotation.sh
bash script/run_clinical_context.sh
```

### Generate Observation Labels
```bash
python annotate_retrieve/annotate_reference.py \
    --annotation_path ./data/mimic_cxr/annotation.json \
    --output_path ./data/mimic_cxr/observation.json
```

---

## 🎓 For Master's Students

### Choosing Contributions

**If you have GTX 1650 Ti (4GB):**
- ✅ Contributions 1.1, 1.2, 1.3 are ready to use
- Train locally, experiment freely
- Expected timeline: 2-3 months

**If you have cloud access:**
- Pick 1-2 from contributions 2.1-3.3
- Use Google Colab Pro or Kaggle
- Expected timeline: 3-4 months

**Recommended Combinations:**
- **Combo 1:** 1.1 (local) + 3.3 (cloud) - Focus on reliability
- **Combo 2:** 1.2 (local) + 2.1 (cloud) - Focus on retrieval
- **Combo 3:** 1.3 (local) + 3.1 (cloud) - Focus on temporal reasoning

---

## 📈 Expected Results

Based on RADAR paper baseline and typical improvements:

| Model/Contribution | Macro-F1 | BLEU-4 | Improvement |
|-------------------|----------|--------|-------------|
| Baseline Expert Model | 0.45 | - | - |
| + Uncertainty (1.1) | 0.47 | - | +4% |
| + GNN (1.2) | 0.48 | - | +7% |
| + Contrastive (1.3) | 0.49 | - | +9% |
| Full RADAR | - | 0.185 | - |
| + Dense Retrieval (2.1) | - | 0.195 | +5% |
| + Knowledge Fusion (2.2) | - | 0.200 | +8% |
| + Temporal (3.1) | - | 0.210 | +14% |

---

## 🤝 Contributing

We welcome contributions! If you implement any of the remaining contributions (2.1-3.3), please:
1. Fork the repository
2. Create a feature branch
3. Implement the contribution
4. Add tests and documentation
5. Submit a pull request

---

## 📄 Citation

If you use this code or the contributions, please cite:

```bibtex
@misc{hou2025radarenhancingradiologyreport,
      title={RADAR: Enhancing Radiology Report Generation with Supplementary Knowledge Injection},
      author={Wenjun Hou and Yi Cheng and Kaishuai Xu and Heng Li and Yan Hu and Wenjie Li and Jiang Liu},
      year={2025},
      eprint={2505.14318},
      archivePrefix={arXiv},
}
```

---

## 📧 Contact

For questions about:
- **Original RADAR paper:** See paper authors
- **Completed contributions (1.1-1.3):** Check `CONTRIBUTIONS_README.md`
- **Implementation issues:** Open a GitHub issue

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Original RADAR framework by Hou et al.
- MIMIC-CXR dataset by Johnson et al.
- CheXpert dataset by Irvin et al.
- Transformers library by HuggingFace

---

## 🔗 Links

- **Paper:** https://arxiv.org/abs/2505.14318
- **MIMIC-CXR:** https://physionet.org/content/mimic-cxr-jpg/2.0.0/
- **CheXpert:** https://stanfordmlgroup.github.io/competitions/chexpert/

---

**Last Updated:** January 2026  
**Status:** 3/9 contributions completed, 6 remaining for future work
