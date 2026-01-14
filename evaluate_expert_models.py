"""
Evaluation and Comparison script for all Expert Model Contributions.

This script:
1. Loads all trained models (baseline, uncertainty, gnn, contrastive)
2. Evaluates them on the test set
3. Compares performance across all metrics
4. Generates comparison tables and visualizations

Usage:
    python evaluate_expert_models.py --test_all
    python evaluate_expert_models.py --model_type uncertainty --checkpoint path/to/model.safetensors
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoModel,
)
from safetensors.torch import load_file

# Import models and dataset
from annotate_retrieve.modeling_expert_model import ExpertModel
from annotate_retrieve.modeling_expert_model_uncertainty import (
    UncertaintyAwareExpertModel,
    UncertaintyMetrics
)
from annotate_retrieve.modeling_expert_model_gnn import HierarchicalExpertModel
from annotate_retrieve.modeling_expert_model_contrastive import ContrastiveExpertModel
from train_expert_models import ExpertModelDataset, collate_fn, OBSERVATION_NAMES


class ModelEvaluator:
    """Comprehensive evaluator for Expert Models"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
    
    def evaluate_model(self, model, data_loader, model_name="model"):
        """
        Evaluate a single model comprehensively.
        
        Returns:
            Dictionary with all metrics
        """
        model.eval()
        all_preds = []
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {model_name}"):
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                logits = model(pixel_values, input_ids, attention_mask)
                probs = torch.sigmoid(logits)
                
                all_logits.append(logits.cpu())
                all_preds.append(probs.cpu())
                all_labels.append(labels.cpu())
        
        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        metrics = self._compute_all_metrics(all_preds, all_labels, all_logits)
        metrics["model_name"] = model_name
        
        self.results[model_name] = metrics
        return metrics
    
    def evaluate_uncertainty_model(self, model, data_loader, model_name="uncertainty"):
        """
        Evaluate uncertainty model with additional calibration metrics.
        """
        model.eval()
        all_preds = []
        all_uncertainties = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {model_name}"):
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                mean_pred, uncertainty, calibrated_pred = model.forward_with_uncertainty(
                    pixel_values, input_ids, attention_mask
                )
                
                all_preds.append(calibrated_pred.cpu())
                all_uncertainties.append(uncertainty.cpu())
                all_labels.append(labels.cpu())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_uncertainties = torch.cat(all_uncertainties, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Standard metrics
        metrics = self._compute_all_metrics(
            all_preds.numpy(), 
            all_labels.numpy(),
            all_preds.numpy()  # Use probs as logits for uncertainty model
        )
        
        # Uncertainty-specific metrics
        metrics["ece"] = UncertaintyMetrics.expected_calibration_error(
            all_preds, all_uncertainties, all_labels
        )
        metrics["uncertainty_correlation"] = UncertaintyMetrics.compute_uncertainty_quality(
            all_preds, all_uncertainties, all_labels
        )
        metrics["mean_uncertainty"] = all_uncertainties.mean().item()
        metrics["std_uncertainty"] = all_uncertainties.std().item()
        
        # Per-class uncertainty
        metrics["per_class_uncertainty"] = all_uncertainties.mean(dim=0).tolist()
        
        metrics["model_name"] = model_name
        self.results[model_name] = metrics
        
        return metrics
    
    def _compute_all_metrics(self, predictions, labels, logits):
        """Compute comprehensive metrics"""
        binary_preds = (predictions > 0.5).astype(float)
        
        # Basic metrics
        tp = (binary_preds * labels).sum(axis=0)
        fp = (binary_preds * (1 - labels)).sum(axis=0)
        fn = ((1 - binary_preds) * labels).sum(axis=0)
        tn = ((1 - binary_preds) * (1 - labels)).sum(axis=0)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        
        # Macro and Micro metrics
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()
        
        micro_tp = tp.sum()
        micro_fp = fp.sum()
        micro_fn = fn.sum()
        micro_precision = micro_tp / (micro_tp + micro_fp + 1e-8)
        micro_recall = micro_tp / (micro_tp + micro_fn + 1e-8)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
        
        # AUC-ROC (per class and macro)
        auc_scores = []
        for i in range(labels.shape[1]):
            if labels[:, i].sum() > 0 and labels[:, i].sum() < len(labels):
                auc = roc_auc_score(labels[:, i], predictions[:, i])
                auc_scores.append(auc)
            else:
                auc_scores.append(0.0)
        
        macro_auc = np.mean([a for a in auc_scores if a > 0])
        
        # Average Precision (per class and macro)
        ap_scores = []
        for i in range(labels.shape[1]):
            if labels[:, i].sum() > 0:
                ap = average_precision_score(labels[:, i], predictions[:, i])
                ap_scores.append(ap)
            else:
                ap_scores.append(0.0)
        
        macro_ap = np.mean([a for a in ap_scores if a > 0])
        
        # 5-class metrics (common observations)
        common_indices = [7, 1, 5, 4, 9]  # Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion
        five_class_f1 = f1[common_indices].mean()
        five_class_auc = np.mean([auc_scores[i] for i in common_indices if auc_scores[i] > 0])
        
        return {
            # Macro metrics
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "macro_auc": float(macro_auc),
            "macro_ap": float(macro_ap),
            
            # Micro metrics
            "micro_precision": float(micro_precision),
            "micro_recall": float(micro_recall),
            "micro_f1": float(micro_f1),
            
            # 5-class metrics
            "5class_macro_f1": float(five_class_f1),
            "5class_macro_auc": float(five_class_auc),
            
            # Per-class metrics
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(),
            "per_class_auc": auc_scores,
            "per_class_ap": ap_scores,
            "per_class_specificity": specificity.tolist(),
        }
    
    def compare_models(self):
        """Generate comparison table across all evaluated models"""
        if not self.results:
            print("No models evaluated yet!")
            return None
        
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        # Main metrics table
        headers = ["Model", "Macro-F1", "Micro-F1", "5-Class F1", "Macro-AUC", "Macro-AP"]
        print(f"\n{'Main Metrics':^80}")
        print("-"*80)
        print(f"{'Model':<20} {'Macro-F1':>10} {'Micro-F1':>10} {'5-Class F1':>12} {'Macro-AUC':>12} {'Macro-AP':>10}")
        print("-"*80)
        
        for model_name, metrics in self.results.items():
            print(f"{model_name:<20} {metrics['macro_f1']:>10.4f} {metrics['micro_f1']:>10.4f} "
                  f"{metrics['5class_macro_f1']:>12.4f} {metrics['macro_auc']:>12.4f} {metrics['macro_ap']:>10.4f}")
        
        print("-"*80)
        
        # Uncertainty metrics (if available)
        uncertainty_models = [m for m in self.results if "ece" in self.results[m]]
        if uncertainty_models:
            print(f"\n{'Uncertainty Metrics':^80}")
            print("-"*80)
            print(f"{'Model':<20} {'ECE':>10} {'Uncert-Corr':>12} {'Mean Uncert':>12} {'Std Uncert':>10}")
            print("-"*80)
            
            for model_name in uncertainty_models:
                metrics = self.results[model_name]
                print(f"{model_name:<20} {metrics['ece']:>10.4f} {metrics['uncertainty_correlation']:>12.4f} "
                      f"{metrics['mean_uncertainty']:>12.4f} {metrics['std_uncertainty']:>10.4f}")
            
            print("-"*80)
        
        # Per-class F1 comparison
        print(f"\n{'Per-Class F1 Scores':^80}")
        print("-"*80)
        header = f"{'Observation':<30}"
        for model_name in self.results:
            header += f" {model_name[:10]:>10}"
        print(header)
        print("-"*80)
        
        for i, obs_name in enumerate(OBSERVATION_NAMES):
            row = f"{obs_name:<30}"
            for model_name in self.results:
                f1 = self.results[model_name]["per_class_f1"][i]
                row += f" {f1:>10.4f}"
            print(row)
        
        print("-"*80)
        
        return self.results
    
    def save_results(self, output_path):
        """Save results to JSON file"""
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def load_model(model_type, checkpoint_path, config, text_model, device):
    """Load a trained model from checkpoint"""
    if model_type == "baseline":
        model = ExpertModel(config=config, text_model=text_model)
    elif model_type == "uncertainty":
        model = UncertaintyAwareExpertModel(config=config, text_model=text_model)
    elif model_type == "gnn":
        model = HierarchicalExpertModel(config=config, text_model=text_model)
    elif model_type == "contrastive":
        model = ContrastiveExpertModel(config=config, text_model=text_model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
    
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Expert Models")
    
    parser.add_argument("--model_type", type=str, default=None,
                        choices=["baseline", "uncertainty", "gnn", "contrastive"],
                        help="Type of model to evaluate (single model)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--test_all", action="store_true",
                        help="Evaluate all available models")
    
    # Data arguments
    parser.add_argument("--image_path", type=str, default="./data/mimic_cxr/images/")
    parser.add_argument("--annotation_path", type=str, default="./data/mimic_cxr/annotation.json")
    parser.add_argument("--clinical_context_path", type=str, default="./data/mimic_cxr/clinical_context.json")
    parser.add_argument("--observation_path", type=str, default="./data/mimic_cxr/observation.json")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/expert_models/")
    
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_path", type=str, default="./results/expert_model_comparison.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Initialize
    vision_model_name = "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft"
    text_model_name = "emilyalsentzer/Bio_ClinicalBERT"
    
    print("Loading tokenizer and processor...")
    processor = AutoFeatureExtractor.from_pretrained(vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name)
    
    config = AutoConfig.from_pretrained(vision_model_name)
    config.num_observation = 14
    config.pretrained_visual_extractor = vision_model_name
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = ExpertModelDataset(
        root_path=args.image_path,
        annotation_path=args.annotation_path,
        clinical_context_path=args.clinical_context_path,
        observation_path=args.observation_path,
        tokenizer=tokenizer,
        processor=processor,
        split="test"
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(device=args.device)
    
    if args.test_all:
        # Evaluate all available models
        model_types = ["baseline", "uncertainty", "gnn", "contrastive"]
        
        for model_type in model_types:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"best_{model_type}_model.safetensors")
            
            if os.path.exists(checkpoint_path):
                print(f"\n{'='*60}")
                print(f"Evaluating {model_type.upper()} model")
                print(f"{'='*60}")
                
                # Need fresh text_model for each
                text_model_fresh = AutoModel.from_pretrained(text_model_name)
                model = load_model(model_type, checkpoint_path, config, text_model_fresh, args.device)
                
                if model_type == "uncertainty":
                    evaluator.evaluate_uncertainty_model(model, test_loader, model_type)
                else:
                    evaluator.evaluate_model(model, test_loader, model_type)
            else:
                print(f"Checkpoint not found for {model_type}: {checkpoint_path}")
        
        # Compare all models
        evaluator.compare_models()
        
    elif args.model_type and args.checkpoint:
        # Evaluate single model
        print(f"\nEvaluating {args.model_type} model from {args.checkpoint}")
        model = load_model(args.model_type, args.checkpoint, config, text_model, args.device)
        
        if args.model_type == "uncertainty":
            metrics = evaluator.evaluate_uncertainty_model(model, test_loader, args.model_type)
        else:
            metrics = evaluator.evaluate_model(model, test_loader, args.model_type)
        
        print(f"\nResults for {args.model_type}:")
        print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"  Micro-F1: {metrics['micro_f1']:.4f}")
        print(f"  5-Class F1: {metrics['5class_macro_f1']:.4f}")
        print(f"  Macro-AUC: {metrics['macro_auc']:.4f}")
    
    else:
        print("Please specify --test_all or both --model_type and --checkpoint")
        return
    
    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    evaluator.save_results(args.output_path)


if __name__ == "__main__":
    main()
