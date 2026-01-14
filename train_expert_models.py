"""
Training script for all 3 Expert Model Contributions:
- Contribution 1.1: Uncertainty-Aware Expert Model
- Contribution 1.2: Hierarchical Multi-Label Classification with GNN
- Contribution 1.3: Contrastive Learning Pre-training

Usage:
    python train_expert_models.py --model_type uncertainty --batch_size 8 --epochs 10
    python train_expert_models.py --model_type gnn --batch_size 8 --epochs 10
    python train_expert_models.py --model_type contrastive --batch_size 4 --epochs 5 --pretrain
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoModel,
)
from safetensors.torch import save_file, load_file

# Import our custom models
from annotate_retrieve.modeling_expert_model import ExpertModel
from annotate_retrieve.modeling_expert_model_uncertainty import (
    UncertaintyAwareExpertModel,
    UncertaintyMetrics
)
from annotate_retrieve.modeling_expert_model_gnn import (
    HierarchicalExpertModel,
    HierarchicalLoss
)
from annotate_retrieve.modeling_expert_model_contrastive import (
    ContrastiveExpertModel,
    ContrastiveLoss,
    MultiViewAugmentation
)


# CheXpert observation names
OBSERVATION_NAMES = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]


class ExpertModelDataset(Dataset):
    """Dataset for training Expert Models"""
    
    def __init__(
        self,
        root_path,
        annotation_path,
        clinical_context_path,
        observation_path,
        tokenizer,
        processor,
        split="train",
        max_text_length=512,
    ):
        self.root_path = Path(root_path)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_text_length = max_text_length
        self.split = split
        
        # Load annotations
        with open(annotation_path, "r") as f:
            annotations = json.load(f)
        
        # Load clinical context
        with open(clinical_context_path, "r") as f:
            self.clinical_contexts = json.load(f)
        
        # Load observation labels
        with open(observation_path, "r") as f:
            self.observations = json.load(f)
        
        # Get data for this split
        split_key = split if split != "valid" else "val"
        self.data = annotations.get(split_key, {})
        
        # Filter samples with valid observations
        self.samples = []
        for idx, sample in self.data.items():
            if "findings" in sample and idx in self.observations:
                self.samples.append({
                    "id": idx,
                    "image_path": sample["image_path"],
                    "observations": self.observations[idx]
                })
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image_path = self.root_path / sample["image_path"]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        # Get clinical context
        study_id = sample["image_path"].split("/")[2] if "/" in sample["image_path"] else sample["id"]
        clinical_context = self.clinical_contexts.get(study_id, {})
        context_text = self._construct_clinical_context(clinical_context)
        
        # Tokenize text
        text_inputs = self.tokenizer(
            context_text,
            padding="max_length",
            max_length=self.max_text_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Create label vector
        labels = torch.zeros(14)
        for obs in sample["observations"]:
            if obs in OBSERVATION_NAMES:
                labels[OBSERVATION_NAMES.index(obs)] = 1.0
        
        return {
            "id": sample["id"],
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "labels": labels
        }
    
    def _construct_clinical_context(self, context):
        """Construct clinical context string"""
        parts = []
        for key in ["Indication", "History", "Comparison", "Technique"]:
            if key in context and context[key]:
                parts.append(f"{key}: {context[key]}")
        return "\n".join(parts) if parts else "No clinical context available."


def collate_fn(batch):
    """Custom collate function"""
    return {
        "ids": [item["id"] for item in batch],
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch])
    }


def create_model(model_type, config, text_model):
    """Create model based on type"""
    if model_type == "baseline":
        return ExpertModel(config=config, text_model=text_model)
    elif model_type == "uncertainty":
        return UncertaintyAwareExpertModel(
            config=config,
            text_model=text_model,
            num_mc_samples=10,
            dropout_rate=0.1
        )
    elif model_type == "gnn":
        return HierarchicalExpertModel(
            config=config,
            text_model=text_model,
            num_gnn_layers=2
        )
    elif model_type == "contrastive":
        return ContrastiveExpertModel(
            config=config,
            text_model=text_model,
            projection_dim=256
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_metrics(predictions, labels):
    """Compute evaluation metrics"""
    predictions = (predictions > 0.5).float()
    
    # Per-class metrics
    tp = (predictions * labels).sum(dim=0)
    fp = (predictions * (1 - labels)).sum(dim=0)
    fn = ((1 - predictions) * labels).sum(dim=0)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Macro and Micro F1
    macro_f1 = f1.mean().item()
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_precision = micro_tp / (micro_tp + micro_fp + 1e-8)
    micro_recall = micro_tp / (micro_tp + micro_fn + 1e-8)
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)).item()
    
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_class_f1": f1.tolist()
    }


def train_baseline(model, train_loader, val_loader, args, device):
    """Train baseline or uncertainty model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_f1 = 0.0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits = model(pixel_values, input_ids, attention_mask)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(torch.sigmoid(logits).detach().cpu())
            train_labels.append(labels.cpu())
            
            pbar.set_postfix({"loss": loss.item()})
        
        scheduler.step()
        
        # Compute training metrics
        train_preds = torch.cat(train_preds, dim=0)
        train_labels = torch.cat(train_labels, dim=0)
        train_metrics = compute_metrics(train_preds, train_labels)
        
        # Validation
        val_metrics, val_loss = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Macro-F1: {train_metrics['macro_f1']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Macro-F1: {val_metrics['macro_f1']:.4f}, Micro-F1: {val_metrics['micro_f1']:.4f}")
        
        # Save best model
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            save_path = os.path.join(args.output_dir, f"best_{args.model_type}_model.safetensors")
            save_file(model.state_dict(), save_path)
            print(f"  Saved best model with Macro-F1: {best_val_f1:.4f}")
    
    return best_val_f1


def train_gnn(model, train_loader, val_loader, args, device):
    """Train GNN model with hierarchical loss"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = HierarchicalLoss(bce_weight=1.0, consistency_weight=0.1, correlation_weight=0.05)
    
    best_val_f1 = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits = model(pixel_values, input_ids, attention_mask)
            loss, loss_dict = criterion(logits, labels, model)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(torch.sigmoid(logits).detach().cpu())
            train_labels.append(labels.cpu())
            
            pbar.set_postfix({
                "loss": loss_dict["total_loss"],
                "bce": loss_dict["bce_loss"],
                "cons": loss_dict["consistency_loss"]
            })
        
        scheduler.step()
        
        train_preds = torch.cat(train_preds, dim=0)
        train_labels = torch.cat(train_labels, dim=0)
        train_metrics = compute_metrics(train_preds, train_labels)
        
        val_metrics, val_loss = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Macro-F1: {train_metrics['macro_f1']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Macro-F1: {val_metrics['macro_f1']:.4f}, Micro-F1: {val_metrics['micro_f1']:.4f}")
        
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            save_path = os.path.join(args.output_dir, f"best_{args.model_type}_model.safetensors")
            save_file(model.state_dict(), save_path)
            print(f"  Saved best model with Macro-F1: {best_val_f1:.4f}")
    
    return best_val_f1


def train_contrastive(model, train_loader, val_loader, args, device):
    """Train contrastive model with pre-training + fine-tuning"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    contrastive_criterion = ContrastiveLoss(use_hard_negatives=True)
    
    # Phase 1: Contrastive Pre-training
    if args.pretrain:
        print("\n=== Phase 1: Contrastive Pre-training ===")
        for epoch in range(args.pretrain_epochs):
            model.train()
            pretrain_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{args.pretrain_epochs}")
            for batch in pbar:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                optimizer.zero_grad()
                image_embeds, text_embeds, logit_scale = model.contrastive_forward(
                    pixel_values, input_ids, attention_mask
                )
                loss, loss_dict = contrastive_criterion(
                    image_embeds, text_embeds, logit_scale, labels
                )
                loss.backward()
                optimizer.step()
                
                pretrain_loss += loss.item()
                pbar.set_postfix({"loss": loss_dict["contrastive_loss"]})
            
            print(f"  Pretrain Loss: {pretrain_loss/len(train_loader):.4f}")
    
    # Phase 2: Fine-tuning
    print("\n=== Phase 2: Fine-tuning ===")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val_f1 = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        pbar = tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits = model(pixel_values, input_ids, attention_mask)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(torch.sigmoid(logits).detach().cpu())
            train_labels.append(labels.cpu())
            
            pbar.set_postfix({"loss": loss.item()})
        
        scheduler.step()
        
        train_preds = torch.cat(train_preds, dim=0)
        train_labels = torch.cat(train_labels, dim=0)
        train_metrics = compute_metrics(train_preds, train_labels)
        
        val_metrics, val_loss = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Macro-F1: {train_metrics['macro_f1']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Macro-F1: {val_metrics['macro_f1']:.4f}, Micro-F1: {val_metrics['micro_f1']:.4f}")
        
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            save_path = os.path.join(args.output_dir, f"best_{args.model_type}_model.safetensors")
            save_file(model.state_dict(), save_path)
            print(f"  Saved best model with Macro-F1: {best_val_f1:.4f}")
    
    return best_val_f1


def evaluate(model, data_loader, device):
    """Evaluate model on validation/test set"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in data_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(pixel_values, input_ids, attention_mask)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    metrics = compute_metrics(all_preds, all_labels)
    avg_loss = total_loss / len(data_loader)
    
    return metrics, avg_loss


def evaluate_uncertainty(model, data_loader, device):
    """Evaluate uncertainty model with calibration metrics"""
    model.eval()
    all_preds = []
    all_uncertainties = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating uncertainty"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
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
    metrics = compute_metrics(all_preds, all_labels)
    
    # Uncertainty metrics
    ece = UncertaintyMetrics.expected_calibration_error(all_preds, all_uncertainties, all_labels)
    uncertainty_correlation = UncertaintyMetrics.compute_uncertainty_quality(
        all_preds, all_uncertainties, all_labels
    )
    
    metrics["ece"] = ece
    metrics["uncertainty_correlation"] = uncertainty_correlation
    metrics["mean_uncertainty"] = all_uncertainties.mean().item()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Expert Models")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="uncertainty",
                        choices=["baseline", "uncertainty", "gnn", "contrastive"],
                        help="Type of model to train")
    
    # Data arguments
    parser.add_argument("--image_path", type=str, default="./data/mimic_cxr/images/",
                        help="Path to images")
    parser.add_argument("--annotation_path", type=str, default="./data/mimic_cxr/annotation.json",
                        help="Path to annotation file")
    parser.add_argument("--clinical_context_path", type=str, default="./data/mimic_cxr/clinical_context.json",
                        help="Path to clinical context file")
    parser.add_argument("--observation_path", type=str, default="./data/mimic_cxr/observation.json",
                        help="Path to observation labels")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--pretrain", action="store_true",
                        help="Enable contrastive pre-training (for contrastive model)")
    parser.add_argument("--pretrain_epochs", type=int, default=5,
                        help="Number of pre-training epochs")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints/expert_models/",
                        help="Output directory for checkpoints")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training {args.model_type.upper()} Expert Model")
    print(f"{'='*60}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    
    # Initialize tokenizer and processor
    vision_model_name = "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft"
    text_model_name = "emilyalsentzer/Bio_ClinicalBERT"
    
    print(f"\nLoading models...")
    processor = AutoFeatureExtractor.from_pretrained(vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name)
    
    # Configure model
    config = AutoConfig.from_pretrained(vision_model_name)
    config.num_observation = 14
    config.pretrained_visual_extractor = vision_model_name
    
    # Create model
    model = create_model(args.model_type, config, text_model)
    model = model.to(args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create datasets
    print(f"\nLoading datasets...")
    train_dataset = ExpertModelDataset(
        root_path=args.image_path,
        annotation_path=args.annotation_path,
        clinical_context_path=args.clinical_context_path,
        observation_path=args.observation_path,
        tokenizer=tokenizer,
        processor=processor,
        split="train"
    )
    
    val_dataset = ExpertModelDataset(
        root_path=args.image_path,
        annotation_path=args.annotation_path,
        clinical_context_path=args.clinical_context_path,
        observation_path=args.observation_path,
        tokenizer=tokenizer,
        processor=processor,
        split="val"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Train model
    print(f"\nStarting training...")
    if args.model_type in ["baseline", "uncertainty"]:
        best_f1 = train_baseline(model, train_loader, val_loader, args, args.device)
    elif args.model_type == "gnn":
        best_f1 = train_gnn(model, train_loader, val_loader, args, args.device)
    elif args.model_type == "contrastive":
        best_f1 = train_contrastive(model, train_loader, val_loader, args, args.device)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best Validation Macro-F1: {best_f1:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print(f"{'='*60}")
    
    # Additional evaluation for uncertainty model
    if args.model_type == "uncertainty":
        print(f"\nEvaluating uncertainty calibration...")
        # Load best model
        best_model_path = os.path.join(args.output_dir, f"best_{args.model_type}_model.safetensors")
        state_dict = load_file(best_model_path)
        model.load_state_dict(state_dict)
        
        uncertainty_metrics = evaluate_uncertainty(model, val_loader, args.device)
        print(f"\nUncertainty Metrics:")
        print(f"  ECE (Expected Calibration Error): {uncertainty_metrics['ece']:.4f}")
        print(f"  Uncertainty-Error Correlation: {uncertainty_metrics['uncertainty_correlation']:.4f}")
        print(f"  Mean Uncertainty: {uncertainty_metrics['mean_uncertainty']:.4f}")


if __name__ == "__main__":
    main()
