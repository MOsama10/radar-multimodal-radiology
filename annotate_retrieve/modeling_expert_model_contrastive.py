import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers import SwinConfig
from transformers import AutoModel


class ContrastiveExpertModel(PreTrainedModel):
    """
    Expert Model with Contrastive Learning Pre-training.
    
    Contribution 1.3: Contrastive Learning for Expert Model
    - CLIP-style contrastive pre-training between images and observation descriptions
    - Hard negative mining for similar but different observations
    - Multi-view augmentation support
    """
    def __init__(self, config: SwinConfig, text_model=None, projection_dim=256):
        super().__init__(config)
        self.text_model = text_model
        self.projection_dim = projection_dim
        
        hidden_size = config.hidden_size + text_model.config.hidden_size
        num_observation = config.num_observation
        
        # Projection heads for contrastive learning
        self.image_projection = nn.Sequential(
            nn.Linear(config.hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_model.config.hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Classification head (for fine-tuning after pre-training)
        self.observation_cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_observation),
        )
        
        # Learnable temperature parameter for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ln(1/0.07)
        
        self.model = AutoModel.from_pretrained(self.config.pretrained_visual_extractor)
        
        # Observation descriptions for contrastive learning
        self.observation_descriptions = self._get_observation_descriptions()
    
    def _get_observation_descriptions(self):
        """
        Clinical descriptions for each observation (for contrastive learning).
        """
        descriptions = {
            0: "Enlarged cardiomediastinum with widened mediastinal contour",
            1: "Cardiomegaly with enlarged cardiac silhouette",
            2: "Lung opacity with increased density in lung parenchyma",
            3: "Lung lesion with focal abnormality or mass",
            4: "Pulmonary edema with interstitial or alveolar fluid accumulation",
            5: "Consolidation with dense opacification of lung tissue",
            6: "Pneumonia with infectious infiltrate in the lungs",
            7: "Atelectasis with collapsed or airless lung tissue",
            8: "Pneumothorax with air in the pleural space",
            9: "Pleural effusion with fluid in the pleural cavity",
            10: "Other pleural abnormality or pleural thickening",
            11: "Fracture with bone discontinuity or break",
            12: "Support devices including tubes, lines, or medical equipment",
            13: "No finding with normal chest radiograph appearance"
        }
        return descriptions
    
    def encode_image(self, input_pixels):
        """Encode image to projection space"""
        image_features = self.model(input_pixels).pooler_output
        image_embeds = self.image_projection(image_features)
        # L2 normalize
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        return image_embeds
    
    def encode_text(self, input_ids, attention_mask):
        """Encode text to projection space"""
        text_features = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output
        text_embeds = self.text_projection(text_features)
        # L2 normalize
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        return text_embeds
    
    def forward(
        self,
        input_pixels: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
    ):
        """Standard classification forward pass"""
        image_embeds = self.model(input_pixels).pooler_output
        text_embeds = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output
        
        combined_embeds = torch.cat((image_embeds, text_embeds), dim=-1)
        observation_cls_logits = self.observation_cls(combined_embeds)
        
        return observation_cls_logits
    
    def contrastive_forward(
        self,
        input_pixels: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
    ):
        """
        Forward pass for contrastive learning.
        
        Returns:
            image_embeds: Normalized image embeddings
            text_embeds: Normalized text embeddings
            logit_scale: Temperature parameter
        """
        image_embeds = self.encode_image(input_pixels)
        text_embeds = self.encode_text(input_ids, attention_mask)
        
        return image_embeds, text_embeds, self.logit_scale.exp()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss with hard negative mining.
    Based on CLIP-style symmetric cross-entropy loss.
    """
    def __init__(self, use_hard_negatives=True, hard_negative_weight=2.0):
        super(ContrastiveLoss, self).__init__()
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_weight = hard_negative_weight
    
    def forward(self, image_embeds, text_embeds, logit_scale, labels=None):
        """
        Compute contrastive loss.
        
        Args:
            image_embeds: [batch_size, projection_dim]
            text_embeds: [batch_size, projection_dim]
            logit_scale: Temperature parameter
            labels: [batch_size, num_observations] for hard negative mining
            
        Returns:
            loss: Contrastive loss value
            loss_dict: Dictionary with detailed losses
        """
        batch_size = image_embeds.size(0)
        
        # Compute similarity matrix
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        
        # Ground truth: diagonal elements are positive pairs
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=image_embeds.device)
        
        # Standard contrastive loss
        loss_i2t = F.cross_entropy(logits_per_image, ground_truth)
        loss_t2i = F.cross_entropy(logits_per_text, ground_truth)
        contrastive_loss = (loss_i2t + loss_t2i) / 2
        
        # Hard negative mining
        if self.use_hard_negatives and labels is not None:
            hard_negative_loss = self._compute_hard_negative_loss(
                logits_per_image, labels
            )
            total_loss = contrastive_loss + self.hard_negative_weight * hard_negative_loss
        else:
            hard_negative_loss = torch.tensor(0.0)
            total_loss = contrastive_loss
        
        loss_dict = {
            'contrastive_loss': contrastive_loss.item(),
            'hard_negative_loss': hard_negative_loss.item() if isinstance(hard_negative_loss, torch.Tensor) else 0.0,
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _compute_hard_negative_loss(self, logits, labels):
        """
        Compute hard negative loss for similar but different observations.
        
        Args:
            logits: Similarity matrix [batch_size, batch_size]
            labels: Observation labels [batch_size, num_observations]
            
        Returns:
            hard_negative_loss: Scalar loss
        """
        batch_size = logits.size(0)
        
        # Compute label similarity (Jaccard similarity)
        label_similarity = self._compute_label_similarity(labels)
        
        # Find hard negatives: high label similarity but not exact match
        hard_negative_mask = (label_similarity > 0.3) & (label_similarity < 1.0)
        
        if hard_negative_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Penalize high similarity scores for hard negatives
        hard_negative_logits = logits[hard_negative_mask]
        hard_negative_loss = F.relu(hard_negative_logits - 0.5).mean()
        
        return hard_negative_loss
    
    def _compute_label_similarity(self, labels):
        """
        Compute Jaccard similarity between label sets.
        
        Args:
            labels: [batch_size, num_observations]
            
        Returns:
            similarity: [batch_size, batch_size]
        """
        batch_size = labels.size(0)
        labels_binary = (labels > 0.5).float()
        
        # Intersection
        intersection = labels_binary @ labels_binary.t()
        
        # Union
        sum_labels = labels_binary.sum(dim=1, keepdim=True)
        union = sum_labels + sum_labels.t() - intersection
        
        # Jaccard similarity
        similarity = intersection / (union + 1e-8)
        
        return similarity


class MultiViewAugmentation:
    """
    Multi-view augmentation for chest X-ray images.
    """
    def __init__(self, image_size=384):
        self.image_size = image_size
    
    def __call__(self, image):
        """
        Apply random augmentations to create multiple views.
        
        Args:
            image: PIL Image or tensor
            
        Returns:
            view1, view2: Two augmented views
        """
        import torchvision.transforms as T
        
        # Define augmentation pipeline
        augmentation = T.Compose([
            T.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
            ], p=0.5),
            T.RandomApply([
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        view1 = augmentation(image)
        view2 = augmentation(image)
        
        return view1, view2


class ContrastivePretrainer:
    """
    Utility class for contrastive pre-training workflow.
    """
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = ContrastiveLoss(use_hard_negatives=True)
    
    def pretrain_step(self, images, texts, text_masks, labels=None):
        """
        Single pre-training step.
        
        Args:
            images: Batch of images
            texts: Batch of text descriptions
            text_masks: Attention masks for text
            labels: Optional observation labels for hard negative mining
            
        Returns:
            loss: Loss value
            loss_dict: Detailed loss components
        """
        self.model.train()
        
        images = images.to(self.device)
        texts = texts.to(self.device)
        text_masks = text_masks.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        # Forward pass
        image_embeds, text_embeds, logit_scale = self.model.contrastive_forward(
            images, texts, text_masks
        )
        
        # Compute loss
        loss, loss_dict = self.criterion(image_embeds, text_embeds, logit_scale, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), loss_dict
    
    def finetune_step(self, images, texts, text_masks, labels):
        """
        Fine-tuning step with classification loss.
        
        Args:
            images: Batch of images
            texts: Batch of text descriptions
            text_masks: Attention masks for text
            labels: Observation labels
            
        Returns:
            loss: Loss value
        """
        self.model.train()
        
        images = images.to(self.device)
        texts = texts.to(self.device)
        text_masks = text_masks.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        logits = self.model(images, texts, text_masks)
        
        # Classification loss
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
