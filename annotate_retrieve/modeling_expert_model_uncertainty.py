import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers import SwinConfig
from transformers import AutoModel


class UncertaintyAwareExpertModel(PreTrainedModel):
    """
    Expert Model with Uncertainty Quantification using Monte Carlo Dropout
    and Temperature Scaling for calibration.
    
    Contribution 1.1: Uncertainty-Aware Expert Model
    - Monte Carlo Dropout for uncertainty estimation
    - Temperature scaling for probability calibration
    - Observation-specific confidence thresholds
    """
    def __init__(self, config: SwinConfig, text_model=None, num_mc_samples=10, dropout_rate=0.1):
        super().__init__(config)
        self.text_model = text_model
        self.num_mc_samples = num_mc_samples
        self.dropout_rate = dropout_rate
        
        hidden_size = config.hidden_size + text_model.config.hidden_size
        num_observation = config.num_observation
        
        # MLP with dropout for uncertainty estimation
        self.observation_cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size // 2, num_observation),
        )
        
        # Temperature parameter for calibration (learnable)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Observation-specific confidence thresholds (learnable)
        self.confidence_thresholds = nn.Parameter(torch.ones(num_observation) * 0.5)
        
        self.model = AutoModel.from_pretrained(self.config.pretrained_visual_extractor)

    def forward(
        self,
        input_pixels: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
    ):
        """Standard forward pass (single prediction)"""
        image_embeds = self.model(input_pixels).pooler_output
        text_embeds = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output
        
        combined_embeds = torch.cat((image_embeds, text_embeds), dim=-1)
        observation_cls_logits = self.observation_cls(combined_embeds)
        
        return observation_cls_logits

    def forward_with_uncertainty(
        self,
        input_pixels: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        return_all_samples=False,
    ):
        """
        Forward pass with uncertainty estimation using Monte Carlo Dropout.
        
        Args:
            input_pixels: Image tensor
            input_ids: Text input IDs
            attention_mask: Text attention mask
            return_all_samples: If True, return all MC samples
            
        Returns:
            mean_pred: Mean prediction across MC samples
            uncertainty: Standard deviation across MC samples
            calibrated_pred: Temperature-scaled predictions
            all_samples: All MC samples (if return_all_samples=True)
        """
        # Enable dropout during inference for MC sampling
        self.observation_cls.train()
        
        predictions = []
        for _ in range(self.num_mc_samples):
            logits = self.forward(input_pixels, input_ids, attention_mask)
            # Apply temperature scaling and sigmoid
            calibrated_logits = logits / self.temperature
            probs = torch.sigmoid(calibrated_logits)
            predictions.append(probs)
        
        # Stack all predictions
        all_predictions = torch.stack(predictions)  # [num_mc_samples, batch_size, num_observations]
        
        # Compute statistics
        mean_pred = all_predictions.mean(dim=0)
        uncertainty = all_predictions.std(dim=0)
        
        # Calibrated prediction (using mean of calibrated samples)
        calibrated_pred = mean_pred
        
        if return_all_samples:
            return mean_pred, uncertainty, calibrated_pred, all_predictions
        else:
            return mean_pred, uncertainty, calibrated_pred

    def get_confident_observations(
        self,
        mean_pred: torch.Tensor,
        uncertainty: torch.Tensor,
        use_adaptive_threshold=True,
    ):
        """
        Get high-confidence observations based on prediction and uncertainty.
        
        Args:
            mean_pred: Mean predictions [batch_size, num_observations]
            uncertainty: Uncertainty estimates [batch_size, num_observations]
            use_adaptive_threshold: Use learned observation-specific thresholds
            
        Returns:
            confident_mask: Boolean mask of confident predictions
            confidence_scores: Combined confidence scores
        """
        # Confidence score: high prediction, low uncertainty
        confidence_scores = mean_pred * (1 - uncertainty)
        
        if use_adaptive_threshold:
            # Use learned observation-specific thresholds
            thresholds = torch.sigmoid(self.confidence_thresholds).unsqueeze(0)
        else:
            # Use fixed threshold
            thresholds = 0.5
        
        confident_mask = confidence_scores > thresholds
        
        return confident_mask, confidence_scores

    def calibrate_temperature(self, val_loader, device='cuda'):
        """
        Calibrate temperature parameter on validation set using NLL loss.
        Should be called after training the main model.
        
        Args:
            val_loader: Validation data loader
            device: Device to run calibration on
        """
        self.eval()
        self.to(device)
        
        # Collect all predictions and labels
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = self.forward(images, input_ids, attention_mask)
                all_logits.append(logits)
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                all_logits / self.temperature,
                all_labels.float()
            )
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        print(f"Calibrated temperature: {self.temperature.item():.4f}")
        
        return self.temperature.item()


class UncertaintyMetrics:
    """Utility class for computing uncertainty calibration metrics"""
    
    @staticmethod
    def expected_calibration_error(predictions, uncertainties, labels, num_bins=10):
        """
        Compute Expected Calibration Error (ECE)
        
        Args:
            predictions: Model predictions [N, num_observations]
            uncertainties: Uncertainty estimates [N, num_observations]
            labels: Ground truth labels [N, num_observations]
            num_bins: Number of bins for calibration
            
        Returns:
            ece: Expected Calibration Error
        """
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        
        ece = 0.0
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find predictions in this bin
            in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
            
            if in_bin.sum() > 0:
                # Average confidence in bin
                avg_confidence = predictions[in_bin].mean()
                # Average accuracy in bin
                avg_accuracy = (predictions[in_bin] > 0.5) == labels[in_bin]
                avg_accuracy = avg_accuracy.mean()
                
                # Weighted contribution to ECE
                ece += (in_bin.sum() / len(predictions)) * abs(avg_confidence - avg_accuracy)
        
        return ece
    
    @staticmethod
    def compute_uncertainty_quality(predictions, uncertainties, labels):
        """
        Compute correlation between uncertainty and prediction error
        High correlation means uncertainty is well-calibrated
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            labels: Ground truth labels
            
        Returns:
            correlation: Pearson correlation coefficient
        """
        errors = torch.abs(predictions - labels.float())
        
        # Flatten for correlation computation
        errors_flat = errors.flatten()
        uncertainties_flat = uncertainties.flatten()
        
        # Compute Pearson correlation
        correlation = torch.corrcoef(torch.stack([errors_flat, uncertainties_flat]))[0, 1]
        
        return correlation.item()
