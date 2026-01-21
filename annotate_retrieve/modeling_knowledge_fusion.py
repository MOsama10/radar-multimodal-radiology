import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    num_fusion_layers: int = 2
    conflict_threshold: float = 0.5
    device: str = 'cuda'


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        if self.all_head_size != config.hidden_size:
            logger.warning(f"hidden_size {config.hidden_size} not divisible by num_attention_heads {config.num_attention_heads}")
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
    
    def transpose_for_scores(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        try:
            x = x.view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            return x.permute(0, 2, 1, 3)
        except RuntimeError as e:
            logger.error(f"Shape mismatch in transpose_for_scores: {e}")
            raise
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        try:
            query_layer = self.transpose_for_scores(self.query(query), batch_size)
            key_layer = self.transpose_for_scores(self.key(key), batch_size)
            value_layer = self.transpose_for_scores(self.value(value), batch_size)
            
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / (self.attention_head_size ** 0.5)
            
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            context_layer = context_layer.view(batch_size, -1, self.all_head_size)
            
            attention_output = self.output(context_layer)
            
            return attention_output, attention_probs
        
        except Exception as e:
            logger.error(f"Error in cross-attention: {e}")
            raise


class KnowledgeGatingMechanism(nn.Module):
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        self.pf_gate = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.intermediate_size, 1),
            nn.Sigmoid()
        )
        
        self.sf_gate = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.intermediate_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, pf_features: torch.Tensor, sf_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            pf_pooled = pf_features.mean(dim=1)
            sf_pooled = sf_features.mean(dim=1)
            
            pf_weight = self.pf_gate(pf_pooled)
            sf_weight = self.sf_gate(sf_pooled)
            
            total_weight = pf_weight + sf_weight + 1e-8
            pf_weight = pf_weight / total_weight
            sf_weight = sf_weight / total_weight
            
            pf_weight = pf_weight.unsqueeze(1)
            sf_weight = sf_weight.unsqueeze(1)
            
            return pf_weight, sf_weight
        
        except Exception as e:
            logger.error(f"Error in gating mechanism: {e}")
            raise


class ImageConditionedFusion(nn.Module):
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        self.image_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        
        self.fusion_layer = nn.Linear(config.hidden_size * 2, config.hidden_size)
    
    def forward(self, image_features: torch.Tensor, pf_features: torch.Tensor,
                sf_features: torch.Tensor) -> torch.Tensor:
        try:
            batch_size = pf_features.size(0)
            seq_len = pf_features.size(1)
            
            image_cond = self.image_projection(image_features)
            image_cond = image_cond.unsqueeze(1).expand(-1, seq_len, -1)
            
            modulated_pf = pf_features * (1 + 0.1 * torch.tanh(image_cond))
            modulated_sf = sf_features * (1 + 0.1 * torch.tanh(image_cond))
            
            concatenated = torch.cat([modulated_pf, modulated_sf], dim=-1)
            fused = self.fusion_layer(concatenated)
            
            return fused
        
        except Exception as e:
            logger.error(f"Error in image-conditioned fusion: {e}")
            raise


class ConflictDetector(nn.Module):
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        self.conflict_scorer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.intermediate_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, pf_features: torch.Tensor, sf_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            concatenated = torch.cat([pf_features, sf_features], dim=-1)
            conflict_scores = self.conflict_scorer(concatenated).squeeze(-1)
            conflict_mask = conflict_scores > self.config.conflict_threshold
            
            return conflict_scores, conflict_mask
        
        except Exception as e:
            logger.error(f"Error in conflict detection: {e}")
            raise


class AttentionBasedKnowledgeFusion(nn.Module):
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        self.pf_to_sf_attention = MultiHeadCrossAttention(config)
        self.sf_to_pf_attention = MultiHeadCrossAttention(config)
        
        self.gating = KnowledgeGatingMechanism(config)
        self.image_fusion = ImageConditionedFusion(config)
        self.conflict_detector = ConflictDetector(config)
        
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout)
        )
    
    def forward(self, pf_features: torch.Tensor, sf_features: torch.Tensor,
                image_features: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict:
        try:
            device = pf_features.device
            if image_features is not None and image_features.device != device:
                image_features = image_features.to(device)
            
            max_len = max(pf_features.size(1), sf_features.size(1))
            pf_padded = F.pad(pf_features, (0, 0, 0, max_len - pf_features.size(1)))
            sf_padded = F.pad(sf_features, (0, 0, 0, max_len - sf_features.size(1)))
            
            pf_attended, pf_attention = self.pf_to_sf_attention(pf_padded, sf_padded, sf_padded, attention_mask)
            sf_attended, sf_attention = self.sf_to_pf_attention(sf_padded, pf_padded, pf_padded, attention_mask)
            
            pf_gate, sf_gate = self.gating(pf_attended, sf_attended)
            
            gated_pf = pf_attended * pf_gate
            gated_sf = sf_attended * sf_gate
            
            if image_features is not None:
                fused = self.image_fusion(image_features, gated_pf, gated_sf)
            else:
                fused = gated_pf + gated_sf
            
            residual = pf_padded + sf_padded
            fused = self.layer_norm_1(fused + residual)
            
            fused_output = self.ffn(fused)
            fused = self.layer_norm_2(fused + fused_output)
            
            conflict_scores, conflict_mask = self.conflict_detector(pf_padded, sf_padded)
            
            return {
                'fused_features': fused,
                'pf_gates': pf_gate,
                'sf_gates': sf_gate,
                'conflict_scores': conflict_scores,
                'conflict_mask': conflict_mask,
                'pf_attention': pf_attention,
                'sf_attention': sf_attention
            }
        
        except Exception as e:
            logger.error(f"Error in fusion forward pass: {e}")
            raise


def create_fusion_model(hidden_size: int = 768, device: str = 'cuda') -> AttentionBasedKnowledgeFusion:
    config = FusionConfig(hidden_size=hidden_size, device=device)
    return AttentionBasedKnowledgeFusion(config)


if __name__ == "__main__":
    logger.info("CONTRIBUTION 2.2: ATTENTION-BASED KNOWLEDGE FUSION")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        logger.info("Building fusion model...")
        fusion_model = create_fusion_model(device=str(device))
        
        batch_size = 2
        logger.info(f"Creating test data (batch_size={batch_size})...")
        
        pf_features = torch.randn(batch_size, 10, 768, device=device)
        sf_features = torch.randn(batch_size, 8, 768, device=device)
        image_features = torch.randn(batch_size, 768, device=device)
        
        logger.info(f"PF shape: {pf_features.shape}")
        logger.info(f"SF shape: {sf_features.shape}")
        logger.info(f"Image shape: {image_features.shape}")
        
        logger.info("Running fusion forward pass...")
        results = fusion_model(pf_features, sf_features, image_features)
        
        logger.info("Results:")
        logger.info(f"Fused features shape: {results['fused_features'].shape}")
        logger.info(f"Conflicts detected: {results['conflict_mask'].sum().item()}")
        logger.info(f"PF gate mean: {results['pf_gates'].mean():.4f}")
        logger.info(f"SF gate mean: {results['sf_gates'].mean():.4f}")
        logger.info("Fusion model working correctly")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()