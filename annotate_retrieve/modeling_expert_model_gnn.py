import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers import SwinConfig
from transformers import AutoModel


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network layer for modeling observation dependencies.
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        
        # Multi-head attention parameters
        self.W = nn.Parameter(torch.zeros(size=(num_heads, in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(num_heads, 2 * out_features, 1)))
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
    
    def forward(self, h, adj):
        """
        Args:
            h: Node features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
        Returns:
            out: Updated node features [batch_size, num_nodes, out_features]
        """
        batch_size, num_nodes, _ = h.size()
        
        # Multi-head attention
        h_prime = []
        for head in range(self.num_heads):
            # Linear transformation
            Wh = torch.matmul(h, self.W[head])  # [batch_size, num_nodes, out_features]
            
            # Attention mechanism
            a_input = self._prepare_attentional_mechanism_input(Wh)
            e = self.leakyrelu(torch.matmul(a_input, self.a[head]).squeeze(3))
            
            # Mask attention based on adjacency
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=2)
            attention = self.dropout_layer(attention)
            
            # Apply attention
            h_prime_head = torch.matmul(attention, Wh)
            h_prime.append(h_prime_head)
        
        # Concatenate or average heads
        out = torch.mean(torch.stack(h_prime), dim=0)
        
        return F.elu(out)
    
    def _prepare_attentional_mechanism_input(self, Wh):
        """Prepare input for attention mechanism"""
        batch_size, num_nodes, out_features = Wh.size()
        
        # Repeat for all pairs
        Wh_repeated_in_chunks = Wh.repeat_interleave(num_nodes, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, num_nodes, 1)
        
        # Concatenate
        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2
        )
        
        return all_combinations_matrix.view(batch_size, num_nodes, num_nodes, 2 * out_features)


class HierarchicalExpertModel(PreTrainedModel):
    """
    Expert Model with Graph Neural Network for modeling observation dependencies.
    
    Contribution 1.2: Hierarchical Multi-Label Classification with GNN
    - Graph Attention Network for observation dependencies
    - Clinical knowledge graph encoding
    - Consistency loss for impossible combinations
    """
    def __init__(self, config: SwinConfig, text_model=None, num_gnn_layers=2):
        super().__init__(config)
        self.text_model = text_model
        self.num_gnn_layers = num_gnn_layers
        
        hidden_size = config.hidden_size + text_model.config.hidden_size
        num_observation = config.num_observation
        
        # Initial feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(
                in_features=hidden_size // 2,
                out_features=hidden_size // 2,
                num_heads=4,
                dropout=0.1
            )
            for _ in range(num_gnn_layers)
        ])
        
        # Final classification layer
        self.observation_cls = nn.Linear(hidden_size // 2, 1)
        
        # Learnable node embeddings for each observation
        self.node_embeddings = nn.Parameter(torch.randn(num_observation, hidden_size // 2))
        
        # Clinical knowledge graph adjacency matrix
        self.register_buffer('adj_matrix', self._create_clinical_knowledge_graph(num_observation))
        
        self.model = AutoModel.from_pretrained(self.config.pretrained_visual_extractor)
    
    def _create_clinical_knowledge_graph(self, num_observation=14):
        """
        Create adjacency matrix encoding clinical relationships between observations.
        
        Observations (CheXpert 14):
        0: Enlarged Cardiomediastinum
        1: Cardiomegaly
        2: Lung Opacity
        3: Lung Lesion
        4: Edema
        5: Consolidation
        6: Pneumonia
        7: Atelectasis
        8: Pneumothorax
        9: Pleural Effusion
        10: Pleural Other
        11: Fracture
        12: Support Devices
        13: No Finding
        """
        # Initialize with self-connections
        adj = torch.eye(num_observation)
        
        # Define clinical correlations (bidirectional)
        correlations = [
            # Cardiomegaly related
            (0, 1),  # Enlarged Cardiomediastinum <-> Cardiomegaly
            (1, 4),  # Cardiomegaly <-> Edema
            (1, 9),  # Cardiomegaly <-> Pleural Effusion
            
            # Lung pathologies
            (2, 5),  # Lung Opacity <-> Consolidation
            (2, 6),  # Lung Opacity <-> Pneumonia
            (5, 6),  # Consolidation <-> Pneumonia
            (2, 7),  # Lung Opacity <-> Atelectasis
            
            # Pleural conditions
            (9, 10), # Pleural Effusion <-> Pleural Other
            
            # Edema relationships
            (4, 9),  # Edema <-> Pleural Effusion
            (4, 2),  # Edema <-> Lung Opacity
            
            # Pneumonia relationships
            (6, 5),  # Pneumonia <-> Consolidation
            (6, 9),  # Pneumonia <-> Pleural Effusion
            
            # Atelectasis relationships
            (7, 2),  # Atelectasis <-> Lung Opacity
            (7, 9),  # Atelectasis <-> Pleural Effusion
        ]
        
        # Add correlations (bidirectional)
        for i, j in correlations:
            adj[i, j] = 1
            adj[j, i] = 1
        
        return adj
    
    def forward(
        self,
        input_pixels: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
    ):
        """
        Forward pass with GNN-based observation modeling.
        
        Returns:
            observation_logits: [batch_size, num_observations]
        """
        batch_size = input_pixels.size(0)
        
        # Extract image and text features
        image_embeds = self.model(input_pixels).pooler_output
        text_embeds = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output
        
        # Combine features
        combined_embeds = torch.cat((image_embeds, text_embeds), dim=-1)
        features = self.feature_projection(combined_embeds)  # [batch_size, hidden_size//2]
        
        # Initialize node features by combining image features with node embeddings
        node_features = self.node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        node_features = node_features + features.unsqueeze(1)
        
        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            node_features = gnn_layer(node_features, self.adj_matrix)
        
        # Final classification for each observation
        observation_logits = self.observation_cls(node_features).squeeze(-1)
        
        return observation_logits
    
    def compute_consistency_loss(self, logits, labels):
        """
        Compute consistency loss to penalize clinically impossible combinations.
        
        Args:
            logits: Predicted logits [batch_size, num_observations]
            labels: Ground truth labels [batch_size, num_observations]
            
        Returns:
            consistency_loss: Scalar loss value
        """
        probs = torch.sigmoid(logits)
        
        # Define mutually exclusive pairs (can't both be positive)
        exclusive_pairs = [
            (13, 0),  # No Finding vs Enlarged Cardiomediastinum
            (13, 1),  # No Finding vs Cardiomegaly
            (13, 2),  # No Finding vs Lung Opacity
            (13, 4),  # No Finding vs Edema
            (13, 5),  # No Finding vs Consolidation
            (13, 6),  # No Finding vs Pneumonia
            (13, 7),  # No Finding vs Atelectasis
            (13, 8),  # No Finding vs Pneumothorax
            (13, 9),  # No Finding vs Pleural Effusion
        ]
        
        consistency_loss = 0.0
        for i, j in exclusive_pairs:
            # Penalize if both are predicted as positive
            both_positive = probs[:, i] * probs[:, j]
            consistency_loss += both_positive.mean()
        
        return consistency_loss
    
    def compute_correlation_loss(self, logits):
        """
        Encourage correlated observations to have similar predictions.
        
        Args:
            logits: Predicted logits [batch_size, num_observations]
            
        Returns:
            correlation_loss: Scalar loss value
        """
        probs = torch.sigmoid(logits)
        
        # Define positively correlated pairs (should be similar)
        correlated_pairs = [
            (0, 1),  # Enlarged Cardiomediastinum & Cardiomegaly
            (1, 4),  # Cardiomegaly & Edema
            (5, 6),  # Consolidation & Pneumonia
        ]
        
        correlation_loss = 0.0
        for i, j in correlated_pairs:
            # L2 distance between correlated predictions
            diff = (probs[:, i] - probs[:, j]) ** 2
            correlation_loss += diff.mean()
        
        return correlation_loss


class HierarchicalLoss(nn.Module):
    """
    Combined loss function for hierarchical multi-label classification.
    """
    def __init__(self, bce_weight=1.0, consistency_weight=0.1, correlation_weight=0.05):
        super(HierarchicalLoss, self).__init__()
        self.bce_weight = bce_weight
        self.consistency_weight = consistency_weight
        self.correlation_weight = correlation_weight
    
    def forward(self, logits, labels, model):
        """
        Compute combined loss.
        
        Args:
            logits: Predicted logits
            labels: Ground truth labels
            model: HierarchicalExpertModel instance
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        # Consistency loss
        consistency_loss = model.compute_consistency_loss(logits, labels)
        
        # Correlation loss
        correlation_loss = model.compute_correlation_loss(logits)
        
        # Total loss
        total_loss = (
            self.bce_weight * bce_loss +
            self.consistency_weight * consistency_loss +
            self.correlation_weight * correlation_loss
        )
        
        loss_dict = {
            'bce_loss': bce_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'correlation_loss': correlation_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
