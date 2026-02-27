import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ViT
from typing import Optional, Tuple
import math

class ViT3D(nn.Module):
    """
    3D ViT encoder from MONAI with pretrained weights loading capability
    """
    def __init__(self, in_channels: int = 1, img_size: Tuple[int, int, int] = (96, 96, 96), pretrained: bool = True, simclr_ckpt_path: Optional[str] = "/media/sdb/divyanshu/divyanshu/Brainiac_revision/checkpoints/simclr_vitb_checkpoints/brainiac_trainval32k_simclr_normandscaling_vitb_cls_normonly_biasbeforenorm_lr0005_best-model-epoch=18-train_loss=0.00.ckpt"):
        super().__init__()
        
        # Create ViT backbone with same architecture as SimCLR
        self.backbone = ViT(
            in_channels=in_channels,
            img_size=img_size,  # Use configurable image size
            patch_size=(16, 16, 16),
            hidden_size=768,  # Standard for ViT-B
            mlp_dim=3072,
            num_layers=12,
            num_heads=12, 
            save_attn=True,
        )
        
        # Record feature dimension (hidden_size for ViT)
        self.feature_dim = 768
        
        # Layer norm for output features (for explicit normalization)
        #self.norm = nn.LayerNorm(self.feature_dim)
        
        # Load pretrained weights if specified
        if pretrained and simclr_ckpt_path:
            print("Loading pretrained weights from SimCLR vit checkpoint, with corrected backbone prefix!")
            self._load_pretrained_weights(simclr_ckpt_path)
    
    def _load_pretrained_weights(self, simclr_ckpt_path: str):
        """Load pretrained weights from SimCLR checkpoint"""
        try:
            ckpt = torch.load(simclr_ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            
            # Extract only backbone weights from SimCLR checkpoint
            backbone_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("backbone."):
                    # Remove "backbone." prefix
                    new_key = key[9:]  # len("backbone.") = 9
                    backbone_state_dict[new_key] = value
            
            # Load the backbone weights
            self.backbone.load_state_dict(backbone_state_dict, strict=True)
            print("Loaded pretrained ViT weights from SimCLR checkpoint!")
            
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            features: [B, feature_dim] - Mean-pooled patch tokens (excluding CLS token)
        """
        # Get features from ViT backbone
        features = self.backbone(x)
        
        # Handle tuple return (when save_attn=True)
        if isinstance(features, tuple):
            features = features[0]  # Extract the features tensor
        
        # features shape: [batch_size, num_tokens, hidden_dim]
        # num_tokens = 1 (CLS at position 0) + num_patches (positions 1 onwards)
        
        # ===== MEAN POOLING APPROACH (CURRENT) =====
        # Mean pool over patch tokens only (excluding CLS token at position 0)
        # This aggregates information from all spatial patch locations
        patch_tokens = features[:, 1:, :]  # [batch_size, num_patches, hidden_dim] - skip CLS token
        mean_pooled = patch_tokens.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Apply layer norm for stable features (if needed)
        # mean_pooled = self.norm(mean_pooled)
        
        return mean_pooled
        
        # ===== CLS TOKEN APPROACH (OLD - COMMENTED OUT) =====
        # # Use CLS token (first token) as global representation
        # # features[0] shape: [batch_size, num_tokens, hidden_dim]
        # # features[0][:, 0] gets CLS token: [batch_size, hidden_dim]
        # cls_token = features[0][:, 0]  # Shape: [batch_size, 768]
        # 
        # # Apply layer norm for stable features
        # #cls_token = self.norm(cls_token)
        # 
        # return cls_token

class TemporalTransformer(nn.Module):
    """
    Temporal Transformer that attends over a padded sequence of embeddings.
    Uses causal masking for temporal ordering.
    Returns CLS token output.
    Positional embeddings are added externally (in ViTVisionPipeline).
    """
    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int = 6,
        depth: int = 1,
        heads: int = 4,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        final_embed_dim: int = 512,  # Kept for backward compatibility but not used
    ):
        super().__init__()
        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings removed - they'll be added externally in ViTVisionPipeline
        
        # Transformer encoder (with causal masking)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        
        # Initialize CLS token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
    
    def _build_causal_mask(self, seq_len, device):
        """
        Build causal attention mask.
        Mask shape: [seq_len, seq_len], True means position is masked
        Lower-triangular (including diagonal) = False (not masked)
        Upper-triangular = True (masked)
        CLS token (first row) can attend to all positions
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        # Do NOT mask first row (CLS token): allow CLS to attend to all positions
        mask[0, :] = False
        return mask  # [1+T, 1+T]

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          seq_embeddings: [B, T, E]  (T ≤ max_seq_len) - Already has positional embeddings added externally
          pad_mask:        [B, T]     (True = real scan, False = pad)
        Returns:
          embedding: [B, embed_dim]  CLS token output
        """
        B, T, E = seq_embeddings.shape
        
        # Prepend CLS token to sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, E]
        x = torch.cat([cls_tokens, seq_embeddings], dim=1)  # [B, 1+T, E]
        # Note: seq_embeddings already has positional encodings added externally
        
        # Build key_padding_mask for transformer: True = mask (padded) position
        # CLS token is never masked
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
        key_mask = torch.cat([cls_mask, ~pad_mask], dim=1)  # [B, 1+T]
        
        # Build causal attention mask
        attn_mask = self._build_causal_mask(T + 1, x.device)  # [1+T, 1+T]
        
        # Apply transformer with causal mask and padding mask
        x = self.transformer(x, mask=attn_mask, src_key_padding_mask=key_mask)
        
        # ===== CLS TOKEN APPROACH (CURRENT) =====
        # Extract CLS token output
        cls_output = x[:, 0, :]  # [B, E]
        return cls_output
        
        # ===== MEAN POOLING APPROACH (COMMENTED OUT - CAN REVERT) =====
        # # Extract scan embeddings (skip CLS token at position 0)
        # scan_embeddings = x[:, 1:, :]  # [B, T, E] - Skip CLS token
        # 
        # # Mean pool over scan embeddings, masking out padded positions
        # # pad_mask: True = real scan, False = pad
        # # We need to mask padded positions for mean pooling
        # pad_mask_expanded = pad_mask.unsqueeze(-1).float()  # [B, T, 1]
        # masked_embeddings = scan_embeddings * pad_mask_expanded  # [B, T, E]
        # sum_embeddings = masked_embeddings.sum(dim=1)  # [B, E]
        # num_real_scans = pad_mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # [B, 1]
        # mean_output = sum_embeddings / num_real_scans  # [B, E]
        # 
        # return mean_output

def get_sinusoidal_position_encoding(max_seq_len, d_model):
    """Generate sinusoidal positional encodings (Vaswani et al., 2017)."""
    position = torch.arange(max_seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    
    pos_encoding = torch.zeros(max_seq_len, d_model)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    
    return pos_encoding.unsqueeze(0)  # [1, max_seq_len, d_model]

class ViTVisionPipeline(nn.Module):
    """
    Complete vision pipeline: 3D ViT + Temporal Transformer
    Similar to CVAE pattern: positional encodings added externally, causal masking in transformer
    Output dimension is vit.feature_dim (768 for ViT-B)
    """
    def __init__(
        self,
        in_channels: int = 1,
        vit_pretrained: bool = True,
        ## load the vit brainaic weights 
        simclr_ckpt_path: Optional[str] = "/media/sdb/divyanshu/divyanshu/Brainiac_revision/checkpoints/simclr_vitb_checkpoints/brainiac_trainval32k_simclr_normandscaling_vitb_cls_normonly_biasbeforenorm_lr0005_best-model-epoch=18-train_loss=0.00.ckpt",
        max_seq_len: int = 6,
        temp_depth: int = 1,
        temp_heads: int = 4,
        temp_mlp_dim: int = 3072,
        embedding_dim: int = 512,  # Kept for backward compatibility but not used
        dropout: float = 0.1,
    ):
        super().__init__()
        # vision encoder
        self.vit = ViT3D(
            in_channels=in_channels,
            img_size=(96, 96, 96),  # Default size for ViTVisionPipeline
            pretrained=vit_pretrained, 
            simclr_ckpt_path="/media/sdb/divyanshu/divyanshu/Brainiac_revision/checkpoints/simclr_vitb_checkpoints/brainiac_trainval32k_simclr_normandscaling_vitb_cls_normonly_biasbeforenorm_lr0005_best-model-epoch=18-train_loss=0.00.ckpt"
        )

        # temporal transformer (outputs vit.feature_dim, not embedding_dim)
        self.temporal = TemporalTransformer(
            embed_dim=self.vit.feature_dim,
            max_seq_len=max_seq_len,
            depth=temp_depth,
            heads=temp_heads,
            mlp_dim=temp_mlp_dim,
            dropout=dropout,
            final_embed_dim=embedding_dim  # Not used anymore
        )
        
        # Sinusoidal positional embeddings for temporal encoder (fixed, not learned)
        # Similar to CVAE pattern - added externally before temporal transformer
        self.register_buffer('encoder_pos_embedding', 
                             get_sinusoidal_position_encoding(max_seq_len, self.vit.feature_dim))
        
        # Store output dimension
        self.output_dim = self.vit.feature_dim

    def forward(
        self,
        volumes: torch.Tensor,
        pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          volumes: [B, T, C, D, H, W] padded with zeros for missing scans
          pad_mask: [B, T] True=real scan, False=pad
        Returns:
          embedding: [B, vit.feature_dim] (768 for ViT-B) - CLS token from temporal transformer
        """
        B, T, C, D, H, W = volumes.shape
        # flatten time for vit
        v_flat = volumes.view(B * T, C, D, H, W)                        # [B*T, C, D, H, W]
        feats = self.vit(v_flat)                                        # [B*T, feature_dim]
        seq = feats.view(B, T, -1)                                      # [B, T, feature_dim]
        
        # ===== NO POSITIONAL EMBEDDINGS (CURRENT) =====
        # Pass ViT features directly to temporal transformer (no positional encoding)
        # Temporal ordering is enforced through causal masking in the transformer
        enc_input = seq  # [B, T, 768]
        
        # ===== WITH POSITIONAL EMBEDDINGS (COMMENTED OUT - CAN REVERT) =====
        # # Add positional embeddings to vision features for temporal ordering
        # # Similar to CVAE pattern (line 188 in generative_cvae_model.py)
        # enc_input = seq + self.encoder_pos_embedding[:, :T, :]  # [B, T, 768] + [1, T, 768]
        
        # temporal transformer with causal mask (returns CLS token)
        return self.temporal(enc_input, pad_mask)  # [B, 768] 