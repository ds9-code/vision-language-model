import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics.classification import AUROC, F1Score, Accuracy

from models.causal_vit_vision_model import ViT3D
from models.text_model import TextEncoder
from configs.config import config


class SimpleCrossAttentionLayer(nn.Module):
    """
    cross-attention layer where text CLS token queries image features.
  
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        img_features: torch.Tensor,  # [B, T, E]
        img_key_padding_mask: torch.Tensor,  # [B, T], True for PAD
        txt_cls: torch.Tensor,  # [B, E] - CLS token 
    ) -> torch.Tensor:
       
        txt_cls_expanded = txt_cls.unsqueeze(1)  # CLS
        
        # Cross-attention: CLS token queries image features
        txt_attended, _ = self.cross_attn(
            query=txt_cls_expanded,  # CLS token as query (already normalized)
            key=img_features,        # Image features as keys and values 
            value=img_features,      
            key_padding_mask=img_key_padding_mask,  
            need_weights=False,
        )
        
        # Residual connection and squeeze back to [B, E]
        txt_attended = txt_cls_expanded + txt_attended
        return txt_attended.squeeze(1)  # [B, E]


class SimpleCrossAttentionTransformer(nn.Module):
    """
    Transformer where text CLS token queries image features.
    No self-attention, no CLS token prepending, just direct cross-attention.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(                  # stack multiple layers 
            [
                SimpleCrossAttentionLayer(
                    embed_dim=embed_dim, num_heads=heads, dropout=dropout
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        img_embeddings: torch.Tensor,  # [B, T, E]
        img_pad_mask: torch.Tensor,  # [B, T], True = real, False = pad
        txt_cls: torch.Tensor,  # [B, E] - CLS token from text encoder
    ) -> torch.Tensor:
        # Make image key padding mask
        img_key_padding_mask = ~img_pad_mask  # [B, T]

        # Pass through cross-attention layers
        txt_attended = txt_cls
        for layer in self.layers:
            txt_attended = layer(
                img_features=img_embeddings,
                img_key_padding_mask=img_key_padding_mask,
                txt_cls=txt_attended,
            )

        return txt_attended # Take the output from the last layer 


class FineGrainedMRITextCrossAttentionModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        # Save hyperparameters
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("__") and not callable(v)}
        self.save_hyperparameters(config_dict)

        # ViT backbone 
        self.vit = ViT3D(
            img_size=config.image_size,
            pretrained=True,
            simclr_ckpt_path=None
        )

        # MLP to project image features: 768 -> 512 (ViT outputs 768)
        self.img_mlp = nn.Sequential(
            nn.Linear(self.vit.feature_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Text encoder (frozen) + norm
        self.text_model = TextEncoder(
            model_name=config.text_model_name,
            max_length=config.text_max_length,
            embedding_dim=config.embedding_dim,
            freeze=True,
        )
        self.txt_layer_norm = nn.LayerNorm(config.embedding_dim)

        # Simple cross-attention transformer (no self-attention)
        self.cross_attention = SimpleCrossAttentionTransformer(
            embed_dim=config.embedding_dim,
            depth=config.cross_attn_depth,
            heads=config.cross_attn_heads,
            dropout=config.cross_attn_dropout,
        )

        # Binary classifier
        self.classifier = nn.Linear(config.embedding_dim, 1)

        # Loss and metrics
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.val_auroc = AUROC(task="binary")
        self.val_f1 = F1Score(task="binary", average="macro")
        self.val_acc = Accuracy(task="binary")

    def forward(self, volumes, pad_mask, texts):
        # volumes: [B, T, C, D, H, W]; pad_mask: [B, T] True=real, False=pad
        B, T, C, D, H, W = volumes.shape
        v_flat = volumes.view(B * T, C, D, H, W)
        feats = self.vit(v_flat)  #  get the vit embeddings as batched across batch and timepoints 
        img_embs = self.img_mlp(feats)  # [B*T, embedding_dim]
        img_seq = img_embs.view(B, T, -1)  # reshape to get back the batch and timepoint dim

        # Get CLS token from text encoder and normalize
        txt_cls = self.text_model(texts)  # [B, E]
        txt_cls = self.txt_layer_norm(txt_cls)  

        # Cross-attend: CLS token attends to image features 
        pooled_emb = self.cross_attention(
            img_embeddings=img_seq,
            img_pad_mask=pad_mask,
            txt_cls=txt_cls,
        )

        logits = self.classifier(pooled_emb)
        return logits, pooled_emb

    def training_step(self, batch, batch_idx):
        volumes, texts, pad_mask, labels = batch
        logits, _ = self(volumes, pad_mask, texts)
        logits = logits.squeeze(1)
        labels = labels.float()
        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=volumes.size(0), sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=volumes.size(0), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        volumes, texts, pad_mask, labels = batch
        logits, _ = self(volumes, pad_mask, texts)
        logits = logits.squeeze(1)
        labels_float = labels.float()
        loss = self.loss_fn(logits, labels_float)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        self.val_auroc.update(probs, labels)
        self.val_f1.update(preds, labels)
        self.val_acc.update(preds, labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=volumes.size(0), sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        try:
            epoch_auc = self.val_auroc.compute()
            epoch_f1 = self.val_f1.compute()
            epoch_acc = self.val_acc.compute()
            self.log("val_auc", epoch_auc, prog_bar=True, batch_size=config.batch_size, sync_dist=True)
            self.log("val_f1", epoch_f1, prog_bar=False, batch_size=config.batch_size, sync_dist=True)
            self.log("val_acc", epoch_acc, prog_bar=True, batch_size=config.batch_size, sync_dist=True)
        finally:
            self.val_auroc.reset()
            self.val_f1.reset()
            self.val_acc.reset()

    def configure_optimizers(self):
        backbone_params = list(self.vit.backbone.parameters())
        other_params = [p for n, p in self.named_parameters() if not n.startswith("vit.backbone")]
        optimizer = AdamW(
            [
                {"params": backbone_params, "lr": config.learning_rate},
                {"params": other_params, "lr": config.learning_rate},
            ],
            weight_decay=config.weight_decay,
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.warmup_steps,
            T_mult=2,
            eta_min=config.learning_rate * 0.1,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
