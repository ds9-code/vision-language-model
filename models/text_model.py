import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    """
    Text encoder based on ClinicalBERT, with option to freeze base model.
    """
    def __init__(
        self,
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        max_length=128,
        embedding_dim=512,
        freeze=True  # <--- Add this flag
    ):
        super().__init__()
        
        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Projection layer to desired embedding dimension
        self.proj = nn.Linear(self.model.config.hidden_size, embedding_dim)
        
        self.max_length = max_length

        # Optionally freeze the transformer backbone
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def tokenize(self, texts):
        """
        Tokenize a batch of texts
        """
        return self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def forward(self, texts):
        """
        Args:
            texts: List of text strings
        """
        # Tokenize inputs
        encoded_inputs = self.tokenize(texts)
        
        # Move to the device that the model is on
        device = next(self.parameters()).device
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        
        # Get model outputs
        with torch.set_grad_enabled(self.proj.weight.requires_grad):  # Ensure correct grad context if frozen
            outputs = self.model(**encoded_inputs)
            # Use CLS token (position 0) as the text representation
            last_hidden_state = outputs.last_hidden_state  # [B, seq_len, hidden_size]
            cls_embedding = last_hidden_state[:, 0, :]  # [B, hidden_size] - CLS token only
            
            # Project to desired embedding dimension
            embedding = self.proj(cls_embedding)
            # No L2 normalization
        
        return embedding
