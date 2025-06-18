from typing import Literal

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from pfam_classifier.module import ResidualCNNBlock, get_positional_encoding


class TransformerClassifier(nn.Module):
    """Transformer classifier.
    
    Args:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The dimension of the model.
        nhead (int): The number of attention heads.
        num_layers (int): The number of layers in the transformer encoder.
        dim_feedforward (int): The dimension of the feedforward network.
        dropout (float): The dropout rate.
        num_classes (int): The number of classes.
        max_seq_len (int): The maximum length of the sequence.
        pos_encoding_type (Literal["sin", "learnable", "rope"]): The type of 
            positional encoding.
        padding_idx (int): The padding index.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_classes: int = 1000,
        max_seq_len: int = 300,
        pos_encoding_type: Literal["sin", "learnable", "rope"] = "sin",
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        dim = d_model // nhead if pos_encoding_type == 'rope' else d_model
        self.pos_encoder = get_positional_encoding(
            pos_encoding_type, dim, max_seq_len + 1
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.classifier = nn.Linear(d_model, num_classes)
        
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.pos_encoding_type = pos_encoding_type
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize the weights of the module."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_seq_repr(
        self, 
        seq: torch.Tensor, 
        seq_len: torch.Tensor = None
    ) -> torch.Tensor:
        """Get the sequence representation.
        
        Args:
            seq (torch.Tensor): Sequence tensor. Shape: [batch_size, seq_len].
            seq_len (torch.Tensor): Length of the sequence. Shape: [batch_size].
                This class does not require `seq_len` as input.

        Returns:
            `torch.Tensor`: The output tensor. Shape: [batch_size, num_classes].
        """
        # Add CLS token at the beginning
        batch_size = seq.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = self.embedding(seq)
        x = torch.cat([cls_tokens, x], dim=1)
        seq_len = self.max_seq_len + 1
        
        # Apply positional encoding
        if self.pos_encoding_type == 'rope':
            shape = (-1, seq_len, self.nhead, self.d_model // self.nhead)
            x = self.pos_encoder(x.reshape(shape))
            x = x.reshape(batch_size, seq_len, self.d_model)
        else:
            x = self.pos_encoder(x)
        
        # Extend padding mask to include CLS token
        padding_mask = seq == self.padding_idx
        cls_mask = torch.zeros(batch_size, 1, device=seq.device).bool()
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        
        # Transformer encoder expects [batch_size, seq_len, d_model]
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        return output[:, 0]
    
    def forward(self, seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            seq (torch.Tensor): Sequence tensor. Shape: [batch_size, seq_len].
            seq_len (torch.Tensor): Length of the sequence. Shape: [batch_size].
        """
        output = self.get_seq_repr(seq, seq_len)
        output = self.classifier(output)
        return output


class GRUClassifier(nn.Module):
    """GRU classifier.
    
    Args:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The dimension of the model.
        num_layers (int): The number of layers in the GRU.
        dropout (float): The dropout rate.
        num_classes (int): The number of classes.
        padding_idx (int): The padding index.
    """
    def __init__(
        self, 
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 1000,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            d_model, 
            d_model, 
            num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        self.fc = nn.Linear(d_model * 2, num_classes)
    
    def forward(self, seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            seq (torch.Tensor): Sequence tensor. Shape: [batch_size, seq_len].
            seq_len (torch.Tensor): Length of the sequence. Shape: [batch_size].

        Returns:
            `torch.Tensor`: The output tensor. Shape: [batch_size, num_classes].
        """
        input_emb = self.emb_dropout(self.embedding(seq))
        packed_input = pack_padded_sequence(
            input_emb, 
            seq_len.tolist(), 
            batch_first=True, 
            enforce_sorted=False
        )
        _, hidden = self.gru(packed_input)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.fc(hidden)
        return output


class CNNClassifier(nn.Module):
    """CNN classifier.
    
    Args:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The dimension of the model.
        num_classes (int): The number of classes.
        num_layers (int): The number of layers in the CNN.
        dropout (float): The dropout rate.
        padding_idx (int): The padding index.
        max_seq_len (int): The maximum length of the sequence.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_classes: int,
        num_layers: int,
        dropout: float = 0.5,
        padding_idx: int = 0,
        max_seq_len: int = 300
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )
        
        self.convs = nn.ModuleList([
            ResidualCNNBlock(d_model, i + 2)
            for i in range(num_layers)
        ])
        
        # Attention pooling layer
        self.maxpool = nn.MaxPool1d(kernel_size=3)
        self.dropout = nn.Dropout(dropout)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(d_model * max_seq_len // 3, num_classes)

    def forward(self, seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            seq (torch.Tensor): Sequence tensor. Shape: [batch_size, seq_len].
            seq_len (torch.Tensor): Length of the sequence. Shape: [batch_size].

        Returns:
            `torch.Tensor`: The output tensor. Shape: [batch_size, num_classes].
        """
        x = self.embedding(seq)
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        
        x = self.maxpool(x)
        x = self.dropout(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        return x