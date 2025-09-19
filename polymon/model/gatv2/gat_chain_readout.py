import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from polymon.data.polymer import Polymer
from polymon.model.base import BaseModel
from polymon.model.utils import MLP, ReadoutPhase, init_weight
from polymon.model.register import register_init_params


@register_init_params
class GATv2ChainReadout(BaseModel):
    """GATv2 with chain readout.
    
    Args:
        num_atom_features (int): The number of atom features.
        hidden_dim (int): The number of hidden dimensions.
        num_layers (int): The number of layers.
        num_heads (int): The number of heads. Default to :obj:`8`.
        pred_hidden_dim (int): The number of hidden dimensions for the prediction 
            MLP. Default to :obj:`128`.
        pred_dropout (float): The dropout rate for the prediction MLP. Default to :obj:`0.2`.
        pred_layers (int): The number of layers for the prediction MLP. Default to :obj:`2`.
        activation (str): The activation function. Default to :obj:`'prelu'`.
        num_tasks (int): The number of tasks. Default to :obj:`1`.
        bias (bool): Whether to use bias. Default to :obj:`True`.
        dropout (float): The dropout rate. Default to :obj:`0.1`.
        edge_dim (int): The number of edge dimensions.
        num_descriptors (int): The number of descriptors. Default to :obj:`0`.
        chain_length (int): The length of the chain. Default to :obj:`10`.
    """
    def __init__(
        self, 
        num_atom_features: int, 
        hidden_dim: int, 
        num_layers: int, 
        num_heads: int=8, 
        pred_hidden_dim: int=128, 
        pred_dropout: float=0.2, 
        pred_layers:int=2,
        activation: str='prelu', 
        num_tasks: int = 1,
        bias: bool = True, 
        dropout: float = 0.1, 
        edge_dim: int = None,
        num_descriptors: int = 0,
        chain_length: int = 10,
    ):
        super().__init__()

        # update phase
        feature_per_layer = [num_atom_features + num_descriptors] + [hidden_dim] * num_layers
        layers = []
        for i in range(num_layers):
            layer = GATv2Conv(
                in_channels=feature_per_layer[i] * (1 if i == 0 else num_heads),
                out_channels=feature_per_layer[i + 1],
                heads=num_heads,
                concat=True if i < len(feature_per_layer) - 2 else False,
                edge_dim=edge_dim,
                dropout=dropout,
                bias=bias
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # readout phase
        self.readout = TransformerChainReadout(
            dim=hidden_dim,
            chain_length=chain_length,
            num_heads=num_heads,
            num_layers=3,
            dim_feedforward=pred_hidden_dim,
            dropout=dropout,
        )

        # prediction phase
        self.predict = MLP(
            input_dim=feature_per_layer[-1] * 2,
            hidden_dim=pred_hidden_dim,
            output_dim=num_tasks,
            n_layers=pred_layers,
            dropout=pred_dropout,
            activation=activation
        )
        self.num_descriptors = num_descriptors
        
    def forward(self, batch: Polymer): 
        """Forward pass.
        
        Args:
            batch (Polymer): The batch of data.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        x = batch.x.float()
        if self.num_descriptors > 0:
            x = torch.cat([x, batch.descriptors[batch.batch]], dim=1)
        
        for layer in self.layers:
            x = layer(x, batch.edge_index, batch.edge_attr)
        
        output = self.readout(x, batch.batch)

        return self.predict(output)


class TransformerChainReadout(nn.Module):
    def __init__(
        self,
        dim: int,
        chain_length: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.readout = ReadoutPhase(dim)
        self.chain_length = chain_length
        self.positional_encoding = symmetric_positional_encoding(chain_length, 2*dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2*dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 2*dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.transformer_encoder.apply(init_weight)
        
    def forward(self, x, batch):
        output = self.readout(x, batch) # [batch_size, 2*dim]
        output = output.unsqueeze(1).repeat(1, self.chain_length, 1) # [batch_size, chain_length, 2*dim]
        output = output + self.positional_encoding.to(output.device)
        output = torch.cat([self.cls_token.expand(output.shape[0], -1, -1), output], dim=1)
        output = self.transformer_encoder(output)
        output = output[:, 0, :]
        return output


def symmetric_positional_encoding(seq_len, d_model):
    """
    Generate reversal-invariant (symmetric) sinusoidal positional encoding.

    Args:
        seq_len (int): Length of the sequence (polymer length).
        d_model (int): Embedding dimension (must be even for sin/cos pairing).

    Returns:
        torch.Tensor of shape (seq_len, d_model) with symmetric encodings.
    """
    # Symmetric position index: distance from nearest end
    positions = np.array([min(i, seq_len - 1 - i) for i in range(seq_len)])
    
    # Expand to shape (seq_len, 1) for broadcasting
    positions = positions[:, np.newaxis]

    # Standard transformer scaling factor
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # Apply sin to even indices, cos to odd indices
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)

    return torch.tensor(pe, dtype=torch.float32)