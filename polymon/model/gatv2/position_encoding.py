from typing import Dict, Any, Literal
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (AttentiveFP, BatchNorm, DimeNetPlusPlus,
                                GATv2Conv, GINConv, PNAConv, TransformerConv,
                                global_add_pool, global_max_pool)
from torch_geometric.utils import degree
from functools import partial

from polymon.data.polymer import Polymer
from polymon.model.base import BaseModel
from polymon.model.utils import MLP, ReadoutPhase, init_weight
from polymon.model.register import register_init_params


@register_init_params
class GATv2_PE(BaseModel):
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
        position_encoding_type: Literal['sin', 'rope', 'learned'] = 'sin',
    ):
        super().__init__()
        
        if position_encoding_type == 'sin':
            self.position_encoding = partial(sinusoidal_encoding, dim=hidden_dim)
        elif position_encoding_type == 'rope':
            self.position_encoding = partial(rope_encoding, dim=hidden_dim)
        elif position_encoding_type == 'learned':
            self.position_encoding = nn.Embedding(250, hidden_dim)
            nn.init.normal_(self.position_encoding.weight, std=0.02)
        else:
            raise ValueError(f'Invalid position encoding type: {position_encoding_type}')
        self.position_encoding_type = position_encoding_type

        self.encoder = nn.Linear(num_atom_features, hidden_dim, bias=False)
        self.encoder.apply(init_weight)

        # update phase
        feature_per_layer = [hidden_dim + num_descriptors] + [hidden_dim] * num_layers
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
        self.atom_weighting = nn.Sequential(
            nn.Linear(feature_per_layer[-1], 1),
            nn.Sigmoid()
        )
        self.atom_weighting.apply(init_weight)

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
        x = batch.x.float()
        x = self.encoder(x)
        
        # x: [num_nodes, num_features]
        if self.position_encoding_type =='rope':
            x = rope_encoding(x, batch.relative_position)
        else:
            x = self.position_encoding(batch.relative_position) + x
        
        
        if self.num_descriptors > 0:
            x = torch.cat([x, batch.descriptors[batch.batch]], dim=1)
        
        for layer in self.layers:
            x = layer(x, batch.edge_index, batch.edge_attr)
        
        batch_index = batch.batch
        weighted = self.atom_weighting(x)
        output1 = global_max_pool(x, batch_index)
        output2 = global_add_pool(weighted * x, batch_index)
        output = torch.cat([output1, output2], dim=1)
        
        # if self.num_descriptors > 0:
        #     output = torch.cat([output, batch.descriptors], dim=1)

        return self.predict(output)


def sinusoidal_encoding(distances, dim):
    """
    distances: tensor of shape (n_nodes,) with integer distances
    dim: embedding dimension
    returns: tensor of shape (n_nodes, dim)
    """
    n_nodes = distances.size(0)
    pe = torch.zeros(n_nodes, dim).to(distances.device)
    position = distances.unsqueeze(1)  # (n_nodes, 1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=distances.device) * 
                         -(math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def rope_encoding(x, distances):
    """
    x: (n_nodes, dim) input features
    distances: (n_nodes,) integer relative distances
    returns: (n_nodes, dim) features after rotary positional encoding
    """
    n_nodes, dim = x.shape
    assert dim % 2 == 0, "dim must be even for RoPE"

    # compute frequencies
    half_dim = dim // 2
    freq_seq = torch.arange(0, half_dim, dtype=torch.float, device=x.device)
    freq_seq = 1.0 / (10000 ** (freq_seq / half_dim))  # (half_dim,)

    # angle for each distance
    theta = distances.unsqueeze(1) * freq_seq  # (n_nodes, half_dim)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # split x into pairs
    x1, x2 = x[:, 0::2], x[:, 1::2]  # (n_nodes, half_dim) each

    # rotate
    x1_new = x1 * cos_theta - x2 * sin_theta
    x2_new = x1 * sin_theta + x2 * cos_theta

    # interleave back
    x_out = torch.stack([x1_new, x2_new], dim=-1).reshape(n_nodes, dim)
    return x_out