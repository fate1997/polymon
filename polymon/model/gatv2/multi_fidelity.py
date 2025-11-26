from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool

from polymon.data.polymer import Polymer
from polymon.model.base import BaseModel
from polymon.model.register import register_init_params
from polymon.model.utils import init_weight


@register_init_params
class GATv2_Source(BaseModel):
    """GATv2 with source-specific heads.
    
    Args:
        num_atom_features (int): The number of atom features.
        hidden_dim (int): The number of hidden dimensions.
        num_layers (int): The number of layers.
        num_heads (int): The number of heads. Default to :obj:`8`.
        pred_hidden_dim (int): The number of hidden dimensions for the prediction 
            MLP. Default to :obj:`128`.
        num_tasks (int): The number of tasks. Default to :obj:`1`.
        bias (bool): Whether to use bias. Default to :obj:`True`.
        dropout (float): The dropout rate. Default to :obj:`0.1`.
        edge_dim (int): The number of edge dimensions.
        source_names (List[str]): The names of the sources. Default to :obj:`['internal']`.
    """
    def __init__(
        self, 
        num_atom_features: int, 
        hidden_dim: int, 
        num_layers: int, 
        num_heads: int=8, 
        pred_hidden_dim: int=128, 
        num_tasks: int = 1,
        bias: bool = True, 
        dropout: float = 0.1, 
        edge_dim: int = None,
        source_names: List[int] = [1],
        **kwargs,
    ):
        super().__init__()

        # update phase
        feature_per_layer = [num_atom_features] + [hidden_dim] * num_layers
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
        self.source_names = source_names
        self.num_sources = len(source_names)
        self.source_specific_head = SourceSpecificHead(
            input_dim=feature_per_layer[-1] * 2,
            hidden_dim=pred_hidden_dim,
            output_dim=num_tasks,
            num_sources=self.num_sources
        )
        
    def forward(self, batch: Polymer):
        """Forward pass.
        
        Args:
            batch (Polymer): The batch of data. It should have :obj:`source` 
                attribute.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        x = batch.x.float()
        for layer in self.layers:
            x = layer(x, batch.edge_index, batch.edge_attr)
        
        batch_index = batch.batch
        weighted = self.atom_weighting(x)
        output1 = global_max_pool(x, batch_index)
        output2 = global_add_pool(weighted * x, batch_index)
        output = torch.cat([output1, output2], dim=1)
        
        batch_size = batch.batch.max() + 1
        source = getattr(batch, 'source', [1] * batch_size)
        indices = (source.detach().cpu().numpy()[:, None] == np.array(self.source_names)[None, :]).argmax(axis=1)
        indices = torch.from_numpy(indices).to(batch.x.device)
        return self.source_specific_head(output, indices)


class SourceSpecificHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_sources=1):
        super().__init__()
        # Store all weights/biases in embeddings
        self.W1 = nn.Embedding(num_sources, input_dim * hidden_dim)
        self.b1 = nn.Embedding(num_sources, hidden_dim)
        self.W2 = nn.Embedding(num_sources, hidden_dim * output_dim)
        self.b2 = nn.Embedding(num_sources, output_dim)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_sources = num_sources

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize W1 and W2 with Xavier uniform like nn.Linear
        for i in range(self.num_sources):
            nn.init.xavier_uniform_(self.W1.weight[i].view(self.input_dim, self.hidden_dim))
            nn.init.zeros_(self.b1.weight[i])
            nn.init.xavier_uniform_(self.W2.weight[i].view(self.hidden_dim, self.output_dim))
            nn.init.zeros_(self.b2.weight[i])

    def forward(self, graph_repr, source_idx):
        # Lookup parameters for each task
        W1 = self.W1(source_idx).view(-1, self.input_dim, self.hidden_dim)   # (B, in, hid)
        b1 = self.b1(source_idx)                                             # (B, hid)
        W2 = self.W2(source_idx).view(-1, self.hidden_dim, self.output_dim)  # (B, hid, out)
        b2 = self.b2(source_idx)                                             # (B, out)

        # Forward pass
        h = torch.bmm(graph_repr.unsqueeze(1), W1).squeeze(1) + b1  # (B, hid)
        h = F.relu(h)
        out = torch.bmm(h.unsqueeze(1), W2).squeeze(1) + b2         # (B, out)

        return out
