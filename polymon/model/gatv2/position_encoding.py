from typing import Dict, Any, Literal

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (AttentiveFP, BatchNorm, DimeNetPlusPlus,
                                GATv2Conv, GINConv, PNAConv, TransformerConv,
                                global_add_pool, global_max_pool)
from torch_geometric.utils import degree

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
    ):
        super().__init__()
        
        self.position_encoding = nn.Embedding(250, num_atom_features)

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
        self.atom_weighting = nn.Sequential(
            nn.Linear(feature_per_layer[-1], 1),
            nn.Sigmoid()
        )
        self.atom_weighting.apply(init_weight)
        nn.init.normal_(self.position_encoding.weight, std=0.02)

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
        
        # x: [num_nodes, num_features]
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