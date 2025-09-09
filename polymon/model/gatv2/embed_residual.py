from typing import Any, Dict, Literal

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool

from polymon.data.polymer import Polymer
from polymon.model.base import BaseModel
from polymon.model.register import register_init_params
from polymon.model.utils import MLP, init_weight
from polymon.model.gnn import GATv2


@register_init_params
class GATv2EmbedResidual(BaseModel):
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
        pretrained_model: GATv2 = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

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
        self.predict = MLP(
            input_dim=feature_per_layer[-1] * 2,
            hidden_dim=pred_hidden_dim,
            output_dim=num_tasks,
            n_layers=pred_layers,
            dropout=pred_dropout,
            activation=activation
        )
        self.pretrained_model = pretrained_model
        if self.pretrained_model is not None:
            print(f'Using pretrained model: {pretrained_model.__class__.__name__}')
            if pretrained_model.hidden_dim != hidden_dim:
                print(f'Resizing pretrained model from {pretrained_model.hidden_dim*2} to {hidden_dim*2}')
                self.pretrained_encoder = nn.Linear(pretrained_model.hidden_dim*2, hidden_dim*2)
                self.pretrained_encoder.apply(init_weight)
        self.num_descriptors = num_descriptors
        if self.num_descriptors > 0:
            print(f'Using {num_descriptors} descriptors')
            self.descriptor_embedding = nn.Linear(num_descriptors, hidden_dim * 2)
            self.descriptor_embedding.apply(init_weight)
        
    def forward(self, batch: Polymer): 
        x = batch.x.float()
        for layer in self.layers:
            x = layer(x, batch.edge_index, batch.edge_attr)
        
        batch_index = batch.batch
        weighted = self.atom_weighting(x)
        output1 = global_max_pool(x, batch_index)
        output2 = global_add_pool(weighted * x, batch_index)
        output = torch.cat([output1, output2], dim=1)
        
        if self.pretrained_model is not None:
            embeddings = self.pretrained_model.get_embeddings(batch)
            if self.pretrained_model.hidden_dim != self.hidden_dim:
                embeddings = self.pretrained_encoder(embeddings)
            output = output + embeddings
        if self.num_descriptors > 0:
            descriptors = self.descriptor_embedding(batch.descriptors)
            output = output + descriptors
        
        return self.predict(output)