import torch
import torch.nn as nn
from torch_geometric.nn import (AttentiveFP, DimeNetPlusPlus, GATv2Conv,
                                global_add_pool, global_max_pool)

from polymon.data.polymer import Polymer
from polymon.model.module import MLP, init_weight


class GATv2(nn.Module):
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
        super(GATv2, self).__init__()

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
            input_dim=feature_per_layer[-1] * 2 + num_descriptors,
            hidden_dim=pred_hidden_dim,
            output_dim=num_tasks,
            n_layers=pred_layers,
            dropout=pred_dropout,
            activation=activation
        )
        self.num_descriptors = num_descriptors
        
    def forward(self, batch: Polymer): 
        x = batch.x.float()
        for layer in self.layers:
            x = layer(x, batch.edge_index, batch.edge_attr)
        
        batch_index = batch.batch
        weighted = self.atom_weighting(x)
        output1 = global_max_pool(x, batch_index)
        output2 = global_add_pool(weighted * x, batch_index)
        output = torch.cat([output1, output2], dim=1)
        
        if self.num_descriptors > 0:
            output = torch.cat([output, batch.descriptors], dim=1)

        return self.predict(output)


class AttentiveFPWrapper(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int = 2,
        dropout: float = 0
    ):
        super(AttentiveFPWrapper, self).__init__()

        self.attentivefp = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
        )

    def forward(self, batch: Polymer):
        return self.attentivefp(
            batch.x, 
            batch.edge_index, 
            batch.edge_attr, 
            batch.batch
        )


class DimeNetPP(DimeNetPlusPlus):
    """DimeNet++ model wrapper."""
    def __init__(
        self, 
        hidden_channels: int=128,
        out_channels: int=1,
        num_blocks: int=3,
        int_emb_size: int=64,
        basis_emb_size: int=8,
        out_emb_channels: int=256,
        num_spherical: int=7,
        num_radial: int=6,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 2,
        act: str = 'swish',
        output_initializer: str = 'zeros'
    ):
        super().__init__(
            hidden_channels, 
            out_channels, 
            num_blocks, 
            int_emb_size, 
            basis_emb_size,
            out_emb_channels, 
            num_spherical, 
            num_radial, 
            cutoff, 
            max_num_neighbors, 
            envelope_exponent, 
            num_before_skip, 
            num_after_skip, 
            num_output_layers, 
            act, 
            output_initializer
        )
        
    def forward(self, data: Polymer):
        z, pos, batch = data.z, data.pos, data.batch
        return super().forward(z, pos, batch)