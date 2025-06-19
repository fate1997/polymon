import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool

from polymon.data.polymer import Polymer

def seed_all():
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

#!TODO: Implement
class GATv2(nn.Module):
    def __init__(
        self, 
        num_atom_features: int, 
        hidden_dim: int, 
        num_layers: int, 
        num_heads: int, 
        pred_hidden_dim: int=128, 
        pred_dropout: float=0.2, 
        pred_layers:int=2,
        activation: str='prelu', 
        residual: bool = True, 
        num_tasks: int = 1,
        bias: bool = True, 
        dropout: float = 0.1, 
        edge_dim: int = None
    ):
        super(GATv2_PyG, self).__init__()

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
            input_dim=feature_per_layer[-1] * 2 * 2,
            hidden_dim=pred_hidden_dim,
            output_dim=num_tasks,
            n_layers=pred_layers,
            dropout=pred_dropout,
            activation=activation
        )        
        
    def forward(self, batch: Polymer): 
        x = batch.x
        for layer in self.layers:
            x = layer(x, batch.edge_index)
        
        batch_index = batch.batch
        mask = batch.comp.bool()
        batch_index[mask] = len(batch) + batch_index[mask]
        weighted = self.atom_weighting(x)
        output1 = global_max_pool(x, batch_index)
        output2 = global_add_pool(weighted * x, batch_index)
        output = torch.cat([output1, output2], dim=1)
        
        output = torch.cat([output[:len(output)//2], output[len(output)//2:]], dim=1)
        return self.predict(output)