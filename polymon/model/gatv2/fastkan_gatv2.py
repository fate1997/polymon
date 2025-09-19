import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool

from polymon.data.polymer import Polymer
from polymon.model.base import BaseModel
from polymon.model.kan.fast_kan import FastKAN
from polymon.model.register import register_init_params
from polymon.model.utils import init_weight


@register_init_params
class FastKAN_GATv2(BaseModel):
    """Fast KAN-augmented GATv2.
    
    Args:
        num_atom_features (int): The number of atom features.
        hidden_dim (int): The number of hidden dimensions.
        num_layers (int): The number of layers.
        num_heads (int): The number of heads. Default to :obj:`8`.
        pred_hidden_dim (int): The number of hidden dimensions for the prediction 
            MLP. Default to :obj:`128`.
        grid_min (float): The minimum value of the grid. Default to :obj:`-2.0`.
        grid_max (float): The maximum value of the grid. Default to :obj:`2.0`.
        num_grids (int): The number of grids. Default to :obj:`8`.
        num_tasks (int): The number of tasks. Default to :obj:`1`.
        bias (bool): Whether to use bias. Default to :obj:`True`.
        dropout (float): The dropout rate. Default to :obj:`0.1`.
        edge_dim (int): The number of edge dimensions.
        num_descriptors (int): The number of descriptors. Default to :obj:`0`.
    """
    def __init__(
        self, 
        num_atom_features: int, 
        hidden_dim: int, 
        num_layers: int, 
        num_heads: int=8, 
        pred_hidden_dim: int=128, 
        grid_min: float=-2.0,
        grid_max: float=2.0,
        num_grids: int=8,
        num_tasks: int = 1,
        bias: bool = True, 
        dropout: float = 0.1, 
        edge_dim: int = None,
        num_descriptors: int = 0,
        
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
        self.atom_weighting = nn.Sequential(
            nn.Linear(feature_per_layer[-1], 1),
            nn.Sigmoid()
        )
        self.atom_weighting.apply(init_weight)

        # prediction phase
        self.predict = FastKAN(
            layers_hidden=[feature_per_layer[-1] * 2, pred_hidden_dim, num_tasks],
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
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
        
        batch_index = batch.batch
        weighted = self.atom_weighting(x)
        output1 = global_max_pool(x, batch_index)
        output2 = global_add_pool(weighted * x, batch_index)
        output = torch.cat([output1, output2], dim=1)
        
        if self.num_descriptors > 0:
            output = torch.cat([output, batch.descriptors], dim=1)

        return self.predict(output)