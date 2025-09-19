from typing import Dict, Any, Literal

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv

from polymon.data.polymer import Polymer
from polymon.model.utils import MLP, KANLinear, ReadoutPhase, RedrawProjection
from polymon.model.register import register_init_params
from polymon.model.base import BaseModel
from polymon.model.gps.conv import GPSConv
from polymon.model.kan.fast_kan import FastKAN


@register_init_params
class GraphGPS(BaseModel):
    """GraphGPS model.
    
    Args:
        in_channels (int): The number of input channels.
        edge_dim (int): The number of edge dimensions.
        heads (int): The number of heads. Default to :obj:`4`.
        hidden_dim (int): The number of hidden dimensions. Default to :obj:`64`.
        num_layers (int): The number of layers. Default to :obj:`6`.
        walk_length (int): The length of the walk. Default to :obj:`20`.
        pe_dim (int): The dimension of the positional encoding. Default to :obj:`8`.
        attn_type (Literal['performer', 'multihead']): The type of attention. 
            Default to :obj:`'multihead'`.
        attn_kwargs (Dict[str, Any]): The keyword arguments for the attention.
        grid_size (int): The size of the grid. Default to :obj:`3`.
    """
    def __init__(
        self, 
        in_channels: int,
        edge_dim: int,
        heads: int = 4,
        hidden_dim: int = 64, 
        num_layers: int = 6,
        walk_length: int = 20,
        pe_dim: int = 8, 
        attn_type: Literal['performer', 'multihead'] = 'multihead', 
        attn_kwargs: Dict[str, Any]=None,
    ):
        super().__init__()

        self.node_emb = nn.Linear(in_channels, hidden_dim - pe_dim, bias=False)
        self.pe_lin = nn.Linear(walk_length, pe_dim)
        self.pe_norm = nn.BatchNorm1d(walk_length)
        self.edge_emb = nn.Linear(edge_dim, hidden_dim, bias=False)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            network = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = GPSConv(hidden_dim, GINEConv(network), heads=heads,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.readout = ReadoutPhase(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, batch: Polymer):
        """Forward pass.
        
        Args:
            batch (Polymer): The batch of data. It should have :obj:`pe` attribute.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        x_pe = self.pe_norm(batch.pe)
        x = torch.cat((self.node_emb(batch.x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(batch.edge_attr.float())

        for conv in self.convs: 
            x = conv(x, batch.edge_index, batch.batch, edge_attr=edge_attr)
        x = self.readout(x, batch.batch)
        return self.mlp(x)


@register_init_params
class KAN_GPS(BaseModel):
    """KAN-augmented GraphGPS model.
    
    Args:
        in_channels (int): The number of input channels.
        edge_dim (int): The number of edge dimensions.
        heads (int): The number of heads. Default to :obj:`4`.
        hidden_dim (int): The number of hidden dimensions. Default to :obj:`64`.
        num_layers (int): The number of layers. Default to :obj:`6`.
        walk_length (int): The length of the walk. Default to :obj:`20`.
        pe_dim (int): The dimension of the positional encoding. Default to :obj:`8`.
        attn_type (Literal['performer', 'multihead', 'fastkan']): The type of attention. 
            Default to :obj:`'fastkan'`.
        attn_kwargs (Dict[str, Any]): The keyword arguments for the attention.
        grid_size (int): The size of the grid. Default to :obj:`3`.
    """
    def __init__(
        self, 
        in_channels: int,
        edge_dim: int,
        heads: int = 4,
        hidden_dim: int = 64, 
        num_layers: int = 6,
        walk_length: int = 20,
        pe_dim: int = 8, 
        attn_type: Literal['performer', 'multihead', 'fastkan'] = 'fastkan', 
        attn_kwargs: Dict[str, Any]=None,
        grid_size: int = 3,
    ):
        super().__init__()

        self.node_emb = KANLinear(in_channels, hidden_dim - pe_dim, grid_size, add_bias=False)
        self.pe_lin = KANLinear(walk_length, pe_dim, grid_size)
        self.pe_norm = nn.BatchNorm1d(walk_length)
        self.edge_emb = KANLinear(edge_dim, hidden_dim, grid_size, add_bias=False)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            network = FastKAN(layers_hidden=[hidden_dim] * 2)
            conv = GPSConv(hidden_dim, GINEConv(network), heads=heads,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)
        self.readout = ReadoutPhase(hidden_dim)

        self.mlp = FastKAN(layers_hidden=[2 * hidden_dim, hidden_dim, 1])
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, batch: Polymer):
        """Forward pass.
        
        Args:
            batch (Polymer): The batch of data. It should have :obj:`pe` attribute.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        x_pe = self.pe_norm(batch.pe)
        x = torch.cat((self.node_emb(batch.x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(batch.edge_attr.float())

        for conv in self.convs: 
            x = conv(x, batch.edge_index, batch.batch, edge_attr=edge_attr)
        x = self.readout(x, batch.batch)
        return self.mlp(x)