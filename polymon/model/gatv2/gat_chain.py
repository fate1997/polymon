import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool

from polymon.data.polymer import Polymer
from polymon.model.base import BaseModel
from polymon.model.utils import MLP, init_weight
from polymon.model.register import register_init_params



@register_init_params
class GATChain(BaseModel):
    def __init__(
        self,
        num_atom_features: int, 
        hidden_dim: int, 
        num_layers: int, 
        num_heads: int=8, 
        dropout: float=0.1,
        pred_hidden_dim: int=128, 
        pred_dropout: float=0.2, 
        pred_layers:int=2,
        activation: str='prelu', 
        chain_length: int=10,
    ):
        super(GATChain, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                GATChainLayer(
                    num_atom_features if i == 0 else hidden_dim * chain_length,
                    hidden_dim // num_heads,
                    num_heads,
                    chain_length,
                    dropout=dropout,
                ))
        hidden_dim = hidden_dim * chain_length
        self.atom_weighting = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.atom_weighting.apply(init_weight)

        # prediction phase
        self.predict = MLP(
            input_dim=hidden_dim * 2,
            hidden_dim=pred_hidden_dim,
            output_dim=1,
            n_layers=pred_layers,
            dropout=pred_dropout,
            activation=activation
        )
        
    def forward(self, batch: Polymer): 
        x = batch.x.float()
        for layer in self.layers:
            x = layer(x, batch.edge_index, batch.bridge_index)
        
        batch_index = batch.batch
        weighted = self.atom_weighting(x)
        output1 = global_max_pool(x, batch_index)
        output2 = global_add_pool(weighted * x, batch_index)
        output = torch.cat([output1, output2], dim=1)

        return self.predict(output)


class GATChainLayer(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        output_dim: int,
        num_heads: int,
        chain_length: int,
        activation: nn.Module = nn.PReLU(),
        concat: bool = True,
        residual: bool = True,
        bias: bool = True,
        dropout: float = 0.1,
    ):
        super(GATChainLayer, self).__init__()

        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.chain_length = chain_length
        self.residual = residual
        self.activation = activation
        self.concat = concat
        self.dropout = dropout

        # Embedding by linear projection
        d_model = output_dim * num_heads
        self.linear_input = nn.Linear(num_node_features, d_model, bias=False)
        self.positional_encoding = SinPositionalEncoding(d_model, chain_length)
        self.double_attn = nn.Parameter(torch.Tensor(1, chain_length*num_heads, output_dim))

        # Bias and concat
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(chain_length*d_model))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        if residual:
            if num_node_features == d_model:
                self.residual_linear = nn.Identity()
            else:
                self.residual_linear = nn.Linear(num_node_features, d_model*chain_length, bias=False)
        else:
            self.register_parameter('residual_linear', None)

        # Some fixed function
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_input.weight)

        nn.init.xavier_uniform_(self.double_attn)
        if self.residual:
            if self.num_node_features != self.num_heads * self.output_dim:
                nn.init.xavier_uniform_(self.residual_linear.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        bridge_index: torch.Tensor,
    ):
        """
        Args:
            x: [num_nodes, num_input_features]
            edge_index: [2, num_edges]
            bridge_index: [2, num_bridges]
        """
        input_x = x.clone()
        n, c, h, d = x.shape[0], self.chain_length, self.num_heads, self.output_dim
        x = self.linear_input(self.dropout(x)).unsqueeze(1).repeat(1, c, 1)
        x = self.positional_encoding(x).reshape(n, c, h, d)

        # 1. Messages between bonded nodes
        src_bond, dst_bond = edge_index
        bond_attn = self.act((x[src_bond] + x[dst_bond])) # [num_bonds, c, h, d]
        
        # 2. Messages between bridge nodes
        src_bridge, dst_bridge = bridge_index
        channels = torch.arange(0, self.chain_length).to(x.device) # [c]
        prev_channels = torch.clamp(channels - 1, min=0)
        next_channels = torch.clamp(channels + 1, max=self.chain_length - 1)
        src2dst = self.act(x[dst_bridge] + x[src_bridge][:, next_channels])
        #! TODO: This is not correct.
        src2dst[:, -1] = 0 # no message for the last channel
        dst2src = self.act(x[src_bridge] + x[dst_bridge][:, prev_channels])
        dst2src[:, 0] = 0 # no message for the first channel
        reverse_bridge_index = torch.stack([dst_bridge, src_bridge], dim=0)
        bridge_index = torch.cat([bridge_index, reverse_bridge_index], dim=1)
        bridge_attn = torch.cat([src2dst, dst2src], dim=0)
        
        # 3. Merge messages from bonded and bridge nodes
        x_src = torch.cat([x[src_bond], x[src_bridge], x[dst_bridge]], dim=0).reshape(-1, c*h, d)
        edge_attn = torch.cat([bond_attn, bridge_attn], dim=0).reshape(-1, c*h, d)
        edge_index = torch.cat([edge_index, bridge_index], dim=1)
        edge_src_index, edge_dst_index = edge_index

        # Edge attention coefficients
        edge_attn = (self.double_attn * edge_attn).sum(-1)
        exp_edge_attn = (edge_attn - edge_attn.max()).exp()

        # sum the edge scores to destination node
        edge_node_score_sum = torch.zeros([n, c*h],
                                          dtype=exp_edge_attn.dtype,
                                          device=exp_edge_attn.device)
        edge_dst_index_broadcast = edge_dst_index.unsqueeze(-1).expand_as(exp_edge_attn)
        edge_node_score_sum.scatter_add_(0, edge_dst_index_broadcast, exp_edge_attn)

        # normalized edge attention
        # edge_attn shape = [num_edges, num_heads, 1]
        exp_edge_attn = exp_edge_attn / (edge_node_score_sum.index_select(0, edge_dst_index) + 1e-16)
        exp_edge_attn = self.dropout(exp_edge_attn).unsqueeze(-1)

        # summation from one-hop atom
        edge_x_projected = x_src.index_select(0, edge_src_index) * exp_edge_attn
        edge_output = torch.zeros([n, c*h, d],
                                  dtype=exp_edge_attn.dtype,
                                  device=exp_edge_attn.device)
        edge_dst_index_broadcast = (edge_dst_index.unsqueeze(-1)).unsqueeze(-1).expand_as(edge_x_projected)
        edge_output.scatter_add_(0, edge_dst_index_broadcast, edge_x_projected)

        output = edge_output
        # residual, concat, bias, activation
        if self.residual:
            output += self.residual_linear(input_x).view(n, c*h, d)
        if self.concat:
            output = output.view(-1, h*c*d)
        else:
            output = output.mean(dim=1)

        if self.bias is not None:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output


class SinPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding. The positional encoding is calculated
    using the following formula:

    .. math::
        PE(pos, 2i) = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)
        PE(pos, 2i+1) = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)
        
    where :math:`pos` is the position of the token in the sequence and 
    :math:`d_{model}` is the dimension of the model.

    Args:
        d_model (int): The dimension of the model.
        max_len (int): The maximum length of the sequence.
    """
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        
        double_i = torch.arange(0, d_model, step=2).float().unsqueeze(0)
        div_term = 10000.0 ** (double_i / d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe = pe.unsqueeze(0)
        
        self.pe = nn.Parameter(pe, requires_grad=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]