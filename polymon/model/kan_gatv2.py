import torch
import torch.nn as nn

from polymon.data.polymer import Polymer
from polymon.model.module import MLP, KANLinear, ReadoutPhase
from polymon.model.register import register_init_params
from polymon.model.base import BaseModel


@register_init_params
class KAN_GATv2(BaseModel):
    def __init__(
        self, 
        num_node_features: int, 
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 8,
        grid_size: int = 3,
        dropout: float = 0.1,
        pred_hidden_dim: int=128, 
        pred_dropout: float=0.2, 
        pred_layers:int=2,
    ):
        self.num_layers = num_layers
        self.input_dim = num_node_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.grid_size = grid_size
        self.dropout = dropout
        
        super(KAN_GATv2, self).__init__()
        self.message_passing = nn.ModuleList()
        for i in range(self.num_layers):
            self.message_passing.append(
                KAN_GATv2Layer(
                    self.input_dim if i == 0 else self.hidden_dim,
                    self.hidden_dim // self.num_heads,
                    self.num_heads,
                    dropout=self.dropout,
                    grid_size=self.grid_size,
                ))
        self.readout = ReadoutPhase(self.hidden_dim)
        self.predict = MLP(
            input_dim=self.hidden_dim * 2,
            hidden_dim=pred_hidden_dim,
            output_dim=1,
            n_layers=pred_layers,
            dropout=pred_dropout,
            activation='prelu',
            kan_mode=False,
            grid_size=self.grid_size,
        )
    
    def forward(self, batch: Polymer):
        x = batch.x
        for layer in self.message_passing:
            x = layer(x, batch.edge_index)
        output = self.readout(x, batch.batch)
        return self.predict(output)


class KAN_GATv2Layer(nn.Module):
    def __init__(
        self, 
        num_node_features: int, 
        output_dim: int, 
        num_heads: int,
        grid_size: int,
        activation=nn.PReLU(), 
        concat: bool = True, 
        residual: bool = True,
        bias: bool = True, dropout: float = 0.1, share_weights: bool = False):
        super(KAN_GATv2Layer, self).__init__()

        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.residual = residual
        self.activation = activation
        self.concat = concat
        self.dropout = dropout
        self.share_weights = share_weights
        self.grid_size = grid_size

        # Embedding by linear projection
        self.linear_src = KANLinear(num_node_features, output_dim * num_heads, grid_size, add_bias=False)
        if self.share_weights:
            self.linear_dst = self.linear_src
        else:
            self.linear_dst = KANLinear(num_node_features, output_dim * num_heads, grid_size, add_bias=False)

        # The learnable parameters to compute attention coefficients
        self.double_attn = nn.Parameter(torch.Tensor(1, num_heads, output_dim))

        # Bias and concat
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(output_dim * num_heads))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        if residual:
            if num_node_features == num_heads * output_dim:
                self.residual_linear = nn.Identity()
            else:
                self.residual_linear = KANLinear(num_node_features, num_heads * output_dim, grid_size, add_bias=False)
        else:
            self.register_parameter('residual_linear', None)

        # Some fixed function
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def init_params(self):
        nn.init.xavier_normal_(self.double_attn)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x, edge_index):

        # Input preprocessing
        edge_src_index, edge_dst_index = edge_index

        # Projection on the new space
        src_projected = self.linear_src(self.dropout(x)).view(-1, self.num_heads, self.output_dim)
        dst_projected = self.linear_dst(self.dropout(x)).view(-1, self.num_heads, self.output_dim)
        

        #######################################
        ############## Edge Attn ##############
        #######################################

        # Edge attention coefficients
        edge_attn = self.leakyReLU((src_projected.index_select(0, edge_src_index)
                                    + dst_projected.index_select(0, edge_dst_index)))
        edge_attn = (self.double_attn * edge_attn).sum(-1)
        exp_edge_attn = (edge_attn - edge_attn.max()).exp()

        # sum the edge scores to destination node
        num_nodes = x.shape[0]
        edge_node_score_sum = torch.zeros([num_nodes, self.num_heads],
                                          dtype=exp_edge_attn.dtype,
                                          device=exp_edge_attn.device)
        edge_dst_index_broadcast = edge_dst_index.unsqueeze(-1).expand_as(exp_edge_attn)
        edge_node_score_sum.scatter_add_(0, edge_dst_index_broadcast, exp_edge_attn)

        # normalized edge attention
        # edge_attn shape = [num_edges, num_heads, 1]
        exp_edge_attn = exp_edge_attn / (edge_node_score_sum.index_select(0, edge_dst_index) + 1e-16)
        exp_edge_attn = self.dropout(exp_edge_attn).unsqueeze(-1)

        # summation from one-hop atom
        edge_x_projected = src_projected.index_select(0, edge_src_index) * exp_edge_attn
        edge_output = torch.zeros([num_nodes, self.num_heads, self.output_dim],
                                  dtype=exp_edge_attn.dtype,
                                  device=exp_edge_attn.device)
        edge_dst_index_broadcast = (edge_dst_index.unsqueeze(-1)).unsqueeze(-1).expand_as(edge_x_projected)
        edge_output.scatter_add_(0, edge_dst_index_broadcast, edge_x_projected)

        output = edge_output
        # residual, concat, bias, activation
        if self.residual:
            output += self.residual_linear(x).view(num_nodes, -1, self.output_dim)
        if self.concat:
            output = output.view(-1, self.num_heads * self.output_dim)
        else:
            output = output.mean(dim=1)

        if self.bias is not None:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output