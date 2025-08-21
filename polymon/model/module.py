import math
from functools import partial
from typing import Callable, Union, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, zeros_
from torch_geometric.nn import global_add_pool, global_max_pool
from torch_geometric.nn.attention import PerformerAttention

ACTIVATION_REGISTER = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU(),
    'none': nn.Identity(),
    'prelu': nn.PReLU(),
}


def get_activation(activation: str):
    activation = activation.lower()
    if activation not in ACTIVATION_REGISTER:
        raise ValueError(f'Activation function {activation} not supported.')
    return ACTIVATION_REGISTER[activation]


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias != None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GRU):
        nn.init.orthogonal_(m.all_weights[0][0])
        nn.init.orthogonal_(m.all_weights[0][1])
        nn.init.zeros_(m.all_weights[0][2])
        nn.init.zeros_(m.all_weights[0][3])
    elif isinstance(m, nn.Embedding):
        nn.init.constant_(m.weight, 1)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class MLP(nn.Module):
    r"""Multi-layer perceptron.
    
    Args:
        input_dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        output_dim (int): Output dimension.
        n_layers (int): Number of layers.
        dropout (float): Dropout rate.
        activation (str): Activation function.
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 n_layers: int, 
                 dropout: float, 
                 activation: str,
                 kan_mode: bool = False,
                 grid_size: int = 10):
        super().__init__()
        if kan_mode:
            linear_layer = partial(KANLinear, grid_size=grid_size) # First layer might should not have bias
        else:
            linear_layer = nn.Linear
        
        activation = get_activation(activation)
        
        if n_layers == 1:
            self.layers = linear_layer(input_dim, output_dim)
        else:
            self.layers = []
            for i in range(n_layers - 1):
                self.layers.append(linear_layer(input_dim if i == 0 else hidden_dim, hidden_dim))
                self.layers.append(activation)
                self.layers.append(nn.LayerNorm(hidden_dim))
                self.layers.append(nn.Dropout(dropout))

            self.layers.append(linear_layer(hidden_dim, output_dim))
            self.layers = nn.Sequential(*self.layers)
        
        self.layers.apply(init_weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of :class:`MLP`.

        Args:
            x (Tensor): Input tensor.      
        """
        output = self.layers(x)
        return output


class ReadoutPhase(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # readout phase
        self.weighting = nn.Linear(dim, 1) 
        self.score = nn.Sigmoid() 
        
        nn.init.xavier_uniform_(self.weighting.weight)
        nn.init.constant_(self.weighting.bias, 0)
    
    def forward(self, x, batch):
        weighted = self.weighting(x)
        score = self.score(weighted)
        output1 = global_add_pool(score * x, batch)
        output2 = global_max_pool(x, batch)
        
        output = torch.cat([output1, output2], dim=1)
        return output


class DenseLayer(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = kaiming_uniform_,
        bias_init: Callable = zeros_,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(DenseLayer, self).__init__(in_features, out_features, bias)

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation is None:
            self._activation = nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented.")

    def reset_parameters(self):
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L106
        self.weight_init(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias_init(self.bias)


class ScaledSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


class KANLinear(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        out_dim: int, 
        grid_size: int, 
        add_bias: bool = True
    ):
        super(KANLinear,self).__init__()
        self.grid_size= grid_size
        self.add_bias = add_bias
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.fouriercoeffs = nn.Parameter(torch.randn(2, out_dim, input_dim, grid_size) / 
                                             (np.sqrt(input_dim) * np.sqrt(grid_size)))
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self,x):

        xshp = x.shape
        outshape = xshp[0:-1] + (self.out_dim,)
        x = x.view(-1, self.input_dim)
        k = torch.reshape(torch.arange(1, self.grid_size+1, device=x.device), (1, 1, 1, self.grid_size))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.grid_size))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.grid_size))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        if self.add_bias:
            y += self.bias
        
        y = y.view(outshape)
        return y


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1