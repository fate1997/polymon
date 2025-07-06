import math
from typing import Callable, Union
from torch.nn.init import kaiming_uniform_, zeros_

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool

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
                 activation: str):
        super().__init__()
        
        activation = get_activation(activation)
        
        if n_layers == 1:
            self.layers = nn.Linear(input_dim, output_dim)
        else:
            self.layers = []
            for i in range(n_layers - 1):
                self.layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
                self.layers.append(activation)
                self.layers.append(nn.LayerNorm(hidden_dim))
                self.layers.append(nn.Dropout(dropout))

            self.layers.append(nn.Linear(hidden_dim, output_dim))
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