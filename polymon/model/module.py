from typing import Literal

import torch
import torch.nn as nn


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


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding"""
    def __init__(self, d_model: int, max_len):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class RotaryPositionalEncoding(nn.Module):
    """This class implements Rotary Positional Embeddings (RoPE) and is adapted
    from `torchtune.modules.positional_embeddings`.
    
    Args:
        d_model (int): The dimension of the model.
        max_len (int): The maximum length of the sequence.
    """
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.dim = d_model
        self.base = 10000
        self.max_seq_len = max_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)

        rope_cache = self.cache[:seq_len]

        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        x_out = x_out.flatten(3)
        return x_out.type_as(x)


def get_positional_encoding(
    encoding_type: Literal["sin", "learnable", "rope"],
    d_model: int,
    max_len: int,
) -> nn.Module:
    """Get the positional encoding module.
    
    Args:
        encoding_type (Literal["sin", "learnable", "rope"]): The type of 
            positional encoding.
        d_model (int): The dimension of the model.
        max_len (int): The maximum length of the sequence.

    Returns:
        `nn.Module`: The positional encoding module.
    """
    encoding_type = encoding_type.lower()
    if encoding_type == "sin":
        return SinPositionalEncoding(d_model, max_len)
    elif encoding_type == "learnable":
        return LearnablePositionalEncoding(d_model, max_len)
    elif encoding_type == "rope":
        return RotaryPositionalEncoding(d_model, max_len)
    else:
        raise ValueError(f"Unknown positional encoding type: {encoding_type}")
    

class ResidualCNNBlock(nn.Module):
    """This module is implemented based on this repo:
    https://github.com/anindya-vedant/Genetic-ProtCNN/tree/master
    
    """
    def __init__(
        self,
        d_model: int,
        d_rate: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_rate = d_rate
        
        self.bn1 = nn.BatchNorm1d(d_model)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv1d(
            d_model, d_model, kernel_size=1, dilation=d_rate, padding='same'
        )
        
        self.bn2 = nn.BatchNorm1d(d_model)
        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)
        
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv2(x)
        
        return x + residual

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