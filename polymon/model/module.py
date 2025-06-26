import torch
import torch.nn as nn

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