from polymon.data.polymer import Polymer
from polymon.model.base import BaseModel
from polymon.model.register import register_init_params
from polymon.model.utils import MLP


@register_init_params
class MLPWrapper(BaseModel):
    """MLPWrapper model.
    
    Args:
        in_channels (int): The number of input channels.
        hidden_dim (int): The number of hidden dimensions.
        num_layers (int): The number of layers.
        dropout (float): The dropout rate.
    """
    def __init__(
        self, 
        in_channels: int, 
        hidden_dim: int, 
        num_layers: int, 
        dropout: float,
        activation: str = 'prelu',
    ):
        super(MLPWrapper, self).__init__()
        self.mlp = MLP(
            input_dim=in_channels,
            hidden_dim=hidden_dim,
            output_dim=1,
            n_layers=num_layers,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, batch: Polymer):
        y_pred = self.mlp(batch.descriptors)
        return y_pred