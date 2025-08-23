import torch
from torch import nn

from polymon.data.polymer import Polymer
from polymon.model.base import BaseModel
from polymon.model.register import register_init_params
from polymon.model.utils import KANLinear


@register_init_params
class FourierKANWrapper(BaseModel):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        grid_size: int = 5,
        add_bias: bool = True,
        add_act: bool = False,
    ):
        super(FourierKANWrapper, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers.append(KANLinear(
                input_dim=in_channels if i == 0 else hidden_dim,
                out_dim=hidden_dim,
                grid_size=grid_size,
                add_bias=add_bias,
            ))
            if add_act:
                self.layers.append(nn.SiLU())
        self.layers.append(KANLinear(
            input_dim=hidden_dim,
            out_dim=1,
            grid_size=grid_size,
            add_bias=add_bias,
        ))
        
    def forward(self, batch: Polymer):
        desc = batch.descriptors
        for layer in self.layers:
            desc = layer(desc)
        return desc