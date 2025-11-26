from kan import KAN

from polymon.model.base import BaseModel
from polymon.model.register import register_init_params
from polymon.data.polymer import Polymer


@register_init_params
class KANWrapper(BaseModel):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int = 1,
        grid: int = 5,
        k: int = 3,
        device: str = 'cuda',
    ):
        super(KANWrapper, self).__init__()
        self.kan = KAN(
            width=[in_channels] + [hidden_dim] * num_layers + [1],
            grid=grid,
            k=k,
            auto_save=False,
            device=device,
            ckpt_path='.kan/'
        )
    
    def forward(self, batch: Polymer):
        desc = batch.descriptors
        y = self.kan(desc)
        return y