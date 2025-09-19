from torch import nn

from polymon.data.polymer import Polymer
from polymon.model.base import BaseModel
from polymon.model.gnn import GINConv
from polymon.model.register import register_init_params
from polymon.model.utils import MLP, ReadoutPhase
from polymon.model.kan.fast_kan import FastKAN


@register_init_params
class KAN_GIN(BaseModel):
    """KAN-augmented GIN.
    
    Args:
        num_atom_features (int): The number of atom features.
        hidden_dim (int): The number of hidden dimensions.
        num_layers (int): The number of layers.
        dropout (float): The dropout rate. Default to :obj:`0.2`.
        n_mlp_layers (int): The number of MLP layers. Default to :obj:`2`.
        pred_hidden_dim (int): The number of hidden dimensions for the prediction 
            MLP. Default to :obj:`128`.
        pred_dropout (float): The dropout rate for the prediction MLP. Default to :obj:`0.2`.
        pred_layers (int): The number of layers for the prediction MLP. Default to :obj:`2`.
        grid_size (int): The number of grid points. Default to :obj:`10`.
    """
    def __init__(
        self,
        num_atom_features: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float=0.2,
        n_mlp_layers: int=2,
        pred_hidden_dim: int=128,
        pred_dropout: float=0.2,
        pred_layers:int=2,
        grid_size: int = 10,
    ):
        super().__init__()

        # GAT layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = GINConv(
                nn=MLP(
                    input_dim=num_atom_features if i==0 else hidden_dim, 
                    hidden_dim=hidden_dim, 
                    output_dim=hidden_dim,
                    n_layers=n_mlp_layers,
                    dropout=dropout,
                    activation='prelu',
                    kan_mode=True,
                    grid_size=grid_size,
                )
            )
            self.layers.append(layer)
        
        # Readout phase
        self.readout = ReadoutPhase(hidden_dim)

        # prediction phase
        self.predict = MLP(
            input_dim=2*hidden_dim,
            hidden_dim=pred_hidden_dim,
            output_dim=1,
            n_layers=pred_layers,
            dropout=pred_dropout,
            activation='prelu',
            kan_mode=True,
            grid_size=grid_size,
        )

    def forward(self, batch: Polymer):
        """Forward pass.
        
        Args:
            batch (Polymer): The batch of data.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        x = batch.x
        for layer in self.layers:
            x = layer(x, batch.edge_index)

        mol_repr_all = self.readout(x, batch.batch)
        
        return self.predict(mol_repr_all)
    

@register_init_params
class FastKAN_GIN(BaseModel):
    """Fast KAN-augmented GIN.
    
    Args:
        num_atom_features (int): The number of atom features.
        hidden_dim (int): The number of hidden dimensions.
        num_layers (int): The number of layers.
        dropout (float): The dropout rate. Default to :obj:`0.2`.
        n_mlp_layers (int): The number of MLP layers. Default to :obj:`2`.
        pred_hidden_dim (int): The number of hidden dimensions for the prediction 
            MLP. Default to :obj:`128`.
        grid_min (float): The minimum value of the grid. Default to :obj:`-4.0`.
        grid_max (float): The maximum value of the grid. Default to :obj:`3.0`.
        num_grids (int): The number of grids. Default to :obj:`10`.
    """
    def __init__(
        self,
        num_atom_features: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float=0.2,
        n_mlp_layers: int=2,
        pred_hidden_dim: int=128,
        grid_min: float=-4.0,
        grid_max: float=3.0,
        num_grids: int=10,
    ):
        super().__init__()

        # GAT layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = GINConv(
                nn=FastKAN(
                    layers_hidden=[num_atom_features if i==0 else hidden_dim, hidden_dim, hidden_dim],
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                )
            )
            self.layers.append(layer)
        
        self.predict = FastKAN(
            layers_hidden=[2*hidden_dim, pred_hidden_dim, 1],
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
        )
        self.readout = ReadoutPhase(hidden_dim)

    def forward(self, batch: Polymer):
        """Forward pass.
        
        Args:
            batch (Polymer): The batch of data.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        x = batch.x
        for layer in self.layers:
            x = layer(x, batch.edge_index)

        mol_repr_all = self.readout(x, batch.batch)
        
        return self.predict(mol_repr_all)
                    