from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, SAGEConv

from polymon.model.base import BaseModel
from polymon.model.register import register_init_params
from polymon.model.utils import MLP, ReadoutPhase


@register_init_params
class GATv2SAGE(BaseModel):
    """GATv2 with SAGEConv.
    
    Args:
        num_atom_features (int): The number of atom features.
        hidden_dim (int): The number of hidden dimensions.
        num_layers (int): The number of layers.
        num_heads (int): The number of heads. Default to :obj:`8`.
        pred_hidden_dim (int): The number of hidden dimensions for the prediction 
            MLP. Default to :obj:`128`.
        pred_dropout (float): The dropout rate for the prediction MLP. Default to :obj:`0.2`.
        pred_layers (int): The number of layers for the prediction MLP. Default to :obj:`2`.
        activation (str): The activation function. Default to :obj:`'prelu'`.
        num_tasks (int): The number of tasks. Default to :obj:`1`.
        bias (bool): Whether to use bias. Default to :obj:`True`.
        dropout (float): The dropout rate. Default to :obj:`0.1`.
        edge_dim (int): The number of edge dimensions.
        sage_aggr (str): The aggregation function. Default to :obj:`'mean'`.
        sage_normalize (bool): Whether to normalize the output. Default to :obj:`False`.
        sage_project (bool): Whether to project the output. Default to :obj:`False`.
    """
    def __init__(
        self, 
        num_atom_features: int, 
        hidden_dim: int, 
        num_layers: int, 
        num_heads: int=8, 
        pred_hidden_dim: int=128, 
        pred_dropout: float=0.2, 
        pred_layers:int=2,
        activation: str='prelu', 
        num_tasks: int = 1,
        bias: bool = True, 
        dropout: float = 0.1, 
        edge_dim: int = None,
        sage_aggr: str = 'mean',
        sage_normalize: bool = False,
        sage_project: bool = False,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.input_dim = num_atom_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_tasks = num_tasks
        self.bias = bias
        self.edge_dim = edge_dim

        # GAT layers
        layers = []
        feature_per_layer = [num_atom_features] + [hidden_dim] * num_layers
        for i in range(num_layers):
            layer = GATv2Conv(
                in_channels=feature_per_layer[i] * (1 if i == 0 else num_heads),
                out_channels=feature_per_layer[i + 1],
                heads=num_heads,
                concat=True if i < len(feature_per_layer) - 2 else False,
                edge_dim=edge_dim,
                dropout=dropout,
                bias=bias
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        
        sage_layer = SAGEConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            aggr=sage_aggr,
            normalize=sage_normalize,
            project=sage_project
        )
        self.sage_layer = sage_layer

        # Readout phase
        self.readout_func = ReadoutPhase(hidden_dim)

        # prediction phase
        self.predict = MLP(
            input_dim=self.hidden_dim * 2,
            hidden_dim=pred_hidden_dim,
            output_dim=num_tasks,
            n_layers=pred_layers,
            dropout=pred_dropout,
            activation=activation
        )

    def forward(self, data: Data):
        """Forward pass.
        
        Args:
            data (Data): The batch of data.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        data = data.sort(sort_by_row=False)
        x = data.x
        
        for layer in self.layers:
            x = layer(x, data.edge_index, data.edge_attr)

        x = self.sage_layer(x, data.edge_index)

        mol_repr = self.readout_func(x, data.batch)

        return self.predict(mol_repr)