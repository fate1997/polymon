from typing import Dict, Any, Literal

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (AttentiveFP, BatchNorm, DimeNetPlusPlus,
                                GATv2Conv, GINConv, PNAConv, TransformerConv,
                                global_add_pool, global_max_pool)
from torch_geometric.utils import degree

from polymon.data.polymer import Polymer
from polymon.model.base import BaseModel
from polymon.model.utils import MLP, ReadoutPhase, init_weight
from polymon.model.register import register_init_params


@register_init_params
class GATv2(BaseModel):
    r"""GATv2 model.
    
    Args:
        num_atom_features (int): The number of atom features.
        hidden_dim (int): The number of hidden dimensions.
        num_layers (int): The number of layers.
        num_heads (int): The number of heads. Default to :obj:`8`.
        pred_hidden_dim (int): The number of hidden dimensions in the prediction 
            layer. Default to :obj:`128`.
        pred_dropout (float): The dropout rate in the prediction layer. Default to 
            :obj:`0.2`.
        pred_layers (int): The number of layers in the prediction layer. Default 
            to :obj:`2`.
        activation (str): The activation function. Default to :obj:`prelu`.
        num_tasks (int): The number of tasks. Default to :obj:`1`.
        bias (bool): Whether to use bias in the GATv2Conv layers. Default to 
            :obj:`True`.
        dropout (float): The dropout rate in the GATv2Conv layers. Default to 
            :obj:`0.1`.
        edge_dim (int): The number of edge features. Default to :obj:`None`.
        num_descriptors (int): The number of descriptors. If not zero, the 
            descriptors will be concatenated to the output of the model. Default 
            to :obj:`0`.
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
        num_descriptors: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # update phase
        feature_per_layer = [num_atom_features] + [hidden_dim] * num_layers
        layers = []
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

        # readout phase
        self.atom_weighting = nn.Sequential(
            nn.Linear(feature_per_layer[-1], 1),
            nn.Sigmoid()
        )
        self.atom_weighting.apply(init_weight)

        # prediction phase
        self.predict = MLP(
            input_dim=feature_per_layer[-1] * 2 + num_descriptors,
            hidden_dim=pred_hidden_dim,
            output_dim=num_tasks,
            n_layers=pred_layers,
            dropout=pred_dropout,
            activation=activation
        )
        self.num_descriptors = num_descriptors
        
    def forward(self, batch: Polymer): 
        """Forward pass.
        
        Args:
            batch (Polymer): The batch of data.
        
        Returns:
            torch.Tensor: The output of the model.
        """
        x = batch.x.float()
        # if self.num_descriptors > 0:
        #     x = torch.cat([x, batch.descriptors[batch.batch]], dim=1)
        
        for layer in self.layers:
            x = layer(x, batch.edge_index, batch.edge_attr)
        
        batch_index = batch.batch
        weighted = self.atom_weighting(x)
        output1 = global_max_pool(x, batch_index)
        output2 = global_add_pool(weighted * x, batch_index)
        output = torch.cat([output1, output2], dim=1)
        
        if self.num_descriptors > 0:
            output = torch.cat([output, batch.descriptors], dim=1)

        return self.predict(output)
    
    def get_embeddings(self, batch: Polymer):
        """Get global graph embeddings.
        
        Args:
            batch (Polymer): The batch of data.
        
        Returns:
            torch.Tensor: The embeddings of the batch.
        """
        x = batch.x.float()
        for layer in self.layers:
            x = layer(x, batch.edge_index, batch.edge_attr)
        batch_index = batch.batch
        weighted = self.atom_weighting(x)
        output1 = global_max_pool(x, batch_index)
        output2 = global_add_pool(weighted * x, batch_index)
        output = torch.cat([output1, output2], dim=1)
        return output


@register_init_params
class AttentiveFPWrapper(BaseModel):
    r"""AttentiveFP model wrapper.
    
    Args:
        in_channels (int): The number of input channels.
        hidden_dim (int): The number of hidden dimensions.
        edge_dim (int): The number of edge features.
        num_layers (int): The number of layers.
        out_channels (int): The number of output channels. Default to :obj:`1`.
        num_timesteps (int): The number of timesteps. Default to :obj:`2`.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        edge_dim: int,
        num_layers: int,
        out_channels: int = 1,
        num_timesteps: int = 2,
    ):
        super(AttentiveFPWrapper, self).__init__()

        self.attentivefp = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=out_channels,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
        )

    def forward(self, batch: Polymer):
        """Forward pass.
        
        Args:
            batch (Polymer): The batch of data.
        
        Returns:
            torch.Tensor: The output of the model.
        """
        return self.attentivefp(
            batch.x, 
            batch.edge_index, 
            batch.edge_attr, 
            batch.batch
        )


@register_init_params
class DimeNetPP(DimeNetPlusPlus, BaseModel):
    r"""DimeNet++ model wrapper.
    
    :no-index:
    
    Args:
        hidden_dim (int): The number of hidden dimensions. Default to :obj:`128`.
        out_channels (int): The number of output channels. Default to :obj:`1`.
        num_layers (int): The number of layers. Default to :obj:`3`.
        int_emb_size (int): The number of embedding dimensions for the integer 
            features. Default to :obj:`64`.
        basis_emb_size (int): The number of embedding dimensions for the basis 
            features. Default to :obj:`8`.
        out_emb_channels (int): The number of output embedding channels. Default 
            to :obj:`256`.
        num_spherical (int): The number of spherical harmonics. Default to :obj:`7`.
        num_radial (int): The number of radial basis functions. Default to :obj:`6`.
        cutoff (float): The cutoff radius. Default to :obj:`5.0`.
        max_num_neighbors (int): The maximum number of neighbors. Default to 
            :obj:`32`.
        envelope_exponent (int): The exponent of the envelope function. Default 
            to :obj:`5`.
        num_before_skip (int): The number of layers before skip connections. 
            Default to :obj:`1`.
        num_after_skip (int): The number of layers after skip connections. 
            Default to :obj:`2`.
        num_output_layers (int): The number of output layers. Default to :obj:`2`.
        act (str): The activation function. Default to :obj:`swish`.
        output_initializer (str): The initializer for the output layer. Default to 
            :obj:`zeros`.
    """
    def __init__(
        self, 
        hidden_dim: int=128,
        out_channels: int=1,
        num_layers: int=3,
        int_emb_size: int=64,
        basis_emb_size: int=8,
        out_emb_channels: int=256,
        num_spherical: int=7,
        num_radial: int=6,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 2,
        act: str = 'swish',
        output_initializer: str = 'zeros'
    ):
        super().__init__(
            hidden_dim, 
            out_channels, 
            num_layers, 
            int_emb_size, 
            basis_emb_size,
            out_emb_channels, 
            num_spherical, 
            num_radial, 
            cutoff, 
            max_num_neighbors, 
            envelope_exponent, 
            num_before_skip, 
            num_after_skip, 
            num_output_layers, 
            act, 
            output_initializer
        )
        
    def forward(self, data: Polymer):
        """Forward pass.
        
        Args:
            data (Polymer): The batch of data.
        
        Returns:
            torch.Tensor: The output of the model.
        """
        z, pos, batch = data.z, data.pos, data.batch
        return super().forward(z, pos, batch)


@register_init_params
class GATPort(BaseModel):
    r"""GATPort model wrapper.
    
    Args:
        num_atom_features (int): The number of atom features.
        hidden_dim (int): The number of hidden dimensions.
        num_layers (int): The number of layers.
        num_heads (int): The number of heads. Default to :obj:`8`.
        pred_hidden_dim (int): The number of hidden dimensions in the prediction 
            layer. Default to :obj:`128`.
        pred_dropout (float): The dropout rate in the prediction layer. Default to 
            :obj:`0.2`.
        pred_layers (int): The number of layers in the prediction layer. Default 
            to :obj:`2`.
        activation (str): The activation function. Default to :obj:`prelu`.
        num_tasks (int): The number of tasks. Default to :obj:`1`.
        bias (bool): Whether to use bias in the GATv2Conv layers. Default to 
            :obj:`True`.
        dropout (float): The dropout rate in the GATv2Conv layers. Default to 
            :obj:`0.1`.
        edge_dim (int): The number of edge features. Default to :obj:`None`.
        num_descriptors (int): The number of descriptors. If not zero, the 
            descriptors will be concatenated to the output of the model. Default 
            to :obj:`0`.
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
        num_descriptors: int = 0,
    ):
        super().__init__()

        # update phase
        feature_per_layer = [num_atom_features] + [hidden_dim] * num_layers
        layers = []
        port_layers = []
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
            
            if i == num_layers - 1:
                continue
            port_layer = nn.TransformerEncoderLayer(
                d_model=feature_per_layer[i + 1] * num_heads,
                nhead=num_heads,
                dim_feedforward=pred_hidden_dim,
                dropout=pred_dropout,
            )
            port_layers.append(port_layer)
            
        self.layers = nn.ModuleList(layers)
        self.port_layers = nn.ModuleList(port_layers)

        # readout phase
        self.atom_weighting = nn.Sequential(
            nn.Linear(feature_per_layer[-1], 1),
            nn.Sigmoid()
        )
        self.atom_weighting.apply(init_weight)

        # prediction phase
        self.predict = MLP(
            input_dim=feature_per_layer[-1] * 2 + num_descriptors,
            hidden_dim=pred_hidden_dim,
            output_dim=num_tasks,
            n_layers=pred_layers,
            dropout=pred_dropout,
            activation=activation
        )
        self.num_descriptors = num_descriptors
        
    def forward(self, batch: Polymer): 
        x = batch.x.float()
        port_mask = batch.z == 0
        for i, layer in enumerate(self.layers):
            x = layer(x, batch.edge_index, batch.edge_attr)
            if i < len(self.layers) - 1:
                x = self.update_port_features(x, port_mask, batch.batch, i)
        
        batch_index = batch.batch
        weighted = self.atom_weighting(x)
        output1 = global_max_pool(x, batch_index)
        output2 = global_add_pool(weighted * x, batch_index)
        output = torch.cat([output1, output2], dim=1)
        
        if self.num_descriptors > 0:
            output = torch.cat([output, batch.descriptors], dim=1)

        return self.predict(output)

    def update_port_features(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        batch: torch.Tensor,
        ith_layer: int,
        max_num: int = 4,
    ) -> torch.Tensor:
        
        batch_size = batch.max().item() + 1
        dim = x.size(1)
        max_total = batch_size * max_num
        
        special_nodes = x[mask]
        special_batches = batch[mask]
        sorted_batches, perm = special_batches.sort()
        special_nodes_sorted = special_nodes[perm]

        # For each batch, create row indices to place the special nodes
        counts = torch.bincount(special_batches, minlength=batch_size)
        output = torch.zeros((max_total, dim), dtype=x.dtype, device=x.device)

        idx_in_batch = torch.cat([torch.arange(c, device=batch.device) for c in counts])
        mask_keep = idx_in_batch < max_num  # mask for truncating per batch
        valid_nodes = special_nodes_sorted[mask_keep]
        valid_batches = sorted_batches[mask_keep]
        idx_in_batch = idx_in_batch[mask_keep]

        flat_index = valid_batches * max_num + idx_in_batch

        output[flat_index] = valid_nodes
        port_features = output.view(batch_size, max_num, dim)
        
        # Update port features
        row_idx = torch.arange(max_num, device=counts.device).unsqueeze(0)
        padding_mask = row_idx >= counts.unsqueeze(1)
        output = self.port_layers[ith_layer](
            port_features.transpose(0, 1),
            src_key_padding_mask=padding_mask
        ).transpose(0, 1) # (batch_size, max_num, dim)

        # Assign port features to the original features
        special_indices = mask.nonzero(as_tuple=False).squeeze(1)  # (n_special,)
        valid_mask = idx_in_batch < output.shape[1]
        special_indices = special_indices[valid_mask]
        special_batches = special_batches[valid_mask]
        idx_in_batch = idx_in_batch[valid_mask]
        updated = output[special_batches, idx_in_batch]
        x[special_indices] = updated
        return x


@register_init_params
class GATv2VirtualNode(BaseModel):
    r"""GATv2VirtualNode model. Add virtual node as the graph node and use its
    features as the graph embedding.
    
    Args:
        num_atom_features (int): The number of atom features.
        hidden_dim (int): The number of hidden dimensions.
        num_layers (int): The number of layers.
        num_heads (int): The number of heads. Default to :obj:`8`.
        pred_hidden_dim (int): The number of hidden dimensions in the prediction 
            layer. Default to :obj:`128`.
        pred_dropout (float): The dropout rate in the prediction layer. Default to 
            :obj:`0.2`.
        pred_layers (int): The number of layers in the prediction layer. Default 
            to :obj:`2`.
        activation (str): The activation function. Default to :obj:`prelu`.
        num_tasks (int): The number of tasks. Default to :obj:`1`.
        bias (bool): Whether to use bias in the GATv2Conv layers. Default to 
            :obj:`True`.
        dropout (float): The dropout rate in the GATv2Conv layers. Default to 
            :obj:`0.1`.
        edge_dim (int): The number of edge features. Default to :obj:`None`.
        num_descriptors (int): The number of descriptors. If not zero, the 
            descriptors will be concatenated to the output of the model. Default 
            to :obj:`0`.
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
        num_descriptors: int = 0,
    ):
        super().__init__()

        # update phase
        feature_per_layer = [num_atom_features] + [hidden_dim] * num_layers
        layers = []
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
        self.project_vn = nn.Linear(num_descriptors, num_atom_features)

        # prediction phase
        self.predict = MLP(
            input_dim=feature_per_layer[-1],
            hidden_dim=pred_hidden_dim,
            output_dim=num_tasks,
            n_layers=pred_layers,
            dropout=pred_dropout,
            activation=activation
        )
        
    def forward(self, batch: Polymer): 
        """Forward pass.
        
        Args:
            batch (Polymer): The batch of data. It should have :obj:`descriptors`
                attribute.
        
        Returns:
            torch.Tensor: The output of the model.
        """
        virtual_features = self.project_vn(batch.descriptors)
        x = self.add_virtual_features(batch.x, batch.batch, virtual_features)
        
        # Pass through GNN layers
        for layer in self.layers:
            x = layer(x, batch.edge_index, batch.edge_attr)

        # Get features of last atom in each molecule
        batch_idx = batch.batch
        _, last_indices = torch.unique_consecutive(
            batch_idx, return_inverse=False, return_counts=True
        )
        last_pos = torch.cumsum(last_indices, dim=0) - 1  # subtract 1 to get the index
        mask = torch.zeros_like(batch_idx, dtype=torch.bool)
        mask[last_pos] = True
        
        output = x[mask]

        return self.predict(output)
    
    def add_virtual_features(
        self,
        features: torch.Tensor,
        batch_indices: torch.Tensor,
        virtual_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add virtual features to the last element of each batch using vectorized operations.
        
        Args:
            features: Input features tensor [num_elements, feature_dim]
            batch_indices: Batch indices for each element [num_elements]
            virtual_features: Virtual features to add [batch_size, feature_dim]
            
        Returns:
            Updated features tensor
        """
        batch_size = batch_indices.max().item() + 1
        device = features.device
        
        # Use scatter_reduce to find the maximum index for each batch
        element_range = torch.arange(batch_indices.size(0), device=device)
        last_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        last_indices.scatter_reduce_(0, batch_indices, element_range, reduce='amax')
        
        # Create a mask for the last elements
        last_element_mask = torch.zeros_like(batch_indices, dtype=torch.bool)
        last_element_mask[last_indices] = True
        
        # Add virtual features to the last elements
        features[last_element_mask] = features[last_element_mask] + virtual_features
        return features


@register_init_params
class GIN(BaseModel):
    r"""GIN model wrapper.
    
    Args:
        num_atom_features (int): The number of atom features.
        hidden_dim (int): The number of hidden dimensions.
        num_layers (int): The number of layers.
        dropout (float): The dropout rate. Default to :obj:`0.2`.
        n_mlp_layers (int): The number of layers in the MLP. Default to :obj:`2`.
        pred_hidden_dim (int): The number of hidden dimensions in the prediction 
            layer. Default to :obj:`128`.
        pred_dropout (float): The dropout rate in the prediction layer. Default to 
            :obj:`0.2`.
        pred_layers (int): The number of layers in the prediction layer. Default 
            to :obj:`2`.
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
    ):
        super(GIN, self).__init__()

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
                    activation='prelu'
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
            activation='prelu'
        )

    def forward(self, batch: Polymer):
        """Forward pass.
        
        Args:
            batch (Polymer): The batch of data.
        
        Returns:
            torch.Tensor: The output of the model.
        """
        x = batch.x
        for layer in self.layers:
            x = layer(x, batch.edge_index)

        mol_repr_all = self.readout(x, batch.batch)
        
        return self.predict(mol_repr_all)


@register_init_params
class PNA(BaseModel):
    r"""PNA model wrapper.
    
    Args:
        in_channels (int): The number of input channels.
        hidden_dim (int): The number of hidden dimensions.
        num_layers (int): The number of layers.
        deg (torch.Tensor): The degree tensor.
        towers (int): The number of towers. Default to :obj:`1`.
        edge_dim (int): The number of edge features. Default to :obj:`None`.
        pred_hidden_dim (int): The number of hidden dimensions in the prediction 
            layer. Default to :obj:`128`.
        pred_dropout (float): The dropout rate in the prediction layer. Default to 
            :obj:`0.2`.
        pred_layers (int): The number of layers in the prediction layer. Default 
            to :obj:`2`.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        deg: torch.Tensor,
        towers: int = 1,
        edge_dim: int = None,
        pred_hidden_dim: int=128,
        pred_dropout: float=0.2,
        pred_layers:int=2,
    ):
        super().__init__()
        
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            conv = PNAConv(
                in_channels=in_channels if i == 0 else hidden_dim, 
                out_channels=hidden_dim,
                aggregators=aggregators, 
                scalers=scalers,
                deg=deg,
                edge_dim=edge_dim, 
                towers=towers, 
                pre_layers=1, 
                post_layers=1,
                divide_input=False
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_dim))

        self.readout = ReadoutPhase(hidden_dim)

        self.predict = MLP(
            input_dim=2*hidden_dim,
            hidden_dim=pred_hidden_dim,
            output_dim=1,
            n_layers=pred_layers,
            dropout=pred_dropout,
            activation='prelu'
        )

    def forward(self, batch: Polymer):
        """Forward pass.
        
        Args:
            batch (Polymer): The batch of data. It should have been preprocessed
                by :class:`polymon.data.utils.AddRandomWalkPE`.
        
        Returns:
            torch.Tensor: The output of the model.
        """
        x = batch.x
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, batch.edge_index, batch.edge_attr.float())))
        return self.predict(self.readout(x, batch.batch))
    
    @classmethod
    def compute_deg(
        cls,
        train_loader: DataLoader,
    ) -> torch.Tensor:
        """Compute the degree tensor.
        
        Args:
            train_loader (DataLoader): The training loader.
        
        Returns:
            torch.Tensor: The degree tensor.
        """
        max_degree = -1
        for data in train_loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in train_loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        return deg


@register_init_params
class GraphTransformer(BaseModel):
    r"""GraphTransformer model wrapper.
    
    Args:
        in_channels (int): The number of input channels.
        hidden_dim (int): The number of hidden dimensions.
        num_layers (int): The number of layers.
        num_heads (int): The number of heads. Default to :obj:`8`.
        dropout (float): The dropout rate. Default to :obj:`0.2`.
        pred_hidden_dim (int): The number of hidden dimensions in the prediction 
            layer. Default to :obj:`128`.
        pred_dropout (float): The dropout rate in the prediction layer. Default to 
            :obj:`0.2`.
        pred_layers (int): The number of layers in the prediction layer. Default 
            to :obj:`2`.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 8,
        dropout: float=0.2,
        pred_hidden_dim: int=128,
        pred_dropout: float=0.2,
        pred_layers:int=2,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = TransformerConv(
                in_channels=in_channels if i == 0 else hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
            )
            self.layers.append(layer)
        
        self.readout = ReadoutPhase(hidden_dim)

        self.predict = MLP(
            input_dim=2*hidden_dim,
            hidden_dim=pred_hidden_dim,
            output_dim=1,
            n_layers=pred_layers,
            dropout=pred_dropout,
            activation='prelu'
        )
    
    def forward(self, batch: Polymer):
        """Forward pass.
        
        Args:
            batch (Polymer): The batch of data.
        
        Returns:
            torch.Tensor: The output of the model.
        """
        x = batch.x
        for layer in self.layers:
            x = layer(x, batch.edge_index)
        return self.predict(self.readout(x, batch.batch))