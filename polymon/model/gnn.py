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

from polymon.model.esa.esa_model import ESA
from polymon.model.esa.esa_module import GatedMLPMulti
from polymon.model.esa.esa_utils import BN, LN, nearest_multiple_of_8
from torch_geometric.utils import to_dense_batch

import math
from typing import List

@register_init_params
class GATv2(BaseModel):
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
        feature_per_layer = [num_atom_features + num_descriptors] + [hidden_dim] * num_layers
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
            input_dim=feature_per_layer[-1] * 2,
            hidden_dim=pred_hidden_dim,
            output_dim=num_tasks,
            n_layers=pred_layers,
            dropout=pred_dropout,
            activation=activation
        )
        self.num_descriptors = num_descriptors
        
    def forward(self, batch: Polymer): 
        x = batch.x.float()
        if self.num_descriptors > 0:
            x = torch.cat([x, batch.descriptors[batch.batch]], dim=1)
        
        for layer in self.layers:
            x = layer(x, batch.edge_index, batch.edge_attr)
        
        batch_index = batch.batch
        weighted = self.atom_weighting(x)
        output1 = global_max_pool(x, batch_index)
        output2 = global_add_pool(weighted * x, batch_index)
        output = torch.cat([output1, output2], dim=1)
        
        # if self.num_descriptors > 0:
        #     output = torch.cat([output, batch.descriptors], dim=1)

        return self.predict(output)


@register_init_params
class AttentiveFPWrapper(BaseModel):
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
        return self.attentivefp(
            batch.x, 
            batch.edge_index, 
            batch.edge_attr, 
            batch.batch
        )


@register_init_params
class DimeNetPP(DimeNetPlusPlus, BaseModel):
    """DimeNet++ model wrapper."""
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
        z, pos, batch = data.z, data.pos, data.batch
        return super().forward(z, pos, batch)


@register_init_params
class GATPort(BaseModel):
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
        x = batch.x
        for layer in self.layers:
            x = layer(x, batch.edge_index)

        mol_repr_all = self.readout(x, batch.batch)
        
        return self.predict(mol_repr_all)


@register_init_params
class PNA(BaseModel):
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
        x = batch.x
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, batch.edge_index, batch.edge_attr.float())))
        return self.predict(self.readout(x, batch.batch))
    
    @classmethod
    def compute_deg(
        cls,
        train_loader: DataLoader,
    ) -> torch.Tensor:
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
<<<<<<< HEAD
class ESAWrapper(BaseModel):
    def __init__(
        self,
        task_type: Literal['regression'],
        num_features: int,
        graph_dim: int,
        edge_dim: int,
        batch_size: int = 128,
        lr: float = 0.001,
        linear_output_size: int = 1,
        scaler=None,
        xformers_or_torch_attn: str = "torch",
        hidden_dims: List[int] = [256, 256, 256, 256],
        num_heads: List[int] = [4, 4, 4, 4],
        num_sabs: int = None,
        sab_dropout: float = 0.0,
        mab_dropout: float = 0.0,
        pma_dropout: float = 0.0,
        apply_attention_on: str = "edge",
        layer_types: List[str] = ['M', 'M', 'S', 'P'],
        use_mlps: bool = False,
        set_max_items: int=0,
        regression_loss_fn: str = "mae",
        early_stopping_patience: int = 30,
        optimiser_weight_decay: float = 1e-10,
        mlp_hidden_size: int = 512,
        mlp_type: str = "gated_mlp",
        attn_residual_dropout: float = 0.0,
        norm_type: str = "LN",
        triu_attn_mask: bool = False,
        output_save_dir: str = None,
        use_bfloat16: bool = True,
        is_node_task: bool = False,
        train_mask = None,
        val_mask = None,
        test_mask = None,
        posenc: str = None,
        num_mlp_layers: int = 2,
        pre_or_post: str = "post",
        pma_residual_dropout: float = 0,
        use_mlp_ln: bool = True,
        mlp_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        assert task_type in ["regression"] 
        
        self.task_type = task_type
        self.num_features = num_features
        self.graph_dim = graph_dim
        self.edge_dim = edge_dim
        
        self.batch_size = batch_size
        self.lr = lr
        self.linear_output_size = linear_output_size
        self.scaler = scaler
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.num_sabs = num_sabs
        self.sab_dropout = sab_dropout
        self.mab_dropout = mab_dropout
        self.pma_dropout = pma_dropout
        self.apply_attention_on = apply_attention_on
        self.layer_types = layer_types
        self.use_mlps = use_mlps
        self.set_max_items = set_max_items
        self.regression_loss_fn = regression_loss_fn
        self.early_stopping_patience = early_stopping_patience
        self.optimiser_weight_decay = optimiser_weight_decay
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_type = mlp_type
        self.attn_residual_dropout = attn_residual_dropout
        self.norm_type = norm_type
        self.triu_attn_mask = triu_attn_mask
        self.output_save_dir = output_save_dir
        self.use_bfloat16 = use_bfloat16
        self.is_node_task = is_node_task
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.posenc = posenc
        self.num_mlp_layers = num_mlp_layers
        self.pre_or_post = pre_or_post
        self.pma_residual_dropout = pma_residual_dropout
        self.use_mlp_ln = use_mlp_ln
        self.mlp_dropout = mlp_dropout

        print(num_heads)
        
        if self.norm_type == "BN":
            norm_fn = BN
        elif self.norm_type == "LN":
            norm_fn = LN
        
        if self.apply_attention_on == "node":
            in_dim = self.num_features
            
            if self.mlp_type == "standard":
                self.node_mlp = MLP(
                    input_dim=in_dim,
                    hidden_dim=self.mlp_hidden_size,
                    output_dim=self.mlp_hidden_size,
                    n_layers=self.num_mlp_layers,
                    dropout=self.mlp_dropout,
                ) # to be modified
            elif self.mlp_type == "gated_mlp":
                self.node_mlp = GatedMLPMulti(
                    in_dim = in_dim,
                    out_dim = self.hidden_dims[0],
                    inter_dim = 128,
                    # activation = F.silu,
                    dropout_p = 0,
                    num_layers = self.num_mlp_layers,
                )
                
        elif self.apply_attention_on == "edge":
            in_dim = self.num_features
            in_dim = in_dim * 2
            if self.edge_dim is not None:
                in_dim += self.edge_dim
            
            if self.mlp_type == "standard":
                self.node_edge_mlp = MLP(
                    input_dim = in_dim,
                    hidden_dim = self.mlp_hidden_size,
                    output_dim = self.mlp_hidden_size,
                    n_layers = self.num_mlp_layers,
                    dropout = self.mlp_dropout,
                )
            elif self.mlp_type == "gated_mlp":
                self.node_edge_mlp = GatedMLPMulti(
                    in_dim = in_dim,
                    out_dim = self.hidden_dims[0],
                    inter_dim = 128,
                    # activation = F.silu,
                    dropout_p = 0,
                    num_layers = self.num_mlp_layers,
                )
                
        self.mlp_norm = norm_fn(self.hidden_dims[0])

        
        st_args = dict(
            num_outputs = 32,
            dim_output = self.graph_dim,
            xformers_or_torch_attn = self.xformers_or_torch_attn,
            dim_hidden = self.hidden_dims,
            num_heads = self.num_heads,
            sab_dropout = self.sab_dropout,
            mab_dropout = self.mab_dropout,
            pma_dropout = self.pma_dropout,
            use_mlps = self.use_mlps,
            mlp_hidden_size = self.mlp_hidden_size,
            mlp_type = self.mlp_type,
            node_or_edge = self.apply_attention_on,
            residual_dropout = self.attn_residual_dropout,
            set_max_items = nearest_multiple_of_8(self.set_max_items + 1),
            use_bfloat16 = self.use_bfloat16,
            layer_types = self.layer_types,
            num_mlp_layers = self.num_mlp_layers,
            pre_or_post = self.pre_or_post,
            pma_residual_dropout = self.pma_residual_dropout,
            use_mlp_ln = self.use_mlp_ln,
            mlp_dropout = self.mlp_dropout,
        )
        self.st_fast = ESA(**st_args)
        
        if self.mlp_type == "standard":
            self.output_mlp = MLP(
                in_dim = self.graph_dim,
                inter_dim = 128,
                out_dim = self.linear_output_size,
                use_ln = False,
                dropout_p = 0,
                num_layers = self.num_mlp_layers if self.num_mlp_layers > 1 else self.num_mlp_layers + 1,
            )
        elif self.mlp_type == "gated_mlp":
            self.output_mlp = GatedMLPMulti(
                in_dim = self.graph_dim,
                out_dim = self.linear_output_size,
                inter_dim = 128,
                # activation = F.silu,    
                dropout_p = 0,
                num_layers = self.num_mlp_layers,
            )
        
        self.output_mlp = nn.Linear(self.graph_dim, self.linear_output_size)
        
    def forward(self, batch: Polymer):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_mapping = batch.batch
        max_node = batch.max_node_global
        max_edge = batch.max_edge_global
        
        x = x.float()

        if self.apply_attention_on == "edge":
            num_max_items = max_edge
            num_max_items = torch.max(num_max_items).item()
            num_max_items = nearest_multiple_of_8(num_max_items + 1)
            source = x[edge_index[0, :], :]
            target = x[edge_index[1, :], :]
            h = torch.cat([source, target], dim=1)
            
            if self.edge_dim is not None and edge_attr is not None:
                h = torch.cat([h, edge_attr.float()], dim=1)

            h = self.node_edge_mlp(h)
            edge_batch_index = batch_mapping.index_select(0, edge_index[0, :])
            h, _ = to_dense_batch(h, edge_batch_index, fill_value=0, max_num_nodes=num_max_items)
            h = self.st_fast(h, edge_index, batch_mapping, num_max_items = num_max_items)
        else:
            num_max_items = max_node
            num_max_items = torch.max(num_max_items).item()
            num_max_items = nearest_multiple_of_8(num_max_items + 1)

            h = self.mlp_norm(self.node_mlp(x))
            h, dense_batch_index = to_dense_batch(h, batch_mapping, fill_value=0, max_num_nodes=num_max_items)
            h = self.st_fast(h, batch_mapping, num_max_items = num_max_items)

            if self.is_node_task:
                h = h[dense_batch_index]

        predictions = torch.flatten(self.output_mlp(h))

        return predictions.unsqueeze(1)
            
            
        
        
        
=======
class GraphTransformer(BaseModel):
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
        x = batch.x
        for layer in self.layers:
            x = layer(x, batch.edge_index)
        return self.predict(self.readout(x, batch.batch))
>>>>>>> main
