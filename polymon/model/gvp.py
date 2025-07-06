# GVP implementation from DiffHopp https://github.com/jostorge/diffusion-hopping/tree/main
import math
from abc import ABC
from functools import partial
from typing import Optional, Tuple, Union

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter_mean

from polymon.model.register import register_init_params
from polymon.model.base import BaseModel
from polymon.model.module import DenseLayer, ReadoutPhase, MLP
from polymon.data.polymer import Polymer

s_V = Tuple[torch.Tensor, torch.Tensor]


class GVPLayerNorm(nn.Module):
    def __init__(self, dims: Tuple[int, int], eps: float=0.00001) ->None:
        super().__init__()
        self.eps = math.sqrt(eps)
        self.scalar_size, self.vector_size = dims
        self.feature_layer_norm = nn.LayerNorm(self.scalar_size, eps=eps)
    
    def forward(self, x:Union[torch.Tensor, s_V]) -> Union[torch.Tensor, s_V]:
        if self.vector_size == 0:
            return self.feature_layer_norm(x)

        s, V = x
        if self.scalar_size!=1:
            s = self.feature_layer_norm(s)
        norm = torch.clip(
            torch.linalg.vector_norm(V, dim=(-1,-2), keepdim=True)
            / math.sqrt(self.vector_size),
            min=self.eps
        )

        V = V / norm
        return s, V


class GVPDropout(nn.Module):
    def __init__(self, p: float=0.5) -> None:
        super().__init__()
        self.dropout_features = nn.Dropout(p)
        self.dropout_vector = nn.Dropout1d(p)
    
    def forward(self, x: Union[torch.Tensor, s_V]) -> Union[torch.Tensor, s_V]:
        if isinstance(x, torch.Tensor):
            return self.dropout_features(x)

        s, V = x
        s = self.dropout_features(s)
        V = self.dropout_vector(V)
        return s, V


class GVPMessagePassing(MessagePassing, ABC):
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        edge_dims: Tuple[int, int],
        hidden_dims: Optional[Tuple[int, int]] = None,
        activations=(F.relu, torch.sigmoid),
        vector_gate: bool = False,
        attention: bool = True,
        aggr: str = "add",
        normalization_factor: float = 1.0,
    ):
        super().__init__(aggr)
        if hidden_dims is None:
            hidden_dims = out_dims

        in_scalar, in_vector = in_dims
        hidden_scalar, hidden_vector = hidden_dims

        edge_scalar, edge_vector = edge_dims

        self.out_scalar, self.out_vector = out_dims
        self.in_vector = in_vector
        self.hidden_scalar = hidden_scalar
        self.hidden_vector = hidden_vector
        self.normalization_factor = normalization_factor

        GVP_ = partial(GVP, activations=activations, vector_gate=vector_gate)
        self.edge_gvps = nn.Sequential(
            GVP_(
                (2 * in_scalar + edge_scalar, 2 * in_vector + edge_vector),
                hidden_dims,
            ),
            GVP_(hidden_dims, hidden_dims),
            GVP_(hidden_dims, out_dims, activations=(None, None)),
        )

        self.attention = attention
        if attention:
            self.attention_gvp = GVP_(
                out_dims,
                (1, 0),
                activations=(torch.sigmoid, None),
            )

    def forward(self, x: s_V, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> s_V:
        s, V = x
        v_dim = V.shape[-1]
        V = torch.flatten(V, start_dim=-2, end_dim=-1)
        return self.propagate(edge_index, s=s, V=V, edge_attr=edge_attr, v_dim=v_dim)

    def message(self, s_i, s_j, V_i, V_j, edge_attr, v_dim):
        V_i = V_i.view(*V_i.shape[:-1], self.in_vector, v_dim)
        V_j = V_j.view(*V_j.shape[:-1], self.in_vector, v_dim)
        edge_scalar, edge_vector = edge_attr

        s = torch.cat([s_i, s_j, edge_scalar], dim=-1)
        V = torch.cat([V_i, V_j, edge_vector], dim=-2)
        s, V = self.edge_gvps((s, V))

        if self.attention:
            att = self.attention_gvp((s, V))
            s, V = att * s, att[..., None] * V
        return self._combine(s, V)

    def update(self, aggr_out: torch.Tensor) -> s_V:
        s_aggr, V_aggr = self._split(aggr_out, self.out_scalar, self.out_vector)
        if self.aggr == "add" or self.aggr == "sum":
            s_aggr = s_aggr / self.normalization_factor
            V_aggr = V_aggr / self.normalization_factor
        return s_aggr, V_aggr

    @staticmethod
    def _combine(s, V) -> torch.Tensor:
        V = torch.flatten(V, start_dim=-2, end_dim=-1)
        return torch.cat([s, V], dim=-1)

    @staticmethod
    def _split(s_V: torch.Tensor, scalar: int, vector: int) -> s_V:
        s = s_V[..., :scalar]
        V = s_V[..., scalar:]
        V = V.view(*V.shape[:-1], vector, -1)
        return s, V

    def reset_parameters(self):
        for gvp in self.edge_gvps:
            gvp.reset_parameters()
        if self.attention:
            self.attention_gvp.reset_parameters()


class GVPConvLayer(GVPMessagePassing, ABC):
    def __init__(
        self,
        node_dims: Tuple[int, int],
        edge_dims: Tuple[int, int],
        drop_rate: float = 0.0,
        activations=(F.relu, torch.sigmoid),
        vector_gate: bool = False,
        residual: bool = True,
        attention: bool = True,
        aggr: str = "add",
        normalization_factor: float = 1.0,
    ):
        super().__init__(
            node_dims,
            node_dims,
            edge_dims,
            hidden_dims=node_dims,
            activations=activations,
            vector_gate=vector_gate,
            attention=attention,
            aggr=aggr,
            normalization_factor=normalization_factor,
        )
        self.residual = residual
        self.drop_rate = drop_rate
        GVP_ = partial(GVP, activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([GVPLayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([GVPDropout(drop_rate) for _ in range(2)])

        self.ff_func = nn.Sequential(
            GVP_(node_dims, node_dims),
            GVP_(node_dims, node_dims, activations=(None, None)),
        )
        self.residual = residual

    def forward(
        self,
        x: Union[s_V, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> s_V:

        s, V = super().forward(x, edge_index, edge_attr)
        if self.residual:
            s, V = self.dropout[0]((s, V))
            s, V = x[0] + s, x[1] + V
            s, V = self.norm[0]((s, V))

        x = (s, V)
        s, V = self.ff_func(x)

        if self.residual:
            s, V = self.dropout[1]((s, V))
            s, V = s + x[0], V + x[1]
            s, V = self.norm[1]((s, V))

        return s, V


class GVP(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        activations=(F.relu, torch.sigmoid),
        vector_gate: bool = False,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        in_scalar, in_vector = in_dims
        out_scalar, out_vector = out_dims
        self.sigma, self.sigma_plus = activations

        if self.sigma is None:
            self.sigma = nn.Identity()
        if self.sigma_plus is None:
            self.sigma_plus = nn.Identity()

        self.h = max(in_vector, out_vector)
        self.W_h = nn.Parameter(torch.empty((self.h, in_vector)))
        self.W_mu = nn.Parameter(torch.empty((out_vector, self.h)))

        self.W_m = nn.Linear(self.h + in_scalar, out_scalar)
        self.v = in_vector
        self.mu = out_vector
        self.n = in_scalar
        self.m = out_scalar
        self.vector_gate = vector_gate

        if vector_gate:
            self.sigma_g = nn.Sigmoid()
            self.W_g = nn.Linear(out_scalar, out_vector)

        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.W_h, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.W_mu, a=math.sqrt(5))
        self.W_m.reset_parameters()
        if self.vector_gate:
            self.W_g.reset_parameters()

    def forward(self, x: Union[torch.Tensor, s_V]) -> Union[torch.Tensor, s_V]:
        """Geometric vector perceptron"""
        s, V = (
            x if self.v > 0 else (x, torch.empty((x.shape[0], 0, 3), device=x.device))
        )

        assert (
            s.shape[-1] == self.n
        ), f"{s.shape[-1]} != {self.n} Scalar dimension mismatch"
        assert (
            V.shape[-2] == self.v
        ), f" {V.shape[-2]} != {self.v} Vector dimension mismatch"
        assert V.shape[0] == s.shape[0], "Batch size mismatch"

        V_h = self.W_h @ V
        V_mu = self.W_mu @ V_h
        s_h = torch.clip(torch.norm(V_h, dim=-1), min=self.eps)
        s_hn = torch.cat([s, s_h], dim=-1)
        s_m = self.W_m(s_hn)
        s_dash = self.sigma(s_m)
        if self.vector_gate:
            V_dash = self.sigma_g(self.W_g(self.sigma_plus(s_m)))[..., None] * V_mu
        else:
            v_mu = torch.clip(torch.norm(V_mu, dim=-1, keepdim=True), min=self.eps)
            V_dash = self.sigma_plus(v_mu) * V_mu
        return (s_dash, V_dash) if self.mu > 0 else s_dash


class GVPNetwork(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        hidden_dims: Tuple[int, int],
        num_layers: int,
        drop_rate: float = 0.0,
        attention: bool = False,
        normalization_factor: float=100.0,
        aggr: str = "add",
        activations=(F.silu, None),
        vector_gate: bool = True,
        eps=1e-4
    ) -> None:
        super().__init__()
        edge_dims = (1,1)

        self.eps = eps
        self.embedding_in = nn.Sequential(
            GVPLayerNorm(in_dims), 
            GVP(
                in_dims,
                hidden_dims,
                activations=(None,None),
                vector_gate=vector_gate
            ),
        )
        self.embedding_out = nn.Sequential(
            GVPLayerNorm(hidden_dims),
            GVP(
                hidden_dims,
                out_dims,
                activations=activations,
                vector_gate=vector_gate
            ),
        )
        self.edge_embedding = nn.Sequential(
            GVPLayerNorm(edge_dims),
            GVP(
                edge_dims,
                (hidden_dims[0],1),
                activations=(None, None),
                vector_gate=vector_gate
            )
        )

        self.layers = nn.ModuleList(
            [
                GVPConvLayer(
                    hidden_dims,
                    (hidden_dims[0], 1),
                    drop_rate=drop_rate,
                    activations=activations,
                    vector_gate=vector_gate,
                    residual=True,
                    attention=attention,
                    aggr=aggr,
                    normalization_factor=normalization_factor,
                )
                for _ in range(num_layers)
            ]
        )

    def get_edge_attr(self, edge_index, pos) -> s_V:
        V = pos[edge_index[0]] - pos[edge_index[1]]  # [n_edges, 3]
        s = torch.linalg.norm(V, dim=-1, keepdim=True)  # [n_edges, 1]
        V = (V / torch.clip(s, min=self.eps))[..., None, :]  # [n_edges, 1, 3]
        return s, V
    
    def forward(self, h, pos, edge_index) -> s_V:
        edge_attr = self.get_edge_attr(edge_index, pos)
        edge_attr = self.edge_embedding(edge_attr)

        h = self.embedding_in(h)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        
        return self.embedding_out(h)


@register_init_params
class GVPModel(BaseModel):
    def __init__(
        self, 
        in_node_nf: int, 
        hidden_dim: int = 64, 
        num_layers: int = 4, 
        attention: bool = False,
        drop_rate: float = 0.0,
        normalization_factor: float = 100,
        pred_hidden_dim: int = 128,
        pred_dropout: float = 0.2,
        pred_layers: int = 2,
    ):
        super().__init__()
        self.h_embedding = DenseLayer(in_node_nf, hidden_dim, activation='silu')
        self.gvp = GVPNetwork(
            in_dims=(hidden_dim, 0),
            out_dims=(hidden_dim, 1),
            hidden_dims=(hidden_dim, hidden_dim//2),
            drop_rate=drop_rate,
            vector_gate=True,
            num_layers=num_layers,
            attention=attention,
            normalization_factor=normalization_factor,
        )
        self.readout = ReadoutPhase(hidden_dim)
        
        self.predict = MLP(
            input_dim=2*hidden_dim,
            hidden_dim=pred_hidden_dim,
            output_dim=1,
            n_layers=pred_layers,
            dropout=pred_dropout,
            activation='prelu'
        )
    
    def forward(
        self, 
        batch: Polymer,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the EGNNDenoiser model.
        Args:
            t: Time. [batch_size, 1]
            x: Positions. [n_nodes, 3]
            h: Features. [n_nodes, h_dims]
            edge_index: [2, n_edges]
            context: [batch_size, context_node_nf]
            batch: [n_nodes]
        """
        x = batch.pos
        h = batch.x
        batch = batch.batch
        edge_index = getattr(batch, 'edge_index', None)
        
        # 1. Concatenate time and context (if provided) to h
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if edge_index is None:
            edge_index = radius_graph(x, r=5, batch=batch, max_num_neighbors=100)

        # 2. Forward pass through GVP
        h = self.h_embedding(h)
        h_final, _ = self.gvp(h, x, edge_index)

        return self.predict(self.readout(h_final, batch))