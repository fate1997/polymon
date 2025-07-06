from torch import nn
import torch.nn.functional as F

from polymon.model.esa.esa_module import SABComplete, PMAComplete
from polymon.model.esa.esa_utils import get_adj_mask_from_edge_index_node, get_adj_mask_from_edge_index_edge


class ESA(nn.Module):
    def __init__(
        self,
        num_outputs,
        dim_output,
        dim_hidden,
        num_heads,
        layer_types,
        node_or_edge="edge",
        xformers_or_torch_attn="xformers",
        pre_or_post="pre",
        norm_type="LN",
        sab_dropout=0.0,
        mab_dropout=0.0,
        pma_dropout=0.0,
        residual_dropout=0.0,
        pma_residual_dropout=0.0,
        use_mlps=False,
        mlp_hidden_size=64,
        num_mlp_layers=2,
        mlp_type="gated_mlp",
        mlp_dropout=0.0,
        use_mlp_ln=False,
        set_max_items=0,
        use_bfloat16=True,
    ):
        super(ESA, self).__init__()

        print(layer_types, dim_hidden, num_heads)
        assert len(layer_types) == len(dim_hidden) and len(layer_types) == len(num_heads)

        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.layer_types = layer_types
        self.node_or_edge = node_or_edge
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.pre_or_post = pre_or_post
        self.norm_type = norm_type
        self.sab_dropout = sab_dropout
        self.mab_dropout = mab_dropout
        self.pma_dropout = pma_dropout
        self.residual_dropout = residual_dropout
        self.pma_residual_dropout = pma_residual_dropout
        self.use_mlps = use_mlps
        self.mlp_hidden_size = mlp_hidden_size
        self.num_mlp_layers = num_mlp_layers
        self.mlp_type = mlp_type
        self.mlp_dropout = mlp_dropout
        self.use_mlp_ln = use_mlp_ln
        self.set_max_items = set_max_items
        self.use_bfloat16 = use_bfloat16
        
        layer_tracker = 0

        self.encoder = []

        pma_encountered = False
        dim_pma = -1

        has_pma = "P" in self.layer_types

        for lt in self.layer_types:
            layer_in_dim = dim_hidden[layer_tracker]
            layer_num_heads = num_heads[layer_tracker]
            if lt != "P":
                if has_pma:
                    layer_out_dim = dim_hidden[layer_tracker + 1]
                else:
                    layer_out_dim = dim_hidden[layer_tracker]
            else:
                layer_out_dim = -1

            if lt == "S" and not pma_encountered:
                self.encoder.append(
                    SABComplete(
                        layer_in_dim,
                        layer_out_dim,
                        layer_num_heads,
                        idx=0,
                        dropout=sab_dropout,
                        node_or_edge=node_or_edge,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        residual_dropout=residual_dropout,
                        use_mlp=use_mlps,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=set_max_items,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=len(dim_hidden) * 2,
                    )
                )
                
                print(f"Added encoder SAB ({layer_in_dim}, {layer_out_dim}, {layer_num_heads})")

            if lt == "M" and not pma_encountered:
                self.encoder.append(
                    SABComplete(
                        layer_in_dim,
                        layer_out_dim,
                        layer_num_heads,
                        idx=1,
                        dropout=mab_dropout,
                        node_or_edge=node_or_edge,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        residual_dropout=residual_dropout,
                        use_mlp=use_mlps,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=set_max_items,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=len(dim_hidden) * 2,
                    )
                )
                
                print(f"Added encoder MAB ({layer_in_dim}, {layer_out_dim}, {layer_num_heads})")
                
            if lt == "P":
                pma_encountered = True
                dim_pma = layer_in_dim
                self.decoder = [
                    PMAComplete(
                        layer_in_dim,
                        layer_num_heads,
                        num_outputs,
                        dropout=pma_dropout,
                        residual_dropout=pma_residual_dropout,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        use_mlp=use_mlps,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=set_max_items,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=len(dim_hidden) * 2,
                    )
                ]

                print(f"Added decoder PMA ({layer_in_dim}, {layer_num_heads})")

            if lt == "S" and pma_encountered:
                self.decoder.append(
                    SABComplete(
                        layer_in_dim,
                        layer_out_dim,
                        layer_num_heads,
                        idx=2,
                        dropout=sab_dropout,
                        node_or_edge=node_or_edge,
                        xformers_or_torch_attn=xformers_or_torch_attn,
                        pre_or_post=pre_or_post,
                        residual_dropout=residual_dropout,
                        use_mlp=use_mlps,
                        mlp_hidden_size=mlp_hidden_size,
                        mlp_dropout=mlp_dropout,
                        num_mlp_layers=num_mlp_layers,
                        use_mlp_ln=use_mlp_ln,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                        set_max_items=set_max_items,
                        use_bfloat16=use_bfloat16,
                        num_layers_for_residual=len(dim_hidden) * 2,
                    )
                )

                print(f"Added decoder SAB ({layer_in_dim}, {layer_out_dim}, {layer_num_heads})")

            if lt != "P":
                layer_tracker += 1

        self.encoder = nn.Sequential(*self.encoder)
        if pma_encountered:
            self.decoder = nn.Sequential(*self.decoder)

        self.decoder_linear = nn.Linear(dim_hidden[-1], dim_output, bias=True)

        if has_pma and dim_hidden[0] != dim_pma:
            self.out_proj = nn.Linear(dim_hidden[0], dim_pma)

            self.dim_pma = dim_pma


    def forward(self, X, edge_index, batch_mapping, num_max_items):

        if self.node_or_edge == "node":
            adj_mask = get_adj_mask_from_edge_index_node(
                edge_index=edge_index,
                batch_mapping=batch_mapping,
                batch_size=X.shape[0],
                max_items=self.set_max_items,
                xformers_or_torch_attn=self.xformers_or_torch_attn,
                use_bfloat16=self.use_bfloat16,
            )
        elif self.node_or_edge == "edge":
            adj_mask = get_adj_mask_from_edge_index_edge(
                edge_index=edge_index,
                batch_mapping=batch_mapping,
                batch_size=X.shape[0],
                max_items=self.set_max_items,
                xformers_or_torch_attn=self.xformers_or_torch_attn,
                use_bfloat16=self.use_bfloat16,
            )
        enc, _, _, _, _ = self.encoder((X, edge_index, batch_mapping, num_max_items, adj_mask))
        if hasattr(self, "dim_pma") and self.dim_hidden[0] != self.dim_pma:
            X = self.out_proj(X)

        enc = enc + X

        if hasattr(self, "decoder"):
            out, _, _, _, _ = self.decoder((enc, edge_index, batch_mapping, num_max_items, adj_mask))
            out = out.mean(dim=1)
        else:
            out = enc

        return F.mish(self.decoder_linear(out))