########################################################
################# ESA related modules ##################
########################################################
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
import torch.nn.functional as F

# from polymon.model.module import MLP
from polymon.model.esa.esa_utils import BN, LN
import polymon.model.esa.admin_torch as admin_torch



class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, dropout_p=0.0, xformers_or_torch_attn="xformers"):
        super(MAB, self).__init__()

        self.xformers_or_torch_attn = xformers_or_torch_attn

        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_V

        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.fc_q = nn.Linear(dim_Q, dim_V, bias=True)
        self.fc_k = nn.Linear(dim_K, dim_V, bias=True)
        self.fc_v = nn.Linear(dim_K, dim_V, bias=True)
        self.fc_o = nn.Linear(dim_V, dim_V, bias=True)

        # NOTE: xavier_uniform_ might work better for a few datasets
        xavier_normal_(self.fc_q.weight)
        xavier_normal_(self.fc_k.weight)
        xavier_normal_(self.fc_v.weight)
        xavier_normal_(self.fc_o.weight)

        # NOTE: this constant bias might work better for a few datasets
        # constant_(self.fc_q.bias, 0.01)
        # constant_(self.fc_k.bias, 0.01)
        # constant_(self.fc_v.bias, 0.01)
        # constant_(self.fc_o.bias, 0.01)

        # NOTE: this additional LN for queries/keys might be useful for some
        # datasets (currently it looks like DOCKSTRING)
        # It is similar to this paper https://arxiv.org/pdf/2302.05442
        # and https://github.com/lucidrains/x-transformers?tab=readme-ov-file#qk-rmsnorm

        # self.ln_q = nn.LayerNorm(dim_Q, eps=1e-8)
        # self.ln_k = nn.LayerNorm(dim_K, eps=1e-8)


    def forward(self, Q, K, adj_mask=None):
        batch_size = Q.size(0)
        E_total = self.dim_V
        assert E_total % self.num_heads == 0, "Embedding dim is not divisible by nheads"
        head_dim = E_total // self.num_heads

        Q = self.fc_q(Q)
        V = self.fc_v(K)
        K = self.fc_k(K)

        # Additional normalisation for queries/keys. See above
        # Q = self.ln_q(Q).to(torch.bfloat16)
        # K = self.ln_k(K).to(torch.bfloat16)

        Q = Q.view(batch_size, -1, self.num_heads, head_dim)
        K = K.view(batch_size, -1, self.num_heads, head_dim)
        V = V.view(batch_size, -1, self.num_heads, head_dim)

        if self.xformers_or_torch_attn in ["torch"]:
            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)


        if adj_mask is not None:
            adj_mask = adj_mask.to(Q.device)
            adj_mask = adj_mask.expand(-1, self.num_heads, -1, -1)

        if self.xformers_or_torch_attn == "xformers":
            # out = memory_efficient_attention(Q, K, V, attn_bias=adj_mask, p=self.dropout_p if self.training else 0)
            # out = out.reshape(batch_size, -1, self.num_heads * head_dim)
            pass
            
        elif self.xformers_or_torch_attn in ["torch"]:
            with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                out = F.scaled_dot_product_attention(
                    Q, K, V, attn_mask=adj_mask, dropout_p=self.dropout_p if self.training else 0, is_causal=False
                )
            out = out.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)


        out = out + F.mish(self.fc_o(out))

        return out


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn)

    def forward(self, X, adj_mask=None):
        return self.mab(X, X, adj_mask=adj_mask)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, dropout, xformers_or_torch_attn):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_normal_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, dropout_p=dropout, xformers_or_torch_attn=xformers_or_torch_attn)

    def forward(self, X, adj_mask=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, adj_mask=adj_mask)



class SABComplete(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        dropout,
        idx,
        norm_type,
        use_mlp=False,
        mlp_hidden_size=64,
        mlp_type="standard",
        node_or_edge="edge",
        xformers_or_torch_attn="xformers",
        residual_dropout=0,
        set_max_items=0,
        use_bfloat16=True,
        num_mlp_layers=3,
        pre_or_post="pre",
        num_layers_for_residual=0,
        use_mlp_ln=False,
        mlp_dropout=0,
    ):
        super(SABComplete, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.use_mlp = use_mlp
        self.idx = idx
        self.mlp_hidden_size = mlp_hidden_size
        self.node_or_edge = node_or_edge
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.residual_dropout = residual_dropout
        self.set_max_items = set_max_items
        self.use_bfloat16 = use_bfloat16
        self.num_mlp_layers = num_mlp_layers
        self.pre_or_post = pre_or_post

        if self.pre_or_post == "post":
            self.residual_attn = admin_torch.as_module(num_layers_for_residual)
            self.residual_mlp = admin_torch.as_module(num_layers_for_residual)

        if dim_in != dim_out:
            self.proj_1 = nn.Linear(dim_in, dim_out)
    

        self.sab = SAB(dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn)

        if self.idx != 2:
            bn_dim = self.set_max_items
        else:
            bn_dim = 32

        if norm_type == "LN":
            if self.pre_or_post == "post":
                if self.idx != 2:
                    self.norm = LN(dim_out, num_elements=self.set_max_items)
                else:
                    self.norm = LN(dim_out)
            else:
                if self.idx != 2:
                    self.norm = LN(dim_in, num_elements=self.set_max_items)
                else:
                    self.norm = LN(dim_in)
                    
        elif norm_type == "BN":
            self.norm = BN(bn_dim)

        self.mlp_type = mlp_type

        if self.use_mlp:
            if self.mlp_type == "standard":
                self.mlp = SmallMLP(
                    in_dim=dim_out,
                    out_dim=dim_out,
                    inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout,
                    num_layers=num_mlp_layers,
                    use_ln=use_mlp_ln,
                )
            
            elif self.mlp_type == "gated_mlp":
                self.mlp = GatedMLPMulti(
                    in_dim=dim_out,
                    out_dim=dim_out,
                    inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout,
                    num_layers=num_mlp_layers,
                    use_ln=use_mlp_ln,
                )

        if norm_type == "LN":
            if self.idx != 2:
                self.norm_mlp = LN(dim_out, num_elements=self.set_max_items)
            else:
                self.norm_mlp = LN(dim_out)
                
        elif norm_type == "BN":
            self.norm_mlp = BN(bn_dim)


    def forward(self, inp):
        X, edge_index, batch_mapping, max_items, adj_mask = inp

        if self.pre_or_post == "pre":
            X = self.norm(X)

        if self.idx == 1:
            out_attn = self.sab(X, adj_mask)
        else:
            out_attn = self.sab(X, None)

        if out_attn.shape[-1] != X.shape[-1]:
            X = self.proj_1(X)

        if self.pre_or_post == "pre":
            out = X + out_attn
        
        if self.pre_or_post == "post":
            out = self.residual_attn(X, out_attn)
            out = self.norm(out)

        if self.use_mlp:
            if self.pre_or_post == "pre":
                out_mlp = self.norm_mlp(out)
                out_mlp = self.mlp(out_mlp)
                if out.shape[-1] == out_mlp.shape[-1]:
                    out = out_mlp + out

            if self.pre_or_post == "post":
                out_mlp = self.mlp(out)
                if out.shape[-1] == out_mlp.shape[-1]:
                    out = self.residual_mlp(out, out_mlp)
                out = self.norm_mlp(out)

        if self.residual_dropout > 0:
            out = F.dropout(out, p=self.residual_dropout)

        return out, edge_index, batch_mapping, max_items, adj_mask


class PMAComplete(nn.Module):
    def __init__(
        self,
        dim_hidden,
        num_heads,
        num_outputs,
        norm_type,
        dropout=0,
        use_mlp=False,
        mlp_hidden_size=64,
        mlp_type="standard",
        xformers_or_torch_attn="xformers",
        set_max_items=0,
        use_bfloat16=True,
        num_mlp_layers=3,
        pre_or_post="pre",
        num_layers_for_residual=0,
        residual_dropout=0,
        use_mlp_ln=False,
        mlp_dropout=0,
    ):
        super(PMAComplete, self).__init__()

        self.use_mlp = use_mlp
        self.mlp_hidden_size = mlp_hidden_size
        self.num_heads = num_heads
        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.set_max_items = set_max_items
        self.use_bfloat16 = use_bfloat16
        self.residual_dropout = residual_dropout
        self.num_mlp_layers = num_mlp_layers
        self.pre_or_post = pre_or_post

        if self.pre_or_post == "post":
            self.residual_attn = admin_torch.as_module(num_layers_for_residual)
            self.residual_mlp = admin_torch.as_module(num_layers_for_residual)

        self.pma = PMA(dim_hidden, num_heads, num_outputs, dropout, xformers_or_torch_attn)

        if norm_type == "LN":
            self.norm = LN(dim_hidden)
        elif norm_type == "BN":
            self.norm = BN(self.set_max_items)

        self.mlp_type = mlp_type

        if self.use_mlp:
            if self.mlp_type == "standard":
                self.mlp = SmallMLP(
                    in_dim=dim_hidden,
                    out_dim=dim_hidden,
                    inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout,
                    num_layers=num_mlp_layers,
                    use_ln=use_mlp_ln,
                )

            elif self.mlp_type == "gated_mlp":
                self.mlp = GatedMLPMulti(
                    in_dim=dim_hidden,
                    out_dim=dim_hidden,
                    inter_dim=mlp_hidden_size,
                    dropout_p=mlp_dropout,
                    num_layers=num_mlp_layers,
                    use_ln=use_mlp_ln,
                )

        if norm_type == "LN":
            self.norm_mlp = LN(dim_hidden)
        elif norm_type == "BN":
            self.norm_mlp = BN(32)


    def forward(self, inp):
        X, edge_index, batch_mapping, max_items, adj_mask = inp

        if self.pre_or_post == "pre":
            X = self.norm(X)

        out_attn = self.pma(X)

        if self.pre_or_post == "pre" and out_attn.shape[-2] == X.shape[-2]:
            out = X + out_attn
        
        elif self.pre_or_post == "post" and out_attn.shape[-2] == X.shape[-2]:
            out = self.residual_attn(X, out_attn)
            out = self.norm(out)
        
        else:
            out = out_attn

        if self.use_mlp:
            if self.pre_or_post == "pre":
                out_mlp = self.norm_mlp(out)
                out_mlp = self.mlp(out_mlp)
                if out.shape[-2] == out_mlp.shape[-2]:
                    out = out_mlp + out

            if self.pre_or_post == "post":
                out_mlp = self.mlp(out)
                if out.shape[-2] == out_mlp.shape[-2]:
                    out = self.residual_mlp(out, out_mlp)
                out = self.norm_mlp(out)

        if self.residual_dropout > 0:
            out = F.dropout(out, p=self.residual_dropout)

        return out, edge_index, batch_mapping, max_items, adj_mask


class SmallMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        inter_dim,
        out_dim,
        dropout_p=0.0,
        num_layers=2,
        use_ln=False,
    ):
        super().__init__()

        self.mlp = []

        if num_layers == 1:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
                nn.Mish(),
            )
        else:
            for i in range(num_layers):
                if i == 0:
                    self.mlp.append(nn.Linear(in_dim, inter_dim))
                    if use_ln:
                        self.mlp.append(nn.LayerNorm(inter_dim))
                    self.mlp.append(nn.Mish())
                elif i != num_layers - 1:
                    self.mlp.append(nn.Linear(inter_dim, inter_dim))
                    if use_ln:
                        self.mlp.append(nn.LayerNorm(inter_dim))
                    self.mlp.append(nn.Mish())
                else:
                    self.mlp.append(nn.Linear(inter_dim, out_dim))

                if dropout_p > 0:
                    self.mlp.append(nn.Dropout(p=dropout_p))

            self.mlp = nn.Sequential(*self.mlp)


    def forward(self, x):
        return self.mlp(x)


class GatedMLPSingle(nn.Module):
    def __init__(
        self,
        in_dim,
        inter_dim,
        out_dim,
        dropout_p=0.0,
        use_ln=False,
    ):
        super().__init__()

        # Uncomment if you want dropout here
        # self.dropout_p = dropout_p

        self.fc1 = nn.Linear(in_dim, 2 * inter_dim, bias=True)
        self.fc2 = nn.Linear(inter_dim, out_dim, bias=True)
        self.use_ln = use_ln

        if self.use_ln:
            self.ln = nn.LayerNorm(2 * inter_dim, eps=1e-8)

        # if dropout_p > 0:
        #     self.dropout = nn.Dropout(p=dropout_p)


    def forward(self, x):
        if self.use_ln:
            y = self.ln(self.fc1(x))
        else:
            y = self.fc1(x)

        y, gate = y.chunk(2, dim=-1)
        #y = swiglu(gate, y)
        y = y * F.gelu(gate)
        # if self.dropout_p > 0:
        #     y = self.dropout(y)
        y = self.fc2(y)
        
        return y
    

class GatedMLPMulti(nn.Module):
    def __init__(
        self,
        in_dim,
        inter_dim,
        out_dim,
        dropout_p=0.0,
        num_layers=2,
        use_ln=False,
    ):
        super().__init__()

        self.mlp = []

        if num_layers == 1:
            self.mlp = nn.Sequential(
                GatedMLPSingle(in_dim, inter_dim, out_dim, dropout_p=dropout_p, use_ln=False),
                nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
                nn.Mish(),
            )
        else:
            for i in range(num_layers):
                if i == 0:
                    self.mlp.append(GatedMLPSingle(in_dim, inter_dim, inter_dim, dropout_p=dropout_p, use_ln=use_ln))
                elif i != num_layers - 1:
                    self.mlp.append(GatedMLPSingle(inter_dim, inter_dim, inter_dim, dropout_p=dropout_p, use_ln=use_ln))
                else:
                    self.mlp.append(GatedMLPSingle(inter_dim, inter_dim, out_dim, dropout_p=dropout_p, use_ln=use_ln))
                
                if dropout_p > 0:
                    self.mlp.append(nn.Dropout(p=dropout_p))

                self.mlp.append(nn.Mish())

        self.mlp = nn.Sequential(*self.mlp)


    def forward(self, x):
        return self.mlp(x)