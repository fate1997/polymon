import torch
from torch import nn
from torch_geometric.utils import unbatch_edge_index
from typing import Tuple
import math

class BatchRenorm(torch.jit.ScriptModule):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
        affine: bool = True,
    ):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(num_features, dtype=torch.float))
        self.register_buffer("running_std", torch.ones(num_features, dtype=torch.float))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.weight = torch.nn.Parameter(torch.ones(num_features, dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.zeros(num_features, dtype=torch.float))
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self) -> torch.Tensor:
        return (2 / 35000 * self.num_batches_tracked + 25 / 35).clamp_(1.0, 3.0)

    @property
    def dmax(self) -> torch.Tensor:
        return (5 / 20000 * self.num_batches_tracked - 25 / 20).clamp_(0.0, 5.0)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Mask is a boolean tensor used for indexing, where True values are padded
        i.e for 3D input, mask should be of shape (batch_size, seq_len)
        mask is used to prevent padded values from affecting the batch statistics
        """
        self._check_input_dim(x)
        if x.dim() > 2:
            x = x.transpose(1, -1)

        if self.training:
            dims = [i for i in range(x.dim() - 1)]
            if mask is not None:
                z = x[~mask]
                batch_mean = z.mean(0)
                batch_std = z.std(0, unbiased=False) + self.eps
            else:
                batch_mean = x.mean(dims)
                batch_std = x.std(dims, unbiased=False) + self.eps

            r = (batch_std.detach() / self.running_std.view_as(batch_std)).clamp_(1 / self.rmax, self.rmax)
            d = (
                (batch_mean.detach() - self.running_mean.view_as(batch_mean)) / self.running_std.view_as(batch_std)
            ).clamp_(-self.dmax, self.dmax)
            x = (x - batch_mean) / batch_std * r + d
            self.running_mean += self.momentum * (batch_mean.detach() - self.running_mean)
            self.running_std += self.momentum * (batch_std.detach() - self.running_std)
            self.num_batches_tracked += 1
        else:
            x = (x - self.running_mean) / self.running_std
        if self.affine:
            x = self.weight * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")

# class BatchRenorm(nn.Module):
#     def __init__(self, dim, eps=1e-8, momentum=0.1):
#         super(BatchRenorm, self).__init__()

#         self.dim = dim

#         self.bn = BatchRenorm1d(num_features=dim, eps=eps, momentum=momentum)

#     def forward(self, x):
#         return self.bn(x)


class BN(nn.Module):
    def __init__(self, dim, num_elements=None):
        super(BN, self).__init__()

        # self.bn = nn.BatchNorm1d(dim, eps=1e-8, momentum=0.1)
        self.bn = BatchRenorm1d(dim, eps=1e-8, momentum=0.1)

    def forward(self, x):
        return self.bn(x)


class LN(nn.Module):
    def __init__(self, hidden_dim, num_elements=None):
        super(LN, self).__init__()

        # if num_elements is not None:
        #     self.ln = nn.LayerNorm([num_elements, hidden_dim], eps=1e-8)
        # else:
        #     self.ln = nn.LayerNorm(hidden_dim, eps=1e-8)

        self.ln = nn.LayerNorm(hidden_dim, eps=1e-8)

    def forward(self, x):
        return self.ln(x)



########################################################
################# ESA edge utils #######################
########################################################

DEVICE = torch.device("cpu")

def get_adj_mask_from_edge_index_node(
    edge_index, batch_size, max_items, batch_mapping, xformers_or_torch_attn, use_bfloat16=True, device=DEVICE
):
    if xformers_or_torch_attn in ["torch"]:
        empty_mask_fill_value = False
        mask_dtype = torch.bool
        edge_mask_fill_value = True
    else:
        empty_mask_fill_value = -99999
        mask_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        edge_mask_fill_value = 0

    adj_mask = torch.full(
        size=(batch_size, max_items, max_items),
        fill_value=empty_mask_fill_value,
        device=device,
        dtype=mask_dtype,
        requires_grad=False,
    )

    edge_index_unbatched = unbatch_edge_index(edge_index, batch_mapping)
    edge_batch_non_cumulative = torch.cat(edge_index_unbatched, dim=1)

    edge_batch_mapping = batch_mapping.index_select(0, edge_index[0, :])

    adj_mask[
        edge_batch_mapping, edge_batch_non_cumulative[0, :], edge_batch_non_cumulative[1, :]
    ] = edge_mask_fill_value

    if xformers_or_torch_attn in ["torch"]:
        adj_mask = ~adj_mask

    adj_mask = adj_mask.unsqueeze(1)
    return adj_mask


def create_edge_adjacency_mask(edge_index, num_edges):
    # Get all the nodes in the edge index (source and target separately for undirected graphs)
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    # Create expanded versions of the source and target node tensors
    expanded_source_nodes = source_nodes.unsqueeze(1).expand(-1, num_edges)
    expanded_target_nodes = target_nodes.unsqueeze(1).expand(-1, num_edges)

    # Create the adjacency mask where an edge is adjacent if either node matches either node of other edges
    source_adjacency = expanded_source_nodes == expanded_source_nodes.t()
    target_adjacency = expanded_target_nodes == expanded_target_nodes.t()
    cross_adjacency = (expanded_source_nodes == expanded_target_nodes.t()) | (
        expanded_target_nodes == expanded_source_nodes.t()
    )

    adjacency_mask = source_adjacency | target_adjacency | cross_adjacency

    # Mask out self-adjacency by setting the diagonal to False
    adjacency_mask.fill_diagonal_(0)  # We use "0" here to indicate False in PyTorch boolean context

    return adjacency_mask


def get_first_unique_index(t):
    # This is taken from Stack Overflow :)
    unique, idx, counts = torch.unique(t, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    #zero = torch.tensor([0], device=DEVICE)
    zero = torch.zeros(1, dtype=cum_sum.dtype, device = cum_sum.device)
    cum_sum = torch.cat((zero, cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]

    return first_indicies


def generate_consecutive_tensor(input_tensor, final):
    # Calculate the length of each segment
    lengths = input_tensor[1:] - input_tensor[:-1]

    # Append the final length
    #lengths = torch.cat((lengths, torch.tensor([final - input_tensor[-1]], dtype=lengths.dtype, device = lengths.device)))
    extra = lengths.new_tensor([final - input_tensor[-1]])
    lengths = torch.cat((lengths, extra))
    # Create ranges for each segment
    ranges = [torch.arange(0, length, device=DEVICE) for length in lengths]

    # Concatenate all ranges into a single tensor
    result = torch.cat(ranges)

    return result

# This is needed if the standard "nonzero" method from PyTorch fails
# This alternative is slower but allows bypassing the problem until 64-bit
# support is available
def nonzero_chunked(ten, num_chunks):
    # This is taken from this pull request
    # https://github.com/facebookresearch/segment-anything/pull/569/files
    b, w_h = ten.shape
    total_elements = b * w_h

    # Maximum allowable elements in one chunk - as torch is using 32 bit integers for this function
    max_elements_per_chunk = 2**31 - 1

    # Calculate the number of chunks needed
    if total_elements % max_elements_per_chunk != 0:
        num_chunks += 1

    # Calculate the actual chunk size
    chunk_size = b // num_chunks
    if b % num_chunks != 0:
        chunk_size += 1

    # List to store the results from each chunk
    all_indices = []

    # Loop through the diff tensor in chunks
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, b)
        chunk = ten[start:end, :]

        # Get non-zero indices for the current chunk
        indices = chunk.nonzero()

        # Adjust the row indices to the original tensor
        indices[:, 0] += start

        all_indices.append(indices)

    # Concatenate all the results
    change_indices = torch.cat(all_indices, dim=0)

    return change_indices


def get_adj_mask_from_edge_index_edge(
    edge_index,
    batch_size,
    max_items,
    batch_mapping,
    xformers_or_torch_attn,
    use_bfloat16=True,
    device=DEVICE,
):
    
    edge_index = edge_index.to(device)
    batch_mapping = batch_mapping.to(device)
    
    if xformers_or_torch_attn in ["torch"]:
        empty_mask_fill_value = False
        mask_dtype = torch.bool
        edge_mask_fill_value = True
    else:
        empty_mask_fill_value = -99999
        mask_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        edge_mask_fill_value = 0

    adj_mask = torch.full(
        size=(batch_size, max_items, max_items),
        fill_value=empty_mask_fill_value,
        device=device,
        dtype=mask_dtype,
        requires_grad=False,
    )

    edge_batch_mapping = batch_mapping.index_select(0, edge_index[0, :])

    edge_adj_matrix = create_edge_adjacency_mask(edge_index, edge_index.shape[1])
    edge_adj_matrix = edge_adj_matrix.to(device)
    edge_batch_index_to_original_index = generate_consecutive_tensor(
        get_first_unique_index(edge_batch_mapping), edge_batch_mapping.shape[0]
    ).to(device)

    try:
        eam_nonzero = edge_adj_matrix.nonzero()
    except:
        # Adjust chunk size as desired
        eam_nonzero = nonzero_chunked(edge_adj_matrix, 3)
    eam_nonzero = eam_nonzero.to(device)

    adj_mask[
        edge_batch_mapping.index_select(0, eam_nonzero[:, 0]),
        edge_batch_index_to_original_index.index_select(0, eam_nonzero[:, 0]),
        edge_batch_index_to_original_index.index_select(0, eam_nonzero[:, 1]),
    ] = edge_mask_fill_value


    if xformers_or_torch_attn in ["torch"]:
        adj_mask = ~adj_mask

    adj_mask = adj_mask.unsqueeze(1)
    return adj_mask


def get_max_node_edge_global(dataset) -> Tuple[int, int]:
    max_node_global = 0
    max_edge_global = 0

    for data in dataset:
        num_edges = data.edge_index.size(1)
        num_nodes = data.x.size(0)
        if num_edges > max_edge_global:
            max_edge_global = num_edges
        if num_nodes > max_node_global:
            max_node_global = num_nodes
    return max_edge_global, max_node_global

def nearest_multiple_of_8(x):
    return math.ceil(x / 8) * 8