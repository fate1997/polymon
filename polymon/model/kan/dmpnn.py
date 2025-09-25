from typing import Any, List, Sequence, Union

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch

from polymon.model.register import register_init_params
from polymon.model.base import BaseModel
from polymon.model.utils import KANLinear


@register_init_params
class KAN_DMPNN(BaseModel):
    """KAN-augmented DMPNN.
    
    Args:
        mode (str): The mode of the model. Default to :obj:`regression`.
        n_classes (int): The number of classes. Default to :obj:`3`.
        n_tasks (int): The number of tasks. Default to :obj:`1`.
        global_features_size (int): The size of the global features. Default to 
            :obj:`0`.
        atom_fdim (int): The number of atom features. Default to :obj:`133`.
        bond_fdim (int): The number of bond features. Default to :obj:`14`.
        hidden_dim (int): The number of hidden dimensions. Default to :obj:`300`.
        num_layers (int): The number of layers. Default to :obj:`3`.
        bias (bool): Whether to use bias. Default to :obj:`False`.
        enc_activation (str): The activation function for the encoder. Default to 
            :obj:`relu`.
        enc_dropout_p (float): The dropout rate for the encoder. Default to 
            :obj:`0.0`.
        aggregation (str): The aggregation function. Default to :obj:`mean`.
        aggregation_norm (Union[int, float]): The normalization factor for the 
            aggregation. Default to :obj:`100`.
        ffn_hidden (int): The number of hidden dimensions for the FFN. Default to 
            :obj:`300`.
        ffn_activation (str): The activation function for the FFN. Default to 
            :obj:`relu`.
        ffn_layers (int): The number of layers for the FFN. Default to :obj:`3`.
        ffn_dropout_p (float): The dropout rate for the FFN. Default to :obj:`0.0`.
        ffn_dropout_at_input_no_act (bool): Whether to apply dropout at the input 
            without activation. Default to :obj:`True`.
    """

    def __init__(
        self,
        mode: str = 'regression',
        n_classes: int = 3,
        n_tasks: int = 1,
        global_features_size: int = 0,
        atom_fdim: int = 133,
        bond_fdim: int = 14,
        hidden_dim: int = 300,
        num_layers: int = 3,
        bias: bool = False,
        enc_activation: str = 'relu',
        enc_dropout_p: float = 0.0,
        aggregation: str = 'mean',
        aggregation_norm: Union[int, float] = 100,
        ffn_hidden: int = 300,
        ffn_activation: str = 'relu',
        ffn_layers: int = 3,
        ffn_dropout_p: float = 0.0,
        ffn_dropout_at_input_no_act: bool = True,
        grid_size: int = 3,
    ):
        super().__init__()
        self.mode: str = mode
        self.n_classes: int = n_classes
        self.n_tasks: int = n_tasks

        # get encoder
        self.encoder: nn.Module = KAN_DMPNNEncoderLayer(
            atom_fdim=atom_fdim,
            bond_fdim=bond_fdim,
            d_hidden=hidden_dim,
            depth=num_layers,
            bias=bias,
            activation=enc_activation,
            dropout_p=enc_dropout_p,
            aggregation=aggregation,
            aggregation_norm=aggregation_norm,
            grid_size=grid_size,
        )

        # get input size for ffn
        ffn_input: int = hidden_dim + global_features_size

        # get output size for ffn
        if self.mode == 'regression':
            ffn_output: int = self.n_tasks
        elif self.mode == 'classification':
            ffn_output = self.n_tasks * self.n_classes

        # get ffn
        self.ffn: nn.Module = PositionwiseFeedForward(
            d_input=ffn_input,
            d_hidden=ffn_hidden,
            d_output=ffn_output,
            activation=ffn_activation,
            n_layers=ffn_layers,
            dropout_p=ffn_dropout_p,
            dropout_at_input_no_act=ffn_dropout_at_input_no_act)

    def forward(
            self,
            pyg_batch: Batch) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """Forward pass.

        Args:
            pyg_batch (Batch): The batch of data. It should be preprocessed by
                :class:`polymon.data.utils.DMPNNTransform`.

        Returns:
            Union[torch.Tensor, Sequence[torch.Tensor]]: The output of the model.
        """
        atom_features: torch.Tensor = pyg_batch['atom_features']
        f_ini_atoms_bonds: torch.Tensor = pyg_batch['f_ini_atoms_bonds']
        atom_to_incoming_bonds: torch.Tensor = pyg_batch[
            'atom_to_incoming_bonds']
        mapping: torch.Tensor = pyg_batch['mapping']
        global_features: torch.Tensor = pyg_batch['global_features']

        # Steps to get `molecules_unbatch_key`:
        # 1. Get the tensor containing the indices of first atoms of each molecule
        # 2. Get the tensor containing number of atoms of each molecule
        #     by taking the difference between consecutive indices.
        # 3. Convert the tensor to a list.
        molecules_unbatch_key: List = torch.diff(
            pyg_batch._slice_dict['atom_features']).tolist()

        # num_molecules x (enc_hidden + global_features_size)
        encodings: torch.Tensor = self.encoder(atom_features, f_ini_atoms_bonds,
                                               atom_to_incoming_bonds, mapping,
                                               global_features,
                                               molecules_unbatch_key)

        # ffn_output (`self.n_tasks` or `self.n_tasks * self.n_classes`)
        output: torch.Tensor = self.ffn(encodings)

        final_output: Union[torch.Tensor, Sequence[torch.Tensor]]

        if self.mode == 'regression':
            final_output = output
        elif self.mode == 'classification':
            if self.n_tasks == 1:
                output = output.view(-1, self.n_classes)
                final_output = nn.functional.softmax(output, dim=1), output
            else:
                output = output.view(-1, self.n_tasks, self.n_classes)
                final_output = nn.functional.softmax(output, dim=2), output

        return final_output


class KAN_DMPNNEncoderLayer(nn.Module):
    def __init__(self,
                 atom_fdim: int = 133,
                 bond_fdim: int = 14,
                 d_hidden: int = 300,
                 depth: int = 3,
                 bias: bool = False,
                 activation: str = 'relu',
                 dropout_p: float = 0.0,
                 aggregation: str = 'mean',
                 aggregation_norm: Union[int, float] = 100,
                 grid_size: int = 3,
                 ):
        super(KAN_DMPNNEncoderLayer, self).__init__()
        self.atom_fdim = atom_fdim
        self.concat_fdim = atom_fdim + bond_fdim

        self.depth: int = depth
        self.aggregation: str = aggregation
        self.aggregation_norm: Union[int, float] = aggregation_norm

        if activation == 'relu':
            self.activation: nn.modules.activation.Module = nn.ReLU()

        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.1)

        elif activation == 'prelu':
            self.activation = nn.PReLU()

        elif activation == 'tanh':
            self.activation = nn.Tanh()

        elif activation == 'selu':
            self.activation = nn.SELU()

        elif activation == 'elu':
            self.activation = nn.ELU()

        self.dropout: nn.modules.dropout.Module = nn.Dropout(dropout_p)

        # Input
        self.W_i: nn.Module = KANLinear(self.concat_fdim, d_hidden, add_bias=bias, grid_size=grid_size)

        # Shared weight matrix across depths (default):
        # For messages hidden states
        self.W_h: nn.Module = KANLinear(d_hidden, d_hidden, add_bias=bias, grid_size=grid_size)

        # For atom hidden states
        self.W_o: nn.Module = KANLinear(self.atom_fdim + d_hidden, d_hidden, grid_size=grid_size)

    def _get_updated_atoms_hidden_state(
            self, atom_features: torch.Tensor, h_message: torch.Tensor,
            atom_to_incoming_bonds: torch.Tensor) -> torch.Tensor:
        """
        Method to compute atom hidden states.

        Parameters
        ----------
        atom_features: torch.Tensor
            Tensor containing atoms features.
        h_message: torch.Tensor
            Tensor containing hidden states of messages.
        atom_to_incoming_bonds: torch.Tensor
            Tensor containing mapping from atom index to list of indicies of incoming bonds.

        Returns
        -------
        atoms_hidden_states: torch.Tensor
            Tensor containing atom hidden states.
        """
        messages_to_atoms: torch.Tensor = h_message[atom_to_incoming_bonds].sum(
            1)  # num_atoms x hidden_size
        atoms_hidden_states: torch.Tensor = self.W_o(
            torch.cat((atom_features, messages_to_atoms),
                      1))  # num_atoms x hidden_size
        atoms_hidden_states = self.activation(
            atoms_hidden_states)  # num_atoms x hidden_size
        atoms_hidden_states = self.dropout(
            atoms_hidden_states)  # num_atoms x hidden_size
        return atoms_hidden_states  # num_atoms x hidden_size

    def _readout(self, atoms_hidden_states: torch.Tensor,
                 molecules_unbatch_key: List) -> torch.Tensor:
        """
        Method to execute the readout phase. (compute molecules encodings from atom hidden states)

        Parameters
        ----------
        atoms_hidden_states: torch.Tensor
            Tensor containing atom hidden states.
        molecules_unbatch_key: List
            List containing number of atoms in various molecules of a batch

        Returns
        -------
        molecule_hidden_state: torch.Tensor
            Tensor containing molecule encodings.
        """
        mol_vecs: List = []
        atoms_hidden_states_split: Sequence[Tensor] = torch.split(
            atoms_hidden_states, molecules_unbatch_key)
        mol_vec: torch.Tensor
        for mol_vec in atoms_hidden_states_split:
            if self.aggregation == 'mean':
                mol_vec = mol_vec.sum(dim=0) / len(mol_vec)
            elif self.aggregation == 'sum':
                mol_vec = mol_vec.sum(dim=0)
            elif self.aggregation == 'norm':
                mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
            else:
                raise Exception("Invalid aggregation")
            mol_vecs.append(mol_vec)

        molecule_hidden_state: torch.Tensor = torch.stack(mol_vecs, dim=0)
        return molecule_hidden_state  # num_molecules x hidden_size

    def forward(self, atom_features: torch.Tensor,
                f_ini_atoms_bonds: torch.Tensor,
                atom_to_incoming_bonds: torch.Tensor, mapping: torch.Tensor,
                global_features: torch.Tensor,
                molecules_unbatch_key: List) -> torch.Tensor:
        """
        Output computation for the DMPNNEncoderLayer.

        Steps:

        - Get original bond hidden states from concatenation of initial atom and bond features. (``input``)
        - Get initial messages hidden states. (``message``)
        - Execute message passing step for ``self.depth - 1`` iterations.
        - Get atom hidden states using atom features and message hidden states.
        - Get molecule encodings.
        - Concatenate global molecular features and molecule encodings.

        Parameters
        ----------
        atom_features: torch.Tensor
            Tensor containing atoms features.
        f_ini_atoms_bonds: torch.Tensor
            Tensor containing concatenated feature vector which contains concatenation of initial atom and bond features.
        atom_to_incoming_bonds: torch.Tensor
            Tensor containing mapping from atom index to list of indicies of incoming bonds.
        mapping: torch.Tensor
            Tensor containing the mapping that maps bond index to 'array of indices of the bonds'
            incoming at the initial atom of the bond (excluding the reverse bonds).
        global_features: torch.Tensor
            Tensor containing molecule features.
        molecules_unbatch_key: List
            List containing number of atoms in various molecules of a batch

        Returns
        -------
        output: torch.Tensor
            Tensor containing the encodings of the molecules.
        """
        input: torch.Tensor = self.W_i(
            f_ini_atoms_bonds)  # num_bonds x hidden_size
        message: torch.Tensor = self.activation(
            input)  # num_bonds x hidden_size

        for _ in range(1, self.depth):
            message = message[mapping].sum(1)  # num_bonds x hidden_size
            h_message: torch.Tensor = input + self.W_h(
                message)  # num_bonds x hidden_size
            h_message = self.activation(h_message)  # num_bonds x hidden_size
            h_message = self.dropout(h_message)  # num_bonds x hidden_size

        # num_atoms x hidden_size
        atoms_hidden_states: torch.Tensor = self._get_updated_atoms_hidden_state(
            atom_features, h_message, atom_to_incoming_bonds)

        # num_molecules x hidden_size
        output: torch.Tensor = self._readout(atoms_hidden_states,
                                             molecules_unbatch_key)

        # concat global features
        if global_features.size()[0] != 0:
            if len(global_features.shape) == 1:
                global_features = global_features.view(len(output), -1)
            output = torch.cat([output, global_features], dim=1)

        return output  # num_molecules x hidden_size


class PositionwiseFeedForward(nn.Module):
    """PositionwiseFeedForward is a layer used to define the position-wise feed-forward (FFN) algorithm for the Molecular Attention Transformer [1]_

    Each layer in the MAT encoder contains a fully connected feed-forward network which applies two linear transformations and the given activation function.
    This is done in addition to the SublayerConnection module.

    Note: This modified version of `PositionwiseFeedForward` class contains `dropout_at_input_no_act` condition to facilitate its use in defining
        the feed-forward (FFN) algorithm for the Directed Message Passing Neural Network (D-MPNN) [2]_

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
    .. [2] Analyzing Learned Molecular Representations for Property Prediction https://arxiv.org/pdf/1904.01561.pdf

    Examples
    --------
    >>> from deepchem.models.torch_models.layers import PositionwiseFeedForward
    >>> feed_fwd_layer = PositionwiseFeedForward(d_input = 2, d_hidden = 2, d_output = 2, activation = 'relu', n_layers = 1, dropout_p = 0.1)
    >>> input_tensor = torch.tensor([[1., 2.], [5., 6.]])
    >>> output_tensor = feed_fwd_layer(input_tensor)
  """

    def __init__(self,
                 d_input: int = 1024,
                 d_hidden: int = 1024,
                 d_output: int = 1024,
                 activation: str = 'leakyrelu',
                 n_layers: int = 1,
                 dropout_p: float = 0.0,
                 dropout_at_input_no_act: bool = False):
        """Initialize a PositionwiseFeedForward layer.

        Parameters
        ----------
        d_input: int
            Size of input layer.
        d_hidden: int (same as d_input if d_output = 0)
            Size of hidden layer.
        d_output: int (same as d_input if d_output = 0)
            Size of output layer.
        activation: str
            Activation function to be used. Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
            'tanh' for TanH, 'selu' for SELU, 'elu' for ELU and 'linear' for linear activation.
        n_layers: int
            Number of layers.
        dropout_p: float
            Dropout probability.
        dropout_at_input_no_act: bool
            If true, dropout is applied on the input tensor. For single layer, it is not passed to an activation function.
        """
        super(PositionwiseFeedForward, self).__init__()

        self.dropout_at_input_no_act: bool = dropout_at_input_no_act

        if activation == 'relu':
            self.activation: Any = nn.ReLU()

        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.1)

        elif activation == 'prelu':
            self.activation = nn.PReLU()

        elif activation == 'tanh':
            self.activation = nn.Tanh()

        elif activation == 'selu':
            self.activation = nn.SELU()

        elif activation == 'elu':
            self.activation = nn.ELU()

        elif activation == "linear":
            self.activation = lambda x: x

        self.n_layers: int = n_layers
        d_output = d_output if d_output != 0 else d_input
        d_hidden = d_hidden if d_hidden != 0 else d_input

        if n_layers == 1:
            self.linears: Any = [nn.Linear(d_input, d_output)]

        else:
            self.linears = [nn.Linear(d_input, d_hidden)] + [
                nn.Linear(d_hidden, d_hidden) for _ in range(n_layers - 2)
            ] + [nn.Linear(d_hidden, d_output)]

        self.linears = nn.ModuleList(self.linears)
        dropout_layer = nn.Dropout(dropout_p)
        self.dropout_p = nn.ModuleList([dropout_layer for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output Computation for the PositionwiseFeedForward layer.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        """
        if not self.n_layers:
            return x

        if self.n_layers == 1:
            if self.dropout_at_input_no_act:
                return self.linears[0](self.dropout_p[0](x))
            else:
                return self.dropout_p[0](self.activation(self.linears[0](x)))

        else:
            if self.dropout_at_input_no_act:
                x = self.dropout_p[0](x)
            for i in range(self.n_layers - 1):
                x = self.dropout_p[i](self.activation(self.linears[i](x)))
            return self.linears[-1](x)