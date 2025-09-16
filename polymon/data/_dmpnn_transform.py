from typing import List, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from polymon.data.polymer import Polymer


class DMPNNTransform(BaseTransform):
    def __init__(self, max_num_bonds: int = 500):
        super().__init__()
        self.max_num_bonds = max_num_bonds

    def __call__(self, data: Polymer) -> '_ModData':
        mapper = _MapperDMPNN(data)
        y = getattr(data, 'y', torch.tensor([], device=data.x.device))
        x = data['x']
        edge_index = data['edge_index']
        edge_attr = data['edge_attr']
        smiles = data['smiles']
        atom_features, f_ini_atoms_bonds, atom_to_incoming_bonds, mapping, global_features = mapper.values

        data = _ModData(required_inc=len(f_ini_atoms_bonds),
                        atom_features=atom_features.float(),
                        f_ini_atoms_bonds=f_ini_atoms_bonds.float(),
                        atom_to_incoming_bonds=atom_to_incoming_bonds,
                        mapping=mapping,
                        global_features=global_features)
        required_padding: int = self.max_num_bonds - data['atom_to_incoming_bonds'].shape[1]
        data['atom_to_incoming_bonds'] = nn.functional.pad(
                        data['atom_to_incoming_bonds'],
                        (0, required_padding, 0, 0),
                        mode='constant',
                        value=-1)
        data['mapping'] = nn.functional.pad(
                        data['mapping'], (0, required_padding, 0, 0),
                        mode='constant',
                        value=-1)
        data['y'] = y
        data['x'] = x
        data['edge_index'] = edge_index
        data['edge_attr'] = edge_attr
        data['smiles'] = smiles
        return data


class _ModData(Data):
    """Modified version of Data class of pytorch-geometric to enable batching process to
    custom increment values in certain keys.
    """

    def __init__(self, required_inc, *args, **kwargs):
        """Initialize the _ModData class"""
        super().__init__(*args, **kwargs)
        self.required_inc = required_inc  # required increment

    def __inc__(self, key, value, *args, **kwargs):
        """
        Modified __inc__() to increment 'atom_to_incoming_bonds' and 'mapping' keys
        based given required increment value (example, number of bonds in the molecule)
        """
        if key in ['atom_to_incoming_bonds', 'mapping']:
            return self.required_inc
        else:
            return super().__inc__(key, value, *args, **kwargs)


class _MapperDMPNN:
    def __init__(self, graph: Polymer):
        """
        Parameters
        ----------
        graph: GraphData
            GraphData object.
        """
        self.num_atoms: int = graph.num_nodes
        self.num_atom_features: int = graph.num_node_features
        self.num_bonds: int = graph.num_edges
        self.num_bond_features: int = graph.num_edge_features
        self.atom_features: Tensor = graph.x
        self.bond_features: Tensor = graph.edge_attr
        self.bond_index: Tensor = graph.edge_index
        self.global_features: Tensor = getattr(graph, 'descriptors', torch.tensor([], device=graph.x.device))  # type: ignore
        # mypy check is ignored for global_features as it is not a default attribute
        # of GraphData. It is created during runtime using **kwargs.

        # mapping from bond index to the index of the atom (where the bond is coming from)
        self.bond_to_ini_atom: np.ndarray

        # mapping from bond index to concat(in_atom, bond) features
        self.f_ini_atoms_bonds: np.ndarray = np.empty(0)

        # mapping from atom index to list of indicies of incoming bonds
        self.atom_to_incoming_bonds: np.ndarray

        # mapping which maps bond index to 'array of indices of the bonds' incoming at the initial atom of the bond (excluding the reverse bonds)
        self.mapping: np.ndarray = np.empty(0)

        if self.num_bonds == 0:
            self.bond_to_ini_atom = np.empty(0)
            self.f_ini_atoms_bonds = np.zeros(
                (1, self.num_atom_features + self.num_bond_features))

            self.atom_to_incoming_bonds = np.asarray([[-1]] * self.num_atoms,
                                                     dtype=int)
            self.mapping = np.asarray([[-1]], dtype=int)

        else:
            self.bond_to_ini_atom = self.bond_index[0]
            self._get_f_ini_atoms_bonds()  # its zero padded at the end

            self.atom_to_incoming_bonds = self._get_atom_to_incoming_bonds()
            self._generate_mapping()  # its padded with -1 at the end
        self.device = self.atom_features.device

    @property
    def values(self) -> Sequence[np.ndarray]:
        """
        Returns the required mappings:
        - atom features
        - concat features (atom + bond)
        - atom to incoming bonds mapping
        - mapping
        - global features
        """
        f_ini_atoms_bonds = torch.from_numpy(self.f_ini_atoms_bonds).to(self.device)
        atom_to_incoming_bonds = torch.from_numpy(self.atom_to_incoming_bonds).to(self.device)
        mapping = torch.from_numpy(self.mapping).to(self.device)
        global_features = self.global_features
        return self.atom_features, f_ini_atoms_bonds, atom_to_incoming_bonds, mapping, global_features

    def _get_f_ini_atoms_bonds(self):
        """Method to get `self.f_ini_atoms_bonds`"""
        self.f_ini_atoms_bonds = np.hstack(
            (self.atom_features[self.bond_to_ini_atom], self.bond_features))

        # zero padded at the end
        self.f_ini_atoms_bonds = np.pad(self.f_ini_atoms_bonds,
                                        ((0, 1), (0, 0)))

    def _generate_mapping(self):
        """
        Generate mapping, which maps bond index to 'array of indices of the bonds'
        incoming at the initial atom of the bond (reverse bonds are not considered).

        Steps:
        - Get mapping based on `self.atom_to_incoming_bonds` and `self.bond_to_ini_atom`.
        - Replace reverse bond indices with -1.
        - Pad the mapping with -1.
        """

        # get mapping which maps bond index to 'array of indices of the bonds' incoming at the initial atom of the bond
        self.mapping = self.atom_to_incoming_bonds[self.bond_to_ini_atom]
        self._replace_rev_bonds()

        # padded with -1 at the end
        self.mapping = np.pad(self.mapping, ((0, 1), (0, 0)),
                              constant_values=-1)

    def _get_atom_to_incoming_bonds(self) -> np.ndarray:
        """Method to get atom_to_incoming_bonds mapping"""
        # mapping from bond index to the index of the atom (where the bond if going to)
        bond_to_final_atom: np.ndarray = self.bond_index[1]

        # mapping from atom index to list of indicies of incoming bonds
        a2b: List = []
        for i in range(self.num_atoms):
            a2b.append(list(np.where(bond_to_final_atom == i)[0]))

        # get maximum number of incoming bonds
        max_num_bonds: int = max(
            1, max(len(incoming_bonds) for incoming_bonds in a2b))

        # Make number of incoming bonds equal to maximum number of bonds.
        # This is done by appending -1 to fill remaining space at each atom indices.
        a2b = [
            a2b[a] + [-1] * (max_num_bonds - len(a2b[a]))
            for a in range(self.num_atoms)
        ]

        return np.asarray(a2b, dtype=int)

    def _replace_rev_bonds(self):
        """Method to get b2revb and replace the reverse bond indices with -1 in mapping."""
        # mapping from bond index to the index of the reverse bond
        b2revb = np.empty(self.num_bonds, dtype=int)
        for i in range(self.num_bonds):
            if i % 2 == 0:
                b2revb[i] = i + 1
            else:
                b2revb[i] = i - 1

        for count, i in enumerate(b2revb):
            self.mapping[count][np.where(self.mapping[count] == i)] = -1