from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch
from polymon.setting import MAX_SEQ_LEN, SMILES_VOCAB
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors as RDKitDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.ML.Descriptors.MoleculeDescriptors import \
    MolecularDescriptorCalculator
from scipy.sparse import coo_matrix

from rdkit.Chem import AllChem, Descriptors3D

from polymon.setting import MAX_SEQ_LEN, SMILES_VOCAB


FEATURIZER_REGISTRY: Dict[str, 'Featurizer'] = {}

def register_cls(name: str):
    def decorator(cls):
        FEATURIZER_REGISTRY[name] = cls
        return cls
    return decorator


class Featurizer(ABC):
    _avail_features: List[str] = []
    
    def __init__(self, feature_names: List[str] = None):
        self.feature_names = feature_names
        if feature_names is not None:
            self._check_features(feature_names)
    
    def _check_features(self, feature_names: List[str]):
        for feature_name in feature_names:
            assert feature_name in self._avail_features, \
                f'{feature_name} is not available in {self.__class__.__name__}'
    
    @abstractmethod
    def __call__(self, rdmol: Chem.Mol) -> Dict[str, torch.Tensor]:
        pass


########################################################
############# Add new featurizers below ################
########################################################

@register_cls('x')
class AtomFeaturizer(Featurizer):
    """Featurize atoms in a molecule. Default features include one-hot encoding
    of atomic numbers.
    """
    _avail_features: List[str] = ['degree', 'is_aromatic']
    def __init__(
        self,
        feature_names: List[str] = None,
        unique_atom_nums: List[int] = None,
    ):
        super().__init__(feature_names)
        self.unique_atom_nums = unique_atom_nums
    
    def __call__(
        self,
        rdmol: Chem.Mol,
    ) -> Dict[str, torch.Tensor]:
        atom_num = self.get_atom_num(rdmol, self.unique_atom_nums)
        if self.feature_names is None:
            return {'x': atom_num}
        
        x = []
        for atom in rdmol.GetAtoms():
            atom_features = []
            for feature_name in self.feature_names:
                atom_features.append(getattr(self, feature_name)(atom))
            x.append(torch.cat(atom_features))
        feature_exclude_atom_num = torch.stack(x, dim=0)
        return {'x': torch.cat([atom_num, feature_exclude_atom_num], dim=1).float()}
    
    @staticmethod
    def get_atom_num(
        rdmol: Chem.Mol,
        unique_atom_nums: List[int],
    ) -> torch.Tensor:
        atom_nums = [atom.GetAtomicNum() for atom in rdmol.GetAtoms()]
        indices = []
        for num in atom_nums:
            try:
                indices.append(unique_atom_nums.index(num))
            except ValueError:
                # If atomic number not found, use the last column (unknown atoms)
                indices.append(len(unique_atom_nums))
        
        atom_nums, indices = torch.tensor(atom_nums), torch.tensor(indices)
        # Add one extra column for unknown atomic numbers
        onehot = torch.zeros(len(atom_nums), len(unique_atom_nums) + 1).to(int)
        onehot.scatter_(1, indices.unsqueeze(1), 1)
        return onehot
    
    def degree(self, atom: Chem.Atom) -> torch.Tensor:
        degree_choice = [0, 1, 2, 3, 4]
        onehot = torch.zeros(len(degree_choice) + 1)
        try:
            onehot[degree_choice.index(atom.GetTotalDegree())] = 1
        except ValueError:
            # If degree not found, use the last column (unknown degrees)
            onehot[-1] = 1
        return onehot
    
    def is_aromatic(self, atom: Chem.Atom) -> torch.Tensor:
        return torch.tensor([int(atom.GetIsAromatic())])


@register_cls('edge')
class BondFeaturizer(Featurizer):
    _avail_features: List[str] = ['fully_connected_edges', 'bond']
    
    def __call__(self, rdmol: Chem.Mol) -> Dict[str, torch.Tensor]:
        if self.feature_names is None:
            return {}
        
        assert len(self.feature_names) == 1, \
            'Only one feature name is supported for bond featurizer'
        feature_name = self.feature_names[0]
        return getattr(self, feature_name)(rdmol)
    
    def fully_connected_edges(self, rdmol: Chem.Mol) -> Dict[str, torch.Tensor]:
        n_nodes = rdmol.GetNumAtoms()
        edges = torch.combinations(torch.arange(n_nodes), r=2).T
        reversed_edges = torch.stack([edges[1], edges[0]])
        return {'edge_index': torch.cat([edges, reversed_edges], dim=1)}
    
    def bond(self, rdmol: Chem.Mol) -> Dict[str, torch.Tensor]:
        adj = Chem.GetAdjacencyMatrix(rdmol)
        coo_adj = coo_matrix(adj)
        edge_index = torch.from_numpy(np.vstack([coo_adj.row, 
                                                 coo_adj.col])).long()
        edge_attr = []
        for i, j in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            bond = rdmol.GetBondBetweenAtoms(i, j)
            bond_type = bond.GetBondType()
            bond_type_one_hot_encoding = [
                int(bond_type == Chem.rdchem.BondType.SINGLE),
                int(bond_type == Chem.rdchem.BondType.DOUBLE),
                int(bond_type == Chem.rdchem.BondType.TRIPLE),
                int(bond_type == Chem.rdchem.BondType.AROMATIC)
            ]
            edge_attr.append(torch.tensor(bond_type_one_hot_encoding))
        edge_attr = torch.stack(edge_attr, dim=0)
        return {'edge_index': edge_index, 'edge_attr': edge_attr}


@register_cls('pos')
class PosFeaturizer(Featurizer):
    def __call__(self, rdmol: Chem.Mol) -> Dict[str, torch.Tensor]:
        if rdmol.GetNumConformers() == 0:
            # If no conformer, use RDKit to generate one
            AllChem.EmbedMolecule(rdmol)
            AllChem.MMFFOptimizeMolecule(rdmol)
            pos = torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
            pos -= pos.mean(dim=0)
            return {'pos': pos}

        pos = torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
        pos -= pos.mean(dim=0)
        return {'pos': pos}


@register_cls('z')
class AtomNumFeaturizer(Featurizer):
    def __call__(
        self,
        rdmol: Chem.Mol,
    ) -> Dict[str, torch.Tensor]:
        atom_nums = [atom.GetAtomicNum() for atom in rdmol.GetAtoms()]
        return {'z': torch.tensor(atom_nums)}


@register_cls('seq')
class SeqFeaturizer(Featurizer):
    # Double tokens
    DOUBLE_TOKEN_DICT = {
        'Br': 'R',
        'Cl': 'G',
        'Si': 'X',
    }

    # Special tokens
    SOS = '$'
    EOS = '!'
    PAD = ' '
    
    def __call__(
        self,
        rdmol: Chem.Mol,
    ) -> Dict[str, torch.Tensor]:
        smiles = Chem.MolToSmiles(rdmol)
        for k, v in self.DOUBLE_TOKEN_DICT.items():
            smiles = smiles.replace(k, v)
        
        # Add special tokens (start, end, pad)
        if len(smiles) > MAX_SEQ_LEN - 2:
            smiles = smiles[:MAX_SEQ_LEN - 2]
        revised_smiles = f"{self.SOS}{smiles}{self.EOS}"
        revised_smiles = revised_smiles.ljust(MAX_SEQ_LEN, self.PAD)
        
        # Encode SMILES
        seq = list(map(lambda x: SMILES_VOCAB.index(x), revised_smiles))
        seq = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
        seq_len = torch.tensor(len(smiles) + 2, dtype=torch.long)
        return {'seq': seq, 'seq_len': seq_len}


@register_cls('desc')
class DescFeaturizer(Featurizer):
    """Featurize descriptors of a molecule. Features should be [1, num_features]
    """
    _avail_features: List[str] = ['rdkit2d', 'ecfp4', 'rdkit3d', 'mordred']

    def __init__(
        self,
        feature_names: List[str] = None,
    ):
        super().__init__(feature_names)
    
    def __call__(
        self,
        rdmol: Chem.Mol,
    ) -> Dict[str, torch.Tensor]:
        descriptors = []
        for feature_name in self.feature_names:
            descriptors.append(getattr(self, feature_name)(rdmol))
        return {'descriptors': torch.concatenate(descriptors, dim=1)}

    def rdkit2d(
        self,
        rdmol: Chem.Mol,
    ) -> torch.Tensor:
        unrobust_indices = [10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25]
        desc_names = [
            x[0] for i, x in enumerate(RDKitDescriptors._descList) \
                if i not in unrobust_indices
        ]
        desc_calculator = MolecularDescriptorCalculator(desc_names)
        descs = desc_calculator.CalcDescriptors(rdmol)
        descs = torch.tensor(list(descs), dtype=torch.float).unsqueeze(0)
        return descs
    
    def ecfp4(
        self,
        rdmol: Chem.Mol,
    ) -> torch.Tensor:
        ecfp4 = GetMorganFingerprintAsBitVect(rdmol, 4, nBits=2048)
        ecfp4 = torch.tensor(list(ecfp4), dtype=torch.float).unsqueeze(0)
        return ecfp4
    
    def mordred(
        self,
        rdmol: Chem.Mol,
    ) -> torch.Tensor:
        '''
        pip install mordred
        '''
        from mordred import Calculator, descriptors
        calc = Calculator(descriptors, ignore_3D=True)
        descs = torch.tensor(calc(rdmol)[2:], dtype=torch.float).unsqueeze(0)
        
    def rdkit3d(
        self,
        rdmol: Chem.Mol,
    ) -> torch.Tensor:
        smiles = Chem.MolToSmiles(rdmol)
        if rdmol.GetNumConformers() == 0:
            rdmol = Chem.AddHs(rdmol)
            for atom in rdmol.GetAtoms():
                nbrs = atom.GetNeighbors()
                if len(nbrs) == 0 or atom.GetAtomicNum() != 0:
                    continue
                bond = rdmol.GetBondBetweenAtoms(atom.GetIdx(), nbrs[0].GetIdx())
                if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    atom.SetAtomicNum(1)
                elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    atom.SetAtomicNum(8)
            try:
                ps = AllChem.ETKDGv3()
                ps.randomSeed = 42
                AllChem.EmbedMolecule(rdmol, ps)
            except Exception as e:
                return torch.full((1, len(Descriptors3D.descList)), float('inf'))

        if rdmol.GetNumConformers() == 0:
            return torch.full((1, len(Descriptors3D.descList)), float('inf'))
    
        desc_dict = Descriptors3D.CalcMolDescriptors3D(rdmol)
        descs = list(desc_dict.values())
        descs = torch.tensor(descs, dtype=torch.float).unsqueeze(0)

        return descs


########################################################
############# End of new featurizers ###################
########################################################

AVAIL_FEATURES = set(FEATURIZER_REGISTRY.keys())
for key, cls in FEATURIZER_REGISTRY.items():
    AVAIL_FEATURES.update(cls._avail_features)


class ComposeFeaturizer:
    def __init__(self, names: List[str], config: dict = None):
        invalid_names = set(names) - set(AVAIL_FEATURES)
        if invalid_names:
            raise ValueError(f'Invalid feature names: {invalid_names}')
        
        if config is None:
            config = {}
        
        featurizers = []
        for key, cls in FEATURIZER_REGISTRY.items():
            feature_names = []
            for name in names:
                if name == key:
                    featurizers.append(cls(**config.get(key, {})))
                elif name in cls._avail_features:
                    feature_names.append(name)
            if feature_names:
                featurizers.append(cls(feature_names, **config.get(key, {})))
        self.featurizers = featurizers
    
    def __call__(self, rdmol: Chem.Mol) -> Dict[str, torch.Tensor]:
        mol_dict = {}
        for featurizer in self.featurizers:
            mol_dict.update(featurizer(rdmol))
        return mol_dict