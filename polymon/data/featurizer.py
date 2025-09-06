import hashlib
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors as RDKitDescriptors
from rdkit.Chem import Descriptors3D, MACCSkeys, rdChemReactions
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.ML.Descriptors.MoleculeDescriptors import \
    MolecularDescriptorCalculator
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_undirected

from polymon.data.polymer import OligomerBuilder
from polymon.setting import (CGCNN_ELEMENT_INFO, DEFAULT_ATOM_FEATURES,
                             GEOMETRY_VOCAB, MAX_SEQ_LEN, MORDRED_UNSTABLE_IDS,
                             SMILES_VOCAB)

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
    _avail_features: List[str] = [
        'degree', 
        'is_aromatic', 
        'chiral_tag', 
        'num_hydrogens', 
        'hybridization', 
        'mass', 
        'formal_charge', 
        'is_attachment',
        # 'xenonpy_atom',
        'cgcnn',
        'source',
    ]
    def __init__(
        self,
        feature_names: List[str] = None,
        unique_atom_nums: List[int] = None,
        unique_sources: List[str] = None,
    ):
        super().__init__(feature_names)
        self.unique_atom_nums = unique_atom_nums
        if self.feature_names is None:
            self.feature_names = DEFAULT_ATOM_FEATURES
        self.unique_sources = unique_sources
    
    def __call__(
        self,
        rdmol: Chem.Mol,
    ) -> Dict[str, torch.Tensor]:
        atom_num = self.get_atom_num(rdmol, self.unique_atom_nums)
        feature_names = deepcopy(self.feature_names)
        if 'source' in feature_names:
            feature_names.remove('source')
        
        x = []
        for atom in rdmol.GetAtoms():
            atom_features = []
            for feature_name in feature_names:
                atom_features.append(getattr(self, feature_name)(atom, rdmol))
            x.append(torch.cat(atom_features))
        feature_exclude_atom_num = torch.stack(x, dim=0)
        
        if 'source' in self.feature_names:
            source_feature = torch.zeros(len(self.unique_sources))
            source_feature[self.unique_sources.index(rdmol.GetProp('Source'))] = 1
            feature_exclude_atom_num = torch.cat([
                feature_exclude_atom_num, 
                source_feature.repeat(feature_exclude_atom_num.shape[0], 1)
            ], dim=1)
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
    
    def degree(self, atom: Chem.Atom, rdmol: Chem.Mol=None) -> torch.Tensor:
        degree_choice = [0, 1, 2, 3, 4]
        onehot = torch.zeros(len(degree_choice) + 1)
        try:
            onehot[degree_choice.index(atom.GetTotalDegree())] = 1
        except ValueError:
            # If degree not found, use the last column (unknown degrees)
            onehot[-1] = 1
        return onehot
    
    def is_aromatic(self, atom: Chem.Atom, rdmol: Chem.Mol=None) -> torch.Tensor:
        return torch.tensor([int(atom.GetIsAromatic())])
    
    def chiral_tag(self, atom: Chem.Atom, rdmol: Chem.Mol=None) -> torch.Tensor:
        chiral_tag_choice = list(range(len(Chem.ChiralType.names)-1))
        chiral_tag = atom.GetChiralTag()
        onehot = torch.zeros(len(chiral_tag_choice) + 1)
        try:
            onehot[chiral_tag_choice.index(chiral_tag)] = 1
        except ValueError:
            onehot[-1] = 1
        return onehot
    
    def num_hydrogens(self, atom: Chem.Atom, rdmol: Chem.Mol=None) -> torch.Tensor:
        num_hydrogen_choice = list(range(len(Chem.rdchem.ChiralType.names)-1))
        num_hydrogen = atom.GetNumImplicitHs()
        onehot = torch.zeros(len(num_hydrogen_choice) + 1)
        try:
            onehot[num_hydrogen_choice.index(num_hydrogen)] = 1
        except ValueError:
            onehot[-1] = 1
        return onehot
    
    def hybridization(self, atom: Chem.Atom, rdmol: Chem.Mol=None) -> torch.Tensor:
        hybridization_choices = list(range(len(Chem.HybridizationType.names)-1))
        hybridization = int(atom.GetHybridization())
        onehot = torch.zeros(len(hybridization_choices) + 1)
        try:
            onehot[hybridization_choices.index(hybridization)] = 1
        except ValueError:
            onehot[-1] = 1
        return onehot
    
    def mass(self, atom: Chem.Atom, rdmol: Chem.Mol=None) -> torch.Tensor:
        return torch.tensor([atom.GetMass() / 100])
    
    def formal_charge(self, atom: Chem.Atom, rdmol: Chem.Mol=None) -> torch.Tensor:
        return torch.tensor([atom.GetFormalCharge() / 10])
    
    def is_attachment(self, atom: Chem.Atom, rdmol: Chem.Mol=None) -> torch.Tensor:
        return torch.tensor([int(atom.GetAtomicNum() == 0)])
    
    def xenonpy_atom(self, atom: Chem.Atom, rdmol: Chem.Mol=None) -> torch.Tensor:
        from polymon.setting import XENONPY_ELEMENTS_INFO

        # preset.sync('elements_completed')
        symbol = Chem.GetPeriodicTable().GetElementSymbol(atom.GetAtomicNum())
        return torch.tensor(XENONPY_ELEMENTS_INFO.loc[symbol].values)
    
    def cgcnn(self, atom: Chem.Atom, rdmol: Chem.Mol=None) -> torch.Tensor:
        atom_num = atom.GetAtomicNum()
        CGCNN_ELEMENT_INFO['0'] = [0] * len(CGCNN_ELEMENT_INFO['1'])
        return torch.tensor(CGCNN_ELEMENT_INFO[str(atom_num)])


@register_cls('edge')
class BondFeaturizer(Featurizer):
    _avail_features: List[str] = [
        'fully_connected_edges', 
        'bond', 
        'periodic_bond',
        'virtual_bond',
    ]
    
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
            # bond_stereo = bond.GetStereo()
            # bond_stereo_one_hot_encoding = [
            #     int(bond_stereo == Chem.rdchem.BondStereo.STEREOCIS),
            #     int(bond_stereo == Chem.rdchem.BondStereo.STEREOTRANS),
            #     int(bond_stereo == Chem.rdchem.BondStereo.STEREOANY),
            #     int(bond_stereo == Chem.rdchem.BondStereo.STEREONONE),
            # ]
            # bond_is_in_ring = [int(bond.IsInRing())]
            # bond_is_conjugated = [int(bond.GetIsConjugated())]
            
            # attr = torch.cat([
            #     torch.tensor(bond_type_one_hot_encoding),
            #     torch.tensor(bond_stereo_one_hot_encoding),
            #     torch.tensor(bond_is_in_ring),
            #     torch.tensor(bond_is_conjugated),
            # ], dim=0)
            # edge_attr.append(attr)
            edge_attr.append(torch.tensor(bond_type_one_hot_encoding))
            
        edge_attr = torch.stack(edge_attr, dim=0)
        return {'edge_index': edge_index, 'edge_attr': edge_attr}
    
    def periodic_bond(self, rdmol: Chem.Mol) -> Dict[str, torch.Tensor]:
        bond_info = self.bond(rdmol)
        bond_index = bond_info['edge_index']
        bond_attr = bond_info['edge_attr']
        
        # Add bonds between attachment points
        # attachments = [
        #     atom.GetIdx() for atom in rdmol.GetAtoms() \
        #         if atom.GetAtomicNum() == 0
        # ]
        attachments = []
        for atom in rdmol.GetAtoms():
            if atom.GetPropsAsDict().get('attachment', 'False') == 'True':
                attachments.append(atom.GetIdx())
        
        bond_attr = torch.cat([bond_attr, torch.zeros(bond_attr.shape[0], 1)], dim=1)
        for i in range(len(attachments)):
            for j in range(i + 1, len(attachments)):
                attach_bond_index = torch.tensor([
                    [attachments[i], attachments[j]],
                    [attachments[j], attachments[i]],
                ]).T
                attach_bond_attr = torch.tensor([0] * bond_attr.shape[1])
                attach_bond_attr[-1] = 1
                attach_bond_attr[0] = 1
                attach_bond_attr = attach_bond_attr.unsqueeze(0)
                attach_bond_attr = torch.cat([attach_bond_attr, attach_bond_attr], dim=0)
                bond_index = torch.cat([bond_index, attach_bond_index], dim=1)
                bond_attr = torch.cat([bond_attr, attach_bond_attr], dim=0)
        return {'edge_index': bond_index, 'edge_attr': bond_attr}
    
    def virtual_bond(self, rdmol: Chem.Mol) -> Dict[str, torch.Tensor]:
        """Add virtual bonds between virtual node and each other node"""
        n_nodes = rdmol.GetNumAtoms()
        bonds = self.bond(rdmol)
        bond_index = bonds['edge_index']
        bond_attr = bonds['edge_attr']
        n_bonds = bond_index.shape[1]
        n_bond_features = bond_attr.shape[1]
        
        # Add bond index between virtual node and each other node
        virtual_bond_index = torch.stack([
            torch.arange(n_nodes),
            n_nodes * torch.ones(n_nodes, dtype=torch.long),
        ])
        
        # Add additional bond attributes for virtual bonds and bonds
        virtual_bond_attr = torch.zeros(virtual_bond_index.shape[1], n_bond_features + 1)
        virtual_bond_attr[:, -1] = 1
        bond_attr = torch.cat([bond_attr, torch.zeros(n_bonds, 1)], dim=1)
        
        # Add another direction of virtual bond
        virtual_bond_index, virtual_bond_attr = to_undirected(
            virtual_bond_index, virtual_bond_attr
        )
        
        bond_attr = torch.cat([bond_attr, virtual_bond_attr], dim=0)
        bond_index = torch.cat([bond_index, virtual_bond_index], dim=1)
        
        return {'edge_index': bond_index, 'edge_attr': bond_attr}


@register_cls('bridge')
class BridgeFeaturizer(Featurizer):
    def __call__(self, rdmol: Chem.Mol) -> Dict[str, torch.Tensor]:
        attachments = [
            atom.GetIdx() for atom in rdmol.GetAtoms() \
                if atom.GetAtomicNum() == 0
        ]
        if len(attachments) != 2:
            return {'bridge_index': None}
        return {'bridge_index': torch.tensor([[attachments[0]], [attachments[1]]])}

@register_cls('pos')
class PosFeaturizer(Featurizer):
    def __call__(self, rdmol: Chem.Mol) -> Dict[str, torch.Tensor]:
        if rdmol.GetNumConformers() == 0:
            rdmol = self.get_embeded_rdmol(rdmol)
            if rdmol is None:
                return {'pos': None}

        pos = torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
        pos -= pos.mean(dim=0)
        return {'pos': pos}
    
    def get_embeded_rdmol(self, rdmol: Chem.Mol, sanitize: bool = False) -> Optional[Chem.Mol]:
        rdmol = deepcopy(rdmol)
        # If no conformer, load from geometry_vocab.sdf or generate one
        smiles = Chem.MolToSmiles(rdmol)
        os.makedirs(str(GEOMETRY_VOCAB), exist_ok=True)
        hash_digest = hashlib.sha256(smiles.encode('utf-8')).hexdigest()
        geometry_file = GEOMETRY_VOCAB / f'{hash_digest}.sdf'
        if not geometry_file.exists():
            rdmol.SetProp('smiles', smiles)
            rdmol = self.polymer2monomer(rdmol)
            if rdmol is None:
                return None
            rdmol = self.init_geometry(rdmol)
            if rdmol is None:
                return None
            sdf_writer = Chem.SDWriter(str(geometry_file))
            sdf_writer.write(rdmol)
            sdf_writer.close()
        else:
            rdmol = Chem.MolFromMolFile(
                str(geometry_file), sanitize=sanitize, removeHs=False
            )
        if rdmol.GetNumConformers() == 0:
            return None
        return rdmol
    
    @staticmethod
    def init_geometry(mol: Chem.Mol) -> Chem.Mol:
        try:
            ps = AllChem.ETKDGv3()
            ps.randomSeed = 42
            AllChem.EmbedMolecule(mol, ps)
            if mol.GetNumConformers() > 0:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
            
            return mol
        except Exception as e:
            print(f"Error initializing geometry for {Chem.MolToSmiles(mol)}: {e}")
            return None
    
    @staticmethod
    def polymer2monomer(rdmol: Chem.Mol) -> Chem.Mol:
        attachments = [atom for atom in rdmol.GetAtoms() if atom.GetSymbol() == '*']
        if len(attachments) != 2:
            print(f'Number of attachments is not 2.')
            return None
        rdmol.SetIntProp('attachment1', attachments[0].GetIdx())
        rdmol.SetIntProp('attachment2', attachments[1].GetIdx())
        
        # Get the neighbors of the attachments
        attachment1, attachment2 = attachments
        attachment1_nbrs = attachment1.GetNeighbors()
        attachment2_nbrs = attachment2.GetNeighbors()
        if len(attachment1_nbrs) != 1 or len(attachment2_nbrs) != 1:
            print(f'Attachment has more than one neighbor.')
            return None
        
        # Set the atomic numbers of the attachments based on neighbors
        attachment1.SetAtomicNum(attachment2_nbrs[0].GetAtomicNum())
        attachment2.SetAtomicNum(attachment1_nbrs[0].GetAtomicNum())
        if '*' in Chem.MolToSmiles(rdmol):
            print(f'Attachment is not removed for {rdmol.GetProp("smiles")}')
        
        return rdmol


@register_cls('z')
class AtomNumFeaturizer(Featurizer):
    def __call__(
        self,
        rdmol: Chem.Mol,
    ) -> Dict[str, torch.Tensor]:
        atom_nums = [atom.GetAtomicNum() for atom in rdmol.GetAtoms()]
        return {'z': torch.tensor(atom_nums)}


@register_cls('relative_position')
class RelativePositionFeaturizer(Featurizer):
    def __call__(
        self,
        rdmol: Chem.Mol,
    ) -> Dict[str, torch.Tensor]:
        attachments = [atom for atom in rdmol.GetAtoms() if atom.GetSymbol() == '*']
        # Find the length of the shortest path between atoms and their closest attachment
        pe = []
        for atom in rdmol.GetAtoms():
            if atom.GetSymbol() == '*':
                pe.append(0)
            else:
                min_dist = float('inf')
                for attachment in attachments:
                    shortest_path = Chem.GetShortestPath(rdmol, atom.GetIdx(), attachment.GetIdx())
                    dist = len(shortest_path) - 1
                    if dist < min_dist:
                        min_dist = dist
                pe.append(min_dist)
        return {'relative_position': torch.LongTensor(pe)}


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
    
    _avail_features: List[str] = [
        'rdkit2d', 
        'ecfp4', 
        'rdkit3d', 
        'mordred', 
        'maccs', 
        'oligomer_rdkit2d', 
        'oligomer_mordred',
        'oligomer_ecfp4',
        'xenonpy_desc',
        'mordred3d',
    ]

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
        unrobust_indices = [10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 42]
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
        descs = calc(rdmol)
        descs = [x for i, x in enumerate(descs) if i not in MORDRED_UNSTABLE_IDS]
        descs = torch.tensor(descs, dtype=torch.float).unsqueeze(0)
        return descs
    
    def maccs(
        self,
        rdmol: Chem.Mol,
    ) -> torch.Tensor:
        maccs = list(MACCSkeys.GenMACCSKeys(rdmol))
        maccs = torch.tensor(maccs, dtype=torch.float).unsqueeze(0)
        return maccs
        
    def rdkit3d(
        self,
        rdmol: Chem.Mol,
    ) -> torch.Tensor:
        if rdmol.GetNumConformers() == 0:
            rdmol = PosFeaturizer().get_embeded_rdmol(rdmol)

        if rdmol is None:
            return torch.full((1, len(Descriptors3D.descList)), float('inf'))
    
        desc_dict = Descriptors3D.CalcMolDescriptors3D(rdmol)
        descs = list(desc_dict.values())
        descs = torch.tensor(descs, dtype=torch.float).unsqueeze(0)
        return descs
    
    def mordred3d(
        self,
        rdmol: Chem.Mol,
    ) -> torch.Tensor:
        if rdmol.GetNumConformers() == 0:
            rdmol = PosFeaturizer().get_embeded_rdmol(rdmol, sanitize=True)

        from mordred import Calculator, descriptors
        calc = Calculator(descriptors, ignore_3D=False)
        
        if rdmol is None:
            return torch.full((1, len(calc.descriptors)), float('inf'))
        descs = calc(rdmol)
        descs = torch.tensor(descs, dtype=torch.float).unsqueeze(0)
        return descs
    
    def oligomer_rdkit2d(
        self,
        rdmol: Chem.Mol,
    ) -> torch.Tensor:
        rdmol_smiles = Chem.MolToSmiles(rdmol)
        oligomer = OligomerBuilder.get_oligomer(rdmol_smiles, 2)
        return self.rdkit2d(oligomer)
    
    def oligomer_mordred(
        self,
        rdmol: Chem.Mol,
    ) -> torch.Tensor:
        rdmol_smiles = Chem.MolToSmiles(rdmol)
        oligomer = OligomerBuilder.get_oligomer(rdmol_smiles, 2)
        return self.mordred(oligomer)
    
    def xenonpy_desc(
        self,
        rdmol: Chem.Mol,
    ) -> torch.Tensor:
        from collections import Counter

        from xenonpy.datatools import preset
        from xenonpy.descriptor import Compositions
        
        cal = Compositions(elemental_info=preset.elements_completed)
        get_symbol = Chem.GetPeriodicTable().GetElementSymbol
        symbols = [get_symbol(atom.GetAtomicNum()) for atom in rdmol.GetAtoms()]
        counts = Counter(symbols)
        comp = dict(counts)
        descriptor = cal.transform([comp])
        return torch.from_numpy(descriptor.to_numpy())


class RDMolPreprocessor:
    AVAIL_PREPROCESSORS = set(
        ['monomer']
    )
    
    @staticmethod
    def monomer(
        rdmol: Chem.Mol,
    ) -> Chem.Mol:
        rdmol = deepcopy(rdmol)
        attachments = [atom for atom in rdmol.GetAtoms() if atom.GetSymbol() == '*']
        
        ids = []
        for attachment in attachments:
            attachment_nbrs = attachment.GetNeighbors()
            for attachment_nbr in attachment_nbrs:
                attachment_nbr.SetProp('attachment', 'True')
            ids.append(attachment.GetIdx())
        
        ids = sorted(ids, reverse=True)
        rwmol = Chem.RWMol(rdmol)
        for i in ids:
            rwmol.RemoveAtom(i)
        rdmol = rwmol.GetMol()
        return rdmol


########################################################
############# End of new featurizers ###################
########################################################

AVAIL_FEATURES = set(FEATURIZER_REGISTRY.keys())
for key, cls in FEATURIZER_REGISTRY.items():
    AVAIL_FEATURES.update(cls._avail_features)


class ComposeFeaturizer:
    def __init__(
        self, 
        names: List[str], 
        config: dict = None, 
        add_hydrogens: bool = False
    ):
        self.names = names
        self.config = config
        self.add_hydrogens = add_hydrogens
        
        preprocessor = set(names) & RDMolPreprocessor.AVAIL_PREPROCESSORS
        assert len(preprocessor) <= 1, \
            f'Only one preprocessor is allowed, but got {preprocessor}'
        if preprocessor:
            self.preprocessor = getattr(RDMolPreprocessor, list(preprocessor)[0])
        else:
            self.preprocessor = None
        names = [name for name in names if name not in RDMolPreprocessor.AVAIL_PREPROCESSORS]
        
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
        if self.add_hydrogens:
            rdmol = Chem.AddHs(rdmol)
        
        if self.preprocessor is not None:
            rdmol = self.preprocessor(rdmol)
        
        mol_dict = {}
        for featurizer in self.featurizers:
            mol_dict.update(featurizer(rdmol))
        
        if 'virtual_bond' in self.names:
            if 'x' in mol_dict:
                mol_dict['x'] = torch.cat([
                    mol_dict['x'], torch.zeros(1, mol_dict['x'].shape[1])
                ], dim=0)
            if 'z' in mol_dict:
                mol_dict['z'] = torch.cat([
                    mol_dict['z'], torch.zeros(1)
                ], dim=0)
        
        return mol_dict
