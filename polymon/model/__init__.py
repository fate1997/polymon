import typing
from typing import Any, Dict

from polymon.model.gatv2.fastkan_gatv2 import FastKAN_GATv2
from polymon.model.gatv2.gat_chain import GATChain
from polymon.model.gatv2.gat_chain_readout import GATv2ChainReadout
from polymon.model.gatv2.kan_gatv2 import KAN_GATv2
from polymon.model.gnn import (GIN, PNA, AttentiveFPWrapper, DimeNetPP,
                               GATPort, GATv2, GATv2VirtualNode,
                               GraphTransformer)
from polymon.model.gps.gps import KAN_GPS, GraphGPS
from polymon.model.gvp import GVPModel
from polymon.model.kan.efficient_kan import EfficientKANWrapper
from polymon.model.kan.fast_kan import FastKANWrapper
# from polymon.model.kan.vanilla import KANWrapper
from polymon.model.kan.fourier_kan import FourierKANWrapper
from polymon.model.gatv2.lineevo import GATv2LineEvo
from polymon.model.gatv2.gatv2_sage import GATv2SAGE
from polymon.model.gatv2.multi_fidelity import GATv2_Source
from polymon.model.gatv2.position_encoding import GATv2_PE
from polymon.model.gatv2.embed_residual import GATv2EmbedResidual
from polymon.model.kan.gin import KAN_GIN, FastKAN_GIN
from polymon.model.kan.gcn import KAN_GCN
from polymon.model.dmpnn import DMPNN
from polymon.model.kan.dmpnn import KAN_DMPNN

if typing.TYPE_CHECKING:
    from polymon.model.base import BaseModel

def build_model(
    model_type: str, 
    num_node_features: int,
    num_edge_features: int,
    num_descriptors: int,
    hparams: Dict[str, Any]
) -> 'BaseModel':
    if model_type == 'gatv2':
        input_args = {
            'num_atom_features': num_node_features,
            'edge_dim': num_edge_features,
            'num_descriptors': num_descriptors,
        }
        hparams.update(input_args)
        model = GATv2(**hparams)
    elif model_type == 'attentivefp':
        input_args = {
            'in_channels': num_node_features,
            'edge_dim': num_edge_features,
            'out_channels': 1,
        }
        hparams.update(input_args)
        model = AttentiveFPWrapper(**hparams)
    elif model_type == 'dimenetpp':
        input_args = {
            'out_channels': 1,
        }
        hparams.update(input_args)
        model = DimeNetPP(**hparams)
    elif model_type == 'gatport':
        input_args = {
            'num_atom_features': num_node_features,
            'edge_dim': num_edge_features,
            'num_descriptors': num_descriptors,
        }
        hparams.update(input_args)
        model = GATPort(**hparams)
    elif model_type == 'gatv2vn':
        input_args = {
            'num_atom_features': num_node_features,
            'edge_dim': num_edge_features,
            'num_descriptors': num_descriptors,
        }
        hparams.update(input_args)
        model = GATv2VirtualNode(**hparams)
    elif model_type == 'gin':
        input_args = {
            'num_atom_features': num_node_features,
        }
        hparams.update(input_args)
        model = GIN(**hparams)
    elif model_type == 'pna':
        input_args = {
            'in_channels': num_node_features,
            'edge_dim': num_edge_features,
        }
        hparams.update(input_args)
        model = PNA(**hparams)
    elif model_type == 'gvp':
        input_args = {
            'in_node_nf': num_node_features,
        }
        hparams.update(input_args)
        model = GVPModel(**hparams)
    elif model_type == 'gatchain':
        input_args = {
            'num_atom_features': num_node_features,
        }
        hparams.update(input_args)
        model = GATChain(**hparams)
    elif model_type == 'gatv2chainreadout':
        input_args = {
            'num_atom_features': num_node_features,
            'edge_dim': num_edge_features,
            'num_descriptors': num_descriptors,
        }
        hparams.update(input_args)
        model = GATv2ChainReadout(**hparams)
    elif model_type == 'gt':
        input_args = {
            'in_channels': num_node_features,
        }
        hparams.update(input_args)
        model = GraphTransformer(**hparams)
    elif model_type == 'kan_gatv2':
        input_args = {
            'num_node_features': num_node_features,
        }
        hparams.update(input_args)
        model = KAN_GATv2(**hparams)
    elif model_type == 'gps':
        input_args = {
            'in_channels': num_node_features,
            'edge_dim': num_edge_features,
        }
        hparams.update(input_args)
        model = GraphGPS(**hparams)
    elif model_type == 'kan_gps':
        input_args = {
            'in_channels': num_node_features,
            'edge_dim': num_edge_features,
        }
        hparams.update(input_args)
        model = KAN_GPS(**hparams)
    elif model_type == 'fastkan':
        input_args = {
            'in_channels': num_descriptors,
        }
        hparams.update(input_args)
        model = FastKANWrapper(**hparams)
    elif model_type == 'efficientkan':
        input_args = {
            'in_channels': num_descriptors,
        }
        hparams.update(input_args)
        model = EfficientKANWrapper(**hparams)
    # elif self.model_type == 'kan':
    #     input_args = {
    #         'in_channels': self.dataset[0].descriptors.shape[1],
    #         'device': self.device,
    #     }
    #     input_args.update(hparams)
    #     model = KANWrapper(**input_args)
    elif model_type == 'fourierkan':
        input_args = {
            'in_channels': num_descriptors,
        }
        hparams.update(input_args)
        model = FourierKANWrapper(**hparams)
    elif model_type == 'fastkan_gatv2':
        input_args = {
            'num_atom_features': num_node_features,
            'edge_dim': num_edge_features,
            'num_descriptors': num_descriptors,
        }
        hparams.update(input_args)
        model = FastKAN_GATv2(**hparams)
    elif model_type == 'gatv2_lineevo':
        input_args = {
            'num_atom_features': num_node_features,
        }
        hparams.update(input_args)
        model = GATv2LineEvo(**hparams)
    elif model_type == 'gatv2_sage':
        input_args = {
            'num_atom_features': num_node_features,
            'edge_dim': num_edge_features,
        }
        hparams.update(input_args)
        model = GATv2SAGE(**hparams)
    elif model_type == 'gatv2_source':
        input_args = {
            'num_atom_features': num_node_features,
            'edge_dim': num_edge_features,
        }
        hparams.update(input_args)
        model = GATv2_Source(**hparams)
    elif model_type == 'gatv2_pe':
        input_args = {
            'num_atom_features': num_node_features,
            'edge_dim': num_edge_features,
        }
        hparams.update(input_args)
        model = GATv2_PE(**hparams)
    elif model_type == 'gatv2_embed_residual':
        input_args = {
            'num_atom_features': num_node_features,
            'edge_dim': num_edge_features,
            'num_descriptors': num_descriptors,
        }
        hparams.update(input_args)
        model = GATv2EmbedResidual(**hparams)
    elif model_type == 'kan_gin':
        input_args = {
            'num_atom_features': num_node_features,
        }
        hparams.update(input_args)
        model = KAN_GIN(**hparams)
    elif model_type == 'fastkan_gin':
        input_args = {
            'num_atom_features': num_node_features,
        }
        hparams.update(input_args)
        model = FastKAN_GIN(**hparams)
    elif model_type == 'kan_gcn':
        input_args = {
            'num_node_features': num_node_features,
        }
        hparams.update(input_args)
        model = KAN_GCN(**hparams)
    elif model_type == 'dmpnn':
        input_args = {
            'atom_fdim': num_node_features,
            'bond_fdim': num_edge_features,
            'global_features_size': num_descriptors,
        }
        hparams.update(input_args)
        model = DMPNN(**hparams)
    elif model_type == 'kan_dmpnn':
        input_args = {
            'atom_fdim': num_node_features,
            'bond_fdim': num_edge_features,
            'global_features_size': num_descriptors,
        }
        hparams.update(input_args)
        model = KAN_DMPNN(**hparams)
    else:
        raise ValueError(f"Model type {model_type} not implemented")
    
    return model