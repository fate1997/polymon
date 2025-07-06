# import pandas as pd
# from polymon.data.dataset import PolymerDataset
# from polymon.model.gnn import ESAWrapper


# data_path = './database/internal/train.csv'
# feature_names = ['x', 'bond', 'z']
# label_column = 'Rg'

# dataset = PolymerDataset(
#     raw_csv_path=data_path,
#     feature_names=feature_names,
#     label_column=label_column,
#     smiles_column='SMILES',
#     identifier_column='id',
#     force_reload=True
# )


# model = ESAWrapper(
#     task_type="regression",
#     num_features = dataset[0].x.shape[1],
#     graph_dim = 512,
#     edge_dim = dataset[0].edge_attr.shape[1],
#     batch_size = 128,
#     lr = 1e-4,
#     linear_output_size = 1,
#     xformers_or_torch_attn='torch',
#     hidden_dims = [256, 256, 256, 256],
#     num_heads = [4, 4, 4, 4],
#     apply_attention_on = "edge",
#     layer_types = ['M', 'M', 'S', 'P'],
#     set_max_items = dataset[0].max_edge_global,
#     optimiser_weight_decay = 1e-10,

#     use_mlps = True,
#     mlp_hidden_size = 512,
#     mlp_type = 'gated_mlp',
#     mlp_layers = 2,
#     use_mlp_ln = True,
#     mlp_dropout = 0.0,

#     gradient_clip_val = 0.5,
#     use_bfloat16 = True,

#     pre_or_post = 'post',
#     seed = 42,
# )

# loaders = dataset.get_loaders(batch_size=128, n_train=0.8, n_val=0.1)
# train_loader, val_loader, test_loader = loaders

# for batch in train_loader:
#     y = batch.y
#     predictions = model(batch)
#     print(predictions.shape, y.shape)
#     break

from io import StringIO

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol

from tqdm import tqdm


def update_psmiles(smiles: str, identifier: int) -> str:

    mol = Chem.MolFromSmiles(smiles)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    star_atoms_id = []
    for idx, atom_symbol in enumerate(atoms):
        if atom_symbol == '*':
            star_atoms_id.append(idx)

    star_pair_list = []
    for star_id in star_atoms_id:
        star_pair_list.append(star_id)
        star_atom = mol.GetAtomWithIdx(star_id)
        neighbors = star_atom.GetNeighbors()
        assert len(neighbors) == 1, 'Error star neighbor num'
        for neighbor in neighbors:
            star_pair_list.append(neighbor.GetIdx())
            
    pair_1_star = star_pair_list[0]
    pair_1 = star_pair_list[3]
    atom = mol.GetAtomWithIdx(pair_1_star)
    atom.SetAtomicNum(mol.GetAtomWithIdx(pair_1).GetAtomicNum())

    pair_2_star = star_pair_list[2]
    pair_2 = star_pair_list[1]
    atom = mol.GetAtomWithIdx(pair_2_star)
    atom.SetAtomicNum(mol.GetAtomWithIdx(pair_2).GetAtomicNum())

    new_smiles = Chem.MolToSmiles(mol)

    return identifier, new_smiles, (pair_1_star, pair_2_star)
        
def init_geometry(mol: Mol) -> Mol:
    try:
        mol = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 42
        AllChem.EmbedMolecule(mol, ps)
        if mol.GetNumConformers() > 0:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        
        return mol
    except Exception as e:
        print(f"Error initializing geometry for {Chem.MolToSmiles(mol)}: {e}")
        return None
    
def write_sdf(emol: Mol, path: str):
    sio = StringIO()
    with Chem.SDWriter(sio) as w:
        w.write(emol)
    with open(path, 'a') as f:
        f.write(sio.getvalue())

def main():
    df = pd.read_csv('./database/internal/train.csv')
    smiles_list = df['SMILES'].tolist()
    smiles_id = dict(zip(df['id'], df['SMILES']))
    print('Number of original smiles:', len(smiles_list))
    filtered_smiles_id = []
    for id, smiles in smiles_id.items():
        if smiles.count('*') != 2 or any(atom.GetSymbol() in ['Li', 'Na', 'K', 'Rb', 
                                'Cs', 'Fr', 'Be', 'Mg', 
                                'Ca', 'Sr', 'Ba', 'Ra', 
                                'Sc', 'Y', 'La', 'Ce', 
                                'Pr', 'Nd', 'Pm', 'Sm', 
                                'Eu', 'Gd', 'Tb', 'Dy', 
                                'Ho', 'Er', 'Tm', 'Yb', 
                                'Lu', 'Hf', 'Ta', 'W',
                                'Re', 'Os', 'Ir', 'Pt', 
                                'Au', 'Hg', 'Tl', 'Pb', 
                                'Bi', 'Po', 'At', 'Rn',
                                'Cd', 'Ca', 'Se'] for atom in Chem.MolFromSmiles(smiles).GetAtoms()):
            filtered_smiles_id.append(id)
    
    df_valid = df[~df['id'].isin(filtered_smiles_id)]
    df_invalid = df[df['id'].isin(filtered_smiles_id)]
    pd.DataFrame({'id': df_invalid['id'], 'SMILES': df_invalid['SMILES']}).to_csv('./results/invalid_smiles.csv', index=False)
    print('Number of valid smiles:', len(df_valid))
    valid_smiles_dict = dict(zip(df_valid['id'], df_valid['SMILES']))
    valid_smiles_dict = {k: Chem.MolToSmiles(Chem.MolFromSmiles(v)) for k, v in valid_smiles_dict.items()}
    
    update_smiles_dict = {}
    for id, smiles in valid_smiles_dict.items():
        identifier, new_smiles, (pair_1_star, pair_2_star) = update_psmiles(smiles, id)
        update_smiles_dict[identifier] = [new_smiles, (pair_1_star, pair_2_star)]
    
    for id, (new_smiles, (pair_1_star, pair_2_star)) in tqdm(update_smiles_dict.items(), desc='Initializing geometry and writing sdf'):
        mol = Chem.MolFromSmiles(new_smiles, sanitize=True)
        emol = init_geometry(mol)
        emol.SetProp('_Name', str(id))
        emol.SetProp('dummy_1', str(pair_1_star))
        emol.SetProp('dummy_2', str(pair_2_star))
        emol.SetProp('psmiles', str(valid_smiles_dict[id]))
        write_sdf(emol, './results/train_emols.sdf')
        
if __name__ == '__main__':
    main()
        
    
    
    
    
    