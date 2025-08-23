import pathlib
import pandas as pd
import json


REPO_DIR = pathlib.Path(__file__).parent.parent
PRETRAINED_MODELS = ['polycl', 'polybert', 'gaff2_mod']
POLYCL_DIR = REPO_DIR / 'polymon' / 'model' / 'polycl' / 'model_utils'
POLYCL_MODEL_PTH = REPO_DIR / 'polymon' / 'model' / 'polycl' / 'polycl.pth'
POLYBERT_DIR = 'kuelumbus/polyBERT'

TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

MAX_SEQ_LEN = 120
SMILES_VOCAB = [
    ' ', '$', '!', '#', '%', '(', ')', '*', '+', '-', '/', '0', '1', '2', '3', 
    '4', '5', '6', '7', '8', '9', '=', '@', 'B', 'C', 'F', 'G', 'H', 'N', 'O', 
    'P', 'S', 'T', '[', '\\', ']', 'a', 'c', 'd', 'e', 'i', 'l', 'n', 'o', 'r', 
    's', 'R', 'G', 'X'
]
UNIQUE_ATOM_NUMS = [
    0, 1, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 20, 32, 34, 35, 48, 50, 52
]

with open(REPO_DIR / 'polymon' / 'data' / 'mordred_unstable.txt', 'r') as f:
    MORDRED_UNSTABLE_IDS = f.read().splitlines()
    MORDRED_UNSTABLE_IDS = list(map(int, MORDRED_UNSTABLE_IDS))


XENONPY_ELEMENTS_INFO = pd.read_csv(
    REPO_DIR / 'polymon' / 'data' / 'xenonpy_elements.csv',
    index_col=0,
)

GEOMETRY_VOCAB = REPO_DIR / 'database' / 'geometry_vocab'

with open(REPO_DIR / 'polymon' / 'data' / 'cgcnn.json', 'r') as f:
    CGCNN_ELEMENT_INFO = json.load(f)


DEFAULT_ATOM_FEATURES = [
    'degree', 
    'is_aromatic', 
    'chiral_tag', 
    'num_hydrogens', 
    'hybridization', 
    'mass', 
    'formal_charge', 
    'is_attachment',
]