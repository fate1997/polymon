import pathlib


REPO_DIR = pathlib.Path(__file__).parent.parent

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