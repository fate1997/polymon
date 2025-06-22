from functools import partial
from typing import List

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from polymon.data.polymer import Polymer
from polymon.model.polycl import polycl
from polymon.setting import POLYBERT_DIR, POLYCL_DIR, POLYCL_MODEL_PTH


def get_polycl_embeddings(
    smiles_list: List[str],
    device: str = 'cpu',
    batch_size: int = 64,
    pretrained_dir: str = POLYCL_DIR,
    pretrained_model_pth: str = POLYCL_MODEL_PTH,
) -> torch.Tensor:
    smiles_list = [to_psmiles(smiles) for smiles in smiles_list]

    model_config = polycl.set_dropout(
        AutoConfig.from_pretrained(pretrained_dir), 
        dropout = False
    )
    model_arc = AutoModel.from_config(config = model_config)
    model = polycl.polyCL(encoder = model_arc, pooler = "cls")

    # 1. Load the pre-trained weights trained by PolyCL
    model.from_pretrained(pretrained_model_pth)
    model = model.to(device)
    model.eval()
    
    # 2. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
    tokenizer = partial(
        tokenizer,
        max_length=512, 
        padding="max_length", 
        truncation=False, 
        return_tensors='pt'
    )

    # 3. Get the embeddings of polymers
    with torch.no_grad():
        polymer_embeddings = []
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i: i+batch_size]
            batch_encoding = tokenizer(batch_smiles)
            batch_encoding = {k: v.to(device) for k, v in batch_encoding.items()}
            batch_embeddings, _ = model(batch_encoding)
            polymer_embeddings.append(batch_embeddings.cpu())
        polymer_embeddings = torch.cat(polymer_embeddings, dim=0)

    return polymer_embeddings


def get_polybert_embeddings(
    smiles_list: List[str],
    device: str = 'cpu',
    batch_size: int = 64,
    pretrained_dir: str = POLYBERT_DIR,
) -> torch.Tensor:
    smiles_list = [to_psmiles(smiles) for smiles in smiles_list]

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        s = token_embeddings.size()
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(s).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / \
            torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    model = AutoModel.from_pretrained(pretrained_dir)
    model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
    tokenizer = partial(
        tokenizer,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        polymer_embeddings = []
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i: i+batch_size]
            batch_encoding = tokenizer(batch_smiles)
            batch_encoding = {k: v.to(device) for k, v in batch_encoding.items()}
            batch_model_output = model(**batch_encoding)
            batch_embeddings = mean_pooling(
                batch_model_output, 
                batch_encoding['attention_mask']
            )
            polymer_embeddings.append(batch_embeddings.cpu())
        polymer_embeddings = torch.cat(polymer_embeddings, dim=0)

    return polymer_embeddings


def assign_pretrained_embeddings(
    data_list: List[Polymer],
    embeddings: torch.Tensor,
):
    for i, data in enumerate(data_list):
        if getattr(data, 'descriptors', None) is None:
            data.descriptors = embeddings[i].unsqueeze(0)
        else:
            data.descriptors = torch.cat(
                [data.descriptors, embeddings[i].unsqueeze(0)], dim=1
            )

# to psmiles(polyBERT)
def to_psmiles(smiles):
    return smiles.replace("*", "[*]")

# to smiles
def to_smiles(psmiles):
    return  psmiles.replace("[*]", "*")