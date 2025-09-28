# A unified framework for polymer property prediction
We provide a unified framework for polymer property prediction. This framework is designed to be flexible and easy to use.

## Installation
```bash
pip install polymon
```

## Usage
### Train
```bash
polymon train --help
```

### Merge
```bash
polymon merge --help
```

### Predict
```bash
polymon predict --help
```

## Citation
To be updated.

## Progress
GNNs
- [ ] GATv2
- [ ] GIN
- [ ] DMPNN
- [ ] AttentiveFP
- [ ] GPS
- [ ] PNA
- [ ] GT
- [ ] GCN

Density Multi-fidelity
- [ ] finetune
- [ ] label residual
- [ ] emb residual
- [ ] multi-head

Residual
- [ ] graph embeddings from other properties (4 x 5)
- [ ] Rg-residual
- [x] Density-residual vdw
- [x] Density-residual fedors
- [ ] Density-residual ibm
- [ ] Tg-residual
- [ ] Atom-contribution (re-implement with training set to train atom contribution)

Actively learning
- [ ] Rg + 1st round
- [ ] Rg + 2nd round
- [ ] Rg + 3rd round

Representation
- [ ] GATv2-vn
- [x] GATv2 periodic
- [ ] KAN-GATv2
- [ ] KAN-GPS
- [ ] KAN-GIN
- [ ] KAN-DMPNN
- [ ] FastKAN-GATv2
- [ ] GATv2-chain-readout
- [ ] GATv2-lineevo-readout
- [x] GATv2_PE
