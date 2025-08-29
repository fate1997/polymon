# polymon
This is for the Kaggle competition about polymer property prediction.

## Requirements

```bash
pip install tabpfn==2.0.9
pip install rdkit==2023.09.3
```
## Overall TODO
1. `Dataset`: Merge datasets more wisely (e.g., using uncertainty to merge datasets)
2. `Feature`: Explore new node features, bond features, and graph features.
3. `Graph`: How to construct a good polymer graph?
4. `Model`: Explore more GNNs, Transformers, or other models. Check the existing hyperparameters.
5. `Training`: Add learning rate scheduler, cross-validation, multi-task training, scaffold-based splitting, ensemble, etc.

## TODO
- [x] Dataset ablation study. Find the best sources for each property. = = rather than using the previous datasets.
- [ ] Loss weights for different fidelities.
- [ ] Maybe add pruner for hyper-parameter optimization to accelerate the process.
- [ ] Non-bonded (based on conformer or Transformer)
- [ ] Source as one-hot encoding!
- [ ] Group contribution in https://github.com/IBM/polymer_property_prediction/blob/main/src/polymer_property_prediction/polymer_properties_from_smiles.py
- [ ] Atom features from pretrained GNN models
- [x] RF -> IMPORTANT FEATURES -> GNN = = not so good
- [ ] Use linear first in the KAN-GNN models (https://github.com/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks)
- [ ] Pretrained on our own dataset following [MIPS](https://github.com/wjxts/MIPS), [FragNet](https://github.com/pnnl/FragNet) or [SimSGT](https://github.com/syr-cn/SimSGT)
- [ ] Review: https://www.sciencedirect.com/science/article/pii/S0079642525001227#b1005

- [x] Try to assign 3D coordinates to polymers, and use `DimeNet++` to predict the properties.
- [ ] Calculate `Rg` based on MD files
- [x] Test `GATv2` with bond features
- [x] Test `GATv2` with different loss functions l1-loss and mse-loss
- [x] Implement cross-validation methods for DL models
- [x] GIN, GCN2, PNA, ESA implementation and hyperparameter optimization
- [x] Test performance of `GATPort`
- [x] Virtual node (use descriptors as the initial features) (atom + descriptors seems have potential)
- [x] Incorporate more atom features and bond features and tried more GNN models.
- [x] log file is only shown in the first one ('Rg'), this might be solved by using `loguru`
- [x] Allow fine-tuning the `TabPFN` model. **[To be Considered]**
- [x] Extract more data points from pdb files **[To be Considered]**
- [x] Standardize the data point deduplication process
- [x] Monomer grow to dimer and then calculate the descriptors

## Progress
**2025-08-19**
Dataset ablation study ($^1$ means the dataset is splitted randomly):
- Tc: 
  - previous$^1$: 0.0470 (local); 0.069 (kaggle)
  - only internal$^1$: 0.0538 (local); 0.070 (kaggle)
  - \+ official_external: 0.0678 (local)
  - \+ official_external (internal serves as test/val): 0.0.0614 (local); 0.071 (kaggle)
  - \+ Kaggle: 0.0663 (local)
  - \+ PI1070: 0.0651 (local); 0.069 (kaggle)
  - \+ PI1070 (internal serves as test/val): 0.0592 (local); 0.068 (kaggle)
  - \+ PI1070 (internal serves as val in production): 0.0589 (local); 0.068 (kaggle)
- Density (all the below are using internal as test/val):
  - previous (only internal)$^1$: 0.0188; 0.067 (kaggle)
  - \+ MAFA-MD: 0.0297 (local); 
  - \+ MAFA-exp: 0.0224 (local); 0.069 (kaggle)
  - \+ GREA: 0.0501 (local); 
  - \+ PI1070:0.0252 (local)
  - \+ PI1070 (linear fitting): 0.0331 (local)
  - \+ PI1070 (only bias): 0.0310 (local)
- Rg:
  - previous$^1$: 0.0458 (local); 0.067 (kaggle)
  - only internal$^1$: 0.0739 (local)
  - \+ PI1070: 0.0814 (local)
- FFV:
  - previous$^1$: 0.0082 (local); 0.067 (kaggle)
  - only internal$^1$: 0.0099 (local)
  - \+ official_external: 0.0087 (local)
  - \+ unknown (have the same number of data points as official_external): 0.0094 (local)
- Tg:
  - previous$^1$: 0.0280 (local)
  - only internal$^1$: 0.0798, 0.0965, 0.0875
  - \+ official_external: 0.0880, 0.0925, 0.0716
  - \+ MAFA-exp: 0.0843, 0.0943, 0.0841
  - \+ MAFA-MD: 0.0970, 0.1076, 0.087
  - \+ SC-Bicerano: 0.0841, 0.0913, 0.0779
  - \+ SC-JCIM: 0.0824, 0.0882, 0.0669
  - \+ GREA: 0.0918, 0.0947, 0.0949
  - \+ PolyMetriX: 0.0924, 0.0899, 0.0934
  - \+ LT-exp: 0.0855, 0.0975, 0.0685
  - \+ LT-MD: 0.0869, 0.0967, 0.0692
  - \+ HT-PolyInfo: 0.0972, 0.0939, 0.0939
  - \+ HT-exp: 0.0897, 0.0966, 
  - \+ HT-MD: 0.0864 (local)

**2025-07-03**
- Finally know why `GATv2` is good after refactoring. Just because add hydrogens = =.

**2025-07-01**
- `GATv2VirtualNode` is not good. Tg val R2 is 0.855 even after 800 epochs.
- 

**2025-06-30**
- `GATv2` is good!
- Hyperparameter optimization works! Especially for `Tg`, but not for `Rg`
- Train longer, the performance for `Tg` is better.

**2025-06-25**
- `Rg` + `Tc`: TabPFN - Mordred is great!

**2025-06-23**
- `Density_merged.csv` is not good, it seems that the data points are not accurate
- 

**2025-06-22**
- `TunedTabPFNRegressor` run faster, but the performance is not as good as vanilla one. Meanwhile, it is hard to use the best parameters for production model. Note this model also has decision tree involved.
- `AutoTabPFNRegressor` did not give good results.

**2025-06-21**
Conclusions:
- `tabpfn`'s performance is incrediable.
- `GNN` models' performance on FFV can beat traditional models except `tabpfn`.

| model | Tg | FFV | Tc | Density | Rg | score | extra_info |
|-------|----|-----|----|---------|----|-------|------------|
| xgb | 0.0957 | 0.0147 | 0.0505 | 0.0336 | 0.1545 | 0.0798 | rdkit2d |
| rf | 0.0923 | 0.0128 | 0.0512 | 0.0271 | 0.0872 | 0.0617 | rdkit2d |
| catboost | 0.0881 | 0.0136 | 0.0543 | 0.0431 | 0.1365 | 0.0766 | rdkit2d |
| tabpfn | 0.0756 | 0.0091 | 0.0514 | 0.0207 | 0.0829 | 0.0548 | rdkit2d |
| lgbm | 0.1112 | 0.0251 | 0.1034 | 0.061 | 0.095 | 0.0882 | rdkit2d |
|-------|----|-----|----|---------|----|-------|------------|
| gatv2 | 0.0787 | 0.0115 | 0.0560 | 0.0252 | 0.0941 | 0.0604 | 32hidden-huberloss-edge |
| attentivefp | 0.0844 | 0.0115 | 0.0565 | 0.0216 | 0.1069 | 0.0640 | 32hidden-huberloss-edge |