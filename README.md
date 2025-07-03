# polymon
This is for the Kaggle competition about polymer property prediction.

## Requirements

```bash
pip install tabpfn==2.0.9
pip install rdkit==2023.09.3
```

## TODO
- [ ] Try to assign 3D coordinates to polymers, and use `DimeNet++` to predict the properties.
- [ ] Calculate `Rg` based on MD files
- [ ] Test `GATv2` with bond features
- [ ] Test `GATv2` with different loss functions l1-loss and mse-loss
- [ ] Implement cross-validation methods for DL models
- [ ] GIN, GCN2, PNA, ESA implementation and hyperparameter optimization
- [ ] Test performance of `GATPort`
- [x] Virtual node (use descriptors as the initial features) (atom + descriptors seems have potential)
- [x] Incorporate more atom features and bond features and tried more GNN models.
- [x] log file is only shown in the first one ('Rg'), this might be solved by using `loguru`
- [x] Allow fine-tuning the `TabPFN` model. **[To be Considered]**
- [x] Extract more data points from pdb files **[To be Considered]**
- [x] Standardize the data point deduplication process
- [x] Monomer grow to dimer and then calculate the descriptors

## Progress

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