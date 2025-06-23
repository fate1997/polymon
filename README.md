# polymon
This is for the Kaggle competition about polymer property prediction.

## Requirements

```bash
pip install tabpfn==2.0.9
pip install rdkit==2023.09.3
```

## TODO
- [ ] Try to assign 3D coordinates to polymers, and use `DimeNet++` to predict the properties.
- [ ] Incorporate more atom features and bond features and tried more GNN models.
- [ ] log file is only shown in the first one ('Rg')
- [ ] From the results below, GNN models and tabpfn models are the most promising models. Therefore, the next step is to find more descriptors for `TabPFN` and explore more GNN models.
- [ ] Allow fine-tuning the `TabPFN` model.
- [ ] Calculate `Rg` based on `cif` files

## Progress
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