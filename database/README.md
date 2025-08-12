# Database

## Format
The database is a pandas dataframe with the following columns:
- **SMILES**: The SMILES string of the polymer. `*` is used to represent the attachment point.
- **Tg**: The glass transition temperature of the polymer. Unit: $^\circ$C
- **FFV**: The free volume fraction of the polymer. Unit: 1
- **Rg**: The radius of gyration of the polymer. Unit: $\rm\AA$
- **Density**: The density of the polymer. Unit: $\rm g/cm^3$
- **Tc**: Thermal conductivity. Unit: $\rm W/(m\cdot K)$
- **Source**: The source of the data. Details are shown below.
- **Uncertainty**: The uncertainty of the data.
  - `0`: The data is from the internal training database.
  - `1`: The data is from the official external database.
  - `2`: The data is from the collected MD-simulated database.
  - `3`: The data is from the collected experimental database.
  - `4`: Unknown source.

## Sources
| Source | Datasets | Description | Reference | Location | 
|--------|-------------|-----------|-----------|-----------|
| internal | `Tg`, `FFV`, `Rg`, `Density`, `Tc` | Original training data from `kaggle` | [kaggle](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data) | `internal` |
| official_external | `Tg`, `FFV`, `Tc` | Official external database | [kaggle](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data) | `official_external` |
| MAFA-exp | `Density`, `Tg` | Experimental data | [MAFA](https://pubs.acs.org/doi/10.1021/acsapm.0c00524) | `external/ap0c00524_si_001.csv` |
| MAFA-MD | `Density`, `Tg` | From MD simulations | [MAFA](https://pubs.acs.org/doi/10.1021/acsapm.0c00524) | `external/ap0c00524_si_001.csv` |
| SC-Bicerano | `Tg` | Experimental data | [SC](https://www.nature.com/articles/s41597-024-03212-4) | `external/Bicerano_bigsmiles.csv` |
| SC-JCIM | `Tg` | Experimental data | [SC](https://www.nature.com/articles/s41597-024-03212-4) | `external/JCIM_sup_bigsmiles.csv` |
| GREA | `Density`, `Tg` | from PolyInfo| [GREA](https://arxiv.org/pdf/2206.02886) | `external/density_raw.csv`, `external/tg_raw.csv` |
| unknown | `FFV` | Unknown source for homopolymer and polyamide| unknown | `external/FFV_homopolymer.xlsx`, `external/FFV_polyamides.xlsx` |
| PolyMetriX | `Tg` | Curated database | [PolyMetriX](https://github.com/lamalab-org/PolyMetriX?tab=readme-ov-file) | `external/LAMALAB_CURATED_Tg_structured_polymerclass.csv` |
| PI1070 | `Tc`, `Density`, `Rg` | From MD simulations, contain std | [PI1070](https://github.com/RadonPy/RadonPy/blob/develop/data/PI1070.csv) | `external/PI1070.csv` |
| LT-exp | `Tg` | Experimental data | [LT](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01031) | `external/Supplement_Dataset1_MD_100.xlsx` |
| LT-MD | `Tg` | From MD simulations | [LT](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01031) | `external/Supplement_Dataset1_MD_100.xlsx` |
| Kaggle | `Tc` | From Kaggle | non-official | `external/Tc_kaggle.csv` |
| HT-PolyInfo | `Tg` | From PolyInfo | [HT](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c00550) | `external/Tg.csv` |
| HT-exp | `Tg` | From journals | [HT](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c00550) | `external/Tg_OOD_EXP.csv` |
| HT-MD | `Tg` | From MD simulations | [HT](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c00550) | `external/Tg_OOD_ME.csv` |
