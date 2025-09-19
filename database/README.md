# Database
We collected the following databases, and they are from various sources. If you want to use them, please cite the reference.

## Format
The database is a pandas dataframe with the following columns:
- **SMILES**: The SMILES string of the polymer. `*` is used to represent the attachment point.
- **Tg**: The glass transition temperature of the polymer. Unit: $^\circ$C
- **FFV**: The free volume fraction of the polymer. Unit: 1
- **Rg**: The radius of gyration of the polymer. Unit: $\rm\AA$
- **Density**: The density of the polymer. Unit: $\rm g/cm^3$
- **Tc**: Thermal conductivity. Unit: $\rm W/(m\cdot K)$
- **Source**: The source of the data. Details are shown below.

## Sources
| Source | Datasets | Description | Reference | Location | License |
|--------|-------------|-----------|-----------|-----------|-----------|
| Kaggle | `Tg`, `FFV`, `Rg`, `Density`, `Tc` | Original training data from `kaggle` | [kaggle](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data) | `kaggle.csv` | MIT |
| MAFA-exp | `Density`, `Tg` | Experimental data | [MAFA](https://pubs.acs.org/doi/10.1021/acsapm.0c00524) | `external/ap0c00524_si_001.csv` | N/A |
| MAFA-MD | `Density`, `Tg` | From MD simulations | [MAFA](https://pubs.acs.org/doi/10.1021/acsapm.0c00524) | `external/ap0c00524_si_001.csv` | N/A |
| SC-Bicerano | `Tg` | Experimental data | [SC](https://www.nature.com/articles/s41597-024-03212-4) | `external/Bicerano_bigsmiles.csv` | N/A |
| SC-JCIM | `Tg` | Experimental data | [SC](https://www.nature.com/articles/s41597-024-03212-4) | `external/JCIM_sup_bigsmiles.csv` | N/A |
| GREA | `Density`, `Tg` | from PolyInfo| [GREA](https://arxiv.org/pdf/2206.02886) | `external/density_raw.csv`, `external/tg_raw.csv` | MIT |
| PolyMetriX | `Tg` | Curated database | [PolyMetriX](https://github.com/lamalab-org/PolyMetriX?tab=readme-ov-file) | `external/LAMALAB_CURATED_Tg_structured_polymerclass.csv` | MIT |
| PI1070 | `Tc`, `Density`, `Rg` | From MD simulations, contain std | [PI1070](https://github.com/RadonPy/RadonPy/blob/develop/data/PI1070.csv) | `external/PI1070.csv` | BSD 3-Clause |
| LT-exp | `Tg` | Experimental data | [LT](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01031) | `external/Supplement_Dataset1_MD_100.xlsx` | N/A |
| LT-MD | `Tg` | From MD simulations | [LT](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01031) | `external/Supplement_Dataset1_MD_100.xlsx` | N/A |
| HT-PolyInfo | `Tg` | From PolyInfo | [HT](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c00550) | `external/Tg.csv` | Apache-2.0 |
| HT-exp | `Tg` | From journals | [HT](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c00550) | `external/Tg_OOD_EXP.csv` | Apache-2.0 |
| HT-MD | `Tg` | From MD simulations | [HT](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c00550) | `external/Tg_OOD_ME.csv` | Apache-2.0 |

## Reference
[Kaggle] Gang Liu, Jiaxin Xu, Eric Inae, Yihan Zhu, Ying Li, Tengfei Luo, Meng Jiang, Yao Yan, Walter Reade, Sohier Dane, Addison Howard, and María Cruz. NeurIPS - Open Polymer Prediction 2025. https://kaggle.com/competitions/neurips-open-polymer-prediction-2025, 2025. Kaggle.
[MAFA] Afzal, M. A. F., Browning, A. R., Goldberg, A., Halls, M. D., Gavartin, J. L., Morisato, T., ... & Goose, J. E. (2020). High-throughput molecular dynamics simulations and validation of thermophysical properties of polymers for various applications. ACS Applied Polymer Materials, 3(2), 620-630.
[SC] Choi, S., Lee, J., Seo, J., Han, S. W., Lee, S. H., Seo, J. H., & Seok, J. (2024). Automated BigSMILES conversion workflow and dataset for homopolymeric macromolecules. Scientific data, 11(1), 371.
[GREA] Liu, G., Zhao, T., Xu, J., Luo, T., & Jiang, M. (2022, August). Graph rationalization with environment-based augmentations. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 1069-1078).
[PolyMetriX] lamalab-org. (2025, August 11). PolyMetriX (Version 0.2.0) [Computer software]. GitHub. https://github.com/lamalab-org/PolyMetriX
[PI1070] Y. Hayashi, J. Shiomi, J. Morikawa, R. Yoshida, "RadonPy: Automated Physical Property Calculation using All-atom Classical Molecular Dynamics Simulations for Polymer Informatics," npj Comput. Mater., 8:222 (2022)
[LT] Tao, L., Varshney, V., & Li, Y. (2021). Benchmarking machine learning models for polymer informatics: an example of glass transition temperature. Journal of Chemical Information and Modeling, 61(11), 5395-5413.
[HT] Tang, H., Yue, T., & Li, Y. (2025). Assessing Uncertainty in Machine Learning for Polymer Property Prediction: A Benchmark Study. Journal of Chemical Information and Modeling.
