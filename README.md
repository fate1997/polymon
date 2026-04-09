<img src="assets/polymon.png" alt="Polymon Icon">

`PolyMon` is a unified framework for polymer property prediction. It combines traditional machine learning methods (Random Forest, XGBoost, LightGBM, CatBoost, TabPFN) with state-of-the-art deep learning models (Graph Neural Networks including GATv2, GIN, PNA, DimeNet++, and KAN-based architectures).

<p align="center">
  <img src="assets/framework.png" alt="framework" width="400">
</p>

## Features

- **Multiple Model Types**: Support for both tabular ML models and graph-based deep learning models
- **Flexible Training Strategies**: K-fold cross-validation, hyperparameter optimization, ensemble learning, multi-fidelity learning, and active learning
- **Comprehensive Descriptors**: RDKit 2D/3D descriptors, ECFP fingerprints, Mordred descriptors, and graph-based representations
- **Multiple Properties**: Predict glass transition temperature (Tg), fractional free volume (FFV), radius of gyration (Rg), density, and thermal conductivity (Tc)
- **Easy-to-Use CLI**: Simple command-line interface for training, prediction, and active learning recommendations

## Installation

### Prerequisites

This package requires `torch>=2.2.2` and `torch_geometric>=2.5.3`. We recommend installing PyTorch with CUDA support first.

> **Note:** RDKit (a required dependency) is not yet compatible with NumPy 2.x. If you have NumPy 2.x installed, downgrade it with `pip install 'numpy<2'` before installing PolyMon.

### Step 1: Install PyTorch and PyTorch Geometric

```bash
# For CUDA 11.8
conda install -y pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
                 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
```

### Step 2: Install PolyMon

```bash
pip install polymon
```

### Development Installation

```bash
git clone https://github.com/fate1997/polymon.git
cd polymon
pip install -e .
```

## Quick Start

### Training a Model

Train a tabular model (e.g., Random Forest) with RDKit 2D descriptors:

```bash
polymon train \
    --raw-csv ./database/database.csv \
    --sources Kaggle PI1070 PolyMetriX \
    --labels Tg \
    --feature-names rdkit2d \
    --model rf \
    --n-fold 5 \
    --out-dir ./results
```

Train a graph neural network (GNN) model:

```bash
polymon train \
    --raw-csv ./database/database.csv \
    --sources Kaggle PI1070 PolyMetriX \
    --labels Tg \
    --model gatv2 \
    --n-fold 5 \
    --n-trials 15 \
    --num-epochs 2500 \
    --out-dir ./results
```

### Making Predictions

```bash
polymon predict \
    --model-path ./results/gatv2/Tg/train/gatv2_Tg.pt \
    --csv-path ./data/new_polymers.csv \
    --smiles-column SMILES
```

### Active Learning Recommendations

```bash
polymon rec \
    --pool-csv ./database/pool.csv \
    --trained-model ./results/gatv2/Rg/train/gatv2_Rg-KFold.pt \
    --acquisition uncertainty \
    --sample-size 20 \
    --save-path recommended.csv
```

## Available Models

### Tabular Models (for use with `--feature-names`)

| Model | CLI Name | Description |
|-------|----------|-------------|
| Random Forest | `rf` | Ensemble of decision trees |
| XGBoost | `xgb` | Gradient boosting framework |
| LightGBM | `lgbm` | Light gradient boosting machine |
| CatBoost | `catboost` | Gradient boosting on decision trees |
| TabPFN | `tabpfn` | Prior-data trained network |

### Deep Learning Models

| Model | CLI Name | Description |
|-------|----------|-------------|
| GATv2 | `gatv2` | Graph Attention Network v2 |
| GIN | `gin` | Graph Isomorphism Network |
| PNA | `pna` | Principal Neighbourhood Aggregation |
| AttentiveFP | `attentivefp` | Attention-based molecular fingerprinting |
| DimeNet++ | `dimenetpp` | Directional message passing |
| GPS | `gps` | Graph Positional Encoding network |
| KAN-GATv2 | `fastkan_gatv2` | Kolmogorov-Arnold Network + GATv2 |
| KAN-GPS | `kan_gps` | Kolmogorov-Arnold Network + GPS |

## Available Descriptors

### For Tabular Models (`--feature-names`)

- `rdkit2d`: RDKit 2D molecular descriptors
- `ecfp4`: Extended Connectivity Fingerprints (ECFP4)
- `mordred`: 1800+ Mordred descriptors
- `maccs`: MACCS keys
- `xenonpy_desc`: XenonPy elemental composition descriptors

### For Graph Models

Graph models automatically use molecular graph features. Additional descriptors can be added via `--descriptors`:
- `rdkit2d`, `ecfp4`, `mordred`, `maccs`, `xenonpy_desc`
- `oligomer_rdkit2d`, `oligomer_mordred`, `oligomer_ecfp4` (for oligomer representations)

## Target Properties

| Property | Symbol | Unit | Description |
|----------|--------|------|-------------|
| Glass Transition Temperature | `Tg` | K | Temperature at which polymer transitions from glassy to rubbery |
| Fractional Free Volume | `FFV` | - | Fraction of volume not occupied by polymer chains |
| Radius of Gyration | `Rg` | Å | Measure of polymer chain size |
| Density | `Density` | g/cm³ | Mass per unit volume |
| Thermal Conductivity | `Tc` | W/m·K | Heat transfer capability |

## Advanced Usage

### Hyperparameter Optimization

```bash
polymon train \
    --labels Tg \
    --model gatv2 \
    --n-trials 15 \
    --n-fold 5 \
    --raw-csv ./database/database.csv
```

### Multi-Fidelity Learning (Fine-tuning)

```bash
# Train on low-fidelity data first
polymon train \
    --labels Density \
    --model gatv2 \
    --sources MD-simulation \
    --run-production

# Fine-tune on high-fidelity data
polymon train \
    --labels Density \
    --model gatv2 \
    --sources Experimental \
    --finetune \
    --pretrained-model ./results/gatv2/Density/production/gatv2_Density.pt \
    --finetune-csv-path ./database/experimental.csv
```

### Ensemble Learning

```bash
polymon train \
    --labels Rg \
    --model gatv2 \
    --n-estimator 10 \
    --ensemble-type voting \
    --raw-csv ./database/database.csv
```

### Delta-Learning with Empirical Estimators

```bash
polymon train \
    --labels Density \
    --model gatv2 \
    --train-residual \
    --estimator-name Density-IBM \
    --raw-csv ./database/database.csv
```

## Python API

For more advanced usage, you can use the Python API directly:

```python
from polymon.model.base import ModelWrapper

# Load a trained model
model = ModelWrapper.from_file('results/gatv2/Tg/train/gatv2_Tg.pt')

# Make predictions
predictions = model.predict(['*C*', '*CC*', '*CCC*'])
print(predictions)
```

## Citation

If you use PolyMon in your research, please cite:

```bibtex
@article{polymon2024,
  title={PolyMon: A Unified Framework for Polymer Property Prediction},
  author={Ren, Gaopeng and Yang, Yijie and Zhou, Jiajun and Jelfs, Kim E.},
  journal={Journal of Chemical Information and Modeling},
  year={2024}
}
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
