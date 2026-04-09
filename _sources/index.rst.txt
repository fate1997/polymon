Welcome to PolyMon's documentation!
=====================================

PolyMon is a unified framework for polymer property prediction. It combines traditional machine learning methods (Random Forest, XGBoost, LightGBM, CatBoost, TabPFN) with state-of-the-art deep learning models (Graph Neural Networks including GATv2, GIN, PNA, DimeNet++, and KAN-based architectures).

**Key Features:**

- Multiple model types: Tabular ML and Graph Neural Networks
- Flexible training strategies: Cross-validation, hyperparameter optimization, ensemble learning
- Advanced techniques: Multi-fidelity learning, delta-learning, active learning
- Comprehensive descriptors: RDKit, Mordred, ECFP fingerprints, graph-based representations
- Support for 5 key polymer properties: Tg, FFV, Rg, Density, Tc

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   examples

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   modules/data
   modules/model
   modules/estimator
   modules/exp

.. toctree::
   :maxdepth: 1
   :caption: CLI

   cli

Quick Start
-----------

**Installation:**

.. code-block:: bash

   # Install PyTorch first
   conda install -y pytorch==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
   pip install torch_geometric

   # Install PolyMon
   pip install polymon

**Train a model:**

.. code-block:: bash

   # Tabular model
   polymon train --labels Tg --model rf --feature-names rdkit2d --n-fold 5

   # Graph neural network
   polymon train --labels Rg --model gatv2 --n-fold 5 --num-epochs 2500

**Make predictions:**

.. code-block:: bash

   polymon predict --model-path model.pt --csv-path data.csv --smiles-column SMILES

For detailed examples, see the :doc:`examples` page.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
