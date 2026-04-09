Command-Line Interface
======================

The ``polymon`` CLI provides three main commands for training, prediction, and active learning:

.. contents::
   :local:
   :depth: 1

Train Command
-------------

The ``train`` command is used to train machine learning or deep learning models for polymer property prediction.

**Usage:**

.. code-block:: bash

   polymon train [OPTIONS]

**Required Arguments:**

``--labels``
   Target property/properties to predict. Choices: ``Tg``, ``FFV``, ``Density``, ``Rg``, ``Tc``.
   Multiple labels can be specified to train multiple models.

**Common Optional Arguments:**

``--raw-csv`` ``PATH``
   Path to the raw CSV file containing polymer data (default: ``database/database.csv``)

``--sources`` ``SOURCE [SOURCE ...]``
   Data sources to filter from the dataset (default: ``['Kaggle']``)
   Common sources: ``Kaggle``, ``PI1070``, ``PolyMetriX``, ``MAFA-exp``

``--model`` ``NAME``
   Model type to train:

   *Tabular models (use with ``--feature-names``):*
      - ``rf``: Random Forest
      - ``xgb``: XGBoost
      - ``lgbm``: LightGBM
      - ``catboost``: CatBoost
      - ``tabpfn``: TabPFN

   *Graph Neural Networks:*
      - ``gatv2``: Graph Attention Network v2
      - ``gin``: Graph Isomorphism Network
      - ``pna``: Principal Neighbourhood Aggregation
      - ``attentivefp``: Attentive Fingerprinting
      - ``dimenetpp``: DimeNet++
      - ``gps``: Graph with Positional Encoding
      - ``fastkan_gatv2``: KAN + GATv2
      - ``kan_gps``: KAN + GPS

``--feature-names`` ``FEATURE [FEATURE ...]``
   Feature names for tabular models (default: ``['rdkit2d']``)
   Choices: ``rdkit2d``, ``ecfp4``, ``mordred``, ``maccs``, ``xenonpy_desc``

``--n-fold`` ``INT``
   Number of folds for cross-validation (default: ``1``)
   Use ``5`` or ``10`` for reliable performance estimates

``--n-trials`` ``INT``
   Number of trials for hyperparameter optimization (default: ``None``)
   When specified, enables Optuna-based optimization

``--out-dir`` ``PATH``
   Directory to save training results (default: ``./results``)

``--tag`` ``NAME``
   Identifier for organizing this training run (default: ``debug``)

**Deep Learning Specific Arguments:**

``--descriptors`` ``FEATURE [FEATURE ...]``
   Additional descriptors to concatenate with graph features

``--hidden-dim`` ``INT``
   Hidden dimension for neural networks (default: ``32``)

``--num-layers`` ``INT``
   Number of layers in the neural network (default: ``3``)

``--num-epochs`` ``INT``
   Maximum number of training epochs (default: ``2500``)

``--lr`` ``FLOAT``
   Learning rate (default: ``1e-3``)

``--batch-size`` ``INT``
   Batch size for training (default: ``128``)

``--early-stopping-patience`` ``INT``
   Patience for early stopping (default: ``250``)

``--device`` ``NAME``
   Device for training: ``cuda`` or ``cpu`` (default: ``cuda``)

**Advanced Training Arguments:**

``--hparams-from`` ``PATH``
   Path to hyperparameters file (``.json``, ``.pt``, or ``.pkl``) to reuse from a previous run

``--run-production``
   Enable production mode (95:5 train:val split, no test set)

``--finetune``
   Enable fine-tuning of a pretrained model

``--pretrained-model`` ``PATH``
   Path to pretrained model for fine-tuning

``--finetune-csv-path`` ``PATH``
   Path to CSV file with fine-tuning data

``--train-residual``
   Train on residuals from a base estimator or low-fidelity model

``--estimator-name`` ``NAME``
   Name of empirical estimator for delta-learning (e.g., ``Density-IBM``, ``Rg-monomer``)

``--low-fidelity-model`` ``PATH``
   Path to low-fidelity model for residual learning

``--emb-model`` ``PATH``
   Path to embedding model for property knowledge transfer

``--n-estimator`` ``INT``
   Number of estimators for ensemble learning (default: ``1``, ``>1`` enables ensemble)

``--ensemble-type`` ``TYPE``
   Type of ensemble: ``voting``, ``bagging``, ``gradient_boosting``, ``snapshot``, ``soft_gradient_boosting``

``--split-mode`` ``MODE``
   Data splitting strategy: ``random``, ``source``, ``scaffold`` (default: ``random``)

``--normalizer-type`` ``TYPE``
   Label normalization: ``normalizer``, ``log_normalizer``, ``none`` (default: ``normalizer``)

``--augmentation``
   Enable data augmentation (oligomer building)

``--remove-hydrogens``
   Remove hydrogens from molecular graphs

``--seed`` ``INT``
   Random seed for reproducibility (default: ``42``)

**Examples:**

Train a Random Forest model with 5-fold cross-validation:

.. code-block:: bash

   polymon train --labels Tg --model rf --feature-names rdkit2d --n-fold 5

Train a GNN with hyperparameter optimization:

.. code-block:: bash

   polymon train --labels Rg --model gatv2 --n-trials 15 --n-fold 5 --num-epochs 2500

Fine-tune a pretrained model:

.. code-block:: bash

   polymon train --labels Density --model gatv2 --finetune \
       --pretrained-model ./results/gatv2/Density/model.pt \
       --finetune-csv-path ./experimental.csv

Recommend Command
-----------------

The ``rec`` command recommends molecules for active learning based on acquisition functions.

**Usage:**

.. code-block:: bash

   polymon rec [OPTIONS]

**Required Arguments:**

``--pool-csv`` ``PATH``
   Path to CSV file with candidate molecules (must contain a ``SMILES`` column)

``--trained-model`` ``PATH``
   Path to trained model file (``.pt`` or ``.pkl``)

**Optional Arguments:**

``--acquisition`` ``FUNCTION``
   Acquisition function: ``epig`` (expected improvement), ``uncertainty``, or ``random`` (default: ``uncertainty``)

``--model-type`` ``TYPE``
   Type of model: ``kfold`` or ``ensemble`` (default: ``kfold``)

``--sample-size`` ``INT``
   Number of molecules to recommend (default: ``100``)

``--save-path`` ``PATH``
   Path to save recommended molecules as CSV (default: ``None``)

**Examples:**

Select 20 most uncertain samples:

.. code-block:: bash

   polymon rec --pool-csv pool.csv --trained-model model.pt \
       --acquisition uncertainty --sample-size 20 --save-path recommended.csv

Predict Command
----------------

The ``predict`` command makes predictions on new polymer data using a trained model.

**Usage:**

.. code-block:: bash

   polymon predict [OPTIONS]

**Required Arguments:**

``--model-path`` ``PATH``
   Path to trained model file (``.pt`` or ``.pkl``)

``--csv-path`` ``PATH``
   Path to CSV file with molecules to predict

``--smiles-column`` ``NAME``
   Name of column containing SMILES strings

**Examples:**

Predict properties for new polymers:

.. code-block:: bash

   polymon predict --model-path ./results/gatv2/Tg/model.pt \
       --csv-path ./new_data.csv --smiles-column SMILES

Python API
----------

For more advanced usage, you can use the Python API directly:

.. code-block:: python

   from polymon.model.base import ModelWrapper

   # Load a trained model
   model = ModelWrapper.from_file('results/gatv2/Tg/model.pt')

   # Make predictions
   predictions = model.predict(['*C*', '*CC*', '*CCC*'])
   print(predictions)
