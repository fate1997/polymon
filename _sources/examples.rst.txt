.. _examples:
.. index:: Examples

Usage Examples
===============

This page provides comprehensive examples for using PolyMon, organized by use case.

.. contents::
   :local:
   :depth: 2

Basic Training
--------------

Train a Tabular Model (Random Forest)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train a Random Forest model using RDKit 2D descriptors:

.. code-block:: bash

   polymon train \
       --raw-csv ./database/database.csv \
       --sources Kaggle PI1070 PolyMetriX \
       --labels Tg \
       --feature-names rdkit2d \
       --model rf \
       --n-fold 5 \
       --out-dir ./results \
       --tag my_experiment

**Arguments explained:**

* ``--raw-csv``: Path to your dataset CSV file
* ``--sources``: Data sources to use (filters the dataset)
* ``--labels``: Target property to predict (``Tg``, ``FFV``, ``Density``, ``Rg``, ``Tc``)
* ``--feature-names``: Molecular descriptors to use
* ``--model``: Model type (``rf``, ``xgb``, ``lgbm``, ``catboost``, ``tabpfn``)
* ``--n-fold``: Number of cross-validation folds
* ``--out-dir``: Output directory for results
* ``--tag``: Identifier for this experiment

Train Multiple Properties at Once
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   polymon train \
       --raw-csv ./database/database.csv \
       --sources Kaggle PI1070 \
       --labels Tg FFV Density Rg Tc \
       --feature-names rdkit2d \
       --model rf \
       --n-fold 5 \
       --out-dir ./results

Using Pre-defined Hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train with hyperparameters from a previous run:

.. code-block:: bash

   polymon train \
       --raw-csv ./database/database.csv \
       --labels Rg \
       --feature-names ecfp4 \
       --model rf \
       --n-fold 5 \
       --hparams-from ./results/rf/rf-Rg-ecfp4.pkl \
       --out-dir ./results \
       --tag reuse_hparams

Hyperparameter Optimization
----------------------------

Optimize Tabular Model Hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Optuna to find the best hyperparameters:

.. code-block:: bash

   polymon train \
       --raw-csv ./database/database.csv \
       --sources Kaggle PI1070 PolyMetriX \
       --labels Tg Tc FFV Density Rg \
       --feature-names rdkit2d \
       --model rf \
       --n-fold 5 \
       --n-trials 15 \
       --out-dir ./results \
       --tag hparam_opt

* ``--n-trials``: Number of optimization trials (default: 15)

Optimize GNN Hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   polymon train \
       --raw-csv ./database/database.csv \
       --sources Kaggle PI1070 PolyMetriX \
       --labels Tc Tg FFV Density Rg \
       --model gatv2 \
       --n-trials 15 \
       --n-fold 5 \
       --out-dir ./results \
       --tag gnn_opt \
       --num-epochs 2500 \
       --early-stopping-patience 250

Neural Network Training
------------------------

Basic GNN Training
~~~~~~~~~~~~~~~~~~

Train a GATv2 model:

.. code-block:: bash

   polymon train \
       --raw-csv ./database/database.csv \
       --sources Kaggle PI1070 PolyMetriX \
       --labels Rg \
       --model gatv2 \
       --n-fold 5 \
       --num-epochs 2500 \
       --out-dir ./results

GNN with Additional Descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine graph features with molecular descriptors:

.. code-block:: bash

   polymon train \
       --raw-csv ./database/database.csv \
       --labels Density \
       --model gatv2 \
       --descriptors rdkit2d \
       --n-fold 5 \
       --num-epochs 2500 \
       --out-dir ./results

Advanced Training Strategies
-----------------------------

Multi-Fidelity Learning
~~~~~~~~~~~~~~~~~~~~~~~

**Fine-tune the Prediction Head**

Train on low-fidelity data (e.g., MD simulations), then fine-tune on high-fidelity experimental data:

.. code-block:: bash

   # Step 1: Train on low-fidelity data
   polymon train \
       --tag pretrained \
       --labels Density \
       --model gatv2 \
       --sources MD-data \
       --run-production \
       --num-epochs 2500

   # Step 2: Fine-tune on high-fidelity data
   polymon train \
       --tag finetune \
       --labels Density \
       --model gatv2 \
       --n-fold 5 \
       --finetune \
       --finetune-csv-path database/experimental.csv \
       --sources Experimental \
       --pretrained-model ./results/gatv2/Density/production/gatv2_Density.pt \
       --hparams-from ./results/gatv2/Density/hparams.json

**Label Residual Learning**

Train a model to predict the residual (difference) between low-fidelity predictions and ground truth:

.. code-block:: bash

   polymon train \
       --tag label_residual \
       --labels Density \
       --model gatv2 \
       --n-fold 5 \
       --train-residual \
       --sources Experimental \
       --low-fidelity-model ./results/gatv2/Density/production/gatv2_Density.pt \
       --hparams-from ./results/gatv2/Density/hparams.json

**Embedding Residual Learning**

Use pre-trained embeddings to guide learning:

.. code-block:: bash

   polymon train \
       --tag emb_residual \
       --labels Density \
       --model gatv2_embed_residual \
       --n-fold 5 \
       --sources Experimental \
       --emb-model ./results/gatv2/Density/production/gatv2_Density.pt \
       --hparams-from ./results/gatv2/Density/hparams.json

Delta-Learning (Property Transfer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Property Knowledge Transfer**

Use knowledge from one property to predict another:

.. code-block:: bash

   # Train on TC first
   polymon train --labels Tc --model gatv2 --run-production

   # Use TC embeddings to predict Density
   polymon train \
       --tag from_Tc \
       --labels Density \
       --model gatv2_embed_residual \
       --n-fold 5 \
       --sources PI1070 Kaggle PolyMetriX \
       --emb-model ./results/gatv2/Tc/production/gatv2_Tc.pt \
       --hparams-from ./results/gatv2/Density/hparams.json

**Empirical Equation Baseline**

Train on residuals from empirical equations:

.. code-block:: bash

   # Using IBM group contribution method as baseline
   polymon train \
       --tag ibm_residual \
       --labels Density \
       --model gatv2 \
       --n-fold 5 \
       --train-residual \
       --sources PI1070 Kaggle PolyMetriX \
       --estimator-name Density-IBM \
       --hparams-from ./results/gatv2/Density/hparams.json

Available empirical estimators:

* ``Density-IBM``: IBM group contribution method
* ``Density-Fedors``: Fedors group contribution method
* ``Density-vdw``: van der Waals volume method
* ``Rg-monomer``: Monomer-based Rg estimation

Ensemble Learning
~~~~~~~~~~~~~~~~~

**Voting Ensemble**

.. code-block:: bash

   polymon train \
       --raw-csv ./database/database.csv \
       --sources Kaggle PI1070 \
       --labels Rg \
       --model gatv2 \
       --hparams-from ./hparams.json \
       --n-estimators 10 \
       --ensemble-type voting \
       --tag voting_ensemble

**Bagging Ensemble**

.. code-block:: bash

   polymon train \
       --raw-csv ./database/database.csv \
       --sources Kaggle PI1070 \
       --labels Rg \
       --model gatv2 \
       --hparams-from ./hparams.json \
       --n-estimators 10 \
       --ensemble-type bagging \
       --tag bagging_ensemble

Periodic Graph Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For polymers with periodic boundary conditions:

.. code-block:: bash

   polymon train \
       --raw-csv ./database/database.csv \
       --sources Kaggle PI1070 PolyMetriX \
       --labels Tc Tg FFV Density Rg \
       --model gatv2 \
       --n-fold 5 \
       --n-trials 15 \
       --additional-features monomer periodic_bond \
       --tag periodic

Prediction
----------

Single Model Prediction
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   polymon predict \
       --model-path ./results/mlp/Rg/train/mlp_Rg.pt \
       --csv-path ./data/new_polymers.csv \
       --smiles-column SMILES

K-Fold Model Prediction
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   polymon predict \
       --model-path ./results/gatv2/Rg/train/gatv2_Rg-KFold.pt \
       --csv-path ./data/new_polymers.csv \
       --smiles-column SMILES

Ensemble Model Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   polymon predict \
       --model-path ./results/gatv2/Rg/ensemble/production/gatv2_Rg-ensemble.pt \
       --csv-path ./data/new_polymers.csv \
       --smiles-column SMILES

Python API for Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from polymon.model.base import ModelWrapper
   from polymon.model.ensemble import EnsembleModelWrapper

   # Single model or K-fold model
   model = ModelWrapper.from_file('results/mlp/Rg/train/mlp_Rg-KFold.pt')
   predictions = model.predict(['*C*', '*CCC*'])
   print(predictions)  # Shape: [N, 1] for single, [N, k] for k-fold

   # Ensemble model
   ensemble = EnsembleModelWrapper.from_file('results/gatv2/Rg/ensemble/gatv2_Rg-ensemble.pt')
   predictions = ensemble.predict(['*C*', '*CC*'])
   print(predictions)

Active Learning
---------------

Uncertainty Sampling
~~~~~~~~~~~~~~~~~~~~

Select the most uncertain samples for labeling:

.. code-block:: bash

   polymon rec \
       --pool-csv ./database/pool.csv \
       --trained-model ./results/mlp/Rg/train/mlp_Rg-KFold.pt \
       --acquisition uncertainty \
       --sample-size 20 \
       --save-path recommended.csv

Expected Improvement (EPIG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   polymon rec \
       --pool-csv ./database/pool.csv \
       --trained-model ./results/gatv2/Rg/train/gatv2_Rg-KFold.pt \
       --acquisition epig \
       --sample-size 50 \
       --save-path recommended.csv

Random Sampling
~~~~~~~~~~~~~~~

.. code-block:: bash

   polymon rec \
       --pool-csv ./database/pool.csv \
       --trained-model ./results/gatv2/Rg/train/gatv2_Rg-KFold.pt \
       --acquisition random \
       --sample-size 20 \
       --save-path recommended.csv

**Acquisition functions explained:**

* ``uncertainty``: Selects samples with highest prediction uncertainty
* ``epig``: Expected improvement in generalization
* ``random``: Random sampling (baseline)

Common Workflows
----------------

Full ML Pipeline
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Step 1: Hyperparameter optimization
   polymon train \
       --labels Tg \
       --model rf \
       --feature-names rdkit2d \
       --n-trials 20 \
       --n-fold 5 \
       --raw-csv ./database/database.csv

   # Step 2: Train final model with best hyperparameters
   polymon train \
       --labels Tg \
       --model rf \
       --feature-names rdkit2d \
       --hparams-from ./results/rf/Tg/hparams_opt/hparams.json \
       --run-production

   # Step 3: Make predictions on new data
   polymon predict \
       --model-path ./results/rf/Tg/production/rf_Tg.pt \
       --csv-path ./new_data.csv \
       --smiles-column SMILES

Active Learning Loop
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Initial training
   polymon train --labels Rg --model gatv2 --n-fold 5 --run-production

   # Iteratively select and label new samples
   for i in {1..5}; do
       # Recommend new samples
       polymon rec \
           --pool-csv pool.csv \
           --trained-model ./results/gatv2/Rg/production/gatv2_Rg.pt \
           --acquisition uncertainty \
           --sample-size 20 \
           --save-path batch_$i.csv

       # Add labeled data to training set and retrain
       # ... (label the samples externally)
       polymon train --labels Rg --model gatv2 --n-fold 5 --run-production
   done

Model Comparison
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Train multiple models for comparison
   for model in rf xgb lgbm catboost gatv2 gin pna; do
       polymon train \
           --labels Tg \
           --model $model \
           --n-fold 5 \
           --n-trials 15 \
           --tag comparison
   done

Tips and Best Practices
------------------------

1. **Start with tabular models**: They train faster and provide good baselines
2. **Use cross-validation**: K-fold CV (``--n-fold 5``) gives more reliable estimates
3. **Optimize hyperparameters**: Use ``--n-trials`` to automatically find good hyperparameters
4. **Monitor training**: Check logs in the output directory for training progress
5. **Use appropriate descriptors**: ``rdkit2d`` works well for most properties; ``ecfp4`` for structure-activity relationships
6. **Consider data sources**: Use ``--sources`` to filter data by experimental/computational source
