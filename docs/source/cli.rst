This document provides instructions on how to use the ``polymon`` command-line interface (CLI).

The ``polymon`` CLI has three main modes:

- `Train`_: Train a machine learning or deep learning model.
- `Merge`_: Merge two datasets.
- `Predict`_: Predict labels for a given dataset.

Train
============

This command is used to train a model.

**Usage:**

.. code-block:: bash

   polymon train [OPTIONS]

**Arguments:**

.. list-table::
   :widths: 50 10 20 50
   :header-rows: 1

   * - Supported Arguments in ``train``
     - Type
     - Default
     - Description
   * - ``--raw-csv``
     - str
     - ``database/database.csv``
     - Path to the raw csv file.
   * - ``--sources``
     - str (multiple)
     - ``['Kaggle']``
     - Sources to use for training.
   * - ``--tag``
     - str
     - ``debug``
     - Tag to use for training.
   * - ``--labels``
     - str (multiple)
     - **Required**
     - Labels to use for training.
   * - ``--feature-names``
     - str (multiple)
     - ``['rdkit2d']``
     - Feature names to use for training.
   * - ``--n-trials``
     - int
     - ``None``
     - Number of trials to run for hyperparameter optimization.
   * - ``--out-dir``
     - str
     - ``./results``
     - Path to the output directory.
   * - ``--hparams-from``
     - str
     - ``None``
     - Path to the hparams file. Allowed formats: .json, .pt, .pkl.
   * - ``--n-fold``
     - int
     - ``1``
     - Number of folds to use for cross-validation.
   * - ``--split-mode``
     - str
     - ``random``
     - Mode to split the data into training, validation, and test sets.
   * - ``--seed``
     - int
     - ``42``
     - Seed to use for training.
   * - ``--remove-hydrogens``
     - bool
     - ``False``
     - Whether to remove hydrogens from the molecules.
   * - ``--descriptors``
     - str (multiple)
     - ``None``
     - Descriptors to use for training. For ML models, this must be specified.
   * - ``--model``
     - str
     - ``rf``
     - Model to use for training.
   * - ``--hidden-dim``
     - int
     - ``32``
     - Hidden dimension of the model.
   * - ``--num-layers``
     - int
     - ``3``
     - Number of layers of the model.
   * - ``--batch-size``
     - int
     - ``128``
     - Batch size to use for training.
   * - ``--lr``
     - float
     - ``1e-3``
     - Learning rate to use for training.
   * - ``--num-epochs``
     - int
     - ``2500``
     - Number of epochs to use for training.
   * - ``--early-stopping-patience``
     - int
     - ``250``
     - Number of epochs to wait before early stopping.
   * - ``--device``
     - str
     - ``cuda``
     - Device to use for training.
   * - ``--run-production``
     - bool
     - ``False``
     - Whether to run the training in production mode, which means train:val:test splits will be forced to 0.95:0.05:0.0.
   * - ``--finetune``
     - bool
     - ``False``
     - Whether to finetune the model.
   * - ``--finetune-csv-path``
     - str
     - ``None``
     - Path to the csv file to finetune the model on.
   * - ``--pretrained-model``
     - str
     - ``None``
     - Path to the pretrained model.
   * - ``--n-estimator``
     - int
     - ``1``
     - Number of estimators to use for training.
   * - ``--additional-features``
     - str (multiple)
     - ``None``
     - Additional features to use for training.
   * - ``--skip-train``
     - bool
     - ``False``
     - Whether to skip the training step.
   * - ``--low-fidelity-model``
     - str
     - ``None``
     - Path to the low fidelity model.
   * - ``--estimator-name``
     - str
     - ``None``
     - Name of the estimator to give base predictions.
   * - ``--emb-model``
     - str
     - ``None``
     - Name of the embedding model for base graph embeddings.
   * - ``--ensemble-type``
     - str
     - ``voting``
     - Type of ensemble to use for training.
   * - ``--train-residual``
     - bool
     - ``False``
     - Whether to train the residual of the model.
   * - ``--normalizer-type``
     - str
     - ``normalizer``
     - Type of normalizer to use for training. Choices: ``normalizer``, ``log_normalizer``, ``none``.
   * - ``--augmentation``
     - bool
     - ``False``
     - Whether to use data augmentation.

Merge
============

This command is used to merge two datasets.

**Usage:**

.. code-block:: bash

   polymon merge [OPTIONS]

**Arguments:**

.. list-table::
   :widths: 30 10 20 50
   :header-rows: 1

   * - Supported Arguments in ``merge``
     - Type
     - Default
     - Description
   * - ``--sources``
     - str (multiple)
     - **Required**
     - Sources to merge.
   * - ``--label``
     - str
     - **Required**
     - Label to merge.
   * - ``--hparams-from``
     - str
     - **Required**
     - Path to the hparams file.
   * - ``--acquisition``
     - str
     - **Required**
     - Acquisition function to use for merging. Choices: ``epig``, ``uncertainty``, ``difference``.
   * - ``--sample-size``
     - int
     - ``20``
     - Sample size to use for merging.
   * - ``--uncertainty-threshold``
     - float
     - ``0.1``
     - Uncertainty threshold to use for merging.
   * - ``--difference-threshold``
     - float
     - ``0.1``
     - Difference threshold to use for merging.
   * - ``--target-size``
     - int
     - ``1000``
     - Target size to use for merging.
   * - ``--base-csv``
     - str
     - ``None``
     - Path to the base csv file.

Predict
============

This command is used to predict labels for a given dataset.

**Usage:**

.. code-block:: bash

   polymon predict [OPTIONS]

**Arguments:**

.. list-table::
   :widths: 30 10 20 50
   :header-rows: 1

   * - Argument
     - Type
     - Default
     - Description
   * - ``--model-path``
     - str
     - **Required**
     - Path to the model.
   * - ``--csv-path``
     - str
     - **Required**
     - Path to the csv file.
   * - ``--smiles-column``
     - str
     - **Required**
     - Name of the smiles column.
