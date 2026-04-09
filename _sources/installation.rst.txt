.. _installation:
.. index:: Installation

Installation
============

PolyMon requires Python 3.8+ and several dependencies including PyTorch and PyTorch Geometric.

Prerequisites
-------------

PyTorch Installation
~~~~~~~~~~~~~~~~~~~~

PolyMon requires ``torch>=2.2.2`` and ``torch_geometric>=2.5.3``. We recommend installing PyTorch first with CUDA support if available.

**For CUDA 11.8:**

.. code-block:: bash

    conda install -y pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
                     pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install torch_geometric
    pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.3.0+cu118.html

**For CPU only:**

.. code-block:: bash

    conda install -y pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 -c pytorch
    pip install torch_geometric
    pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.3.0+cpu.html

**For other CUDA versions**, please refer to the `PyTorch Geometric installation guide <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`__.

Install PolyMon
~~~~~~~~~~~~~~~

**Via pip (recommended):**

.. code-block:: bash

    # NumPy 2.x is not yet compatible with RDKit, so we install NumPy 1.x first
    pip install 'numpy<2'
    pip install polymon

**From source:**

.. code-block:: bash

    git clone https://github.com/fate1997/polymon.git
    cd polymon
    pip install 'numpy<2'
    pip install -e .

Verify Installation
-------------------

After installation, verify that PolyMon is working:

.. code-block:: bash

    polymon --help

You should see the help message with available commands: ``train``, ``rec``, and ``predict``.

Optional Dependencies
---------------------

For full functionality, the following packages are automatically installed:

- **RDKit**: Chemical informatics and molecular featurization
- **Mordred**: Additional molecular descriptors
- **XenonPy**: Element-based descriptors
- **XGBoost/LightGBM/CatBoost**: Gradient boosting frameworks
- **Optuna**: Hyperparameter optimization
- **PyTorch Lightning**: Training utilities

Troubleshooting
---------------

**Issue**: ``ImportError: No module named 'torch_geometric'``

**Solution**: Make sure you installed PyTorch Geometric after PyTorch. Reinstall if needed:

.. code-block:: bash

    pip install torch_geometric --force-reinstall

**Issue**: NumPy/AttributeError warnings when running polymon commands

**Solution**: RDKit is not yet compatible with NumPy 2.x. Downgrade NumPy:

.. code-block:: bash

    pip install 'numpy<2'

**Issue**: CUDA errors

**Solution**: Verify your PyTorch installation includes CUDA support:

.. code-block:: bash

    python -c "import torch; print(torch.cuda.is_available())"

If ``False``, reinstall PyTorch with CUDA support using the commands above.
