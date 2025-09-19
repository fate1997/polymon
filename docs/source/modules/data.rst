polymon.data
====================

.. contents:: Contents
    :local:


dataset
---------------------------

.. automodule:: polymon.data.dataset
   :members:
   :show-inheritance:
   :undoc-members:

dedup
-------------------------

.. automodule:: polymon.data.dedup
   :members:
   :show-inheritance:
   :undoc-members:

featurizer
------------------------------

.. list-table:: **Summary of Supported Featurizers**
   :header-rows: 1

   * - Name
     - Featurizer
     - Attributes in :class:`Polymer`
   * - x
     - ``AtomFeaturizer``
     - :obj:`x`
   * - edge
     - ``BondFeaturizer``
     - :obj:`edge_index` and :obj:`edge_attr`
   * - pos
     - ``PosFeaturizer``
     - :obj:`pos`
   * - z
     - ``AtomNumFeaturizer``
     - :obj:`z`
   * - relative_position
     - ``RelativePositionFeaturizer``
     - :obj:`relative_position`
   * - seq
     - ``SeqFeaturizer``
     - :obj:`seq` and :obj:`seq_len`
   * - desc
     - ``DescFeaturizer``
     - :obj:`descriptors`
   * - monomer
     - ``RDMolPreprocessor``
     - :obj:`monomer`

.. list-table:: **Available Feature Names**
   :header-rows: 1

   * - Feature Name
     - Available Features
     - Description
   * - x
     - 
     - Node features (default set, :obj:`polymon.setting.DEFAULT_ATOM_FEATURES`).
   * - 
     - xenonpy_atom
     - XenonPy atom features
   * - 
     - cgcnn
     - CGCNN atom features
   * - 
     - source
     - Source features
   * - z
     - 
     - Atomic numbers as integers
   * - edge
     - 
     - Default edge features (:obj:`bond`)
   * - 
     - bond
     - Chemical bond features
   * - 
     - fully_connected_edges
     - Fully connected edge indices
   * - 
     - periodic_bond
     - Add bonds between attachment points to the chemical bonds
   * - 
     - virtual_bond
     - Add virtual bonds between virtual node and each other node
   * - pos
     - 
     - 3D coordinates
   * - relative_position
     - 
     - The relative position of the atom to the nearest attachment
   * - seq
     - 
     - SMILES sequence features
   * - desc
     - 
     - Default descriptor features (:obj:`rdkit2d`)
   * - 
     - rdkit2d
     - RDKit 2D descriptors
   * - 
     - ecfp4
     - ECFP4 fingerprints
   * - 
     - rdkit3d
     - RDKit 3D descriptors
   * - 
     - mordred
     - Mordred descriptors
   * - 
     - maccs
     - MACCS keys
   * - 
     - oligomer_rdkit2d
     - RDKit 2D descriptors of the oligomer
   * - 
     - oligomer_mordred
     - Mordred descriptors of the oligomer
   * - 
     - oligomer_ecfp4
     - ECFP4 fingerprints of the oligomer
   * - 
     - xenonpy_desc
     - XenonPy composition descriptors
   * - 
     - mordred3d
     - Mordred 3D descriptors
   * - 
     - fedors_density
     - Estimated density from Fedors method
   * - monomer
     - 
     - Preprocess molecule as monomer (remove attachment points)
   * - polycl
     - 
     - Pretrained Polycl embeddings
   * - polybert
     - 
     - Pretrained PolyBERT embeddings
   * - gaff2_mod
     - 
     - Pretrained GAFF2 descriptors



.. autoclass:: polymon.data.featurizer.AtomFeaturizer
   :members: __call__

.. autoclass:: polymon.data.featurizer.BondFeaturizer
   :members: __call__

.. autoclass:: polymon.data.featurizer.PosFeaturizer
   :members: __call__

.. autoclass:: polymon.data.featurizer.AtomNumFeaturizer
   :members: __call__

.. autoclass:: polymon.data.featurizer.RelativePositionFeaturizer
   :members: __call__

.. autoclass:: polymon.data.featurizer.SeqFeaturizer
   :members: __call__

.. autoclass:: polymon.data.featurizer.DescFeaturizer
   :members: __call__

.. autoclass:: polymon.data.featurizer.RDMolPreprocessor
   :members: monomer

polymer
---------------------------

.. automodule:: polymon.data.polymer
   :members:
   :show-inheritance:
   :undoc-members:
