polymon.model
===========================

ModelWrapper
------------------------------------------

.. autoclass:: polymon.model.base.ModelWrapper
   :members:
   :show-inheritance:
   :undoc-members:


KFoldModel
------------------------------------------

.. autoclass:: polymon.model.base.KFoldModel
   :members:
   :show-inheritance:
   :undoc-members:

LinearEnsembleRegressor
------------------------------------------

.. autoclass:: polymon.model.ensemble.LinearEnsembleRegressor
   :members:
   :show-inheritance:
   :undoc-members:

EnsembleModelWrapper
------------------------------------------

.. autoclass:: polymon.model.ensemble.EnsembleModelWrapper
   :members:
   :show-inheritance:
   :undoc-members:

Models
----------

.. automodule:: polymon.model
   :members:
   :show-inheritance:
   :undoc-members:

.. list-table:: Available Models in ``polymon.model``
   :header-rows: 1

   * - Model Type
     - Class Name
     - Description
   * - ``gatv2``
     - :ref:`GATv2 <model-gatv2>`
     - `Graph Attention Network v2 <https://arxiv.org/abs/2105.14491>`_
   * - ``attentivefp``
     - :ref:`AttentiveFPWrapper <model-attentivefp>`
     - `AttentiveFP <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_
   * - ``dimenetpp``
     - :ref:`DimeNetPP <model-dimenetpp>`
     - `DimeNet++ <https://arxiv.org/abs/2011.14115>`_
   * - ``gatv2vn``
     - :ref:`GATv2VirtualNode <model-gatv2vn>`
     - GATv2 with virtual node
   * - ``gin``
     - :ref:`GIN <model-gin>`
     - `Graph Isomorphism Network <https://arxiv.org/abs/1810.00826>`_
   * - ``pna``
     - :ref:`PNA <model-pna>`
     - `Principal Neighbourhood Aggregation <https://arxiv.org/abs/2004.05718>`_
   * - ``gvp``
     - :ref:`GVPModel <model-gvp>`
     - `Geometric Vector Perceptron <https://arxiv.org/abs/2009.01411>`_
   * - ``gatv2chainreadout``
     - :ref:`GATv2ChainReadout <model-gatv2chainreadout>`
     - GATv2 with chain readout
   * - ``gt``
     - :ref:`GraphTransformer <model-gt>`
     - `Graph Transformer <https://arxiv.org/abs/2009.03509>`_
   * - ``kan_gatv2``
     - :ref:`KAN_GATv2 <model-kan_gatv2>`
     - KAN-augmented GATv2
   * - ``gps``
     - :ref:`GraphGPS <model-gps>`
     - `Graph GPS <https://arxiv.org/abs/2205.12454>`_
   * - ``kan_gps``
     - :ref:`KAN_GPS <model-kan_gps>`
     - KAN-augmented GraphGPS
   * - ``fastkan``
     - :ref:`FastKANWrapper <model-fastkan>`
     - `Fast KAN for descriptors <https://github.com/ZiyaoLi/fast-kan>`_
   * - ``efficientkan``
     - :ref:`EfficientKANWrapper <model-efficientkan>`
     - `Efficient KAN for descriptors <https://github.com/Blealtan/efficient-kan>`_
   * - ``fourierkan``
     - :ref:`FourierKANWrapper <model-fourierkan>`
     - `Fourier KAN for descriptors <https://github.com/GistNoesis/FourierKAN>`_
   * - ``fastkan_gatv2``
     - :ref:`FastKAN_GATv2 <model-fastkan_gatv2>`
     - FastKAN-augmented GATv2
   * - ``gatv2_lineevo``
     - :ref:`GATv2LineEvo <model-gatv2_lineevo>`
     - `GATv2 with line evolution <https://pubs.acs.org/doi/10.1021/acs.jcim.3c00059>`_
   * - ``gatv2_sage``
     - :ref:`GATv2SAGE <model-gatv2_sage>`
     - GATv2 with SAGE aggregation
   * - ``gatv2_source``
     - :ref:`GATv2_Source <model-gatv2_source>`
     - GATv2 for multi-fidelity/source
   * - ``gatv2_pe``
     - :ref:`GATv2_PE <model-gatv2_pe>`
     - GATv2 with position encoding
   * - ``gatv2_embed_residual``
     - :ref:`GATv2EmbedResidual <model-gatv2_embed_residual>`
     - GATv2 with embedding residuals
   * - ``kan_gin``
     - :ref:`KAN_GIN <model-kan_gin>`
     - KAN-augmented GIN
   * - ``fastkan_gin``
     - :ref:`FastKAN_GIN <model-fastkan_gin>`
     - FastKAN-augmented GIN
   * - ``kan_gcn``
     - :ref:`KAN_GCN <model-kan_gcn>`
     - KAN-augmented GCN
   * - ``dmpnn``
     - :ref:`DMPNN <model-dmpnn>`
     - `Directed Message Passing Neural Network <https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237>`_
   * - ``kan_dmpnn``
     - :ref:`KAN_DMPNN <model-kan_dmpnn>`
     - KAN-augmented DMPNN

.. note::
   The ``model_type`` string is used as the key in configuration and when calling :func:`polymon.model.build_model`.

.. toctree::
   :maxdepth: 1
   :hidden:

.. _model-gatv2:


gatv2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: polymon.model.gnn.GATv2
   :members: forward, get_embeddings
   :show-inheritance:
   :undoc-members:

.. _model-attentivefp:

attentivefp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.gnn.AttentiveFPWrapper
   :members:
   :show-inheritance:
   :undoc-members:

.. _model-dimenetpp:

dimenetpp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.gnn.DimeNetPP
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-gatv2vn:

gatv2vn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.gnn.GATv2VirtualNode
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-gin:

gin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.gnn.GIN
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-pna:

pna
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.gnn.PNA
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-gvp:

gvp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.gvp.GVPModel
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-gatv2chainreadout:

gatv2chainreadout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.gatv2.gat_chain_readout.GATv2ChainReadout
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-gt:

gt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.gnn.GraphTransformer
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-kan_gatv2:

kan_gatv2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.gatv2.kan_gatv2.KAN_GATv2
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-gps:

gps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.gps.gps.GraphGPS
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-kan_gps:

kan_gps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.gps.gps.KAN_GPS
   :members: forward
   :show-inheritance:

.. _model-fastkan:

fastkan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.kan.fast_kan.FastKANWrapper
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-efficientkan:

efficientkan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.kan.efficient_kan.EfficientKANWrapper
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-fourierkan:

fourierkan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.kan.fourier_kan.FourierKANWrapper
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-fastkan_gatv2:

fastkan_gatv2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.gatv2.fastkan_gatv2.FastKAN_GATv2
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-gatv2_lineevo:

gatv2_lineevo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.gatv2.lineevo.GATv2LineEvo
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-gatv2_sage:

gatv2_sage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.gatv2.gatv2_sage.GATv2SAGE
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-gatv2_source:

gatv2_source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.gatv2.multi_fidelity.GATv2_Source
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-gatv2_pe:

gatv2_pe
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.gatv2.position_encoding.GATv2_PE
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-gatv2_embed_residual:

gatv2_embed_residual
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.gatv2.embed_residual.GATv2EmbedResidual
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-kan_gin:

kan_gin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.kan.gin.KAN_GIN
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-fastkan_gin:

fastkan_gin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.kan.gin.FastKAN_GIN
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-kan_gcn:

kan_gcn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.kan.gcn.KAN_GCN
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-dmpnn:

dmpnn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.dmpnn.DMPNN
   :members: forward
   :show-inheritance:
   :undoc-members:

.. _model-kan_dmpnn:

kan_dmpnn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: polymon.model.kan.dmpnn.KAN_DMPNN
   :members: forward
   :show-inheritance:
   :undoc-members: