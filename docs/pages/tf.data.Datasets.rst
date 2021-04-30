Using tf.data.Datasets for loading
==================================

Using datasets for loading is a little complicated because of the precise nature of the feeding of the data to different devices. There are several helpers though to try and make this a bit easier.

.. contents:: The available modules are:
   :depth: 2
   :local:

DatasetGradientIMNN
-------------------

.. autoclass:: imnn.DatasetGradientIMNN
  :members:

  .. autoclasstoc::

DatasetNumericalGradientIMNN
----------------------------

.. autoclass:: imnn.DatasetNumericalGradientIMNN
  :members:

  .. autoclasstoc::

TFRecords (writer)
------------------

.. autoclass:: imnn.TFRecords
  :members:

  .. autoclasstoc::
