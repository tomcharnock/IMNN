Parent classes
==============

There are many different cases which might be desirable for fitting an IMNN, i.e. generating simulations on-the-fly with :func:`~imnn.SimulatorIMNN`, or with a fixed set of analytic gradients with :func:`~imnn.GradientIMNN` or performing numerical gradients for the derivatives with :func:`~imnn.NumericalGradientIMNN`, or when the datasets are very large (either in number of elements - ``n_s`` - or in shape - ``input_shape``) then the gradients can be manually aggregated using :func:`~imnn.AggregatedSimulatorIMNN`, :func:`imnn.AggregatedGradientIMNN` or :func:`~imnn.AggregatedNumericalGradientIMNN`. These are all wrappers around a base class :func:`~imnn.imnn._imnn._IMNN` and optionally an aggregation class :func:`~imnn.imnn._aggregated_imnn._AggregatedIMNN`. For completeness these are documented here.

.. contents:: The available modules are:
   :depth: 2
   :local:

Base class
----------

.. autoclass:: imnn.imnn._imnn._IMNN
   :members:

   .. autoclasstoc::

Aggregation of gradients
------------------------

.. autoclass:: imnn.imnn._aggregated_imnn._AggregatedIMNN
  :members:

  .. autoclasstoc::
