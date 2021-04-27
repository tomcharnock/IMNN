Likelihood-free inference modules
=================================

To be able to make use of the outputs of the IMNN it is best to use
likelihood-free inference (this is true when using any sort of neural network,
but anyway - I digress). Any lfi or sbi module should be able to use these
outputs where the network (once trained) can be added to the simulation as a
compression step. For ease of use there are some jax implementations of a few
different approximate Bayesian computation algorithms. These also come with a
plotting function. Examples are available
`here <examples/lfi/mean_and_variance/mean_and_variance.html>`_. Note that
the module is not fully tested, but seems to be functionally working. There are
more functions available with ``pip install imnn-tf`` where there is a previous
implementation with more general versions of the same functions and checkout
also `github.com/justinalsing/pydelfi
<https://github.com/justinalsing/pydelfi/tree/tf2-tom>`_

.. contents:: The available modules are:
   :depth: 2
   :local:


LikelihoodFreeInference
-----------------------

.. autoclass:: imnn.lfi.LikelihoodFreeInference
  :members:

  .. autoclasstoc::

GaussianApproximation
---------------------

.. autoclass:: imnn.lfi.GaussianApproximation
  :members:

  .. autoclasstoc::

ApproximateBayesianComputation
------------------------------

.. autoclass:: imnn.lfi.ApproximateBayesianComputation
  :members:

  .. autoclasstoc::

PopulationMonteCarlo
--------------------

.. autoclass:: imnn.lfi.PopulationMonteCarlo
  :members:

  .. autoclasstoc::
