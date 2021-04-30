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

Doing likelihood-free inference
-------------------------------

With a trained IMNN it is possible to get an estimate of some data using

.. code-block:: python

    estimate = IMNN.get_estimate(target_data)

Along with the Fisher information from the network, we can use this to make a Gaussian approximation of the posterior under the assumption that the fiducial parameter values used to calculate the Fisher information coincide with the parameter estimate. This posterior can be calculated using the ``imnn.lfi`` module. For all of the available functions in the lfi module a `TensorFlow Probability <https://www.tensorflow.org/probability/>`_-like distribution is used for the prior, e.g. for a uniform distribution for two parameters between 0 and 10 each we could write

.. code-block:: python

    import tensorflow_probability
    tfp = tensorflow_probability.substrates.jax

    prior = tfp.distributions.Blockwise(
    [tfp.distributions.Uniform(low=low, high=high)
     for low, high in zip([0., 0.], [10., 10.])])

     prior.low = np.array([0., 0.])
     prior.high = np.array([10., 10.])

We set the values of ``prior.low`` and ``prior.high`` since they are used to define the plotting ranges. Note that ``prior.event_shape`` should be equal to ``n_params``, i.e. the number of parameters in the physical model.

Gaussian Approximation
______________________

The ``GaussianApproximation`` simply evaluates a multivariate Gaussian with mean at ``estimate`` and covariance given by ``np.linalg.inv(IMNN.F)`` on a grid defined by the prior ranges.

.. code-block:: python

    GA = imnn.lfi.GaussianApproximation(
      parameter_estimates=estimate,
      invF=np.linalg.inv(IMNN.F),
      prior=prior,
      gridsize=100)

And corner plots of the Gaussian approximation can be made using

.. code-block:: python

    GA.marginal_plot(
      ax=None,                   # Axes object to plot (constructs new if None)
      ranges=None,               # Ranges for each parameter (None=preset)
      marginals=None,            # Marginal distributions to plot (None=preset)
      known=None,                # Plots known parameter values if not None
      label=None,                # Adds legend element if not None
      axis_labels=None,          # Adds labels to the axes if not None
      levels=None,               # Plot specified approx significance contours
      linestyle="solid",         # Linestyle for the contours
      colours=None,              # Colour for the contours
      target=None,               # If multiple target data, which index to plot
      format=False,              # Whether to set up the plot decoration
      ncol=2,                    # Number of columns in the legend
      bbox_to_anchor=(1.0, 1.0)) # Where to place the legend

Note that this approximation shouldn't be necessarily a good estimate of the true posterior, for that actual LFI methods should be used.

Approximate Bayesian Computation
________________________________

To generate simulations and accept or reject these simulations based on a distance based criterion from some target data we can use

.. code-block:: python

    ABC = imnn.lfi.ApproximateBayesianComputation(
      target_data=target_data,
      prior=prior,
      simulator=simulator,
      compressor=IMNN.get_estimate,
      gridsize=100,
      F=IMNN.F,
      distance_measure=None)

This takes in the target data and compresses it using the provided compressor (like ``IMNN.get_estimate``). The Fisher information matrix can be provided to rescale the parameter directions to make meaningful distance measurements as long as summaries are parameter estimates. If a different distance measure is better for the specific problem this can be passed as a function. Note that if simulations have already been done for the ABC and only the plotting and the acceptance and rejection is needed then ``simulator`` can be set to ``None``. The ABC can actually be run by calling the module

.. code-block:: python

    parameters, summaries, distances = ABC(
        ϵ=None,             # The size of the epsilon ball to accept summaries
        rng=None,           # Random number generator for params and simulation
        n_samples=None,     # The number of samples to run (at one time)
        parameters=None,    # Values of parameters with premade compressed sims
        summaries=None,     # Premade compressed sims to avoid running new sims
        min_accepted=None,  # Num of required sims in epsilon ball (iterative)
        max_iterations=10,  # Max num of iterations to try and get min_accepted
        smoothing=None,     # Amount of smoothing on the histogrammed marginals
        replace=False)      # Whether to remove all previous run summaries

If not run and no premade simulations have been made then ``n_samples`` and ``rng`` must be passed. Note that if ``ϵ`` is too large then the accepted samples should not be considered to be drawn from the posterior but rather some partially marginalised part of the joint distribution of summaries and parameters, and hence it can be very misleading - ``ϵ`` should be a small as possible! Like with the ``GaussianApproximation`` there is a ``ABC.marginal_plot(...)`` but the parameter samples can also be plotted as a scatter plot on the corner plot

.. code-block:: python

    ABC.scatter_plot(
        ax=None,                 # Axes object to plot (constructs new if None)
        ranges=None,             # Ranges for each parameter (None=preset)
        points=None,             # Parameter values to scatter (None=preset)
        label=None,              # Adds legend element if not None
        axis_labels=None,        # Adds labels to the axes if not None
        colours=None,            # Colour for the scatter points (and hists)
        hist=True,               # Whether to plot 1D histograms of points
        s=5,                     # Marker size for points
        alpha=1.,                # Amount of transparency for the points
        figsize=(10, 10),        # Size of the figure if not premade
        linestyle="solid",       # Linestyle for the histograms
        target=None,             # If multiple target data, which index to plot
        ncol=2,                  # Number of columns in the legend
        bbox_to_anchor=(0., 1.)) # Where to place the legend

And the summaries can also be plotted on a corner plot with exactly the same parameters as ``scatter_plot`` (apart from ``gridsize`` being added) but if ``points`` is left ``None`` then ``ABC.summaries.accepted`` is used instead and the ranges calculated from these values. If points is supplied but ``ranges`` is None then the ranges are calculated from the minimum and maximum values of the points are used as the edges.

.. code-block:: python

    ABC.scatter_summaries(
        ax=None,
        ranges=None,
        points=None,
        label=None,
        axis_labels=None,
        colours=None,
        hist=True,
        s=5,
        alpha=1.,
        figsize=(10, 10),
        linestyle="solid",
        gridsize=100,
        target=None,
        format=False,
        ncol=2,
        bbox_to_anchor=(0.0, 1.0))

Population Monte Carlo
______________________

To more efficiently accept samples than using a simple ABC where samples are drawn from the prior, the `PopulationMonteCarlo`_ provides a JAX accelerated iterative acceptance and rejection scheme where each iteration the population of samples with summaries closest to the summary of the desired target defines a new proposal distribution to force a fixed population to converge towards the posterior without setting an explicit size for the epsilon ball of normal ABC. The PMC is stopped using a criterion on the number of accepted proposals compared to the number of total draws from the proposal. When this gets very small it suggests the distribution is stationary and that the proposal has been reached. It works similarly to ``ApproximateBayesianComputation``.

.. code-block:: python

    PMC = imnn.lfi.PopulationMonteCarlo(
      target_data=target_data,
      prior=prior,
      simulator=simulator,
      compressor=IMNN.get_estimate,
      gridsize=100,
      F=IMNN.F,
      distance_measure=None)

And it can be run using

.. code-block:: python

    parameters, summaries, distances = PMC(
        rng,                     # Random number generator for params and sims
        n_points,                # Number of points from the final distribution
        percentile=None,         # Percentage of points making the population
        acceptance_ratio=0.1,    # Fraction of accepted draws vs total draws
        max_iteration=10,        # Maximum number of iterations of the PMC
        max_acceptance=1,        # Maximum number of tries to get an accepted
        max_samples=int(1e5),    # Maximum number of attempts to get parameter
        n_initial_points=None,   # Number of points in the initial ABC step
        n_parallel_simulations=None, # Number of simulations to do in parallel
        proposed=None,           # Prerun parameter values for the initial ABC
        summaries=None,          # Premade compressed simulations for ABC
        distances=None,          # Precalculated distances for the initial ABC
        smoothing=None,          # Amount of smoothing on histogrammed marginal
        replace=False)           # Whether to remove all previous run summaries

The same plotting functions as `ApproximateBayesianComputation`_ are also available in the PMC

Note
....

There seems to be a bug in PopulationMonteCarlo and the parallel sampler is turned off


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
