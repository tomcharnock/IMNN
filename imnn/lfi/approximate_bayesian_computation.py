import jax
import jax.numpy as np
from imnn.lfi import LikelihoodFreeInference
from imnn.utils import container
from scipy.ndimage import gaussian_filter
from imnn.utils.utils import _check_input


class ApproximateBayesianComputation(LikelihoodFreeInference):
    """Approximate Bayesian computation

    Approximate Bayesian computation involves drawing parameter values from a
    prior, :math:`\\boldsymbol{\\theta}_i\\leftarrow p(\\boldsymbol{\\theta})`,
    and generating simulations at this value using a model,
    :math:`{\\bf d}_i = f(\\boldsymbol{\\theta}_i)`, which can then be
    compressed down to some informative summary,
    :math:`{\\bf x}_i =g({\\bf d}_i)`. By compressing some observed target to
    get :math:`{\\bf x}^\\textrm{obs} =g({\\bf d}^\\textrm{obs})` we can
    simulate from the prior distribution of possible parameter values to create
    a set of summaries :math:`\\{{\\bf x}_i|i\\in[1, n_\\textrm{sims}]\\}`. The
    approximate posterior distribution is then obtained as the samples of
    parameter values whose summaries of simulations have an infinitely small
    difference from the summary of the target observation:

    .. math::

        p(\\boldsymbol{\\theta}|{\\bf d}^\\textrm{obs})\\approx
        \\{\\boldsymbol{\\theta}_i|
        i~\\textrm{where}~\\lim_{\\epsilon\\to0}
        D(g(f(\\boldsymbol{\\theta}_i), g({\\bf d}^\\textrm{obs}))<\\epsilon\\}

    Note that the distance measure :math:`D(x, y)` is somewhat ambiguous, but
    in the infinitessimal limit will be more or less flat. Sensible choices for
    this distance measure can be made, i.e. for parameter estimates a Fisher
    scaled squared distance, but it is not wholly trivial in all cases.

    This class contains several methods for making simulations, compressing
    them and doing distance based acceptance and rejection of the parameters
    used to make the simulations from the compressed summaries of target data.
    In particular either a single set of simulations can be run and the
    distance calculated, simulations can be run until a desired number of
    accepted points within achosen distance are obtained or if there are
    already a set of run simulations, different distance values can be chosen
    for acceptance and rejection of points. There are also several plotting
    methods for corner plots.

    Note that if the summaries of the data are not parameter estimates then the
    distance measure must be optimal for the summary space to make sense.

    Parameters
    ----------
    target_summaries : float(n_targets, n_summaries)
        The compressed data to be infered
    n_summaries : int
        The size of the output of the compressor for a single simulation
    n_targets : int
        The number of different targets to be infered
    F : float(n_params, n_params)
        The Fisher information matrix for rescaling distance measure
    parameters : utils container
        Holds accepted and rejected parameters and some summaries of results
    summaries : utils container
        Holds accepted and rejected summaries and some summaries of results
    distances : utils container
        Holds accepted and rejected distances of summaries to targets and
        some summaries of results

    Methods
    -------
    simulator
        A simulator which takes in an array of parameters and returns
        simulations made at those parameter values
    compressor
        A function which takes in an array of simulations and compresses them.
        If no compression is needed this can be an identity function
    distance_measure:
        Either ``F_distance`` or ``euclidean_distance`` depending on inputs
    """
    def __init__(self, target_data, prior, simulator, compressor,
                 gridsize=100, F=None, distance_measure=None):
        """Constructor method

        Parameters
        ----------
        target_data: float(n_targets, input_data)
            Data (or batch of data) to infer parameter values at
        prior: fn
            A prior distribution which can be evaluated and sampled from
            (should also contain a ``low`` and a ``high`` attribute with
            appropriate ranges)
        simulator: fn
            A function which takes in a batch of parameter values (and random
            seeds) and returns a simulation made at each parameter value
        compressor: fn
            A function which takes a batch of simulations and returns their
            compressed summaries for each simulation (can be identity function
            for no compression)
        gridsize : int or list, default=100
            The number of grid points to evaluate the marginal distribution on
            for every parameter (int) or each parameter (list)
        F : list or float({n_targets}, n_params, n_params) or None
            The Fisher information matrix to rescale parameter directions for
            measuring the distance. If None, this is set to the identity matrix
        distance_measure : fn or None
            A distance measuring function between a batch of summaries and a
            single set of summaries for a target. If None, ``F_distance``
            (Euclidean scaled by the Fisher information) is used. This is only
        """
        super().__init__(
            prior=prior,
            gridsize=gridsize)
        self.simulator = simulator
        self.compressor = compressor
        if len(target_data.shape) == 1:
            target_data = np.expand_dims(target_data, 0)
        self.target_summaries = self.compressor(target_data)
        self.n_summaries = self.target_summaries.shape[-1]
        self.n_targets = self.target_summaries.shape[0]
        if F is not None:
            if self.n_summaries != self.n_params:
                raise ValueError(
                    "If using the Fisher information to scale the distance " +
                    "then the compressor must return parameter estimates. " +
                    "The compressor returns summaries with shape " +
                    f"{(self.n_summaries,)}, but should return estamites " +
                    f"with shape {(self.n_params)} if the prior is correct.")
            if isinstance(F, list):
                self.F = np.stack([
                    _check_input(f, (self.n_params, self.n_params), "F")
                    for f in F], 0)
            else:
                if F.shape == (self.n_params, self.n_params):
                    self.F = np.expand_dims(F, 0)
                    if self.n_targets > 1:
                        self.F = np.repeat(self.F, self.n_targets)
                else:
                    self.F = _check_input(
                        F, (self.n_targets, self.n_params, self.n_params), "F")
            self.distance_measure = self.F_distance

        if distance_measure is not None:
            self.distance_measure = distance_measure
        elif F is None:
            self.distance_measure = self.euclidean_distance
            self.F = np.zeros(self.n_targets)

        self.parameters = container()
        self.parameters.n_params = self.n_params
        self.summaries = container()
        self.summaries.n_summaries = self.n_summaries
        self.distances = container()
        self.distances.n_targets = self.n_targets

    def __call__(self, ϵ=None, rng=None, n_samples=None, parameters=None,
                 summaries=None, min_accepted=None, max_iterations=10,
                 smoothing=None, replace=False):
        """Run the ABC

        Parameters
        ----------
        ϵ : float or float(n_targets) or None, default=None
            The acceptance distance between summaries from simulations and the
            summary of the target data. A different epsilon can be passed for
            each target. If None (and `rng` and `n_samples` are not None) then
            samples are obtained but no acceptance and rejection takes place
        rng : int(2,) or None, default=None
            A random number generator for generating simulations and randomly
            drawing parameter values from the prior. Can be None if passing
            `parameters` and `summaries` to set prerun simulations or if `ϵ`
            is not None for resetting the accepted parameters and summaries
        n_samples : int or None, default=None
            Number of simulations to make (or how many to make each iteration
            if `min_accepted` is not None). Can be None if passing `parameters`
            and `summaries` to set prerun simulations or if `ϵ` is not None for
            resetting the accepted parameters and summaries
        parameters : float(any, n_params) or None, default=None
            The pre-obtained parameter values to set to the class (`summaries`
            must not be None)
        summaries : float(any, n_summaries) or None, default=None
            The summaries of prerun simulations (at parameter values
            corresponding to the values in `parameters`) to set the class
            attributes
        min_accepted : int or None, default=None
            A minimum number of accepted samples desired within a chosen `ϵ`
            value. If None and `ϵ`, `rng`, `n_samples` are not None then just
            `n_samples` will be run
        max_iterations : int, default=10
            A maximum number of while loop iterations to run when getting a
            minimum number of accepted samples
        smoothing : float or None, default=None
            A Gaussian smoothing for the marginal distributions
        replace : bool, default=False
            Whether to replace the summaries, parameters and distances already
            obtained when running again

        Returns
        -------
        parameters container:
            All parameters with accepted and rejected attributes
        summaries container:
            All summaries with accepted and rejected attributes
        distances container:
            All distances with accepted and rejected attributes

        Todo
        ----
        type checking and pytests
        """
        if ((self.parameters.all is None)
                and ((rng is None) or (n_samples is None))
                and ((parameters is None) or (summaries is None))):
            raise ValueError(
                "No samples currently available. If running simulations " +
                "`rng` must be a JAX prng and `n_samples` an integer number " +
                "of samples. If using premade summaries `parameters` and " +
                "`summaries` must be numpy arrays.")
        if ((min_accepted is not None) and (ϵ is None)):
            raise ValueError("`ϵ` must be passed if passing `min_accepted`")
        if (rng is not None) and (n_samples is not None):
            if ((ϵ is not None) and (min_accepted is not None)):
                self.set_samples(
                    *self.get_min_accepted(
                        rng=rng, ϵ=ϵ, accepted=min_accepted,
                        n_simulations=n_samples,
                        max_iterations=max_iterations),
                    replace=True)
            else:
                self.set_samples(
                    *self.get_samples(rng=rng, n_samples=n_samples),
                    replace=replace)
        if (parameters is not None) and (summaries is not None):
            self.set_samples(parameters, summaries, replace=replace)
        if ϵ is not None:
            self.set_accepted(ϵ, smoothing=smoothing)
        return self.parameters, self.distances, self.summaries

    def set_samples(self, parameters, summaries, distances=None,
                    replace=False):
        """Fills containers and calculates distances

        All parameters and summaries of simulations made at those parameters
        (and their distance from the summary of the observed data) are kept in
        containers for ease of use. This function sets the ``all`` attribute of
        these containers with the passed summaries and parameters (and
        distances if provided or it calculates the distances). These are
        concatenated to existing values unless ``replace = True`` in which the
        existing values are removed and overwritten with the passed summaries,
        parameters and distances.

        Parameters
        ----------
        parameters : float(any, n_params)
            The pre-obtained parameter values to set to the class
        summaries : float(any, n_summaries)
            The summaries of prerun simulations (at parameter values
            corresponding to the values in `parameters`)
        distances : float(n_targets, any) or None, default=None
            The distances of the summaries from the summaries of each target.
            If None then the distances will be calculated
        replace : bool, default=False
            Whether to replace the summaries, parameters and distances already
            obtained
        """
        if distances is None:
            distances = jax.vmap(
                lambda target, F: self.distance_measure(
                    summaries, target, F))(self.target_summaries, self.F)
        if (self.parameters.all is None) or (replace):
            self.parameters.all = parameters
            self.summaries.all = summaries
            self.distances.all = distances
        else:
            self.parameters.all = np.concatenate(
                [self.parameters.all, parameters], axis=0)
            self.summaries.all = np.concatenate(
                [self.summaries.all, summaries], axis=0)
            self.distances.all = np.concatenate(
                [self.distances.all, distances], axis=0)
        self.parameters.size = self.parameters.all.shape[0]
        self.summaries.size = self.summaries.all.shape[0]
        self.distances.size = self.distances.all.shape[-1]

    def get_samples(self, rng, n_samples, dist=None):
        """Draws parameters from prior and makes and compresses simulations

        Simulations are done `n_samples` in parallel

        Parameters
        ----------
        rng : int(2,)
            A random number generator to draw the parameters and make
            simulations with
        n_samples : int
            Number of parallel samples to draw
        dist : fn or None, default=None
            A distribution to sample, if None the samples will be drawn from
            the prior set in the class initialisation
        """
        rng, key = jax.random.split(rng)
        if dist is None:
            parameters = self.prior.sample(n_samples, seed=rng)
        else:
            parameters = dist.sample(n_samples, seed=rng)
        summaries = self.compressor(self.simulator(key, parameters))
        return parameters, summaries

    def set_accepted(self, ϵ, smoothing=None):
        """Sets the accepted and rejected attributes of the containers

        Using a distance (or list of distances for each target) cutoff between
        simulation summaries and summaries from some target the accepted and
        rejected parameter values (and summaries) can be defined. These points
        are used to make an approximate set of marginal distributions for
        plotting based on histogramming the points - where smoothing can be
        performed on the histogram to avoid undersampling artefacts.

        Parameters
        ----------
        ϵ : float or float(n_targets)
            The acceptance distance between summaries from simulations and the
            summary of the target data. A different epsilon can be passed for
            each target.
        smoothing : float or None, default=None
            A Gaussian smoothing for the marginal distributions

        Methods
        -------
        get_accepted:
            Returns a boolean array with whether summaries are within `ϵ`
        """
        def get_accepted(distances, ϵ):
            """ Returns a boolean array with whether summaries are within `ϵ`

            Parameters
            ----------
            distances : float(any)
                The distances between the summary of a target and the summaries
                of the run simulations
            ϵ : float
                The acceptance distance between summaries from simulations and
                the summary of the target data
            """
            return np.less(distances, ϵ)

        if not isinstance(ϵ, list):
            accepted = jax.vmap(
                lambda distances: get_accepted(distances, ϵ))(
                self.distances.all)
        else:
            accepted = jax.vmap(get_accepted)(self.distances.all, ϵ)

        rejected = ~accepted
        accepted_inds = [np.argwhere(accept)[:, 0] for accept in accepted]
        rejected_inds = [np.argwhere(reject)[:, 0] for reject in rejected]
        self.parameters.ϵ = ϵ
        self.summaries.ϵ = ϵ
        self.distances.ϵ = ϵ
        self.parameters.accepted = [self.parameters.all[inds]
                                    for inds in accepted_inds]
        self.parameters.n_accepted = np.array([
            self.parameters.accepted[i].shape[0]
            for i in range(self.n_targets)])
        self.summaries.accepted = [self.summaries.all[inds]
                                   for inds in accepted_inds]
        self.summaries.n_accepted = np.array([
            self.summaries.accepted[i].shape[0]
            for i in range(self.n_targets)])
        self.distances.accepted = [
            self.distances.all[i, inds]
            for i, inds in enumerate(accepted_inds)]
        self.distances.n_accepted = np.array([
            self.distances.accepted[i].shape[0]
            for i in range(self.n_targets)])
        self.parameters.rejected = [
            self.parameters.all[inds] for inds in rejected_inds]
        self.summaries.rejected = [
            self.summaries.all[inds] for inds in rejected_inds]
        self.distances.rejected = [
            self.distances.all[i, inds]
            for i, inds in enumerate(rejected_inds)]
        self.marginals = self.get_marginals(smoothing=smoothing)

    def get_min_accepted(self, rng, ϵ, accepted, n_simulations=1,
                         max_iterations=10, smoothing=None, verbose=True):
        """Iteratively run ABC until a minimum number of samples are accepted

        Setting a maximum distance that simulation summaries can be allowed to
        be from the summaries of each target an iterative scheme can be used to
        get ``n_samples`` simulations at a time, compress them and calculate
        the distances until the minimum number of desired samples are within
        the allowed cutoff distance. If multiple targets are being inferred
        then the simulations are run until all targets have at least the
        minimum number of accepted samples.

        Parameters
        ----------
        rng : int(2,)
            A random number generator to draw the parameters and make
            simulations with
        ϵ : float or float(n_targets)
            The acceptance distance between summaries from simulations and the
            summary of the target data. A different epsilon can be passed for
            each target.
        accepted : int
            The minimum number of samples to be accepted within the `ϵ` cutoff
        n_simulations : int, default=1
            The number of simulations to do at once
        max_iterations : int, default=10
            Maximum number of iterations in the while loop to prevent infinite
            runs. Note if max_iterations is reached then there are probably
            not the required number of accepted samples
        smoothing : float or None, default=None
            A Gaussian smoothing for the marginal distributions
        verbose : bool, default=True
            Whether to print out the running stats (number accepted,
            iterations, etc)

        Returns
        -------
        float(any, n_params)
            All parameters values drawn
        float(any, n_summaries)
            All summaries of simulations made
        float(n_target, any)
            The distance of every summary to each target
        """
        @jax.jit
        def loop(inputs):
            return jax.lax.while_loop(loop_cond, loop_body, inputs)

        def loop_cond(inputs):
            return np.logical_and(
                np.less(np.min(inputs[-2]), accepted),
                np.less(inputs[-1], max_iterations))

        def loop_body(inputs):
            rng, parameters, summaries, distances, n_accepted, iteration = \
                inputs
            rng, key = jax.random.split(rng)
            parameter_samples = self.prior.sample(n_simulations, seed=key)
            rng, key = jax.random.split(rng)
            summary_samples = self.compressor(
                self.simulator(key, parameter_samples))
            distance_samples = jax.vmap(
                lambda target, F: self.distance_measure(
                    summary_samples, target, F))(
                self.target_summaries, self.F)
            indices = jax.lax.dynamic_slice(
                np.arange(n_simulations * max_iterations),
                [n_simulations * iteration],
                [n_simulations])
            parameters = jax.ops.index_update(
                parameters,
                jax.ops.index[indices],
                parameter_samples)
            summaries = jax.ops.index_update(
                summaries,
                jax.ops.index[indices],
                summary_samples)
            distances = jax.ops.index_update(
                distances,
                jax.ops.index[:, indices],
                distance_samples)
            n_accepted = np.int32(np.less(distances, ϵ).sum(1))
            return rng, parameters, summaries, distances, n_accepted, \
                iteration + np.int32(1)

        parameters = np.ones((max_iterations * n_simulations, self.n_params))
        summaries = np.ones(
            (max_iterations * n_simulations,
             self.n_summaries))
        distances = np.ones((
            self.n_targets,
            max_iterations * n_simulations)) * np.inf
        if self.parameters.all is not None:
            parameters = np.vstack([parameters, self.parameters.all])
            summaries = np.vstack([summaries, self.summaries.all])
            distances = np.hstack([distances, self.distances.all])
            if self.parameters.n_accepted is None:
                self.set_accepted(ϵ, smoothing=smoothing)
            n_accepted = np.int32(self.parameters.n_accepted)
        else:
            n_accepted = np.zeros(self.n_targets, dtype=np.int32)
        current_accepted = n_accepted
        iteration = 0
        _, parameters, summaries, distances, n_accepted, iteration = loop(
            (rng, parameters, summaries, distances, n_accepted, iteration))
        keep = ~np.any(np.isinf(distances), 0)
        parameters = parameters[keep]
        summaries = summaries[keep]
        distances = distances[:, keep]
        if verbose:
            print(f"{n_accepted - current_accepted} accepted in last ",
                  f"{iteration} iterations ",
                  f"({n_simulations * iteration} simulations done).")
        return parameters, summaries, distances

    def set_marginals(self, accepted_parameters=None, ranges=None,
                      gridsize=None, smoothing=None):
        """Wrapper for ``get_marginals`` to set the ``marginals`` attribute

        Parameters
        ----------
        accepted_parameters : float(any, n_params) or None, default=None
            An array of all accepted parameter values
        ranges : list or None, default=None
            A list of arrays containing the bin centres for the marginal
            distribution obtained by histogramming for each parameter.
        gridsize : list or None, default=None
            The number of grid points to evaluate the marginal distribution on
            for each parameter
        smoothing : float or None, default=None
            A Gaussian smoothing for the marginal distributions
        """
        self.marginals = self.get_marginals(
            accepted_parameters=accepted_parameters, ranges=ranges,
            gridsize=gridsize, smoothing=smoothing)

    def get_marginals(self, accepted_parameters=None, ranges=None,
                      gridsize=None, smoothing=None):
        """ Creates the 1D and 2D marginal distribution list for plotting

        Using list of parameter values (accepted by the ABC) an approximate set
        of marginal distributions for plotting are created based on
        histogramming the points. Smoothing can be performed on the histogram
        to avoid undersampling artefacts.

        For every parameter the full distribution is summed over every other
        parameter to get the 1D marginals and for every combination the 2D
        marginals are calculated by summing over the remaining parameters. The
        list is made up of a list of n_params lists which contain n_columns
        number of objects.

        Parameters
        ----------
        accepted_parameters : float(any, n_params) or None, default=None
            An array of all accepted parameter values. If None, the accepted
            parameters from the `parameters` class attribute are used
        ranges : list or None, default=None
            A list of arrays containing the bin centres for the marginal
            distribution obtained by histogramming for each parameter. If None
            the ranges are constructed from the ranges of the prior
            distribution.
        gridsize : list or None, default=None
            The number of grid points to evaluate the marginal distribution on
            for each parameter. This needs to be set if ranges is passed (and
            different from the gridsize set on initialisation)
        smoothing : float or None, default=None
            A Gaussian smoothing for the marginal distributions. Smoothing not
            done if smoothing is None

        Returns
        -------
        list of lists:
            The 1D and 2D marginal distributions for each parameter (of pair)
        """
        if accepted_parameters is None:
            accepted_parameters = self.parameters.accepted
        if ranges is None:
            ranges = [
                np.hstack([range, np.array([range[1] - range[0]])])
                - (range[1] - range[0]) / 2 for range in self.ranges]
        if gridsize is None:
            gridsize = self.gridsize
        if smoothing is not None:
            def smooth(x):
                return gaussian_filter(x, smoothing, mode="mirror")
        else:
            def smooth(x):
                return x
        marginals = []
        for row in range(self.n_params):
            marginals.append([])
            for column in range(self.n_params):
                if column == row:
                    marginals[row].append(
                        np.array([
                            smooth(np.histogram(
                                parameters[:, column],
                                bins=ranges[column],
                                density=True)[0])
                            for parameters in accepted_parameters]))
                elif column < row:
                    marginals[row].append(
                        np.array([
                            smooth(np.histogramdd(
                                parameters[:, [column, row]],
                                bins=[ranges[column], ranges[row]],
                                density=True)[0])
                            for parameters in accepted_parameters]))
        return marginals

    def scatter_plot(self, ax=None, ranges=None, points=None, label=None,
                     axis_labels=None, colours=None, hist=True, s=5, alpha=1.,
                     figsize=(10, 10), linestyle="solid", target=None, ncol=2,
                     bbox_to_anchor=(0.0, 1.0)):
        """Plot a scatter corner plot of all accepted parameter pair values

        Plots scatter plots for parameter values in 2D subplots (and optionally
        the histogram of points in the 1D diagonal subplots).

        Parameters
        ----------
        ax : list of matplotlib axes or None, default=None
            If ax is None a new figure and axes are created to make a corner
            plot otherwise a set of axes are formatted correctly for a corner
            plot. If existing ax is not correctly shaped there will be an error
        ranges : list, default=None
            The list of ranges to set the number of rows and columns (if this
            is None the ranges set on initialisation will be used)
        points : float(n_targets, n_points, {n_params} or {n_summaries})
            The points to scatter plot, if None the class instance of
            `parameters.accepted` will be used
        label : str or None, default=None
            Name to be used in the legend
        axis_labels : list of str or None, default=None
            A list of names for each parameter, no axis labels if None
        colours : str or list or None, default=None
            The colour or list of colours to use for the different targets, if
            None then the normal matplotlib colours are used
        hist : bool, default=True
            Whether or not to plot histograms on the diagonal of the corner
            plot
        s : float, default=5.
            The size of the marker points in the scatter plot
        alpha : float, default=1.
            The amount of alpha colour for the marker points
        figsize : tuple, default=(10, 10)
            The size (in inches) to create a figure (if ax is None)
        linestyle : str, default="solid"
            Linestyle for the histograms
        target : None or int or list, default=None
            The indices to choose which target's points are plotted
        format : bool, default=False
            If formatting is not needed
        ncols : int, default=2
            Number of columns for the legend
        bbox_to_anchor : tuple, default=(0.0, 1.0)
            Position to fix the legend to

        Returns
        -------
        axes object:
            The scatter plot axes
        """
        if ranges is None:
            ranges = self.ranges
        if points is None:
            points = self.parameters.accepted
        return self._scatter_plot(
            ax=ax, ranges=ranges, points=points, label=label,
            axis_labels=axis_labels, colours=colours, hist=hist, s=s,
            alpha=alpha, figsize=figsize, linestyle=linestyle, target=target,
            ncol=ncol, bbox_to_anchor=bbox_to_anchor)

    def scatter_summaries(self, ax=None, ranges=None, points=None, label=None,
                          axis_labels=None, colours=None, hist=True, s=5,
                          alpha=1., figsize=(10, 10), linestyle="solid",
                          gridsize=100, target=None, format=False, ncol=2,
                          bbox_to_anchor=(0.0, 1.0)):
        """Plot a scatter corner plot of all accepted summary pair values

        Plots scatter plots for summary values in 2D subplots (and optionally
        the histogram of points in the 1D diagonal subplots).

        Parameters
        ----------
        ax : list of matplotlib axes or None, default=None
            If ax is None a new figure and axes are created to make a corner
            plot otherwise a set of axes are formatted correctly for a corner
            plot. If existing ax is not correctly shaped there will be an error
        ranges : list, default=None
            The list of ranges to set the number of rows and columns (if this
            is None the ranges set on initialisation will be used)
        points : float(n_targets, n_points, {n_params} or {n_summaries})
            The points to scatter plot, if None the class instance of
            `summaries.accepted` will be used
        label : str or None, default=None
            Name to be used in the legend
        axis_labels : list of str or None, default=None
            A list of names for each parameter, no axis labels if None
        colours : str or list or None, default=None
            The colour or list of colours to use for the different targets, if
            None then the normal matplotlib colours are used
        hist : bool, default=True
            Whether or not to plot histograms on the diagonal of the corner
            plot
        s : float, default=5.
            The size of the marker points in the scatter plot
        alpha : float, default=1.
            The amount of alpha colour for the marker points
        figsize : tuple, default=(10, 10)
            The size (in inches) to create a figure (if ax is None)
        linestyle : str, default="solid"
            Linestyle for the histograms
        target : None or int or list, default=None
            The indices to choose which target's points are plotted
        format : bool, default=False
            If formatting is not needed
        ncols : int, default=2
            Number of columns for the legend
        bbox_to_anchor : tuple, default=(0.0, 1.0)
            Position to fix the legend to

        Returns
        -------
        axes object:
            The scatter plot axes

        Todo
        ----
        There are many extra matplotlib parameters which could be passed,
        although this is not done because the axis is returned which can then
        be manipulated.
        """
        if points is None:
            points = self.summaries.accepted
            n_summaries = self.n_summaries
        else:
            n_summaries = points[0].shape[-1]
        gridsize = self.get_gridsize(gridsize, n_summaries)
        targets, n_targets = self.target_choice(target)
        if ranges is None:
            ranges = [
                np.linspace(
                    np.min(
                        np.array([
                            np.min(points[target][:, summary])
                            for target in targets])),
                    np.max(
                        np.array([
                            np.max(points[target][:, summary])
                            for target in targets])),
                    gridsize[summary])
                for summary in range(n_summaries)]
        return self._scatter_plot(
            ax=ax, ranges=ranges, points=points, label=label,
            axis_labels=axis_labels, colours=colours, hist=hist, s=s,
            alpha=alpha, figsize=figsize, linestyle=linestyle, target=target,
            format=format, ncol=ncol, bbox_to_anchor=bbox_to_anchor)

    def F_distance(self, x, y, F):
        """Default distance measure, squared difference scaled by Fisher matrix

        The target summaries are expanded on the zeroth axis so that it can
        broadcast with the simulation summaries.

        Parameters
        ----------
        x : (any, n_params)
            Envisioned as a whole batch of summaries of simulations
        y : (n_params)
            Envisioned as a target summary
        F : (n_params)
            Fisher information matrix to scale the summary direction
        """
        difference = x - np.expand_dims(y, 0)
        return np.sqrt(
            np.einsum(
                "ik,kl,il->i",
                difference,
                F,
                difference))

    def euclidean_distance(self, x, y, aux=None):
        """Euclidean distance between simulation summaries and target summary

        The target summaries are expanded on the zeroth axis so that it can
        broadcast with the simulation summaries.

        Parameters
        ----------
        x : (any, n_params)
            Envisioned as a whole batch of summaries of simulations
        y : (n_params)
            Envisioned as a target summary
        aux : None, default=Note
            Empty holder so that function works in the same way as `F_distance`
        """
        difference = x - np.expand_dims(y, 0)
        return np.sqrt(np.sum(np.square(difference), -1))
