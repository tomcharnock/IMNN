from imnn.lfi import ApproximateBayesianComputation
import jax
import jax.numpy as np
import tensorflow_probability
from functools import partial
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


class PopulationMonteCarlo(ApproximateBayesianComputation):
    """ Population driven approximate Bayesian computation sample retrieval

    SEEMS TO BE A BUG IN THIS MODULE CURRENTLY!

    Approximate Bayesian computation tends to be very expensive because it
    depends on drawing samples from all over the prior and a lot of the domain
    may be extremely unlikely without good knowledge of the sampling
    distribution. To make sampling more efficient, the population Monte Carlo
    iteratively updates the proposal distribution for new samples based on the
    density of the closest summaries of simulations to the summary of the
    target.

    Parameters
    ----------
    acceptance_reached : float(n_targets,)
        The number of draws accepted over the number over the number of draws
        made for each target
    iterations : int(n_targets,)
        The number of iterations of population updates for each target
    total_draws : int(n_targets,)
        The total number of draws made (equivalent to number of simulations
        made) for each target
    """
    def __init__(self, target_data, prior, simulator, compressor, F=None,
                 gridsize=100, distance_measure=None):
        """ Constructor method

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
        F : list or float({n_targets}, n_params, n_params) or None
            The Fisher information matrix to rescale parameter directions for
            measuring the distance. If None, this is set to the identity matrix
        gridsize : int or list, default=100
            The number of grid points to evaluate the marginal distribution on
            for every parameter (int) or each parameter (list)
        distance_measure : fn or None
            A distance measuring function between a batch of summaries and a
            single set of summaries for a target. If None, ``F_distance``
            (Euclidean scaled by the Fisher information) is used. This is only
        """
        super().__init__(
            target_data=target_data,
            prior=prior,
            simulator=simulator,
            compressor=compressor,
            gridsize=gridsize,
            F=F,
            distance_measure=distance_measure)
        self.acceptance_reached = np.zeros(self.n_targets)
        self.iterations = np.zeros(self.n_targets, dtype=np.int32)
        self.total_draws = np.zeros(self.n_targets, dtype=np.int32)

    def __call__(self, rng, n_points,
                 percentile=None, acceptance_ratio=0.1, max_iteration=10,
                 max_acceptance=1, max_samples=int(1e5),
                 n_initial_points=None, n_parallel_simulations=None,
                 proposed=None, summaries=None, distances=None, smoothing=None,
                 replace=False):
        """Run the PMC

        Parameters
        ----------
        rng : int(2,)
            A random number generator for generating simulations and randomly
            drawing parameter values from the prior
        n_points : int
            Number of points desired in the final (approximate) posterior
            (note that if `acceptance_ratio` is large then this approximation
            will be bad)
        percentile : int or None, default=None
            The percentage of points to define as the population, i.e. 75 for
            75% of the points. If None only the furthest point is moved one at
            a time
        acceptance_ratio : float
            The ratio of the number of accepted points to total proposals. When
            this gets small it suggests that points aren't being added to the
            population any more because the population is stationary
        max_iteration : int, default=10
            The cutoff number of iterations to break out of the while loop even
            if `acceptance_ratio` is not reached
        max_acceptance : int, default=1
            The cutoff number of attempts in a single iteration to attempt to
            get a sample accepted
        max_samples : int, default=100000
            The number of attempts to get parameter values from the truncated
            normal distribution
        n_initial_points : int or None, default=None
            The number of points to run in the initial ABC to define the
            population before the PMC starts. The PMC will always run from
            scratch if n_initial_points is passed
        n_parallel_simulations : int or None, default=None
            number of simulations to do at once, the innermost (closest)
            summaries are the only ones accepted, but this can massively reduce
            sampling time as long as the simulations can be easily parallelised
        proposed : float(any, n_params) or None, default=None
            A set of proposed parameters which have been used to premake
            simulations. Summaries of these simulations must also be passed as
            `summaries`. These can be used instead of running an initial ABC
            step
        summaries : float(any, n_summaries) or None, default=None
            A set of summaries of simulations which have been premade at
            parameter values corresponding to `proposed`. These can be used
            instead of running an initial ABC step
        distances : float(n_targets, any) or None, default=None
            An optional distance calculation from `summaries`, if this is not
            passed then distances is calculated in the call
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

        Raises
        ------
        ValueError
            if `n_initial_points` is less than `n_points`

        Todo
        ----
        type checking and pytests need writing
        """
        if n_initial_points is not None:
            if n_initial_points < n_points:
                raise ValueError(
                    "`n_initial_points` must be greater than or equal to " +
                    "the final number of points (`n_points`)")
            if n_parallel_simulations is not None:
                rng, *keys = jax.random.split(
                    rng, num=n_parallel_simulations + 1)
                proposed, summaries = jax.vmap(
                    lambda key: self.get_samples(
                        key, n_initial_points // n_parallel_simulations))(
                    np.array(keys))
            else:
                rng, *keys = jax.random.split(rng, num=n_initial_points + 1)
                proposed, summaries = jax.vmap(
                    lambda key: self.get_samples(key, 1))(np.array(keys))
            proposed = proposed.reshape((n_initial_points, -1))
            summaries = summaries.reshape((n_initial_points, -1))
            distances = jax.vmap(
                lambda target, F: self.distance_measure(
                    summaries, target, F))(self.target_summaries, self.F)
        elif (proposed is not None) and (summaries is not None):
            if distances is None:
                distances = jax.vmap(
                    lambda target, F: self.distance_measure(
                        summaries, target, F))(self.target_summaries, self.F)
        elif (self.parameters.all is not None) and (not replace):
            proposed = self.parameters.all.reshape((-1, self.n_params))
            summaries = self.summaries.all.reshape((-1, self.n_summaries))
            distances = jax.vmap(
                lambda target, F: self.distance_measure(
                    summaries, target, F))(self.target_summaries, self.F)
        else:
            raise ValueError(
                "`proposed` and `summaries` (and optionally `distances`) or " +
                "`n_initial_points` must be provided if PMC has not been " +
                "previously called")

        sample_indices = np.argsort(distances, axis=1)[:, :n_points]
        samples = jax.vmap(lambda x: proposed[x])(sample_indices)
        summaries = jax.vmap(lambda x: summaries[x])(sample_indices)
        distances = np.take(distances, sample_indices)

        weighting = self.prior.prob(samples)

        if percentile is None:
            ϵ_ind = -1
        else:
            ϵ_ind = int(percentile / 100 * n_points)

        key = np.array(jax.random.split(rng, num=self.n_targets))
        (rng, samples, summaries, distances, weighting, acceptance_reached,
         iteration_counter, total_draws) = jax.vmap(
            partial(
                self.move_samples,
                ϵ_ind=ϵ_ind,
                acceptance_ratio=acceptance_ratio,
                max_iteration=max_iteration,
                max_acceptance=max_acceptance,
                max_samples=max_samples,
                n_parallel_simulations=n_parallel_simulations))(
            key, samples, summaries, distances, weighting,
            self.target_summaries, self.F)
        self.set_samples(samples, summaries, distances=distances,
                         replace=replace)
        self.set_accepted(smoothing=smoothing)
        self.acceptance_reached = self.acceptance_reached + acceptance_reached
        self.iterations = self.iterations + iteration_counter
        self.total_draws = self.total_draws + total_draws
        print(
            f"Acceptance reached {self.acceptance_reached} in " +
            f"{self.iterations} iterations with a total of " +
            f"{self.total_draws} draws")
        return self.parameters, self.distances, self.summaries

    def move_samples(
            self, rng, samples, summaries, distances, weighting, target, F,
            ϵ_ind=None, acceptance_ratio=None, max_iteration=None,
            max_acceptance=None, max_samples=None,
            n_parallel_simulations=None):
        """ Loop which updates the population iteratively for individual target

        The values of the parameters corresponding to the most distant
        summaries to the summary of the target data are considered to be the
        means of a new proposal distribution with a covariance given by the
        population weighted samples (weighted from the prior in the first
        instance). From this proposal distribution a new sample (or a parallel
        set of samples) is proposed and a simulation made at each sample,
        compressed and the distance from the summary of the target data
        calculated. If this distance is within the population then that
        proposal is accepted and the weighting is updated with this new sample
        and the new furthest samples will be updated with this new proposal
        distribution on the next iteration. This is continued until the number
        of accepted draws is small compared to the total number of draws (this
        is the `acceptance_ratio`). At this point the distribution is somewhat
        stationary (as long as `acceptance_ratio` is small enough) and the
        samples approximately come from the target distribution.

        Parameters
        ----------
        rng : int(2,)
            A random number generator to draw the parameters and make
            simulations with
        samples : float(n_points, n_params)
            The parameter values from the ABC step
        summaries : float(n_points, n_summaries)
            The summaries of prerun simulations from the ABC step (at parameter
            values corresponding to the values in `samples`)
        distances : float(n_points,)
            The distances of the summaries from the summaries of a target
            from the ABC step
        weighting : float(n_points,)
            The value of the prior evaluated at the parameter values obtained
            in the ABC step
        target : float(n_summaries,)
            The summary of the target to infer the parameter values of
        F : float(n_params, n_params)
            The Fisher information matrix to rescale parameter directions for
            measuring the distance. This should be set to the identity matrix
            float(n_summaries, n_summaries) if the summaries are not parameter
            estimates
        ϵ_ind : float, default=None
            The index of the outer most population simulation
        acceptance_ratio : float, default=None
            The ratio of the number of accepted points to total proposals. When
            this gets small it suggests that points aren't being added to the
            population any more because the population is stationary
        max_iteration : int, default=None
            The cutoff number of iterations to break out of the while loop even
            if `acceptance_ratio` is not reached
        max_acceptance : int, default=None
            The cutoff number of attempts in a single iteration to attempt to
            get a sample accepted
        max_samples : int, default=None
            The number of attempts to get parameter values from the truncated
            normal distribution
        n_parallel_simulations : int or None, default=None
            number of simulations to do at once, the innermost (closest)
            summaries are the only ones accepted, but this can massively reduce
            sampling time as long as the simulations can be easily parallelised

        Returns
        -------
        int(2,):
            A random number generator to draw the parameters and make
            simulations with
        float(n_points, n_params):
            The accepted parameter values
        float(n_points, n_summaries):
            The summaries corresponding to the accepted parameter values, i.e.
            the closest summaries to the summary of the target data
        float(n_points,):
            The distances of the summaries from the summaries of the target
        float(n_points,):
            The value of the weighting from by the population
        float:
            The acceptance ratio reach by the final iteration
        int:
            The number of iterations run
        int:
            The total number of draws from the proposal distribution over all
            iterations

        Methods
        -------
        single_iteration_condition:
            Checks if the acceptance ratio or maximum iterations is reached
        single_iteration:
            Runs a single (or parallel set of) simulation(s), accepts if closer
        """
        def single_iteration_condition(args):
            """Checks if the acceptance ratio or maximum iterations is reached

            Parameters
            ----------
            args : tuple
                loop variables (described in `single_iteration`)

            Returns
            -------
            bool:
                True if acceptance_ratio is reached or the maximum number of
                iterations is reached
            """
            return np.logical_and(
                np.greater(args[-3], acceptance_ratio),
                np.less(args[-2], max_iteration))

        def single_iteration(args):
            """Runs single (parallel set of) simulation(s), accepts if closer

            Parameters
            ----------
            args : tuple
                - *rng* **int(2,)** -- A random number generator to draw the
                  parameters and make simulations with
                - *loc* **float(n_parallel_sims, n_params)** -- The current
                  iteration parameter values to use as the mean of the proposal
                  distribution for the next iterations parameters
                - *scale* **float(n_parallel_sims, n_params, n_params)** -- The
                  population weighted covariance for the proposal distribution
                - *summ* **float(n_parallel_sims, n_summaries)** -- The values
                  of the simulation summaries from this iteration
                - *dis* **float(n_parallel_sims,)** -- The distances between
                  the summaries being moved and the target summaries
                - *draws* **int** -- The number of draws taken so far
                - *accepted* **int** -- The number of proposals accepted so far
                - *acceptance_counter* **int** -- The total number of
                  iterations attempting to get an accepted proposal

            Returns
            -------
            tuple
                Same as input loop variables

            Methods
            -------
            single_acceptance_condition:
                checks if proposal has been accepted or max iterations reached
            single_acceptance:
                draws a proposal, simulates and compresses and checks distances
            """
            def single_acceptance_condition(args):
                """checks proposal has been accepted or max iterations reached

                Parameters
                ----------
                args : tuple
                    see loop variable in `single_iteration`

                Returns
                -------
                bool:
                    True if proposal not accepted and number of attempts to get
                    an accepted proposal not yet reached
                """
                return np.logical_and(
                    np.less(args[-2], 1),
                    np.less(args[-1], max_acceptance))

            def single_acceptance(args):
                """Draws a proposal, simulates and compresses, checks distance

                A new proposal is drawn from a truncated multivariate normal
                distribution whose mean is centred on the parameter to move and
                the covariance is set by the population. From this proposed
                parameter value a simulation is made and compressed and the
                distance from the target is calculated. If this distance is
                less than the current position then the proposal is accepted.

                Parameters
                ----------
                args : tuple
                    see loop variable in `single_iteration`

                Returns
                -------
                bool:
                    True if proposal not accepted and number of attempts to get
                    an accepted proposal not yet reached

                Todo
                ----
                Parallel sampling is currently commented out
                """
                (rng, loc, scale, summ, dis, draws, accepted,
                 acceptance_counter) = args
                rng, key = jax.random.split(rng)
                proposed, summaries = self.get_samples(
                    key, None, dist=tmvn(
                        loc, scale, self.prior.low, self.prior.high,
                        max_counter=max_samples))
                distances = np.squeeze(
                    self.distance_measure(
                        np.expand_dims(summaries, 0),
                        target,
                        F))
                # if n_parallel_simulations is not None:
                #     min_distance_index = np.argmin(distances)
                #     min_distance = distances[min_distance_index]
                #     closer = np.less(min_distance, ϵ)
                #     loc = jax.lax.cond(
                #         closer,
                #         lambda _ : proposed[min_distance_index],
                #         lambda _ : loc,
                #         None)
                #     summ = jax.lax.cond(
                #         closer,
                #         lambda _ : summaries[min_distance_index],
                #         lambda _ : summ,
                #         None)
                #     dis = jax.lax.cond(
                #         closer,
                #         lambda _ : distances[min_distance_index],
                #         lambda _ : dis,
                #         None)
                #     iteration_draws = n_parallel_simulations \
                #         - np.isinf(distances).sum()
                #     draws += iteration_draws
                #     accepted = closer.sum()
                # else:
                closer = np.less(distances, np.min(dis))
                loc = jax.lax.cond(
                    closer,
                    lambda _: proposed,
                    lambda _: loc,
                    None)
                summ = jax.lax.cond(
                    closer,
                    lambda _: summaries,
                    lambda _: summ,
                    None)
                dis = jax.lax.cond(
                    closer,
                    lambda _: distances,
                    lambda _: dis,
                    None)
                iteration_draws = 1 - np.isinf(distances).sum()
                draws += iteration_draws
                accepted = closer.sum()
                return (rng, loc, scale, summ, dis, draws, accepted,
                        acceptance_counter + 1)

            (rng, samples, summaries, distances, weighting, acceptance_reached,
             iteration_counter, total_draws) = args
            n_to_move = samples[ϵ_ind:].shape[0]
            cov = self.w_cov(samples, weighting)
            scale = np.linalg.cholesky(cov)
            rng, *keys = jax.random.split(rng, num=n_to_move + 1)

            results = jax.vmap(
                lambda key, loc, scale, summaries, distances, draws, accepted,
                acceptance_counter: jax.lax.while_loop(
                    single_acceptance_condition,
                    single_acceptance,
                    (key, loc, scale, summaries, distances, draws, accepted,
                     acceptance_counter)))(
                np.array(keys),
                samples[ϵ_ind:],
                np.repeat(np.expand_dims(scale, 0), n_to_move, axis=0),
                summaries[ϵ_ind:],
                distances[ϵ_ind:],
                np.zeros(n_to_move, dtype=np.int32),
                np.zeros(n_to_move, dtype=np.int32),
                np.zeros(n_to_move))

            weighting = jax.vmap(
                lambda proposed: (
                    self.prior.prob(proposed)
                    / (np.sum(weighting * tfd.MultivariateNormalTriL(
                        loc=proposed,
                        scale_tril=np.repeat(
                            np.expand_dims(scale, 0),
                            samples.shape[0],
                            axis=0)).prob(proposed)))))(
                np.vstack([samples[:ϵ_ind], results[1]]))
            samples = jax.ops.index_update(
                samples,
                jax.ops.index[ϵ_ind:, :],
                results[1])
            summaries = jax.ops.index_update(
                summaries,
                jax.ops.index[ϵ_ind:, :],
                results[3])
            distances = jax.ops.index_update(
                distances,
                jax.ops.index[ϵ_ind:],
                results[4])
            sample_indices = np.argsort(distances)
            samples = samples[sample_indices]
            summaries = summaries[sample_indices]
            distances = distances[sample_indices]
            weighting = weighting[sample_indices]
            acceptance_reached = results[-2].sum() / results[-3].sum()
            return (rng, samples, summaries, distances, weighting,
                    acceptance_reached, iteration_counter + 1,
                    total_draws + results[-3].sum())

        acceptance_reached = np.inf
        iteration_counter = 0
        total_draws = 0
        return jax.lax.while_loop(
            single_iteration_condition,
            single_iteration,
            (rng, samples, summaries, distances, weighting, acceptance_reached,
             iteration_counter, total_draws))

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
        parameters : float(n_targets, n_points, n_params)
            The pre-obtained parameter values to set to the class
        summaries : float(n_targets, n_points, n_summaries)
            The summaries of prerun simulations (at parameter values
            corresponding to the values in `parameters`)
        distances : float(n_targets, n_points) or None, default=None
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
                [self.parameters.all, parameters], axis=1)
            self.summaries.all = np.concatenate(
                [self.summaries.all, summaries], axis=1)
            self.distances.all = np.concatenate(
                [self.distances.all, distances], axis=1)
        self.parameters.size = self.parameters.all.shape[0]
        self.summaries.size = self.summaries.all.shape[0]
        self.distances.size = self.distances.all.shape[-1]

    def set_accepted(self, smoothing=None):
        """Container values to list of accepted attributes, builds marginals

        Parameters
        ----------
        smoothing : float or None, default=None
            A Gaussian smoothing for the marginal distributions
        """
        self.parameters.accepted = [params for params in self.parameters.all]
        self.parameters.n_accepted = [
            params.shape[0] for params in self.parameters.all]
        self.summaries.accepted = [
            summaries for summaries in self.summaries.all]
        self.summaries.n_accepted = [
            summaries.shape[0] for summaries in self.summaries.all]
        self.distances.accepted = [
            distances for distances in self.distances.all]
        self.distances.n_accepted = [
            distances.shape[0] for distances in self.distances.all]
        self.parameters.rejected = None
        self.summaries.rejected = None
        self.distances.rejected = None
        self.marginals = self.get_marginals(smoothing=smoothing)

    def w_cov(self, proposed, weighting):

        weighted_samples = proposed * weighting[:, np.newaxis]
        return weighted_samples.T.dot(
            weighted_samples) / weighting.T.dot(weighting)


class tmvn():
    """Truncated multivariate normal distribution (for sampling)

    Parameters
    ----------
    loc : float(any, n_params)
        The mean of any number of input distributions
    scale : float(any, n_params, n_params)
        The cholesky matrix of the covariance
    low : float(any, n_params)
        The minimum value for the truncation for each parameter
    high : float(any, n_params)
        The maximum value for the truncation for each parameter
    n_samples : int or None
        The number of different distributions to make (equivalent to
        `batch_shape`)
    n_params : int
        The number of parameters (equivalent to `event_shape`)
    max_counter : int
        Number of iterations to try to get accepted samples (within truncation)
    """
    def __init__(self, loc, scale, low, high, max_counter=int(1e3)):
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        if len(loc.shape) > 1:
            self.n_samples = loc.shape[0]
        else:
            self.n_samples = None
        self.n_params = low.shape[0]
        self.max_counter = max_counter

    def mvn(self, rng, loc):
        u = jax.random.normal(rng, shape=(self.n_params,))
        return loc + u.dot(self.scale)

    def w_cond(self, args):
        _, loc, counter = args
        return np.logical_and(
            np.logical_or(
                np.any(np.greater(loc, self.high)),
                np.any(np.less(loc, self.low))),
            np.less(counter, self.max_counter))

    def __sample(self, args):
        rng, loc, counter = args
        rng, key = jax.random.split(rng)
        return (rng, self.mvn(key, loc), counter + 1)

    def _sample(self, rng, loc):
        rng, key = jax.random.split(rng)
        _, loc, counter = jax.lax.while_loop(
            self.w_cond,
            self.__sample,
            (rng, self.mvn(key, loc), 0))
        return jax.lax.cond(
            np.greater_equal(counter, self.max_counter),
            lambda _: np.nan * np.ones((self.n_params,)),
            lambda _: loc,
            None)

    def _sample_n(self, rng, loc, n=None):
        if n is None:
            return self._sample(rng, loc)
        else:
            key = jax.random.split(rng, num=n)
            return jax.vmap(self._sample)(
                key, np.repeat(loc[np.newaxis], n, axis=0))

    def sample(self, shape=None, seed=None):
        if shape is None:
            if self.n_samples is None:
                return self._sample_n(seed, self.loc)
            else:
                key = jax.random.split(seed, num=self.n_samples)
                return jax.vmap(
                    lambda key, loc: self._sample_n(key, loc))(key, self.loc)
        elif len(shape) == 1:
            if self.n_samples is None:
                return self._sample_n(seed, self.loc, n=shape[0])
            else:
                key = jax.random.split(seed, num=self.n_samples)
                return jax.vmap(
                    lambda key, loc: self._sample_n(key, loc, n=shape[0]))(
                    key, self.loc)
        else:
            key = jax.random.split(seed, num=shape[-1])
            return jax.vmap(
                lambda key: self.sample(
                    shape=tuple(shape[:-1]), seed=key), out_axes=-2)(key)
