# TODO
# Add kde as an alternative to the histogram

import jax
import jax.numpy as np
from functools import partial
from imnn.lfi import LikelihoodFreeInference
from imnn.utils import container
from scipy.ndimage import gaussian_filter
from imnn.utils.utils import _check_input


class ApproximateBayesianComputation(LikelihoodFreeInference):
    """Automatic Bayesian computation

    Parameters
    ----------
    target_summaries : float(n_targets, n_summaries)
        The compressed data to be infered
    n_summaries : int
        The size of the output of the compressor for a single simulation
    n_targets : int
        The number of different targets to be infered
    invF : float(n_params, n_params)
        The inverse Fisher information matrix for rescaling distance measure
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
            if isinstance(F, list):
                self.invF = [
                    np.linalg.inv(
                        _check_input(f, (self.n_params, self.n_params), "F"))
                    for f in F]
            else:
                if F.shape == (self.n_params, self.n_params):
                    self.invF = np.expand_dims(np.linalg.inv(F), 0)
                    if self.n_targets > 1:
                        self.invF = np.repeat(self.invF, self.n_targets)
                else:
                    self.invF = jax.vmap(np.linalg.inv)(
                        _check_input(
                            F,
                            (self.n_targets, self.n_params, self.n_params),
                            "F"))
        if distance_measure is None:
            if self.invF is not None:
                self.distance_measure = self.F_distance
            else:
                self.distance_measure = self.euclidean_distance

        self.parameters = container()
        self.parameters.n_params = self.n_params
        self.summaries = container()
        self.summaries.n_summaries = self.n_summaries
        self.distances = container()
        self.distances.n_targets = self.n_targets

    def __call__(self, ϵ=None, rng=None, n_samples=None, parameters=None,
                 summaries=None, min_accepted=None, max_iterations=1,
                 smoothing=None, replace=False):
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
        if distances is None:
            distances = self.distance_measure(self.target_summaries, summaries)
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

    def get_samples(self, rng, n_samples, dist=None):
        rng, key = jax.random.split(rng)
        if dist is None:
            parameters = self.prior.sample(n_samples, seed=rng)
        else:
            parameters = dist.sample(n_samples, seed=rng)
        summaries = self.compressor(self.simulator(key, parameters))
        return np.stack(parameters, -1), summaries

    def set_accepted(self, ϵ, smoothing=None):
        def get_accepted(distances, ϵ):
            return np.less(distances, ϵ)
        if not isinstance(ϵ, list):
            accepted = jax.vmap(
                lambda distances: get_accepted(distances, ϵ))(
                self.distances.all)
        else:
            print(self.distances.all.shape)
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

    @partial(jax.jit, static_argnums=(0, 7, 8, 9, 10))
    def _get_min_accepted(
            self,
            rng, parameters, summaries, distances, n_accepted, iteration,
            ϵ, accepted, n_simulations, max_iterations):
        def loop_cond(inputs):
            return np.logical_and(
                np.less(inputs[-2], accepted),
                np.less(inputs[-1], max_iterations))

        def loop_body(inputs):
            rng, parameters, summaries, distances, n_accepted, iteration = \
                inputs
            rng, key = jax.random.split(rng)
            parameter_samples = self.prior.sample(n_simulations, seed=key)
            rng, key = jax.random.split(rng)
            summary_samples = self.compressor(
                self.simulator(key, parameter_samples))
            distance_samples = self.distance_measure(
                summary_samples, self.target_summaries).T
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
            n_accepted = np.min(np.less(distances, ϵ).sum(1))
            return rng, parameters, summaries, distances, n_accepted, \
                iteration + 1

        inputs = (rng, parameters, summaries, distances, n_accepted,
                  iteration)
        return jax.lax.while_loop(loop_cond, loop_body, inputs)

    def get_min_accepted(self, rng, ϵ, accepted, n_simulations=1,
                         max_iterations=1, smoothing=None, verbose=True):
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
            n_accepted = np.min(self.parameters.n_accepted)
        else:
            n_accepted = 0
        current_accepted = n_accepted
        iteration = 0
        _, parameters, summaries, distances, n_accepted, iteration = \
            self._get_min_accepted(
                rng, parameters, summaries, distances, n_accepted, iteration,
                ϵ, accepted, n_simulations, max_iterations)
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
        self.marginals = self.get_marginals(
            accepted_parameters=accepted_parameters, ranges=ranges,
            gridsize=gridsize, smoothing=smoothing)

    def get_marginals(self, accepted_parameters=None, ranges=None,
                      gridsize=None, smoothing=None):
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
                            np.min(self.summaries.accepted[target][:, summary])
                            for target in targets])),
                    np.max(
                        np.array([
                            np.max(self.summaries.accepted[target][:, summary])
                            for target in targets])),
                    gridsize[summary])
                for summary in range(n_summaries)]
        return self._scatter_plot(
            ax=ax, ranges=ranges, points=points, label=label,
            axis_labels=axis_labels, colours=colours, hist=hist, s=s,
            alpha=alpha, figsize=figsize, linestyle=linestyle, target=target,
            format=format, ncol=ncol, bbox_to_anchor=bbox_to_anchor)

    def F_distance(self, x, y, F=None):
        if F is None:
            F = self.F
        difference = np.expand_dims(x, 1) - np.expand_dims(y, 0)
        return np.sqrt(
            np.einsum(
                "ijk,kl,ijl->i",
                difference,
                F,
                difference))

    def euclidean_distance(self, x, y, aux=None):
        difference = np.expand_dims(x, 1) - np.expand_dims(y, 0)
        return np.sqrt(np.sum(np.square(difference), -1))
