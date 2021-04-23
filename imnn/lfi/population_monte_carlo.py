from imnn.lfi import ApproximateBayesianComputation
import jax
import jax.numpy as np
import tensorflow_probability
from functools import partial
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

class PopulationMonteCarlo(ApproximateBayesianComputation):
    def __init__(self, target_data, prior, simulator, compressor,
                 gridsize=100, F=None, distance_measure=None, verbose=True):
        super().__init__(
            target_data=target_data,
            prior=prior,
            simulator=simulator,
            compressor=compressor,
            gridsize=gridsize,
            F=F,
            distance_measure=distance_measure,
            verbose=verbose)
        self.acceptance_reached = np.zeros(self.n_targets)
        self.iterations = np.zeros(self.n_targets, dtype=np.int32)
        self.total_draws = np.zeros(self.n_targets, dtype=np.int32)

    def __call__(self, rng, n_initial_points, n_points,
                 percentile=75, acceptance_ratio=0.1, max_iteration=10,
                 max_acceptance=10, max_samples=int(1e3),
                 n_parallel_simulations=None, smoothing=None, replace=False):
        rng, key = jax.random.split(rng)

        if n_initial_points < n_points:
            raise ValueError(
                "`n_initial_points` must be greater than or equal to the " +
                "final number of points (`n_points`)")

        proposed, summaries = self.get_samples(key, n_initial_points)
        distances = jax.vmap(
            lambda target, f: self.distance_measure(summaries, target, f))(
            self.target_summaries, self.F)

        # if n_parallel_simulations is not None:
        #     proposed = proposed.reshape(
        #         (n_initial_points * n_parallel_simulations, -1))
        #     summaries = summaries.reshape(
        #         (n_initial_points * n_parallel_simulations, -1))
        #     distances = distances.reshape((self.n_targets, -1))

        sample_indices = np.argsort(distances, axis=1)[:, :n_points]
        samples = jax.vmap(lambda x: proposed.T[x])(sample_indices)
        summaries = jax.vmap(lambda x: summaries[x])(sample_indices)
        distances = np.take(distances, sample_indices)

        weighting = self.prior.prob(samples)

        if percentile is None:
            ϵ_ind = -1
            to_accept = 1
        else:
            ϵ_ind = int(percentile / 100 * n_points)
            to_accept = n_points - ϵ_ind

        key = jax.random.split(rng, num=self.n_targets)
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
        def single_iteration_condition(args):
            return np.logical_and(
                np.greater(args[-3], acceptance_ratio),
                np.less(args[-2], max_iteration))

        def single_iteration(args):
            def single_acceptance_condition(args):
                return np.logical_and(
                    np.less(args[-2], 1),
                    np.less(args[-1], max_acceptance))

            def single_acceptance(args):
                rng, loc, summ, dis, draws, accepted, acceptance_counter = args
                rng, key = jax.random.split(rng)
                proposed, summaries = self.get_samples(
                    key, None, dist=tmvn(
                        loc, scale, self.prior.low, self.prior.high,
                        max_counter=max_samples))
                distances = np.squeeze(
                    self.distance_measure(
                        np.expand_dims(target, 0),
                        np.expand_dims(summaries, 0),
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
                closer = np.less(distances, dis)
                loc = jax.lax.cond(closer,
                    lambda _ : proposed,
                    lambda _ : loc,
                    None)
                summ = jax.lax.cond(closer,
                    lambda _ : summaries,
                    lambda _ : summ,
                    None)
                dis = jax.lax.cond(closer,
                    lambda _ : distances,
                    lambda _ : dis,
                    None)
                iteration_draws = 1 - np.isinf(distances).sum()
                draws += iteration_draws
                accepted = closer.sum()
                return (rng, loc, summ, dis, draws, accepted,
                        acceptance_counter+1)

            (rng, samples, summaries, distances, weighting, acceptance_reached,
             iteration_counter, total_draws) = args
            ϵ = distances[ϵ_ind]
            loc = samples[ϵ_ind:]
            cov = self.w_cov(samples, weighting)
            inv_cov = np.linalg.inv(cov)
            scale = np.linalg.cholesky(cov)
            rng, *key = jax.random.split(rng, num=loc.shape[0] + 1)
            draws = np.zeros(loc.shape[0], dtype=np.int32)
            accepted = np.zeros(loc.shape[0], dtype=np.int32)
            acceptance_counter = np.zeros(loc.shape[0], dtype=np.int32)

            results = jax.vmap(
                lambda key, loc, summaries, distances, draws, accepted,
                    acceptance_counter : jax.lax.while_loop(
                    single_acceptance_condition,
                    single_acceptance,
                    (key, loc, summaries, distances, draws, accepted,
                     acceptance_counter)))(
                np.array(key), loc, summaries[ϵ_ind:], distances[ϵ_ind:],
                draws, accepted, acceptance_counter)

            weighting = jax.vmap(
                lambda proposed : (
                    self.prior.prob(proposed)
                    / (np.sum(weighting * tfd.MultivariateNormalTriL(
                        loc=samples,
                        scale_tril=np.repeat(
                            scale[np.newaxis],
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
                results[2])
            distances = jax.ops.index_update(
                distances,
                jax.ops.index[ϵ_ind:],
                results[3])
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

    def set_accepted(self, smoothing=None):
        self.parameters.accepted = [params for params in self.parameters.all]
        self.parameters.n_accepted = [params.shape[0]
            for params in self.parameters.all]
        self.summaries.accepted = [summaries
            for summaries in self.summaries.all]
        self.summaries.n_accepted = [summaries.shape[0]
            for summaries in self.summaries.all]
        self.distances.accepted = [distances
            for distances in self.distances.all]
        self.distances.n_accepted = [distances.shape[0]
            for distances in self.distances.all]
        self.parameters.rejected = None
        self.summaries.rejected = None
        self.distances.rejected = None
        self.marginals = self.get_marginals(smoothing=smoothing)

    def w_cov(self, proposed, weighting):
        weighted_samples = proposed * weighting[:, np.newaxis]
        return weighted_samples.T.dot(
            weighted_samples) / weighting.T.dot(weighting)

class tmvn():
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
        return (rng, self.mvn(key, loc), counter+1)

    def _sample(self, rng, loc):
        rng, key = jax.random.split(rng)
        _, loc, counter = jax.lax.while_loop(
            self.w_cond,
            self.__sample,
            (rng, self.mvn(key, loc), 0))
        return jax.lax.cond(
            np.greater_equal(counter, self.max_counter),
            lambda _ : np.nan * np.ones((self.n_params,)),
            lambda _ : loc,
            None)

    def _sample_n(self, rng, loc, n=None):
        if n is None:
            return self._sample(rng, loc)
        else:
            key = jax.random.split(rng, num=n)
            return jax.vmap(self._sample)(key,
                np.repeat(loc[np.newaxis], n, axis=0))

    def sample(self, shape=None, seed=None):
        if shape is None:
            if self.n_samples is None:
                return self._sample_n(seed, self.loc)
            else:
                key = jax.random.split(seed, num_self.n_samples)
                return jax.vmap(
                    lambda key, loc : self._sample_n(key, loc))(key, self.loc)
        elif len(shape) == 1:
            if self.n_samples is None:
                return self._sample_n(seed, self.loc, n=shape[0])
            else:
                key = jax.random.split(seed, num_self.n_samples)
                return jax.vmap(
                    lambda key, loc : self._sample_n(key, loc, n=shape[0]))(
                    key, self.loc)
        else:
            key = jax.random.split(seed, num=shape[-1])
            return jax.vmap(
                lambda key: self.sample(
                    shape=tuple(shape[:-1]), seed=key), out_axes=-2)(key)
