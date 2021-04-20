from test.gradient_defaults import gradientTests
from test.defaults import simulator
from functools import partial
import jax
import jax.numpy as np


def get_numerical_derivative(
        key, θ=np.array([0., 1.]), δθ=np.array([0.1, 0.1])):
    a = simulator(key, (θ[0] - δθ[0] / 2., θ[1]))
    b = simulator(key, (θ[0], θ[1] - δθ[1] / 2.))
    c = simulator(key, (θ[0] + δθ[0] / 2., θ[1]))
    d = simulator(key, (θ[0], θ[1] + δθ[1] / 2.))
    return np.stack([np.stack([a, b], 0), np.stack([c, d], 0)], 0)


rng = jax.random.PRNGKey(0)
rng, model_key, data_key, fit_key, stats_key = jax.random.split(rng, num=5)
simulation_key, validation_simulation_key = jax.random.split(data_key)

training_keys = jax.random.split(simulation_key, num=1000)
validation_keys = jax.random.split(validation_simulation_key, num=1000)

fiducial = jax.vmap(
    partial(
        simulator,
        θ=np.array([0., 1.])))(
    np.array(training_keys))
numerical_derivative = jax.vmap(get_numerical_derivative)(
    np.array(training_keys))
validation_fiducial = jax.vmap(
    partial(
        simulator,
        θ=np.array([0., 1.])))(
    np.array(validation_keys))
validation_numerical_derivative = jax.vmap(get_numerical_derivative)(
    np.array(validation_keys))


class numericalGradientTests(gradientTests):
    def __init__(
            self, δθ=np.array([0.1, 0.1]), fiducial=fiducial,
            derivative=numerical_derivative,
            validation_fiducial=validation_fiducial,
            validation_derivative=validation_numerical_derivative, **kwargs):
        super().__init__(
            fiducial=fiducial,
            derivative=derivative,
            validation_fiducial=validation_fiducial,
            validation_derivative=validation_derivative,
            **kwargs)
        self.δθ = δθ
        self.kwargs["δθ"] = δθ
        self.reduced_kwargs["δθ"] = δθ
        self.arrays = ["θ_fid", "δθ", "fiducial", "derivative",
                       "validation_fiducial", "validation_derivative"]
