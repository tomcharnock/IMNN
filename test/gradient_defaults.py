from test.defaults import defaultTests, simulator_gradient
from functools import partial
import jax
import jax.numpy as np

rng = jax.random.PRNGKey(0)
rng, model_key, data_key, fit_key, stats_key = jax.random.split(rng, num=5)
simulation_key, validation_simulation_key = jax.random.split(data_key)

training_keys = jax.random.split(simulation_key, num=1000)
validation_keys = jax.random.split(validation_simulation_key, num=1000)

fiducial, derivative = jax.vmap(
    partial(
        simulator_gradient,
        θ=np.array([0., 1.])))(
    np.array(training_keys))
validation_fiducial, validation_derivative = jax.vmap(
    partial(
        simulator_gradient,
        θ=np.array([0., 1.])))(
    np.array(validation_keys))


class gradientTests(defaultTests):
    def __init__(
            self, fiducial=fiducial, derivative=derivative,
            validation_fiducial=validation_fiducial,
            validation_derivative=validation_derivative, **kwargs):
        super().__init__(**kwargs)
        self.fiducial = fiducial
        self.derivative = derivative
        self.validation_fiducial = validation_fiducial
        self.validation_derivative = validation_derivative
        self.reduced_derivative = derivative[:self.reduced_n_d]
        self.reduced_validation_derivative = \
            validation_derivative[:self.reduced_n_d]
        self.kwargs["fiducial"] = fiducial
        self.kwargs["derivative"] = derivative
        self.kwargs["validation_fiducial"] = validation_fiducial
        self.kwargs["validation_derivative"] = validation_derivative
        self.reduced_kwargs["fiducial"] = fiducial
        self.reduced_kwargs["derivative"] = derivative[:self.reduced_n_d]
        self.reduced_kwargs["validation_fiducial"] = validation_fiducial
        self.reduced_kwargs["validation_derivative"] = \
            validation_derivative[:self.reduced_n_d]
        self.arrays = ["θ_fid", "fiducial", "derivative",
                       "validation_fiducial", "validation_derivative"]

    def preload(self, dictionary, state=False, validate=False):
        if state:
            dictionary["key_or_state"] = self.state
        if not validate:
            dictionary.pop("validation_fiducial")
            dictionary.pop("validation_derivative")
        return dictionary

    def specific_exceptions(self, variable, input_variable, kwargs):
        if variable == "validation_fiducial":
            if "validation_fiducial" not in kwargs.keys():
                return True
        if variable == "validation_derivative":
            if "validation_derivative" not in kwargs.keys():
                return True
        return False
