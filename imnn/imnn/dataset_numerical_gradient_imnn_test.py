import pytest
import jax
import jax.numpy as np
import tensorflow as tf
from test.aggregated_defaults import aggregatedTests
from test.numerical_gradient_defaults import numericalGradientTests
from imnn.imnn import DatasetNumericalGradientIMNN


class datasetNumericalGradientTests(
        aggregatedTests, numericalGradientTests):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.kwargs["fiducial"] = self.fiducial.reshape(
            (self.n_devices,
             self.n_s // (self.n_devices * self.n_per_device),
             self.n_per_device)
            + self.input_shape)

        self.kwargs["derivative"] = self.derivative.reshape(
            (self.n_devices,
             2 * self.n_params * self.n_d //
                (self.n_devices * self.n_per_device),
             self.n_per_device)
            + self.input_shape)

        self.kwargs["validation_fiducial"] = self.validation_fiducial.reshape(
            (self.n_devices,
             self.n_s // (self.n_devices * self.n_per_device),
             self.n_per_device)
            + self.input_shape)

        self.kwargs["validation_derivative"] = \
            self.validation_derivative.reshape(
                (self.n_devices,
                 2 * self.n_params * self.n_d //
                    (self.n_devices * self.n_per_device),
                 self.n_per_device)
                + self.input_shape)

        self.reduced_kwargs["fiducial"] = self.fiducial.reshape(
            (self.n_devices,
             self.n_s // (self.n_devices * self.n_per_device),
             self.n_per_device)
            + self.input_shape)

        self.reduced_kwargs["derivative"] = \
            self.reduced_derivative.reshape(
                (self.n_devices,
                 2 * self.n_params * self.reduced_n_d //
                    (self.n_devices * self.n_per_device),
                 self.n_per_device)
                + self.input_shape)

        self.reduced_kwargs["validation_fiducial"] = \
            self.validation_fiducial.reshape(
                (self.n_devices,
                 self.n_s // (self.n_devices * self.n_per_device),
                 self.n_per_device)
                + self.input_shape)

        self.reduced_kwargs["validation_derivative"] = \
            self.reduced_validation_derivative.reshape(
                (self.n_devices,
                 2 * self.n_params * self.reduced_n_d //
                    (self.n_devices * self.n_per_device),
                 self.n_per_device)
                + self.input_shape)

        self.arrays = ["θ_fid", "δθ"]

    def preload(self, dictionary, state=False, validate=False):
        dictionary["host"] = jax.devices("cpu")[0]
        dictionary["devices"] = jax.devices()
        dictionary["fiducial"] = [
            tf.data.Dataset.from_tensor_slices(
                fid).repeat().as_numpy_iterator()
            for fid in dictionary["fiducial"]]
        dictionary["derivative"] = [
            tf.data.Dataset.from_tensor_slices(
                der).repeat().as_numpy_iterator()
            for der in dictionary["derivative"]]
        if state:
            dictionary["key_or_state"] = self.state
        if (not self.simulate) and (not validate):
            dictionary.pop("validation_fiducial")
            dictionary.pop("validation_derivative")
        else:
            dictionary["validation_fiducial"] = [
                tf.data.Dataset.from_tensor_slices(
                    fid).repeat().as_numpy_iterator()
                for fid in dictionary["validation_fiducial"]]
            dictionary["validation_derivative"] = [
                tf.data.Dataset.from_tensor_slices(
                    der).repeat().as_numpy_iterator()
                for der in dictionary["validation_derivative"]]

        return dictionary

    def splitting_resize(self, _kwargs):
        _kwargs["fiducial"] = None
        _kwargs["derivative"] = None
        if "validation_fiducial" in _kwargs.keys():
            _kwargs["validation_fiducial"] = None
            _kwargs["validation_derivative"] = None
        return _kwargs


test = datasetNumericalGradientTests(
    imnn=DatasetNumericalGradientIMNN,
    filename="dataset_numerical_gradient",
    n_per_device=100)


@pytest.mark.parametrize("kwargs", [test.kwargs, test.reduced_kwargs])
@pytest.mark.parametrize("state", [True, False])
@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize(
    "input_variable",
    [None, list(), 1., 1, np.zeros((1,)), test.rng, tuple(), (0, 0),
     (test.model[0], 0), test.bad_model, test.state])
@pytest.mark.parametrize("variable", test.kwargs.keys())
def test_initialisation_parameters_(
        variable, kwargs, input_variable, state, validate):
    test.initialise_parameters(
        variable, kwargs, input_variable, state=state, validate=validate)


@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("state", [False, True])
@pytest.mark.parametrize("variable", ["n_s", "n_d", "same"])
def test_splitting_(variable, validate, state):
    test.splitting(variable, test.kwargs, state=state, validate=validate)


@pytest.mark.parametrize("kwargs", [test.kwargs, test.reduced_kwargs])
@pytest.mark.parametrize("state", [True, False])
@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize(
    "input_variable", [None, list(), 1., 1, np.zeros((1,)), test.rng])
@pytest.mark.parametrize("variable", test.fit_kwargs.keys())
def test_fit_parameters_(variable, kwargs, input_variable, state, validate):
    test.fit_parameters(
        variable, kwargs, test.fit_kwargs, input_variable, state=state,
        validate=validate)


@pytest.mark.parametrize("state", [True, False])
@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("fit", [True, False])
@pytest.mark.parametrize("none_first", [True, False])
@pytest.mark.parametrize("kwargs", [test.kwargs, test.reduced_kwargs])
def test_combined_running_test_(kwargs, state, validate, fit, none_first):
    test.combined_running_test(
        [test.single_target_data, test.batch_target_data], kwargs,
        test.fit_kwargs, state=state, validate=validate, fit=fit,
        none_first=none_first, aggregated=True)
