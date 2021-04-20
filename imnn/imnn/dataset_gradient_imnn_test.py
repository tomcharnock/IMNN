import pytest
import jax
import jaxlib
import jax.numpy as np
import tensorflow as tf
from test.aggregated_defaults import aggregatedTests
from test.gradient_defaults import gradientTests
from imnn.imnn import DatasetGradientIMNN


class datasetGradientTests(
        aggregatedTests, gradientTests):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.kwargs.pop("fiducial")
        self.kwargs.pop("derivative")
        self.kwargs.pop("validation_fiducial")
        self.kwargs.pop("validation_derivative")
        self.reduced_kwargs.pop("fiducial")
        self.reduced_kwargs.pop("derivative")
        self.reduced_kwargs.pop("validation_fiducial")
        self.reduced_kwargs.pop("validation_derivative")

        self.kwargs["main"] = (
            self.fiducial[:self.n_d].reshape(
                (self.n_devices,
                 self.n_d // (self.n_devices * self.n_per_device),
                 self.n_per_device)
                + self.input_shape),
            self.derivative.reshape(
                (self.n_devices,
                 self.n_d // (self.n_devices * self.n_per_device),
                 self.n_per_device)
                + self.input_shape + (self.n_params,)))

        self.kwargs["validation_main"] = (
            self.validation_fiducial[:self.n_d].reshape(
                (self.n_devices,
                 self.n_d // (self.n_devices * self.n_per_device),
                 self.n_per_device)
                + self.input_shape),
            self.validation_derivative.reshape(
                (self.n_devices,
                 self.n_d // (self.n_devices * self.n_per_device),
                 self.n_per_device)
                + self.input_shape + (self.n_params,)))

        self.kwargs["remaining"] = self.fiducial[self.n_d:].reshape(
            (self.n_devices,
             (self.n_s - self.n_d) // (self.n_devices * self.n_per_device),
             self.n_per_device)
            + self.input_shape)

        self.kwargs["validation_remaining"] = \
            self.validation_fiducial[self.n_d:].reshape(
                (self.n_devices,
                 (self.n_s - self.n_d) // (self.n_devices * self.n_per_device),
                 self.n_per_device)
                + self.input_shape)

        self.reduced_kwargs["main"] = (
            self.fiducial[:self.reduced_n_d].reshape(
                (self.n_devices,
                 self.reduced_n_d // (self.n_devices * self.n_per_device),
                 self.n_per_device)
                + self.input_shape),
            self.reduced_derivative.reshape(
                (self.n_devices,
                 self.reduced_n_d // (self.n_devices * self.n_per_device),
                 self.n_per_device)
                + self.input_shape + (self.n_params,)))

        self.reduced_kwargs["validation_main"] = (
            self.validation_fiducial[:self.reduced_n_d].reshape(
                (self.n_devices,
                 self.reduced_n_d // (self.n_devices * self.n_per_device),
                 self.n_per_device)
                + self.input_shape),
            self.reduced_validation_derivative.reshape(
                (self.n_devices,
                 self.reduced_n_d // (self.n_devices * self.n_per_device),
                 self.n_per_device)
                + self.input_shape + (self.n_params,)))

        self.reduced_kwargs["remaining"] = \
            self.fiducial[self.reduced_n_d:].reshape(
                (self.n_devices,
                 (self.n_s - self.reduced_n_d) //
                    (self.n_devices * self.n_per_device),
                 self.n_per_device)
                + self.input_shape)

        self.reduced_kwargs["validation_remaining"] = \
            self.validation_fiducial[self.reduced_n_d:].reshape(
                (self.n_devices,
                 (self.n_s - self.reduced_n_d) //
                    (self.n_devices * self.n_per_device),
                 self.n_per_device)
                + self.input_shape)

        self.arrays = ["θ_fid"]

    def preload(self, dictionary, state=False, validate=False):
        dictionary["host"] = jax.devices("cpu")[0]
        dictionary["devices"] = jax.devices()
        dictionary["main"] = [
            tf.data.Dataset.zip(
                (tf.data.Dataset.from_tensor_slices(fid),
                 tf.data.Dataset.from_tensor_slices(der))
            ).repeat().as_numpy_iterator()
            for fid, der in zip(*dictionary["main"])]
        dictionary["remaining"] = [
            tf.data.Dataset.from_tensor_slices(
                fid).repeat().as_numpy_iterator()
            for fid in dictionary["remaining"]]
        if state:
            dictionary["key_or_state"] = self.state
        if (not self.simulate) and (not validate):
            dictionary.pop("validation_main")
            dictionary.pop("validation_remaining")
        else:
            dictionary["validation_main"] = [
                tf.data.Dataset.zip(
                    (tf.data.Dataset.from_tensor_slices(fid),
                     tf.data.Dataset.from_tensor_slices(der))
                ).repeat().as_numpy_iterator()
                for fid, der in zip(*dictionary["validation_main"])]
            dictionary["validation_remaining"] = [
                tf.data.Dataset.from_tensor_slices(
                    fid).repeat().as_numpy_iterator()
                for fid in dictionary["validation_remaining"]]

        return dictionary

    def specific_exceptions(self, variable, input_variable, kwargs):
        if variable == "validation_main":
            if "validation_main" not in kwargs.keys():
                return True
        if variable == "validation_remaining":
            if "validation_remaining" not in kwargs.keys():
                return True
        if variable == "devices":
            if input_variable is list():
                if len(input_variable) < 1:
                    kwargs[variable] = input_variable
                    with pytest.raises(ValueError) as info:
                        self.imnn(**kwargs)
                    assert info.match("`devices` has no elements in")
                    return True
                if not all(
                        [isinstance(device, jaxlib.xla_extension.Device)
                         for device in input_variable]):
                    kwargs[variable] = input_variable
                    with pytest.raises(TypeError) as info:
                        self.imnn(**kwargs)
                    assert info.match(
                        "`all elements of `devices` must be xla devices")
                    return True
        if variable == "host":
            if input_variable is None:
                return False
            if not isinstance(input_variable, jaxlib.xla_extension.Device):
                kwargs[variable] = input_variable
                with pytest.raises(TypeError) as info:
                    self.imnn(**kwargs)
                assert info.match(
                    "`host` must be an xla device but is a "
                    f"{type(input_variable)}")
                return True
        return False

    def splitting_resize(self, _kwargs):
        _kwargs["main"] = None
        _kwargs["remaining"] = None
        if "validation_main" in _kwargs.keys():
            _kwargs["validation_main"] = None
            _kwargs["validation_remaining"] = None
        return _kwargs


test = datasetGradientTests(
    imnn=DatasetGradientIMNN,
    filename="dataset_gradient",
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
